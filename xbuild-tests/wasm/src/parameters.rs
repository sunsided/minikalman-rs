//! Parameter estimation: dual RLS + Kalman filter vs a joint augmented-state EKF.
//!
//! The plant and estimators of the `augmented-state` and `rls-adaptive` examples. A motor's
//! speed follows `xₖ₊₁ = a·xₖ + b·uₖ + w`, measured as `z = x + v`. At step 400 the load
//! changes and the true `(a, b)` jump from `(0.9, 0.5)` to `(0.7, 1.0)`.
//!
//! - **Dual**: recursive least squares identifies `(a, b)` with forgetting factor `μ`, and a
//!   one-state Kalman filter takes `A = â`, `B = b̂` from it each step.
//! - **Joint**: one extended Kalman filter on the augmented state `[x, a, b]`, with the
//!   parameters as a random walk of variance `q` per step.

use minikalman::extended::{FilterStorage as EkfStorage, ObservationStorage as EkfObservationStorage};
use minikalman::matrix::MatrixData;
use minikalman::prelude::*;
use minikalman::regular::{ControlStorage, FilterStorage, ObservationStorage};
use minikalman::rls::EstimatorStorage;

use crate::lab::{Noise, Trace, knob, record, rms};

/// The true `(a, b)` before the load change; also every estimator's initial guess.
const NOMINAL: [f64; 2] = [0.9, 0.5];

/// The true `(a, b)` after the load change.
const CHANGED: [f64; 2] = [0.7, 1.0];

/// Number of steps.
const STEPS: usize = 800;

/// The step at which the load changes.
const CHANGE_AT: usize = 400;

/// Standard deviation of the process noise `w`.
const PROCESS_SIGMA: f64 = 0.05;

/// How many steps the input holds its value before it may switch.
const INPUT_HOLD: usize = 20;

/// Initial variance of each parameter, in both estimators.
const PARAMETER_VARIANCE: f64 = 1.0;

/// The bound on `|â − a|` for re-identification.
const A_BOUND: f64 = 0.05;

/// The bound on `|b̂ − b|` for re-identification.
const B_BOUND: f64 = 0.1;

/// The trace channels, in order.
mod channel {
    /// True speed.
    pub const SPEED: usize = 0;
    /// Measured speed.
    pub const MEASUREMENT: usize = 1;
    /// True `a`, then `b`.
    pub const THETA: usize = 2;
    /// Joint EKF: speed, `â`, `b̂`, `σ_a`, `σ_b`.
    pub const JOINT: usize = 4;
    /// Dual RLS + KF: speed, `â`, `b̂`, `σ_a`, `σ_b`.
    pub const DUAL: usize = 9;
    /// The input `u`.
    pub const INPUT: usize = 14;
    /// Number of channels.
    pub const COUNT: usize = 15;
}

/// Writes one estimator's step: speed, parameters and their standard deviations.
#[allow(clippy::cast_possible_truncation)]
fn put(trace: &mut Trace<'_>, base: usize, k: usize, x: f64, theta: [f64; 2], variance: [f64; 2]) {
    trace.set(base, k, x as f32);
    trace.set(base + 1, k, theta[0] as f32);
    trace.set(base + 2, k, theta[1] as f32);
    trace.set(base + 3, k, variance[0].max(0.0).sqrt() as f32);
    trace.set(base + 4, k, variance[1].max(0.0).sqrt() as f32);
}

/// Simulates one run and writes it to the trace.
///
/// Channels: see [`channel`]. Summary: `[speed RMSE after the change (joint, dual), RMS of
/// â − a once settled (joint, dual), RMS of b̂ − b once settled (joint, dual), steps to
/// re-identify (joint, dual; −1 for never), speed RMSE before the change (joint, dual)]`.
/// "Settled" is from 50 steps after the change on.
///
/// * `seed` - Noise seed.
/// * `sensor_sigma` - Sensor noise standard deviation, `0.01..=1`.
/// * `log_q` - `log₁₀` of the joint filter's parameter process noise, `−8..=−1`.
/// * `forgetting` - The dual estimator's forgetting factor `μ`, `0.8..=1`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_lines)]
pub extern "C" fn parameters_run(seed: u32, sensor_sigma: f32, log_q: f32, forgetting: f32) -> u32 {
    let sensor = f64::from(knob(sensor_sigma, 0.01, 1.0));
    let q = 10f64.powf(f64::from(knob(log_q, -8.0, -1.0)));
    let forgetting = f64::from(knob(forgetting, 0.8, 1.0));

    let mut ekf_storage = EkfStorage::<3, f64>::new(0.0);
    let mut ekf_observation_storage = EkfObservationStorage::<3, 1, f64>::new(0.0);
    let mut rls_storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut kf_storage = FilterStorage::<1, f64>::new(0.0);
    let mut control_storage = ControlStorage::<1, 1, f64>::new(0.0);
    let mut kf_observation_storage = ObservationStorage::<1, 1, f64>::new(0.0);

    let mut noise = Noise::new(seed);
    record(channel::COUNT, STEPS, |trace, summary| {
        // Joint EKF on [x, a, b].
        let mut ekf = ekf_storage.build();
        let mut ekf_measurement = ekf_observation_storage.build();
        ekf.state_vector_mut().apply(|s| {
            s[1] = NOMINAL[0];
            s[2] = NOMINAL[1];
        });
        ekf.estimate_covariance_mut().apply(|p| {
            p.set_at(0, 0, 1.0);
            p.set_at(1, 1, PARAMETER_VARIANCE);
            p.set_at(2, 2, PARAMETER_VARIANCE);
        });
        ekf.direct_process_noise_mut().apply(|n| {
            n.set_at(0, 0, PROCESS_SIGMA * PROCESS_SIGMA);
            n.set_at(1, 1, q);
            n.set_at(2, 2, q);
        });
        ekf_measurement.observation_jacobian_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
        ekf_measurement
            .measurement_noise_covariance_mut()
            .apply(|r| r.set_at(0, 0, sensor * sensor));

        // Dual: RLS feeding a one-state Kalman filter.
        let mut rls = rls_storage.build(PARAMETER_VARIANCE);
        rls.set_forgetting_factor(forgetting);
        rls.set_measurement_variance(sensor * sensor);
        rls.parameters_mut().apply(|theta| {
            theta.set(0, 0, NOMINAL[0]);
            theta.set(1, 0, NOMINAL[1]);
        });
        let mut kf = kf_storage.build();
        let mut control = control_storage.build();
        let mut kf_measurement = kf_observation_storage.build();
        kf.estimate_covariance_mut().set(0, 0, 1.0);
        kf_measurement.observation_matrix_mut().set(0, 0, 1.0);
        kf_measurement.measurement_noise_covariance_mut().set(0, 0, sensor * sensor);

        let (mut x, mut u) = (0.0f64, 1.0f64);
        for k in 0..STEPS {
            let theta = if k < CHANGE_AT { NOMINAL } else { CHANGED };
            x = theta[0] * x + theta[1] * u + PROCESS_SIGMA * noise.gaussian();
            let z = x + sensor * noise.gaussian();
            #[allow(clippy::cast_possible_truncation)]
            {
                trace.set(channel::SPEED, k, x as f32);
                trace.set(channel::MEASUREMENT, k, z as f32);
                trace.set(channel::THETA, k, theta[0] as f32);
                trace.set(channel::THETA + 1, k, theta[1] as f32);
                trace.set(channel::INPUT, k, u as f32);
            }

            // Joint: F = ∂f/∂s at the current estimate.
            let (x_hat, a_hat) = (ekf.state_vector()[0], ekf.state_vector()[1]);
            ekf.state_transition_jacobian_mut().apply(|f| {
                f.make_identity();
                f.set_at(0, 0, a_hat);
                f.set_at(0, 1, x_hat);
                f.set_at(0, 2, u);
            });
            ekf.predict_nonlinear(|s, next| {
                next[0] = s[1] * s[0] + s[2] * u;
                next[1] = s[1];
                next[2] = s[2];
            });
            ekf_measurement.measurement_vector_mut().apply(|m| m[0] = z);
            ekf.correct_nonlinear(&mut ekf_measurement, |s, m| m[0] = s[0]);
            let s = ekf.state_vector();
            let p = ekf.estimate_covariance();
            put(
                trace,
                channel::JOINT,
                k,
                s[0],
                [s[1], s[2]],
                [p.get_at(1, 1), p.get_at(2, 2)],
            );

            // Dual: identify zₖ ≈ a·x̂ₖ₋₁ + b·uₖ₋₁ with the filter's previous estimate.
            let phi = [kf.state_vector().get(0, 0), u];
            rls.update(&phi, z);
            let theta_hat = [rls.parameters().get_row(0), rls.parameters().get_row(1)];
            let mut p_theta = [0.0; 4];
            rls.covariance(&mut MatrixData::new_mut::<2, 2, f64>(&mut p_theta));
            // The parameter uncertainty adds φᵀ·P_θ·φ to the model's process noise.
            let model_variance = phi[0] * phi[0] * p_theta[0]
                + 2.0 * phi[0] * phi[1] * p_theta[1]
                + phi[1] * phi[1] * p_theta[3];
            kf.state_transition_mut().set(0, 0, theta_hat[0]);
            kf.direct_process_noise_mut()
                .set(0, 0, PROCESS_SIGMA * PROCESS_SIGMA + model_variance);
            control.control_matrix_mut().set(0, 0, theta_hat[1]);
            control.control_vector_mut().set(0, 0, u);
            kf_measurement.measurement_vector_mut().set(0, 0, z);
            kf.predict().control(&mut control);
            kf.correct(&mut kf_measurement);
            put(
                trace,
                channel::DUAL,
                k,
                kf.state_vector().get(0, 0),
                theta_hat,
                [p_theta[0], p_theta[3]],
            );

            if (k + 1) % INPUT_HOLD == 0 && noise.chance(0.5) {
                u = -u;
            }
        }

        let series = |trace: &Trace<'_>, c: usize, truth: usize, range: core::ops::Range<usize>| {
            rms(range.map(|k| f64::from(trace.get(c, k) - trace.get(truth, k)))) as f32
        };
        let settled = CHANGE_AT + 50..STEPS;
        for (i, base) in [channel::JOINT, channel::DUAL].into_iter().enumerate() {
            summary[i] = series(trace, base, channel::SPEED, CHANGE_AT..STEPS);
            summary[2 + i] = series(trace, base + 1, channel::THETA, settled.clone());
            summary[4 + i] = series(trace, base + 2, channel::THETA + 1, settled.clone());
            #[allow(clippy::cast_precision_loss)]
            let reidentified = (CHANGE_AT..STEPS)
                .find(|&k| {
                    f64::from((trace.get(base + 1, k) - trace.get(channel::THETA, k)).abs()) < A_BOUND
                        && f64::from((trace.get(base + 2, k) - trace.get(channel::THETA + 1, k)).abs())
                            < B_BOUND
                })
                .map_or(-1.0, |k| (k + 1 - CHANGE_AT) as f32);
            summary[6 + i] = reidentified;
            summary[8 + i] = series(trace, base, channel::SPEED, 50..CHANGE_AT);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_reidentify_and_joint_is_faster() {
        parameters_run(42, 0.2, -4.0, 0.98);
        let s: [f32; 10] = crate::lab::summary();
        assert!(s[6] > 0.0 && s[7] > 0.0, "{s:?}");
        assert!(s[6] < s[7], "joint {} vs dual {} steps", s[6], s[7]);
    }

    #[test]
    fn extreme_knobs_stay_finite() {
        for args in [(0, 0.0, -99.0, 0.0), (u32::MAX, 99.0, 99.0, 99.0)] {
            parameters_run(args.0, args.1, args.2, args.3);
            let s: [f32; 2] = crate::lab::summary();
            assert!(s.iter().all(|v| v.is_finite()), "{args:?}: {s:?}");
        }
    }
}
