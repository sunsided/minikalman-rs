//! Outliers: the plain Kalman update, an NIS-gated update and the Student-t robust update on
//! the same measurements.
//!
//! A target moves along a smooth 1D path and a sensor measures its position with Gaussian
//! noise; a share of the readings is replaced by gross outliers. Three constant-velocity
//! filters run on the same readings:
//!
//! - **standard**: `correct`, which takes every reading at face value;
//! - **gated**: `correct` with `set_gate`, which rejects a reading whose NIS exceeds the gate,
//!   optionally re-acquiring after a number of consecutive rejections by inflating `P`;
//! - **robust**: `correct_robust` with Student-t noise, which down-weights a reading by how
//!   far it is from the prediction instead of a hard accept or reject.
//!
//! The filters assume constant velocity with white-noise acceleration. The truth is a smooth
//! swing that this model follows through its process noise, plus an optional step at 15 s
//! that no motion model predicts. After such a step, every reading is far from the
//! prediction: the gate rejects them all and the gated filter coasts on its model (a gate
//! lockout) unless it re-acquires.

use minikalman::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer;
use minikalman::models::motion;
use minikalman::prelude::*;
use minikalman::regular::{FilterStorage, ObservationStorage};

use crate::lab::{Noise, knob, record, rms};

/// Number of steps of a run.
const STEPS: usize = 300;

/// Time step in seconds.
const DT: f32 = 0.1;

/// Acceleration variance of the filters' constant-velocity model, in (m/s²)².
const ACCELERATION_VARIANCE: f32 = 2.0;

/// The trace channels, in order.
#[allow(dead_code)]
mod channel {
    /// True position.
    pub const TRUTH: usize = 0;
    /// Measured position.
    pub const MEASUREMENT: usize = 1;
    /// `1` where the reading is an injected outlier.
    pub const OUTLIER: usize = 2;
    /// Position estimate of the standard filter; the gated and robust follow.
    pub const ESTIMATE: usize = 3;
    /// Position standard deviation of the standard filter; the gated and robust follow.
    pub const SIGMA: usize = 6;
    /// `1` where the gated filter rejected the reading.
    pub const REJECTED: usize = 9;
    /// The Student-t weight `w` of the robust filter.
    pub const WEIGHT: usize = 10;
    /// The prior NIS of the gated filter's reading, also for rejected readings.
    pub const NIS: usize = 11;
    /// `1` where the gated filter re-acquired by inflating `P` after this reading.
    pub const REACQUIRED: usize = 12;
    /// Number of channels.
    pub const COUNT: usize = 13;
}

/// The factor by which the gated filter inflates `P` to re-acquire after a lockout.
const REACQUIRE_INFLATION: f32 = 100.0;

/// The time in seconds at which the target jumps, if `jump` is not zero.
const JUMP_AT: f32 = 15.0;

/// The true position at time `t`: a slow swing with a drift, and a sudden jump by `jump`.
fn truth(t: f32, jump: f32) -> f32 {
    6.0 * (0.35 * t).sin() + 0.5 * t + if t >= JUMP_AT { jump } else { 0.0 }
}

/// Simulates one run and writes it to the trace.
///
/// Channels: see [`channel`]. Summary: `[rmse standard, rmse gated, rmse robust,
/// rmse of the clean readings, outliers, gated rejections, rejections that were outliers,
/// mean robust weight on outliers, mean robust weight on clean readings, re-acquisitions]`.
///
/// * `seed` - Noise seed.
/// * `sigma` - Measurement noise standard deviation in m, `0.05..=5`.
/// * `rate` - Share of outliers, `0..=0.6`.
/// * `size` - Outlier size in multiples of `sigma`, `2..=60`.
/// * `gate` - NIS gate of the gated filter, `0.5..=100` (chi-square, one degree of freedom).
/// * `dof` - Degrees of freedom `ν` of the robust filter, `0.5..=100`.
/// * `jump` - A sudden change of the true position at 15 s in m, `-20..=20`: a real change
///   that a hard gate can mistake for an outlier.
/// * `reacquire` - After this many consecutive rejections, the gated filter inflates `P` by
///   100 so the next reading passes the gate again; `0` turns this off, at most `50`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_lines)]
pub extern "C" fn outliers_run(
    seed: u32,
    sigma: f32,
    rate: f32,
    size: f32,
    gate: f32,
    dof: f32,
    jump: f32,
    reacquire: u32,
) -> u32 {
    let reacquire = reacquire.min(50);
    let jump = knob(jump, -20.0, 20.0);
    let sigma = knob(sigma, 0.05, 5.0);
    let rate = knob(rate, 0.0, 0.6);
    let size = knob(size, 2.0, 60.0);
    let gate = knob(gate, 0.5, 100.0);
    let dof = knob(dof, 0.5, 100.0);

    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    let mut filters = [const { FilterStorage::<2, f32>::new(0.0) }; 3];
    let mut observations = [const { ObservationStorage::<2, 1, f32>::new(0.0) }; 3];
    let mut workspace = [0.0f32; 1];
    let robust = RobustCorrection {
        weight: RobustWeight::StudentT { dof },
        iterations: 20,
    };

    let mut noise = Noise::new(seed);
    record(channel::COUNT, STEPS, |trace, summary| {
        let mut filters = filters.each_mut().map(FilterStorage::build);
        let mut observations = observations.each_mut().map(ObservationStorage::build);
        let start = truth(0.0, jump);
        for (filter, observation) in filters.iter_mut().zip(observations.iter_mut()) {
            filter
                .state_transition_mut()
                .apply(|a| model.fill_transition(a));
            filter
                .direct_process_noise_mut()
                .apply(|q| model.fill_process_noise(q));
            filter.state_vector_mut().set(0, 0, start);
            filter.estimate_covariance_mut().apply(|p| {
                p.set(0, 0, sigma * sigma);
                p.set(1, 1, 4.0);
            });
            observation.observation_matrix_mut().set(0, 0, 1.0);
            observation
                .measurement_noise_covariance_mut()
                .set(0, 0, sigma * sigma);
        }
        observations[1].set_gate(Some(gate));

        let (mut outliers, mut rejected, mut caught) = (0u32, 0u32, 0u32);
        let (mut consecutive, mut reacquisitions) = (0u32, 0u32);
        let (mut weight_outlier, mut weight_clean) = (0.0f64, 0.0f64);
        for k in 0..STEPS {
            #[allow(clippy::cast_precision_loss)]
            let x = truth(k as f32 * DT, jump);
            let is_outlier = noise.chance(f64::from(rate));
            let mut z = x + sigma * noise.gaussian() as f32;
            if is_outlier {
                let sign = if noise.chance(0.5) { 1.0 } else { -1.0 };
                z += sign * size * sigma * (0.75 + 0.5 * noise.uniform() as f32);
                outliers += 1;
            }
            trace.set(channel::TRUTH, k, x);
            trace.set(channel::MEASUREMENT, k, z);
            trace.set(channel::OUTLIER, k, f32::from(u8::from(is_outlier)));

            for (i, (filter, observation)) in
                filters.iter_mut().zip(observations.iter_mut()).enumerate()
            {
                filter.predict();
                observation.measurement_vector_mut().set(0, 0, z);
                let status = if i == 2 {
                    let mut workspace =
                        TemporaryResidualCovarianceInvertedMatrixBuffer::<1, f32, _>::from(
                            workspace.as_mut_slice(),
                        );
                    filter.correct_robust(observation, &mut workspace, robust)
                } else {
                    filter.correct(observation)
                };
                trace.set(channel::ESTIMATE + i, k, filter.state_vector().get(0, 0));
                trace.set(
                    channel::SIGMA + i,
                    k,
                    filter.estimate_covariance().get(0, 0).max(0.0).sqrt(),
                );
                if i == 1 {
                    let gated = status == CorrectionStatus::Gated;
                    trace.set(channel::REJECTED, k, f32::from(u8::from(gated)));
                    trace.set(channel::NIS, k, observation.nis());
                    rejected += u32::from(gated);
                    caught += u32::from(gated && is_outlier);
                    consecutive = if gated { consecutive + 1 } else { 0 };
                    let reacquiring = reacquire > 0 && consecutive >= reacquire;
                    if reacquiring {
                        // Admit that the model lost the target: widen P until readings pass.
                        filter.estimate_covariance_mut().apply(|p| {
                            p.as_mut_slice()
                                .iter_mut()
                                .for_each(|v| *v *= REACQUIRE_INFLATION)
                        });
                        consecutive = 0;
                        reacquisitions += 1;
                    }
                    trace.set(channel::REACQUIRED, k, f32::from(u8::from(reacquiring)));
                }
            }

            let w = f64::from(observations[2].robust_weight());
            trace.set(channel::WEIGHT, k, w as f32);
            if is_outlier {
                weight_outlier += w;
            } else {
                weight_clean += w;
            }
        }

        let error = |trace: &crate::lab::Trace<'_>, c: usize| {
            rms((0..STEPS).map(|k| f64::from(trace.get(c, k) - trace.get(channel::TRUTH, k))))
        };
        for (i, value) in summary.iter_mut().take(3).enumerate() {
            *value = error(trace, channel::ESTIMATE + i) as f32;
        }
        summary[3] = rms((0..STEPS)
            .filter(|&k| trace.get(channel::OUTLIER, k) == 0.0)
            .map(|k| f64::from(trace.get(channel::MEASUREMENT, k) - trace.get(channel::TRUTH, k))))
            as f32;
        summary[4] = outliers as f32;
        summary[5] = rejected as f32;
        summary[6] = caught as f32;
        #[allow(clippy::cast_precision_loss)]
        let clean = (STEPS as u32 - outliers).max(1);
        summary[7] = (weight_outlier / f64::from(outliers.max(1))) as f32;
        summary[8] = (weight_clean / f64::from(clean)) as f32;
        summary[9] = reacquisitions as f32;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the summary of the last run.
    fn summary() -> [f32; 9] {
        crate::lab::summary()
    }

    #[test]
    fn robust_and_gated_beat_standard_under_outliers() {
        outliers_run(1, 0.5, 0.1, 15.0, 9.0, 4.0, 0.0, 0);
        let s = summary();
        assert!(s[2] < 0.5 * s[0], "robust {} vs standard {}", s[2], s[0]);
        assert!(s[1] < 0.5 * s[0], "gated {} vs standard {}", s[1], s[0]);
        assert!(s[7] < 0.2 && s[8] > 0.8, "weights {} {}", s[7], s[8]);
    }

    #[test]
    fn without_outliers_all_agree() {
        outliers_run(2, 0.5, 0.0, 15.0, 9.0, 4.0, 0.0, 0);
        let s = summary();
        assert!(
            (s[1] - s[0]).abs() < 0.05 && (s[2] - s[0]).abs() < 0.05,
            "{s:?}"
        );
    }

    #[test]
    fn a_real_jump_locks_the_gate_out_unless_it_reacquires() {
        outliers_run(1, 0.5, 0.1, 15.0, 9.0, 4.0, 3.0, 0);
        let locked: [f32; 10] = crate::lab::summary();
        // The plain and robust filters follow the jump; the gated one coasts away.
        assert!(
            locked[1] > 5.0 * locked[2],
            "gated {} vs robust {}",
            locked[1],
            locked[2]
        );
        assert!(locked[0] < 1.5, "plain {}", locked[0]);
        assert_eq!(locked[9], 0.0);

        outliers_run(1, 0.5, 0.1, 15.0, 9.0, 4.0, 3.0, 5);
        let recovered: [f32; 10] = crate::lab::summary();
        assert!(recovered[9] >= 1.0, "re-acquired {} times", recovered[9]);
        assert!(
            recovered[1] < 0.2 * locked[1],
            "gated {} after re-acquiring",
            recovered[1]
        );
    }

    #[test]
    fn extreme_knobs_stay_finite() {
        for args in [
            (0, 0.0, 0.0, 0.0, 0.0, 0.0, -99.0, 0),
            (u32::MAX, 99.0, 1.0, 999.0, 999.0, 999.0, 99.0, 999),
        ] {
            outliers_run(
                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
            );
            assert!(summary()[..3].iter().all(|v| v.is_finite()), "{args:?}");
        }
    }
}
