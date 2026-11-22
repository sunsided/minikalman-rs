//! Smoothing: the filter, a fixed-lag smoother and the RTS smoother on the same run.
//!
//! A target moves along a smooth 1D path, measured every step except during a gap. The
//! constant-velocity filter estimates each step from the readings up to it, so in the gap it
//! can only coast on its model. The fixed-lag smoother refines each step with the
//! [`LAG`]` − 1` steps after it, online. The RTS smoother runs backwards over the whole
//! recorded history, so every step sees every reading, including those after the gap.

use minikalman::matrix::MatrixDataArray;
use minikalman::models::motion;
use minikalman::prelude::*;
use minikalman::regular::{FilterStorage, ObservationStorage};
use minikalman::smoother::{FixedLagSmoother, RtsSmoother, RtsSmootherStorage};

use crate::lab::{Noise, Trace, knob, record, rms};

/// Number of steps of a run.
const STEPS: usize = 250;

/// Time step in seconds.
const DT: f32 = 0.1;

/// The step at which the gap in the readings starts (10 s).
const GAP_START: usize = 100;

/// The window of the fixed-lag smoother in steps (1.5 s).
pub const LAG: usize = 15;

/// A recorded state vector.
type Vector = MatrixDataArray<2, 1, 2, f32>;

/// A recorded covariance.
type Square = MatrixDataArray<2, 2, 4, f32>;

/// The trace channels, in order.
pub mod channel {
    /// True position.
    pub const TRUTH: usize = 0;
    /// Measured position, NaN in the gap.
    pub const MEASUREMENT: usize = 1;
    /// Filtered position and its standard deviation.
    pub const FILTER: usize = 2;
    /// Fixed-lag smoothed position and its standard deviation, NaN for the last steps.
    pub const LAG: usize = 4;
    /// RTS smoothed position and its standard deviation.
    pub const RTS: usize = 6;
    /// Number of channels.
    pub const COUNT: usize = 8;
}

/// The true position at time `t`.
fn truth(t: f32) -> f32 {
    5.0 * (0.4 * t).sin() + 2.0 * (0.9 * t).cos() + 0.3 * t
}

/// Writes a position and its standard deviation to `channel` and `channel + 1`.
fn put(
    trace: &mut Trace<'_>,
    channel: usize,
    k: usize,
    x: &impl Matrix<2, 1, f32>,
    p: &impl Matrix<2, 2, f32>,
) {
    trace.set(channel, k, x.get(0, 0));
    trace.set(channel + 1, k, p.get(0, 0).max(0.0).sqrt());
}

/// Simulates one run and writes it to the trace.
///
/// Channels: see [`channel`]. Summary: `[rmse filter, rmse fixed-lag, rmse RTS,
/// rmse filter in the gap, rmse RTS in the gap, largest σ of the filter in the gap,
/// largest σ of the RTS smoother in the gap, RMS reading error, all steps smoothed (1/0)]`.
/// The fixed-lag RMSE covers the steps it has smoothed.
///
/// * `seed` - Noise seed.
/// * `sigma` - Measurement noise standard deviation in m, `0.05..=5`.
/// * `q` - Acceleration variance of the model in (m/s²)², `0.001..=100`.
/// * `gap` - Length of the gap in the readings in seconds, from 10 s on, `0..=10`.
#[unsafe(no_mangle)]
pub extern "C" fn smoothing_run(seed: u32, sigma: f32, q: f32, gap: f32) -> u32 {
    let sigma = knob(sigma, 0.05, 5.0);
    let q = knob(q, 0.001, 100.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let gap = (knob(gap, 0.0, 10.0) / DT).round() as usize;
    let in_gap = |k: usize| (GAP_START..GAP_START + gap).contains(&k);

    let model = motion::constant_velocity(DT, q);
    let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
    let mut observation_storage = ObservationStorage::<2, 1, f32>::new(0.0);
    let mut rts_storage = RtsSmootherStorage::<2, f32>::new(0.0);
    let mut x = [const { Vector::new_unchecked([0.0; 2]) }; STEPS];
    let mut p = [const { Square::new_unchecked([0.0; 4]) }; STEPS];
    let mut x_predicted = [const { Vector::new_unchecked([0.0; 2]) }; STEPS];
    let mut p_predicted = [const { Square::new_unchecked([0.0; 4]) }; STEPS];
    let mut lag = FixedLagSmoother::<2, LAG, f32, _, _, _, _, _, _>::new(
        RtsSmoother::new(
            Square::new_unchecked([0.0; 4]),
            Square::new_unchecked([0.0; 4]),
            Square::new_unchecked([0.0; 4]),
            Vector::new_unchecked([0.0; 2]),
        ),
        Vector::new_unchecked([0.0; 2]),
        Square::new_unchecked([0.0; 4]),
    );

    let mut noise = Noise::new(seed);
    record(channel::COUNT, STEPS, |trace, summary| {
        let mut filter = filter_storage.build();
        let mut measurement = observation_storage.build();
        filter
            .state_transition_mut()
            .apply(|a| model.fill_transition(a));
        filter
            .direct_process_noise_mut()
            .apply(|n| model.fill_process_noise(n));
        filter.estimate_covariance_mut().apply(|m| {
            m.set(0, 0, 25.0);
            m.set(1, 1, 25.0);
        });
        measurement.observation_matrix_mut().set(0, 0, 1.0);
        measurement
            .measurement_noise_covariance_mut()
            .set(0, 0, sigma * sigma);

        for k in 0..STEPS {
            #[allow(clippy::cast_precision_loss)]
            let truth = truth(k as f32 * DT);
            trace.set(channel::TRUTH, k, truth);
            trace.set(channel::LAG, k, f32::NAN);
            trace.set(channel::LAG + 1, k, f32::NAN);

            filter.predict();
            filter
                .state_vector()
                .inspect(|v| v.copy_to(&mut x_predicted[k]));
            filter
                .estimate_covariance()
                .inspect(|m| m.copy_to(&mut p_predicted[k]));

            let z = truth + sigma * noise.gaussian() as f32;
            if in_gap(k) {
                trace.set(channel::MEASUREMENT, k, f32::NAN);
            } else {
                trace.set(channel::MEASUREMENT, k, z);
                measurement.measurement_vector_mut().set(0, 0, z);
                filter.correct(&mut measurement);
            }
            filter.state_vector().inspect(|v| v.copy_to(&mut x[k]));
            filter
                .estimate_covariance()
                .inspect(|m| m.copy_to(&mut p[k]));
            put(trace, channel::FILTER, k, &x[k], &p[k]);

            let pushed = filter
                .state_transition()
                .inspect(|a| lag.push(a, &x[k], &p[k], &x_predicted[k], &p_predicted[k]));
            if pushed.is_some() {
                put(
                    trace,
                    channel::LAG,
                    k + 1 - LAG,
                    lag.state(),
                    lag.covariance(),
                );
            }
        }

        let mut rts = rts_storage.build();
        let complete = filter
            .state_transition()
            .inspect(|a| rts.smooth(a, &mut x, &mut p, &x_predicted, &p_predicted));
        for k in 0..STEPS {
            put(trace, channel::RTS, k, &x[k], &p[k]);
        }

        let error = |trace: &Trace<'_>, c: usize, keep: &dyn Fn(usize) -> bool| {
            rms((0..STEPS)
                .filter(|&k| keep(k) && !trace.get(c, k).is_nan())
                .map(|k| f64::from(trace.get(c, k) - trace.get(channel::TRUTH, k))))
                as f32
        };
        let largest = |trace: &Trace<'_>, c: usize| {
            (0..STEPS)
                .filter(|&k| in_gap(k))
                .map(|k| trace.get(c, k))
                .fold(0.0f32, f32::max)
        };
        summary[0] = error(trace, channel::FILTER, &|_| true);
        summary[1] = error(trace, channel::LAG, &|_| true);
        summary[2] = error(trace, channel::RTS, &|_| true);
        summary[3] = error(trace, channel::FILTER, &in_gap);
        summary[4] = error(trace, channel::RTS, &in_gap);
        summary[5] = largest(trace, channel::FILTER + 1);
        summary[6] = largest(trace, channel::RTS + 1);
        summary[7] = error(trace, channel::MEASUREMENT, &|_| true);
        summary[8] = f32::from(u8::from(complete));
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothers_beat_the_filter() {
        smoothing_run(1, 0.8, 4.0, 3.0);
        let s: [f32; 9] = crate::lab::summary();
        assert!(s[2] < s[1] && s[1] < s[0], "{s:?}");
        assert!(s[4] < 0.5 * s[3], "gap: RTS {} vs filter {}", s[4], s[3]);
        assert!(s[6] < s[5], "the RTS σ in the gap is smaller");
        assert_eq!(s[8], 1.0);
    }

    #[test]
    fn extreme_knobs_stay_finite() {
        for args in [(0, 0.0, 0.0, 0.0), (u32::MAX, 99.0, 999.0, 99.0)] {
            smoothing_run(args.0, args.1, args.2, args.3);
            let s: [f32; 3] = crate::lab::summary();
            assert!(s.iter().all(|v| v.is_finite()), "{args:?}: {s:?}");
        }
    }
}
