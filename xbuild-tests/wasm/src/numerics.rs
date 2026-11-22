//! Numerics: the standard and the Joseph covariance update in `f32`, `I16F16` and `I32F32`.
//!
//! The scenario of the crate's `numerical_robustness` test bench: a 2D constant-velocity
//! model (`x, y, vx, vy`) with up to ten scalar position sensors along directions 18° apart,
//! applied one after another between two predictions. With precise sensors the steady-state
//! position variance drops below the resolution of `I16F16` (`2⁻¹⁶ ≈ 1.5·10⁻⁵`). After the
//! last correction of each step, the run records how far `P` is from symmetric, its smallest
//! diagonal entry, whether it still has a Cholesky factor in its own type, and the position
//! error.
//!
//! Fixed-point overflow wraps in this release build rather than panicking.

use fixed::types::{I16F16, I32F32};
use minikalman::matrix::MatrixDataArray;
use minikalman::num_traits::ToPrimitive;
use minikalman::prelude::*;
use minikalman::regular::{FilterStorage, ObservationStorage};

use crate::lab::{Noise, Trace, knob, record};

/// Time step in seconds.
const DT: f64 = 0.1;

/// The largest number of sensors.
const MAX_SENSORS: usize = 10;

/// The largest number of steps.
const MAX_STEPS: usize = 600;

/// The trace channels of one variant; variant `v` starts at `v × PER_VARIANT`.
pub mod channel {
    /// The largest relative asymmetry `|Pij − Pji| / √(Pii·Pjj)`.
    pub const ASYMMETRY: usize = 0;
    /// The smallest diagonal entry of `P`.
    pub const MIN_DIAGONAL: usize = 1;
    /// `1` if `P` has a Cholesky factor in its own type, else `0`.
    pub const POSITIVE_DEFINITE: usize = 2;
    /// The position error in m.
    pub const ERROR: usize = 3;
    /// Number of channels per variant.
    pub const PER_VARIANT: usize = 4;
}

/// Number of variants: `f32`, `I16F16` and `I32F32`, each standard and Joseph.
const VARIANTS: usize = 6;

/// The settings of a run.
#[derive(Clone, Copy)]
struct Settings {
    /// Number of steps.
    steps: usize,
    /// Number of scalar sensors per step.
    sensors: usize,
    /// Measurement noise variance of each sensor.
    r: f64,
    /// Process noise variance per state and step.
    q: f64,
    /// Noise seed.
    seed: u32,
}

/// Converts a constant to `T`, saturating where it does not fit.
fn num<T: MatrixDataType + ToPrimitive>(value: f64) -> T {
    T::from_f64(value).unwrap_or_else(|| T::from_f64(value.signum() * 1e4).unwrap_or(T::zero()))
}

/// Converts `value` to `f64`.
fn to_f64<T: ToPrimitive>(value: T) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

/// Runs one variant in type `T` and writes its channels from `base` on.
///
/// Returns `(steps with P not positive definite, first such step or -1, largest asymmetry,
/// final position error)`.
#[allow(clippy::cast_precision_loss)]
fn run<T>(
    settings: Settings,
    update: CovarianceUpdate,
    trace: &mut Trace<'_>,
    base: usize,
) -> [f32; 4]
where
    T: MatrixDataType + ToPrimitive,
{
    let mut storage = FilterStorage::<4, T>::new(T::zero());
    let mut sensor_storage: [ObservationStorage<4, 1, T>; MAX_SENSORS] =
        core::array::from_fn(|_| ObservationStorage::new(T::zero()));
    let mut filter = storage.build();
    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set(0, 2, num(DT));
        a.set(1, 3, num(DT));
    });
    filter
        .direct_process_noise_mut()
        .apply(|n| (0..4).for_each(|i| n.set(i, i, num(settings.q))));
    filter.estimate_covariance_mut().make_scalar(num(10.0));

    let directions: [(f64, f64); MAX_SENSORS] =
        core::array::from_fn(|k| (k as f64 * 18f64).to_radians().sin_cos());
    let mut sensors = sensor_storage.each_mut().map(ObservationStorage::build);
    for (sensor, &(sin, cos)) in sensors.iter_mut().zip(&directions) {
        sensor.set_covariance_update(update);
        sensor.observation_matrix_mut().apply(|h| {
            h.set(0, 0, num(cos));
            h.set(0, 1, num(sin));
        });
        sensor
            .measurement_noise_covariance_mut()
            .make_scalar(num(settings.r));
    }

    let mut noise = Noise::new(settings.seed);
    let (mut indefinite, mut first, mut worst, mut error) = (0u32, -1.0f32, 0.0f64, 0.0f64);
    for step in 0..settings.steps {
        filter.predict();
        let t = DT * step as f64;
        let (px, py) = (0.3 * t, -0.2 * t);
        for (sensor, &(sin, cos)) in sensors.iter_mut().zip(&directions).take(settings.sensors) {
            let reading = cos * px + sin * py + settings.r.sqrt() * noise.gaussian();
            sensor.measurement_vector_mut().set(0, 0, num(reading));
            filter.correct(sensor);
        }

        let p = filter.estimate_covariance();
        let get = |i: usize, j: usize| to_f64(p.get(i, j));
        let mut asymmetry = 0.0f64;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let scale = (get(i, i) * get(j, j)).abs().sqrt();
                if scale > 0.0 {
                    asymmetry = asymmetry.max((get(i, j) - get(j, i)).abs() / scale);
                }
            }
        }
        let min_diagonal = (0..4).map(|i| get(i, i)).fold(f64::INFINITY, f64::min);
        let mut copy = MatrixDataArray::<4, 4, 16, T>::new_unchecked([T::zero(); 16]);
        p.inspect(|m| m.copy_to(&mut copy));
        let positive_definite = copy.cholesky_decompose_lower();
        let x = filter.state_vector();
        error = (to_f64(x.get(0, 0)) - px).hypot(to_f64(x.get(1, 0)) - py);

        trace.set(base + channel::ASYMMETRY, step, asymmetry as f32);
        trace.set(base + channel::MIN_DIAGONAL, step, min_diagonal as f32);
        trace.set(
            base + channel::POSITIVE_DEFINITE,
            step,
            f32::from(u8::from(positive_definite)),
        );
        trace.set(base + channel::ERROR, step, error as f32);
        if !positive_definite {
            indefinite += 1;
            if first < 0.0 {
                first = step as f32;
            }
        }
        worst = worst.max(asymmetry);
    }

    [indefinite as f32, first, worst as f32, error as f32]
}

/// Runs all six variants on the same readings and writes them to the trace.
///
/// Variants, in order: `f32` standard, `f32` Joseph, `I16F16` standard, `I16F16` Joseph,
/// `I32F32` standard, `I32F32` Joseph. Channels: four per variant, see [`channel`].
/// Summary: four values per variant, `[steps with P not positive definite, first such step
/// (−1 for none), largest asymmetry, final position error]`.
///
/// * `seed` - Noise seed.
/// * `sensors` - Scalar sensors per step, `2..=10` (one cannot observe both axes).
/// * `log_r` - `log₁₀` of each sensor's noise variance, `−6..=0`.
/// * `log_q` - `log₁₀` of the process noise variance per state and step, `−8..=−1`.
/// * `steps` - Number of steps, `10..=600`.
#[cfg_attr(not(test), unsafe(no_mangle))]
pub extern "C" fn numerics_run(seed: u32, sensors: u32, log_r: f32, log_q: f32, steps: u32) -> u32 {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let settings = Settings {
        steps: knob(steps as f32, 10.0, MAX_STEPS as f32) as usize,
        sensors: knob(sensors as f32, 2.0, MAX_SENSORS as f32) as usize,
        r: 10f64.powf(f64::from(knob(log_r, -6.0, 0.0))),
        q: 10f64.powf(f64::from(knob(log_q, -8.0, -1.0))),
        seed,
    };

    record(
        channel::PER_VARIANT * VARIANTS,
        settings.steps,
        |trace, summary| {
            let updates = [CovarianceUpdate::Standard, CovarianceUpdate::Joseph];
            for (u, update) in updates.into_iter().enumerate() {
                let results = [
                    run::<f32>(settings, update, trace, u * channel::PER_VARIANT),
                    run::<I16F16>(settings, update, trace, (2 + u) * channel::PER_VARIANT),
                    run::<I32F32>(settings, update, trace, (4 + u) * channel::PER_VARIANT),
                ];
                for (t, result) in results.iter().enumerate() {
                    let v = 2 * t + u;
                    summary[4 * v..4 * v + 4].copy_from_slice(result);
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_point_and_i32f32_stay_positive_definite() {
        numerics_run(1, 4, -2.0, -5.0, 100);
        let s: [f32; 24] = crate::lab::summary();
        for v in [0, 1, 4, 5] {
            assert_eq!(s[4 * v], 0.0, "variant {v}: {s:?}");
            assert!(s[4 * v + 3] < 0.2, "variant {v} error {}", s[4 * v + 3]);
        }
        // The Joseph form is exactly symmetric.
        assert_eq!(s[4 + 2], 0.0);
        assert_eq!(s[5 * 4 + 2], 0.0);
        // The standard form is not, in f32.
        assert!(s[2] > 0.0);
    }
}
