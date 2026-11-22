//! Online adaptation of the process noise covariance `Q` from the state corrections.
//!
//! Shared by the regular, extended and unscented filters, enabled with their
//! `set_process_noise_adaptation`. After every applied correction,
//!
//! ```text
//! Δx = x⁺ − x⁻ = K×y
//! Q ← α×Q + (1−α)×Δx×Δxᵀ
//! ```
//!
//! then every diagonal entry of `Q` is raised to at least the floor.
//!
//! See [`NoiseAdaptation`] for the method, how `Q` stays positive semi-definite, and its
//! limits.

use crate::kalman::{CorrectionStatus, DirectProcessNoiseCovarianceMatrixMut, StateVectorMut};
use crate::matrix::{Matrix, MatrixDataType, RowMajorSequentialDataMut};
use crate::observations::NoiseAdaptation;

/// The process noise adaptation of a filter: the settings, plus the update of `Q`.
///
/// The update is a function pointer so that the filters' `correct*` methods can adapt `Q`
/// without requiring a mutable `Q` type: it is only created by [`ProcessNoiseAdapter::new`],
/// where `Q` is known to be mutable.
pub(crate) struct ProcessNoiseAdapter<const STATES: usize, T, Q> {
    /// The forgetting factor and the diagonal floor.
    settings: NoiseAdaptation<T>,

    /// Blends the state correction `Δx` into `Q`; [`adapt_process_noise`] for the filter's `Q`.
    update: fn(&mut Q, &[T; STATES], NoiseAdaptation<T>),
}

impl<const STATES: usize, T, Q> Clone for ProcessNoiseAdapter<STATES, T, Q>
where
    T: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<const STATES: usize, T, Q> Copy for ProcessNoiseAdapter<STATES, T, Q> where T: Copy {}

impl<const STATES: usize, T, Q> ProcessNoiseAdapter<STATES, T, Q>
where
    T: MatrixDataType,
{
    /// Creates the adaptation of the mutable `Q` with the given settings.
    pub(crate) fn new(settings: NoiseAdaptation<T>) -> Self
    where
        Q: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
    {
        Self {
            settings,
            update: adapt_process_noise::<STATES, T, Q>,
        }
    }

    /// Returns the forgetting factor and the diagonal floor.
    pub(crate) fn settings(&self) -> NoiseAdaptation<T> {
        self.settings
    }
}

/// Runs `correct` on `x` and, if it was applied and `adapter` is set, adapts `Q` from the state
/// correction `Δx = x⁺ − x⁻`.
///
/// Without an adapter, this is exactly `correct(x)`: `x` is not copied.
#[allow(non_snake_case)]
pub(crate) fn correct_adapting_process_noise<const STATES: usize, T, X, Q>(
    adapter: Option<ProcessNoiseAdapter<STATES, T, Q>>,
    x: &mut X,
    Q: &mut Q,
    correct: impl FnOnce(&mut X) -> CorrectionStatus,
) -> CorrectionStatus
where
    T: MatrixDataType,
    X: StateVectorMut<STATES, T>,
{
    let Some(adapter) = adapter else {
        return correct(x);
    };

    // The copies stay on get: as slice copies, they shifted the code of the inlined correction
    // on Cortex-M4F at opt-level 3 (benches/mcu: +5 instructions for the 2×1 step) although
    // they only run with the adaptation on.
    let mut dx = [T::zero(); STATES];
    for (i, d) in dx.iter_mut().enumerate() {
        *d = x.as_matrix().get(i, 0);
    }

    let status = correct(x);
    if !status.is_applied() {
        return status;
    }

    for (i, d) in dx.iter_mut().enumerate() {
        *d = x.as_matrix().get(i, 0) - *d;
    }

    (adapter.update)(Q, &dx, adapter.settings);
    status
}

/// Adapts `Q ← α×Q + (1−α)×Δx×Δxᵀ`, then raises every diagonal entry to at least the floor.
///
/// Writes the lower triangle and mirrors it, so `Q` stays symmetric. See the
/// [module documentation](self).
///
/// With fixed-point types, `Δx×Δxᵀ` can overflow for large corrections; that panics in debug
/// builds and wraps in release builds, as for `ε×εᵀ` in the update of `R`.
#[allow(non_snake_case)]
fn adapt_process_noise<const STATES: usize, T, Q>(
    Q: &mut Q,
    dx: &[T; STATES],
    adaptation: NoiseAdaptation<T>,
) where
    T: MatrixDataType,
    Q: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
{
    let alpha = adaptation.forgetting;
    let weight = T::one() - alpha;
    // A slice of the static length, so that the loops need no bounds checks.
    let q = &mut Q.as_matrix_mut().as_mut_slice()[..STATES * STATES];

    for i in 0..STATES {
        for j in 0..=i {
            let value = alpha * q[i * STATES + j] + weight * dx[i] * dx[j];
            q[i * STATES + j] = value;
            q[j * STATES + i] = value;
        }
    }

    crate::observations::raise_diagonal_to_floor::<STATES, T>(q, adaptation.floor);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::types::DirectProcessNoiseCovarianceMatrixMutBuffer;
    use crate::matrix::{MatrixData, MatrixDataMut, MatrixMut};
    use crate::test_bits::{ToBits, assert_bits_eq, test_value};

    /// The `get`/`set` loops that [`adapt_process_noise`] replaced.
    #[allow(non_snake_case)]
    fn old_adapt<const N: usize, T: MatrixDataType>(
        Q: &mut impl MatrixMut<N, N, T>,
        dx: &[T; N],
        adaptation: NoiseAdaptation<T>,
    ) {
        let alpha = adaptation.forgetting;
        let weight = T::one() - alpha;
        for i in 0..N {
            for j in 0..=i {
                let value = alpha * Q.get(i, j) + weight * dx[i] * dx[j];
                Q.set(i, j, value);
                Q.set(j, i, value);
            }
        }

        for i in 0..N {
            if Q.get(i, i) < adaptation.floor {
                Q.set(i, i, adaptation.floor);
            }
        }
    }

    /// Runs [`adapt_process_noise`] and the old loops on the same `Q` and `Δx` and demands
    /// bit-identical results.
    fn assert_matches_old<const N: usize, T: MatrixDataType + ToBits>(seed: u32, scale: f64) {
        let c = |i: usize| T::from_f64(test_value(seed, i) * scale).unwrap();
        let adaptation = NoiseAdaptation {
            forgetting: T::from_f64(0.75).unwrap(),
            floor: T::from_f64(0.25 * scale).unwrap(),
        };
        let mut dx: [T; N] = core::array::from_fn(|i| c(100 + i));
        dx[0] = T::zero();
        let initial: [T; 36] = core::array::from_fn(c);

        let (mut new, mut old) = (initial, initial);
        let mut q: DirectProcessNoiseCovarianceMatrixMutBuffer<N, T, MatrixDataMut<'_, N, N, T>> =
            (&mut new[..]).into();
        adapt_process_noise(&mut q, &dx, adaptation);
        old_adapt(
            &mut MatrixData::new_mut::<N, N, T>(&mut old),
            &dx,
            adaptation,
        );
        assert_bits_eq(&new, &old, "Q");
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    fn adapt_process_noise_matches_the_old_loops() {
        for seed in 0..4 {
            assert_matches_old::<2, f32>(seed, 1.0);
            assert_matches_old::<4, f32>(seed, 1.0);
            assert_matches_old::<6, f64>(seed, 1.0);
        }
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn adapt_process_noise_matches_the_old_loops_fixed() {
        for seed in 0..4 {
            assert_matches_old::<4, fixed::types::I16F16>(seed, 0.5);
            assert_matches_old::<6, fixed::types::I32F32>(seed, 1.0);
        }
    }
}
