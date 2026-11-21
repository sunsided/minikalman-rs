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
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};
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
    let Q = Q.as_matrix_mut();

    for i in 0..STATES {
        for j in 0..=i {
            let value = alpha * Q.get(i, j) + weight * dx[i] * dx[j];
            Q.set(i, j, value);
            Q.set(j, i, value);
        }
    }

    for i in 0..STATES {
        if Q.get(i, i) < adaptation.floor {
            Q.set(i, i, adaptation.floor);
        }
    }
}
