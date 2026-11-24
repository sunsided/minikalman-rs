//! Storage for the recursive least-squares estimator.

use super::layout::flat;
use crate::matrix::{MatrixData, MatrixDataMut, MatrixDataType};
use crate::rls::RecursiveLeastSquares;

/// A [`RecursiveLeastSquares`] estimator whose buffers borrow an [`EstimatorStorage`].
///
/// The [`Forgetting`](crate::rls::Forgetting) mode `F` defaults to
/// [`Exponential`](crate::rls::Exponential).
pub type BorrowedEstimator<'a, const N: usize, T, F = crate::rls::Exponential> =
    RecursiveLeastSquares<
        N,
        T,
        MatrixDataMut<'a, N, 1, T>,
        MatrixDataMut<'a, N, N, T>,
        MatrixDataMut<'a, N, 1, T>,
        MatrixDataMut<'a, N, 1, T>,
        F,
    >;

/// Storage for the buffers of a [`RecursiveLeastSquares`] estimator of `N` parameters.
///
/// Holds `θ`, `P` (or its UD factors) and two scratch vectors as nested arrays,
/// `N² + 3·N` values in total, and allocates nothing. [`new`](Self::new) is a `const fn`, so
/// the storage can live in a `static`; see
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
///
/// ## Example
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
/// use minikalman::prelude::*;
/// use minikalman::rls::EstimatorStorage;
///
/// // N² + 3·N values, nothing else.
/// assert_eq!(core::mem::size_of::<EstimatorStorage<3, f32>>(), 18 * 4);
///
/// let mut storage = EstimatorStorage::<3, f32>::new(0.0);
/// let mut rls = storage.build(100.0);
/// assert!(rls.update(&[1.0, 0.0, 0.0], 2.0).is_applied());
/// assert!(rls.parameters().get_row(0) > 1.9);
/// # }
/// ```
pub struct EstimatorStorage<const N: usize, T> {
    /// The parameter vector `θ`.
    theta: [[T; 1]; N],
    /// The covariance `P` or its UD factors.
    p: [[T; N]; N],
    /// Scratch vector `Uᵀ·φ` or `P·φ`.
    temp_f: [[T; 1]; N],
    /// Scratch vector for the gain.
    temp_k: [[T; 1]; N],
}

impl<const N: usize, T: Copy> EstimatorStorage<N, T> {
    /// Creates the storage with every entry set to `init`, e.g. `0.0` or `I16F16::ZERO`.
    pub const fn new(init: T) -> Self {
        Self {
            theta: [[init; 1]; N],
            p: [[init; N]; N],
            temp_f: [[init; 1]; N],
            temp_k: [[init; 1]; N],
        }
    }
}

impl<const N: usize, T: MatrixDataType> EstimatorStorage<N, T> {
    /// Wires a [`RecursiveLeastSquares`] estimator whose buffers borrow this storage, and
    /// resets it to `θ = 0` and `P = p0·I`, see [`RecursiveLeastSquares::new`].
    pub fn build(&mut self, p0: T) -> BorrowedEstimator<'_, N, T> {
        RecursiveLeastSquares::new(
            MatrixData::new_mut::<N, 1, T>(flat(&mut self.theta)),
            MatrixData::new_mut::<N, N, T>(flat(&mut self.p)),
            MatrixData::new_mut::<N, 1, T>(flat(&mut self.temp_f)),
            MatrixData::new_mut::<N, 1, T>(flat(&mut self.temp_k)),
            p0,
        )
    }
}
