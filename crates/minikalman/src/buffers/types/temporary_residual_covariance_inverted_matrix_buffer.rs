use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::TemporaryResidualCovarianceInvertedMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for the temporary inverted innovation residual covariance matrix (`num_measurements` × `num_measurements`).
///
/// This matrix represents the inverse of the innovation (residual) covariance matrix, \( S^{-1} \).
/// It quantifies the weight given to the innovation (residual) in the update step of the Kalman Filter.
/// By inverting the innovation covariance matrix, \( S^{-1} \) provides a measure of the certainty
/// of the innovation, allowing the Kalman gain to optimally adjust the state estimate based on
/// the difference between the predicted and actual measurements. This inverse matrix ensures that
/// the filter accurately balances the contributions of the state prediction and the measurement
/// update in minimizing the overall estimation error.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = TemporaryResidualCovarianceInvertedMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = TemporaryResidualCovarianceInvertedMatrixBuffer::<2, f32, _>::from(data.as_mut_slice());
/// ```
pub struct TemporaryResidualCovarianceInvertedMatrixBuffer<const OBSERVATIONS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

// -----------------------------------------------------------

impl<'a, const OBSERVATIONS: usize, T> From<&'a mut [T]>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, OBSERVATIONS, OBSERVATIONS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * OBSERVATIONS <= value.len());
        }
        Self::new(MatrixData::new_mut::<OBSERVATIONS, OBSERVATIONS, T>(value))
    }
}

impl<const OBSERVATIONS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataArray<OBSERVATIONS, OBSERVATIONS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * OBSERVATIONS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<OBSERVATIONS, OBSERVATIONS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, T, M>
    TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        OBSERVATIONS * OBSERVATIONS
    }

    pub const fn is_empty(&self) -> bool {
        OBSERVATIONS * OBSERVATIONS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const OBSERVATIONS: usize, T, M> RowMajorSequentialData<OBSERVATIONS, OBSERVATIONS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const OBSERVATIONS: usize, T, M> RowMajorSequentialDataMut<OBSERVATIONS, OBSERVATIONS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const OBSERVATIONS: usize, T, M> Matrix<OBSERVATIONS, OBSERVATIONS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
}

impl<const OBSERVATIONS: usize, T, M> MatrixMut<OBSERVATIONS, OBSERVATIONS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
}

impl<const OBSERVATIONS: usize, T, M> TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    type Target = M;
    type TargetMut = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T, M> Index<usize>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const OBSERVATIONS: usize, T, M> IndexMut<usize>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, T, M> IntoInnerData
    for TemporaryResidualCovarianceInvertedMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_array() {
        let value: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> =
            data.as_mut_slice().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn test_access() {
        let mut value: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> = [0.0; 25].into();

        // Set values.
        {
            let matrix = value.as_matrix_mut();
            for i in 0..matrix.cols() {
                matrix.set_symmetric(0, i, i as _);
                matrix.set_at(i, i, i as _);
            }
        }

        // Update values.
        for i in 0..value.len() {
            value[i] += 10.0;
        }

        // Get values.
        {
            let matrix = value.as_matrix();
            for i in 0..matrix.rows() {
                assert_eq!(matrix.get_at(0, i), 10.0 + i as f32);
                assert_eq!(matrix.get_at(i, 0), 10.0 + i as f32);
            }
        }

        assert_eq!(value.into_inner(),
                   [
                       10.0, 11.0, 12.0, 13.0, 14.0,
                       11.0, 11.0, 10.0, 10.0, 10.0,
                       12.0, 10.0, 12.0, 10.0, 10.0,
                       13.0, 10.0, 10.0, 13.0, 10.0,
                       14.0, 10.0, 10.0, 10.0, 14.0,
                   ]);
    }
}
