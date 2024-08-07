use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::TemporaryHPMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// This matrix represents the product of the observation model and the estimate covariance, \( H×P \).
/// It quantifies how the uncertainty in the state estimate propagates into the measurement space.
/// The resulting matrix captures the influence of the current state uncertainty on the predicted
/// measurements, providing an intermediate step in calculating the innovation (residual) covariance matrix.
/// This product helps to incorporate the effects of state estimation uncertainty into the measurement
/// update process, ensuring that the Kalman Filter accurately adjusts the state estimate based on the
/// observed data.
///
/// # See also
/// * [`TemporaryPHTMatrixBuffer`](crate::buffers::types::TemporaryPHTMatrixBuffer).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::TemporaryHPMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = TemporaryHPMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = TemporaryHPMatrixBuffer::<2, 2, f32, _>::from(data.as_mut_slice());
/// ```
pub struct TemporaryHPMatrixBuffer<const OBSERVATIONS: usize, const STATES: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: Matrix<OBSERVATIONS, STATES, T>;

// -----------------------------------------------------------

impl<'a, const OBSERVATIONS: usize, const STATES: usize, T> From<&'a mut [T]>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, MatrixDataMut<'a, OBSERVATIONS, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= value.len());
        }
        Self::new(MatrixData::new_mut::<OBSERVATIONS, STATES, T>(value))
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TemporaryHPMatrixBuffer<
        OBSERVATIONS,
        STATES,
        T,
        MatrixDataArray<OBSERVATIONS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<OBSERVATIONS, STATES, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        OBSERVATIONS * STATES
    }

    pub const fn is_empty(&self) -> bool {
        OBSERVATIONS * STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    RowMajorSequentialData<OBSERVATIONS, STATES, T>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: Matrix<OBSERVATIONS, STATES, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    RowMajorSequentialDataMut<OBSERVATIONS, STATES, T>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> Matrix<OBSERVATIONS, STATES, T>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> MatrixMut<OBSERVATIONS, STATES, T>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    TemporaryHPMatrix<OBSERVATIONS, STATES, T>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
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

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> Index<usize>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> IndexMut<usize>
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> IntoInnerData
    for TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T> + IntoInnerData,
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
        let value: TemporaryHPMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: TemporaryHPMatrixBuffer<5, 3, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: TemporaryHPMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn test_access() {
        let mut value: TemporaryHPMatrixBuffer<5, 5, f32, _> = [0.0; 25].into();

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
