use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::KalmanGainMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for the Kalman Gain matrix (`num_states` × `num_measurements`), typically denoted "K".
///
/// Determines how much the predictions should be corrected based on the measurements.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::KalmanGainMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = KalmanGainMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = KalmanGainMatrixBuffer::<2, 2, f32, _>::from(data.as_mut_slice());
/// ```
pub struct KalmanGainMatrixBuffer<const STATES: usize, const OBSERVATIONS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, OBSERVATIONS, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, const OBSERVATIONS: usize, T> From<&'a mut [T]>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, MatrixDataMut<'a, STATES, OBSERVATIONS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * OBSERVATIONS <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, OBSERVATIONS, T>(value))
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for KalmanGainMatrixBuffer<
        STATES,
        OBSERVATIONS,
        T,
        MatrixDataArray<STATES, OBSERVATIONS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * OBSERVATIONS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, OBSERVATIONS, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * OBSERVATIONS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * OBSERVATIONS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    RowMajorSequentialData<STATES, OBSERVATIONS, T>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    RowMajorSequentialDataMut<STATES, OBSERVATIONS, T>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> Matrix<STATES, OBSERVATIONS, T>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> MatrixMut<STATES, OBSERVATIONS, T>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> KalmanGainMatrix<STATES, OBSERVATIONS, T>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
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

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> Index<usize>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> IndexMut<usize>
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> IntoInnerData
    for KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T> + IntoInnerData,
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
        let value: KalmanGainMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: KalmanGainMatrixBuffer<5, 3, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: KalmanGainMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn test_access() {
        let mut value: KalmanGainMatrixBuffer<5, 5, f32, _> = [0.0; 25].into();

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
