use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::CrossCovarianceMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for cross-covariance matrix (`num_states` × `num_observations`).
///
/// Stores the cross-covariance between state and observation sigma points,
/// used to compute the Kalman gain in the unscented correction step.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::CrossCovarianceMatrixBuffer;
/// use minikalman::prelude::*;
///
/// const NUM_STATES: usize = 3;
/// const NUM_OBS: usize = 2;
///
/// let mut data = [0.0_f32; { NUM_STATES * NUM_OBS }];
/// let buffer = CrossCovarianceMatrixBuffer::<NUM_STATES, NUM_OBS, f32, _>::from(data.as_mut_slice());
/// ```
pub struct CrossCovarianceMatrixBuffer<const STATES: usize, const OBSERVATIONS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, OBSERVATIONS, T>;

impl<const STATES: usize, const OBSERVATIONS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for CrossCovarianceMatrixBuffer<
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

impl<'a, const STATES: usize, const OBSERVATIONS: usize, T> From<&'a mut [T]>
    for CrossCovarianceMatrixBuffer<
        STATES,
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, STATES, OBSERVATIONS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * OBSERVATIONS <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, OBSERVATIONS, T>(value))
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
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

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    RowMajorSequentialData<STATES, OBSERVATIONS, T>
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
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
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> Matrix<STATES, OBSERVATIONS, T>
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> MatrixMut<STATES, OBSERVATIONS, T>
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    CrossCovarianceMatrix<STATES, OBSERVATIONS, T>
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    type Target = M;
    type TargetMut = M;

    #[inline(always)]
    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    #[inline(always)]
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> Index<usize>
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> IndexMut<usize>
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> IntoInnerData
    for CrossCovarianceMatrixBuffer<STATES, OBSERVATIONS, T, M>
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
        const STATES: usize = 3;
        const OBS: usize = 2;
        const TOTAL: usize = STATES * OBS;
        let value: CrossCovarianceMatrixBuffer<STATES, OBS, f32, _> = [0.0; TOTAL].into();
        assert_eq!(value.len(), TOTAL);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut_slice() {
        const STATES: usize = 3;
        const OBS: usize = 2;
        let mut data = [0.0_f32; STATES * OBS];
        let value: CrossCovarianceMatrixBuffer<STATES, OBS, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), STATES * OBS);
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_new_and_accessors() {
        const STATES: usize = 3;
        const OBS: usize = 2;
        let data = MatrixData::new_array::<STATES, OBS, { STATES * OBS }, f32>([1.0; STATES * OBS]);
        let buffer = CrossCovarianceMatrixBuffer::new(data);
        assert_eq!(buffer.len(), STATES * OBS);
        assert!(!buffer.is_empty());
        assert!(buffer.is_valid());
        assert_eq!(buffer[0], 1.0);
    }

    #[test]
    fn test_index_mut() {
        const STATES: usize = 3;
        const OBS: usize = 2;
        let mut data = [0.0_f32; STATES * OBS];
        let mut buffer: CrossCovarianceMatrixBuffer<STATES, OBS, f32, _> =
            data.as_mut_slice().into();
        buffer[3] = 42.0;
        assert_eq!(buffer[3], 42.0);
        assert_eq!(data[3], 42.0);
    }

    #[test]
    fn test_as_matrix_mut() {
        const STATES: usize = 3;
        const OBS: usize = 2;
        let mut data = [0.0_f32; STATES * OBS];
        let mut buffer: CrossCovarianceMatrixBuffer<STATES, OBS, f32, _> =
            data.as_mut_slice().into();
        {
            let mat = buffer.as_matrix_mut();
            mat.set(1, 1, 99.0);
        }
        assert_eq!(buffer[OBS + 1], 99.0);
    }

    #[test]
    fn test_is_empty() {
        let data = MatrixData::new_array::<0, 0, 0, f32>([]);
        let buffer = CrossCovarianceMatrixBuffer::<0, 0, f32, _>::new(data);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_into_inner() {
        const STATES: usize = 3;
        const OBS: usize = 2;
        let arr: [f32; STATES * OBS] = [1.0; STATES * OBS];
        let data = MatrixData::new_array::<STATES, OBS, { STATES * OBS }, f32>(arr);
        let buffer = CrossCovarianceMatrixBuffer::new(data);
        let inner = buffer.into_inner();
        assert_eq!(inner.as_ref()[0], 1.0);
    }
}
