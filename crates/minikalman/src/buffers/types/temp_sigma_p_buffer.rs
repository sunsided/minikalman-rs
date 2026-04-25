use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::TempSigmaPMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for temporary sigma covariance computation (`num_states` × `num_states`).
///
/// Temporary matrix used during sigma point covariance reconstruction.
/// Can be aliased with other P-sized temporaries when memory is constrained.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::TempSigmaPMatrixBuffer;
/// use minikalman::prelude::*;
///
/// const NUM_STATES: usize = 3;
///
/// let mut data = [0.0_f32; { NUM_STATES * NUM_STATES }];
/// let buffer = TempSigmaPMatrixBuffer::<NUM_STATES, f32, _>::from(data.as_mut_slice());
/// ```
pub struct TempSigmaPMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

impl<const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TempSigmaPMatrixBuffer<STATES, T, MatrixDataArray<STATES, STATES, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, STATES, TOTAL, T>(value))
    }
}

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for TempSigmaPMatrixBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, STATES, T>(value))
    }
}

impl<const STATES: usize, T, M> TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * STATES
    }

    pub const fn is_empty(&self) -> bool {
        STATES * STATES == 0
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, T, M> RowMajorSequentialData<STATES, STATES, T>
    for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const STATES: usize, T, M> RowMajorSequentialDataMut<STATES, STATES, T>
    for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T> for TempSigmaPMatrixBuffer<STATES, T, M> where
    M: MatrixMut<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> MatrixMut<STATES, STATES, T>
    for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
}

impl<const STATES: usize, T, M> TempSigmaPMatrix<STATES, T> for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
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

impl<const STATES: usize, T, M> Index<usize> for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const STATES: usize, T, M> IntoInnerData for TempSigmaPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
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
        const TOTAL: usize = STATES * STATES;
        let value: TempSigmaPMatrixBuffer<STATES, f32, _> = [0.0; TOTAL].into();
        assert_eq!(value.len(), TOTAL);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut_slice() {
        const STATES: usize = 3;
        let mut data = [0.0_f32; STATES * STATES];
        let value: TempSigmaPMatrixBuffer<STATES, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), STATES * STATES);
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_new_and_accessors() {
        const STATES: usize = 3;
        let data = MatrixData::new_array::<STATES, STATES, { STATES * STATES }, f32>(
            [1.0; STATES * STATES],
        );
        let buffer = TempSigmaPMatrixBuffer::new(data);
        assert_eq!(buffer.len(), STATES * STATES);
        assert!(!buffer.is_empty());
        assert!(buffer.is_valid());
        assert_eq!(buffer[0], 1.0);
    }

    #[test]
    fn test_index_mut() {
        const STATES: usize = 3;
        let mut data = [0.0_f32; STATES * STATES];
        let mut buffer: TempSigmaPMatrixBuffer<STATES, f32, _> = data.as_mut_slice().into();
        buffer[4] = 42.0;
        assert_eq!(buffer[4], 42.0);
        assert_eq!(data[4], 42.0);
    }

    #[test]
    fn test_as_matrix_mut() {
        const STATES: usize = 3;
        let mut data = [0.0_f32; STATES * STATES];
        let mut buffer: TempSigmaPMatrixBuffer<STATES, f32, _> = data.as_mut_slice().into();
        {
            let mat = buffer.as_matrix_mut();
            mat.set(1, 1, 99.0);
        }
        assert_eq!(buffer[STATES + 1], 99.0);
    }

    #[test]
    fn test_is_empty() {
        let data = MatrixData::new_array::<0, 0, 0, f32>([]);
        let buffer = TempSigmaPMatrixBuffer::<0, f32, _>::new(data);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_into_inner() {
        const STATES: usize = 3;
        let arr: [f32; STATES * STATES] = [1.0; STATES * STATES];
        let data = MatrixData::new_array::<STATES, STATES, { STATES * STATES }, f32>(arr);
        let buffer = TempSigmaPMatrixBuffer::new(data);
        let inner = buffer.into_inner();
        assert_eq!(inner.as_ref()[0], 1.0);
    }
}
