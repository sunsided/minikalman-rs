use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::SigmaPropagatedMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{AsMatrix, AsMatrixMut, RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for propagated sigma points after state transition (`num_states` × `num_sigma_points`).
///
/// Stores the sigma points after they have been propagated through the nonlinear
/// state transition function.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::SigmaPropagatedMatrixBuffer;
/// use minikalman::prelude::*;
///
/// const NUM_STATES: usize = 3;
/// const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
///
/// let mut data = [0.0_f32; { NUM_STATES * NUM_SIGMA }];
/// let buffer = SigmaPropagatedMatrixBuffer::<NUM_STATES, NUM_SIGMA, f32, _>::from(data.as_mut_slice());
/// ```
pub struct SigmaPropagatedMatrixBuffer<const STATES: usize, const NUM_SIGMA: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, NUM_SIGMA, T>;

impl<const STATES: usize, const NUM_SIGMA: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for SigmaPropagatedMatrixBuffer<
        STATES,
        NUM_SIGMA,
        T,
        MatrixDataArray<STATES, NUM_SIGMA, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * NUM_SIGMA <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, NUM_SIGMA, TOTAL, T>(value))
    }
}

impl<'a, const STATES: usize, const NUM_SIGMA: usize, T> From<&'a mut [T]>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, MatrixDataMut<'a, STATES, NUM_SIGMA, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * NUM_SIGMA <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, NUM_SIGMA, T>(value))
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M>
    SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * NUM_SIGMA
    }

    pub const fn is_empty(&self) -> bool {
        STATES * NUM_SIGMA == 0
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> RowMajorSequentialData<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M>
    RowMajorSequentialDataMut<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> Matrix<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> MatrixMut<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> SigmaPropagatedMatrix<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
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

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> AsMatrix<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    type Target = M;
    #[inline(always)]
    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> AsMatrixMut<STATES, NUM_SIGMA, T>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    type TargetMut = M;
    #[inline(always)]
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> Index<usize>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> IndexMut<usize>
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T, M> IntoInnerData
    for SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, M>
where
    M: MatrixMut<STATES, NUM_SIGMA, T> + IntoInnerData,
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
        const SIGMA: usize = 7;
        const TOTAL: usize = STATES * SIGMA;
        let value: SigmaPropagatedMatrixBuffer<STATES, SIGMA, f32, _> = [0.0; TOTAL].into();
        assert_eq!(value.len(), TOTAL);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut_slice() {
        const STATES: usize = 3;
        const SIGMA: usize = 7;
        let mut data = [0.0_f32; STATES * SIGMA];
        let value: SigmaPropagatedMatrixBuffer<STATES, SIGMA, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), STATES * SIGMA);
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_new_and_accessors() {
        const STATES: usize = 3;
        const SIGMA: usize = 7;
        let data =
            MatrixData::new_array::<STATES, SIGMA, { STATES * SIGMA }, f32>([1.0; STATES * SIGMA]);
        let buffer = SigmaPropagatedMatrixBuffer::new(data);
        assert_eq!(buffer.len(), STATES * SIGMA);
        assert!(!buffer.is_empty());
        assert!(buffer.is_valid());
        assert_eq!(buffer[0], 1.0);
    }

    #[test]
    fn test_index_mut() {
        const STATES: usize = 3;
        const SIGMA: usize = 7;
        let mut data = [0.0_f32; STATES * SIGMA];
        let mut buffer: SigmaPropagatedMatrixBuffer<STATES, SIGMA, f32, _> =
            data.as_mut_slice().into();
        buffer[5] = 42.0;
        assert_eq!(buffer[5], 42.0);
        assert_eq!(data[5], 42.0);
    }

    #[test]
    fn test_as_matrix() {
        const STATES: usize = 3;
        const SIGMA: usize = 7;
        let data =
            MatrixData::new_array::<STATES, SIGMA, { STATES * SIGMA }, f32>([5.0; STATES * SIGMA]);
        let buffer = SigmaPropagatedMatrixBuffer::new(data);
        let mat = AsMatrix::as_matrix(&buffer);
        assert_eq!(mat.get(0, 0), 5.0);
    }

    #[test]
    fn test_as_matrix_mut() {
        const STATES: usize = 3;
        const SIGMA: usize = 7;
        let mut data = [0.0_f32; STATES * SIGMA];
        let mut buffer: SigmaPropagatedMatrixBuffer<STATES, SIGMA, f32, _> =
            data.as_mut_slice().into();
        {
            let mat = AsMatrixMut::as_matrix_mut(&mut buffer);
            mat.set(1, 2, 99.0);
        }
        assert_eq!(buffer[SIGMA + 2], 99.0);
    }

    #[test]
    fn test_is_empty() {
        let data = MatrixData::new_array::<0, 0, 0, f32>([]);
        let buffer = SigmaPropagatedMatrixBuffer::<0, 0, f32, _>::new(data);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_into_inner() {
        const STATES: usize = 3;
        const SIGMA: usize = 7;
        let arr: [f32; STATES * SIGMA] = [1.0; STATES * SIGMA];
        let data = MatrixData::new_array::<STATES, SIGMA, { STATES * SIGMA }, f32>(arr);
        let buffer = SigmaPropagatedMatrixBuffer::new(data);
        let inner = buffer.into_inner();
        assert_eq!(inner.as_ref()[0], 1.0);
    }
}
