use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{SigmaWeightsVector, SigmaWeightsVectorMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{AsMatrix, AsMatrixMut, RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for sigma point weights (`num_sigma_points`).
///
/// Stores the covariance weights (W_c) used for combining sigma points during
/// the unscented transform. The mean weight for the first sigma point (W_m0)
/// is computed separately from `lambda / (n + lambda)` and not stored here.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::SigmaWeightsVectorBuffer;
/// use minikalman::prelude::*;
///
/// const NUM_SIGMA: usize = 7;
///
/// // From owned data
/// let buffer = SigmaWeightsVectorBuffer::new(MatrixData::new_array::<NUM_SIGMA, 1, NUM_SIGMA, f32>([0.0; NUM_SIGMA]));
///
/// // From a reference
/// let mut data = [0.0_f32; NUM_SIGMA];
/// let buffer = SigmaWeightsVectorBuffer::<NUM_SIGMA, f32, _>::from(data.as_mut_slice());
/// ```
pub struct SigmaWeightsVectorBuffer<const NUM_SIGMA: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<NUM_SIGMA, 1, T>;

impl<const NUM_SIGMA: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, MatrixDataArray<NUM_SIGMA, 1, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(NUM_SIGMA <= TOTAL);
        }
        Self::new(MatrixData::new_array::<NUM_SIGMA, 1, TOTAL, T>(value))
    }
}

impl<'a, const NUM_SIGMA: usize, T> From<&'a mut [T]>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, MatrixDataMut<'a, NUM_SIGMA, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(NUM_SIGMA <= value.len());
        }
        Self::new(MatrixData::new_mut::<NUM_SIGMA, 1, T>(value))
    }
}

impl<const NUM_SIGMA: usize, T, M> SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        NUM_SIGMA
    }

    pub const fn is_empty(&self) -> bool {
        NUM_SIGMA == 0
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const NUM_SIGMA: usize, T, M> RowMajorSequentialData<NUM_SIGMA, 1, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const NUM_SIGMA: usize, T, M> RowMajorSequentialDataMut<NUM_SIGMA, 1, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const NUM_SIGMA: usize, T, M> Matrix<NUM_SIGMA, 1, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
}

impl<const NUM_SIGMA: usize, T, M> MatrixMut<NUM_SIGMA, 1, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
}

impl<const NUM_SIGMA: usize, T, M> AsMatrix<NUM_SIGMA, 1, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    type Target = M;

    #[inline(always)]
    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const NUM_SIGMA: usize, T, M> AsMatrixMut<NUM_SIGMA, 1, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    type TargetMut = M;

    #[inline(always)]
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const NUM_SIGMA: usize, T, M> SigmaWeightsVector<NUM_SIGMA, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
}

impl<const NUM_SIGMA: usize, T, M> SigmaWeightsVectorMut<NUM_SIGMA, T>
    for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
}

impl<const NUM_SIGMA: usize, T, M> Index<usize> for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const NUM_SIGMA: usize, T, M> IndexMut<usize> for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const NUM_SIGMA: usize, T, M> IntoInnerData for SigmaWeightsVectorBuffer<NUM_SIGMA, T, M>
where
    M: MatrixMut<NUM_SIGMA, 1, T> + IntoInnerData,
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
        const N: usize = 7;
        let value: SigmaWeightsVectorBuffer<N, f32, _> = [0.0; N].into();
        assert_eq!(value.len(), N);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        const N: usize = 7;
        let mut data = [0.0_f32; N];
        let value: SigmaWeightsVectorBuffer<N, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), N);
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }
}
