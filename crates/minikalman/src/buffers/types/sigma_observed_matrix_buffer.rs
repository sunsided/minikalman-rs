use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::SigmaObservedMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Mutable buffer for observed sigma points (`num_observations` × `num_sigma_points`).
///
/// Stores the sigma points after they have been propagated through the nonlinear
/// observation function into measurement space.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::SigmaObservedMatrixBuffer;
/// use minikalman::prelude::*;
///
/// const NUM_OBS: usize = 2;
/// const NUM_SIGMA: usize = 7;
///
/// let mut data = [0.0_f32; { NUM_OBS * NUM_SIGMA }];
/// let buffer = SigmaObservedMatrixBuffer::<NUM_OBS, NUM_SIGMA, f32, _>::from(data.as_mut_slice());
/// ```
pub struct SigmaObservedMatrixBuffer<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>;

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for SigmaObservedMatrixBuffer<
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        MatrixDataArray<OBSERVATIONS, NUM_SIGMA, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * NUM_SIGMA <= TOTAL);
        }
        Self::new(MatrixData::new_array::<OBSERVATIONS, NUM_SIGMA, TOTAL, T>(
            value,
        ))
    }
}

impl<'a, const OBSERVATIONS: usize, const NUM_SIGMA: usize, T> From<&'a mut [T]>
    for SigmaObservedMatrixBuffer<
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        MatrixDataMut<'a, OBSERVATIONS, NUM_SIGMA, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * NUM_SIGMA <= value.len());
        }
        Self::new(MatrixData::new_mut::<OBSERVATIONS, NUM_SIGMA, T>(value))
    }
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M>
    SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        OBSERVATIONS * NUM_SIGMA
    }

    pub const fn is_empty(&self) -> bool {
        OBSERVATIONS * NUM_SIGMA == 0
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M>
    RowMajorSequentialData<OBSERVATIONS, NUM_SIGMA, T>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M>
    RowMajorSequentialDataMut<OBSERVATIONS, NUM_SIGMA, T>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M> Matrix<OBSERVATIONS, NUM_SIGMA, T>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M> MatrixMut<OBSERVATIONS, NUM_SIGMA, T>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M>
    SigmaObservedMatrix<OBSERVATIONS, NUM_SIGMA, T>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
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

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M> Index<usize>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M> IndexMut<usize>
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const OBSERVATIONS: usize, const NUM_SIGMA: usize, T, M> IntoInnerData
    for SigmaObservedMatrixBuffer<OBSERVATIONS, NUM_SIGMA, T, M>
where
    M: MatrixMut<OBSERVATIONS, NUM_SIGMA, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
