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
