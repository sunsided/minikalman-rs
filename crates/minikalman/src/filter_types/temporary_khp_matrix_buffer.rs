use crate::filter_traits::{ResidualCovarianceMatrix, TemporaryKHPMatrix};
use crate::filter_types::SystemMatrixMutBuffer;
use crate::matrix_traits::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct TemporaryKHPMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

// -----------------------------------------------------------

impl<const STATES: usize, T, M> TemporaryKHPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for TemporaryKHPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T> for TemporaryKHPMatrixBuffer<STATES, T, M> where
    M: MatrixMut<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> TemporaryKHPMatrix<STATES, T>
    for TemporaryKHPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
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

impl<const STATES: usize, T, M> Index<usize> for TemporaryKHPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for TemporaryKHPMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
