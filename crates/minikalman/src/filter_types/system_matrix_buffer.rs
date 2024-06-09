use crate::filter_traits::{SystemMatrix, SystemMatrixMut};
use crate::filter_types::SystemCovarianceMatrixBuffer;
use crate::matrix_traits::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct SystemMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, STATES, T>;

pub struct SystemMatrixMutBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

// -----------------------------------------------------------

impl<const STATES: usize, T, M> SystemMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T> for SystemMatrixBuffer<STATES, T, M> where
    M: Matrix<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> SystemMatrix<STATES, T> for SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> AsMut<[T]> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T> for SystemMatrixMutBuffer<STATES, T, M> where
    M: MatrixMut<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> MatrixMut<STATES, STATES, T> for SystemMatrixMutBuffer<STATES, T, M> where
    M: MatrixMut<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> SystemMatrix<STATES, T> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> SystemMatrixMut<STATES, T> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T, M> Index<usize> for SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> Index<usize> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
