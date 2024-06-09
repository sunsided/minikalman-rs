use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::filter_traits::TemporaryBQMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};

pub struct TemporaryBQMatrixBuffer<const STATES: usize, const INPUTS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, INPUTS, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, T, M> TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsRef<[T]>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsMut<[T]>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Matrix<STATES, INPUTS, T>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> MatrixMut<STATES, INPUTS, T>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> TemporaryBQMatrix<STATES, INPUTS, T>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
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

impl<const STATES: usize, const INPUTS: usize, T, M> Index<usize>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> IndexMut<usize>
    for TemporaryBQMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
