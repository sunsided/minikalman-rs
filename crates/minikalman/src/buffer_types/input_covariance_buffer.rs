use crate::filter_traits::{InputCovarianceMatrix, InputCovarianceMatrixMut};
use crate::matrix_traits::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct InputCovarianceMatrixBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<INPUTS, INPUTS, T>;

pub struct InputCovarianceMatrixMutBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<INPUTS, INPUTS, T>;

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, INPUTS, T>
    for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrix<INPUTS, T>
    for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> AsMut<[T]> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> MatrixMut<INPUTS, INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrix<INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrixMut<INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> IndexMut<usize> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
