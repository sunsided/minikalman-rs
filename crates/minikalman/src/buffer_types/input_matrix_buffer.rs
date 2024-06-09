use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::filter_traits::{InputMatrix, InputMatrixMut};
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::IntoInnerData;

pub struct InputMatrixBuffer<const STATES: usize, const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, INPUTS, T>;

pub struct InputMatrixMutBuffer<const STATES: usize, const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, INPUTS, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsRef<[T]>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Matrix<STATES, INPUTS, T>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrix<STATES, INPUTS, T>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsRef<[T]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsMut<[T]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Matrix<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> MatrixMut<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrix<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixMut<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Index<usize>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Index<usize>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> IndexMut<usize>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, T, M> IntoInnerData
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> IntoInnerData
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
