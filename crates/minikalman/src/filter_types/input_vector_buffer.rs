use crate::filter_traits::InputVector;
use crate::filter_types::MeasurementVectorBuffer;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::prelude::InputVectorMut;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct InputVectorBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<INPUTS, 1, T>;

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> AsMut<[T]> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, 1, T> for InputVectorBuffer<INPUTS, T, M> where
    M: MatrixMut<INPUTS, 1, T>
{
}

impl<const INPUTS: usize, T, M> MatrixMut<INPUTS, 1, T> for InputVectorBuffer<INPUTS, T, M> where
    M: MatrixMut<INPUTS, 1, T>
{
}

impl<const INPUTS: usize, T, M> InputVector<INPUTS, T> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputVectorMut<INPUTS, T> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> IndexMut<usize> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
