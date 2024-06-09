use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::filter_traits::InputVector;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::prelude::InputVectorMut;
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned};

pub struct InputVectorBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<INPUTS, 1, T>;

// -----------------------------------------------------------

impl<'a, const INPUTS: usize, T> From<&'a mut [T]>
    for InputVectorBuffer<INPUTS, T, MatrixDataMut<'a, INPUTS, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(INPUTS, value.len());
        }
        Self::new(MatrixData::new_mut::<INPUTS, 1, T>(value))
    }
}

impl<const INPUTS: usize, T> From<[T; INPUTS]>
    for InputVectorBuffer<INPUTS, T, MatrixDataOwned<INPUTS, 1, INPUTS, T>>
{
    fn from(value: [T; INPUTS]) -> Self {
        Self::new(MatrixData::new_owned::<INPUTS, 1, INPUTS, T>(value))
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> IntoInnerData for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
