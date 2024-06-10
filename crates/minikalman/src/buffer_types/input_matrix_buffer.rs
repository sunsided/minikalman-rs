use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::type_traits::{InputMatrix, InputMatrixMut};
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned, MatrixDataRef};
use minikalman_traits::{Matrix, MatrixMut};

pub struct InputMatrixBuffer<const STATES: usize, const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, INPUTS, T>;

pub struct InputMatrixMutBuffer<const STATES: usize, const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, INPUTS, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, const INPUTS: usize, T> From<&'a [T]>
    for InputMatrixBuffer<STATES, INPUTS, T, MatrixDataRef<'a, STATES, INPUTS, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * INPUTS, value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, INPUTS, T>(value))
    }
}

impl<'a, const STATES: usize, const INPUTS: usize, T> From<&'a mut [T]>
    for InputMatrixBuffer<STATES, INPUTS, T, MatrixDataRef<'a, STATES, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * INPUTS, value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, INPUTS, T>(value))
    }
}

impl<'a, const STATES: usize, const INPUTS: usize, T> From<&'a mut [T]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, MatrixDataMut<'a, STATES, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * INPUTS, value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, INPUTS, T>(value))
    }
}

impl<const STATES: usize, const INPUTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, MatrixDataOwned<STATES, INPUTS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * INPUTS, TOTAL);
        }
        Self::new(MatrixData::new_owned::<STATES, INPUTS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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
