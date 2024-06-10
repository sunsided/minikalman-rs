use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::type_traits::{SystemMatrix, SystemMatrixMut};
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned, MatrixDataRef};
use minikalman_traits::{Matrix, MatrixMut};

pub struct SystemMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, STATES, T>;

pub struct SystemMatrixMutBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, T> From<&'a [T]>
    for SystemMatrixBuffer<STATES, T, MatrixDataRef<'a, STATES, STATES, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * STATES, value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, STATES, T>(value))
    }
}

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for SystemMatrixBuffer<STATES, T, MatrixDataRef<'a, STATES, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * STATES, value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, STATES, T>(value))
    }
}

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for SystemMatrixMutBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * STATES, value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, STATES, T>(value))
    }
}

impl<const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for SystemMatrixMutBuffer<STATES, T, MatrixDataOwned<STATES, STATES, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(STATES * STATES, TOTAL);
        }
        Self::new(MatrixData::new_owned::<STATES, STATES, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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

// -----------------------------------------------------------

impl<const STATES: usize, T, M> IntoInnerData for SystemMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const STATES: usize, T, M> IntoInnerData for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
