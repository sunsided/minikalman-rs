use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::filter_traits::TemporaryStateMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned};

pub struct TemporaryStateMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for TemporaryStateMatrixBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>
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
    for TemporaryStateMatrixBuffer<STATES, T, MatrixDataOwned<STATES, STATES, TOTAL, T>>
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

impl<const STATES: usize, T, M> TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> AsMut<[T]> for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T>
    for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
}

impl<const STATES: usize, T, M> MatrixMut<STATES, STATES, T>
    for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
}

impl<const STATES: usize, T, M> TemporaryStateMatrix<STATES, T>
    for TemporaryStateMatrixBuffer<STATES, T, M>
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

impl<const STATES: usize, T, M> Index<usize> for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> IntoInnerData for TemporaryStateMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
