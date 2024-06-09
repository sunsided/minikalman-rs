use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::filter_traits::TemporaryHPMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::IntoInnerData;

pub struct TemporaryHPMatrixBuffer<const MEASUREMENTS: usize, const STATES: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: Matrix<MEASUREMENTS, STATES, T>;

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsRef<[T]>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsMut<[T]>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Matrix<MEASUREMENTS, STATES, T>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> MatrixMut<MEASUREMENTS, STATES, T>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    TemporaryHPMatrix<MEASUREMENTS, STATES, T>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
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

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Index<usize>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IndexMut<usize>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IntoInnerData
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
