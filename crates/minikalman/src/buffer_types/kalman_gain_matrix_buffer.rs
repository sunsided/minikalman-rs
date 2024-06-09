use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::filter_traits::KalmanGainMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};

pub struct KalmanGainMatrixBuffer<const STATES: usize, const MEASUREMENTS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, MEASUREMENTS, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const MEASUREMENTS: usize, T, M>
    KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> AsRef<[T]>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> AsMut<[T]>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> Matrix<STATES, MEASUREMENTS, T>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> MatrixMut<STATES, MEASUREMENTS, T>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> KalmanGainMatrix<STATES, MEASUREMENTS, T>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
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

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> Index<usize>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
