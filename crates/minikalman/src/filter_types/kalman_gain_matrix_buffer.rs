use crate::filter_traits::{KalmanGainMatrix, ResidualCovarianceMatrix};
use crate::filter_types::InputMatrixMutBuffer;
use crate::matrix_traits::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

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

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> Matrix<STATES, MEASUREMENTS, T>
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
