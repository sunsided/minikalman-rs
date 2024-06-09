use crate::filter_traits::MeasurementProcessNoiseCovarianceMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct MeasurementProcessNoiseCovarianceMatrixBuffer<const MEASUREMENT: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>;

// -----------------------------------------------------------

impl<const MEASUREMENT: usize, T, M>
    MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const MEASUREMENT: usize, T, M> AsRef<[T]>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENT: usize, T, M> AsMut<[T]>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENT: usize, T, M> Matrix<MEASUREMENT, MEASUREMENT, T>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
{
}

impl<const MEASUREMENT: usize, T, M> MatrixMut<MEASUREMENT, MEASUREMENT, T>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
{
}

impl<const MEASUREMENT: usize, T, M> MeasurementProcessNoiseCovarianceMatrix<MEASUREMENT, T>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
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

impl<const MEASUREMENTS: usize, T, M> Index<usize>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
