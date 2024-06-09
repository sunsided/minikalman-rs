use crate::filter_traits::MeasurementVector;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::prelude::MeasurementVectorMut;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct MeasurementVectorBuffer<const MEASUREMENTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<MEASUREMENTS, 1, T>;

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const MEASUREMENTS: usize, T, M> AsRef<[T]> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, T, M> Index<usize> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> AsMut<[T]> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, T, M> Matrix<MEASUREMENTS, 1, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MatrixMut<MEASUREMENTS, 1, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MeasurementVector<MEASUREMENTS, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, T, M> MeasurementVectorMut<MEASUREMENTS, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}
