use crate::buffer_types::InputMatrixMutBuffer;
use crate::filter_traits::{MeasurementTransformationMatrix, MeasurementTransformationMatrixMut};
use crate::matrix_traits::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct MeasurementTransformationMatrixBuffer<
    const MEASUREMENTS: usize,
    const STATES: usize,
    T,
    M,
>(M, PhantomData<T>)
where
    M: Matrix<MEASUREMENTS, STATES, T>;

pub struct MeasurementTransformationMatrixMutBuffer<
    const MEASUREMENTS: usize,
    const STATES: usize,
    T,
    M,
>(M, PhantomData<T>)
where
    M: MatrixMut<MEASUREMENTS, STATES, T>;

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsRef<[T]>
    for MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Matrix<MEASUREMENTS, STATES, T>
    for MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>
    for MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsRef<[T]>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsMut<[T]>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Matrix<MEASUREMENTS, STATES, T>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> MatrixMut<MEASUREMENTS, STATES, T>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementTransformationMatrixMut<MEASUREMENTS, STATES, T>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Index<usize>
    for MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Index<usize>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IndexMut<usize>
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}
