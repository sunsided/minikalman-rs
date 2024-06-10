use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use minikalman_traits::kalman::{
    MeasurementTransformationMatrix, MeasurementTransformationMatrixMut,
};

use minikalman_traits::matrix::{
    IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned, MatrixDataRef,
};
use minikalman_traits::matrix::{Matrix, MatrixMut};

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

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a [T]>
    for MeasurementTransformationMatrixBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataRef<'a, MEASUREMENTS, STATES, T>,
    >
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * STATES, value.len());
        }
        Self::new(MatrixData::new_ref::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a mut [T]>
    for MeasurementTransformationMatrixBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataRef<'a, MEASUREMENTS, STATES, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * STATES, value.len());
        }
        Self::new(MatrixData::new_ref::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a mut [T]>
    for MeasurementTransformationMatrixMutBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataMut<'a, MEASUREMENTS, STATES, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * STATES, value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MeasurementTransformationMatrixMutBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataOwned<MEASUREMENTS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * STATES, TOTAL);
        }
        Self::new(MatrixData::new_owned::<MEASUREMENTS, STATES, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IntoInnerData
    for MeasurementTransformationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IntoInnerData
    for MeasurementTransformationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
