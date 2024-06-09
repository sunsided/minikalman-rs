use crate::filter_traits::MeasurementProcessNoiseCovarianceMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

pub struct MeasurementProcessNoiseCovarianceMatrixBuffer<const MEASUREMENT: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataMut<'a, MEASUREMENTS, MEASUREMENTS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * MEASUREMENTS, value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, MEASUREMENTS, T>(value))
    }
}

impl<const MEASUREMENTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataOwned<MEASUREMENTS, MEASUREMENTS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * MEASUREMENTS, TOTAL);
        }
        Self::new(MatrixData::new_owned::<MEASUREMENTS, MEASUREMENTS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENT: usize, T, M>
    MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENT, T, M>
where
    M: MatrixMut<MEASUREMENT, MEASUREMENT, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
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

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> IntoInnerData
    for MeasurementProcessNoiseCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
