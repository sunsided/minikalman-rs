use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use minikalman_traits::kalman::MeasurementProcessNoiseCovarianceMatrix;
use minikalman_traits::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use minikalman_traits::matrix::{Matrix, MatrixMut};

/// Mutable buffer for the observation covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = MeasurementProcessNoiseCovarianceMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = MeasurementProcessNoiseCovarianceMatrixBuffer::<2, f32, _>::from(data.as_mut());
/// ```
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
            debug_assert!(MEASUREMENTS * MEASUREMENTS <= value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, MEASUREMENTS, T>(value))
    }
}

impl<const MEASUREMENTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MeasurementProcessNoiseCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataArray<MEASUREMENTS, MEASUREMENTS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS * MEASUREMENTS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<MEASUREMENTS, MEASUREMENTS, TOTAL, T>(value))
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

    pub const fn len(&self) -> usize {
        MEASUREMENT * MEASUREMENT
    }

    pub const fn is_empty(&self) -> bool {
        MEASUREMENT * MEASUREMENT == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_array() {
        let value: MeasurementProcessNoiseCovarianceMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: MeasurementProcessNoiseCovarianceMatrixBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: MeasurementProcessNoiseCovarianceMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
