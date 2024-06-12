use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use minikalman_traits::kalman::ResidualCovarianceMatrix;
use minikalman_traits::matrix::{
    IntoInnerData, Matrix, MatrixData, MatrixDataArray, MatrixDataMut, MatrixMut,
};

/// Buffer for the square innovation (residual) covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = InnovationResidualCovarianceMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InnovationResidualCovarianceMatrixBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct InnovationResidualCovarianceMatrixBuffer<const MEASUREMENTS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for InnovationResidualCovarianceMatrixBuffer<
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
    for InnovationResidualCovarianceMatrixBuffer<
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

impl<const MEASUREMENTS: usize, T, M> InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        MEASUREMENTS * MEASUREMENTS
    }

    pub const fn is_empty(&self) -> bool {
        MEASUREMENTS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const MEASUREMENTS: usize, T, M> AsRef<[T]>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, T, M> AsMut<[T]>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, T, M> Matrix<MEASUREMENTS, MEASUREMENTS, T>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MatrixMut<MEASUREMENTS, MEASUREMENTS, T>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> ResidualCovarianceMatrix<MEASUREMENTS, T>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
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
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> IntoInnerData
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
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
        let value: InnovationResidualCovarianceMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let mut data = [0.0_f32; 100];
        let value: InnovationResidualCovarianceMatrixBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: InnovationResidualCovarianceMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
