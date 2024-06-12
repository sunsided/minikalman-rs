use crate::kalman::ResidualCovarianceMatrix;
use crate::matrix::{IntoInnerData, Matrix, MatrixData, MatrixDataArray, MatrixDataMut, MatrixMut};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

/// Buffer for the square innovation (residual) covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::InnovationResidualCovarianceMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = InnovationResidualCovarianceMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InnovationResidualCovarianceMatrixBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct InnovationResidualCovarianceMatrixBuffer<const OBSERVATIONS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

// -----------------------------------------------------------

impl<'a, const OBSERVATIONS: usize, T> From<&'a mut [T]>
    for InnovationResidualCovarianceMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, OBSERVATIONS, OBSERVATIONS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * OBSERVATIONS <= value.len());
        }
        Self::new(MatrixData::new_mut::<OBSERVATIONS, OBSERVATIONS, T>(value))
    }
}

impl<const OBSERVATIONS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InnovationResidualCovarianceMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataArray<OBSERVATIONS, OBSERVATIONS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * OBSERVATIONS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<OBSERVATIONS, OBSERVATIONS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, T, M> InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        OBSERVATIONS * OBSERVATIONS
    }

    pub const fn is_empty(&self) -> bool {
        OBSERVATIONS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const OBSERVATIONS: usize, T, M> AsRef<[T]>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const OBSERVATIONS: usize, T, M> AsMut<[T]>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const OBSERVATIONS: usize, T, M> Matrix<OBSERVATIONS, OBSERVATIONS, T>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
}

impl<const OBSERVATIONS: usize, T, M> MatrixMut<OBSERVATIONS, OBSERVATIONS, T>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
}

impl<const OBSERVATIONS: usize, T, M> ResidualCovarianceMatrix<OBSERVATIONS, T>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
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

impl<const OBSERVATIONS: usize, T, M> Index<usize>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const OBSERVATIONS: usize, T, M> IndexMut<usize>
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, T, M> IntoInnerData
    for InnovationResidualCovarianceMatrixBuffer<OBSERVATIONS, T, M>
where
    M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T> + IntoInnerData,
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
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: InnovationResidualCovarianceMatrixBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
        assert!(!value.is_empty());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: InnovationResidualCovarianceMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
