use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::TemporaryPHTMatrix;
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};

/// Mutable buffer for the temporary P×Hᵀ matrix (`num_states` × `num_measurements`).
///
/// # See also
/// * [`TemporaryHPMatrixBuffer`](crate::buffer_types::TemporaryHPMatrixBuffer).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::TemporaryHPMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = TemporaryHPMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = TemporaryHPMatrixBuffer::<2, 2, f32, _>::from(data.as_mut());
/// ```
pub struct TemporaryPHTMatrixBuffer<const STATES: usize, const OBSERVATIONS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, OBSERVATIONS, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, const OBSERVATIONS: usize, T> From<&'a mut [T]>
    for TemporaryPHTMatrixBuffer<
        STATES,
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, STATES, OBSERVATIONS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * OBSERVATIONS <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, OBSERVATIONS, T>(value))
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TemporaryPHTMatrixBuffer<
        STATES,
        OBSERVATIONS,
        T,
        MatrixDataArray<STATES, OBSERVATIONS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * OBSERVATIONS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, OBSERVATIONS, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * OBSERVATIONS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * OBSERVATIONS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> AsRef<[T]>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> AsMut<[T]>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> Matrix<STATES, OBSERVATIONS, T>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> MatrixMut<STATES, OBSERVATIONS, T>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M>
    TemporaryPHTMatrix<STATES, OBSERVATIONS, T>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
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

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> Index<usize>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> IndexMut<usize>
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const OBSERVATIONS: usize, T, M> IntoInnerData
    for TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, M>
where
    M: MatrixMut<STATES, OBSERVATIONS, T> + IntoInnerData,
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
        let value: TemporaryPHTMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: TemporaryPHTMatrixBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: TemporaryPHTMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
