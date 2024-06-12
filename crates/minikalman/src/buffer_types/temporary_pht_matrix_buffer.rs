use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use minikalman_traits::kalman::TemporaryPHTMatrix;
use minikalman_traits::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use minikalman_traits::matrix::{Matrix, MatrixMut};

/// Mutable buffer for the temporary P×Hᵀ matrix (`num_states` × `num_measurements`).
///
/// # See also
/// * [`TemporaryHPMatrixBuffer`](crate::buffer_types::TemporaryHPMatrixBuffer).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = TemporaryHPMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = TemporaryHPMatrixBuffer::<2, 2, f32, _>::from(data.as_mut());
/// ```
pub struct TemporaryPHTMatrixBuffer<const STATES: usize, const MEASUREMENTS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, MEASUREMENTS, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for TemporaryPHTMatrixBuffer<
        STATES,
        MEASUREMENTS,
        T,
        MatrixDataMut<'a, STATES, MEASUREMENTS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * MEASUREMENTS <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, MEASUREMENTS, T>(value))
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TemporaryPHTMatrixBuffer<
        STATES,
        MEASUREMENTS,
        T,
        MatrixDataArray<STATES, MEASUREMENTS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * MEASUREMENTS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, MEASUREMENTS, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const MEASUREMENTS: usize, T, M>
    TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * MEASUREMENTS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * MEASUREMENTS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> AsRef<[T]>
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> AsMut<[T]>
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> Matrix<STATES, MEASUREMENTS, T>
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> MatrixMut<STATES, MEASUREMENTS, T>
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M>
    TemporaryPHTMatrix<STATES, MEASUREMENTS, T>
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
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
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const MEASUREMENTS: usize, T, M> IntoInnerData
    for TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, M>
where
    M: MatrixMut<STATES, MEASUREMENTS, T> + IntoInnerData,
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
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: TemporaryPHTMatrixBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
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
