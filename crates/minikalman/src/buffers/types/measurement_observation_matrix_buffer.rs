use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{ObservationMatrix, ObservationMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the observation matrix (`num_controls` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ObservationMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ObservationMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = ObservationMatrixBuffer::<2, 2, f32, _>::from(data.as_ref());
/// ```
pub struct ObservationMatrixBuffer<const OBSERVATIONS: usize, const STATES: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: Matrix<OBSERVATIONS, STATES, T>;

/// Mmutable buffer for the observation matrix (`num_controls` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ObservationMatrixMutBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ObservationMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = ObservationMatrixMutBuffer::<2, 2, f32, _>::from(data.as_mut());
/// ```
pub struct ObservationMatrixMutBuffer<const OBSERVATIONS: usize, const STATES: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<OBSERVATIONS, STATES, T>;

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for ObservationMatrixBuffer<
        OBSERVATIONS,
        STATES,
        T,
        MatrixDataArray<OBSERVATIONS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<OBSERVATIONS, STATES, TOTAL, T>(
            value,
        ))
    }
}

impl<'a, const OBSERVATIONS: usize, const STATES: usize, T> From<&'a [T]>
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, MatrixDataRef<'a, OBSERVATIONS, STATES, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= value.len());
        }
        Self::new(MatrixData::new_ref::<OBSERVATIONS, STATES, T>(value))
    }
}

impl<'a, const OBSERVATIONS: usize, const STATES: usize, T> From<&'a mut [T]>
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, MatrixDataRef<'a, OBSERVATIONS, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= value.len());
        }
        Self::new(MatrixData::new_ref::<OBSERVATIONS, STATES, T>(value))
    }
}

impl<'a, const OBSERVATIONS: usize, const STATES: usize, T> From<&'a mut [T]>
    for ObservationMatrixMutBuffer<
        OBSERVATIONS,
        STATES,
        T,
        MatrixDataMut<'a, OBSERVATIONS, STATES, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= value.len());
        }
        Self::new(MatrixData::new_mut::<OBSERVATIONS, STATES, T>(value))
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for ObservationMatrixMutBuffer<
        OBSERVATIONS,
        STATES,
        T,
        MatrixDataArray<OBSERVATIONS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(OBSERVATIONS * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<OBSERVATIONS, STATES, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    ObservationMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: Matrix<OBSERVATIONS, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        OBSERVATIONS * STATES
    }

    pub const fn is_empty(&self) -> bool {
        OBSERVATIONS * STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> AsRef<[T]>
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: Matrix<OBSERVATIONS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> Matrix<OBSERVATIONS, STATES, T>
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: Matrix<OBSERVATIONS, STATES, T>,
{
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    ObservationMatrix<OBSERVATIONS, STATES, T>
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: Matrix<OBSERVATIONS, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        OBSERVATIONS * STATES
    }

    pub const fn is_empty(&self) -> bool {
        OBSERVATIONS * STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> AsRef<[T]>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> AsMut<[T]>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> Matrix<OBSERVATIONS, STATES, T>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> MatrixMut<OBSERVATIONS, STATES, T>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    ObservationMatrix<OBSERVATIONS, STATES, T>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M>
    ObservationMatrixMut<OBSERVATIONS, STATES, T>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> Index<usize>
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: Matrix<OBSERVATIONS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> Index<usize>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> IndexMut<usize>
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> IntoInnerData
    for ObservationMatrixBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T, M> IntoInnerData
    for ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, M>
where
    M: MatrixMut<OBSERVATIONS, STATES, T> + IntoInnerData,
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
        let value: ObservationMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: ObservationMatrixBuffer<5, 3, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ObservationMatrixBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: ObservationMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: ObservationMatrixMutBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ObservationMatrixMutBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: ObservationMatrixMutBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
