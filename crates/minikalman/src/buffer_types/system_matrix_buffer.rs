use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use minikalman_traits::kalman::{SystemMatrix, SystemMatrixMut};
use minikalman_traits::matrix::{
    IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef,
};
use minikalman_traits::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the system matrix (`num_states` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = SystemMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = SystemMatrixBuffer::<2, f32, _>::from(data.as_ref());
/// ```
pub struct SystemMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, STATES, T>;

/// Mutable buffer for the system matrix (`num_states` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = SystemMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = SystemMatrixMutBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct SystemMatrixMutBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for SystemMatrixBuffer<STATES, T, MatrixDataArray<STATES, STATES, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, STATES, TOTAL, T>(value))
    }
}

impl<'a, const STATES: usize, T> From<&'a [T]>
    for SystemMatrixBuffer<STATES, T, MatrixDataRef<'a, STATES, STATES, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, STATES, T>(value))
    }
}

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for SystemMatrixBuffer<STATES, T, MatrixDataRef<'a, STATES, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, STATES, T>(value))
    }
}

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for SystemMatrixMutBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, STATES, T>(value))
    }
}

impl<const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for SystemMatrixMutBuffer<STATES, T, MatrixDataArray<STATES, STATES, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, STATES, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * STATES
    }

    pub const fn is_empty(&self) -> bool {
        STATES * STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T> for SystemMatrixBuffer<STATES, T, M> where
    M: Matrix<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> SystemMatrix<STATES, T> for SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * STATES
    }

    pub const fn is_empty(&self) -> bool {
        STATES * STATES == 0
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> AsMut<[T]> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T> for SystemMatrixMutBuffer<STATES, T, M> where
    M: MatrixMut<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> MatrixMut<STATES, STATES, T> for SystemMatrixMutBuffer<STATES, T, M> where
    M: MatrixMut<STATES, STATES, T>
{
}

impl<const STATES: usize, T, M> SystemMatrix<STATES, T> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> SystemMatrixMut<STATES, T> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T, M> Index<usize> for SystemMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> Index<usize> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> IntoInnerData for SystemMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const STATES: usize, T, M> IntoInnerData for SystemMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
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
        let value: SystemMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: SystemMatrixBuffer<5, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: SystemMatrixBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: SystemMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: SystemMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: SystemMatrixMutBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: SystemMatrixMutBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
