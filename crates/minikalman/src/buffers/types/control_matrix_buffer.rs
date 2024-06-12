use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{ControlMatrix, ControlMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the control matrix (`num_states` × `num_controls`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ControlMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ControlMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = ControlMatrixBuffer::<2, 2, f32, _>::from(data.as_ref());
/// ```
pub struct ControlMatrixBuffer<const STATES: usize, const CONTROLS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, CONTROLS, T>;

/// Mutable buffer for the control matrix (`num_states` × `num_controls`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ControlMatrixMutBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ControlMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = ControlMatrixMutBuffer::<2, 2, f32, _>::from(data.as_mut());
/// ```
pub struct ControlMatrixMutBuffer<const STATES: usize, const CONTROLS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<STATES, CONTROLS, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const CONTROLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for ControlMatrixBuffer<STATES, CONTROLS, T, MatrixDataArray<STATES, CONTROLS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * CONTROLS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, CONTROLS, TOTAL, T>(value))
    }
}

impl<'a, const STATES: usize, const CONTROLS: usize, T> From<&'a [T]>
    for ControlMatrixBuffer<STATES, CONTROLS, T, MatrixDataRef<'a, STATES, CONTROLS, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, CONTROLS, T>(value))
    }
}

impl<'a, const STATES: usize, const CONTROLS: usize, T> From<&'a mut [T]>
    for ControlMatrixBuffer<STATES, CONTROLS, T, MatrixDataRef<'a, STATES, CONTROLS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, CONTROLS, T>(value))
    }
}

impl<'a, const STATES: usize, const CONTROLS: usize, T> From<&'a mut [T]>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataMut<'a, STATES, CONTROLS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, CONTROLS, T>(value))
    }
}

impl<const STATES: usize, const CONTROLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataArray<STATES, CONTROLS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * CONTROLS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, CONTROLS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const CONTROLS: usize, T, M> ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: Matrix<STATES, CONTROLS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * CONTROLS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * CONTROLS == 0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> AsRef<[T]>
    for ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: Matrix<STATES, CONTROLS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> Matrix<STATES, CONTROLS, T>
    for ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: Matrix<STATES, CONTROLS, T>,
{
}

impl<const STATES: usize, const CONTROLS: usize, T, M> ControlMatrix<STATES, CONTROLS, T>
    for ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: Matrix<STATES, CONTROLS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M>
    ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * CONTROLS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * CONTROLS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> AsRef<[T]>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> AsMut<[T]>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> Matrix<STATES, CONTROLS, T>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
}

impl<const STATES: usize, const CONTROLS: usize, T, M> MatrixMut<STATES, CONTROLS, T>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
}

impl<const STATES: usize, const CONTROLS: usize, T, M> ControlMatrix<STATES, CONTROLS, T>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> ControlMatrixMut<STATES, CONTROLS, T>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> Index<usize>
    for ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: Matrix<STATES, CONTROLS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> Index<usize>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> IndexMut<usize>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const CONTROLS: usize, T, M> IntoInnerData
    for ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M> IntoInnerData
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T> + IntoInnerData,
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
        let value: ControlMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: ControlMatrixBuffer<5, 3, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ControlMatrixBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: ControlMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: ControlMatrixMutBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ControlMatrixMutBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: ControlMatrixMutBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
