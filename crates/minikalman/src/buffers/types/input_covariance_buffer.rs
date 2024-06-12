use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{InputCovarianceMatrix, InputCovarianceMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::InputCovarianceMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = InputCovarianceMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = InputCovarianceMatrixBuffer::<2, f32, _>::from(data.as_ref());
/// ```
pub struct InputCovarianceMatrixBuffer<const CONTROLS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<CONTROLS, CONTROLS, T>;

/// Mutable buffer for the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::InputCovarianceMatrixMutBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = InputCovarianceMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InputCovarianceMatrixMutBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct InputCovarianceMatrixMutBuffer<const CONTROLS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<CONTROLS, CONTROLS, T>;

// -----------------------------------------------------------

impl<const CONTROLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputCovarianceMatrixBuffer<CONTROLS, T, MatrixDataArray<CONTROLS, CONTROLS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(CONTROLS * CONTROLS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<CONTROLS, CONTROLS, TOTAL, T>(value))
    }
}

impl<'a, const CONTROLS: usize, T> From<&'a [T]>
    for InputCovarianceMatrixBuffer<CONTROLS, T, MatrixDataRef<'a, CONTROLS, CONTROLS, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(CONTROLS * CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_ref::<CONTROLS, CONTROLS, T>(value))
    }
}

impl<'a, const CONTROLS: usize, T> From<&'a mut [T]>
    for InputCovarianceMatrixBuffer<CONTROLS, T, MatrixDataRef<'a, CONTROLS, CONTROLS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(CONTROLS * CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_ref::<CONTROLS, CONTROLS, T>(value))
    }
}

impl<'a, const CONTROLS: usize, T> From<&'a mut [T]>
    for InputCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataMut<'a, CONTROLS, CONTROLS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(CONTROLS * CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_mut::<CONTROLS, CONTROLS, T>(value))
    }
}

impl<const CONTROLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataArray<CONTROLS, CONTROLS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(CONTROLS * CONTROLS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<CONTROLS, CONTROLS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const CONTROLS: usize, T, M> InputCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        CONTROLS * CONTROLS
    }

    pub const fn is_empty(&self) -> bool {
        CONTROLS == 0
    }
}

impl<const CONTROLS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const CONTROLS: usize, T, M> Matrix<CONTROLS, CONTROLS, T>
    for InputCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
}

impl<const CONTROLS: usize, T, M> InputCovarianceMatrix<CONTROLS, T>
    for InputCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T, M> InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        CONTROLS * CONTROLS
    }

    pub const fn is_empty(&self) -> bool {
        CONTROLS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const CONTROLS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const CONTROLS: usize, T, M> AsMut<[T]> for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const CONTROLS: usize, T, M> Matrix<CONTROLS, CONTROLS, T>
    for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
}

impl<const CONTROLS: usize, T, M> MatrixMut<CONTROLS, CONTROLS, T>
    for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
}

impl<const CONTROLS: usize, T, M> InputCovarianceMatrix<CONTROLS, T>
    for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T, M> InputCovarianceMatrixMut<CONTROLS, T>
    for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T, M> Index<usize> for InputCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const CONTROLS: usize, T, M> Index<usize> for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const CONTROLS: usize, T, M> IndexMut<usize> for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const CONTROLS: usize, T, M> IntoInnerData for InputCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const CONTROLS: usize, T, M> IntoInnerData for InputCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T> + IntoInnerData,
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
        let value: InputCovarianceMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: InputCovarianceMatrixBuffer<5, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
        assert!(!value.is_empty());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: InputCovarianceMatrixBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
        assert!(!value.is_empty());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: InputCovarianceMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: InputCovarianceMatrixMutBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: InputCovarianceMatrixMutBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: InputCovarianceMatrixMutBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
