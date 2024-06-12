use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use minikalman_traits::kalman::{InputMatrix, InputMatrixMut};
use minikalman_traits::matrix::{
    IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef,
};
use minikalman_traits::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the control matrix (`num_states` × `num_inputs`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = InputMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = InputMatrixBuffer::<2, 2, f32, _>::from(data.as_ref());
/// ```
pub struct InputMatrixBuffer<const STATES: usize, const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, INPUTS, T>;

/// Mutable buffer for the control matrix (`num_states` × `num_inputs`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = InputMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InputMatrixMutBuffer::<2, 2, f32, _>::from(data.as_mut());
/// ```
pub struct InputMatrixMutBuffer<const STATES: usize, const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, INPUTS, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputMatrixBuffer<STATES, INPUTS, T, MatrixDataArray<STATES, INPUTS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * INPUTS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, INPUTS, TOTAL, T>(value))
    }
}

impl<'a, const STATES: usize, const INPUTS: usize, T> From<&'a [T]>
    for InputMatrixBuffer<STATES, INPUTS, T, MatrixDataRef<'a, STATES, INPUTS, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * INPUTS <= value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, INPUTS, T>(value))
    }
}

impl<'a, const STATES: usize, const INPUTS: usize, T> From<&'a mut [T]>
    for InputMatrixBuffer<STATES, INPUTS, T, MatrixDataRef<'a, STATES, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * INPUTS <= value.len());
        }
        Self::new(MatrixData::new_ref::<STATES, INPUTS, T>(value))
    }
}

impl<'a, const STATES: usize, const INPUTS: usize, T> From<&'a mut [T]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, MatrixDataMut<'a, STATES, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * INPUTS <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, INPUTS, T>(value))
    }
}

impl<const STATES: usize, const INPUTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, MatrixDataArray<STATES, INPUTS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES * INPUTS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<STATES, INPUTS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * INPUTS == 0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsRef<[T]>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Matrix<STATES, INPUTS, T>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrix<STATES, INPUTS, T>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES * INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        STATES * INPUTS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsRef<[T]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> AsMut<[T]>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Matrix<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> MatrixMut<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrix<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> InputMatrixMut<STATES, INPUTS, T>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Index<usize>
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: Matrix<STATES, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> Index<usize>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> IndexMut<usize>
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, const INPUTS: usize, T, M> IntoInnerData
    for InputMatrixBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, M> IntoInnerData
    for InputMatrixMutBuffer<STATES, INPUTS, T, M>
where
    M: MatrixMut<STATES, INPUTS, T> + IntoInnerData,
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
        let value: InputMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: InputMatrixBuffer<5, 3, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 15);
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: InputMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: InputMatrixMutBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: InputMatrixMutBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: InputMatrixMutBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
