use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use minikalman_traits::kalman::{InputCovarianceMatrix, InputCovarianceMatrixMut};
use minikalman_traits::matrix::{
    IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef,
};
use minikalman_traits::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = InputCovarianceMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = InputCovarianceMatrixBuffer::<2, f32, _>::from(data.as_ref());
/// ```
pub struct InputCovarianceMatrixBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<INPUTS, INPUTS, T>;

/// Mutable buffer for the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = InputCovarianceMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InputCovarianceMatrixMutBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct InputCovarianceMatrixMutBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<INPUTS, INPUTS, T>;

// -----------------------------------------------------------

impl<const INPUTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputCovarianceMatrixBuffer<INPUTS, T, MatrixDataArray<INPUTS, INPUTS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(INPUTS * INPUTS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<INPUTS, INPUTS, TOTAL, T>(value))
    }
}

impl<'a, const INPUTS: usize, T> From<&'a [T]>
    for InputCovarianceMatrixBuffer<INPUTS, T, MatrixDataRef<'a, INPUTS, INPUTS, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(INPUTS * INPUTS <= value.len());
        }
        Self::new(MatrixData::new_ref::<INPUTS, INPUTS, T>(value))
    }
}

impl<'a, const INPUTS: usize, T> From<&'a mut [T]>
    for InputCovarianceMatrixBuffer<INPUTS, T, MatrixDataRef<'a, INPUTS, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(INPUTS * INPUTS <= value.len());
        }
        Self::new(MatrixData::new_ref::<INPUTS, INPUTS, T>(value))
    }
}

impl<'a, const INPUTS: usize, T> From<&'a mut [T]>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, MatrixDataMut<'a, INPUTS, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(INPUTS * INPUTS <= value.len());
        }
        Self::new(MatrixData::new_mut::<INPUTS, INPUTS, T>(value))
    }
}

impl<const INPUTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, MatrixDataArray<INPUTS, INPUTS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(INPUTS * INPUTS <= TOTAL);
        }
        Self::new(MatrixData::new_array::<INPUTS, INPUTS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        INPUTS * INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        INPUTS == 0
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, INPUTS, T>
    for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrix<INPUTS, T>
    for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        INPUTS * INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        INPUTS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> AsMut<[T]> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> MatrixMut<INPUTS, INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrix<INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrixMut<INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> IndexMut<usize> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> IntoInnerData for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const INPUTS: usize, T, M> IntoInnerData for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T> + IntoInnerData,
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
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: InputCovarianceMatrixBuffer<5, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
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
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: InputCovarianceMatrixMutBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: InputCovarianceMatrixMutBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
