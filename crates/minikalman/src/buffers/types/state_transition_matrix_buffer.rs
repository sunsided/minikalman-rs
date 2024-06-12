use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{StateTransitionMatrix, StateTransitionMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the system matrix (`num_states` × `num_states`), typically denoted "A" or "F".
///
/// Describes how the state evolves from one time step to the next in the absence of control inputs.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::StateTransitionMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = StateTransitionMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = StateTransitionMatrixBuffer::<2, f32, _>::from(data.as_ref());
/// ```
pub struct StateTransitionMatrixBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<STATES, STATES, T>;

/// Mutable buffer for the system matrix (`num_states` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::StateTransitionMatrixMutBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = StateTransitionMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = StateTransitionMatrixMutBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct StateTransitionMatrixMutBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, STATES, T>;

// -----------------------------------------------------------

impl<const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for StateTransitionMatrixBuffer<STATES, T, MatrixDataArray<STATES, STATES, TOTAL, T>>
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
    for StateTransitionMatrixBuffer<STATES, T, MatrixDataRef<'a, STATES, STATES, T>>
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
    for StateTransitionMatrixBuffer<STATES, T, MatrixDataRef<'a, STATES, STATES, T>>
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
    for StateTransitionMatrixMutBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>
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
    for StateTransitionMatrixMutBuffer<STATES, T, MatrixDataArray<STATES, STATES, TOTAL, T>>
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

impl<const STATES: usize, T, M> StateTransitionMatrixBuffer<STATES, T, M>
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

impl<const STATES: usize, T, M> AsRef<[T]> for StateTransitionMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T>
    for StateTransitionMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
}

impl<const STATES: usize, T, M> StateTransitionMatrix<STATES, T>
    for StateTransitionMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> StateTransitionMatrixMutBuffer<STATES, T, M>
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

impl<const STATES: usize, T, M> AsRef<[T]> for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> AsMut<[T]> for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, STATES, T>
    for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
}

impl<const STATES: usize, T, M> MatrixMut<STATES, STATES, T>
    for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
}

impl<const STATES: usize, T, M> StateTransitionMatrix<STATES, T>
    for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> StateTransitionMatrixMut<STATES, T>
    for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T, M> Index<usize> for StateTransitionMatrixBuffer<STATES, T, M>
where
    M: Matrix<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> Index<usize> for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for StateTransitionMatrixMutBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> IntoInnerData for StateTransitionMatrixBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const STATES: usize, T, M> IntoInnerData for StateTransitionMatrixMutBuffer<STATES, T, M>
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
        let value: StateTransitionMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: StateTransitionMatrixBuffer<5, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: StateTransitionMatrixBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: StateTransitionMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: StateTransitionMatrixMutBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: StateTransitionMatrixMutBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: StateTransitionMatrixMutBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn array_into_inner() {
        let value: StateTransitionMatrixBuffer<2, f32, _> = [0.0, 1.0, 3.0, 4.0].into();
        let data = value.into_inner();
        assert_eq!(data, [0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    #[rustfmt::skip]
    fn test_access_readonly() {
        let value: StateTransitionMatrixMutBuffer<5, f32, _> = [
            10.0, 11.0, 12.0, 13.0, 14.0,
            11.0, 11.0, 10.0, 10.0, 10.0,
            12.0, 10.0, 12.0, 10.0, 10.0,
            13.0, 10.0, 10.0, 13.0, 10.0,
            14.0, 10.0, 10.0, 10.0, 14.0,
        ].into();

        // Get values.
        {
            let matrix = value.as_matrix();
            for i in 0..matrix.rows() {
                assert_eq!(matrix.get(0, i), 10.0 + i as f32);
                assert_eq!(matrix.get(i, 0), 10.0 + i as f32);
            }
        }

        assert_eq!(value.into_inner(),
                   [
                       10.0, 11.0, 12.0, 13.0, 14.0,
                       11.0, 11.0, 10.0, 10.0, 10.0,
                       12.0, 10.0, 12.0, 10.0, 10.0,
                       13.0, 10.0, 10.0, 13.0, 10.0,
                       14.0, 10.0, 10.0, 10.0, 14.0,
                   ]);
    }

    #[test]
    #[rustfmt::skip]
    fn test_access() {
        let mut value: StateTransitionMatrixMutBuffer<5, f32, _> = [0.0; 25].into();

        // Set values.
        {
            let matrix = value.as_matrix_mut();
            for i in 0..matrix.cols() {
                matrix.set_symmetric(0, i, i as _);
                matrix.set(i, i, i as _);
            }
        }

        // Update values.
        for i in 0..value.len() {
            value[i] += 10.0;
        }

        // Get values.
        {
            let matrix = value.as_matrix();
            for i in 0..matrix.rows() {
                assert_eq!(matrix.get(0, i), 10.0 + i as f32);
                assert_eq!(matrix.get(i, 0), 10.0 + i as f32);
            }
        }

        assert_eq!(value.into_inner(),
                   [
                       10.0, 11.0, 12.0, 13.0, 14.0,
                       11.0, 11.0, 10.0, 10.0, 10.0,
                       12.0, 10.0, 12.0, 10.0, 10.0,
                       13.0, 10.0, 10.0, 13.0, 10.0,
                       14.0, 10.0, 10.0, 10.0, 14.0,
                   ]);
    }
}
