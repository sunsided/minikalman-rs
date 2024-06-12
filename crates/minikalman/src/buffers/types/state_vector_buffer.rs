use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{StateVector, StateVectorMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};

/// Mutable buffer for the state vector (`num_states` × `1`), typically denoted "x".
///
/// Represents the internal state of the system.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::StateVectorBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = StateVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = StateVectorBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct StateVectorBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, 1, T>;

// -----------------------------------------------------------

impl<'a, const STATES: usize, T> From<&'a mut [T]>
    for StateVectorBuffer<STATES, T, MatrixDataMut<'a, STATES, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(STATES <= value.len());
        }
        Self::new(MatrixData::new_mut::<STATES, 1, T>(value))
    }
}

/// # Example
/// Buffers can be trivially constructed from correctly-sized arrays:
///
/// ```
/// # use minikalman::buffers::types::StateVectorBuffer;
/// let _value: StateVectorBuffer<5, f32, _> = [0.0; 5].into();
/// ```
///
/// Invalid buffer sizes fail to compile:
///
/// ```fail_compile
/// # use minikalman::prelude::StateVectorBuffer;
/// let _value: StateVectorBuffer<5, f32, _> = [0.0; 1].into();
/// ```
impl<const STATES: usize, T> From<[T; STATES]>
    for StateVectorBuffer<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>
{
    fn from(value: [T; STATES]) -> Self {
        Self::new(MatrixData::new_array::<STATES, 1, STATES, T>(value))
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        STATES
    }

    pub const fn is_empty(&self) -> bool {
        STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const STATES: usize, T, M> AsRef<[T]> for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const STATES: usize, T, M> AsMut<[T]> for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const STATES: usize, T, M> Matrix<STATES, 1, T> for StateVectorBuffer<STATES, T, M> where
    M: MatrixMut<STATES, 1, T>
{
}

impl<const STATES: usize, T, M> MatrixMut<STATES, 1, T> for StateVectorBuffer<STATES, T, M> where
    M: MatrixMut<STATES, 1, T>
{
}

impl<const STATES: usize, T, M> StateVector<STATES, T> for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T, M> StateVectorMut<STATES, T> for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T, M> Index<usize> for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const STATES: usize, T, M> IndexMut<usize> for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const STATES: usize, T, M> IntoInnerData for StateVectorBuffer<STATES, T, M>
where
    M: MatrixMut<STATES, 1, T> + IntoInnerData,
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
        let value: StateVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: StateVectorBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_access() {
        let mut value: StateVectorBuffer<5, f32, _> = [0.0; 5].into();

        // Set values.
        {
            let matrix = value.as_matrix_mut();
            for i in 0..matrix.rows() {
                matrix.set(i, 0, i as _);
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
                assert_eq!(matrix.get(i, 0), 10.0 + i as f32);
            }
        }

        assert_eq!(value.into_inner(), [10.0, 11.0, 12.0, 13.0, 14.0]);
    }
}
