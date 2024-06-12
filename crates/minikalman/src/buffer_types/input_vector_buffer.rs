use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use minikalman_traits::kalman::{InputVector, InputVectorMut};
use minikalman_traits::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use minikalman_traits::matrix::{Matrix, MatrixMut};

// TODO: Add InputVectorMutBuffer

/// Mutable buffer for the control (input) vector (`num_inputs` × `1`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
/// use minikalman_traits::matrix::MatrixData;
///
/// // From owned data
/// let buffer = InputVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InputVectorBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct InputVectorBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<INPUTS, 1, T>;

// -----------------------------------------------------------

impl<'a, const INPUTS: usize, T> From<&'a mut [T]>
    for InputVectorBuffer<INPUTS, T, MatrixDataMut<'a, INPUTS, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(INPUTS <= value.len());
        }
        Self::new(MatrixData::new_mut::<INPUTS, 1, T>(value))
    }
}

/// # Example
/// Buffers can be trivially constructed from correctly-sized arrays:
///
/// ```
/// # use minikalman::prelude::InputVectorBuffer;
/// let _value: InputVectorBuffer<5, f32, _> = [0.0; 5].into();
/// ```
///
/// Invalid buffer sizes fail to compile:
///
/// ```fail_compile
/// # use minikalman::prelude::InputVectorBuffer;
/// let _value: InputVectorBuffer<5, f32, _> = [0.0; 1].into();
/// ```
impl<const INPUTS: usize, T> From<[T; INPUTS]>
    for InputVectorBuffer<INPUTS, T, MatrixDataArray<INPUTS, 1, INPUTS, T>>
{
    fn from(value: [T; INPUTS]) -> Self {
        Self::new(MatrixData::new_array::<INPUTS, 1, INPUTS, T>(value))
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        INPUTS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> AsMut<[T]> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, 1, T> for InputVectorBuffer<INPUTS, T, M> where
    M: MatrixMut<INPUTS, 1, T>
{
}

impl<const INPUTS: usize, T, M> MatrixMut<INPUTS, 1, T> for InputVectorBuffer<INPUTS, T, M> where
    M: MatrixMut<INPUTS, 1, T>
{
}

impl<const INPUTS: usize, T, M> InputVector<INPUTS, T> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputVectorMut<INPUTS, T> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> IndexMut<usize> for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> IntoInnerData for InputVectorBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, 1, T> + IntoInnerData,
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
        let value: InputVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: InputVectorBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 5);
        assert!(value.is_valid());
    }
}
