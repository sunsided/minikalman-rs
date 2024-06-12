use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{ControlVector, ControlVectorMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};

// TODO: Add ControlVectorMutBuffer

/// Mutable buffer for the control (input) vector (`num_inputs` × `1`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ControlVectorBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ControlVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = ControlVectorBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct ControlVectorBuffer<const CONTROLS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<CONTROLS, 1, T>;

// -----------------------------------------------------------

impl<'a, const CONTROLS: usize, T> From<&'a mut [T]>
    for ControlVectorBuffer<CONTROLS, T, MatrixDataMut<'a, CONTROLS, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(CONTROLS <= value.len());
        }
        Self::new(MatrixData::new_mut::<CONTROLS, 1, T>(value))
    }
}

/// # Example
/// Buffers can be trivially constructed from correctly-sized arrays:
///
/// ```
/// # use minikalman::buffers::types::ControlVectorBuffer;
/// let _value: ControlVectorBuffer<5, f32, _> = [0.0; 5].into();
/// ```
///
/// Invalid buffer sizes fail to compile:
///
/// ```fail_compile
/// # use minikalman::prelude::ControlVectorBuffer;
/// let _value: ControlVectorBuffer<5, f32, _> = [0.0; 1].into();
/// ```
impl<const CONTROLS: usize, T> From<[T; CONTROLS]>
    for ControlVectorBuffer<CONTROLS, T, MatrixDataArray<CONTROLS, 1, CONTROLS, T>>
{
    fn from(value: [T; CONTROLS]) -> Self {
        Self::new(MatrixData::new_array::<CONTROLS, 1, CONTROLS, T>(value))
    }
}

// -----------------------------------------------------------

impl<const CONTROLS: usize, T, M> ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        CONTROLS
    }

    pub const fn is_empty(&self) -> bool {
        CONTROLS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const CONTROLS: usize, T, M> AsRef<[T]> for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const CONTROLS: usize, T, M> AsMut<[T]> for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const CONTROLS: usize, T, M> Matrix<CONTROLS, 1, T> for ControlVectorBuffer<CONTROLS, T, M> where
    M: MatrixMut<CONTROLS, 1, T>
{
}

impl<const CONTROLS: usize, T, M> MatrixMut<CONTROLS, 1, T> for ControlVectorBuffer<CONTROLS, T, M> where
    M: MatrixMut<CONTROLS, 1, T>
{
}

impl<const CONTROLS: usize, T, M> ControlVector<CONTROLS, T> for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T, M> ControlVectorMut<CONTROLS, T>
    for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T, M> Index<usize> for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const CONTROLS: usize, T, M> IndexMut<usize> for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const CONTROLS: usize, T, M> IntoInnerData for ControlVectorBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, 1, T> + IntoInnerData,
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
        let value: ControlVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: ControlVectorBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }
}
