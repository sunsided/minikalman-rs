use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{ControlMatrix, ControlMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Immutable buffer for the control matrix (`num_states` × `num_controls`), typically denoted "B".
///
/// This matrix represents the control input model. It defines how the control inputs
/// influence the state evolution from one time step to the next. The matrix \( B \)
/// is used to incorporate the effect of control inputs into the state transition,
/// allowing the model to account for external controls applied to the system.
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

/// Mutable buffer for the control matrix (`num_states` × `num_controls`), typically denoted "B".
///
/// This matrix represents the control input model. It defines how the control inputs
/// influence the state evolution from one time step to the next. The matrix \( B \)
/// is used to incorporate the effect of control inputs into the state transition,
/// allowing the model to account for external controls applied to the system.
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
/// let buffer = ControlMatrixMutBuffer::<2, 2, f32, _>::from(data.as_mut_slice());
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

impl<const STATES: usize, const CONTROLS: usize, T, M> RowMajorSequentialData<STATES, CONTROLS, T>
    for ControlMatrixBuffer<STATES, CONTROLS, T, M>
where
    M: Matrix<STATES, CONTROLS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
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

impl<const STATES: usize, const CONTROLS: usize, T, M> RowMajorSequentialData<STATES, CONTROLS, T>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, M>
    RowMajorSequentialDataMut<STATES, CONTROLS, T>
    for ControlMatrixMutBuffer<STATES, CONTROLS, T, M>
where
    M: MatrixMut<STATES, CONTROLS, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
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
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ControlMatrixBuffer<5, 3, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
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
        let value: ControlMatrixMutBuffer<5, 3, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: ControlMatrixMutBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn test_access() {
        let mut value: ControlMatrixMutBuffer<5, 5, f32, _> = [0.0; 25].into();

        // Set values.
        {
            let matrix = value.as_matrix_mut();
            for i in 0..matrix.cols() {
                matrix.set_symmetric(0, i, i as _);
                matrix.set_at(i, i, i as _);
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
                assert_eq!(matrix.get_at(0, i), 10.0 + i as f32);
                assert_eq!(matrix.get_at(i, 0), 10.0 + i as f32);
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
