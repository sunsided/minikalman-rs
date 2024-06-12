use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{MeasurementObservationMatrix, MeasurementObservationMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};

/// Immutable buffer for the observation matrix (`num_inputs` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::MeasurementObservationMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = MeasurementObservationMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = MeasurementObservationMatrixBuffer::<2, 2, f32, _>::from(data.as_ref());
/// ```
pub struct MeasurementObservationMatrixBuffer<const MEASUREMENTS: usize, const STATES: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: Matrix<MEASUREMENTS, STATES, T>;

/// Mmutable buffer for the observation matrix (`num_inputs` × `num_states`).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::MeasurementObservationMatrixMutBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = MeasurementObservationMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = MeasurementObservationMatrixMutBuffer::<2, 2, f32, _>::from(data.as_mut());
/// ```
pub struct MeasurementObservationMatrixMutBuffer<
    const MEASUREMENTS: usize,
    const STATES: usize,
    T,
    M,
>(M, PhantomData<T>)
where
    M: MatrixMut<MEASUREMENTS, STATES, T>;

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MeasurementObservationMatrixBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataArray<MEASUREMENTS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<MEASUREMENTS, STATES, TOTAL, T>(
            value,
        ))
    }
}

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a [T]>
    for MeasurementObservationMatrixBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataRef<'a, MEASUREMENTS, STATES, T>,
    >
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS * STATES <= value.len());
        }
        Self::new(MatrixData::new_ref::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a mut [T]>
    for MeasurementObservationMatrixBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataRef<'a, MEASUREMENTS, STATES, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS * STATES <= value.len());
        }
        Self::new(MatrixData::new_ref::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a mut [T]>
    for MeasurementObservationMatrixMutBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataMut<'a, MEASUREMENTS, STATES, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS * STATES <= value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MeasurementObservationMatrixMutBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataArray<MEASUREMENTS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS * STATES <= TOTAL);
        }
        Self::new(MatrixData::new_array::<MEASUREMENTS, STATES, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementObservationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        MEASUREMENTS * STATES
    }

    pub const fn is_empty(&self) -> bool {
        MEASUREMENTS * STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsRef<[T]>
    for MeasurementObservationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Matrix<MEASUREMENTS, STATES, T>
    for MeasurementObservationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementObservationMatrix<MEASUREMENTS, STATES, T>
    for MeasurementObservationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        MEASUREMENTS * STATES
    }

    pub const fn is_empty(&self) -> bool {
        MEASUREMENTS * STATES == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsRef<[T]>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsMut<[T]>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Matrix<MEASUREMENTS, STATES, T>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> MatrixMut<MEASUREMENTS, STATES, T>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementObservationMatrix<MEASUREMENTS, STATES, T>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    MeasurementObservationMatrixMut<MEASUREMENTS, STATES, T>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Index<usize>
    for MeasurementObservationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Index<usize>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IndexMut<usize>
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IntoInnerData
    for MeasurementObservationMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IntoInnerData
    for MeasurementObservationMatrixMutBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T> + IntoInnerData,
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
        let value: MeasurementObservationMatrixBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: MeasurementObservationMatrixBuffer<5, 3, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: MeasurementObservationMatrixBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: MeasurementObservationMatrixBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: MeasurementObservationMatrixMutBuffer<5, 3, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: MeasurementObservationMatrixMutBuffer<5, 3, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 15);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: MeasurementObservationMatrixMutBuffer<5, 3, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }
}
