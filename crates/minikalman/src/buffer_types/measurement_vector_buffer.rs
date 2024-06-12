use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{MeasurementVector, MeasurementVectorMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use crate::matrix::{Matrix, MatrixMut};

// TODO: Add MeasurementVectorMutBuffer

/// Mutable buffer for the observation (measurement) vector (`num_measurements` × `1`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = MeasurementVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = MeasurementVectorBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct MeasurementVectorBuffer<const MEASUREMENTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<MEASUREMENTS, 1, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for MeasurementVectorBuffer<MEASUREMENTS, T, MatrixDataMut<'a, MEASUREMENTS, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert!(MEASUREMENTS <= value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, 1, T>(value))
    }
}

/// # Example
/// Buffers can be trivially constructed from correctly-sized arrays:
///
/// ```
/// # use minikalman::prelude::MeasurementVectorBuffer;
/// let _value: MeasurementVectorBuffer<5, f32, _> = [0.0; 5].into();
/// ```
///
/// Invalid buffer sizes fail to compile:
///
/// ```fail_compile
/// # use minikalman::prelude::MeasurementVectorBuffer;
/// let _value: MeasurementVectorBuffer<5, f32, _> = [0.0; 1].into();
/// ```
impl<const MEASUREMENTS: usize, T> From<[T; MEASUREMENTS]>
    for MeasurementVectorBuffer<MEASUREMENTS, T, MatrixDataArray<MEASUREMENTS, 1, MEASUREMENTS, T>>
{
    fn from(value: [T; MEASUREMENTS]) -> Self {
        Self::new(MatrixData::new_array::<MEASUREMENTS, 1, MEASUREMENTS, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        MEASUREMENTS
    }

    pub const fn is_empty(&self) -> bool {
        MEASUREMENTS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }
}

impl<const MEASUREMENTS: usize, T, M> AsRef<[T]> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, T, M> Index<usize> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> AsMut<[T]> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, T, M> Matrix<MEASUREMENTS, 1, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MatrixMut<MEASUREMENTS, 1, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MeasurementVector<MEASUREMENTS, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, T, M> MeasurementVectorMut<MEASUREMENTS, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> IntoInnerData for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T> + IntoInnerData,
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
        let value: MeasurementVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: MeasurementVectorBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }
}
