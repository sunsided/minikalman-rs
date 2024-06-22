use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::{ControlProcessNoiseCovarianceMatrix, ControlProcessNoiseCovarianceMatrixMut};
use crate::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef};
use crate::matrix::{Matrix, MatrixMut};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// Immutable buffer for the control process noise covariance matrix (`num_controls` × `num_controls`).
///
/// This matrix represents the control process noise covariance. It quantifies the
/// uncertainty introduced by the control inputs, reflecting how much the true state
/// is expected to deviate from the predicted state due to noise and variations
/// in the control process. The matrix is calculated as B×Q×Bᵀ, where B
/// represents the control input model, and Q is the process noise covariance (this matrix).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ControlProcessNoiseCovarianceMatrixBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ControlProcessNoiseCovarianceMatrixBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let data = [0.0; 4];
/// let buffer = ControlProcessNoiseCovarianceMatrixBuffer::<2, f32, _>::from(data.as_ref());
/// ```
#[doc(alias = "ControlCovarianceMatrixBuffer")]
pub struct ControlProcessNoiseCovarianceMatrixBuffer<const CONTROLS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: Matrix<CONTROLS, CONTROLS, T>;

/// Mutable buffer for the control process noise covariance matrix (`num_controls` × `num_controls`).
///
/// This matrix represents the control process noise covariance. It quantifies the
/// uncertainty introduced by the control inputs, reflecting how much the true state
/// is expected to deviate from the predicted state due to noise and variations
/// in the control process. The matrix is calculated as B×Q×Bᵀ, where B
/// represents the control input model, and Q is the process noise covariance (this matrix).
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ControlProcessNoiseCovarianceMatrixMutBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ControlProcessNoiseCovarianceMatrixMutBuffer::new(MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = ControlProcessNoiseCovarianceMatrixMutBuffer::<2, f32, _>::from(data.as_mut_slice());
/// ```
#[doc(alias = "ControlCovarianceMatrixMutBuffer")]
pub struct ControlProcessNoiseCovarianceMatrixMutBuffer<const CONTROLS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<CONTROLS, CONTROLS, T>;

// -----------------------------------------------------------

impl<const CONTROLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for ControlProcessNoiseCovarianceMatrixBuffer<
        CONTROLS,
        T,
        MatrixDataArray<CONTROLS, CONTROLS, TOTAL, T>,
    >
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
    for ControlProcessNoiseCovarianceMatrixBuffer<
        CONTROLS,
        T,
        MatrixDataRef<'a, CONTROLS, CONTROLS, T>,
    >
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
    for ControlProcessNoiseCovarianceMatrixBuffer<
        CONTROLS,
        T,
        MatrixDataRef<'a, CONTROLS, CONTROLS, T>,
    >
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
    for ControlProcessNoiseCovarianceMatrixMutBuffer<
        CONTROLS,
        T,
        MatrixDataMut<'a, CONTROLS, CONTROLS, T>,
    >
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
    for ControlProcessNoiseCovarianceMatrixMutBuffer<
        CONTROLS,
        T,
        MatrixDataArray<CONTROLS, CONTROLS, TOTAL, T>,
    >
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

impl<const CONTROLS: usize, T, M> ControlProcessNoiseCovarianceMatrixBuffer<CONTROLS, T, M>
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

impl<const CONTROLS: usize, T, M> RowMajorSequentialData<CONTROLS, CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const CONTROLS: usize, T, M> Matrix<CONTROLS, CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
}

impl<const CONTROLS: usize, T, M> ControlProcessNoiseCovarianceMatrix<CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T, M> ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
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

impl<const CONTROLS: usize, T, M> RowMajorSequentialData<CONTROLS, CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const CONTROLS: usize, T, M> RowMajorSequentialDataMut<CONTROLS, CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const CONTROLS: usize, T, M> Matrix<CONTROLS, CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
}

impl<const CONTROLS: usize, T, M> MatrixMut<CONTROLS, CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
}

impl<const CONTROLS: usize, T, M> ControlProcessNoiseCovarianceMatrix<CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T, M> ControlProcessNoiseCovarianceMatrixMut<CONTROLS, T>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T, M> Index<usize>
    for ControlProcessNoiseCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: Matrix<CONTROLS, CONTROLS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const CONTROLS: usize, T, M> Index<usize>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const CONTROLS: usize, T, M> IndexMut<usize>
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const CONTROLS: usize, T, M> IntoInnerData
    for ControlProcessNoiseCovarianceMatrixBuffer<CONTROLS, T, M>
where
    M: MatrixMut<CONTROLS, CONTROLS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const CONTROLS: usize, T, M> IntoInnerData
    for ControlProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, M>
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
        let value: ControlProcessNoiseCovarianceMatrixBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_ref() {
        let data = [0.0_f32; 100];
        let value: ControlProcessNoiseCovarianceMatrixBuffer<5, f32, _> = data.as_ref().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
        assert!(!value.is_empty());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ControlProcessNoiseCovarianceMatrixBuffer<5, f32, _> =
            data.as_mut_slice().into();
        assert_eq!(value.len(), 25);
        assert!(value.is_valid());
        assert!(!value.is_empty());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_from_array_invalid_size() {
        let value: ControlProcessNoiseCovarianceMatrixBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    fn test_mut_from_array() {
        let value: ControlProcessNoiseCovarianceMatrixMutBuffer<5, f32, _> = [0.0; 100].into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_mut_from_mut() {
        let mut data = [0.0_f32; 100];
        let value: ControlProcessNoiseCovarianceMatrixMutBuffer<5, f32, _> =
            data.as_mut_slice().into();
        assert_eq!(value.len(), 25);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    #[cfg(feature = "no_assert")]
    fn test_mut_from_array_invalid_size() {
        let value: ControlProcessNoiseCovarianceMatrixMutBuffer<5, f32, _> = [0.0; 1].into();
        assert!(!value.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn test_access() {
        let mut value: ControlProcessNoiseCovarianceMatrixMutBuffer<5, f32, _> = [0.0; 25].into();

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
