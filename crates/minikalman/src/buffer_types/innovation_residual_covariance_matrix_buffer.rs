use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::matrix_traits::{Matrix, MatrixMut};
use crate::type_traits::ResidualCovarianceMatrix;
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned};

/// Buffer for the square innovation (residual) covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Example
/// ```
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = InnovationResidualCovarianceMatrixBuffer::new(MatrixData::new_owned::<2, 2, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InnovationResidualCovarianceMatrixBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct InnovationResidualCovarianceMatrixBuffer<const MEASUREMENTS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for InnovationResidualCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataMut<'a, MEASUREMENTS, MEASUREMENTS, T>,
    >
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * MEASUREMENTS, value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, MEASUREMENTS, T>(value))
    }
}

impl<const MEASUREMENTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InnovationResidualCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataOwned<MEASUREMENTS, MEASUREMENTS, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * MEASUREMENTS, TOTAL);
        }
        Self::new(MatrixData::new_owned::<MEASUREMENTS, MEASUREMENTS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }
}

impl<const MEASUREMENTS: usize, T, M> AsRef<[T]>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, T, M> AsMut<[T]>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, T, M> Matrix<MEASUREMENTS, MEASUREMENTS, T>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MatrixMut<MEASUREMENTS, MEASUREMENTS, T>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> ResidualCovarianceMatrix<MEASUREMENTS, T>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    type Target = M;
    type TargetMut = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const MEASUREMENTS: usize, T, M> Index<usize>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> IntoInnerData
    for InnovationResidualCovarianceMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
