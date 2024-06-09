use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::filter_traits::TemporaryResidualCovarianceInvertedMatrix;
use crate::matrix_traits::{Matrix, MatrixMut};
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned};

pub struct TemporaryResidualCovarianceInvertedMatrixBuffer<const MEASUREMENTS: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<
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

impl<'a, const MEASUREMENTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<
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

impl<const MEASUREMENTS: usize, T, M>
    TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    pub fn new(matrix: M) -> Self {
        Self(matrix, PhantomData::default())
    }
}

impl<const MEASUREMENTS: usize, T, M> AsRef<[T]>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, T, M> AsMut<[T]>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, T, M> Matrix<MEASUREMENTS, MEASUREMENTS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MatrixMut<MEASUREMENTS, MEASUREMENTS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
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
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> IntoInnerData
    for TemporaryResidualCovarianceInvertedMatrixBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, MEASUREMENTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
