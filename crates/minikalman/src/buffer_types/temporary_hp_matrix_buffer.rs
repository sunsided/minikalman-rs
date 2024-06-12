use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use minikalman_traits::kalman::TemporaryHPMatrix;

use minikalman_traits::matrix::{IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut};
use minikalman_traits::matrix::{Matrix, MatrixMut};

pub struct TemporaryHPMatrixBuffer<const MEASUREMENTS: usize, const STATES: usize, T, M>(
    M,
    PhantomData<T>,
)
where
    M: Matrix<MEASUREMENTS, STATES, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, const STATES: usize, T> From<&'a mut [T]>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, MatrixDataMut<'a, MEASUREMENTS, STATES, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * STATES, value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, STATES, T>(value))
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for TemporaryHPMatrixBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataArray<MEASUREMENTS, STATES, TOTAL, T>,
    >
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS * STATES, TOTAL);
        }
        Self::new(MatrixData::new_array::<MEASUREMENTS, STATES, TOTAL, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
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
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: Matrix<MEASUREMENTS, STATES, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> AsMut<[T]>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Matrix<MEASUREMENTS, STATES, T>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> MatrixMut<MEASUREMENTS, STATES, T>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M>
    TemporaryHPMatrix<MEASUREMENTS, STATES, T>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
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

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> Index<usize>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IndexMut<usize>
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, const STATES: usize, T, M> IntoInnerData
    for TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, M>
where
    M: MatrixMut<MEASUREMENTS, STATES, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
