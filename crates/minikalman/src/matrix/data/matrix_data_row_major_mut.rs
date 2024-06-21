use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::matrix::{IntoInnerData, Matrix, MatrixMut, RowMajorSequentialData};
use crate::prelude::RowMajorSequentialDataMut;

/// A mutable owned data matrix.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
#[derive(Debug)]
pub struct MatrixDataRowMajorMut<const ROWS: usize, const COLS: usize, S, T = f32>(
    S,
    PhantomData<T>,
)
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>;

impl<const ROWS: usize, const COLS: usize, S, T> MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    /// Creates a new instance of the [`MatrixDataRowMajorMut`] type.
    #[inline(always)]
    pub const fn new(data: S) -> Self {
        Self(data, PhantomData)
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> Index<usize>
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0.as_slice()[index]
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> IndexMut<usize>
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0.as_mut_slice()[index]
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> IntoInnerData
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    type Target = S;

    #[inline(always)]
    fn into_inner(self) -> Self::Target {
        self.0
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> From<S> for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    #[inline(always)]
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> Matrix<ROWS, COLS, T>
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
}

impl<const ROWS: usize, const COLS: usize, S, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
}

impl<const ROWS: usize, const COLS: usize, S, T> RowMajorSequentialData<ROWS, COLS, T>
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    #[inline(always)]
    fn get_at(&self, row: usize, column: usize) -> T
    where
        T: Copy,
    {
        self.0.get_at(row, column)
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> RowMajorSequentialDataMut<ROWS, COLS, T>
    for MatrixDataRowMajorMut<ROWS, COLS, S, T>
where
    S: RowMajorSequentialDataMut<ROWS, COLS, T>,
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }

    #[inline(always)]
    fn set_at(&mut self, row: usize, column: usize, value: T) {
        self.0.set_at(row, column, value)
    }
}
