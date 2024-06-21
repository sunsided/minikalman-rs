use core::marker::PhantomData;
use core::ops::Index;

use crate::matrix::{IntoInnerData, Matrix, RowMajorSequentialData};

/// An immutable owned data matrix.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
#[derive(Debug)]
pub struct MatrixDataRowMajor<const ROWS: usize, const COLS: usize, S, T = f32>(S, PhantomData<T>)
where
    S: RowMajorSequentialData<ROWS, COLS, T>;

impl<const ROWS: usize, const COLS: usize, S, T> MatrixDataRowMajor<ROWS, COLS, S, T>
where
    S: RowMajorSequentialData<ROWS, COLS, T>,
{
    /// Creates a new instance of the [`RowMajorSequentialData`] type.
    #[inline(always)]
    pub const fn new(data: S) -> Self {
        Self(data, PhantomData)
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> Index<usize>
    for MatrixDataRowMajor<ROWS, COLS, S, T>
where
    S: RowMajorSequentialData<ROWS, COLS, T>,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0.as_slice()[index]
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> IntoInnerData
    for MatrixDataRowMajor<ROWS, COLS, S, T>
where
    S: RowMajorSequentialData<ROWS, COLS, T>,
{
    type Target = S;

    #[inline(always)]
    fn into_inner(self) -> Self::Target {
        self.0
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> From<S> for MatrixDataRowMajor<ROWS, COLS, S, T>
where
    S: RowMajorSequentialData<ROWS, COLS, T>,
{
    #[inline(always)]
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

impl<const ROWS: usize, const COLS: usize, S, T> Matrix<ROWS, COLS, T>
    for MatrixDataRowMajor<ROWS, COLS, S, T>
where
    S: RowMajorSequentialData<ROWS, COLS, T>,
{
}

impl<const ROWS: usize, const COLS: usize, S, T> RowMajorSequentialData<ROWS, COLS, T>
    for MatrixDataRowMajor<ROWS, COLS, S, T>
where
    S: RowMajorSequentialData<ROWS, COLS, T>,
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
