use crate::matrix::{Matrix, MatrixMut, RowMajorSequentialData, RowMajorSequentialDataMut};
use std::ops::{Index, IndexMut};

/// Owned data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `TOTAL` - The total number of matrix cells (i.e., rows × columns)
/// * `T` - The data type.
#[derive(Debug, Clone)]
pub struct MatrixDataArray<const ROWS: usize, const COLS: usize, const TOTAL: usize, T = f32>(
    [T; TOTAL],
);

impl<T> Default for MatrixDataArray<0, 0, 0, T> {
    #[inline]
    fn default() -> Self {
        MatrixDataArray::<0, 0, 0, T>([])
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    /// Creates a new instance of the [`MatrixDataArray`] type.
    pub fn new(data: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            assert_eq!(ROWS * COLS, TOTAL);
        }
        Self(data)
    }

    /// Creates a new instance of the [`MatrixDataArray`] type.
    pub const fn new_unchecked(data: [T; TOTAL]) -> Self {
        Self(data)
    }

    /// Returns the inner array.
    pub fn into_inner(self) -> [T; TOTAL] {
        self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    RowMajorSequentialData<ROWS, COLS, T> for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    RowMajorSequentialDataMut<ROWS, COLS, T> for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> Index<usize>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> IndexMut<usize>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    From<MatrixDataArray<ROWS, COLS, TOTAL, T>> for [T; TOTAL]
{
    #[inline(always)]
    fn from(value: MatrixDataArray<ROWS, COLS, TOTAL, T>) -> Self {
        value.0
    }
}
