use crate::matrix::{IntoInnerData, Matrix, RowMajorSequentialData};
use core::ops::Index;

/// An immutable reference to data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
#[derive(Debug, Clone)]
pub struct MatrixDataRef<'a, const ROWS: usize, const COLS: usize, T = f32>(&'a [T]);

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixDataRef<'a, ROWS, COLS, T> {
    /// Creates a new instance of the [`MatrixDataRef`] type.
    #[inline(always)]
    pub const fn new(data: &'a [T]) -> Self {
        Self(data)
    }

    /// Returns the inner slice reference.
    #[inline(always)]
    pub const fn into_inner(self) -> &'a [T] {
        self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> IntoInnerData
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    type Target = &'a [T];

    #[inline(always)]
    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<&'a [T]>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn from(value: &'a [T]) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<&'a mut [T]>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn from(value: &'a mut [T]) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> RowMajorSequentialData<ROWS, COLS, T>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Index<usize>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<MatrixDataRef<'a, ROWS, COLS, T>>
    for &'a [T]
{
    #[inline(always)]
    fn from(value: MatrixDataRef<'a, ROWS, COLS, T>) -> Self {
        value.0
    }
}
