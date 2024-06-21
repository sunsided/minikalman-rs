use crate::matrix::{
    Matrix, MatrixDataArray, MatrixMut, RowMajorSequentialData, RowMajorSequentialDataMut,
};
use std::ops::{Index, IndexMut};

/// A mutable reference to data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
#[derive(Debug)]
pub struct MatrixDataMut<'a, const ROWS: usize, const COLS: usize, T = f32>(&'a mut [T]);

/// Consumes self and returns the wrapped data.
pub trait IntoInnerData {
    type Target;

    fn into_inner(self) -> Self::Target;
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixDataMut<'a, ROWS, COLS, T> {
    /// Creates a new instance of the [`MatrixDataMut`] type.
    pub fn new(data: &'a mut [T]) -> Self {
        Self(data)
    }

    /// Returns the inner mutable slice reference.
    pub fn into_inner(self) -> &'a mut [T] {
        self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> IntoInnerData
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    type Target = [T; TOTAL];

    #[inline(always)]
    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> IntoInnerData
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    type Target = &'a mut [T];

    #[inline(always)]
    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    #[inline(always)]
    fn from(value: [T; TOTAL]) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<&'a mut [T]>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn from(value: &'a mut [T]) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> RowMajorSequentialData<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> RowMajorSequentialDataMut<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Index<usize>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> IndexMut<usize>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<MatrixDataMut<'a, ROWS, COLS, T>>
    for &'a [T]
{
    #[inline(always)]
    fn from(value: MatrixDataMut<'a, ROWS, COLS, T>) -> Self {
        value.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<MatrixDataMut<'a, ROWS, COLS, T>>
    for &'a mut [T]
{
    #[inline(always)]
    fn from(value: MatrixDataMut<'a, ROWS, COLS, T>) -> Self {
        value.0
    }
}
