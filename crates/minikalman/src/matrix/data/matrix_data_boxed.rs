use crate::matrix::{
    IntoInnerData, Matrix, MatrixMut, RowMajorSequentialData, RowMajorSequentialDataMut,
};

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::boxed::Box;

#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::boxed::Box;

use core::ops::{Index, IndexMut};

/// Owned boxed data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
#[derive(Debug, Clone)]
pub struct MatrixDataBoxed<const ROWS: usize, const COLS: usize, T = f32>(Box<[T]>);

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> MatrixDataBoxed<ROWS, COLS, T> {
    /// Creates a new instance of the [`MatrixDataBoxed`] type.
    pub fn new<B>(data: B) -> Self
    where
        B: Into<Box<[T]>>,
    {
        let data = data.into();
        #[cfg(not(feature = "no_assert"))]
        {
            assert_eq!(ROWS * COLS, data.len());
        }
        Self(data)
    }

    /// Returns the inner array.
    pub fn into_inner(self) -> Box<[T]> {
        self.0
    }
}

impl<const ROWS: usize, const COLS: usize, T> IntoInnerData for MatrixDataBoxed<ROWS, COLS, T> {
    type Target = Box<[T]>;

    #[inline(always)]
    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<const ROWS: usize, const COLS: usize, T, B> From<B> for MatrixDataBoxed<ROWS, COLS, T>
where
    B: Into<Box<[T]>>,
{
    #[inline(always)]
    fn from(value: B) -> Self {
        Self::new(value)
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataBoxed<ROWS, COLS, T>
{
}

impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataBoxed<ROWS, COLS, T>
{
}

impl<const ROWS: usize, const COLS: usize, T> RowMajorSequentialData<ROWS, COLS, T>
    for MatrixDataBoxed<ROWS, COLS, T>
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<const ROWS: usize, const COLS: usize, T> RowMajorSequentialDataMut<ROWS, COLS, T>
    for MatrixDataBoxed<ROWS, COLS, T>
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<const ROWS: usize, const COLS: usize, T> Index<usize> for MatrixDataBoxed<ROWS, COLS, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const ROWS: usize, const COLS: usize, T> IndexMut<usize> for MatrixDataBoxed<ROWS, COLS, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
