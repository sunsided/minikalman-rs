/// Data laid out sequentially and in a row-major layout.
pub trait RowMajorSequentialData<const ROWS: usize, const COLS: usize, T> {
    fn as_slice(&self) -> &[T];

    /// Gets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// * `cols` - The column
    ///
    /// ## Returns
    /// The value at the given cell.
    #[inline(always)]
    #[doc(alias = "matrix_get")]
    fn get_at(&self, row: usize, column: usize) -> T
    where
        T: Copy,
    {
        let m = self.as_slice();
        m[row * COLS + column]
    }
}

/// Data laid out sequentially and in a row-major layout.
pub trait RowMajorSequentialDataMut<const ROWS: usize, const COLS: usize, T>:
    RowMajorSequentialData<ROWS, COLS, T>
{
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Gets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// * `cols` - The column
    ///
    /// ## Returns
    /// The value at the given cell.
    #[inline(always)]
    #[doc(alias = "matrix_set")]
    fn set_at(&mut self, row: usize, column: usize, value: T) {
        let m = self.as_mut_slice();
        m[row * COLS + column] = value;
    }
}
