/// Data laid out sequentially and in a row-major layout.
pub trait RowMajorSequentialData<const ROWS: usize, const COLS: usize, T> {
    /// Gets the data as a slice.
    ///
    /// The data is expected to be in row-major format, i.e. neighboring column values are next
    /// to each other in memory.
    fn as_slice(&self) -> &[T];

    /// Gets the length of the data.
    #[inline(always)]
    fn len(&self) -> usize {
        ROWS * COLS
    }

    /// Gets the length of the underlying data buffer.
    #[inline(always)]
    fn buffer_len(&self) -> usize {
        self.as_slice().len()
    }

    /// Determines if this data is empty.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        ROWS * COLS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    fn is_valid(&self) -> bool {
        self.len() <= self.buffer_len()
    }

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
