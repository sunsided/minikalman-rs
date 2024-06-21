/// Data laid out sequentially and in a row-major layout.
pub trait RowMajorSequentialData<const ROWS: usize, const COLS: usize, T> {
    fn as_slice(&self) -> &[T];
}

/// Data laid out sequentially and in a row-major layout.
pub trait RowMajorSequentialDataMut<const ROWS: usize, const COLS: usize, T>:
    RowMajorSequentialData<ROWS, COLS, T>
{
    fn as_mut_slice(&mut self) -> &mut [T];
}
