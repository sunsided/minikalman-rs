use nalgebra::{Const, IsContiguous, RawStorage};

use crate::matrix::row_major::RowMajorSequentialData;

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(all(feature = "nalgebra", feature = "unsafe"))]
unsafe impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> IsContiguous
    for crate::matrix::MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}

#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "nalgebra", feature = "unsafe", feature = "alloc")))
)]
#[cfg(all(feature = "nalgebra", feature = "unsafe", feature = "alloc"))]
unsafe impl<const ROWS: usize, const COLS: usize, T> IsContiguous
    for crate::matrix::MatrixDataBoxed<ROWS, COLS, T>
{
}

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(all(feature = "nalgebra", feature = "unsafe"))]
unsafe impl<'a, const ROWS: usize, const COLS: usize, T> IsContiguous
    for crate::matrix::MatrixDataRef<'a, ROWS, COLS, T>
{
}

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(all(feature = "nalgebra", feature = "unsafe"))]
unsafe impl<'a, const ROWS: usize, const COLS: usize, T> IsContiguous
    for crate::matrix::MatrixDataMut<'a, ROWS, COLS, T>
{
}

// ---------

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(all(feature = "nalgebra", feature = "unsafe"))]
unsafe impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    RawStorage<T, Const<ROWS>, Const<COLS>>
    for crate::matrix::MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    type RStride = Const<COLS>;
    type CStride = Const<1>;

    #[inline]
    fn ptr(&self) -> *const T {
        // nalgebra reads ROWS×COLS elements through this pointer without bounds checks.
        // Kept even under `no_assert`: it guards memory safety, and folds away when it holds.
        assert!(ROWS * COLS <= TOTAL, "buffer smaller than ROWS × COLS");
        self.as_slice().as_ptr()
    }

    #[inline]
    fn shape(&self) -> (Const<ROWS>, Const<COLS>) {
        (Const, Const)
    }

    #[inline]
    fn strides(&self) -> (Self::RStride, Self::CStride) {
        (Const, Const)
    }

    #[inline]
    fn is_contiguous(&self) -> bool {
        true
    }

    #[inline]
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr(), ROWS * COLS)
    }
}

#[cfg(all(test, feature = "nalgebra", feature = "unsafe"))]
mod tests {
    use crate::matrix::MatrixData;
    use nalgebra::*;

    #[test]
    fn test_matrix() {
        let a_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = MatrixData::new_array::<2, 3, 6, f32>(a_buf);
        let mat = Matrix::from_data(a);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
    }

    #[test]
    #[should_panic(expected = "buffer smaller than ROWS × COLS")]
    fn undersized_buffer_is_rejected() {
        // Two values cannot back a 3×3 matrix; nalgebra would read past the array.
        let a = MatrixData::new_array::<3, 3, 2, f32>([1.0, 2.0]);
        let mat = Matrix::from_data(a);
        let _ = mat.sum();
    }

    /* TODO: Add implementations for all buffer types.
    #[test]
    fn test_state_matrix() {
        let a_buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buffer = ControlMatrixMutBuffer::<3, 5, f32, _>::from(a_buf);
        let mat = Matrix::from_data(buffer.as_matrix());
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 5);
    }
    */
}
