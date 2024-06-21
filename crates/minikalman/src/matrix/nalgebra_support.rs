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

/*
#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(all(feature = "nalgebra", feature = "unsafe"))]
unsafe impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    Storage<T, Const<ROWS>, Const<COLS>> for crate::matrix::MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}
*/

/*
impl<const ROWS: usize, const COLS: usize, T, S> crate::matrix::Matrix<ROWS, COLS, T> for S where
    S: RawStorage<T, Const<ROWS>, Const<COLS>>
{
}
*/

/*
impl<const ROWS: usize, const COLS: usize, T, S> crate::matrix::Matrix<ROWS, COLS, T>
    for Matrix<T, Const<ROWS>, Const<COLS>, S>
where
    T: Scalar,
    S: IsContiguous,
{
}
*/
