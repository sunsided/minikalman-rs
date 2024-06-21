use crate::matrix::row_major::{RowMajorSequentialData, RowMajorSequentialDataMut};
use nalgebra::{ArrayStorage, Const, IsContiguous, Matrix, RawStorage, VecStorage};

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

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl<const ROWS: usize, const COLS: usize, T> RowMajorSequentialData<ROWS, COLS, T>
    for Matrix<T, Const<ROWS>, Const<COLS>, ArrayStorage<T, ROWS, COLS>>
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl<const ROWS: usize, const COLS: usize, T> RowMajorSequentialDataMut<ROWS, COLS, T>
    for Matrix<T, Const<ROWS>, Const<COLS>, ArrayStorage<T, ROWS, COLS>>
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl<const ROWS: usize, const COLS: usize, T> RowMajorSequentialData<ROWS, COLS, T>
    for Matrix<T, Const<ROWS>, Const<COLS>, VecStorage<T, Const<ROWS>, Const<COLS>>>
{
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    #[inline(always)]
    fn get_at(&self, row: usize, column: usize) -> T
    where
        T: Copy,
    {
        self.data.as_slice()[row * COLS + column]
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
impl<const ROWS: usize, const COLS: usize, T> RowMajorSequentialDataMut<ROWS, COLS, T>
    for Matrix<T, Const<ROWS>, Const<COLS>, VecStorage<T, Const<ROWS>, Const<COLS>>>
{
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    #[inline(always)]
    fn set_at(&mut self, row: usize, column: usize, value: T) {
        self.data.as_mut_slice()[row * COLS + column] = value
    }
}

// ---------

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(all(feature = "nalgebra", feature = "unsafe"))]
unsafe impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    RawStorage<T, Const<ROWS>, Const<COLS>>
    for crate::matrix::MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    type RStride = Const<1>;
    type CStride = Const<ROWS>;

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
