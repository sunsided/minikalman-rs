use nalgebra::IsContiguous;

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
