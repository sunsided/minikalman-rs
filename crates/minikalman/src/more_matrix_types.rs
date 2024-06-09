use crate::more_matrix_traits::{Matrix, MatrixMut};
use std::ops::{Index, IndexMut};

/// A builder for a Kalman filter measurements.
pub struct MatrixData;

impl MatrixData {
    /// Creates an empty matrix.
    pub fn empty<T>() -> MatrixDataOwned<0, 0, 0, T>
    where
        T: Default,
    {
        let nothing = [T::default(); 0];
        MatrixDataOwned::<0, 0, 0, T>(nothing)
    }

    /// Creates a new matrix buffer that owns the data.
    pub fn new_owned<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>(
        data: [T; TOTAL],
    ) -> MatrixDataOwned<ROWS, COLS, TOTAL, T> {
        MatrixDataOwned::<ROWS, COLS, TOTAL, T>::new(data)
    }

    /// Creates a new matrix buffer that references the data.
    pub fn new_ref<const ROWS: usize, const COLS: usize, T>(
        data: &[T],
    ) -> MatrixDataRef<ROWS, COLS, T> {
        MatrixDataRef::<ROWS, COLS, T>::new(data)
    }

    /// Creates a new matrix buffer that mutably references the data.
    pub fn new_mut<const ROWS: usize, const COLS: usize, T>(
        data: &mut [T],
    ) -> MatrixDataMut<ROWS, COLS, T> {
        MatrixDataMut::<ROWS, COLS, T>::new(data)
    }
}

/// Owned data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `TOTAL` - The total number of matrix cells (i.e., rows × columns)
pub struct MatrixDataOwned<const ROWS: usize, const COLS: usize, const TOTAL: usize, T = f32>(
    [T; TOTAL],
);

/// An immutable reference to data.
pub struct MatrixDataRef<'a, const ROWS: usize, const COLS: usize, T = f32>(&'a [T]);

/// A mutable reference to data.
pub struct MatrixDataMut<'a, const ROWS: usize, const COLS: usize, T = f32>(&'a mut [T]);

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
    /// Creates a new instance of the [`MatrixDataOwned`] type.
    pub fn new(data: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(ROWS * COLS, TOTAL);
        }
        Self(data)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixDataRef<'a, ROWS, COLS, T> {
    /// Creates a new instance of the [`MatrixDataRef`] type.
    pub fn new(data: &'a [T]) -> Self {
        Self(data)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixDataMut<'a, ROWS, COLS, T> {
    /// Creates a new instance of the [`MatrixDataMut`] type.
    pub fn new(data: &'a mut [T]) -> Self {
        Self(data)
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> AsRef<[T]>
    for MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> AsMut<[T]>
    for MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> AsRef<[T]> for MatrixDataRef<'a, ROWS, COLS, T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> AsRef<[T]> for MatrixDataMut<'a, ROWS, COLS, T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> AsMut<[T]> for MatrixDataMut<'a, ROWS, COLS, T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> Index<usize>
    for MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> IndexMut<usize>
    for MatrixDataOwned<ROWS, COLS, TOTAL, T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Index<usize>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Index<usize>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> IndexMut<usize>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;

    #[test]
    #[rustfmt::skip]
    fn owned_buffer() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut a = MatrixData::new_owned::<2, 3, 6, f32>(a_buf);
        a[2] += 10.0;

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 13.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);
    }

    #[test]
    #[rustfmt::skip]
    fn ref_buffer() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);

        assert_f32_near!(a_buf[0], 1.);
        assert_f32_near!(a_buf[1], 2.);
        assert_f32_near!(a_buf[2], 3.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);
    }

    #[test]
    #[rustfmt::skip]
    fn mut_buffer() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut a = MatrixData::new_mut::<2, 3, f32>(&mut a_buf);
        a[2] += 10.0;

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 13.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);
    }
}
