use core::ops::{Index, IndexMut};

use crate::matrix::traits::{Matrix, MatrixMut};
#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::boxed::Box;

/// A builder for a Kalman filter measurements.
pub struct MatrixData;

impl MatrixData {
    /// Creates an empty matrix.
    pub fn empty<T>() -> MatrixDataArray<0, 0, 0, T>
    where
        T: Default,
    {
        let nothing = [T::default(); 0];
        MatrixDataArray::<0, 0, 0, T>(nothing)
    }

    /// Creates a new matrix buffer that owns the data.
    #[allow(clippy::new_ret_no_self)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<const ROWS: usize, const COLS: usize, T>(init: T) -> MatrixDataBoxed<ROWS, COLS, T>
    where
        T: Copy,
    {
        MatrixDataBoxed::<ROWS, COLS, T>::new(vec![init; ROWS * COLS])
    }

    /// Creates a new matrix buffer that owns the data.
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_boxed<const ROWS: usize, const COLS: usize, T, B>(
        data: B,
    ) -> MatrixDataBoxed<ROWS, COLS, T>
    where
        B: Into<Box<[T]>>,
    {
        MatrixDataBoxed::<ROWS, COLS, T>::new(data.into())
    }

    /// Creates a new matrix buffer that owns the data.
    pub const fn new_array<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>(
        data: [T; TOTAL],
    ) -> MatrixDataArray<ROWS, COLS, TOTAL, T> {
        MatrixDataArray::<ROWS, COLS, TOTAL, T>::new_unchecked(data)
    }

    /// Creates a new matrix buffer that references the data.
    pub const fn new_ref<const ROWS: usize, const COLS: usize, T>(
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
/// * `T` - The data type.
pub struct MatrixDataArray<const ROWS: usize, const COLS: usize, const TOTAL: usize, T = f32>(
    [T; TOTAL],
);

/// Owned boxed data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
pub struct MatrixDataBoxed<const ROWS: usize, const COLS: usize, T = f32>(Box<[T]>);

/// An immutable reference to data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
pub struct MatrixDataRef<'a, const ROWS: usize, const COLS: usize, T = f32>(&'a [T]);

/// A mutable reference to data.
///
/// ## Type arguments
/// * `ROWS` - The number of matrix rows.
/// * `COLS` - The number of matrix columns.
/// * `T` - The data type.
pub struct MatrixDataMut<'a, const ROWS: usize, const COLS: usize, T = f32>(&'a mut [T]);

// TODO: Provide variants that allow taking Box<[T]>

/// Consumes self and returns the wrapped data.
pub trait IntoInnerData {
    type Target;

    fn into_inner(self) -> Self::Target;
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    /// Creates a new instance of the [`MatrixDataArray`] type.
    pub fn new(data: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            assert_eq!(ROWS * COLS, TOTAL);
        }
        Self(data)
    }

    /// Creates a new instance of the [`MatrixDataArray`] type.
    pub const fn new_unchecked(data: [T; TOTAL]) -> Self {
        Self(data)
    }

    /// Returns the inner array.
    pub fn into_inner(self) -> [T; TOTAL] {
        self.0
    }
}

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

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixDataRef<'a, ROWS, COLS, T> {
    /// Creates a new instance of the [`MatrixDataRef`] type.
    pub const fn new(data: &'a [T]) -> Self {
        Self(data)
    }

    /// Returns the inner slice reference.
    pub const fn into_inner(self) -> &'a [T] {
        self.0
    }
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

    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> IntoInnerData for MatrixDataBoxed<ROWS, COLS, T> {
    type Target = Box<[T]>;

    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> IntoInnerData
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    type Target = &'a [T];

    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> IntoInnerData
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    type Target = &'a mut [T];

    fn into_inner(self) -> Self::Target {
        self.into_inner()
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    fn from(value: [T; TOTAL]) -> Self {
        Self::new(value)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T, B> From<B> for MatrixDataBoxed<ROWS, COLS, T>
where
    B: Into<Box<[T]>>,
{
    fn from(value: B) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<&'a [T]>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    fn from(value: &'a [T]) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<&'a mut [T]>
    for MatrixDataRef<'a, ROWS, COLS, T>
{
    fn from(value: &'a mut [T]) -> Self {
        Self::new(value)
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<&'a mut [T]>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
    fn from(value: &'a mut [T]) -> Self {
        Self::new(value)
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> AsRef<[T]>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> AsMut<[T]>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataBoxed<ROWS, COLS, T>
{
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataBoxed<ROWS, COLS, T>
{
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> AsRef<[T]> for MatrixDataBoxed<ROWS, COLS, T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> AsMut<[T]> for MatrixDataBoxed<ROWS, COLS, T> {
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
        self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> AsRef<[T]> for MatrixDataMut<'a, ROWS, COLS, T> {
    fn as_ref(&self) -> &[T] {
        self.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T>
    for MatrixDataMut<'a, ROWS, COLS, T>
{
}

impl<'a, const ROWS: usize, const COLS: usize, T> AsMut<[T]> for MatrixDataMut<'a, ROWS, COLS, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> Index<usize>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> IndexMut<usize>
    for MatrixDataArray<ROWS, COLS, TOTAL, T>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> Index<usize> for MatrixDataBoxed<ROWS, COLS, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
impl<const ROWS: usize, const COLS: usize, T> IndexMut<usize> for MatrixDataBoxed<ROWS, COLS, T> {
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

impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T>
    From<MatrixDataArray<ROWS, COLS, TOTAL, T>> for [T; TOTAL]
{
    fn from(value: MatrixDataArray<ROWS, COLS, TOTAL, T>) -> Self {
        value.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<MatrixDataRef<'a, ROWS, COLS, T>>
    for &'a [T]
{
    fn from(value: MatrixDataRef<'a, ROWS, COLS, T>) -> Self {
        value.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<MatrixDataMut<'a, ROWS, COLS, T>>
    for &'a [T]
{
    fn from(value: MatrixDataMut<'a, ROWS, COLS, T>) -> Self {
        value.0
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> From<MatrixDataMut<'a, ROWS, COLS, T>>
    for &'a mut [T]
{
    fn from(value: MatrixDataMut<'a, ROWS, COLS, T>) -> Self {
        value.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;

    #[cfg(feature = "unsafe")]
    use core::ptr::addr_of;

    #[test]
    #[rustfmt::skip]
    fn owned_buffer() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut a = MatrixData::new_array::<2, 3, 6, f32>(a_buf);
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

    #[test]
    #[rustfmt::skip]
    fn static_buffer() {
        static BUFFER: [f32; 6] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];

        let a = MatrixData::new_ref::<2, 3, f32>(&BUFFER);

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 3.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);
    }

    #[test]
    #[cfg(feature = "unsafe")]
    #[rustfmt::skip]
    fn static_mut_buffer() {
        static mut BUFFER: [f32; 6] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];

        let a = unsafe { MatrixData::new_ref::<2, 3, f32>(&*addr_of!(BUFFER)) };

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 3.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);
    }
}
