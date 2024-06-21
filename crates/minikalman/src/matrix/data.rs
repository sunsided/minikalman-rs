mod matrix_data_array;
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
#[cfg(any(feature = "alloc", feature = "std"))]
mod matrix_data_boxed;
mod matrix_data_mut;
mod matrix_data_ref;
mod matrix_data_row_major;
mod matrix_data_row_major_mut;

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::boxed::Box;

#[cfg(all(feature = "std", not(feature = "alloc")))]
use std::boxed::Box;

pub use matrix_data_array::*;
pub use matrix_data_mut::*;
pub use matrix_data_ref::*;
pub use matrix_data_row_major::*;
pub use matrix_data_row_major_mut::*;

use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};
#[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
#[cfg(any(feature = "alloc", feature = "std"))]
pub use matrix_data_boxed::*;

/// A builder for a Kalman filter measurements.
pub struct MatrixData;

impl MatrixData {
    /// Creates an empty matrix.
    pub fn empty<T>() -> MatrixDataArray<0, 0, 0, T>
    where
        T: Default,
    {
        MatrixDataArray::<0, 0, 0, T>::default()
    }

    /// Creates a new matrix buffer from a given storage.
    #[allow(clippy::new_ret_no_self)]
    pub fn new_from<const ROWS: usize, const COLS: usize, T, S>(
        storage: S,
    ) -> MatrixDataRowMajor<ROWS, COLS, S, T>
    where
        T: Copy,
        S: RowMajorSequentialData<ROWS, COLS, T>,
    {
        MatrixDataRowMajor::from(storage)
    }

    /// Creates a new mutable matrix buffer from a given storage.
    #[allow(clippy::new_ret_no_self)]
    pub fn new_mut_from<const ROWS: usize, const COLS: usize, T, S>(
        storage: S,
    ) -> MatrixDataRowMajorMut<ROWS, COLS, S, T>
    where
        T: Copy,
        S: RowMajorSequentialDataMut<ROWS, COLS, T>,
    {
        MatrixDataRowMajorMut::from(storage)
    }

    /// Creates a new matrix buffer that owns the data.
    #[allow(clippy::new_ret_no_self)]
    #[cfg_attr(docsrs, doc(cfg(any(feature = "alloc", feature = "std"))))]
    #[cfg(any(feature = "alloc", feature = "std"))]
    pub fn new<const ROWS: usize, const COLS: usize, T>(init: T) -> MatrixDataBoxed<ROWS, COLS, T>
    where
        T: Copy,
    {
        MatrixDataBoxed::<ROWS, COLS, T>::new(alloc::vec![init; ROWS * COLS])
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

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::*;

    use crate::prelude::{
        ColumnVector, ColumnVectorMut, RowMajorSequentialData, RowVector, RowVectorMut, Scalar,
        ScalarMut,
    };
    #[cfg(feature = "unsafe")]
    use core::ptr::addr_of;

    #[test]
    #[cfg(feature = "alloc")]
    #[rustfmt::skip]
    fn aray_buffer() {
        let a = MatrixData::new::<2, 3, f32>(0.0);
        assert_eq!(a.len(), 6);
        assert_eq!(a.buffer_len(), 6);
        assert!(!a.is_empty());
        assert!(a.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn array_buffer() {
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

        assert_eq!(a.len(), 6);
        assert_eq!(a.buffer_len(), 6);
        assert!(!a.is_empty());
        assert!(a.is_valid());
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

    #[test]
    #[rustfmt::skip]
    fn from_array() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut a = MatrixDataArray::<2, 3, 6, f32>::from(a_buf);
        a[2] += 10.0;

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 13.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);

        assert_eq!(a.len(), 6);
        assert_eq!(a.buffer_len(), 6);
        assert!(!a.is_empty());
        assert!(a.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn ref_from_ref() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let a = MatrixDataRef::<2, 3, f32>::from(a_buf.as_ref());

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 3.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);

        assert_eq!(a.len(), 6);
        assert_eq!(a.buffer_len(), 6);
        assert!(!a.is_empty());
        assert!(a.is_valid());
    }

    #[test]
    fn data_into_array() {
        let value: MatrixDataArray<4, 1, 4, f32> = [0.0, 1.0, 3.0, 4.0].into();
        let data: [f32; 4] = value.into();
        assert_eq!(data, [0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    #[rustfmt::skip]
    fn ref_from_mut() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let a = MatrixDataRef::<2, 3, f32>::from(a_buf.as_mut_slice());

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 3.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);

        assert_eq!(a.len(), 6);
        assert_eq!(a.buffer_len(), 6);
        assert!(!a.is_empty());
        assert!(a.is_valid());
    }

    #[test]
    #[rustfmt::skip]
    fn mut_from_mut() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut a = MatrixDataMut::<2, 3, f32>::from(a_buf.as_mut_slice());
        a[2] += 10.0;

        assert_f32_near!(a[0], 1.);
        assert_f32_near!(a[1], 2.);
        assert_f32_near!(a[2], 13.);
        assert_f32_near!(a[3], 4.);
        assert_f32_near!(a[4], 5.);
        assert_f32_near!(a[5], 6.);

        assert_eq!(a.len(), 6);
        assert_eq!(a.buffer_len(), 6);
        assert!(!a.is_empty());
        assert!(a.is_valid());
    }

    #[test]
    fn row_vector() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut a = MatrixDataMut::<3, 1, f32>::from(a_buf.as_mut_slice());
        assert_eq!(a.get_row(0), 1.0);
        assert_eq!(a.get_row(1), 2.0);
        assert_eq!(a.get_row(2), 3.0);

        a.set_row(0, 0.0);
        assert_eq!(a.get_row(0), 0.0);
    }

    #[test]
    fn column_vector() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut a = MatrixDataMut::<1, 3, f32>::from(a_buf.as_mut_slice());
        assert_eq!(a.get_col(0), 1.0);
        assert_eq!(a.get_col(1), 2.0);
        assert_eq!(a.get_col(2), 3.0);

        a.set_col(0, 0.0);
        assert_eq!(a.get_col(0), 0.0);
    }

    #[test]
    fn scalar() {
        let mut a_buf = [1.0, 2.0, 3.0];
        let mut a = MatrixDataMut::<1, 1, f32>::from(a_buf.as_mut_slice());
        assert_eq!(a.get_value(), 1.0);

        a.set_value(0.0);
        assert_eq!(a.get_value(), 0.0);
    }
}
