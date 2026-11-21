//! Opt-in interoperability with [nalgebra](https://nalgebra.rs), enabled by the `nalgebra` feature.
//!
//! minikalman stores matrices row-major and nalgebra stores its own matrices column-major. The
//! views returned by [`AsNalgebra::as_nalgebra`] and [`AsNalgebraMut::as_nalgebra_mut`] borrow
//! minikalman's data with row-major strides, so `view[(row, col)]` is the same element as
//! [`Matrix::get(row, col)`](crate::matrix::Matrix::get). The copies ([`AsNalgebra::to_nalgebra`]
//! and [`AsNalgebraMut::copy_from_nalgebra`]) go through these views, which keeps the layout
//! mapping in one place. Neither views nor copies allocate.
//!
//! Do arithmetic on the copies, not the views: see [`AsNalgebra`] for the nalgebra bug this avoids.

use nalgebra::{Const, MatrixView, MatrixViewMut, RawStorage, SMatrix};

use crate::matrix::row_major::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// A read-only nalgebra view of a `ROWS` × `COLS` row-major minikalman matrix.
///
/// The row stride is `COLS` and the column stride is `1`.
pub type RowMajorView<'a, const ROWS: usize, const COLS: usize, T> =
    MatrixView<'a, T, Const<ROWS>, Const<COLS>, Const<COLS>, Const<1>>;

/// A mutable nalgebra view of a `ROWS` × `COLS` row-major minikalman matrix.
///
/// The row stride is `COLS` and the column stride is `1`.
pub type RowMajorViewMut<'a, const ROWS: usize, const COLS: usize, T> =
    MatrixViewMut<'a, T, Const<ROWS>, Const<COLS>, Const<COLS>, Const<1>>;

/// Borrows or copies a minikalman matrix or buffer as a nalgebra matrix.
///
/// Implemented for every type that implements [`RowMajorSequentialData`], i.e. all matrix data
/// types and all filter buffers. Needs only the `nalgebra` feature, not `unsafe`.
///
/// # Arithmetic
///
/// Use the views to read, write and copy elements, and do arithmetic on a
/// [`to_nalgebra`](Self::to_nalgebra) copy. Up to at least nalgebra 0.35, a product whose left
/// operand has a row stride other than 1, such as these views, reads and writes out of bounds
/// ([dimforge/nalgebra#1616](https://github.com/dimforge/nalgebra/issues/1616)).
///
/// # Examples
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::nalgebra::SMatrix;
///
/// // A 2×3 matrix, stored row by row.
/// let a = MatrixData::new_array::<2, 3, 6, f32>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///
/// let view = a.as_nalgebra();
/// assert_eq!(view[(1, 0)], 4.0);
///
/// // `SMatrix::new` takes its arguments in row order, too.
/// let expected = SMatrix::<f32, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
/// assert_eq!(a.to_nalgebra(), expected);
///
/// // Do arithmetic on the owned copy, not on the view (see "Arithmetic" above).
/// let gram = a.to_nalgebra() * expected.transpose();
/// assert_eq!(gram[(0, 1)], 32.0);
/// ```
pub trait AsNalgebra<const ROWS: usize, const COLS: usize, T> {
    /// Borrows the data as a nalgebra matrix view, without copying.
    ///
    /// # Panics
    ///
    /// Panics if the underlying buffer holds fewer than `ROWS × COLS` elements.
    fn as_nalgebra(&self) -> RowMajorView<'_, ROWS, COLS, T>;

    /// Copies the data into a stack-allocated nalgebra matrix.
    ///
    /// The result uses nalgebra's column-major layout and holds the same values at the same
    /// `(row, column)` indices.
    ///
    /// # Panics
    ///
    /// Panics if the underlying buffer holds fewer than `ROWS × COLS` elements.
    fn to_nalgebra(&self) -> SMatrix<T, ROWS, COLS>;
}

/// Mutably borrows a minikalman matrix or buffer as a nalgebra matrix, or fills it from one.
///
/// Implemented for every type that implements [`RowMajorSequentialDataMut`]. Needs only the
/// `nalgebra` feature, not `unsafe`.
///
/// # Examples
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::nalgebra::SMatrix;
///
/// let mut a = MatrixData::new_array::<2, 3, 6, f32>([0.0; 6]);
///
/// // Writes through the view land in the row-major buffer.
/// a.as_nalgebra_mut()[(1, 0)] = 4.0;
/// assert_eq!(a.as_slice()[3], 4.0);
///
/// a.copy_from_nalgebra(&SMatrix::<f32, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
/// assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub trait AsNalgebraMut<const ROWS: usize, const COLS: usize, T>:
    AsNalgebra<ROWS, COLS, T>
{
    /// Mutably borrows the data as a nalgebra matrix view, without copying.
    ///
    /// # Panics
    ///
    /// Panics if the underlying buffer holds fewer than `ROWS × COLS` elements.
    fn as_nalgebra_mut(&mut self) -> RowMajorViewMut<'_, ROWS, COLS, T>;

    /// Overwrites the data with the values of a nalgebra matrix of the same shape.
    ///
    /// Accepts any nalgebra storage, e.g. an [`SMatrix`] or a view.
    ///
    /// # Panics
    ///
    /// Panics if the underlying buffer holds fewer than `ROWS × COLS` elements.
    fn copy_from_nalgebra<S>(&mut self, source: &nalgebra::Matrix<T, Const<ROWS>, Const<COLS>, S>)
    where
        S: RawStorage<T, Const<ROWS>, Const<COLS>>;
}

impl<const ROWS: usize, const COLS: usize, T, D> AsNalgebra<ROWS, COLS, T> for D
where
    T: nalgebra::Scalar,
    D: RowMajorSequentialData<ROWS, COLS, T> + ?Sized,
{
    #[inline]
    fn as_nalgebra(&self) -> RowMajorView<'_, ROWS, COLS, T> {
        // nalgebra checks the slice length and panics if it is too short.
        RowMajorView::from_slice_with_strides_generic(self.as_slice(), Const, Const, Const, Const)
    }

    #[inline]
    fn to_nalgebra(&self) -> SMatrix<T, ROWS, COLS> {
        self.as_nalgebra().clone_owned()
    }
}

impl<const ROWS: usize, const COLS: usize, T, D> AsNalgebraMut<ROWS, COLS, T> for D
where
    T: nalgebra::Scalar,
    D: RowMajorSequentialDataMut<ROWS, COLS, T> + ?Sized,
{
    #[inline]
    fn as_nalgebra_mut(&mut self) -> RowMajorViewMut<'_, ROWS, COLS, T> {
        // nalgebra checks the slice length and panics if it is too short.
        RowMajorViewMut::from_slice_with_strides_generic(
            self.as_mut_slice(),
            Const,
            Const,
            Const,
            Const,
        )
    }

    #[inline]
    fn copy_from_nalgebra<S>(&mut self, source: &nalgebra::Matrix<T, Const<ROWS>, Const<COLS>, S>)
    where
        S: RawStorage<T, Const<ROWS>, Const<COLS>>,
    {
        self.as_nalgebra_mut().copy_from(source);
    }
}

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(feature = "unsafe")]
unsafe impl<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> nalgebra::IsContiguous
    for crate::matrix::MatrixDataArray<ROWS, COLS, TOTAL, T>
{
}

#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "nalgebra", feature = "unsafe", feature = "alloc")))
)]
#[cfg(all(feature = "unsafe", feature = "alloc"))]
unsafe impl<const ROWS: usize, const COLS: usize, T> nalgebra::IsContiguous
    for crate::matrix::MatrixDataBoxed<ROWS, COLS, T>
{
}

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(feature = "unsafe")]
unsafe impl<'a, const ROWS: usize, const COLS: usize, T> nalgebra::IsContiguous
    for crate::matrix::MatrixDataRef<'a, ROWS, COLS, T>
{
}

#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(feature = "unsafe")]
unsafe impl<'a, const ROWS: usize, const COLS: usize, T> nalgebra::IsContiguous
    for crate::matrix::MatrixDataMut<'a, ROWS, COLS, T>
{
}

/// Lets a [`MatrixDataArray`](crate::matrix::MatrixDataArray) back an owned nalgebra matrix
/// via `nalgebra::Matrix::from_data`, with row-major strides.
///
/// Prefer [`AsNalgebra::as_nalgebra`], which needs no `unsafe`. Note that `as_slice()` on such a
/// matrix returns the elements in row-major order, unlike nalgebra's own matrices.
/// Its row stride is `COLS`, so the arithmetic caveat on [`AsNalgebra`] applies here, too.
#[cfg_attr(docsrs, doc(cfg(all(feature = "nalgebra", feature = "unsafe"))))]
#[cfg(feature = "unsafe")]
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
        // Checked: a wrapping ROWS × COLS could otherwise pass for huge const generics.
        assert!(
            ROWS.checked_mul(COLS).is_some_and(|n| n <= TOTAL),
            "buffer smaller than ROWS × COLS"
        );
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
        // SAFETY: `ptr` asserts that the array holds at least ROWS × COLS elements.
        core::slice::from_raw_parts(self.ptr(), ROWS * COLS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::types::*;
    use crate::matrix::{Matrix, MatrixData};
    use nalgebra::SMatrix;

    /// The 2×3 matrix `[[1, 2, 3], [4, 5, 6]]`, built by nalgebra from row-order arguments.
    fn expected_2x3() -> SMatrix<f32, 2, 3> {
        SMatrix::<f32, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    }

    #[test]
    fn view_matches_minikalman_indexing() {
        let a = MatrixData::new_array::<2, 3, 6, f32>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let view = a.as_nalgebra();

        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(view[(r, c)], a.get(r, c));
            }
        }

        assert_eq!(view, expected_2x3());
        assert_eq!(a.to_nalgebra(), expected_2x3());
    }

    #[test]
    fn view_writes_reach_the_row_major_buffer() {
        let mut a = MatrixData::new_array::<2, 3, 6, f32>([0.0; 6]);
        a.as_nalgebra_mut()[(1, 0)] = 7.0;
        assert_eq!(a.as_slice()[3], 7.0);

        a.as_mut_slice()[2] = 9.0;
        assert_eq!(a.as_nalgebra()[(0, 2)], 9.0);
    }

    #[test]
    fn copy_from_nalgebra_keeps_indices() {
        let mut a = MatrixData::new_array::<2, 3, 6, f32>([0.0; 6]);
        a.copy_from_nalgebra(&expected_2x3());
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // A transposed view (non-contiguous storage) is accepted as well.
        let mut b = MatrixData::new_array::<3, 2, 6, f32>([0.0; 6]);
        b.copy_from_nalgebra(&a.as_nalgebra().transpose());
        assert_eq!(b.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(b.to_nalgebra(), expected_2x3().transpose());
    }

    #[test]
    fn all_matrix_data_kinds_have_views() {
        let mut raw = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = crate::matrix::MatrixDataRef::<2, 3, f32>::from(raw.as_slice());
        assert_eq!(r.to_nalgebra(), expected_2x3());

        let mut m = MatrixData::new_mut::<2, 3, f32>(&mut raw);
        m.as_nalgebra_mut()[(0, 0)] = 10.0;
        assert_eq!(m.to_nalgebra()[(0, 0)], 10.0);
        assert_eq!(raw[0], 10.0);

        #[cfg(feature = "alloc")]
        {
            let boxed = MatrixData::new_boxed::<2, 3, f32, _>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            assert_eq!(boxed.to_nalgebra(), expected_2x3());
        }
    }

    #[test]
    fn buffers_have_views() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Non-square buffers: observation (OBS × STATES), gain (STATES × OBS), control (STATES × CONTROLS).
        let mut h = ObservationMatrixMutBuffer::<2, 3, f32, _>::from(data);
        assert_eq!(h.to_nalgebra(), expected_2x3());
        h.as_nalgebra_mut()[(1, 2)] = 60.0;
        assert_eq!(h.as_slice()[5], 60.0);

        let k = KalmanGainMatrixBuffer::<3, 2, f32, _>::from(data);
        assert_eq!(
            k.to_nalgebra(),
            SMatrix::<f32, 3, 2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        );

        let b = ControlMatrixMutBuffer::<3, 2, f32, _>::from(data);
        assert_eq!(b.as_nalgebra()[(2, 1)], 6.0);

        // Vectors.
        let mut x = StateVectorBuffer::<3, f32, _>::from([1.0, 2.0, 3.0]);
        x.copy_from_nalgebra(&nalgebra::Vector3::new(4.0, 5.0, 6.0));
        assert_eq!(x.as_slice(), &[4.0, 5.0, 6.0]);

        // Square covariance.
        let p = EstimateCovarianceMatrixBuffer::<2, f32, _>::from([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p.as_nalgebra()[(1, 0)], 3.0);
    }

    #[test]
    fn filter_state_can_be_read_through_nalgebra() {
        use crate::test_dummies::make_dummy_filter;

        let filter = make_dummy_filter();
        let x = filter.state_vector().to_nalgebra();
        assert_eq!(x.nrows(), 3);
    }

    #[test]
    #[should_panic(expected = "input data buffer too small")]
    fn undersized_buffer_panics_in_view() {
        // Two values cannot back a 3×3 matrix; the view must not read past them.
        let a = MatrixData::new_array::<3, 3, 2, f32>([1.0, 2.0]);
        let _ = a.as_nalgebra();
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn fixed_point_views_and_copies() {
        use fixed::types::I16F16;

        let v = |x: f32| I16F16::from_num(x);
        let mut a = MatrixData::new_array::<2, 3, 6, I16F16>([I16F16::ZERO; 6]);
        let source = SMatrix::<I16F16, 2, 3>::new(v(1.0), v(2.0), v(3.0), v(4.0), v(5.0), v(6.0));

        a.copy_from_nalgebra(&source);
        assert_eq!(a.get(1, 0), v(4.0));
        assert_eq!(a.to_nalgebra(), source);
        assert_eq!(a.as_nalgebra().transpose()[(2, 1)], v(6.0));

        // nalgebra's generic arithmetic works on `fixed` types, too (on owned copies: see
        // dimforge/nalgebra#1616 for strided left operands).
        let product = a.to_nalgebra() * source.transpose();
        assert_eq!(product[(0, 1)], v(32.0));
    }

    #[cfg(feature = "unsafe")]
    #[test]
    fn owned_storage_matches_view() {
        let a = MatrixData::new_array::<2, 3, 6, f32>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mat = nalgebra::Matrix::from_data(a);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        assert_eq!(mat, expected_2x3());
    }

    #[cfg(feature = "unsafe")]
    #[test]
    #[should_panic(expected = "buffer smaller than ROWS × COLS")]
    fn overflowing_dimensions_are_rejected() {
        // (2⁶³) × 2 wraps to 0 in release builds, which would pass an unchecked `<= TOTAL`.
        let a = MatrixData::new_array::<{ 1 << (usize::BITS - 1) }, 2, 0, f32>([]);
        let mat = nalgebra::Matrix::from_data(a);
        let _ = mat.as_ptr();
    }

    #[cfg(feature = "unsafe")]
    #[test]
    #[should_panic(expected = "buffer smaller than ROWS × COLS")]
    fn undersized_buffer_is_rejected() {
        // Two values cannot back a 3×3 matrix; nalgebra would read past the array.
        let a = MatrixData::new_array::<3, 3, 2, f32>([1.0, 2.0]);
        let mat = nalgebra::Matrix::from_data(a);
        let _ = mat.sum();
    }
}
