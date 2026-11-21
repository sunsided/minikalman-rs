//! Opt-in interoperability with [nalgebra](https://nalgebra.rs), enabled by the `nalgebra` feature.
//!
//! minikalman stores matrices row-major and nalgebra stores its own matrices column-major.
//! [`ToNalgebra::to_nalgebra`] and [`CopyFromNalgebra::copy_from_nalgebra`] copy between the two
//! on the stack and keep every value at its `(row, column)` index. Internally they go through a
//! strided view of minikalman's data, which keeps the layout mapping in one place.
//!
//! # Why there are no public zero-copy views
//!
//! Up to at least nalgebra 0.35, nalgebra's `axpy`-based kernels (matrix products, `gemv`, `axpy`)
//! read and write out of bounds when a vector or a left-hand matrix operand has a row stride
//! other than 1 ([dimforge/nalgebra#1616](https://github.com/dimforge/nalgebra/issues/1616), fix
//! in review in [dimforge/nalgebra#1627](https://github.com/dimforge/nalgebra/pull/1627)). A
//! view of a row-major matrix has row stride `COLS`, so safe code like `view * x` would reach
//! undefined behavior. The views stay crate-internal, where only the element-wise copies use
//! them. They can be made public again once a fixed nalgebra release fits the MSRV
//! ([#138](https://github.com/sunsided/minikalman-rs/issues/138)).

use nalgebra::{Const, MatrixView, MatrixViewMut, RawStorage, SMatrix};

use crate::matrix::row_major::{RowMajorSequentialData, RowMajorSequentialDataMut};

/// A read-only nalgebra view of a `ROWS` × `COLS` row-major matrix: row stride `COLS`,
/// column stride `1`.
///
/// Crate-internal; see the module docs for why.
type RowMajorView<'a, const ROWS: usize, const COLS: usize, T> =
    MatrixView<'a, T, Const<ROWS>, Const<COLS>, Const<COLS>, Const<1>>;

/// A mutable nalgebra view of a `ROWS` × `COLS` row-major matrix: row stride `COLS`,
/// column stride `1`.
///
/// Crate-internal; see the module docs for why.
type RowMajorViewMut<'a, const ROWS: usize, const COLS: usize, T> =
    MatrixViewMut<'a, T, Const<ROWS>, Const<COLS>, Const<COLS>, Const<1>>;

/// Panics unless `len` elements can hold a `ROWS` × `COLS` matrix.
///
/// nalgebra's own length check computes `ROWS × COLS` unchecked, so for huge const generics it
/// can wrap in release builds and pass for a short slice.
#[inline]
fn assert_fits<const ROWS: usize, const COLS: usize>(len: usize) {
    assert!(
        ROWS.checked_mul(COLS).is_some_and(|n| n <= len),
        "buffer smaller than ROWS × COLS"
    );
}

/// Borrows row-major `data` as a nalgebra view, without copying.
///
/// Never hand the result to nalgebra arithmetic; see the module docs.
///
/// # Panics
///
/// Panics if `data` holds fewer than `ROWS × COLS` elements.
#[inline]
fn row_major_view<const ROWS: usize, const COLS: usize, T: nalgebra::Scalar>(
    data: &[T],
) -> RowMajorView<'_, ROWS, COLS, T> {
    assert_fits::<ROWS, COLS>(data.len());
    RowMajorView::from_slice_with_strides_generic(data, Const, Const, Const, Const)
}

/// Mutably borrows row-major `data` as a nalgebra view, without copying.
///
/// Never hand the result to nalgebra arithmetic; see the module docs.
///
/// # Panics
///
/// Panics if `data` holds fewer than `ROWS × COLS` elements.
#[inline]
fn row_major_view_mut<const ROWS: usize, const COLS: usize, T: nalgebra::Scalar>(
    data: &mut [T],
) -> RowMajorViewMut<'_, ROWS, COLS, T> {
    assert_fits::<ROWS, COLS>(data.len());
    RowMajorViewMut::from_slice_with_strides_generic(data, Const, Const, Const, Const)
}

/// Copies a minikalman matrix or buffer into a stack-allocated nalgebra matrix.
///
/// Implemented for every type that implements [`RowMajorSequentialData`], i.e. all matrix data
/// types and all filter buffers, read-only ones included. Needs only the `nalgebra` feature, not
/// `unsafe`, and never allocates.
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
/// // `SMatrix::new` takes its arguments in row order, too.
/// let expected = SMatrix::<f32, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
/// let copy = a.to_nalgebra();
/// assert_eq!(copy, expected);
/// assert_eq!(copy[(1, 0)], 4.0);
///
/// // The copy is an ordinary nalgebra matrix.
/// let gram = copy * expected.transpose();
/// assert_eq!(gram[(0, 1)], 32.0);
/// ```
pub trait ToNalgebra<const ROWS: usize, const COLS: usize, T> {
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

/// Fills a minikalman matrix or buffer from a nalgebra matrix.
///
/// Implemented for every type that implements [`RowMajorSequentialDataMut`], i.e. the mutable
/// matrix data types and buffers. Needs only the `nalgebra` feature, not `unsafe`.
///
/// # Examples
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::nalgebra::SMatrix;
///
/// let mut a = MatrixData::new_array::<2, 3, 6, f32>([0.0; 6]);
///
/// a.copy_from_nalgebra(&SMatrix::<f32, 2, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
/// assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub trait CopyFromNalgebra<const ROWS: usize, const COLS: usize, T> {
    /// Overwrites the data with the values of a nalgebra matrix of the same shape.
    ///
    /// Accepts any nalgebra storage, e.g. an [`SMatrix`] or a nalgebra view.
    ///
    /// # Panics
    ///
    /// Panics if the underlying buffer holds fewer than `ROWS × COLS` elements.
    fn copy_from_nalgebra<S>(&mut self, source: &nalgebra::Matrix<T, Const<ROWS>, Const<COLS>, S>)
    where
        S: RawStorage<T, Const<ROWS>, Const<COLS>>;
}

impl<const ROWS: usize, const COLS: usize, T, D> ToNalgebra<ROWS, COLS, T> for D
where
    T: nalgebra::Scalar,
    D: RowMajorSequentialData<ROWS, COLS, T> + ?Sized,
{
    #[inline]
    fn to_nalgebra(&self) -> SMatrix<T, ROWS, COLS> {
        // `clone_owned` copies element by element; it does not use the affected kernels.
        row_major_view::<ROWS, COLS, T>(self.as_slice()).clone_owned()
    }
}

impl<const ROWS: usize, const COLS: usize, T, D> CopyFromNalgebra<ROWS, COLS, T> for D
where
    T: nalgebra::Scalar,
    D: RowMajorSequentialDataMut<ROWS, COLS, T> + ?Sized,
{
    #[inline]
    fn copy_from_nalgebra<S>(&mut self, source: &nalgebra::Matrix<T, Const<ROWS>, Const<COLS>, S>)
    where
        S: RawStorage<T, Const<ROWS>, Const<COLS>>,
    {
        // `copy_from` copies element by element; it does not use the affected kernels.
        row_major_view_mut::<ROWS, COLS, T>(self.as_mut_slice()).copy_from(source);
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
/// `as_slice()` on such a matrix returns the elements in row-major order, unlike nalgebra's own
/// matrices.
///
/// # Warning: no arithmetic
///
/// The type implements only `RawStorage`, not nalgebra's `Storage`, so the matrix itself cannot
/// be multiplied. Views taken from it can, though (`m.column(0)`, `m.fixed_view::<R, C>(0, 0)`),
/// and they keep the row stride `COLS`. They hit
/// [dimforge/nalgebra#1616](https://github.com/dimforge/nalgebra/issues/1616): a product with
/// such a view on the left, `gemv` or `axpy` read and write out of bounds, which is undefined
/// behavior in release builds (with `f32` too). Use the matrix for element access only, or call
/// [`ToNalgebra::to_nalgebra`] and compute on the copy. See
/// [#138](https://github.com/sunsided/minikalman-rs/issues/138).
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
        assert_fits::<ROWS, COLS>(TOTAL);
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
    use crate::matrix::{Matrix, MatrixData, MatrixDataArray};
    use nalgebra::SMatrix;

    /// Gives tests the crate-internal views as methods.
    trait ViewExt<const R: usize, const C: usize, T: nalgebra::Scalar> {
        /// See [`row_major_view`].
        fn as_nalgebra(&self) -> RowMajorView<'_, R, C, T>;

        /// See [`row_major_view_mut`].
        fn as_nalgebra_mut(&mut self) -> RowMajorViewMut<'_, R, C, T>;
    }

    impl<const R: usize, const C: usize, T, D> ViewExt<R, C, T> for D
    where
        T: nalgebra::Scalar,
        D: RowMajorSequentialDataMut<R, C, T> + ?Sized,
    {
        fn as_nalgebra(&self) -> RowMajorView<'_, R, C, T> {
            row_major_view(self.as_slice())
        }

        fn as_nalgebra_mut(&mut self) -> RowMajorViewMut<'_, R, C, T> {
            row_major_view_mut(self.as_mut_slice())
        }
    }

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

    /// Compiles only if `D` can be copied into nalgebra.
    fn has_view<const R: usize, const C: usize, D: ToNalgebra<R, C, f32>>() {}

    /// Compiles only if `D` can be copied into nalgebra and filled from nalgebra.
    fn has_view_mut<const R: usize, const C: usize, D>()
    where
        D: ToNalgebra<R, C, f32> + CopyFromNalgebra<R, C, f32>,
    {
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn every_buffer_type_has_views() {
        /// Storage for a 2×3 matrix.
        type M23 = MatrixDataArray<2, 3, 6, f32>;
        /// Storage for a 3×2 matrix.
        type M32 = MatrixDataArray<3, 2, 6, f32>;
        /// Storage for a 2×2 matrix.
        type M22 = MatrixDataArray<2, 2, 4, f32>;
        /// Storage for a 2-vector.
        type V2 = MatrixDataArray<2, 1, 2, f32>;

        has_view_mut::<2, 1, ControlVectorBuffer<2, f32, V2>>();
        has_view::<3, 2, ControlMatrixBuffer<3, 2, f32, M32>>();
        has_view_mut::<3, 2, ControlMatrixMutBuffer<3, 2, f32, M32>>();
        has_view::<2, 2, ControlProcessNoiseCovarianceMatrixBuffer<2, f32, M22>>();
        has_view_mut::<2, 2, ControlProcessNoiseCovarianceMatrixMutBuffer<2, f32, M22>>();
        has_view_mut::<3, 2, CrossCovarianceMatrixBuffer<3, 2, f32, M32>>();
        has_view_mut::<2, 1, InnovationVectorBuffer<2, f32, V2>>();
        has_view_mut::<2, 3, SigmaPropagatedMatrixBuffer<2, 3, f32, M23>>();
        has_view_mut::<2, 2, InnovationCovarianceMatrixBuffer<2, f32, M22>>();
        has_view_mut::<2, 1, SigmaWeightsVectorBuffer<2, f32, V2>>();
        has_view_mut::<2, 3, SigmaPointMatrixBuffer<2, 3, f32, M23>>();
        has_view_mut::<2, 3, SigmaObservedMatrixBuffer<2, 3, f32, M23>>();
        has_view_mut::<2, 1, StateVectorBuffer<2, f32, V2>>();
        has_view_mut::<2, 2, TempSigmaPMatrixBuffer<2, f32, M22>>();
        has_view_mut::<3, 2, TemporaryBQMatrixBuffer<3, 2, f32, M32>>();
        has_view_mut::<2, 2, TemporaryStateMatrixBuffer<2, f32, M22>>();
        has_view_mut::<2, 2, EstimateCovarianceMatrixBuffer<2, f32, M22>>();
        has_view::<2, 2, DirectProcessNoiseCovarianceMatrixBuffer<2, f32, M22>>();
        has_view_mut::<2, 2, DirectProcessNoiseCovarianceMatrixMutBuffer<2, f32, M22>>();
        has_view_mut::<2, 1, PredictedStateEstimateVectorBuffer<2, f32, V2>>();
        has_view_mut::<2, 3, TemporaryHPMatrixBuffer<2, 3, f32, M23>>();
        has_view::<2, 2, StateTransitionMatrixBuffer<2, f32, M22>>();
        has_view_mut::<2, 2, StateTransitionMatrixMutBuffer<2, f32, M22>>();
        has_view_mut::<3, 2, KalmanGainMatrixBuffer<3, 2, f32, M32>>();
        has_view_mut::<2, 1, MeasurementVectorBuffer<2, f32, V2>>();
        has_view_mut::<2, 2, MeasurementNoiseCovarianceMatrixBuffer<2, f32, M22>>();
        has_view::<2, 3, ObservationMatrixBuffer<2, 3, f32, M23>>();
        has_view_mut::<2, 3, ObservationMatrixMutBuffer<2, 3, f32, M23>>();
        has_view_mut::<2, 2, TemporaryKHPMatrixBuffer<2, f32, M22>>();
        has_view_mut::<3, 2, TemporaryPHTMatrixBuffer<3, 2, f32, M32>>();
        has_view_mut::<2, 2, TemporaryResidualCovarianceInvertedMatrixBuffer<2, f32, M22>>();
    }

    #[test]
    fn filter_state_can_be_read_through_nalgebra() {
        use crate::test_dummies::make_dummy_filter;

        let mut filter = make_dummy_filter();
        filter.state_vector_mut().as_nalgebra_mut()[(2, 0)] = 5.0;
        assert_eq!(filter.state_vector().as_slice()[2], 5.0);
        assert_eq!(filter.state_vector().to_nalgebra()[(2, 0)], 5.0);
    }

    #[test]
    #[should_panic(expected = "buffer smaller than ROWS × COLS")]
    fn undersized_buffer_panics_in_view() {
        // Two values cannot back a 3×3 matrix; the view must not read past them.
        let a = MatrixData::new_array::<3, 3, 2, f32>([1.0, 2.0]);
        let _ = a.to_nalgebra();
    }

    #[test]
    #[should_panic(expected = "buffer smaller than ROWS × COLS")]
    fn overflowing_dimensions_are_rejected_by_views() {
        // (2⁶³) × 2 wraps to 0 in release builds, which would pass nalgebra's own length check.
        let _ = row_major_view::<{ 1 << (usize::BITS - 1) }, 2, f32>(&[]);
    }

    #[test]
    #[should_panic(expected = "buffer smaller than ROWS × COLS")]
    fn overflowing_dimensions_are_rejected_by_mutable_views() {
        let _ = row_major_view_mut::<{ 1 << (usize::BITS - 1) }, 2, f32>(&mut []);
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
