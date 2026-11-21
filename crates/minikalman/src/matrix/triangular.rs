//! Householder triangularization and triangular solves on row-major slices.
//!
//! These are the slice-level cores behind
//! [`MatrixMut::triangularize_upper`](crate::matrix::MatrixMut::triangularize_upper) and the
//! `solve_*_triangular_inplace` methods of [`SquareMatrix`](crate::matrix::SquareMatrix). They
//! are shared with code that works on raw slices, such as the predictively oriented correction.

use crate::matrix::MatrixDataType;

/// Which triangle of a square matrix holds the factor.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Triangle {
    /// The factor `L` is stored on and below the diagonal.
    Lower,
    /// The factor `U` is stored on and above the diagonal.
    Upper,
}

/// Whether a solve uses the factor itself or its transpose.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transpose {
    /// Solve `T×x = b`.
    No,
    /// Solve `Tᵀ×x = b`.
    Yes,
}

/// Replaces the row-major `n×cols` matrix `b` by `T⁻¹×b` (or `T⁻ᵀ×b`) for the triangular
/// row-major `n×n` factor `T` stored in `t`.
///
/// Only the triangle named by `triangle` is read; the other one is ignored. Returns `false`
/// and leaves `b` untouched if a diagonal entry of `T` is zero, which would divide by zero
/// (and panic for fixed-point types).
pub(crate) fn solve_triangular<T>(
    t: &[T],
    n: usize,
    triangle: Triangle,
    transpose: Transpose,
    b: &mut [T],
    cols: usize,
) -> bool
where
    T: MatrixDataType,
{
    if (0..n).any(|i| t[i * n + i] == T::zero()) {
        return false;
    }

    // Element (i, k) of the effective factor, i.e. of `T` or `Tᵀ`.
    let at = |i: usize, k: usize| match transpose {
        Transpose::No => t[i * n + k],
        Transpose::Yes => t[k * n + i],
    };

    // `L` and `Uᵀ` are lower triangular and solved forwards; `U` and `Lᵀ` backwards.
    let forward = (triangle == Triangle::Lower) == (transpose == Transpose::No);
    for col in 0..cols {
        for step in 0..n {
            let i = if forward { step } else { n - 1 - step };
            let known = if forward { 0..i } else { (i + 1)..n };
            let sum = known.map(|k| at(i, k) * b[k * cols + col]).sum::<T>();
            b[i * cols + col] = (b[i * cols + col] - sum) / at(i, i);
        }
    }

    true
}

/// Triangularizes the row-major `rows×cols` matrix `m` in place with Householder reflections.
///
/// Afterwards the first `cols` rows hold the upper triangular `R` with `Mᵀ×M = Rᵀ×R` and a
/// non-negative diagonal, and all rows below are zero. `Q` is never formed.
///
/// Each reflector is computed from the column divided by its largest magnitude, as BLAS `nrm2`
/// does, so no entry is squared unscaled. With fixed-point types, the column norms times
/// `√(2·rows + 2·√rows)` (the bound of `ṽᵀ×column`) must fit the type. A zero column is skipped, which
/// leaves a zero on the diagonal of `R`, i.e. a valid singular factor.
///
/// Returns `false` if `rows < cols`, or if an entry or a diagonal of `R` is not finite (`NaN`
/// or infinite, e.g. after an overflow). `m` is then partially modified.
pub(crate) fn triangularize_upper<T>(m: &mut [T], rows: usize, cols: usize) -> bool
where
    T: MatrixDataType,
{
    if rows < cols {
        return false;
    }

    for j in 0..cols {
        // Entries above the diagonal of column `j` are final since step `j - 1`.
        if !(0..rows).all(|i| is_finite(m[i * cols + j])) {
            return false;
        }

        let Some(diagonal) = reflect_column(m, rows, cols, j) else {
            continue;
        };

        if !is_finite(diagonal) {
            return false;
        }

        m[j * cols + j] = diagonal;
        for i in (j + 1)..rows {
            m[i * cols + j] = T::zero();
        }
    }

    make_diagonal_non_negative(m, cols);
    true
}

/// Applies the Householder reflection that zeroes column `j` below the diagonal to the columns
/// `j + 1..cols`, and returns the new diagonal entry `α = −sign(a₀)·‖a‖`.
///
/// Returns `None` without changing `m` if the column is zero from row `j` on. Column `j` itself
/// is used as scratch for the scaled reflector `v / max|aᵢ|` and must be overwritten by the
/// caller.
fn reflect_column<T>(m: &mut [T], rows: usize, cols: usize, j: usize) -> Option<T>
where
    T: MatrixDataType,
{
    let scale = (j..rows).map(|i| abs(m[i * cols + j])).fold(T::zero(), max);
    if scale == T::zero() {
        return None;
    }

    // ã = a / scale, so |ãᵢ| ≤ 1 and ‖ã‖² ≤ rows cannot overflow. Divided rather than
    // multiplied by `1/scale`, whose rounding in fixed point is a large relative error.
    for i in j..rows {
        m[i * cols + j] = m[i * cols + j] / scale;
    }

    let norm = (j..rows)
        .map(|i| m[i * cols + j] * m[i * cols + j])
        .sum::<T>()
        .square_root();

    // ṽ = ã − α̃·e₀ with α̃ = −sign(ã₀)·‖ã‖, so ṽ₀ = ã₀ + sign(ã₀)·‖ã‖ never cancels.
    let head = m[j * cols + j];
    let alpha = if head < T::zero() { norm } else { -norm };
    m[j * cols + j] = head - alpha;

    // β̃ = 2 / (ṽᵀ×ṽ) = 1 / (‖ã‖·(‖ã‖ + |ã₀|)), with β̃×ṽ×ṽᵀ = β×v×vᵀ. Divided by the
    // denominator (up to about `rows`) rather than multiplied by β̃, which rounds in fixed point.
    let denominator = norm * (norm + abs(head));
    for k in (j + 1)..cols {
        let projection = (j..rows)
            .map(|i| m[i * cols + j] * m[i * cols + k])
            .sum::<T>()
            / denominator;
        for i in j..rows {
            m[i * cols + k] -= projection * m[i * cols + j];
        }
    }

    Some(alpha * scale)
}

/// Negates every row of the upper triangular `cols×cols` block of `m` that has a negative
/// diagonal entry, which keeps `Rᵀ×R` and makes `R` unique.
fn make_diagonal_non_negative<T>(m: &mut [T], cols: usize)
where
    T: MatrixDataType,
{
    for i in 0..cols {
        if m[i * cols + i] < T::zero() {
            for k in i..cols {
                m[i * cols + k] = -m[i * cols + k];
            }
        }
    }
}

/// Returns `true` if `x` is neither `NaN` nor infinite; always `true` for fixed-point types.
#[allow(clippy::eq_op)]
fn is_finite<T>(x: T) -> bool
where
    T: MatrixDataType,
{
    // `x − x` is zero only for finite values: `NaN` and `±inf` both give `NaN`.
    x - x == T::zero()
}

/// Returns `|x|`.
fn abs<T>(x: T) -> T
where
    T: MatrixDataType,
{
    if x < T::zero() {
        -x
    } else {
        x
    }
}

/// Returns the larger of `a` and `b`.
fn max<T>(a: T, b: T) -> T
where
    T: MatrixDataType,
{
    if b > a {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    #[cfg(any(feature = "std", feature = "libm", feature = "fixed"))]
    use super::*;
    use crate::matrix::{Matrix, MatrixData, MatrixMut, RowMajorSequentialData, SquareMatrix};

    /// Returns `Mᵀ×M` of the row-major `rows×cols` matrix `m` in `f64`, with `N = cols²`.
    #[cfg(any(feature = "std", feature = "libm", feature = "fixed"))]
    fn gram<const N: usize>(m: &[f64], rows: usize, cols: usize) -> [f64; N] {
        let mut g = [0.0; N];
        for i in 0..cols {
            for j in 0..cols {
                g[i * cols + j] = (0..rows).map(|k| m[k * cols + i] * m[k * cols + j]).sum();
            }
        }

        g
    }

    /// Asserts that the first `cols` rows of `r` are upper triangular with a non-negative
    /// diagonal and that all rows below are zero.
    #[cfg(any(feature = "std", feature = "libm", feature = "fixed"))]
    fn assert_upper_triangular(r: &[f64], rows: usize, cols: usize) {
        for i in 0..rows {
            for j in 0..cols {
                if i > j {
                    assert_eq!(r[i * cols + j], 0.0, "entry ({i}, {j})");
                }
            }

            if i < cols {
                assert!(r[i * cols + i] >= 0.0, "diagonal {i}");
            }
        }
    }

    /// Asserts that `a` and `b` agree entry by entry to within `tolerance`.
    fn assert_close(a: &[f64], b: &[f64], tolerance: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert!((a - b).abs() <= tolerance, "entry {i}: {a} vs {b}");
        }
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[rustfmt::skip]
    fn stacked_factors_give_the_sum_of_their_products() {
        // M = [L₁ᵀ; L₂ᵀ], so Mᵀ×M = L₁×L₁ᵀ + L₂×L₂ᵀ, the square-root predict prearray.
        let l1 = [
            2.0, 0.0, 0.0,
            0.5, 1.5, 0.0,
            -1.0, 0.3, 0.7];
        let l2 = [
            0.1, 0.0, 0.0,
            0.0, 0.2, 0.0,
            0.05, 0.0, 0.3];
        let mut m = [0.0f64; 18];
        for i in 0..3 {
            for j in 0..3 {
                m[i * 3 + j] = l1[j * 3 + i];
                m[(i + 3) * 3 + j] = l2[j * 3 + i];
            }
        }

        let mut expected = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                expected[i * 3 + j] = (0..3)
                    .map(|k| l1[i * 3 + k] * l1[j * 3 + k] + l2[i * 3 + k] * l2[j * 3 + k])
                    .sum();
            }
        }

        assert!(triangularize_upper(&mut m, 6, 3));
        assert_upper_triangular(&m, 6, 3);
        assert_close(&gram::<9>(&m, 6, 3), &expected, 1e-12);
    }

    #[test]
    fn rejects_wide_and_non_finite_matrices() {
        let mut wide = [1.0f32; 6];
        assert!(!MatrixData::new_mut::<2, 3, f32>(&mut wide).triangularize_upper());

        for value in [f32::NAN, f32::INFINITY] {
            let mut m = [1.0, 2.0, 3.0, value];
            assert!(!MatrixData::new_mut::<2, 2, f32>(&mut m).triangularize_upper());
        }

        // Finite entries whose norm overflows: the diagonal of R is infinite.
        let mut m = [f32::MAX, 0.0, f32::MAX, 0.0];
        assert!(!MatrixData::new_mut::<2, 2, f32>(&mut m).triangularize_upper());
    }

    /// `cholesky(Mᵀ×M)` squares the condition number; the QR does not. With `cond(M) = 10⁶`,
    /// `Mᵀ×M` is numerically singular in `f32` while the QR still recovers `R` closely.
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    #[rustfmt::skip]
    fn qr_beats_cholesky_on_ill_conditioned_matrices() {
        // U×diag(1, 10⁻³, 10⁻⁶)×Vᵀ for orthonormal U (4×3) and V (3×3), rounded to f32.
        let m32: [f32; 12] = [
            2.889_635_3e-1, 5.776_206e-4, 2.883_867_3e-1,
            1.929_456_3e-4, 3.844_046_4e-4, -1.929_456_3e-4,
            5.771_579e-1, -3.850_128e-4, 5.775_426e-1,
            2.887_714e-1, 1.924_050_5e-4, 2.885_788_7e-1,
        ];
        let mut reference = m32.map(f64::from);
        assert!(triangularize_upper(&mut reference, 4, 3));

        let mut qr = m32;
        assert!(triangularize_upper(&mut qr, 4, 3));
        let qr_error = max_error(&qr[..9], &reference[..9]);

        let mut normal = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                normal[i * 3 + j] = (0..4).map(|k| m32[k * 3 + i] * m32[k * 3 + j]).sum();
            }
        }

        // A failed Cholesky counts as worse than any QR result.
        let mut factor = MatrixData::new_mut::<3, 3, f32>(&mut normal);
        let cholesky_error = if factor.cholesky_decompose_lower() {
            let mut r = [0.0f32; 9];
            for i in 0..3 {
                for j in 0..3 {
                    r[i * 3 + j] = factor.get(j, i);
                }
            }
            max_error(&r, &reference[..9])
        } else {
            f64::INFINITY
        };

        assert!(qr_error < 1e-7, "QR error {qr_error}");
        assert!(qr_error * 1000.0 < cholesky_error, "QR {qr_error}, Cholesky {cholesky_error}");
    }

    /// Returns the largest absolute difference between `a` and the `f64` reference `b`.
    #[cfg(any(feature = "std", feature = "libm"))]
    fn max_error(a: &[f32], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max)
    }

    /// Entries around 1000 square to about 10⁶, beyond the `I16F16` range of `±32768`: only the
    /// scaled reflectors keep this from overflowing.
    #[test]
    #[cfg(feature = "fixed")]
    #[rustfmt::skip]
    fn fixed_point_handles_large_entries() {
        use fixed::types::I16F16;

        let m: [f64; 12] = [
            1000.0, -800.0, 250.0,
            -950.0, 600.0, 700.0,
            400.0, 1200.0, -300.0,
            -700.0, -450.0, 900.0,
        ];
        let mut fixed = m.map(I16F16::from_num);
        assert!(MatrixData::new_mut::<4, 3, I16F16>(&mut fixed).triangularize_upper());

        // R from numpy, with the row signs flipped to a non-negative diagonal.
        let expected = [
            1_597.654_530_87, -359.902_587_76, -729.193_938_67,
            0.0, 1_585.235_038_51, -509.349_570_18,
            0.0, 0.0, 813.227_652_73,
        ];
        let r = fixed.map(|x| x.to_num::<f64>());
        assert_close(&r[..9], &expected, 0.05);
        assert_eq!(&r[9..], &[0.0; 3]);
    }

    /// The worst case of the documented `I16F16` limit: with 12 rows, `ṽᵀ×column` reaches
    /// `√(2·12 + 2·√12) ≈ 5.56` times the column norm when the column is parallel to the
    /// reflector. A norm of 5000 gives about 27800, just inside the range of `±32768`.
    #[test]
    #[cfg(feature = "fixed")]
    fn fixed_point_handles_the_documented_limit() {
        use fixed::types::I16F16;

        // Column 0 is constant, so ṽ = (1 + √12, 1, …, 1); column 1 is ṽ scaled to norm 5000.
        let head = 1.0 + 12f64.sqrt();
        let length = (head * head + 11.0).sqrt();
        let mut m = [0.0f64; 24];
        for i in 0..12 {
            m[i * 2] = 1000.0;
            m[i * 2 + 1] = 5000.0 * if i == 0 { head } else { 1.0 } / length;
        }

        let mut fixed = m.map(I16F16::from_num);
        assert!(MatrixData::new_mut::<12, 2, I16F16>(&mut fixed).triangularize_upper());

        let r = fixed.map(|x| x.to_num::<f64>());
        assert_upper_triangular(&r, 12, 2);
        let expected = gram::<4>(&m, 12, 2);
        for (a, e) in gram::<4>(&r, 12, 2).iter().zip(&expected) {
            assert!((a - e).abs() <= 1e-4 * 25e6, "{a} vs {e}");
        }
    }

    /// A zero column (e.g. from a singular process noise factor) is skipped: no division by
    /// zero, which would panic for `I16F16`, and `Rᵀ×R = Mᵀ×M` still holds.
    #[test]
    #[cfg(feature = "fixed")]
    #[rustfmt::skip]
    fn fixed_point_handles_zero_and_rank_deficient_columns() {
        use fixed::types::I16F16;

        // M = [Aᵀ; 0] with a zero middle column and a third column parallel to the first.
        let m: [f64; 18] = [
            1.5, 0.0, 3.0,
            -0.5, 0.0, -1.0,
            2.0, 0.0, 4.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let mut fixed = m.map(I16F16::from_num);
        assert!(MatrixData::new_mut::<6, 3, I16F16>(&mut fixed).triangularize_upper());

        let r = fixed.map(|x| x.to_num::<f64>());
        assert_upper_triangular(&r, 6, 3);
        assert_close(&gram::<9>(&r, 6, 3), &gram::<9>(&m, 6, 3), 1e-3);

        let mut zero = [I16F16::ZERO; 4];
        assert!(MatrixData::new_mut::<2, 2, I16F16>(&mut zero).triangularize_upper());
        assert_eq!(zero, [I16F16::ZERO; 4]);
    }

    /// Returns `T×x` for the `3×3` factor given by `factor(i, k)` and the row-major `3×2` `x`.
    fn product(factor: impl Fn(usize, usize) -> f64, x: &[f64]) -> [f64; 6] {
        let mut b = [0.0; 6];
        for i in 0..3 {
            for c in 0..2 {
                b[i * 2 + c] = (0..3).map(|k| factor(i, k) * x[k * 2 + c]).sum();
            }
        }

        b
    }

    #[test]
    #[rustfmt::skip]
    fn triangular_solves_invert_their_factor() {
        // The 9.0 entries lie outside the factor's triangle and must be ignored.
        let l = MatrixData::new_array::<3, 3, 9, f64>([
            2.0, 9.0, 9.0,
            0.5, 1.5, 9.0,
            -1.0, 0.3, 0.7]);
        let u = MatrixData::new_array::<3, 3, 9, f64>([
            2.0, 0.5, -1.0,
            9.0, 1.5, 0.3,
            9.0, 9.0, 0.7]);
        let x = [1.0, -2.0, 0.5, 3.0, 0.25, -1.5];
        let lower = |i: usize, k: usize| if k <= i { l.get(i, k) } else { 0.0 };
        let upper = |i: usize, k: usize| if k >= i { u.get(i, k) } else { 0.0 };

        let mut b = MatrixData::new_array::<3, 2, 6, f64>(product(lower, &x));
        assert!(l.solve_lower_triangular_inplace(&mut b));
        assert_close(b.as_slice(), &x, 1e-12);

        let mut b = MatrixData::new_array::<3, 2, 6, f64>(product(|i, k| lower(k, i), &x));
        assert!(l.solve_lower_triangular_transposed_inplace(&mut b));
        assert_close(b.as_slice(), &x, 1e-12);

        let mut b = MatrixData::new_array::<3, 2, 6, f64>(product(upper, &x));
        assert!(u.solve_upper_triangular_inplace(&mut b));
        assert_close(b.as_slice(), &x, 1e-12);

        let mut b = MatrixData::new_array::<3, 2, 6, f64>(product(|i, k| upper(k, i), &x));
        assert!(u.solve_upper_triangular_transposed_inplace(&mut b));
        assert_close(b.as_slice(), &x, 1e-12);
    }

    #[test]
    fn triangular_solves_match_the_cholesky_inverse() {
        let mut l = MatrixData::new_array::<2, 2, 4, f64>([4.0, 1.0, 1.0, 3.0]);
        assert!(l.cholesky_decompose_lower());

        let mut inverse = MatrixData::new_array::<2, 2, 4, f64>([0.0; 4]);
        l.invert_l_cholesky(&mut inverse);

        let mut b = MatrixData::new_array::<2, 2, 4, f64>([1.0, 0.0, 0.0, 1.0]);
        assert!(l.solve_lower_triangular_inplace(&mut b));
        assert!(l.solve_lower_triangular_transposed_inplace(&mut b));
        assert_close(b.as_slice(), inverse.as_slice(), 1e-12);
    }

    /// A zero diagonal is reported instead of dividing by zero, which panics for `I16F16`.
    #[test]
    #[cfg(feature = "fixed")]
    fn triangular_solves_reject_a_zero_diagonal() {
        use fixed::types::I16F16;

        let one = I16F16::ONE;
        let t = MatrixData::new_array::<2, 2, 4, I16F16>([one, one, one, I16F16::ZERO]);
        let original = [one, -one];
        let mut b = MatrixData::new_array::<2, 1, 2, I16F16>(original);

        assert!(!t.solve_lower_triangular_inplace(&mut b));
        assert!(!t.solve_lower_triangular_transposed_inplace(&mut b));
        assert!(!t.solve_upper_triangular_inplace(&mut b));
        assert!(!t.solve_upper_triangular_transposed_inplace(&mut b));
        assert_eq!(b.as_slice(), &original);
    }
}
