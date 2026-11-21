//! Factorizations of the innovation covariance `S = H×P×Hᵀ + R` that apply `S⁻¹` by
//! substitution, for the Kalman gain and the normalized innovation squared (NIS).
//!
//! For floating-point types with up to [`SQUARE_ROOT_FREE_MAX`] measurements, `S` is factored
//! as `L×D×Lᵀ` with a unit lower triangular `L` and a diagonal `D` ([`LdlFactor`]), which needs
//! no square root and lives in local storage. For a single measurement this is `K = P×Hᵀ/s`
//! and `NIS = y²/s`. Otherwise, `S` is copied into a scratch matrix (`OBSERVATIONS²` values)
//! and takes the Cholesky factor `S = L×Lᵀ` there ([`CholeskyFactor`]). Either way `S` is never
//! overwritten, and the choice depends only on the element type and the const `OBSERVATIONS`,
//! so it is made at compile time.
//!
//! Fixed-point types always take the Cholesky path: `L×D×Lᵀ` divides by the pivots `Dᵢ`, the
//! squares of the Cholesky pivots `Lᵢᵢ`, so its intermediate values span twice the dynamic
//! range and overflow a fixed-point type near its resolution where the Cholesky path does not
//! (see [`MatrixDataType::SQUARE_ROOT_FREE_GAIN`]).
//!
//! Both factors apply a pivot by multiplying with its reciprocal, formed once per pivot: the
//! gain then costs `OBSERVATIONS` divisions, fewer than the explicit inverse did, which matters
//! where a division is expensive (fixed point, soft float). The one exception is the
//! single-measurement `L×D×Lᵀ`, which divides by `s` (`OBSERVATIONS + 1` divisions per gain) so
//! that `K = P×Hᵀ/s` is correctly rounded.

use crate::matrix::triangular::{Transpose, Triangle, substitute};
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};

/// The largest number of measurements whose innovation covariance is factored square-root free
/// ([`LdlFactor`]) for types with [`MatrixDataType::SQUARE_ROOT_FREE_GAIN`]; larger ones, and
/// all other types, use the Cholesky factor of a copy ([`CholeskyFactor`]).
///
/// The `L×D×Lᵀ` factor lives in an `OBSERVATIONS × OBSERVATIONS` local array, so the bound
/// keeps the stack use small (nine values).
pub(crate) const SQUARE_ROOT_FREE_MAX: usize = 3;

/// Returns `true` if innovation covariances of `OBSERVATIONS` measurements with elements `T`
/// are factored square-root free ([`LdlFactor`]). A constant for every `OBSERVATIONS` and `T`.
pub(crate) const fn is_square_root_free<const OBSERVATIONS: usize, T>() -> bool
where
    T: MatrixDataType,
{
    T::SQUARE_ROOT_FREE_GAIN && OBSERVATIONS <= SQUARE_ROOT_FREE_MAX
}

/// A factorization of a symmetric positive definite innovation covariance `S`.
pub(crate) enum InnovationFactor<'a, const OBSERVATIONS: usize, T> {
    /// `S = L×D×Lᵀ`, in local storage.
    SquareRootFree(LdlFactor<OBSERVATIONS, T>),
    /// `S = L×Lᵀ`, with `L` in the scratch matrix passed to [`new`](Self::new).
    Cholesky(CholeskyFactor<'a, OBSERVATIONS, T>),
}

impl<'a, const OBSERVATIONS: usize, T> InnovationFactor<'a, OBSERVATIONS, T>
where
    T: MatrixDataType,
{
    /// Factors `S`, square-root free where [`is_square_root_free`] says so, and otherwise with
    /// [`cholesky_decompose_lower`](MatrixMut::cholesky_decompose_lower) on a copy in
    /// `scratch`. `S` is never written, and `scratch` only on the Cholesky path.
    ///
    /// Returns `None` if `S` is not positive definite: a pivot is not positive or not finite
    /// (the rule of `cholesky_decompose_lower`).
    #[inline]
    #[allow(non_snake_case)]
    pub(crate) fn new(
        S: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
        scratch: &'a mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    ) -> Option<Self> {
        if is_square_root_free::<OBSERVATIONS, T>() {
            return LdlFactor::new(S.as_slice()).map(Self::SquareRootFree);
        }

        let len = OBSERVATIONS * OBSERVATIONS;
        scratch.as_mut_slice()[..len].copy_from_slice(&S.as_slice()[..len]);
        if !scratch.cholesky_decompose_lower() {
            return None;
        }

        Some(Self::Cholesky(CholeskyFactor::new(scratch.as_slice())))
    }

    /// Returns the normalized innovation squared `yᵀ×S⁻¹×y`.
    #[inline]
    pub(crate) fn nis(&self, y: &impl Matrix<OBSERVATIONS, 1, T>) -> T {
        match self {
            Self::SquareRootFree(factor) => factor.nis(y),
            Self::Cholesky(factor) => factor.nis(y),
        }
    }

    /// Replaces the row `b` (length `OBSERVATIONS`) by `S⁻¹×b`.
    #[inline]
    pub(crate) fn solve_row(&self, b: &mut [T]) {
        match self {
            Self::SquareRootFree(factor) => factor.solve_row(b),
            Self::Cholesky(factor) => factor.solve_row(b),
        }
    }
}

/// Returns the prior NIS `yᵀ×(G + R)⁻¹×y` as the plain correction forms it, given the Cholesky
/// factor `cholesky` of `G + R` that the robust correction has already computed.
///
/// On the Cholesky path this reads `cholesky` and adds no factorization. On the square-root
/// free path it factors `G + R` as `L×D×Lᵀ` in local storage (at most nine values), as the
/// plain correction does; `None` if that rejects a pivot.
#[allow(non_snake_case)]
pub(crate) fn prior_nis<const OBSERVATIONS: usize, T>(
    G: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    cholesky: &[T],
    y: &impl Matrix<OBSERVATIONS, 1, T>,
) -> Option<T>
where
    T: MatrixDataType,
{
    if !is_square_root_free::<OBSERVATIONS, T>() {
        return Some(CholeskyFactor::<OBSERVATIONS, T>::new(cholesky).nis(y));
    }

    let mut s = [[T::zero(); OBSERVATIONS]; OBSERVATIONS];
    for (i, row) in s.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = G.get(i, j) + R.get(i, j);
        }
    }

    LdlFactor::<OBSERVATIONS, T>::new(s.as_flattened()).map(|factor| factor.nis(y))
}

/// The square-root free factorization `S = L×D×Lᵀ` of a small innovation covariance, with a
/// unit lower triangular `L` and the diagonal `D`.
pub(crate) struct LdlFactor<const OBSERVATIONS: usize, T> {
    /// `L` below the diagonal; the diagonal (ones) and the upper triangle are never read.
    l: [[T; OBSERVATIONS]; OBSERVATIONS],
    /// The pivots `Dᵢ`, all positive and finite.
    d: [T; OBSERVATIONS],
    /// `1 / Dᵢ` for more than one measurement; unused (zero) for one, which divides by `D₀`.
    inverse_d: [T; OBSERVATIONS],
}

impl<const OBSERVATIONS: usize, T> LdlFactor<OBSERVATIONS, T>
where
    T: MatrixDataType,
{
    /// Factors the row-major `s`, reading its upper triangle as
    /// [`cholesky_decompose_lower`](MatrixMut::cholesky_decompose_lower) does.
    ///
    /// The pivot `Dⱼ = Sⱼⱼ − Σₖ Lⱼₖ²×Dₖ` is the square of the Cholesky pivot in exact
    /// arithmetic, and is rejected by the same rule: `None` if it is not positive or not finite.
    fn new(s: &[T]) -> Option<Self> {
        let s = &s[..OBSERVATIONS * OBSERVATIONS];
        let mut l = [[T::zero(); OBSERVATIONS]; OBSERVATIONS];
        let mut d = [T::zero(); OBSERVATIONS];
        let mut inverse_d = [T::zero(); OBSERVATIONS];
        for j in 0..OBSERVATIONS {
            // Lⱼₖ×Dₖ for the columns already done; reused for the pivot and the column below.
            let mut scaled = [T::zero(); OBSERVATIONS];
            for k in 0..j {
                scaled[k] = l[j][k] * d[k];
            }

            let pivot = s[j * OBSERVATIONS + j] - (0..j).map(|k| l[j][k] * scaled[k]).sum::<T>();
            if !is_valid_pivot(pivot) {
                return None;
            }

            d[j] = pivot;
            if OBSERVATIONS > 1 {
                inverse_d[j] = pivot.recip();
            }

            for i in (j + 1)..OBSERVATIONS {
                let sum = (0..j).map(|k| l[i][k] * scaled[k]).sum::<T>();
                l[i][j] = (s[j * OBSERVATIONS + i] - sum) * inverse_d[j];
            }
        }

        Some(Self { l, d, inverse_d })
    }

    /// Returns `yᵀ×S⁻¹×y = wᵀ×D⁻¹×w` with `w = L⁻¹×y`, summed as `Σ wᵢ×(wᵢ/Dᵢ)`.
    fn nis(&self, y: &impl Matrix<OBSERVATIONS, 1, T>) -> T {
        let mut w = [T::zero(); OBSERVATIONS];
        w.copy_from_slice(&y.as_slice()[..OBSERVATIONS]);
        self.substitute(Transpose::No, &mut w);
        (0..OBSERVATIONS).map(|i| w[i] * self.scale(i, w[i])).sum()
    }

    /// Replaces the row `b` by `S⁻¹×b = L⁻ᵀ×(D⁻¹×(L⁻¹×b))`.
    fn solve_row(&self, b: &mut [T]) {
        self.substitute(Transpose::No, b);
        for (i, value) in b.iter_mut().enumerate() {
            *value = self.scale(i, *value);
        }

        self.substitute(Transpose::Yes, b);
    }

    /// Returns `value / Dᵢ`: a division for one measurement (correctly rounded), a
    /// multiplication with the reciprocal otherwise. `OBSERVATIONS` is a constant, so the
    /// branch folds away.
    #[inline(always)]
    fn scale(&self, i: usize, value: T) -> T {
        if OBSERVATIONS == 1 {
            value / self.d[i]
        } else {
            value * self.inverse_d[i]
        }
    }

    /// Replaces `b` by `L⁻¹×b` or `L⁻ᵀ×b`; the unit diagonal needs no pivot step.
    #[inline]
    fn substitute(&self, transpose: Transpose, b: &mut [T]) {
        substitute(
            self.l.as_flattened(),
            OBSERVATIONS,
            Triangle::Lower,
            transpose,
            b,
            1,
            |_, rest| rest,
        );
    }
}

/// The lower Cholesky factor `L` of an innovation covariance `S = L×Lᵀ`, for applying `S⁻¹`
/// by substitution without forming it.
///
/// The substitutions multiply with the reciprocals `1/Lᵢᵢ`, formed once per pivot, rather than
/// dividing per entry: that keeps the gain at `OBSERVATIONS` divisions, fewer than the explicit
/// inverse needed.
pub(crate) struct CholeskyFactor<'a, const OBSERVATIONS: usize, T> {
    /// The row-major `OBSERVATIONS × OBSERVATIONS` factor; only the lower triangle is read.
    l: &'a [T],
    /// `1 / Lᵢᵢ`.
    inverse_diagonal: [T; OBSERVATIONS],
}

impl<'a, const OBSERVATIONS: usize, T> CholeskyFactor<'a, OBSERVATIONS, T>
where
    T: MatrixDataType,
{
    /// Wraps the factor `l` from a successful
    /// [`cholesky_decompose_lower`](MatrixMut::cholesky_decompose_lower), whose diagonal is
    /// positive.
    pub(crate) fn new(l: &'a [T]) -> Self {
        let l = &l[..OBSERVATIONS * OBSERVATIONS];
        let mut inverse_diagonal = [T::zero(); OBSERVATIONS];
        for (i, inverse) in inverse_diagonal.iter_mut().enumerate() {
            *inverse = l[i * OBSERVATIONS + i].recip();
        }

        Self {
            l,
            inverse_diagonal,
        }
    }

    /// Returns the normalized innovation squared `yᵀ×S⁻¹×y = ‖L⁻¹×y‖²`, from one forward
    /// substitution.
    pub(crate) fn nis(&self, y: &impl Matrix<OBSERVATIONS, 1, T>) -> T {
        let mut w = [T::zero(); OBSERVATIONS];
        w.copy_from_slice(&y.as_slice()[..OBSERVATIONS]);
        self.substitute(Transpose::No, &mut w);
        w.iter().map(|&w| w * w).sum()
    }

    /// Replaces the row `b` by `S⁻¹×b = L⁻ᵀ×(L⁻¹×b)`, with one forward and one back
    /// substitution.
    fn solve_row(&self, b: &mut [T]) {
        self.substitute(Transpose::No, b);
        self.substitute(Transpose::Yes, b);
    }

    /// Replaces `b` by `L⁻¹×b` or `L⁻ᵀ×b`.
    #[inline]
    fn substitute(&self, transpose: Transpose, b: &mut [T]) {
        let inverse_diagonal = &self.inverse_diagonal;
        substitute(
            self.l,
            OBSERVATIONS,
            Triangle::Lower,
            transpose,
            b,
            1,
            |i, rest| rest * inverse_diagonal[i],
        );
    }
}

/// Returns `true` for a pivot that `cholesky_decompose_lower` would accept: positive (which
/// rejects a NaN) and finite (`pivot − pivot` is zero only for finite values, which rejects
/// `+inf`).
#[allow(clippy::eq_op)]
fn is_valid_pivot<T>(pivot: T) -> bool
where
    T: MatrixDataType,
{
    pivot > T::zero() && pivot - pivot == T::zero()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{MatrixData, SquareMatrix};

    /// Returns whether the row-major 2×2 `s` is accepted by the square-root free factorization
    /// and by `cholesky_decompose_lower`.
    fn accepted(s: [f32; 4]) -> (bool, bool) {
        let mut copy = s;
        let cholesky = MatrixData::new_mut::<2, 2, f32>(&mut copy).cholesky_decompose_lower();
        (LdlFactor::<2, f32>::new(&s).is_some(), cholesky)
    }

    /// Non-positive and non-finite pivots are rejected by both factorizations alike.
    #[test]
    fn rejects_the_pivots_cholesky_rejects() {
        for s in [
            [1.0, 0.5, 0.5, 2.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [f32::NAN, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, f32::INFINITY],
            [f32::MAX, f32::MAX, f32::MAX, 1.0],
        ] {
            let (ldl, cholesky) = accepted(s);
            assert_eq!(ldl, cholesky, "S = {s:?}");
        }
    }

    /// `S⁻¹` applied through `L×D×Lᵀ` matches the explicit inverse for one, two and three
    /// measurements, and so does the NIS.
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn applies_the_inverse() {
        #[rustfmt::skip]
        let s: [f64; 9] = [
            4.0, 1.0, 0.5,
            1.0, 3.0, 0.2,
            0.5, 0.2, 2.0];
        let y = [0.3, -1.2, 0.7];
        assert_applies_the_inverse::<1, 1>(&[4.0], &y);
        assert_applies_the_inverse::<2, 4>(&[4.0, 1.0, 1.0, 3.0], &y);
        assert_applies_the_inverse::<3, 9>(&s, &y);
    }

    /// Fixed-point types never take the square-root free path; floating-point types do up to
    /// three measurements.
    #[test]
    fn dispatch_depends_on_the_type_and_the_size() {
        assert!(is_square_root_free::<1, f32>());
        assert!(is_square_root_free::<3, f64>());
        assert!(!is_square_root_free::<4, f32>());
        #[cfg(feature = "fixed")]
        {
            assert!(!is_square_root_free::<1, fixed::types::I16F16>());
            assert!(!is_square_root_free::<2, fixed::types::I32F32>());
        }
    }

    /// Checks [`LdlFactor`] against `S⁻¹` from the Cholesky inverse, for `M` measurements
    /// (`MM = M²`).
    #[cfg(any(feature = "std", feature = "libm"))]
    fn assert_applies_the_inverse<const M: usize, const MM: usize>(s: &[f64], y: &[f64]) {
        let mut factor = [0.0; MM];
        factor.copy_from_slice(s);
        let mut inverse = [0.0; MM];
        let mut l = MatrixData::new_mut::<M, M, f64>(&mut factor);
        assert!(l.cholesky_decompose_lower());
        l.invert_l_cholesky(&mut MatrixData::new_mut::<M, M, f64>(&mut inverse));

        let ldl = LdlFactor::<M, f64>::new(s).expect("S is positive definite");
        let mut y_m = [0.0; M];
        y_m.copy_from_slice(&y[..M]);
        let mut row = y_m;
        ldl.solve_row(&mut row);

        let mut nis = 0.0;
        for i in 0..M {
            let expected: f64 = (0..M).map(|j| inverse[i * M + j] * y_m[j]).sum();
            assert!((row[i] - expected).abs() < 1e-14, "S⁻¹×y[{i}]");
            nis += y_m[i] * expected;
        }

        let got = ldl.nis(&MatrixData::new_ref::<M, 1, f64>(&y_m));
        assert!((got - nis).abs() < 1e-14, "NIS {got} vs {nis}");
    }
}
