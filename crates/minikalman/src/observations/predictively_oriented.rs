//! # Predictively oriented (PrO) correction for scalar linear observations.
//!
//! The PrO update (Shen, Duran-Martin, Oates, [arXiv:2606.03230](https://arxiv.org/abs/2606.03230);
//! Kean, Shen, Oates, [arXiv:2610.14773](https://arxiv.org/abs/2610.14773)) replaces the Bayesian
//! update with the Gaussian `q = N(m, P)` that minimizes
//!
//! ```text
//! −log ∫ p(z|x) q(x) dx + KL(q ‖ N(x̃, P̃))
//! ```
//!
//! For a linear observation the minimizer is Kalman-shaped, `P⁻¹ = P̃⁻¹ + hᵀ·w·h`, so only the
//! scalar `c = h·P·hᵀ` has to be found. With `g = h·P̃·hᵀ`, the noise variance `r` and the prior
//! innovation `v = z − h·x̃`, the stationarity condition is
//!
//! ```text
//! f(c) = 1/c − 1/g − 1/(c+r) + v²/(c+r+g)² = 0.
//! ```
//!
//! For `c > 0`, `f` is strictly decreasing (`f′ = −1/c² + 1/(c+r)² − 2v²/(c+r+g)³ < 0`) and
//! strictly convex (`f″ = 2/c³ − 2/(c+r)³ + 6v²/(c+r+g)⁴ > 0`), and `f(c) → −1/g` as `c → ∞`,
//! so it has exactly one root. Newton's method started left of the root, where `f > 0`, then
//! increases monotonically towards the root and never overshoots it: every iterate is a valid,
//! positive definite posterior. The start is the larger of two points where `f ≥ 0`: the root of
//! the `v = 0` case, `(√(r² + 4gr) − r)/2`, and `|v|·√g − r − g`, where `v²/(c+r+g)² = 1/g`.
//!
//! Given `c`, the posterior is
//!
//! ```text
//! x = x̃ + P̃·hᵀ · v/(c+r+g)
//! P = P̃ − P̃·hᵀ·h·P̃ · (g−c)/g²
//! ```

use crate::kalman::CorrectionStatus;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};

/// Performs the PrO update of a scalar linear observation from the prior innovation `y = z − H×x`.
///
/// Stores the normalized innovation squared `y²/(H×P×Hᵀ + R)` of the prior in `nis`, exactly as
/// the regular update does, and honors `gate` the same way.
///
/// Leaves `x` and `P` unchanged and returns [`CorrectionStatus::NotPositiveDefinite`] unless both
/// `g = H×P×Hᵀ` and `R` are strictly positive (this also rejects NaN).
///
/// ## Buffers
/// `temp_PHt` may be aliased with `temp_KHP`, so it is copied into `K` before `temp_KHP` is
/// written. The covariance term is formed before `K` is scaled to the gain, which keeps every
/// intermediate of the order of the inputs even for huge outliers.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_scalar<const STATES: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<1, STATES, T>,
    R: &impl Matrix<1, 1, T>,
    y: &impl Matrix<1, 1, T>,
    K: &mut impl MatrixMut<STATES, 1, T>,
    S: &mut impl MatrixMut<1, 1, T>,
    temp_PHt: &mut impl MatrixMut<STATES, 1, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    nis: &mut T,
    gate: Option<T>,
    max_iterations: usize,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // g = H*P*Hᵀ, kept in S.
    P.mult_transb(H, temp_PHt);
    H.mult(temp_PHt, S);
    let g = S.get(0, 0);
    let r = R.get(0, 0);
    let v = y.get(0, 0);

    // Written positively so that NaN is rejected as well.
    let positive = g > T::zero() && r > T::zero();
    if !positive {
        return CorrectionStatus::NotPositiveDefinite;
    }

    *nis = v * v / (g + r);
    if gate.is_some_and(|gate| *nis > gate) {
        return CorrectionStatus::Gated;
    }

    let c = solve_stationarity(g, r, v, max_iterations);

    // P -= (g−c)/g² · (P*Hᵀ)(P*Hᵀ)ᵀ, from a copy in K since temp_PHt may alias temp_KHP.
    K.copy_from(temp_PHt);
    K.multscale_transb(K, (g - c) / g / g, temp_KHP);
    P.sub_inplace_a(temp_KHP);

    // K = P*Hᵀ / (c+r+g); x += K*y
    let mean_gain = (c + r + g).recip();
    for i in 0..STATES {
        K.set(i, 0, K.get(i, 0) * mean_gain);
    }

    K.multadd_rowvector(y, x);
    CorrectionStatus::Applied
}

/// Finds the root `c = H×P×Hᵀ` of the PrO stationarity condition `f(c) = 0` (see the module
/// docs) by Newton's method from a point left of the root.
///
/// Requires `g > 0` and `r > 0`. Stops after `max_iterations` steps or as soon as a step makes
/// no progress in `T`, which happens at the working precision because the iteration converges
/// monotonically. On budget exhaustion the returned `c` lies left of the root: a valid, but
/// less inflated, posterior. `max_iterations = 0` returns the start point.
fn solve_stationarity<T>(g: T, r: T, v: T, max_iterations: usize) -> T
where
    T: MatrixDataType,
{
    let mut c = newton_start(g, r, v);
    for _ in 0..max_iterations {
        let (f, df) = stationarity(c, g, r, v);
        if df < T::zero() {
            let next = c - f / df;
            if next > c {
                c = next;
                continue;
            }
        }

        break;
    }

    c
}

/// Returns a start point `c > 0` with `f(c) ≥ 0`, i.e. left of the root.
///
/// Picks the largest of three candidates that passes `f(c) ≥ 0`:
/// - the large-innovation estimate `|v|·√g − r − g`, where `v²/(c+r+g)² = 1/g`,
/// - the `v = 0` root `(√(r² + 4gr) − r)/2`,
/// - the Kalman filter's `g·r/(g+r)`, which always passes and needs no square root.
///
/// The first two are exact lower bounds, but an approximate square root (e.g. `micromath`) or
/// cancellation in `|v|·√g − g` can push them right of the root, hence the check.
/// `|v|·√g` is formed as a product to avoid overflowing `v²·g`.
fn newton_start<T>(g: T, r: T, v: T) -> T
where
    T: MatrixDataType,
{
    let two = T::one() + T::one();
    let four = two + two;
    let abs_v = if v < T::zero() { -v } else { v };
    let large_innovation = abs_v * g.square_root() - r - g;
    // (√(r² + 4gr) − r)/2, rationalized to avoid cancellation when g ≪ r.
    let no_innovation = two * g * r / ((r * r + four * g * r).square_root() + r);
    let kalman = g * r / (g + r);

    let mut start = kalman;
    for candidate in [no_innovation, large_innovation] {
        if candidate > start && stationarity(candidate, g, r, v).0 >= T::zero() {
            start = candidate;
        }
    }

    start
}

/// Evaluates the stationarity condition `f(c)` and its derivative `f′(c)` (see the module docs).
///
/// Uses `a = v/(c+r+g)` so that neither `v²` nor `(c+r+g)²` is formed on its own.
fn stationarity<T>(c: T, g: T, r: T, v: T) -> (T, T)
where
    T: MatrixDataType,
{
    let two = T::one() + T::one();
    let inv_c = c.recip();
    let inv_cr = (c + r).recip();
    let a = v / (c + r + g);
    let a2 = a * a;

    let f = inv_c - g.recip() - inv_cr + a2;
    let df = inv_cr * inv_cr - inv_c * inv_c - two * a2 / (c + r + g);
    (f, df)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derivation's toy problem `P̃ = H = R = 1`, `v = 3` has its root at `c ≈ 1.5019`.
    #[test]
    fn toy_problem_root() {
        let c = solve_stationarity(1.0_f64, 1.0, 3.0, 50);
        assert!((c - 1.5019).abs() < 1e-4, "c = {c}");
        assert!(stationarity(c, 1.0, 1.0, 3.0).0.abs() < 1e-12);
    }

    /// Without innovation the root has the closed form `(√(r² + 4gr) − r)/2`, which is also the
    /// start point, so no iteration is needed.
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))] // micromath's square root is approximate
    fn no_innovation_root_is_closed_form() {
        let (g, r) = (4.0_f64, 0.5);
        let expected = ((r * r + 4.0 * g * r).sqrt() - r) / 2.0;
        assert!((solve_stationarity(g, r, 0.0, 0) - expected).abs() < 1e-12);
        assert!((solve_stationarity(g, r, 0.0, 50) - expected).abs() < 1e-12);
    }

    /// Ill-conditioned problems and huge innovations converge within a small budget, and the
    /// result is never left of the Kalman filter's `c = g·r/(g+r)` (up to rounding, since the two
    /// coincide for `g ≪ r` and `v = 0`).
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))] // micromath's square root is approximate
    fn converges_on_hard_problems() {
        for &g in &[1e-4_f64, 1.0, 1e4] {
            for &r in &[1e-4_f64, 1.0, 1e4] {
                for &nu in &[0.0_f64, 1.0, 3.0, 1000.0] {
                    let v = -nu * (g + r).sqrt();
                    let c = solve_stationarity(g, r, v, 20);
                    let (f, df) = stationarity(c, g, r, v);
                    assert!(c >= g * r / (g + r) * (1.0 - 1e-12), "g={g} r={r} nu={nu}");
                    assert!((f / df).abs() <= 1e-10 * c, "g={g} r={r} nu={nu} c={c}");
                }
            }
        }
    }

    /// An exhausted budget leaves `c` left of the root, where `f(c) ≥ 0`.
    #[test]
    fn exhausted_budget_stays_left_of_root() {
        let c = solve_stationarity(1e4_f64, 1e-4, 1e2, 1);
        assert!(stationarity(c, 1e4, 1e-4, 1e2).0 >= 0.0);
    }
}
