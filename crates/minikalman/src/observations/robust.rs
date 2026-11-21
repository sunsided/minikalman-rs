//! # Outlier-robust correction by a Student-t measurement noise model.
//!
//! The measurement noise is modelled as Student-t with `ν` degrees of freedom, i.e. as a
//! Gaussian `N(0, R/λ)` whose precision scale is `λ ~ Gamma(ν/2, ν/2)`. The variational Bayes
//! (VB) update of Piché, Särkkä and Hartikainen, "Recursive outlier-robust filtering and
//! smoothing for nonlinear systems using the multivariate Student-t distribution", IEEE MLSP
//! 2012, Algorithm 1, alternates between an ordinary Kalman update with the noise covariance
//! `R/w` and the mean `w = E[λ]` of the posterior of `λ`, starting with `w = 1`:
//!
//! ```text
//! x⁺, P⁺ ← Kalman update of x⁻, P⁻ with R/w
//! D = (z − H×x⁺)(z − H×x⁺)ᵀ + H×P⁺×Hᵀ
//! γ = tr(D×R⁻¹)
//! w = (ν + m) / (ν + γ)
//! ```
//!
//! with `m = OBSERVATIONS`. A large residual makes `γ` large and `w` small, which inflates the
//! effective noise `R/w` and so discounts the measurement. Since `R/w` is a valid noise
//! covariance, the update is an ordinary Kalman update and never makes `P` larger than the
//! prior.
//!
//! ## Measurement space
//!
//! The iteration needs no state-sized buffers. With `G = H×P⁻×Hᵀ`, the prior innovation
//! `y = z − H×x⁻` and `S_w = w×G + R`, the Kalman update with `R/w` has the innovation
//! covariance `G + R/w = S_w/w`, so
//!
//! ```text
//! z − H×x⁺ = (R/w)×(G + R/w)⁻¹×y = R×S_w⁻¹×y
//! H×P⁺×Hᵀ = G − G×(G + R/w)⁻¹×G = G×S_w⁻¹×R
//! γ = bᵀ×R×b + tr(G×S_w⁻¹),   b = S_w⁻¹×y
//! ```
//!
//! Neither `R⁻¹` nor a division by `w` is needed, so a weight near zero (a huge outlier) stays
//! representable in fixed-point types. `b` grows like `R⁻¹×y` and `bᵀ×R×b` with the squared
//! outlier size; both are evaluated from `y` and `b` scaled to at most `1` in magnitude, and
//! `w` is formed from the scaled terms, so they do not overflow either. `S_w⁻¹` itself is
//! bounded by `R⁻¹`, so `R⁻¹` must be representable. The final gain is `K = P⁻×Hᵀ×(w×S_w⁻¹)`.
//!
//! The loop stops once the weight changes by at most `10⁻⁶×w + ε` (with the machine epsilon
//! `ε` of `T`), or after the caller's iteration budget. Every iterate is a valid posterior.

use crate::kalman::CorrectionStatus;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};
use crate::observations::CovarianceUpdate;
use crate::observations::innovation_factor::prior_nis;
use crate::observations::predictively_oriented::vector::machine_epsilon;

/// The relative change of the weight below which the iteration counts as converged.
const RELATIVE_TOLERANCE: f64 = 1e-6;

/// The weight function of a [robust correction](RobustCorrection).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum RobustWeight<T> {
    /// Student-t measurement noise with `dof` (`ν`) degrees of freedom, fitted by variational
    /// Bayes (Piché, Särkkä, Hartikainen, IEEE MLSP 2012, Algorithm 1).
    ///
    /// `dof` must be positive. Small values (e.g. `3` to `5`) discount outliers strongly; as
    /// `dof → ∞`, the weight tends to `1` and the correction to the Kalman update. With
    /// fixed-point types, `dof + OBSERVATIONS` must be representable.
    StudentT {
        /// The degrees of freedom `ν > 0`.
        dof: T,
    },
}

/// The settings of a robust correction, e.g.
/// [`RegularObservation::correct_robust`](crate::regular::RegularObservation::correct_robust).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustCorrection<T> {
    /// The weight function, i.e. the noise model.
    pub weight: RobustWeight<T>,

    /// The budget of weight refreshes. Every refresh costs one Cholesky factorization and one
    /// inversion of an `OBSERVATIONS × OBSERVATIONS` matrix. `0` applies the plain Kalman
    /// update (`w = 1`).
    ///
    /// Algorithm 1 of Piché et al. with `N` fixed-point iterations corresponds to
    /// `iterations = N − 1`, since its last refresh of the weight is not used by the update;
    /// the paper's `N = 5` is `4` here.
    ///
    /// The iteration converges linearly. While `H×P×Hᵀ` is at most about `R`, the weight
    /// settles to a relative change of `10⁻⁶` within about 20 refreshes for any innovation.
    /// When `H×P×Hᵀ` is much larger than `R`, innovations near the size where the weight
    /// drops from about `1` to near `0` converge slowly and can need hundreds of refreshes. A
    /// correction that runs out of the budget applies the last weight and reports
    /// [`CorrectionStatus::NotConverged`]. In the 2-D tracking simulations of the tests
    /// (`H×P×Hᵀ` about a tenth of `R`), a budget of `20` converged in every correction.
    pub iterations: usize,
}

/// The result of [`solve_weight`]: the weight `w` and whether the iteration converged.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WeightSolution<T> {
    /// The weight `w` of the last factorization, i.e. the one the update applies.
    pub(crate) weight: T,

    /// Whether the last refresh changed `w` by at most the tolerance.
    pub(crate) converged: bool,
}

impl<T> WeightSolution<T> {
    /// The status to report once the update with this weight is applied.
    pub(crate) fn status(&self) -> CorrectionStatus {
        if self.converged {
            CorrectionStatus::Applied
        } else {
            CorrectionStatus::NotConverged
        }
    }
}

/// Iterates the Student-t weight `w` in measurement space (see the module docs).
///
/// `G` is `H×P⁻×Hᵀ` (or the predicted measurement covariance of the unscented filter) and `y`
/// the prior innovation. The first factorization uses `w = 1`, i.e. `S = G + R`, so the NIS
/// stored in `nis` is that of the prior, and `gate` is applied to it as in the regular update.
///
/// On success, `S_inv` holds `S_w⁻¹ = (w×G + R)⁻¹` for the returned weight (so
/// `(G + R/w)⁻¹ = w×S_inv`), and `S` the Cholesky factor of `S_w`. Returns [`CorrectionStatus::NotPositiveDefinite`] if any
/// `w×G + R` has no Cholesky factor, or [`CorrectionStatus::Gated`]; neither touches anything
/// but `S`, `S_inv` and `nis`.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn solve_weight<const M: usize, T>(
    G: &impl Matrix<M, M, T>,
    R: &impl Matrix<M, M, T>,
    y: &impl Matrix<M, 1, T>,
    S: &mut impl MatrixMut<M, M, T>,
    S_inv: &mut impl MatrixMut<M, M, T>,
    nis: &mut T,
    gate: Option<T>,
    robust: RobustCorrection<T>,
) -> Result<WeightSolution<T>, CorrectionStatus>
where
    T: MatrixDataType,
{
    let RobustWeight::StudentT { dof } = robust.weight;
    debug_assert!(dof > T::zero(), "the degrees of freedom must be positive");

    let mut solution = WeightSolution {
        weight: T::one(),
        converged: false,
    };

    for iteration in 0..=robust.iterations {
        invert_scaled(G, R, solution.weight, S, S_inv)?;
        if iteration == 0 {
            // The prior NIS as the plain update forms it; on the Cholesky path from the factor
            // that `invert_scaled` just left in `S`.
            *nis = prior_nis(G, R, S.as_slice(), y).ok_or(CorrectionStatus::NotPositiveDefinite)?;
            if gate.is_some_and(|gate| *nis > gate) {
                return Err(CorrectionStatus::Gated);
            }
        }

        if solution.converged || iteration == robust.iterations {
            break;
        }

        let next = student_t_weight(dof, G, R, y, S_inv);
        solution.converged = is_settled(solution.weight, next);
        solution.weight = next;
    }

    Ok(solution)
}

/// Writes `S = w×G + R`, factors it in place and writes `S_inv = S⁻¹`.
///
/// Returns [`CorrectionStatus::NotPositiveDefinite`] if `S` has no Cholesky factor.
#[allow(non_snake_case)]
fn invert_scaled<const M: usize, T>(
    G: &impl Matrix<M, M, T>,
    R: &impl Matrix<M, M, T>,
    w: T,
    S: &mut impl MatrixMut<M, M, T>,
    S_inv: &mut impl MatrixMut<M, M, T>,
) -> Result<(), CorrectionStatus>
where
    T: MatrixDataType,
{
    write_scaled_covariance(G, R, w, S);
    if !S.cholesky_decompose_lower() {
        return Err(CorrectionStatus::NotPositiveDefinite);
    }

    S.invert_l_cholesky(S_inv);
    Ok(())
}

/// Returns the refreshed weight `(ν + m)/(ν + γ)` with `γ = bᵀ×R×b + tr(G×S_w⁻¹)` and
/// `b = S_w⁻¹×y`, given `S_inv = S_w⁻¹`.
///
/// `b` is formed from `y/s` with `s = max(1, maxᵢ |yᵢ|)`, then scaled by
/// `t = max(1, maxᵢ |bᵢ|)`, and the weight is formed as
/// `((ν + m)/(s·t)²) / ((ν + τ)/(s·t)² + b̃ᵀ×R×b̃)` with `b̃ = b/(s·t)` and `τ = tr(G×S_w⁻¹)`.
/// Neither `b` itself (about `R⁻¹×y` for an outlier, which can exceed the range of a
/// fixed-point type while the prior NIS still fits) nor the squared outlier size is formed.
#[allow(non_snake_case)]
fn student_t_weight<const M: usize, T>(
    dof: T,
    G: &impl Matrix<M, M, T>,
    R: &impl Matrix<M, M, T>,
    y: &impl Matrix<M, 1, T>,
    S_inv: &impl Matrix<M, M, T>,
) -> T
where
    T: MatrixDataType,
{
    let s = (0..M).map(|j| abs(y.get(j, 0))).fold(T::one(), max);
    let mut b = [T::zero(); M];
    for (i, value) in b.iter_mut().enumerate() {
        *value = (0..M).map(|j| S_inv.get(i, j) * (y.get(j, 0) / s)).sum();
    }

    let t = b.iter().map(|&v| abs(v)).fold(T::one(), max);
    for value in b.iter_mut() {
        *value = *value / t;
    }

    let mut quadratic = T::zero();
    let mut trace = T::zero();
    for i in 0..M {
        for j in 0..M {
            quadratic += b[i] * R.get(i, j) * b[j];
            trace += G.get(i, j) * S_inv.get(j, i);
        }
    }

    let m = T::from_usize(M).unwrap_or_else(T::one);
    let numerator = (dof + m) / s / s / t / t;
    numerator / ((dof + trace) / s / s / t / t + quadratic)
}

/// Returns whether the weight moved from `previous` to `next` by at most
/// `10⁻⁶×next + ε` (with the machine epsilon `ε` of `T`).
fn is_settled<T>(previous: T, next: T) -> bool
where
    T: MatrixDataType,
{
    let relative = T::from_f64(RELATIVE_TOLERANCE).unwrap_or_else(T::zero);
    abs(next - previous) <= relative * next + machine_epsilon::<T>()
}

/// Multiplies every entry of `A` by `factor`.
#[allow(non_snake_case)]
pub(crate) fn scale<const R: usize, const C: usize, T>(A: &mut impl MatrixMut<R, C, T>, factor: T)
where
    T: MatrixDataType,
{
    for i in 0..R {
        for j in 0..C {
            A.set(i, j, A.get(i, j) * factor);
        }
    }
}

/// Returns the larger of `a` and `b`.
fn max<T>(a: T, b: T) -> T
where
    T: MatrixDataType,
{
    if b > a { b } else { a }
}

/// Returns `|value|`.
fn abs<T>(value: T) -> T
where
    T: MatrixDataType,
{
    if value < T::zero() { -value } else { value }
}

/// Performs the robust update of a linear (or linearized) observation from the prior
/// innovation `y` (see the module docs).
///
/// Writes the weight of the applied update to `weight`. Leaves `x`, `P`, `K` and `weight`
/// unchanged unless the update is applied; see [`solve_weight`] for the other statuses.
///
/// Updates `P` as `covariance_update` says. The Joseph form is evaluated with the unscaled
/// gain `K̃ = P⁻×Hᵀ×S_w⁻¹ = K/w` as
/// `P − w×(K̃×H×P + (K̃×H×P)ᵀ) + w×K̃×S_w×K̃ᵀ`, which equals
/// `P − K×H×P − (K×H×P)ᵀ + K×(H×P×Hᵀ + R/w)×Kᵀ` without forming `R/w`.
///
/// On return, whatever the status, `S` holds `G + R = H×P⁻×Hᵀ + R` of the prior, as after the
/// plain correction; the iteration's factors only live in it in between.
///
/// ## Buffers
/// All buffers are distinct. `G` is the caller's scratch; `temp_PHt` keeps `P⁻×Hᵀ` while the
/// weight is iterated, and `temp_HP` and `temp_KHP` are only written once `K` is formed.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_linear<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    G: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    nis: &mut T,
    weight: &mut T,
    gate: Option<T>,
    robust: RobustCorrection<T>,
    covariance_update: CovarianceUpdate,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // G = H*P*Hᵀ, with temp_PHt = P*Hᵀ kept for the gain.
    P.mult_transb(H, temp_PHt);
    H.mult(temp_PHt, G);

    let solution = match solve_weight(&*G, R, y, S, S_inv, nis, gate, robust) {
        Ok(solution) => solution,
        Err(status) => {
            write_scaled_covariance(G, R, T::one(), S);
            return status;
        }
    };

    // K̃ = P*Hᵀ*S_w⁻¹; the gain is K = w*K̃.
    let w = solution.weight;
    temp_PHt.mult(S_inv, K);
    H.mult(P, temp_HP);
    K.mult(temp_HP, temp_KHP); // temp_KHP = K̃*H*P
    match covariance_update {
        CovarianceUpdate::Standard => {
            scale(temp_KHP, w);
            P.sub_inplace_a(temp_KHP);
        }
        CovarianceUpdate::Joseph => {
            subtract_scaled_with_transpose(P, temp_KHP, w);
            write_scaled_covariance(G, R, w, S);
            // temp_PHt is only written once temp_KHP is consumed.
            K.mult(S, temp_PHt); // temp_PHt = K̃*S_w
            add_scaled_lower_product_mirrored(P, temp_PHt, K, w);
        }
    }

    scale(K, w);
    K.multadd_rowvector(y, x);
    *weight = w;
    write_scaled_covariance(G, R, T::one(), S);
    solution.status()
}

/// Writes `S = w×G + R`.
#[allow(non_snake_case)]
fn write_scaled_covariance<const M: usize, T>(
    G: &impl Matrix<M, M, T>,
    R: &impl Matrix<M, M, T>,
    w: T,
    S: &mut impl MatrixMut<M, M, T>,
) where
    T: MatrixDataType,
{
    for i in 0..M {
        for j in 0..M {
            S.set(i, j, w * G.get(i, j) + R.get(i, j));
        }
    }
}

/// Applies `P ← P − w×(A + Aᵀ)`.
#[allow(non_snake_case)]
fn subtract_scaled_with_transpose<const STATES: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    A: &impl Matrix<STATES, STATES, T>,
    w: T,
) where
    T: MatrixDataType,
{
    for i in 0..STATES {
        for j in 0..STATES {
            let value = P.get(i, j) - w * (A.get(i, j) + A.get(j, i));
            P.set(i, j, value);
        }
    }
}

/// Adds `w×A×Bᵀ` to `P`, computing only the lower triangle and mirroring it, so that `P` comes
/// out exactly symmetric.
#[allow(non_snake_case)]
fn add_scaled_lower_product_mirrored<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    A: &impl Matrix<STATES, OBSERVATIONS, T>,
    B: &impl Matrix<STATES, OBSERVATIONS, T>,
    w: T,
) where
    T: MatrixDataType,
{
    for i in 0..STATES {
        for j in 0..=i {
            let product: T = (0..OBSERVATIONS).map(|l| A.get(i, l) * B.get(j, l)).sum();
            let value = P.get(i, j) + w * product;
            P.set(i, j, value);
            P.set(j, i, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;

    /// Runs [`solve_weight`] on row-major `2×2` `G`, `R` and the innovation `y`, without a gate.
    fn solve(
        g: [f64; 4],
        r: [f64; 4],
        y: [f64; 2],
        dof: f64,
        iterations: usize,
    ) -> WeightSolution<f64> {
        let (mut s, mut s_inv, mut nis) = ([0.0; 4], [0.0; 4], 0.0);
        let robust = RobustCorrection {
            weight: RobustWeight::StudentT { dof },
            iterations,
        };
        solve_weight(
            &MatrixData::new_ref::<2, 2, f64>(&g),
            &MatrixData::new_ref::<2, 2, f64>(&r),
            &MatrixData::new_ref::<2, 1, f64>(&y),
            &mut MatrixData::new_mut::<2, 2, f64>(&mut s),
            &mut MatrixData::new_mut::<2, 2, f64>(&mut s_inv),
            &mut nis,
            None,
            robust,
        )
        .expect("positive definite")
    }

    /// The Student-t weight of Piché et al., Algorithm 1, iterated on the full state for the
    /// `2×2` problem `P = G`, `H = I`, written out with explicit inverses.
    fn full_state_weight(g: [f64; 4], r: [f64; 4], y: [f64; 2], dof: f64, rounds: usize) -> f64 {
        let inv = |a: [f64; 4]| {
            let det = a[0] * a[3] - a[1] * a[2];
            [a[3] / det, -a[1] / det, -a[2] / det, a[0] / det]
        };
        let mul = |a: [f64; 4], b: [f64; 4]| {
            [
                a[0] * b[0] + a[1] * b[2],
                a[0] * b[1] + a[1] * b[3],
                a[2] * b[0] + a[3] * b[2],
                a[2] * b[1] + a[3] * b[3],
            ]
        };

        let mut w = 1.0;
        for _ in 0..rounds {
            let s = [
                g[0] + r[0] / w,
                g[1] + r[1] / w,
                g[2] + r[2] / w,
                g[3] + r[3] / w,
            ];
            let k = mul(g, inv(s));
            let x = [k[0] * y[0] + k[1] * y[1], k[2] * y[0] + k[3] * y[1]];
            let kg = mul(k, g);
            let p = [g[0] - kg[0], g[1] - kg[1], g[2] - kg[2], g[3] - kg[3]];
            let e = [y[0] - x[0], y[1] - x[1]];
            let d = [
                e[0] * e[0] + p[0],
                e[0] * e[1] + p[1],
                e[1] * e[0] + p[2],
                e[1] * e[1] + p[3],
            ];
            let dr = mul(d, inv(r));
            w = (dof + 2.0) / (dof + dr[0] + dr[3]);
        }

        w
    }

    /// The measurement-space iteration reproduces the full-state iteration of the paper (up to
    /// the early exit once the weight has settled).
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))] // micromath's square root is approximate
    fn matches_full_state_iteration() {
        let (g, r) = ([2.0, 0.3, 0.3, 1.0], [0.5, 0.1, 0.1, 0.8]);
        for y in [[0.0, 0.0], [1.0, -0.5], [10.0, 3.0], [300.0, -200.0]] {
            for rounds in 1..6 {
                let solution = solve(g, r, y, 4.0, rounds);
                let reference = full_state_weight(g, r, y, 4.0, rounds);
                assert!(
                    (solution.weight - reference).abs() <= 1e-9 * reference,
                    "y = {y:?}, rounds = {rounds}: {} vs {reference}",
                    solution.weight
                );
            }
        }
    }

    /// The iteration settles within a small budget, and a budget of zero applies `w = 1`
    /// without claiming convergence.
    #[test]
    fn converges_within_a_small_budget() {
        let (g, r) = ([2.0, 0.3, 0.3, 1.0], [0.5, 0.1, 0.1, 0.8]);
        for y in [[0.0, 0.0], [3.0, 0.0], [30.0, 0.0], [3000.0, 0.0]] {
            let solution = solve(g, r, y, 4.0, 20);
            assert!(solution.converged, "y = {y:?}");
            assert!(solution.weight > 0.0 && solution.weight <= 6.0 / 4.0);
        }

        let solution = solve(g, r, [30.0, 0.0], 4.0, 0);
        assert_eq!(solution.weight, 1.0);
        assert!(!solution.converged);
    }

    /// In `I16F16`, the prior NIS of this 250σ outlier (`12500`) fits, but `bᵀ×R×b` tends to
    /// `y² = 62500` as `w → 0` and would overflow; the scaled evaluation keeps the weight finite
    /// and close to the `f64` value.
    #[test]
    #[cfg(feature = "fixed")]
    fn huge_outliers_do_not_overflow_in_fixed_point() {
        use fixed::types::I16F16;

        let to_fixed = |a: &[f64]| {
            let mut out = [I16F16::ZERO; 4];
            for (o, &v) in out.iter_mut().zip(a) {
                *o = I16F16::from_num(v);
            }
            out
        };
        let (g, r, y) = ([4.0, 0.0, 0.0, 4.0], [1.0, 0.0, 0.0, 1.0], [250.0, 0.0]);
        let (gf, rf, yf) = (to_fixed(&g), to_fixed(&r), to_fixed(&y));
        let (mut s, mut s_inv, mut nis) = ([I16F16::ZERO; 4], [I16F16::ZERO; 4], I16F16::ZERO);
        let robust = RobustCorrection {
            weight: RobustWeight::StudentT {
                dof: I16F16::from_num(4),
            },
            iterations: 20,
        };

        let solution = solve_weight(
            &MatrixData::new_ref::<2, 2, I16F16>(&gf),
            &MatrixData::new_ref::<2, 2, I16F16>(&rf),
            &MatrixData::new_ref::<2, 1, I16F16>(&yf[..2]),
            &mut MatrixData::new_mut::<2, 2, I16F16>(&mut s),
            &mut MatrixData::new_mut::<2, 2, I16F16>(&mut s_inv),
            &mut nis,
            None,
            robust,
        )
        .expect("positive definite");

        let reference = solve(g, r, y, 4.0, 20).weight;
        let weight: f64 = solution.weight.to_num();
        assert!(
            weight > 0.0 && (weight - reference).abs() < 1e-3,
            "{weight} vs {reference}"
        );
    }

    /// A regular `I16F16` filter takes a 100σ outlier without overflowing: the state barely
    /// moves, the weight is small and `P` stays positive and no larger than the prior.
    #[test]
    #[cfg(all(feature = "fixed", feature = "alloc"))]
    fn fixed_point_filter_survives_a_huge_outlier() {
        use crate::prelude::*;
        use crate::regular::builder::KalmanFilterBuilder;

        let builder = KalmanFilterBuilder::<2, I16F16>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<2>();
        let mut workspace = BufferBuilder::temp_S_inv::<2>().new::<I16F16>();
        let one = I16F16::from_num(1);

        filter
            .estimate_covariance_mut()
            .as_matrix_mut()
            .make_identity();
        measurement
            .observation_matrix_mut()
            .as_matrix_mut()
            .make_identity();
        measurement
            .measurement_noise_covariance_mut()
            .as_matrix_mut()
            .make_identity();
        measurement
            .measurement_vector_mut()
            .as_matrix_mut()
            .set(0, 0, I16F16::from_num(100));

        let robust = RobustCorrection {
            weight: RobustWeight::StudentT {
                dof: I16F16::from_num(4),
            },
            iterations: 20,
        };
        let status = filter.correct_robust(&mut measurement, &mut workspace, robust);

        assert!(status.is_applied());
        assert!(measurement.robust_weight() < I16F16::from_num(0.01));
        let x = filter.state_vector().as_matrix().get(0, 0);
        assert!(x > I16F16::ZERO && x < one, "x = {x}");
        let p = filter.estimate_covariance().as_matrix().get(0, 0);
        assert!(p > I16F16::from_num(0.9) && p <= one, "P = {p}");
    }

    /// Regression: with a tiny `R` in `I16F16`, `b = S_w⁻¹×y ≈ R⁻¹×y ≈ 35000` overflowed
    /// once the weight dropped, although the prior NIS (about `28000`) still fit. `y` is scaled
    /// before `b` is formed now.
    #[test]
    #[cfg(feature = "fixed")]
    fn tiny_noise_does_not_overflow_b_in_fixed_point() {
        use fixed::types::I16F16;

        let tiny = I16F16::from_bits(3); // 3 ULP, about 4.6e-5
        let mut g = [I16F16::ZERO; 4];
        g[0] = tiny;
        g[3] = tiny;
        let r = g;
        let y = [I16F16::from_num(1.6), I16F16::ZERO];
        let (mut s, mut s_inv, mut nis) = ([I16F16::ZERO; 4], [I16F16::ZERO; 4], I16F16::ZERO);
        let robust = RobustCorrection {
            weight: RobustWeight::StudentT {
                dof: I16F16::from_num(4),
            },
            iterations: 20,
        };

        let solution = solve_weight(
            &MatrixData::new_ref::<2, 2, I16F16>(&g),
            &MatrixData::new_ref::<2, 2, I16F16>(&r),
            &MatrixData::new_ref::<2, 1, I16F16>(&y),
            &mut MatrixData::new_mut::<2, 2, I16F16>(&mut s),
            &mut MatrixData::new_mut::<2, 2, I16F16>(&mut s_inv),
            &mut nis,
            None,
            robust,
        )
        .expect("positive definite");

        assert!(nis > I16F16::from_num(20_000), "NIS = {nis}");
        assert!(solution.weight >= I16F16::ZERO && solution.weight < I16F16::from_num(0.001));
    }
}
