//! # Observations for Regular, Extended, and Unscented Kalman Filters.

use crate::kalman::CorrectionStatus;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

pub mod extended;
pub(crate) mod predictively_oriented;
pub mod regular;
pub(crate) mod robust;
pub mod unscented;

/// Online adaptation of the measurement noise covariance `R` from the residuals of the
/// corrections, enabled per observation with `set_noise_adaptation`.
///
/// After every applied regular or extended correction with the Kalman gain (`correct`,
/// `correct_with`, `correct_nonlinear` and `correct_nonlinear_with`), `R` is replaced by an
/// exponentially weighted average
///
/// ```text
/// ε = z − H×x⁺ = y − H×K×y
/// R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)
/// ```
///
/// where `α` is the [forgetting factor](Self::forgetting), `x⁺` and `P⁺` are the corrected
/// state and covariance, and `ε` is the residual after the correction. Afterwards, every
/// diagonal entry of `R` is raised to at least the [floor](Self::floor). No history window
/// and no extra buffer is needed; the existing temporaries are reused. Corrections that are
/// [gated](crate::prelude::CorrectionStatus::Gated),
/// [not positive definite](crate::prelude::CorrectionStatus::NotPositiveDefinite) or
/// [backed out](crate::prelude::CorrectionStatus::BackedOut) leave `R` unchanged.
///
/// ## Other corrections
///
/// The residual form above needs the optimal gain (see [below](#why-the-other-corrections-differ)),
/// so the corrections with an externally supplied gain (`correct_with_gain*` and
/// `correct_nonlinear_with_gain*`) and the recalibrated extended corrections
/// (`correct_nonlinear_recalibrated*`) use the innovation form instead:
///
/// ```text
/// R ← α×R + (1−α)×(y×yᵀ − H×P⁻×Hᵀ) = R + (1−α)×(y×yᵀ − S)
/// ```
///
/// with the prior innovation `y = z − H×x⁻` and `S = H×P⁻×Hᵀ + R`. For the recalibrated
/// correction, `H` is the Jacobian at the updated state. The diagonal is floored the same way.
/// Because this form subtracts, the candidate is not positive definite by construction; if the
/// floored candidate is not, `R` is left unchanged for that correction. The gain corrections
/// have no gate, so every one of them adapts `R`.
///
/// The predictively oriented correction (`correct_predictively_oriented`) and the robust
/// corrections (`correct_robust*`, `correct_nonlinear_robust*`, `correct_sigma_point_robust*`)
/// do not adapt `R`; the latter use the reweighted `R/w`, whose residuals would bias it.
///
/// ## Background
///
/// This is the residual-based covariance matching of Akhlaghi, Zhou and Huang, "Adaptive
/// Adjustment of Noise Covariance in Kalman Filter for Dynamic State Estimation"
/// (arXiv:1702.00884, 2017), eq. (11), which replaces a moving window by the forgetting
/// factor. The paper prints the prior covariance `P⁻` in that equation; this implementation
/// uses the corrected covariance `P⁺` (see also Mohamed and Schwarz, "Adaptive Kalman
/// Filtering for INS/GPS", 1999), because only then is the true `R` a fixed point: with
/// `ε = R×S⁻¹×y` one gets `E[ε×εᵀ] = R×S⁻¹×R` and `H×P⁺×Hᵀ = R − R×S⁻¹×R`, which sum to
/// `R`. With `P⁻`, the estimate is biased upwards by `H×K×S×Kᵀ×Hᵀ`.
///
/// `α×R` is positive definite and both added terms are positive semi-definite, so the
/// update keeps `R` positive definite in exact arithmetic for `0 < α < 1`. The floor guards
/// against rounding (e.g. `P⁺` losing definiteness, or fixed-point types); it bounds the
/// diagonal only.
///
/// For the extended filter, `ε` is the linearized residual `y − H×K×y` with the Jacobian
/// `H`, not `z − h(x⁺)`; `h` is not evaluated a second time.
///
/// ### Why the other corrections differ
///
/// For any gain `K`, with `P⁺` from the Joseph form and `ε = H×e⁺ + v` (`e⁺` the error of
/// `x⁺`, `v` the measurement noise, `E[e⁺×vᵀ] = −K×R`),
///
/// ```text
/// E[ε×εᵀ] + H×P⁺×Hᵀ = R + 2×H×P⁺×Hᵀ − H×K×R − R×Kᵀ×Hᵀ.
/// ```
///
/// The extra terms cancel only if `P⁺×Hᵀ = K×R`. With the Joseph form, a scalar observation
/// gives `p⁺ = (1−k)²×p + k²×r`, so the condition reads `(1−k)×[(1−k)×p − k×r] = 0`. It holds
/// for the optimal gain `k = p/(p+r)`, i.e. `K = P⁻×Hᵀ×S⁻¹`, and for the degenerate `k = 1`,
/// where `ε = 0` and `R` stops adapting. For any other gain, the residual form converges to a
/// biased `R`.
///
/// - **Supplied gain:** `K` is arbitrary, so the residual form is biased. The prior
///   innovation does not depend on the gain, `E[y×yᵀ] = H×P⁻×Hᵀ + R`, so the innovation form
///   is unbiased for any gain, as long as `P⁻` is the true prior covariance. The Joseph form
///   keeps it so for any gain, given the true `R`.
/// - **Recalibrated:** `K` is optimal for the Jacobian `H₁` at the prior, but `P⁺` is
///   recalibrated with the Jacobian `H₂` at `x⁺`. In the `H₂` linearization the gain is not
///   optimal. For linear observations (`H₂ = H₁`) both forms are exact; otherwise both are
///   biased at first order in `H₂ − H₁` (the residual form through the terms above, the
///   innovation form through `S₂ − S₁`). The innovation form is used because it does not
///   depend on the gain and came closer in simulations: for `h(x) = x²` with a prior spread
///   that is large against `R`, the residual form converged to `3` to `45` times the true
///   `R` and the innovation form to `1.3` to `9` times it, against `1.1` to `2.4` times for
///   `correct_nonlinear`, with many corrections backed out. Where `H×P×Hᵀ ≲ R` and few
///   corrections are backed out, all of them stayed within a few percent of the true `R`.
/// - **Predictively oriented:** the PrO posterior `P` is deliberately inflated and is not the
///   error covariance of `x⁺`, so neither form holds. In closed-loop simulations of a scalar
///   random walk with `Q/R` between `0.01` and `10`, the residual form converged to `1` to `25`
///   times the true `R`, and the innovation form to `0.5` to `0.9` times it, because the inflated
///   `P⁻` overstates `H×P⁻×Hᵀ`. The correction therefore skips the adaptation.
///
/// ## Choosing α and when the estimate diverges
///
/// For a single measurement, the residual form reduces to
/// `R ← R × (1 + (1−α) × (R/S) × (NIS − 1))`, so `R` grows while the
/// [NIS](crate::regular::RegularObservation::nis) is above `1` and shrinks while it is below.
/// The innovation form reduces to `R ← R × (1 + (1−α) × (S/R) × (NIS − 1))`: it reacts more
/// strongly while `H×P×Hᵀ` dominates `R`, where a single small innovation can push `R` down
/// to the floor. Use `α` close to `1` with it.
///
/// - `α` close to `1` (e.g. `0.95` to `0.99`) averages over roughly `1 / (1−α)` corrections
///   and gives a smooth estimate; a small `α` lets single residuals dominate and `R`
///   fluctuates strongly. `α` must lie in `(0, 1)`: `α = 1` freezes `R`, `α > 1` reverses
///   the correction and drives `R` away from the residuals, `α = 0` discards the previous
///   `R`, and `α < 0` subtracts it; outside `(0, 1)`, only the floor keeps the diagonal
///   positive.
/// - With the residual form, `R = 0` is absorbing, and a tiny floor nearly so: at the floor,
///   `R` grows by only `(1−α) × (floor/S) × (NIS − 1)` per correction. Set the floor near the
///   smallest plausible sensor variance.
/// - While `H×P×Hᵀ` dominates `R` (an uncertain or unobservable state), `R/S` is small and
///   the residual form barely adapts. With unobservable states, `P` keeps growing and `R`
///   stalls.
/// - Every excess of the NIS is attributed to `R`, including a process noise `Q` that is too
///   small or a model error. `R` then grows without bound to make up for it, and the filter
///   ignores the measurements; tune `Q` first.
/// - A [gate](crate::regular::RegularObservation::set_gate) rejects corrections before they
///   reach the adaptation. After a real step up in the measurement noise, a tight gate
///   rejects most readings and `R` never catches up; widen or disable the gate when
///   adapting.
/// - With fixed-point types, `ε×εᵀ` can overflow for large residuals, like the NIS.
///
/// ## Unscented observations
///
/// The unscented observation adapts `R` after every applied `correct_sigma_point*`
/// correction, including the recalibrated one. Its predicted measurement covariance `Pzz`
/// takes the place of `H×P⁻×Hᵀ`, and `H×K = Pzz×S⁻¹`. After the gain, only `S = Pzz + R`,
/// `S⁻¹` and `y` are left, so both terms of the residual form above are written in terms of
/// them with `Pzz = S − R`:
///
/// ```text
/// ε = y − H×K×y = y − Pzz×S⁻¹×y = R×S⁻¹×y
/// H×P⁺×Hᵀ = Pzz − Pzz×S⁻¹×Pzz = R − R×S⁻¹×R
/// ```
///
/// Substituting them into `R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)` gives
///
/// ```text
/// ε = R×S⁻¹×y
/// R ← R + (1−α)×(ε×εᵀ − R×S⁻¹×R)
/// ```
///
/// which is the same update for a linear observation, so the true `R` stays a fixed point.
/// The recalibrated correction adapts `R` from the first pass, i.e. the `y` and `S` at the
/// prior: the second pass only recalibrates `P`, and `H×P⁺×Hᵀ` has no closed form in its
/// sigma points. With a measurement residual function, `ε` is computed from the normalized
/// `y`. The update overwrites the inverted innovation covariance temporary.
///
/// ## Process noise `Q`
///
/// The regular, extended and unscented filters adapt their process noise covariance `Q`
/// with the same settings, enabled with their `set_process_noise_adaptation`. After every
/// applied `correct`, `correct_with`, `correct_nonlinear`, `correct_nonlinear_with`,
/// `correct_sigma_point*` or robust (`correct_robust*`, `correct_nonlinear_robust*`,
/// `correct_sigma_point_robust*`) correction,
///
/// ```text
/// Δx = x⁺ − x⁻ = K×y
/// Q ← α×Q + (1−α)×Δx×Δxᵀ
/// ```
///
/// then every diagonal entry of `Q` is raised to at least the floor. The external gain,
/// predictively oriented and extended recalibrated corrections do not adapt `Q`, and neither
/// does the information filter.
///
/// ### Background
///
/// This is the innovation-based estimate of Mohamed and Schwarz, "Adaptive Kalman Filtering
/// for INS/GPS" (Journal of Geodesy 73, 1999), `Q̂ = K×Ĉᵥ×Kᵀ` with the innovation covariance
/// `Ĉᵥ` estimated from the innovations `y`. Akhlaghi, Zhou and Huang, "Adaptive Adjustment of
/// Noise Covariance in Kalman Filter for Dynamic State Estimation" (arXiv:1702.00884, 2017),
/// replace the moving window by the forgetting factor `α`, as for `R` above.
///
/// In expectation, `Δx×Δxᵀ` is `K×S×Kᵀ = P⁻ − P⁺`, the covariance the correction removes. The
/// true `Q` is a fixed point only where the steady state has `A×P⁺×Aᵀ = P⁺`, e.g. a random
/// walk (`A = I`). Otherwise the estimate is biased. Once the gain `K` has settled, every
/// `Δx` points along `K`, so with a single measurement the estimate has rank one along `K`
/// rather than the shape of the true `Q`; e.g. for a constant velocity model with a position
/// sensor, the noise ends up mostly on the position instead of the velocity. The filter can
/// still be consistent, i.e. its NIS averages to the number of measurements. The full Sage-Husa estimate adds
/// `P⁺ − A×P⁺ₖ₋₁×Aᵀ` to remove this bias, but that term is indefinite and would need an
/// eigenvalue projection to keep `Q` positive semi-definite, so it is not used.
///
/// ### Keeping `Q` symmetric positive semi-definite
///
/// No projection is needed: for `0 ≤ α ≤ 1`, `α×Q` is positive semi-definite (PSD) and
/// `Δx×Δxᵀ` is a PSD rank-one term, so their sum is PSD. Only the lower triangle is computed
/// and mirrored, so `Q` stays exactly symmetric under rounding. Raising a diagonal entry adds
/// a non-negative diagonal matrix, which keeps `Q` PSD; the floor may be `0`.
///
/// ### Limits
///
/// - `Δx` lies in the span of the gain `K`, so `Q` only learns the directions the
///   corrections move the state in. In all other directions, `Q` decays by `α` per
///   correction towards the floor.
/// - Every correction adds one `Δx`. With several observations per prediction, each takes
///   part of the correction, and `Q` comes out too small.
/// - Unlike `R = 0`, `Q = 0` is not absorbing: the first non-zero correction makes it
///   `(1−α)×Δx×Δxᵀ`. A `Q` that is far too small still shrinks `P` and the gain, and with
///   them `Δx`, so it can take many corrections to recover; raise the floor if that matters.
/// - Adapting `R` at the same time is poorly identifiable: both explain the same innovations.
///   Adapt one of them at a time, or keep `α` for `Q` closer to `1`.
/// - With fixed-point types, `Δx×Δxᵀ` can overflow for large corrections, like `ε×εᵀ`
///   for `R`.
/// - Gated, not positive definite and backed-out corrections leave `Q` unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseAdaptation<T> {
    /// The forgetting factor `α` in `(0, 1)`: the weight of the previous `R` or `Q`. The new
    /// residual or state correction is weighted with `1 − α`.
    pub forgetting: T,

    /// Lower bound for every diagonal entry of the adapted `R` or `Q`. Should be positive for
    /// `R`; may be `0` for `Q`.
    pub floor: T,
}

/// How the regular and extended corrections with the Kalman gain update the estimate
/// covariance `P`.
///
/// Set per observation with `set_covariance_update` on
/// [`RegularObservation`](crate::regular::RegularObservation) and
/// [`ExtendedObservation`](crate::extended::ExtendedObservation). Applies to `correct`,
/// `correct_with`, `correct_nonlinear` and `correct_nonlinear_with`, and to the robust
/// corrections `correct_robust*` and `correct_nonlinear_robust*` (with the reweighted noise
/// `R/w` in place of `R`). The corrections with an
/// externally supplied gain (`correct_with_gain*`) and the recalibrated extended corrections
/// always use the Joseph form; the predictively oriented corrections write `P` their own way
/// and ignore this setting.
///
/// ## Cost
///
/// [`Joseph`](Self::Joseph) needs no extra buffer. For `N` states and `m` measurements it adds
/// about `2×N×m² + N²×m/2` multiply-adds per correction: it forms `S = H×P×Hᵀ + R` again
/// (after the gain, the `S` buffer holds its Cholesky factor), then `K×S` and the lower
/// triangle of `K×S×Kᵀ`.
///
/// ## What it buys
///
/// Measured with the `numerical_robustness` test bench of this crate:
///
/// - **Symmetry.** The Joseph form leaves `P` exactly symmetric in every type. The standard
///   update leaves a relative asymmetry of about `10⁻⁵` in `f32` and `10⁻³` to `0.3` in
///   `I16F16`, which the prediction carries forward.
/// - **Positive definiteness, ill-conditioned measurements.** For the textbook case
///   `H = [[1,1,1],[1,1,1+δ]]`, `R = δ²×I`, `P = I`, the standard update left `P` indefinite
///   for 16 of 48 values of `δ` in `f32` (from `δ ≈ 9×10⁻³`) and 11 in `f64` (from
///   `δ ≈ 10⁻⁴`); the Joseph form for 7 and 9 (from `δ ≈ 7×10⁻³` and `4×10⁻⁵`). In `I32F32`,
///   it was the other way round: 0 for the standard update, 1 for the Joseph form. It helps
///   in floating point, but it does not fix the case.
/// - **Fixed point.** In `I16F16`, neither form helps once the variances approach the
///   resolution (`2⁻¹⁶`): with ten scalar sensors and a steady-state variance below one LSB,
///   the standard update left `P` indefinite in 16 of 1000 steps, and the Joseph form
///   overflowed while computing the gain. Use a type with more fractional bits, such as `I32F32`, there.
///
/// On well-conditioned problems, both forms agree to a few ULP in `f64`, given an exact square
/// root; with `micromath`'s approximate one, the gain is not quite optimal and they differ more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CovarianceUpdate {
    /// The short form `P ← P − K×H×P`. Cheapest, and exact only for the optimal gain in exact
    /// arithmetic: rounding can make `P` asymmetric or, once `K×H×P` is close to `P`,
    /// indefinite. Opt in where the extra products of the Joseph form do not fit the budget.
    Standard,
    /// The Joseph form (the default) `P ← (I − K×H)×P×(I − K×H)ᵀ + K×R×Kᵀ`, evaluated as
    /// `P − K×H×P − (K×H×P)ᵀ + K×S×Kᵀ`. Only the lower triangle is formed and mirrored, so
    /// `P` comes out exactly symmetric. Positive semi-definite for any gain in exact
    /// arithmetic. After the correction, the observation's `S` holds `H×P⁻×Hᵀ + R` instead of
    /// its Cholesky factor.
    #[default]
    Joseph,
}

/// Computes the normalized innovation squared `NIS = yᵀ×S⁻¹×y`.
#[allow(non_snake_case)]
pub(crate) fn normalized_innovation_squared<const OBSERVATIONS: usize, T>(
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    S_inv: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
) -> T
where
    T: MatrixDataType,
{
    let mut nis = T::zero();
    for i in 0..OBSERVATIONS {
        let mut row = T::zero();
        for j in 0..OBSERVATIONS {
            row += S_inv.get(i, j) * y.get(j, 0);
        }
        nis += y.get(i, 0) * row;
    }
    nis
}

/// Performs the update step of the linear and extended observations from an innovation `y`
/// that is already computed, either as `y = z - H×x` or as `y = z - h(x)`.
///
/// Stores the normalized innovation squared in `nis` whenever `S` can be inverted.
/// Leaves `x` and `P` unchanged if the innovation covariance `S = H×P×Hᵀ + R` is not
/// positive definite, or if `nis` exceeds `gate`. Updates `P` as `covariance_update` says;
/// see [`CovarianceUpdate`]. After an applied correction, adapts `R` if `adaptation` is set;
/// see [`NoiseAdaptation`].
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_with_innovation<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    nis: &mut T,
    gate: Option<T>,
    adaptation: Option<NoiseAdaptation<T>>,
    covariance_update: CovarianceUpdate,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    let status = compute_kalman_gain(P, H, R, y, K, S, S_inv, temp_HP, temp_PHt, nis, gate);
    if !status.is_applied() {
        return status;
    }

    //* Correct state prediction
    //* x = x + K*y

    // x = x + K*y
    K.multadd_rowvector(y, x);

    match covariance_update {
        CovarianceUpdate::Standard => {
            //* Correct state covariances
            //* P = (I-K*H) * P
            //*   = P - K*(H*P)

            // P = P - K*(H*P)
            H.mult(P, temp_HP); // temp_HP = H*P
            K.mult(temp_HP, temp_KHP); // temp_KHP = K*temp_HP
            P.sub_inplace_a(temp_KHP); // P -= temp_KHP
        }
        CovarianceUpdate::Joseph => {
            // S holds its Cholesky factor by now, so Joseph forms S again.
            joseph_update(P, H, R, K, S, temp_HP, temp_PHt, temp_KHP);
        }
    }

    if let Some(adaptation) = adaptation {
        // S_inv and temp_PHt are the only temporaries guaranteed not to alias each other.
        adapt_measurement_noise(adaptation, P, H, R, y, K, S_inv, temp_PHt);
    }

    CorrectionStatus::Applied
}

/// Performs the update step of the linear and extended observations with an externally
/// supplied Kalman gain `K`, from an innovation `y` that is already computed.
///
/// Applies `x = x + K×y` and updates `P` with the Joseph form
/// `P = (I - K×H)×P×(I - K×H)ᵀ + K×R×Kᵀ`, expanded as
/// `P = P - K×H×P - (K×H×P)ᵀ + K×S×Kᵀ` with `S = H×P×Hᵀ + R`, so that `P` stays exactly
/// symmetric, and positive semi-definite for any `K` in exact arithmetic. `S` is not inverted and is left holding
/// `H×P×Hᵀ + R` of the prior `P` and the `R` used in the update. Afterwards, adapts `R`
/// with the innovation form if `adaptation` is set; see [`NoiseAdaptation`].
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_with_gain<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    adaptation: Option<NoiseAdaptation<T>>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // x = x + K*y; x does not enter the covariance update.
    K.multadd_rowvector(y, x);

    joseph_update(P, H, R, K, S, temp_HP, temp_PHt, temp_KHP);

    if let Some(adaptation) = adaptation {
        // S_inv may alias temp_HP and temp_KHP, both of which are consumed by now.
        adapt_measurement_noise_from_innovation(adaptation, R, y, S, S_inv);
    }

    CorrectionStatus::Applied
}

/// Updates `P` with the Joseph form `P = P - K×H×P - (K×H×P)ᵀ + K×S×Kᵀ` for any gain `K`,
/// leaving `S = H×P×Hᵀ + R` of the prior `P` in `S`.
///
/// `temp_HP` must not alias `temp_KHP`; `temp_PHt` may alias either, since it is only written
/// once both are consumed.
#[allow(non_snake_case, clippy::too_many_arguments)]
fn joseph_update<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
) where
    T: MatrixDataType,
{
    // S = H*P*H' + R, from the prior P
    H.mult(P, temp_HP); // temp_HP = H*P
    temp_HP.mult_transb(H, S); // S = temp_HP*Hᵀ
    S.add_inplace_a(R); // S += R

    // P = P - K*H*P - (K*H*P)ᵀ
    K.mult(temp_HP, temp_KHP); // temp_KHP = K*temp_HP
    subtract_with_transpose(P, temp_KHP);

    // P = P + K*S*Kᵀ; temp_PHt MAY alias temp_KHP, so it is only written once temp_KHP is consumed.
    K.mult(S, temp_PHt); // temp_PHt = K*S
    add_lower_product_mirrored(P, temp_PHt, K); // P += temp_PHt*Kᵀ
}

/// Adds `A×Bᵀ` to `P`, computing only the lower triangle of the sum and mirroring it into
/// the upper one, so that `P` comes out exactly symmetric.
///
/// For `A = K×S` and `B = K`, `A×Bᵀ = K×S×Kᵀ` is symmetric in exact arithmetic, but not under
/// rounding: `(K×S)ᵢₗ×Kⱼₗ` and `(K×S)ⱼₗ×Kᵢₗ` round differently.
#[allow(non_snake_case)]
fn add_lower_product_mirrored<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    A: &impl Matrix<STATES, OBSERVATIONS, T>,
    B: &impl Matrix<STATES, OBSERVATIONS, T>,
) where
    T: MatrixDataType,
{
    for i in 0..STATES {
        for j in 0..=i {
            let product: T = (0..OBSERVATIONS).map(|l| A.get(i, l) * B.get(j, l)).sum();
            let value = P.get(i, j) + product;
            P.set(i, j, value);
            P.set(j, i, value);
        }
    }
}

/// Adapts the measurement noise covariance `R` with the innovation form
/// `R ← R + (1−α)×(y×yᵀ − S)`, raising every diagonal entry to at least the floor. See
/// [`NoiseAdaptation`].
///
/// `y` must be the prior innovation and `S = H×P⁻×Hᵀ + R` its covariance, formed with the
/// current `R`. The candidate is checked with a Cholesky decomposition in `temp`; if it is
/// not positive definite, `R` is left unchanged. `R` stays symmetric.
#[allow(non_snake_case)]
pub(crate) fn adapt_measurement_noise_from_innovation<const OBSERVATIONS: usize, T>(
    adaptation: NoiseAdaptation<T>,
    R: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    S: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    temp: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
) where
    T: MatrixDataType,
{
    for i in 0..OBSERVATIONS {
        for j in 0..=i {
            let value = innovation_matched_entry(adaptation, &*R, y, S, i, j);
            temp.set(i, j, value);
            temp.set(j, i, value);
        }
    }

    if !temp.cholesky_decompose_lower() {
        return;
    }

    // Only the lower triangle is read, so mirroring into the upper one is safe in place.
    for i in 0..OBSERVATIONS {
        for j in 0..=i {
            let value = innovation_matched_entry(adaptation, &*R, y, S, i, j);
            R.set(i, j, value);
            R.set(j, i, value);
        }
    }
}

/// Returns the entry `(i, j)` of the innovation-form candidate `R + (1−α)×(y×yᵀ − S)`, with
/// diagonal entries raised to at least the floor.
#[allow(non_snake_case)]
fn innovation_matched_entry<const OBSERVATIONS: usize, T>(
    adaptation: NoiseAdaptation<T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    S: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    i: usize,
    j: usize,
) -> T
where
    T: MatrixDataType,
{
    let weight = T::one() - adaptation.forgetting;
    let value = R.get(i, j) + weight * (y.get(i, 0) * y.get(j, 0) - S.get(i, j));
    if i == j && value < adaptation.floor {
        return adaptation.floor;
    }

    value
}

/// Subtracts `A + Aᵀ` from the square matrix `P` in place.
#[allow(non_snake_case)]
fn subtract_with_transpose<const STATES: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    A: &impl Matrix<STATES, STATES, T>,
) where
    T: MatrixDataType,
{
    for i in 0..STATES {
        for j in 0..STATES {
            let value = P.get(i, j) - A.get(i, j) - A.get(j, i);
            P.set(i, j, value);
        }
    }
}

/// Adapts the measurement noise covariance `R` after an applied correction:
/// `R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)` with the residual `ε = y − H×K×y`, then raises
/// every diagonal entry of `R` to at least the floor. See [`NoiseAdaptation`].
///
/// `P` must already be the corrected covariance `P⁺`. The result is written to the lower
/// triangle and mirrored, so `R` stays symmetric. `temp_eps` and `temp_PHt` are scratch
/// space and must not alias each other.
#[allow(non_snake_case, clippy::too_many_arguments)]
fn adapt_measurement_noise<const STATES: usize, const OBSERVATIONS: usize, T>(
    adaptation: NoiseAdaptation<T>,
    P: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    temp_eps: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
) where
    T: MatrixDataType,
{
    let alpha = adaptation.forgetting;
    let weight = T::one() - alpha;

    // Residual ε = y − H×(K×y): K×y goes to column 0 of temp_PHt, ε to column 0 of temp_eps.
    for k in 0..STATES {
        let ky = (0..OBSERVATIONS).map(|j| K.get(k, j) * y.get(j, 0)).sum();
        temp_PHt.set(k, 0, ky);
    }

    for i in 0..OBSERVATIONS {
        let hky: T = (0..STATES).map(|k| H.get(i, k) * temp_PHt.get(k, 0)).sum();
        temp_eps.set(i, 0, y.get(i, 0) - hky);
    }

    // R ← α×R + (1−α)×ε×εᵀ
    for i in 0..OBSERVATIONS {
        for j in 0..=i {
            let value = alpha * R.get(i, j) + weight * temp_eps.get(i, 0) * temp_eps.get(j, 0);
            R.set(i, j, value);
            R.set(j, i, value);
        }
    }

    // R ← R + (1−α)×H×P⁺×Hᵀ
    P.mult_transb(H, temp_PHt); // temp_PHt = P⁺×Hᵀ
    H.mult(temp_PHt, temp_eps); // temp_eps = H×P⁺×Hᵀ
    for i in 0..OBSERVATIONS {
        for j in 0..=i {
            let value = R.get(i, j) + weight * temp_eps.get(i, j);
            R.set(i, j, value);
            R.set(j, i, value);
        }
    }

    for i in 0..OBSERVATIONS {
        if R.get(i, i) < adaptation.floor {
            R.set(i, i, adaptation.floor);
        }
    }
}

/// Computes the Kalman gain `K = P×Hᵀ×S⁻¹` of the linear and extended observations.
///
/// Stores the normalized innovation squared in `nis` whenever `S = H×P×Hᵀ + R` can be
/// inverted, and leaves `K` unchanged if `S` is not positive definite or `nis` exceeds
/// `gate`. Neither `x` nor `P` is written. On return, `S` holds its Cholesky factor.
#[allow(non_snake_case, clippy::too_many_arguments)]
fn compute_kalman_gain<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    nis: &mut T,
    gate: Option<T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // S = H*P*H' + R
    H.mult(P, temp_HP); // temp = H*P
    temp_HP.mult_transb(H, S); // S = temp*Hᵀ
    S.add_inplace_a(R); // S += R

    // K = P*Hᵀ * S^-1
    if !S.cholesky_decompose_lower() {
        return CorrectionStatus::NotPositiveDefinite;
    }
    S.invert_l_cholesky(S_inv); // S_inv = S^-1

    // Gate on the normalized innovation squared before anything else is written.
    *nis = normalized_innovation_squared(y, S_inv);
    if gate.is_some_and(|gate| *nis > gate) {
        return CorrectionStatus::Gated;
    }

    // NOTE that to allow aliasing of Sinv and temp_PHt, a copy must be performed here
    P.mult_transb(H, temp_PHt); // temp = P*H'
    temp_PHt.mult(S_inv, K); // K = temp*Sinv
    CorrectionStatus::Applied
}

/// Performs the first pass of the recalibrated update of the extended observations from an
/// innovation `y = z - h(x)` that is already computed.
///
/// Computes the gain `K` at the prior as [`update_with_innovation`] does, keeps the prior `x`
/// in the first column of `temp_KHP` and moves the state to `x⁺ = x + K×y`. The caller then
/// writes the Jacobian `H₂` at `x⁺` into `H` and calls [`recalibrate_covariance`]. Returns
/// early, leaving `x` and `P` unchanged, if `S` is not positive definite or the innovation
/// is gated.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_state_for_recalibration<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    nis: &mut T,
    gate: Option<T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    let status = compute_kalman_gain(P, H, R, y, K, S, S_inv, temp_HP, temp_PHt, nis, gate);
    if !status.is_applied() {
        return status;
    }

    // temp_KHP may alias temp_PHt and S_inv, so the prior is only stored once K exists.
    for i in 0..STATES {
        temp_KHP.set(i, 0, x.get(i, 0));
    }

    K.multadd_rowvector(y, x);
    CorrectionStatus::Applied
}

/// Performs the second pass of the recalibrated update of the extended observations, after
/// [`update_state_for_recalibration`] moved `x` to `x⁺` and `H` holds the Jacobian `H₂` at
/// `x⁺`.
///
/// Recalibrates the covariance with the Joseph form
/// `P⁺ = P − K×H₂×P − (K×H₂×P)ᵀ + K×(H₂×P×H₂ᵀ + R)×Kᵀ`. If `trace(P⁺) > trace(P)`, both `x`
/// and `P` are restored and [`CorrectionStatus::BackedOut`] is returned. The trace is
/// compared before `P` is written. Only `temp_HP` and `temp_KHP` are used, since `temp_PHt`
/// and `S_inv` may alias either of them.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn recalibrate_covariance<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // S₂ = H₂*P*H₂ᵀ + R, with temp_HP = H₂*P.
    H.mult(P, temp_HP);
    temp_HP.mult_transb(H, S);
    S.add_inplace_a(R);

    // trace(P⁺) - trace(P) = trace(K*S₂*Kᵀ) - 2*trace(K*H₂*P)
    let cross = trace_of_product(K, temp_HP);
    if trace_of_gain_sandwich(K, S) - cross - cross > T::zero() {
        for i in 0..STATES {
            x.set(i, 0, temp_KHP.get(i, 0));
        }
        return CorrectionStatus::BackedOut;
    }

    // P = P - K*H₂*P - (K*H₂*P)ᵀ + K*S₂*Kᵀ
    K.mult(temp_HP, temp_KHP);
    sub_symmetric_part(P, temp_KHP);
    S.mult_transb(K, temp_HP); // temp_HP = S₂*Kᵀ
    K.mult(temp_HP, temp_KHP);
    P.add_inplace_a(temp_KHP);
    CorrectionStatus::Applied
}

/// Computes `trace(K×S×Kᵀ)` without forming the `STATES × STATES` product.
#[allow(non_snake_case)]
pub(crate) fn trace_of_gain_sandwich<const STATES: usize, const OBSERVATIONS: usize, T>(
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    S: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
) -> T
where
    T: MatrixDataType,
{
    let mut trace = T::zero();
    for i in 0..STATES {
        for j in 0..OBSERVATIONS {
            let mut row = T::zero();
            for l in 0..OBSERVATIONS {
                row += S.get(j, l) * K.get(i, l);
            }
            trace += K.get(i, j) * row;
        }
    }
    trace
}

/// Computes `trace(A×B)` of a `STATES × OBSERVATIONS` matrix `A` and an
/// `OBSERVATIONS × STATES` matrix `B` without forming the product.
#[allow(non_snake_case)]
fn trace_of_product<const STATES: usize, const OBSERVATIONS: usize, T>(
    A: &impl Matrix<STATES, OBSERVATIONS, T>,
    B: &impl Matrix<OBSERVATIONS, STATES, T>,
) -> T
where
    T: MatrixDataType,
{
    let mut trace = T::zero();
    for i in 0..STATES {
        for j in 0..OBSERVATIONS {
            trace += A.get(i, j) * B.get(j, i);
        }
    }
    trace
}

/// Subtracts `M + Mᵀ` from `P` in place.
#[allow(non_snake_case)]
pub(crate) fn sub_symmetric_part<const STATES: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    M: &impl Matrix<STATES, STATES, T>,
) where
    T: MatrixDataType,
{
    for i in 0..STATES {
        for j in 0..STATES {
            P.set(i, j, P.get(i, j) - M.get(i, j) - M.get(j, i));
        }
    }
}
