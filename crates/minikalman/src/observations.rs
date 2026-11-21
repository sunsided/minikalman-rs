//! # Observations for Regular, Extended, and Unscented Kalman Filters.

use crate::kalman::CorrectionStatus;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};
use crate::observations::innovation_factor::{InnovationFactor, is_number};

pub mod extended;
pub mod fixed_gain;
#[cfg(test)]
mod glue_reference;
pub(crate) mod innovation_factor;
pub(crate) mod predictively_oriented;
pub mod regular;
pub(crate) mod robust;
pub(crate) mod scratch;
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
/// about `N×m² + N²×m/2` multiply-adds per correction: `K×S` and the lower triangle of
/// `K×S×Kᵀ`. Both forms reuse the `H×P` the gain needed. The gain never overwrites `S`: for
/// floating-point types with up to three measurements it factors `S` square-root free in local
/// storage, otherwise (fixed point, or more measurements) it factors a copy (`m²` values, in the
/// `S⁻¹` temporary), so the Joseph form reads `S` as formed.
///
/// ## What it buys
///
/// Measured with the `numerical_robustness` test bench of this crate:
///
/// - **Symmetry.** The Joseph form leaves `P` exactly symmetric in every type. The standard
///   update leaves a relative asymmetry of about `10⁻⁵` in `f32` and `10⁻⁴` to `0.03` in
///   `I16F16`. The next prediction makes `P` exactly symmetric again (it computes the lower
///   triangle of `A×P×Aᵀ` and mirrors it), but its lower triangle has seen the asymmetry.
/// - **Positive definiteness, ill-conditioned measurements.** For the textbook case
///   `H = [[1,1,1],[1,1,1+δ]]`, `R = δ²×I`, `P = I`, the exact `P⁺` has eigenvalues of order
///   `δ²`, so near the singular end its definiteness is decided by rounding. The standard
///   update left `P` indefinite for 8 of 48 values of `δ` in `f32` (from `δ ≈ 2×10⁻³`) and 3 in
///   `f64` (from `δ ≈ 3×10⁻⁶`); the Joseph form for 8 and 8 (from `δ ≈ 7×10⁻³` and
///   `6×10⁻⁵`), and in `I32F32` 27 and 11. Neither form fixes the case.
/// - **Fixed point.** In `I16F16`, neither form helps once the variances approach the
///   resolution (`2⁻¹⁶`): with ten scalar sensors and a steady-state variance below one LSB,
///   the standard update left `P` indefinite in 6 of 1000 steps, and the Joseph form
///   diverged (`P` indefinite from the second step on; with the explicit `S⁻¹` it overflowed). Use a type with more fractional bits, such as `I32F32`, there.
///
/// On well-conditioned problems, both forms agree to a few tens of ULP in `f64`, given an exact
/// square root; with `micromath`'s approximate one, the gain is not quite optimal and they differ more.
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
    /// arithmetic.
    ///
    /// With either form, the observation's `S` holds `H×P⁻×Hᵀ + R` of the prior after a regular,
    /// extended or robust correction (also a gated or rejected one): the gain factors it in local
    /// storage (floating point, up to three measurements) or a copy of it. The predictively
    /// oriented corrections document their own `S`.
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
    // Slices of the static length, so that the loops need no bounds checks.
    let s_inv = &S_inv.as_slice()[..OBSERVATIONS * OBSERVATIONS];
    let y = &y.as_slice()[..OBSERVATIONS];
    let mut nis = T::zero();
    for i in 0..OBSERVATIONS {
        let mut row = T::zero();
        for j in 0..OBSERVATIONS {
            row += s_inv[i * OBSERVATIONS + j] * y[j];
        }
        nis += y[i] * row;
    }
    nis
}

/// Performs the update step of the linear and extended observations from an innovation `y`
/// that is already computed, either as `y = z - H×x` or as `y = z - h(x)`.
///
/// Stores the normalized innovation squared in `nis` whenever `S` can be inverted.
/// Leaves `x` and `P` unchanged if the innovation covariance `S = H×P×Hᵀ + R` is not
/// positive definite or the NIS is `NaN` (both [`CorrectionStatus::NotPositiveDefinite`]), or if
/// `nis` exceeds `gate`. Updates `P` as `covariance_update` says;
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
    // S = H*P*Hᵀ + R; temp_HP keeps H*P of the prior P for the covariance update below.
    form_innovation_covariance(P, H, R, S, temp_HP);

    // The gain factors a copy of S in S_inv, so S keeps H*P*Hᵀ + R for the Joseph form.
    let status = gain_from_innovation_covariance(P, H, y, K, S, S_inv, nis, gate);
    if !status.is_applied() {
        return status;
    }

    //* Correct state prediction
    //* x = x + K*y

    // x = x + K*y
    K.multadd_rowvector(y, x);

    // P is still the prior here, so temp_HP still holds H*P.
    match covariance_update {
        CovarianceUpdate::Standard => {
            //* Correct state covariances
            //* P = (I-K*H) * P
            //*   = P - K*(H*P)

            // P = P - K*(H*P)
            K.mult(temp_HP, temp_KHP); // temp_KHP = K*temp_HP
            P.sub_inplace_a(temp_KHP); // P -= temp_KHP
        }
        CovarianceUpdate::Joseph => joseph_update_from_hp(P, K, S, temp_HP, temp_PHt, temp_KHP),
    }

    if let Some(adaptation) = adaptation {
        // The adaptation needs two temporaries; S_inv and temp_PHt are free by now.
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
        // The adaptation reuses S_inv, which is free by now.
        adapt_measurement_noise_from_innovation(adaptation, R, y, S, S_inv);
    }

    CorrectionStatus::Applied
}

/// Updates `P` with the Joseph form `P = P - K×H×P - (K×H×P)ᵀ + K×S×Kᵀ` for any gain `K`,
/// leaving `S = H×P×Hᵀ + R` of the prior `P` in `S`.
///
/// `temp_HP`, `temp_PHt` and `temp_KHP` are distinct buffers; `temp_PHt` is only written
/// once `temp_HP` and `temp_KHP` are consumed.
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
    form_innovation_covariance(P, H, R, S, temp_HP);
    joseph_update_from_hp(P, K, S, temp_HP, temp_PHt, temp_KHP);
}

/// The Joseph update of [`joseph_update`] once `HP = H×P` and `S = H×P×Hᵀ + R` of the prior
/// `P` are formed: `P = P - K×HP - (K×HP)ᵀ + K×S×Kᵀ`.
///
/// `temp_PHt` is only written once `temp_KHP` is consumed.
#[allow(non_snake_case)]
fn joseph_update_from_hp<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    S: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    HP: &impl Matrix<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
) where
    T: MatrixDataType,
{
    // P = P - K*H*P - (K*H*P)ᵀ, on the lower triangle only
    K.mult(HP, temp_KHP); // temp_KHP = K*HP
    subtract_with_transpose_lower(P, temp_KHP);

    // P = P + K*S*Kᵀ, once temp_KHP is consumed. K×S×Kᵀ is symmetric in exact arithmetic but
    // not under rounding; adding its lower triangle and mirroring the sums into the upper one
    // leaves P exactly symmetric.
    K.mult(S, temp_PHt); // temp_PHt = K*S
    temp_PHt.multadd_symmetric_transb(K, P); // P += temp_PHt*Kᵀ
}

/// Forms the innovation covariance `S = H×P×Hᵀ + R`, leaving `H×P` in `temp_HP`.
#[allow(non_snake_case)]
fn form_innovation_covariance<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
) where
    T: MatrixDataType,
{
    H.mult(P, temp_HP); // temp_HP = H*P
    temp_HP.mult_transb(H, S); // S = temp_HP*Hᵀ
    S.add_inplace_a(R); // S += R
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
    // Slices of the static length, so that the loops need no bounds checks.
    let y = &y.as_slice()[..OBSERVATIONS];
    let s = &S.as_slice()[..OBSERVATIONS * OBSERVATIONS];
    {
        let r = &R.as_slice()[..OBSERVATIONS * OBSERVATIONS];
        let candidate = &mut temp.as_mut_slice()[..OBSERVATIONS * OBSERVATIONS];
        for i in 0..OBSERVATIONS {
            for j in 0..=i {
                let value = innovation_matched_entry::<OBSERVATIONS, T>(adaptation, r, y, s, i, j);
                candidate[i * OBSERVATIONS + j] = value;
                candidate[j * OBSERVATIONS + i] = value;
            }
        }
    }

    if !temp.cholesky_decompose_lower() {
        return;
    }

    // Only the lower triangle is read, so mirroring into the upper one is safe in place.
    let r = &mut R.as_mut_slice()[..OBSERVATIONS * OBSERVATIONS];
    for i in 0..OBSERVATIONS {
        for j in 0..=i {
            let value = innovation_matched_entry::<OBSERVATIONS, T>(adaptation, r, y, s, i, j);
            r[i * OBSERVATIONS + j] = value;
            r[j * OBSERVATIONS + i] = value;
        }
    }
}

/// Returns the entry `(i, j)` of the innovation-form candidate `R + (1−α)×(y×yᵀ − S)`, with
/// diagonal entries raised to at least the floor.
///
/// `r` and `s` are the row-major `OBSERVATIONS × OBSERVATIONS` entries of `R` and `S`, `y`
/// the innovation.
#[inline(always)]
fn innovation_matched_entry<const OBSERVATIONS: usize, T>(
    adaptation: NoiseAdaptation<T>,
    r: &[T],
    y: &[T],
    s: &[T],
    i: usize,
    j: usize,
) -> T
where
    T: MatrixDataType,
{
    let weight = T::one() - adaptation.forgetting;
    let ij = i * OBSERVATIONS + j;
    let value = r[ij] + weight * (y[i] * y[j] - s[ij]);
    if i == j && value < adaptation.floor {
        return adaptation.floor;
    }

    value
}

/// Raises every diagonal entry of the row-major `N × N` matrix `m` to at least `floor`.
#[inline(always)]
pub(crate) fn raise_diagonal_to_floor<const N: usize, T>(m: &mut [T], floor: T)
where
    T: MatrixDataType,
{
    for i in 0..N {
        if m[i * N + i] < floor {
            m[i * N + i] = floor;
        }
    }
}

/// Subtracts `A + Aᵀ` from the lower triangle of the square matrix `P` in place.
///
/// The upper triangle is left as it was: the caller overwrites it with the mirrored lower one.
#[allow(non_snake_case)]
fn subtract_with_transpose_lower<const STATES: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    A: &impl Matrix<STATES, STATES, T>,
) where
    T: MatrixDataType,
{
    // Slices of the static length, so that the loops need no bounds checks.
    let p = &mut P.as_mut_slice()[..STATES * STATES];
    let a = &A.as_slice()[..STATES * STATES];
    for i in 0..STATES {
        for j in 0..=i {
            p[i * STATES + j] = p[i * STATES + j] - a[i * STATES + j] - a[j * STATES + i];
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
    // This body stays on get/set although it never runs with the adaptation off: it is
    // inlined into every regular and extended correction, and a slice version shifted the
    // register allocation of the hot path (Cortex-M4F, benches/mcu: up to +5 instructions for
    // the 2×1 correction). Revisit together with an instruction-count check.
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
/// Stores the normalized innovation squared in `nis` whenever `S = H×P×Hᵀ + R` is positive
/// definite, and leaves `K` unchanged if `S` is not positive definite or `nis` exceeds
/// `gate`. Neither `x` nor `P` is written. `S` keeps `H×P×Hᵀ + R`; on the Cholesky path
/// (fixed-point types, or more than three measurements) `temp_factor` receives its Cholesky
/// factor.
#[allow(non_snake_case, clippy::too_many_arguments)]
fn compute_kalman_gain<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_factor: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    nis: &mut T,
    gate: Option<T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    form_innovation_covariance(P, H, R, S, temp_HP);
    gain_from_innovation_covariance(P, H, y, K, S, temp_factor, nis, gate)
}

/// The second half of [`compute_kalman_gain`]: factors `S = H×P×Hᵀ + R`, which
/// [`form_innovation_covariance`] formed, gates on the NIS and solves `K×S = P×Hᵀ` for `K`.
///
/// With a triangular factorization of `S` ([`InnovationFactor`]: `L×D×Lᵀ` without square
/// roots for floating-point types with up to three measurements, `L×Lᵀ` otherwise), the NIS `yᵀ×S⁻¹×y` takes one forward
/// substitution, and every row `k` of the gain solves `S×kᵀ = (P×Hᵀ)ᵀ` with one forward and
/// one back substitution in place. `S⁻¹` is never formed.
///
/// `S` itself keeps `H×P×Hᵀ + R` whatever the outcome: on the square-root free path the factor
/// lives in local storage, otherwise `S` is copied into `temp_factor` (`OBSERVATIONS²` values)
/// and factored there. Neither `x` nor `P` is written, and `K` only once
/// the correction is known to apply.
#[allow(non_snake_case, clippy::too_many_arguments)]
fn gain_from_innovation_covariance<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    temp_factor: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    nis: &mut T,
    gate: Option<T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    let Some(factor) = InnovationFactor::new(S, temp_factor) else {
        return CorrectionStatus::NotPositiveDefinite;
    };

    // Gate on the normalized innovation squared before anything else is written.
    *nis = factor.nis(y);
    if !is_number(*nis) {
        // A NaN would pass the gate; S is not usable as a covariance then.
        return CorrectionStatus::NotPositiveDefinite;
    }

    if gate.is_some_and(|gate| *nis > gate) {
        return CorrectionStatus::Gated;
    }

    // K = P*Hᵀ, then every row k ← S⁻¹×k in place (S is symmetric).
    P.mult_transb(H, K);
    let k = &mut K.as_mut_slice()[..STATES * OBSERVATIONS];
    for row in k.chunks_exact_mut(OBSERVATIONS) {
        factor.solve_row(row);
    }

    CorrectionStatus::Applied
}

/// Runs one correction of the covariance recursion for [`RegularKalman::settle_gain`]: forms
/// `S = H×P×Hᵀ + R` and the gain `K = P×Hᵀ×S⁻¹`, compares every new entry of `K` with the old
/// one while writing it, and updates `P` as `covariance_update` says.
///
/// No state, gate, NIS or noise adaptation is involved. On return, `S` holds the unfactored
/// `H×P×Hᵀ + R` of the `P` passed in (in both covariance update modes). Returns `Ok(true)` if
/// every entry satisfies `|ΔKᵢⱼ| ≤ tolerance×max(1, |Kᵢⱼ|)`. The previous gain needs no copy, so
/// no buffer beyond the observation's own is used; the aliasing rules of the observation
/// temporaries hold: `temp_HP` is formed again after the gain, because it may alias
/// `temp_S_inv` or `temp_PHt`.
///
/// [`RegularKalman::settle_gain`]: crate::regular::RegularKalman::settle_gain
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn settle_gain_step<const STATES: usize, const OBSERVATIONS: usize, T>(
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    tolerance: T,
    covariance_update: CovarianceUpdate,
) -> Result<bool, CorrectionStatus>
where
    T: MatrixDataType,
{
    // S = H*P*Hᵀ + R, factored and inverted.
    H.mult(P, temp_HP);
    temp_HP.mult_transb(H, S);
    S.add_inplace_a(R);
    if !S.cholesky_decompose_lower() {
        return Err(CorrectionStatus::NotPositiveDefinite);
    }
    S.invert_l_cholesky(S_inv);

    // K = P*Hᵀ*S⁻¹; temp_PHt never aliases S_inv.
    P.mult_transb(H, temp_PHt);
    let converged = write_gain_and_compare(K, temp_PHt, S_inv, tolerance);

    // temp_HP may alias S_inv or temp_PHt, so H*P is formed again from the prior P.
    match covariance_update {
        CovarianceUpdate::Standard => {
            H.mult(P, temp_HP);
            temp_HP.mult_transb(H, S);
            S.add_inplace_a(R);
            K.mult(temp_HP, temp_KHP);
            P.sub_inplace_a(temp_KHP);
        }
        CovarianceUpdate::Joseph => joseph_update(P, H, R, K, S, temp_HP, temp_PHt, temp_KHP),
    }

    Ok(converged)
}

/// Writes `K = PHt×S⁻¹` entry by entry and returns whether every entry moved by at most
/// `tolerance×max(1, |Kᵢⱼ|)` from the value it replaced.
///
/// The comparison uses absolute values rather than squares, so that a small tolerance does
/// not round to zero and a large gain does not overflow in fixed point.
#[allow(non_snake_case)]
fn write_gain_and_compare<const STATES: usize, const OBSERVATIONS: usize, T>(
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    PHt: &impl Matrix<STATES, OBSERVATIONS, T>,
    S_inv: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    tolerance: T,
) -> bool
where
    T: MatrixDataType,
{
    // Slices of the static length, so that the loops need no bounds checks.
    let pht = &PHt.as_slice()[..STATES * OBSERVATIONS];
    let s_inv = &S_inv.as_slice()[..OBSERVATIONS * OBSERVATIONS];
    let k = &mut K.as_mut_slice()[..STATES * OBSERVATIONS];
    let mut converged = true;
    for i in 0..STATES {
        for j in 0..OBSERVATIONS {
            let gain: T = (0..OBSERVATIONS)
                .map(|l| pht[i * OBSERVATIONS + l] * s_inv[l * OBSERVATIONS + j])
                .sum();
            let magnitude = abs(gain);
            let scale = if magnitude > T::one() {
                magnitude
            } else {
                T::one()
            };
            converged &= abs(gain - k[i * OBSERVATIONS + j]) <= tolerance * scale;
            k[i * OBSERVATIONS + j] = gain;
        }
    }

    converged
}

/// Returns `|value|` with only the operations every [`MatrixDataType`] has.
#[inline(always)]
fn abs<T>(value: T) -> T
where
    T: MatrixDataType,
{
    if value < T::zero() {
        T::zero() - value
    } else {
        value
    }
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
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    nis: &mut T,
    gate: Option<T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    let status = compute_kalman_gain(P, H, R, y, K, S, S_inv, temp_HP, nis, gate);
    if !status.is_applied() {
        return status;
    }

    // The prior is stored in column 0 of temp_KHP once K exists.
    copy_into_first_column::<STATES, T>(
        &x.as_slice()[..STATES],
        &mut temp_KHP.as_mut_slice()[..STATES * STATES],
    );

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
/// compared before `P` is written. Only `temp_HP` and `temp_KHP` are used.
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
        let prior = &temp_KHP.as_slice()[..STATES * STATES];
        for (x, row) in x.as_mut_slice()[..STATES]
            .iter_mut()
            .zip(prior.chunks_exact(STATES))
        {
            *x = row[0];
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
    // Slices of the static length, so that the loops need no bounds checks.
    let s = &S.as_slice()[..OBSERVATIONS * OBSERVATIONS];
    let k = &K.as_slice()[..STATES * OBSERVATIONS];
    let mut trace = T::zero();
    for k_row in k.chunks_exact(OBSERVATIONS) {
        for (j, s_row) in s.chunks_exact(OBSERVATIONS).enumerate() {
            let mut row = T::zero();
            for l in 0..OBSERVATIONS {
                row += s_row[l] * k_row[l];
            }
            trace += k_row[j] * row;
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
    // Slices of the static length, so that the loops need no bounds checks.
    let a = &A.as_slice()[..STATES * OBSERVATIONS];
    let b = &B.as_slice()[..OBSERVATIONS * STATES];
    let mut trace = T::zero();
    for i in 0..STATES {
        for j in 0..OBSERVATIONS {
            trace += a[i * OBSERVATIONS + j] * b[j * STATES + i];
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
    // Slices of the static length, so that the loops need no bounds checks.
    let p = &mut P.as_mut_slice()[..STATES * STATES];
    let m = &M.as_slice()[..STATES * STATES];
    for i in 0..STATES {
        for j in 0..STATES {
            p[i * STATES + j] = p[i * STATES + j] - m[i * STATES + j] - m[j * STATES + i];
        }
    }
}

/// Copies the `N` entries of `source` into column 0 of the row-major `N × N` matrix `target`.
#[inline(always)]
fn copy_into_first_column<const N: usize, T>(source: &[T], target: &mut [T])
where
    T: Copy,
{
    for (&value, row) in source.iter().zip(target.chunks_exact_mut(N)) {
        row[0] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;
    use crate::test_bits::{ToBits, assert_bits_eq, finite_or_infinity, max_deviation, test_value};

    /// Largest buffer the cases need (6×6); the buffers are longer than most matrices, which
    /// `is_valid` allows.
    const CAP: usize = 36;

    /// The inputs of one correction, row-major in `CAP`-long buffers.
    #[allow(non_snake_case)]
    #[derive(Clone, Copy)]
    struct Buffers<T> {
        x: [T; CAP],
        P: [T; CAP],
        H: [T; CAP],
        R: [T; CAP],
        y: [T; CAP],
        K: [T; CAP],
        S: [T; CAP],
        S_inv: [T; CAP],
        HP: [T; CAP],
        PHt: [T; CAP],
        KHP: [T; CAP],
    }

    /// A well-conditioned correction: `P = B×Bᵀ + I`, `R = D×Dᵀ + I/2`, `H`, `x`, `y` from
    /// [`test_value`], all scaled by `scale` so that they fit fixed point.
    fn case<const N: usize, const M: usize, T: MatrixDataType>(
        seed: u32,
        scale: f64,
    ) -> Buffers<T> {
        let c = |v: f64| T::from_f64(v * scale).unwrap();
        let spd = |n: usize, s: u32, diag: f64, i: usize, j: usize| {
            let dot: f64 = (0..n)
                .map(|k| test_value(s, i * n + k) * test_value(s, j * n + k))
                .sum();
            dot + if i == j { diag } else { 0.0 }
        };
        let z = [T::zero(); CAP];
        let mut b = Buffers {
            x: z,
            P: z,
            H: z,
            R: z,
            y: z,
            K: z,
            S: z,
            S_inv: z,
            HP: z,
            PHt: z,
            KHP: z,
        };
        for i in 0..N {
            b.x[i] = c(test_value(seed, i));
            for j in 0..N {
                b.P[i * N + j] = c(spd(N, seed + 1, 1.0, i, j));
            }
        }
        for i in 0..M {
            b.y[i] = c(test_value(seed + 2, i));
            for j in 0..N {
                b.H[i * N + j] = c(test_value(seed + 3, i * N + j));
            }
            for j in 0..M {
                b.R[i * M + j] = c(spd(M, seed + 4, 0.5, i, j));
            }
        }
        b
    }

    /// The gain as it was formed before the solve: `S⁻¹` explicitly from the Cholesky factor,
    /// the NIS as `yᵀ×S⁻¹×y` and `K = (P×Hᵀ)×S⁻¹`. The reference for the deviation tests.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn explicit_inverse_gain<const N: usize, const M: usize, T: MatrixDataType>(
        P: &impl Matrix<N, N, T>,
        H: &impl Matrix<M, N, T>,
        R: &impl Matrix<M, M, T>,
        y: &impl Matrix<M, 1, T>,
        K: &mut impl MatrixMut<N, M, T>,
        S: &mut impl MatrixMut<M, M, T>,
        S_inv: &mut impl MatrixMut<M, M, T>,
        HP: &mut impl MatrixMut<M, N, T>,
        PHt: &mut impl MatrixMut<N, M, T>,
        nis: &mut T,
        gate: Option<T>,
    ) -> CorrectionStatus {
        form_innovation_covariance(P, H, R, S, HP);
        if !S.cholesky_decompose_lower() {
            return CorrectionStatus::NotPositiveDefinite;
        }

        S.invert_l_cholesky(S_inv);
        *nis = normalized_innovation_squared(y, S_inv);
        if gate.is_some_and(|gate| *nis > gate) {
            return CorrectionStatus::Gated;
        }

        P.mult_transb(H, PHt);
        PHt.mult(S_inv, K);
        CorrectionStatus::Applied
    }

    /// Runs `update_with_innovation` and, on a copy, the update with the explicit-inverse gain
    /// of [`explicit_inverse_gain`]. Demands the same status, `x`, `P` and `K` untouched by a
    /// rejected correction, and `x`, `P`, `K`, `S` and the NIS within `tolerance` (relative,
    /// see [`max_deviation`]). Returns the largest deviation.
    #[allow(non_snake_case)]
    fn assert_close_to_explicit_inverse<const N: usize, const M: usize, T>(
        seed: u32,
        scale: f64,
        update: CovarianceUpdate,
        gate: Option<T>,
        tolerance: f64,
    ) -> f64
    where
        T: MatrixDataType + ToBits,
    {
        let initial = case::<N, M, T>(seed, scale);
        let (mut new, mut old) = (initial, initial);
        let (mut nis_new, mut nis_old) = (T::zero(), T::zero());

        let status_new = {
            let b = &mut new;
            update_with_innovation(
                &mut MatrixData::new_mut::<N, 1, T>(&mut b.x),
                &mut MatrixData::new_mut::<N, N, T>(&mut b.P),
                &MatrixData::new_ref::<M, N, T>(&b.H),
                &mut MatrixData::new_mut::<M, M, T>(&mut b.R),
                &MatrixData::new_ref::<M, 1, T>(&b.y),
                &mut MatrixData::new_mut::<N, M, T>(&mut b.K),
                &mut MatrixData::new_mut::<M, M, T>(&mut b.S),
                &mut MatrixData::new_mut::<M, M, T>(&mut b.S_inv),
                &mut MatrixData::new_mut::<M, N, T>(&mut b.HP),
                &mut MatrixData::new_mut::<N, M, T>(&mut b.PHt),
                &mut MatrixData::new_mut::<N, N, T>(&mut b.KHP),
                &mut nis_new,
                gate,
                None,
                update,
            )
        };

        let status_old = {
            let b = &mut old;
            let mut x = MatrixData::new_mut::<N, 1, T>(&mut b.x);
            let mut P = MatrixData::new_mut::<N, N, T>(&mut b.P);
            let H = MatrixData::new_ref::<M, N, T>(&b.H);
            let R = MatrixData::new_ref::<M, M, T>(&b.R);
            let y = MatrixData::new_ref::<M, 1, T>(&b.y);
            let mut K = MatrixData::new_mut::<N, M, T>(&mut b.K);
            let mut S = MatrixData::new_mut::<M, M, T>(&mut b.S);
            let mut S_inv = MatrixData::new_mut::<M, M, T>(&mut b.S_inv);
            let mut HP = MatrixData::new_mut::<M, N, T>(&mut b.HP);
            let mut PHt = MatrixData::new_mut::<N, M, T>(&mut b.PHt);
            let mut KHP = MatrixData::new_mut::<N, N, T>(&mut b.KHP);
            let status = explicit_inverse_gain(
                &P,
                &H,
                &R,
                &y,
                &mut K,
                &mut S,
                &mut S_inv,
                &mut HP,
                &mut PHt,
                &mut nis_old,
                gate,
            );
            if status.is_applied() {
                K.multadd_rowvector(&y, &mut x);
                match update {
                    CovarianceUpdate::Standard => {
                        H.mult(&P, &mut HP);
                        K.mult(&HP, &mut KHP);
                        P.sub_inplace_a(&KHP);
                    }
                    CovarianceUpdate::Joseph => {
                        joseph_update(&mut P, &H, &R, &K, &mut S, &mut HP, &mut PHt, &mut KHP);
                    }
                }
            }
            status
        };

        assert_eq!(status_new, status_old);
        assert_bits_eq(&new.R[..M * M], &old.R[..M * M], "R");
        if !status_new.is_applied() {
            assert_bits_eq(&new.x[..N], &initial.x[..N], "x");
            assert_bits_eq(&new.P[..N * N], &initial.P[..N * N], "P");
            assert_bits_eq(&new.K[..N * M], &initial.K[..N * M], "K");
        }

        // S keeps H×P×Hᵀ + R of the prior after every outcome and in both forms; the explicit
        // inverse left the Cholesky factor there in the standard form.
        let mut prior = initial;
        form_innovation_covariance(
            &MatrixData::new_ref::<N, N, T>(&prior.P),
            &MatrixData::new_ref::<M, N, T>(&prior.H),
            &MatrixData::new_ref::<M, M, T>(&prior.R),
            &mut MatrixData::new_mut::<M, M, T>(&mut prior.S),
            &mut MatrixData::new_mut::<M, N, T>(&mut prior.HP),
        );
        assert_bits_eq(&new.S[..M * M], &prior.S[..M * M], "S");

        let deviation = [
            max_deviation(&new.x[..N], &old.x[..N]),
            max_deviation(&new.P[..N * N], &old.P[..N * N]),
            max_deviation(&new.K[..N * M], &old.K[..N * M]),
            max_deviation(&[nis_new], &[nis_old]),
        ]
        .into_iter()
        .map(finite_or_infinity)
        .fold(0.0, f64::max);
        assert!(
            deviation <= tolerance,
            "{N}×{M} seed {seed} {update:?}: deviation {deviation:e} > {tolerance:e}"
        );

        deviation
    }

    /// Every shape class (`M < N`, `M = N`, `M > N`) with up to five measurements, plus a
    /// gated correction. Returns the largest deviation from the explicit-inverse gain.
    fn assert_all_shapes<T: MatrixDataType + ToBits>(scale: f64, tolerance: f64) -> f64 {
        let mut deviation = 0.0f64;
        for update in [CovarianceUpdate::Standard, CovarianceUpdate::Joseph] {
            for seed in 0..4 {
                let check = [
                    assert_close_to_explicit_inverse::<2, 1, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<4, 2, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<6, 3, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<3, 3, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<2, 3, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<1, 2, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<6, 4, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<4, 4, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    assert_close_to_explicit_inverse::<3, 5, T>(
                        seed, scale, update, None, tolerance,
                    ),
                    // A gate of zero rejects every correction once the NIS is formed.
                    assert_close_to_explicit_inverse::<4, 2, T>(
                        seed,
                        scale,
                        update,
                        Some(T::zero()),
                        tolerance,
                    ),
                ];
                deviation = check.into_iter().fold(deviation, f64::max);
            }
        }

        deviation
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn update_with_innovation_is_close_to_the_explicit_inverse_f32() {
        assert_all_shapes::<f32>(1.0, 1e-5);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn update_with_innovation_is_close_to_the_explicit_inverse_f64() {
        assert_all_shapes::<f64>(1.0, 1e-13);
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn update_with_innovation_is_close_to_the_explicit_inverse_fixed() {
        assert_all_shapes::<fixed::types::I16F16>(0.5, 1e-2);
        assert_all_shapes::<fixed::types::I32F32>(1.0, 1e-6);
    }

    /// The Joseph covariance update as it was before #167: `P − K×H×P − (K×H×P)ᵀ` on every
    /// entry, then `K×S×Kᵀ` added to the lower triangle and mirrored.
    #[allow(non_snake_case)]
    fn old_joseph<const N: usize, const M: usize, T: MatrixDataType>(b: &mut Buffers<T>) {
        let K = MatrixData::new_ref::<N, M, T>(&b.K);
        let S = MatrixData::new_ref::<M, M, T>(&b.S);
        K.mult(
            &MatrixData::new_ref::<M, N, T>(&b.HP),
            &mut MatrixData::new_mut::<N, N, T>(&mut b.KHP),
        );
        K.mult(&S, &mut MatrixData::new_mut::<N, M, T>(&mut b.PHt));

        let (p, a, ks) = (&mut b.P, &b.KHP, &b.PHt);
        for i in 0..N {
            for j in 0..N {
                p[i * N + j] = p[i * N + j] - a[i * N + j] - a[j * N + i];
            }
        }

        for i in 0..N {
            for j in 0..=i {
                let product: T = (0..M).map(|l| ks[i * M + l] * b.K[j * M + l]).sum();
                let value = p[i * N + j] + product;
                p[i * N + j] = value;
                p[j * N + i] = value;
            }
        }
    }

    /// Runs `joseph_update_from_hp` and the loops it replaced on the same `K`, `S` and `H×P`,
    /// and demands bit-identical results.
    #[allow(non_snake_case)]
    fn assert_joseph_matches_old<const N: usize, const M: usize, T>(seed: u32, scale: f64)
    where
        T: MatrixDataType + ToBits,
    {
        let mut new = case::<N, M, T>(seed, scale);
        new.K = core::array::from_fn(|i| T::from_f64(test_value(seed + 5, i) * scale).unwrap());
        {
            let b = &mut new;
            form_innovation_covariance(
                &MatrixData::new_ref::<N, N, T>(&b.P),
                &MatrixData::new_ref::<M, N, T>(&b.H),
                &MatrixData::new_ref::<M, M, T>(&b.R),
                &mut MatrixData::new_mut::<M, M, T>(&mut b.S),
                &mut MatrixData::new_mut::<M, N, T>(&mut b.HP),
            );
        }

        let mut old = new;
        old_joseph::<N, M, T>(&mut old);
        {
            let b = &mut new;
            joseph_update_from_hp(
                &mut MatrixData::new_mut::<N, N, T>(&mut b.P),
                &MatrixData::new_ref::<N, M, T>(&b.K),
                &MatrixData::new_ref::<M, M, T>(&b.S),
                &MatrixData::new_ref::<M, N, T>(&b.HP),
                &mut MatrixData::new_mut::<N, M, T>(&mut b.PHt),
                &mut MatrixData::new_mut::<N, N, T>(&mut b.KHP),
            );
        }

        assert_bits_eq(&new.P[..N * N], &old.P[..N * N], "P");
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn joseph_update_matches_the_old_loops() {
        for seed in 0..4 {
            assert_joseph_matches_old::<2, 1, f32>(seed, 1.0);
            assert_joseph_matches_old::<4, 2, f32>(seed, 1.0);
            assert_joseph_matches_old::<6, 3, f64>(seed, 1.0);
            assert_joseph_matches_old::<3, 4, f64>(seed, 1.0);
        }
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn joseph_update_matches_the_old_loops_fixed() {
        for seed in 0..4 {
            assert_joseph_matches_old::<4, 2, fixed::types::I16F16>(seed, 0.5);
            assert_joseph_matches_old::<6, 3, fixed::types::I32F32>(seed, 1.0);
        }
    }

    /// A non-positive-definite `S` returns early from both paths alike.
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn update_with_innovation_rejects_an_indefinite_s_like_before() {
        // Negating P and R makes S negative definite.
        for update in [CovarianceUpdate::Standard, CovarianceUpdate::Joseph] {
            assert_close_to_explicit_inverse::<4, 2, f64>(7, -1.0, update, None, 0.0);
            assert_close_to_explicit_inverse::<1, 1, f64>(7, -1.0, update, None, 0.0);
            // More than three measurements take the Cholesky path.
            assert_close_to_explicit_inverse::<5, 4, f64>(7, -1.0, update, None, 0.0);
        }
    }

    /// An `I16F16` correction with `S` at the resolution of the type (`S = 2⁻¹⁶`, one LSB):
    /// `P = [1]`, `H = [2⁻⁸]`, `R = [0]`, `y = 0.6`. A square-root free `y/S` would need about
    /// `39321`, beyond the range of `I16F16`; the Cholesky solve (`y/√S ≈ 153.6`) keeps the
    /// NIS `y²/S ≈ 23593` and the gain `P×Hᵀ/S = 256` in range, as the fixed-point types use it.
    #[test]
    #[cfg(feature = "fixed")]
    #[allow(non_snake_case)]
    fn fixed_point_gain_at_the_resolution_stays_in_range() {
        use fixed::types::I16F16;

        let one = I16F16::ONE;
        let (mut x, mut P, H, mut R) = ([I16F16::ZERO], [one], [one >> 8], [I16F16::ZERO]);
        let y = [I16F16::from_num(0.6)];
        let (mut K, mut S, mut S_inv, mut HP, mut PHt, mut KHP) = (
            [I16F16::ZERO],
            [I16F16::ZERO],
            [I16F16::ZERO],
            [I16F16::ZERO],
            [I16F16::ZERO],
            [I16F16::ZERO],
        );
        let mut nis = I16F16::ZERO;
        let status = update_with_innovation(
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut x),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut P),
            &MatrixData::new_ref::<1, 1, I16F16>(&H),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut R),
            &MatrixData::new_ref::<1, 1, I16F16>(&y),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut K),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut S),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut S_inv),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut HP),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut PHt),
            &mut MatrixData::new_mut::<1, 1, I16F16>(&mut KHP),
            &mut nis,
            None,
            None,
            CovarianceUpdate::Joseph,
        );

        assert_eq!(status, CorrectionStatus::Applied);
        assert_eq!(S[0], I16F16::DELTA, "S keeps H×P×Hᵀ + R");
        assert_eq!(K[0], I16F16::from_num(256));
        assert!(
            (nis.to_num::<f64>() - 0.36 * 65536.0).abs() < 10.0,
            "NIS {nis}"
        );
        assert!((x[0].to_num::<f64>() - 153.6).abs() < 0.01, "x {}", x[0]);
    }

    /// A `NaN` NIS, which every gate comparison lets through, never applies a correction, on
    /// either factorization path.
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))]
    fn nan_nis_is_not_applied() {
        nan_innovation_is_not_applied::<2, 1>();
        nan_innovation_is_not_applied::<6, 4>();
    }

    /// Runs one correction of the [`case`] with a `NaN` in `y` and asserts that it is rejected
    /// with `x`, `P` and `K` untouched.
    fn nan_innovation_is_not_applied<const N: usize, const M: usize>() {
        let initial = case::<N, M, f64>(3, 1.0);
        let mut b = initial;
        b.y[0] = f64::NAN;
        let mut nis = 0.0;
        let status = update_with_innovation(
            &mut MatrixData::new_mut::<N, 1, f64>(&mut b.x),
            &mut MatrixData::new_mut::<N, N, f64>(&mut b.P),
            &MatrixData::new_ref::<M, N, f64>(&b.H),
            &mut MatrixData::new_mut::<M, M, f64>(&mut b.R),
            &MatrixData::new_ref::<M, 1, f64>(&b.y),
            &mut MatrixData::new_mut::<N, M, f64>(&mut b.K),
            &mut MatrixData::new_mut::<M, M, f64>(&mut b.S),
            &mut MatrixData::new_mut::<M, M, f64>(&mut b.S_inv),
            &mut MatrixData::new_mut::<M, N, f64>(&mut b.HP),
            &mut MatrixData::new_mut::<N, M, f64>(&mut b.PHt),
            &mut MatrixData::new_mut::<N, N, f64>(&mut b.KHP),
            &mut nis,
            Some(1e9),
            None,
            CovarianceUpdate::Joseph,
        );

        assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
        assert!(nis.is_nan());
        assert_bits_eq(&b.x[..N], &initial.x[..N], "x");
        assert_bits_eq(&b.P[..N * N], &initial.P[..N * N], "P");
        assert_bits_eq(&b.K[..N * M], &initial.K[..N * M], "K");
    }
}
