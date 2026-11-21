//! # Observations for Regular, Extended, and Unscented Kalman Filters.

use crate::kalman::CorrectionStatus;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

pub mod extended;
pub mod regular;
pub mod unscented;

/// Online adaptation of the measurement noise covariance `R` from the residuals of the
/// corrections, enabled per observation with `set_noise_adaptation`.
///
/// After every applied correction, `R` is replaced by an exponentially weighted average
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
/// [gated](crate::prelude::CorrectionStatus::Gated) or
/// [not positive definite](crate::prelude::CorrectionStatus::NotPositiveDefinite) leave `R`
/// unchanged.
///
/// ## Background
///
/// This is the residual-based covariance matching of Akhlaghi, Zhou and Huang, "Adaptive
/// Adjustment of Noise Covariance in Kalman Filter for Dynamic State Estimation"
/// (arXiv:1702.00884, 2017), eq. (11), which replaces a moving window by the forgetting
/// factor. The paper prints the prior covariance `P⁻` in that equation; this implementation
/// uses the corrected covariance `P⁺` (as in Mohamed and Schwarz, "Adaptive Kalman Filtering
/// for INS/GPS", 1999), because only then is the true `R` a fixed point: with
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
/// ## Choosing α and when the estimate diverges
///
/// For a single measurement, the update reduces to
/// `R ← R × (1 + (1−α) × (R/S) × (NIS − 1))`, so `R` grows while the
/// [NIS](crate::regular::RegularObservation::nis) is above `1` and shrinks while it is below.
///
/// - `α` close to `1` (e.g. `0.95` to `0.99`) averages over roughly `1 / (1−α)` corrections
///   and gives a smooth estimate; a small `α` lets single residuals dominate and `R`
///   fluctuates strongly. `α` must lie in `(0, 1)`: `α ≥ 1` freezes `R`, and `α ≤ 0`
///   discards the previous `R` and can let it lose definiteness (only the floor remains).
/// - `R = 0` is absorbing, so the floor must be positive to let `R` recover.
/// - While `H×P×Hᵀ` dominates `R` (an uncertain or unobservable state), `R/S` is small and
///   `R` barely adapts. With unobservable states, `P` keeps growing and `R` stalls.
/// - Every excess of the NIS is attributed to `R`, including a process noise `Q` that is too
///   small or a model error. `R` then grows without bound to make up for it, and the filter
///   ignores the measurements; tune `Q` first.
/// - A [gate](crate::regular::RegularObservation::set_gate) rejects corrections before they
///   reach the adaptation. After a real step up in the measurement noise, a tight gate
///   rejects most readings and `R` never catches up; widen or disable the gate when
///   adapting.
/// - With fixed-point types, `ε×εᵀ` can overflow for large residuals, like the NIS.
///
/// Only `R` is adapted; the process noise `Q` is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseAdaptation<T> {
    /// The forgetting factor `α` in `(0, 1)`: the weight of the previous `R`. The new
    /// residual is weighted with `1 − α`.
    pub forgetting: T,

    /// Lower bound for every diagonal entry of the adapted `R`. Should be positive.
    pub floor: T,
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
/// positive definite, or if `nis` exceeds `gate`. After an applied correction, adapts `R`
/// if `adaptation` is set; see [`NoiseAdaptation`].
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
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // S = H*P*H' + R
    H.mult(P, temp_HP); // temp = H*P
    temp_HP.mult_transb(H, S); // S = temp*Hᵀ
    S.add_inplace_a(R); // S += R

    //* Calculate Kalman gain
    //* K = P*Hᵀ * S^-1

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

    //* Correct state prediction
    //* x = x + K*y

    // x = x + K*y
    K.multadd_rowvector(y, x);

    //* Correct state covariances
    //* P = (I-K*H) * P
    //*   = P - K*(H*P)

    // P = P - K*(H*P)
    H.mult(P, temp_HP); // temp_HP = H*P
    K.mult(temp_HP, temp_KHP); // temp_KHP = K*temp_HP
    P.sub_inplace_a(temp_KHP); // P -= temp_KHP

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
/// `P = P - K×H×P - (K×H×P)ᵀ + K×S×Kᵀ` with `S = H×P×Hᵀ + R`, so that `P` stays symmetric
/// and positive semi-definite for any `K`. `S` is not inverted and is left holding
/// `H×P×Hᵀ + R`.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_with_gain<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &impl Matrix<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // S = H*P*H' + R, from the prior P
    H.mult(P, temp_HP); // temp_HP = H*P
    temp_HP.mult_transb(H, S); // S = temp_HP*Hᵀ
    S.add_inplace_a(R); // S += R

    // x = x + K*y
    K.multadd_rowvector(y, x);

    // P = P - K*H*P - (K*H*P)ᵀ
    K.mult(temp_HP, temp_KHP); // temp_KHP = K*temp_HP
    subtract_with_transpose(P, temp_KHP);

    // P = P + K*S*Kᵀ; temp_PHt MAY alias temp_KHP, so it is only written once temp_KHP is consumed.
    K.mult(S, temp_PHt); // temp_PHt = K*S
    temp_PHt.multadd_transb(K, P); // P += temp_PHt*Kᵀ
    CorrectionStatus::Applied
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
