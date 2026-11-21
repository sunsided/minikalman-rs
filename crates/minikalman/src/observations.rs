//! # Observations for Regular, Extended, and Unscented Kalman Filters.

use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

pub mod extended;
pub mod regular;
pub mod unscented;

/// The outcome of a correction step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Correction {
    /// The update was applied to `x` and `P`.
    Applied,
    /// The innovation covariance `S` was not positive definite; `x` and `P` are unchanged.
    NotPositiveDefinite,
    /// The normalized innovation squared exceeded the gate; `x` and `P` are unchanged.
    Gated,
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
/// positive definite, or if `nis` exceeds `gate`.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_with_innovation<const STATES: usize, const OBSERVATIONS: usize, T>(
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
    nis: &mut T,
    gate: Option<T>,
) -> Correction
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
        return Correction::NotPositiveDefinite;
    }
    S.invert_l_cholesky(S_inv); // S_inv = S^-1

    // Gate on the normalized innovation squared before anything else is written.
    *nis = normalized_innovation_squared(y, S_inv);
    if gate.is_some_and(|gate| *nis > gate) {
        return Correction::Gated;
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
    Correction::Applied
}
