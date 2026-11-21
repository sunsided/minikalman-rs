//! # Observations for Regular, Extended, and Unscented Kalman Filters.

use crate::kalman::CorrectionStatus;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

pub mod extended;
pub mod regular;
pub mod unscented;

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
