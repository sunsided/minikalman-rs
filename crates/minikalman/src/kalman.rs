pub use filter_trait::*;
pub use matrix_types::*;

pub mod extended;
mod filter_trait;
pub mod information;
mod matrix_types;
pub mod regular;
pub mod unscented;

use crate::matrix::{Matrix, MatrixDataType, MatrixMut};

/// Predicts the estimate covariance, `P = A×P×Aᵀ + Q`, shared by the linear and extended filters.
///
/// With `Some(lambda)`, the propagated covariance is scaled by `1/λ²` before `Q` is added,
/// with `0 < λ ≤ 1` (smaller values mean larger uncertainty).
#[allow(non_snake_case)]
pub(crate) fn predict_covariance<const STATES: usize, T>(
    A: &impl Matrix<STATES, STATES, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    Q: &impl Matrix<STATES, STATES, T>,
    P_temp: &mut impl MatrixMut<STATES, STATES, T>,
    lambda: Option<T>,
) where
    T: MatrixDataType,
{
    A.mult(P, P_temp); // temp = A*P
    match lambda {
        None => P_temp.mult_transb(A, P), // P = temp*Aᵀ
        Some(lambda) => P_temp.multscale_transb(A, (lambda * lambda).recip(), P), // P = temp*Aᵀ * 1/λ²
    }
    Q.add_inplace_b(P); // P = P + Q
}
