pub use filter_trait::*;
pub use matrix_types::*;
pub use time_update::TimeUpdate;

pub mod extended;
mod filter_trait;
pub mod information;
pub mod luenberger;
mod matrix_types;
pub(crate) mod process_noise;
pub mod regular;
pub mod rls;
pub mod square_root;
mod time_update;
pub mod ud;
pub mod unscented;

use crate::matrix::{Matrix, MatrixDataType, MatrixMut};

/// Predicts the estimate covariance, `P = A×P×Aᵀ + Q`, shared by the linear and extended filters.
///
/// With `Some(lambda)`, the propagated covariance is scaled by `1/λ²` before `Q` is added,
/// with `0 < λ ≤ 1`: the fading-memory prediction, see `docs/fading-memory.md` (rendered on
/// `RegularKalman::predict_tuned`).
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
    // A×P×Aᵀ is symmetric: only its lower triangle is computed, then mirrored.
    A.mult(P, P_temp); // temp = A*P
    match lambda {
        None => P_temp.mult_symmetric_transb(A, P), // P = temp*Aᵀ
        Some(lambda) => P_temp.multscale_symmetric_transb(A, (lambda * lambda).recip(), P), // P = temp*Aᵀ * 1/λ²
    }
    Q.add_inplace_b(P); // P = P + Q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;
    use crate::test_bits::{ToBits, assert_bits_eq, test_value};

    /// `L` deterministic values in `[-1, 1)`, seeded with `seed`.
    fn values<const L: usize, T: MatrixDataType>(seed: u32) -> [T; L] {
        core::array::from_fn(|i| T::from_f64(test_value(seed, i)).unwrap())
    }

    /// `P = A×P×Aᵀ + Q` with the full kernels that computed every entry before #167.
    #[allow(non_snake_case)]
    fn full_predict<const N: usize, const L: usize, T: MatrixDataType>(
        A: &[T; L],
        P: &mut [T; L],
        Q: &[T; L],
        lambda: Option<T>,
    ) {
        let a = MatrixData::new_ref::<N, N, T>(A);
        let q = MatrixData::new_ref::<N, N, T>(Q);
        let mut temp = [T::zero(); L];
        let mut t = MatrixData::new_mut::<N, N, T>(&mut temp);
        let mut p = MatrixData::new_mut::<N, N, T>(P);

        a.mult(&p, &mut t);
        match lambda {
            None => t.mult_transb(&a, &mut p),
            Some(lambda) => t.multscale_transb(&a, (lambda * lambda).recip(), &mut p),
        }

        q.add_inplace_b(&mut p);
    }

    /// Asserts that the prediction keeps the lower triangle of the full kernels bit for bit
    /// and mirrors it exactly into the upper one.
    #[allow(non_snake_case)]
    fn check_predict<const N: usize, const L: usize, T>(lambda: Option<T>)
    where
        T: MatrixDataType + ToBits,
    {
        let A = values::<L, T>(1);
        let P0 = values::<L, T>(2);
        // A symmetric Q, so that adding it keeps the mirror exact.
        let mut Q = values::<L, T>(3);
        for i in 0..N {
            for j in 0..i {
                Q[j * N + i] = Q[i * N + j];
            }
        }

        let mut want = P0;
        full_predict::<N, L, T>(&A, &mut want, &Q, lambda);

        let mut got = P0;
        let mut temp = [T::zero(); L];
        predict_covariance::<N, T>(
            &MatrixData::new_ref::<N, N, T>(&A),
            &mut MatrixData::new_mut::<N, N, T>(&mut got),
            &MatrixData::new_ref::<N, N, T>(&Q),
            &mut MatrixData::new_mut::<N, N, T>(&mut temp),
            lambda,
        );

        for i in 0..N {
            for j in 0..=i {
                assert_bits_eq(&[got[i * N + j]], &[want[i * N + j]], "lower");
                assert_bits_eq(&[got[j * N + i]], &[got[i * N + j]], "mirror");
            }
        }
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    fn predict_covariance_mirrors_the_lower_triangle() {
        check_predict::<2, 4, f32>(None);
        check_predict::<4, 16, f32>(None);
        check_predict::<6, 36, f64>(None);
        check_predict::<5, 25, f32>(Some(0.9));
        check_predict::<3, 9, f64>(Some(0.95));
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn predict_covariance_mirrors_the_lower_triangle_fixed() {
        check_predict::<4, 16, fixed::types::I16F16>(None);
        check_predict::<3, 9, fixed::types::I32F32>(None);
    }
}
