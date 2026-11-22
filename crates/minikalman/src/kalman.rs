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

/// Predicts the state, `x = A×x`, in place for the companion transition `A` whose first
/// column is `phi` and whose superdiagonal is one (see `RegularKalman::predict_companion`).
///
/// Row `i` of `A×x` is `φᵢ·x₀ + xᵢ₊₁` (no `xₙ` term in the last row), so the entries are
/// computed in ascending order, each before its `xᵢ₊₁` is overwritten. Every entry equals the
/// one `mult_rowvector` computes with the dense `A` bit for bit, for finite `x`, except that an
/// exact zero may come out with the other sign (the dense sum adds `0·xₖ` terms).
pub(crate) fn predict_companion_state<const STATES: usize, T>(
    phi: &[T; STATES],
    x: &mut impl MatrixMut<STATES, 1, T>,
) where
    T: MatrixDataType,
{
    let x = &mut x.as_mut_slice()[..STATES];
    let x0 = x[0];

    for i in 0..STATES {
        let mut total = phi[i] * x0;
        if i + 1 < STATES {
            total += x[i + 1];
        }

        x[i] = total;
    }
}

/// Predicts the estimate covariance, `P = A×P×Aᵀ + Q`, in `O(n²)` and in place for the
/// companion transition `A` whose first column is `phi` and whose superdiagonal is one.
///
/// With `M = A×P`, row `i` of `M` is `φᵢ·P₀ + Pᵢ₊₁` (row `0` of `P` scaled, plus the next row),
/// and `C = M×Aᵀ` has `Cᵢⱼ = Mᵢ₀·φⱼ + Mᵢ,ⱼ₊₁`. Only the lower triangle of `C` is computed and
/// mirrored, as in [`predict_covariance`], one row `i` at a time: row `i` reads only row `i + 1`
/// of `P`, which no earlier row has written, and row `0`, which is copied first.
///
/// Every sum adds the same non-zero terms in the same order, from `+0`, as the dense
/// `mult` / `mult_symmetric_transb` pair with `A` stored densely; the terms that pair adds on
/// top are `0·Pₖⱼ`, which leave a sum started from `+0` unchanged. The result is therefore
/// the same bit for bit, for finite `P` (the dense path turns `0·∞` into NaN).
///
/// Scratch: one copy of row `0` of `P` on the stack, `STATES` elements.
#[allow(non_snake_case)]
pub(crate) fn predict_companion_covariance<const STATES: usize, T>(
    phi: &[T; STATES],
    P: &mut impl MatrixMut<STATES, STATES, T>,
    Q: &impl Matrix<STATES, STATES, T>,
    lambda: Option<T>,
) where
    T: MatrixDataType,
{
    let scale = lambda.map(|lambda| (lambda * lambda).recip());
    let p = &mut P.as_mut_slice()[..STATES * STATES];
    let row0: [T; STATES] = core::array::from_fn(|j| p[j]);

    for i in 0..STATES {
        let m_i0 = companion_product_entry(phi, &row0, p, i, 0);
        for j in 0..=i {
            let mut total = T::zero();
            total += m_i0 * phi[j];
            if j + 1 < STATES {
                total += companion_product_entry(phi, &row0, p, i, j + 1);
            }

            let total = match scale {
                None => total,
                Some(scale) => total * scale,
            };
            p[i * STATES + j] = total;
            p[j * STATES + i] = total;
        }
    }

    Q.add_inplace_b(P); // P = P + Q
}

/// Entry `(i, k)` of `M = A×P` for the companion `A`: `φᵢ·P₀ₖ + Pᵢ₊₁,ₖ`, summed from `+0`.
///
/// `row0` is the original row `0` of `P`; `p` must still hold the original row `i + 1`.
#[inline(always)]
fn companion_product_entry<const STATES: usize, T>(
    phi: &[T; STATES],
    row0: &[T; STATES],
    p: &[T],
    i: usize,
    k: usize,
) -> T
where
    T: MatrixDataType,
{
    let mut total = T::zero();
    total += phi[i] * row0[k];
    if i + 1 < STATES {
        total += p[(i + 1) * STATES + k];
    }

    total
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

    /// The dense companion matrix: first column `phi`, ones on the superdiagonal.
    fn companion_matrix<const N: usize, const L: usize, T: MatrixDataType>(phi: &[T; N]) -> [T; L] {
        let mut a = [T::zero(); L];
        for i in 0..N {
            a[i * N] = phi[i];
            if i + 1 < N {
                a[i * N + i + 1] = T::one();
            }
        }

        a
    }

    /// Asserts that the companion prediction of `x` and `P` matches the dense one bit for bit
    /// on the given operands, up to the sign of an exact zero in `x`. `P0` may be asymmetric: both paths must read the same entries.
    #[allow(non_snake_case)]
    fn check_companion_on<const N: usize, const L: usize, T>(
        phi: &[T; N],
        x0: &[T; N],
        P0: &[T; L],
        Q: &[T; L],
        lambda: Option<T>,
    ) where
        T: MatrixDataType + ToBits,
    {
        let A = companion_matrix::<N, L, T>(phi);
        let a = MatrixData::new_ref::<N, N, T>(&A);

        let mut want_P = *P0;
        let mut temp = [T::zero(); L];
        predict_covariance::<N, T>(
            &a,
            &mut MatrixData::new_mut::<N, N, T>(&mut want_P),
            &MatrixData::new_ref::<N, N, T>(Q),
            &mut MatrixData::new_mut::<N, N, T>(&mut temp),
            lambda,
        );
        let mut got_P = *P0;
        predict_companion_covariance::<N, T>(
            phi,
            &mut MatrixData::new_mut::<N, N, T>(&mut got_P),
            &MatrixData::new_ref::<N, N, T>(Q),
            lambda,
        );
        assert_bits_eq(&got_P, &want_P, "P");

        let mut want_x = [T::zero(); N];
        a.mult_rowvector(
            &MatrixData::new_ref::<N, 1, T>(x0),
            &mut MatrixData::new_mut::<N, 1, T>(&mut want_x),
        );
        let mut got_x = *x0;
        predict_companion_state::<N, T>(phi, &mut MatrixData::new_mut::<N, 1, T>(&mut got_x));
        // An exact zero may differ in sign only; every other entry is the same bit for bit.
        for (got, want) in got_x.iter_mut().zip(&want_x) {
            if *got == T::zero() && *want == T::zero() {
                *got = *want;
            }
        }

        assert_bits_eq(&got_x, &want_x, "x");
    }

    /// Checks the companion prediction on random operands (asymmetric `P`, symmetric `Q`) and
    /// on an ARMA-like start: `x = 0`, diagonal `P`, a zero and a negative coefficient.
    #[allow(non_snake_case)]
    fn check_companion<const N: usize, const L: usize, T>(lambda: Option<T>)
    where
        T: MatrixDataType + ToBits,
    {
        let phi = values::<N, T>(4);
        let x0 = values::<N, T>(5);
        let P0 = values::<L, T>(6);
        let mut Q = values::<L, T>(7);
        for i in 0..N {
            for j in 0..i {
                Q[j * N + i] = Q[i * N + j];
            }
        }

        check_companion_on::<N, L, T>(&phi, &x0, &P0, &Q, lambda);

        let mut sparse_phi = phi;
        sparse_phi[N - 1] = T::from_f64(-0.5).unwrap();
        sparse_phi[0] = T::from_f64(-0.75).unwrap();
        sparse_phi[N / 2] = T::zero();
        let mut diagonal = [T::zero(); L];
        for i in 0..N {
            diagonal[i * N + i] = T::one();
        }

        check_companion_on::<N, L, T>(&sparse_phi, &[T::zero(); N], &diagonal, &diagonal, lambda);
    }

    /// Runs [`check_companion`] for `n = 1…8`.
    fn check_companion_sizes<T: MatrixDataType + ToBits>(lambda: Option<T>) {
        check_companion::<1, 1, T>(lambda);
        check_companion::<2, 4, T>(lambda);
        check_companion::<3, 9, T>(lambda);
        check_companion::<4, 16, T>(lambda);
        check_companion::<5, 25, T>(lambda);
        check_companion::<6, 36, T>(lambda);
        check_companion::<7, 49, T>(lambda);
        check_companion::<8, 64, T>(lambda);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    fn companion_prediction_matches_the_dense_one() {
        check_companion_sizes::<f32>(None);
        check_companion_sizes::<f32>(Some(0.9));
        check_companion_sizes::<f64>(None);
        check_companion_sizes::<f64>(Some(0.95));
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn companion_prediction_matches_the_dense_one_fixed() {
        use fixed::types::{I16F16, I32F32};

        check_companion_sizes::<I16F16>(None);
        check_companion_sizes::<I16F16>(Some(I16F16::from_num(0.9)));
        check_companion_sizes::<I32F32>(None);
        check_companion_sizes::<I32F32>(Some(I32F32::from_num(0.95)));
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn predict_covariance_mirrors_the_lower_triangle_fixed() {
        check_predict::<4, 16, fixed::types::I16F16>(None);
        check_predict::<3, 9, fixed::types::I32F32>(None);
        check_predict::<4, 16, fixed::types::I16F16>(Some(fixed::types::I16F16::from_num(0.9)));
        check_predict::<3, 9, fixed::types::I32F32>(Some(fixed::types::I32F32::from_num(0.95)));
    }
}
