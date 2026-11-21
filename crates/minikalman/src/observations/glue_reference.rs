//! Pins the slice-based innovation-form noise adaptation, gain and recalibration helpers of
//! [`super`] to the `get`/`set` loops they replaced, bit for bit.
//!
//! The old loops are kept here verbatim as references. The rewrite only changed how entries
//! are addressed, not the order of any operation, so every result must match exactly, for
//! floating-point and fixed-point types alike, including signed zeros.

#![allow(non_snake_case)]

use super::*;
use crate::matrix::MatrixData;
use crate::test_bits::{ToBits, assert_bits_eq, test_value};

/// Largest buffer the cases need (6×6).
const CAP: usize = 36;

/// A deterministic buffer of `test_value`s scaled by `scale`, with an exact `+0` at entry 1
/// and a `-0` at entry 2, so that a changed summation start would show in the sign of a zero.
fn values<T: MatrixDataType>(seed: u32, scale: f64) -> [T; CAP] {
    let mut v: [T; CAP] =
        core::array::from_fn(|i| T::from_f64(test_value(seed, i) * scale).unwrap());
    v[1] = T::zero();
    v[2] = T::from_f64(-0.0).unwrap();
    v
}

/// A symmetric positive definite `n×n` matrix `B×Bᵀ + I` from [`test_value`], scaled.
fn spd<T: MatrixDataType>(n: usize, seed: u32, scale: f64) -> [T; CAP] {
    let mut m = [T::zero(); CAP];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n)
                .map(|k| test_value(seed, i * n + k) * test_value(seed, j * n + k))
                .sum();
            let value = dot + if i == j { 1.0 } else { 0.0 };
            m[i * n + j] = T::from_f64(value * scale).unwrap();
        }
    }
    m
}

/// The `get`/`set` version of [`adapt_measurement_noise_from_innovation`].
fn old_adapt_from_innovation<const M: usize, T: MatrixDataType>(
    adaptation: NoiseAdaptation<T>,
    R: &mut impl MatrixMut<M, M, T>,
    y: &impl Matrix<M, 1, T>,
    S: &impl Matrix<M, M, T>,
    temp: &mut impl MatrixMut<M, M, T>,
) {
    for i in 0..M {
        for j in 0..=i {
            let value = old_innovation_matched_entry(adaptation, &*R, y, S, i, j);
            temp.set(i, j, value);
            temp.set(j, i, value);
        }
    }

    if !temp.cholesky_decompose_lower() {
        return;
    }

    for i in 0..M {
        for j in 0..=i {
            let value = old_innovation_matched_entry(adaptation, &*R, y, S, i, j);
            R.set(i, j, value);
            R.set(j, i, value);
        }
    }
}

/// The `get`/`set` version of [`innovation_matched_entry`].
fn old_innovation_matched_entry<const M: usize, T: MatrixDataType>(
    adaptation: NoiseAdaptation<T>,
    R: &impl Matrix<M, M, T>,
    y: &impl Matrix<M, 1, T>,
    S: &impl Matrix<M, M, T>,
    i: usize,
    j: usize,
) -> T {
    let weight = T::one() - adaptation.forgetting;
    let value = R.get(i, j) + weight * (y.get(i, 0) * y.get(j, 0) - S.get(i, j));
    if i == j && value < adaptation.floor {
        return adaptation.floor;
    }

    value
}

/// The `get`/`set` version of [`write_gain_and_compare`].
fn old_write_gain_and_compare<const N: usize, const M: usize, T: MatrixDataType>(
    K: &mut impl MatrixMut<N, M, T>,
    PHt: &impl Matrix<N, M, T>,
    S_inv: &impl Matrix<M, M, T>,
    tolerance: T,
) -> bool {
    let mut converged = true;
    for i in 0..N {
        for j in 0..M {
            let gain: T = (0..M).map(|l| PHt.get(i, l) * S_inv.get(l, j)).sum();
            let magnitude = abs(gain);
            let scale = if magnitude > T::one() {
                magnitude
            } else {
                T::one()
            };
            converged &= abs(gain - K.get(i, j)) <= tolerance * scale;
            K.set(i, j, gain);
        }
    }

    converged
}

/// The `get`/`set` version of [`trace_of_gain_sandwich`].
fn old_trace_of_gain_sandwich<const N: usize, const M: usize, T: MatrixDataType>(
    K: &impl Matrix<N, M, T>,
    S: &impl Matrix<M, M, T>,
) -> T {
    let mut trace = T::zero();
    for i in 0..N {
        for j in 0..M {
            let mut row = T::zero();
            for l in 0..M {
                row += S.get(j, l) * K.get(i, l);
            }
            trace += K.get(i, j) * row;
        }
    }
    trace
}

/// The `get`/`set` version of [`trace_of_product`].
fn old_trace_of_product<const N: usize, const M: usize, T: MatrixDataType>(
    A: &impl Matrix<N, M, T>,
    B: &impl Matrix<M, N, T>,
) -> T {
    let mut trace = T::zero();
    for i in 0..N {
        for j in 0..M {
            trace += A.get(i, j) * B.get(j, i);
        }
    }
    trace
}

/// The `get`/`set` version of [`sub_symmetric_part`].
fn old_sub_symmetric_part<const N: usize, T: MatrixDataType>(
    P: &mut impl MatrixMut<N, N, T>,
    M: &impl Matrix<N, N, T>,
) {
    for i in 0..N {
        for j in 0..N {
            P.set(i, j, P.get(i, j) - M.get(i, j) - M.get(j, i));
        }
    }
}

/// Runs every rewritten helper and its old loop on the same inputs and demands bit-identical
/// outputs. `floor` is chosen high enough that some diagonal entries are raised.
fn assert_helpers_match<const N: usize, const M: usize, T>(seed: u32, scale: f64)
where
    T: MatrixDataType + ToBits,
{
    let adaptation = NoiseAdaptation {
        forgetting: T::from_f64(0.75).unwrap(),
        floor: T::from_f64(0.9 * scale).unwrap(),
    };
    let P = spd::<T>(N, seed, scale);
    let H = values::<T>(seed + 1, scale);
    let y = values::<T>(seed + 2, scale);
    let K = values::<T>(seed + 3, scale);
    let R = spd::<T>(M, seed + 4, scale);
    let S = spd::<T>(M, seed + 5, scale);
    let tolerance = T::from_f64(1e-3).unwrap();

    let yv = MatrixData::new_ref::<M, 1, T>(&y);
    let k = MatrixData::new_ref::<N, M, T>(&K);
    let s = MatrixData::new_ref::<M, M, T>(&S);

    // Innovation-form adaptation.
    let (mut r_new, mut r_old) = (R, R);
    let (mut t_new, mut t_old) = ([T::zero(); CAP], [T::zero(); CAP]);
    // S = R makes the candidate 0.75×R + 0.25×y×yᵀ positive definite, so R is written.
    let s_r = MatrixData::new_ref::<M, M, T>(&R);
    adapt_measurement_noise_from_innovation(
        adaptation,
        &mut MatrixData::new_mut::<M, M, T>(&mut r_new),
        &yv,
        &s_r,
        &mut MatrixData::new_mut::<M, M, T>(&mut t_new),
    );
    old_adapt_from_innovation(
        adaptation,
        &mut MatrixData::new_mut::<M, M, T>(&mut r_old),
        &yv,
        &s_r,
        &mut MatrixData::new_mut::<M, M, T>(&mut t_old),
    );
    assert_bits_eq(&r_new, &r_old, "innovation-adapted R");
    assert_ne!(r_new[0].to_bits_u64(), R[0].to_bits_u64(), "R was adapted");
    assert_bits_eq(&t_new, &t_old, "innovation-adaptation temp");

    // Gain with convergence check, starting from a gain close to and far from the new one.
    for start in [K, [T::zero(); CAP]] {
        let (mut k_new, mut k_old) = (start, start);
        let pht = MatrixData::new_ref::<N, M, T>(&H);
        let converged_new = write_gain_and_compare(
            &mut MatrixData::new_mut::<N, M, T>(&mut k_new),
            &pht,
            &s,
            tolerance,
        );
        let converged_old = old_write_gain_and_compare(
            &mut MatrixData::new_mut::<N, M, T>(&mut k_old),
            &pht,
            &s,
            tolerance,
        );
        assert_eq!(converged_new, converged_old, "converged");
        assert_bits_eq(&k_new, &k_old, "gain");
    }

    // Traces and the symmetric-part subtraction.
    let ht = MatrixData::new_ref::<M, N, T>(&H);
    assert_bits_eq(
        &[trace_of_gain_sandwich(&k, &s)],
        &[old_trace_of_gain_sandwich(&k, &s)],
        "trace(K×S×Kᵀ)",
    );
    assert_bits_eq(
        &[trace_of_product(&k, &ht)],
        &[old_trace_of_product(&k, &ht)],
        "trace(K×H)",
    );
    let (mut p_new, mut p_old) = (P, P);
    let m = MatrixData::new_ref::<N, N, T>(&H);
    sub_symmetric_part(&mut MatrixData::new_mut::<N, N, T>(&mut p_new), &m);
    old_sub_symmetric_part(&mut MatrixData::new_mut::<N, N, T>(&mut p_old), &m);
    assert_bits_eq(&p_new, &p_old, "P − M − Mᵀ");
}

#[test]
fn noise_adaptation_and_gain_helpers_match_the_old_loops() {
    for seed in 0..4 {
        assert_helpers_match::<2, 1, f32>(seed, 1.0);
        assert_helpers_match::<4, 2, f32>(seed, 1.0);
        assert_helpers_match::<6, 3, f64>(seed, 1.0);
        assert_helpers_match::<3, 4, f64>(seed, 1.0);
    }
}

#[test]
#[cfg(feature = "fixed")]
fn noise_adaptation_and_gain_helpers_match_the_old_loops_fixed() {
    for seed in 0..4 {
        assert_helpers_match::<2, 1, fixed::types::I16F16>(seed, 0.25);
        assert_helpers_match::<4, 2, fixed::types::I16F16>(seed, 0.25);
        assert_helpers_match::<6, 3, fixed::types::I32F32>(seed, 1.0);
    }
}
