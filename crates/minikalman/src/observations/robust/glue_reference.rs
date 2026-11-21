//! Pins the slice-based helpers of the robust correction to the `get`/`set` loops they
//! replaced, bit for bit.
//!
//! The rewrite only changed how entries are addressed, not the order of any operation.

#![allow(non_snake_case)]

use super::*;
use crate::matrix::MatrixData;
use crate::test_bits::{ToBits, assert_bits_eq, test_value};

/// Largest buffer the cases need (6×6).
const CAP: usize = 36;

/// A deterministic buffer of `test_value`s scaled by `scale`, with `+0` and `-0` entries.
fn values<T: MatrixDataType>(seed: u32, scale: f64) -> [T; CAP] {
    let mut v: [T; CAP] =
        core::array::from_fn(|i| T::from_f64(test_value(seed, i) * scale).unwrap());
    v[1] = T::zero();
    v[2] = T::from_f64(-0.0).unwrap();
    v
}

/// The `get`/`set` version of [`student_t_weight`].
fn old_student_t_weight<const M: usize, T: MatrixDataType>(
    dof: T,
    G: &impl Matrix<M, M, T>,
    R: &impl Matrix<M, M, T>,
    y: &impl Matrix<M, 1, T>,
    S_inv: &impl Matrix<M, M, T>,
) -> T {
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

/// Runs every rewritten helper and its old loop on the same inputs and demands bit-identical
/// outputs.
fn assert_helpers_match<const N: usize, const M: usize, T>(seed: u32, magnitude: f64)
where
    T: MatrixDataType + ToBits,
{
    let (a, b, g, r, y) = (
        values::<T>(seed, magnitude),
        values::<T>(seed + 1, magnitude),
        values::<T>(seed + 2, magnitude),
        values::<T>(seed + 3, magnitude),
        values::<T>(seed + 4, magnitude),
    );
    let w = T::from_f64(0.625).unwrap();
    let dof = T::from_f64(4.0).unwrap();
    let gm = MatrixData::new_ref::<M, M, T>(&g);
    let rm = MatrixData::new_ref::<M, M, T>(&r);
    let ym = MatrixData::new_ref::<M, 1, T>(&y);
    let s_inv = MatrixData::new_ref::<M, M, T>(&a);

    let new = student_t_weight(dof, &gm, &rm, &ym, &s_inv);
    let old = old_student_t_weight(dof, &gm, &rm, &ym, &s_inv);
    assert_bits_eq(&[new], &[old], "Student-t weight");

    let (mut new, mut old) = (a, a);
    scale(&mut MatrixData::new_mut::<N, M, T>(&mut new), w);
    for value in &mut old[..N * M] {
        *value = *value * w;
    }
    assert_bits_eq(&new, &old, "scaled A");

    let (mut new, mut old) = (b, b);
    write_scaled_covariance(&gm, &rm, w, &mut MatrixData::new_mut::<M, M, T>(&mut new));
    {
        let S = &mut MatrixData::new_mut::<M, M, T>(&mut old);
        for i in 0..M {
            for j in 0..M {
                S.set(i, j, w * gm.get(i, j) + rm.get(i, j));
            }
        }
    }
    assert_bits_eq(&new, &old, "S = w×G + R");

    let am = MatrixData::new_ref::<N, N, T>(&a);
    let (mut new, mut old) = (b, b);
    subtract_scaled_with_transpose(&mut MatrixData::new_mut::<N, N, T>(&mut new), &am, w);
    {
        let P = &mut MatrixData::new_mut::<N, N, T>(&mut old);
        for i in 0..N {
            for j in 0..N {
                let value = P.get(i, j) - w * (am.get(i, j) + am.get(j, i));
                P.set(i, j, value);
            }
        }
    }
    assert_bits_eq(&new, &old, "P − w×(A + Aᵀ)");

    let an = MatrixData::new_ref::<N, M, T>(&a);
    let bn = MatrixData::new_ref::<N, M, T>(&g);
    let (mut new, mut old) = (b, b);
    add_scaled_lower_product_mirrored(&mut MatrixData::new_mut::<N, N, T>(&mut new), &an, &bn, w);
    {
        let P = &mut MatrixData::new_mut::<N, N, T>(&mut old);
        for i in 0..N {
            for j in 0..=i {
                let product: T = (0..M).map(|l| an.get(i, l) * bn.get(j, l)).sum();
                let value = P.get(i, j) + w * product;
                P.set(i, j, value);
                P.set(j, i, value);
            }
        }
    }
    assert_bits_eq(&new, &old, "P + w×A×Bᵀ");
}

#[test]
fn robust_helpers_match_the_old_loops() {
    for seed in 0..4 {
        assert_helpers_match::<2, 1, f32>(seed, 1.0);
        assert_helpers_match::<4, 2, f32>(seed, 1.0);
        assert_helpers_match::<6, 3, f64>(seed, 1.0);
    }
}

#[test]
#[cfg(feature = "fixed")]
fn robust_helpers_match_the_old_loops_fixed() {
    for seed in 0..4 {
        assert_helpers_match::<4, 2, fixed::types::I16F16>(seed, 0.25);
        assert_helpers_match::<6, 3, fixed::types::I32F32>(seed, 1.0);
    }
}
