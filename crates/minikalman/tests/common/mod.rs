#![allow(dead_code)]

use proptest::prelude::*;

use minikalman::matrix::{MatrixDataArray, MatrixDataBoxed};
use minikalman::prelude::*;

/// Tolerance for floating-point comparisons in proptest assertions.
pub const EPS: f32 = 1e-3;

/// A more relaxed tolerance for complex operations (e.g. matrix inversion).
pub const EPS_RELAXED: f32 = 1e-2;

/// Generates finite `f32` values in range `[-1e3..1e3]`.
pub fn finite_f32() -> impl Strategy<Value = f32> {
    (-1e3f32..1e3f32).prop_filter("exclude non-finite", |x| x.is_finite())
}

/// Generates finite `f32` values in range `[-5..5]` for well-conditioned filter tests.
pub fn small_finite_f32() -> impl Strategy<Value = f32> {
    (-5f32..5f32).prop_filter("exclude non-finite", |x| x.is_finite())
}

/// Generates a `Vec<T>` of length `len` with small finite entries.
pub fn small_finite_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(small_finite_f32(), len..=len)
}

/// Generates a fixed-size array strategy with small finite entries.
pub fn small_fixed_vec<const N: usize>() -> impl Strategy<Value = [f32; N]> {
    small_finite_vec(N).prop_map(|v| {
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(&v);
        arr
    })
}

/// Generates finite `f64` values in range `[-1e3..1e3]`.
pub fn finite_f64() -> impl Strategy<Value = f64> {
    (-1e3f64..1e3f64).prop_filter("exclude non-finite", |x| x.is_finite())
}

/// Generates a `Vec<T>` of length `len` with finite entries.
pub fn finite_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(finite_f32(), len..=len)
}

/// Generates a fixed-size array strategy for small arrays.
pub fn finite_fixed_vec<const N: usize>() -> impl Strategy<Value = [f32; N]> {
    finite_vec(N).prop_map(|v| {
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(&v);
        arr
    })
}

// ---- Concrete identity / zero matrices for test sizes ----

pub fn identity_2() -> MatrixDataArray<2, 2, 4, f32> {
    MatrixDataArray::new([1.0, 0.0, 0.0, 1.0])
}

pub fn identity_3() -> MatrixDataArray<3, 3, 9, f32> {
    MatrixDataArray::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

pub fn zero_2x2() -> MatrixDataArray<2, 2, 4, f32> {
    MatrixDataArray::new([0.0; 4])
}

pub fn zero_2x3() -> MatrixDataArray<2, 3, 6, f32> {
    MatrixDataArray::new([0.0; 6])
}

pub fn zero_3x2() -> MatrixDataArray<3, 2, 6, f32> {
    MatrixDataArray::new([0.0; 6])
}

pub fn zero_3x3() -> MatrixDataArray<3, 3, 9, f32> {
    MatrixDataArray::new([0.0; 9])
}

pub fn zero_3x4() -> MatrixDataArray<3, 4, 12, f32> {
    MatrixDataArray::new([0.0; 12])
}

pub fn zero_4x4() -> MatrixDataArray<4, 4, 16, f32> {
    MatrixDataArray::new([0.0; 16])
}

pub fn zero_2x4() -> MatrixDataArray<2, 4, 8, f32> {
    MatrixDataArray::new([0.0; 8])
}

pub fn zero_4x2() -> MatrixDataArray<4, 2, 8, f32> {
    MatrixDataArray::new([0.0; 8])
}

pub fn zero_4x3() -> MatrixDataArray<4, 3, 12, f32> {
    MatrixDataArray::new([0.0; 12])
}

// ---- Random matrix strategies (concrete sizes) ----

pub fn random_matrix_2x2() -> impl Strategy<Value = MatrixDataArray<2, 2, 4, f32>> {
    finite_vec(4).prop_map(|v| {
        let mut data = [0.0f32; 4];
        data.copy_from_slice(&v);
        MatrixDataArray::new(data)
    })
}

pub fn random_matrix_3x3() -> impl Strategy<Value = MatrixDataArray<3, 3, 9, f32>> {
    finite_vec(9).prop_map(|v| {
        let mut data = [0.0f32; 9];
        data.copy_from_slice(&v);
        MatrixDataArray::new(data)
    })
}

pub fn random_matrix_2x3() -> impl Strategy<Value = MatrixDataArray<2, 3, 6, f32>> {
    finite_vec(6).prop_map(|v| {
        let mut data = [0.0f32; 6];
        data.copy_from_slice(&v);
        MatrixDataArray::new(data)
    })
}

pub fn random_matrix_3x2() -> impl Strategy<Value = MatrixDataArray<3, 2, 6, f32>> {
    finite_vec(6).prop_map(|v| {
        let mut data = [0.0f32; 6];
        data.copy_from_slice(&v);
        MatrixDataArray::new(data)
    })
}

pub fn random_matrix_3x4() -> impl Strategy<Value = MatrixDataArray<3, 4, 12, f32>> {
    finite_vec(12).prop_map(|v| {
        let mut data = [0.0f32; 12];
        data.copy_from_slice(&v);
        MatrixDataArray::new(data)
    })
}

pub fn random_matrix_4x4() -> impl Strategy<Value = MatrixDataArray<4, 4, 16, f32>> {
    finite_vec(16).prop_map(|v| {
        let mut data = [0.0f32; 16];
        data.copy_from_slice(&v);
        MatrixDataArray::new(data)
    })
}

pub fn random_matrix_boxed_3x3() -> impl Strategy<Value = MatrixDataBoxed<3, 3, f32>> {
    finite_vec(9).prop_map(MatrixDataBoxed::new)
}

// ---- SPD matrix strategies (concrete sizes) ----

/// Generates a 2x2 SPD matrix via L*L^T + eps*I.
pub fn spd_matrix_2x2() -> impl Strategy<Value = MatrixDataArray<2, 2, 4, f32>> {
    (
        proptest::collection::vec(-10f32..10f32, 1..=1),
        proptest::collection::vec(-10f32..10f32, 2..=2),
    )
        .prop_map(|(offdiag, diag)| {
            let l_data = [0.5 + diag[0].abs(), 0.0, offdiag[0], 0.5 + diag[1].abs()];
            let l = MatrixDataArray::<2, 2, 4, f32>::new(l_data);
            let mut m = zero_2x2();
            l.mult_transb(&l, &mut m);
            m.as_mut_slice()[0] += 1e-3;
            m.as_mut_slice()[3] += 1e-3;
            m
        })
}

/// Generates a 3x3 SPD matrix via L*L^T + eps*I.
pub fn spd_matrix_3x3() -> impl Strategy<Value = MatrixDataArray<3, 3, 9, f32>> {
    (
        proptest::collection::vec(-10f32..10f32, 3..=3),
        proptest::collection::vec(-10f32..10f32, 3..=3),
    )
        .prop_map(|(offdiag, diag)| {
            let mut l_data = [0.0f32; 9];
            l_data[0] = 0.5 + diag[0].abs();
            l_data[3] = offdiag[0];
            l_data[4] = 0.5 + diag[1].abs();
            l_data[6] = offdiag[1];
            l_data[7] = offdiag[2];
            l_data[8] = 0.5 + diag[2].abs();
            let l = MatrixDataArray::<3, 3, 9, f32>::new(l_data);
            let mut m = zero_3x3();
            l.mult_transb(&l, &mut m);
            for i in 0..3 {
                m.as_mut_slice()[i * 3 + i] += 1e-3;
            }
            m
        })
}

/// Generates a 4x4 SPD matrix via L*L^T + eps*I.
pub fn spd_matrix_4x4() -> impl Strategy<Value = MatrixDataArray<4, 4, 16, f32>> {
    (
        proptest::collection::vec(-10f32..10f32, 6..=6),
        proptest::collection::vec(-10f32..10f32, 4..=4),
    )
        .prop_map(|(offdiag, diag)| {
            let mut l_data = [0.0f32; 16];
            l_data[0] = 0.5 + diag[0].abs();
            l_data[4] = offdiag[0];
            l_data[5] = 0.5 + diag[1].abs();
            l_data[8] = offdiag[1];
            l_data[9] = offdiag[2];
            l_data[10] = 0.5 + diag[2].abs();
            l_data[12] = offdiag[3];
            l_data[13] = offdiag[4];
            l_data[14] = offdiag[5];
            l_data[15] = 0.5 + diag[3].abs();
            let l = MatrixDataArray::<4, 4, 16, f32>::new(l_data);
            let mut m = zero_4x4();
            l.mult_transb(&l, &mut m);
            for i in 0..4 {
                m.as_mut_slice()[i * 4 + i] += 1e-3;
            }
            m
        })
}

// ---- Reference implementations ----

pub fn ref_mult(a: &[f32], b: &[f32], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<f32> {
    let rows_b = cols_a;
    let mut c = vec![0.0f32; rows_a * cols_b];
    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0f32;
            for k in 0..rows_b {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            c[i * cols_b + j] = sum;
        }
    }
    c
}

pub fn ref_add(a: &[f32], b: &[f32], len: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; len];
    for i in 0..len {
        c[i] = a[i] + b[i];
    }
    c
}

pub fn ref_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

// ---- Assertions ----

pub fn assert_matrix_close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "matrix length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let max_abs = av.abs().max(bv.abs());
        let tol = eps + eps * max_abs;
        assert!(
            diff <= tol,
            "element {} differs: {} vs {} (diff={}, tol={})",
            i,
            av,
            bv,
            diff,
            tol
        );
    }
}

pub fn is_symmetric(data: &[f32], n: usize, eps: f32) -> bool {
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (data[i * n + j] - data[j * n + i]).abs();
            if diff > eps {
                return false;
            }
        }
    }
    true
}
