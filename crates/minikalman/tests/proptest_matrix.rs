#![cfg(feature = "std")]
#![forbid(unsafe_code)]

mod common;

use common::*;
use minikalman::matrix::{MatrixDataArray, MatrixDataBoxed};
use minikalman::prelude::*;
use proptest::prelude::*;

// ============================================================================
// A + 0 == A
// ============================================================================

proptest! {
    #[test]
    fn add_zero_is_identity_2x2(
        mat in random_matrix_2x2()
    ) {
        let zero = zero_2x2();
        let mut result = mat.clone();
        result.add_inplace_a(&zero);
        assert_matrix_close(mat.as_slice(), result.as_slice(), EPS);
    }

    #[test]
    fn add_zero_is_identity_3x3(
        mat in random_matrix_3x3()
    ) {
        let zero = zero_3x3();
        let mut result = mat.clone();
        result.add_inplace_a(&zero);
        assert_matrix_close(mat.as_slice(), result.as_slice(), EPS);
    }
}

// ============================================================================
// A + B == B + A
// ============================================================================

proptest! {
    #[test]
    fn add_commutative_3x3(
        a in random_matrix_3x3(),
        b in random_matrix_3x3()
    ) {
        let mut a_plus_b = a.clone();
        a_plus_b.add_inplace_a(&b);

        let mut b_plus_a = b.clone();
        b_plus_a.add_inplace_a(&a);

        assert_matrix_close(a_plus_b.as_slice(), b_plus_a.as_slice(), EPS);
    }
}

// ============================================================================
// (A + B) + C == A + (B + C)
// ============================================================================

proptest! {
    #[test]
    fn add_associative_2x2(
        a in random_matrix_2x2(),
        b in random_matrix_2x2(),
        c in random_matrix_2x2()
    ) {
        let mut left = a.clone();
        left.add_inplace_a(&b);
        left.add_inplace_a(&c);

        let mut right = b.clone();
        right.add_inplace_a(&c);
        let mut a_plus = a.clone();
        a_plus.add_inplace_a(&right);

        assert_matrix_close(left.as_slice(), a_plus.as_slice(), EPS);
    }
}

// ============================================================================
// A - A == 0
// ============================================================================

proptest! {
    #[test]
    fn sub_self_is_zero_3x3(
        a in random_matrix_3x3()
    ) {
        let mut result = zero_3x3();
        a.sub(&a, &mut result);
        assert_matrix_close(result.as_slice(), &[0.0f32; 9], EPS);
    }

    #[test]
    fn sub_self_is_zero_boxed_3x3(
        a in random_matrix_boxed_3x3()
    ) {
        let mut result = MatrixDataBoxed::<3, 3, f32>::new(vec![0.0f32; 9].into_boxed_slice());
        a.sub(&a, &mut result);
        assert_matrix_close(result.as_slice(), &[0.0f32; 9], EPS);
    }
}

// ============================================================================
// A * I == A  and  I * A == A
// ============================================================================

proptest! {
    #[test]
    fn mult_identity_right_3x3(
        a in random_matrix_3x3()
    ) {
        let identity = identity_3();
        let mut result = zero_3x3();
        a.mult(&identity, &mut result);
        assert_matrix_close(a.as_slice(), result.as_slice(), EPS);
    }

    #[test]
    fn mult_identity_left_3x3(
        a in random_matrix_3x3()
    ) {
        let identity = identity_3();
        let mut result = zero_3x3();
        identity.mult(&a, &mut result);
        assert_matrix_close(a.as_slice(), result.as_slice(), EPS);
    }
}

// ============================================================================
// A * (B + C) == A*B + A*C
// ============================================================================

proptest! {
    #[test]
    fn mult_distributive_2x3(
        a in random_matrix_2x2(),
        b in random_matrix_2x3(),
        c in random_matrix_2x3()
    ) {
        let mut b_plus_c = b.clone();
        b_plus_c.add_inplace_a(&c);
        let mut left = zero_2x3();
        a.mult(&b_plus_c, &mut left);

        let mut ab = zero_2x3();
        a.mult(&b, &mut ab);
        let mut ac = zero_2x3();
        a.mult(&c, &mut ac);
        ab.add_inplace_a(&ac);

        assert_matrix_close(left.as_slice(), ab.as_slice(), 1e-1);
    }
}

// ============================================================================
// (A*B)^T == B^T * A^T
// ============================================================================

proptest! {
    #[test]
    fn mult_transpose_property_2x3(
        a in random_matrix_2x3(),
        b in random_matrix_3x2()
    ) {
        let mut ab = zero_2x2();
        a.mult(&b, &mut ab);
        let ab_t = ref_transpose(ab.as_slice(), 2, 2);

        let a_t = ref_transpose(a.as_slice(), 2, 3);
        let b_t = ref_transpose(b.as_slice(), 3, 2);
        let bta_t = ref_mult(&b_t, &a_t, 2, 3, 2);

        assert_matrix_close(&ab_t, &bta_t, EPS);
    }
}

// ============================================================================
// mult_buffered(A, B) == mult(A, B)
// ============================================================================

proptest! {
    #[test]
    fn mult_buffered_agrees_with_mult_2x3(
        a in random_matrix_2x3(),
        b in random_matrix_3x4()
    ) {
        let mut c1 = zero_2x4();
        a.mult(&b, &mut c1);

        let mut c2 = zero_2x4();
        let mut aux = [0.0f32; 3];
        a.mult_buffered(&b, &mut c2, &mut aux);

        assert_matrix_close(c1.as_slice(), c2.as_slice(), EPS);
    }

    #[test]
    fn mult_buffered_agrees_with_mult_3x3(
        a in random_matrix_3x3(),
        b in random_matrix_3x3()
    ) {
        let mut c1 = zero_3x3();
        a.mult(&b, &mut c1);

        let mut c2 = zero_3x3();
        let mut aux = [0.0f32; 3];
        a.mult_buffered(&b, &mut c2, &mut aux);

        assert_matrix_close(c1.as_slice(), c2.as_slice(), EPS);
    }
}

// ============================================================================
// mult_transb(A, B) == A * B^T
// ============================================================================

proptest! {
    #[test]
    fn mult_transb_matches_reference_2x2(
        a in random_matrix_2x3(),
        b in random_matrix_2x3()
    ) {
        let mut c = zero_2x2();
        a.mult_transb(&b, &mut c);

        let b_t = ref_transpose(b.as_slice(), 2, 3);
        let expected = ref_mult(a.as_slice(), &b_t, 2, 3, 2);

        assert_matrix_close(c.as_slice(), &expected, EPS);
    }
}

// ============================================================================
// multadd_transb(A, B, C) == C + A*B^T
// ============================================================================

proptest! {
    #[test]
    fn multadd_transb_matches_reference_2x2(
        a in random_matrix_2x3(),
        b in random_matrix_2x3(),
        c in random_matrix_2x2()
    ) {
        let mut result = c.clone();
        a.multadd_transb(&b, &mut result);

        let b_t = ref_transpose(b.as_slice(), 2, 3);
        let ab_t = ref_mult(a.as_slice(), &b_t, 2, 3, 2);
        let expected = ref_add(c.as_slice(), &ab_t, 4);

        assert_matrix_close(result.as_slice(), &expected, EPS);
    }
}

// ============================================================================
// multscale_transb(A, B, s) == s * A*B^T
// ============================================================================

proptest! {
    #[test]
    fn multscale_transb_matches_reference_2x2(
        a in random_matrix_2x3(),
        b in random_matrix_2x3(),
        s in finite_f32()
    ) {
        let mut c = zero_2x2();
        a.multscale_transb(&b, s, &mut c);

        let b_t = ref_transpose(b.as_slice(), 2, 3);
        let ab_t = ref_mult(a.as_slice(), &b_t, 2, 3, 2);
        let expected: Vec<f32> = ab_t.iter().map(|x| x * s).collect();

        assert_matrix_close(c.as_slice(), &expected, EPS);
    }
}

// ============================================================================
// mult_rowvector(A, x) vs reference
// ============================================================================

proptest! {
    #[test]
    fn mult_rowvector_matches_reference_3x1(
        a in random_matrix_3x2(),
        x_vals in finite_fixed_vec::<2>()
    ) {
        let x = MatrixDataArray::<2, 1, 2, f32>::new(x_vals);
        let mut c = MatrixDataArray::<3, 1, 3, f32>::new([0.0; 3]);
        a.mult_rowvector(&x, &mut c);

        let expected = ref_mult(a.as_slice(), &x_vals, 3, 2, 1);
        assert_matrix_close(c.as_slice(), &expected, EPS);
    }
}

// ============================================================================
// multadd_rowvector(A, x, y) == y + A*x
// ============================================================================

proptest! {
    #[test]
    fn multadd_rowvector_matches_reference_3x1(
        a in random_matrix_3x2(),
        x_vals in finite_fixed_vec::<2>()
    ) {
        let x = MatrixDataArray::<2, 1, 2, f32>::new(x_vals);
        let y_init = [1.0f32, 2.0, 3.0];
        let mut c = MatrixDataArray::<3, 1, 3, f32>::new(y_init);
        a.multadd_rowvector(&x, &mut c);

        let ax = ref_mult(a.as_slice(), &x_vals, 3, 2, 1);
        let expected: Vec<f32> = y_init.iter().zip(ax.iter()).map(|(&y, &ax_val)| y + ax_val).collect();
        assert_matrix_close(c.as_slice(), &expected, EPS);
    }
}

// ============================================================================
// Cholesky: for SPD M, decomposition succeeds and L*L^T ~= M
// ============================================================================

proptest! {
    #[test]
    fn cholesky_reconstructs_spd_3x3(
        spd in spd_matrix_3x3()
    ) {
        let original = spd.clone();
        let mut lower = spd;
        let success = lower.cholesky_decompose_lower();
        prop_assert!(success, "Cholesky should succeed for SPD matrix");

        let mut reconstructed = zero_3x3();
        lower.mult_transb(&lower, &mut reconstructed);

        assert_matrix_close(original.as_slice(), reconstructed.as_slice(), EPS_RELAXED);
    }

    #[test]
    fn cholesky_reconstructs_spd_4x4(
        spd in spd_matrix_4x4()
    ) {
        let original = spd.clone();
        let mut lower = spd;
        let success = lower.cholesky_decompose_lower();
        prop_assert!(success, "Cholesky should succeed for SPD matrix");

        let mut reconstructed = zero_4x4();
        lower.mult_transb(&lower, &mut reconstructed);

        assert_matrix_close(original.as_slice(), reconstructed.as_slice(), EPS_RELAXED);
    }
}

// ============================================================================
// Cholesky + invert: for SPD M, M * M^-1 ~= I
// ============================================================================

proptest! {
    #[test]
    fn cholesky_invert_gives_identity_3x3(
        spd in spd_matrix_3x3()
    ) {
        let mut lower = spd;
        let success = lower.cholesky_decompose_lower();
        prop_assert!(success, "Cholesky should succeed for SPD matrix");

        let mut inv = zero_3x3();
        lower.invert_l_cholesky(&mut inv);

        let mut m = zero_3x3();
        lower.mult_transb(&lower, &mut m);

        let mut product = zero_3x3();
        m.mult(&inv, &mut product);

        assert_matrix_close(product.as_slice(), identity_3().as_slice(), 1e-1);
    }
}

// ============================================================================
// Boxed matrix: mult agrees with array
// ============================================================================

proptest! {
    #[test]
    fn boxed_mult_agrees_with_array_3x3(
        vals in finite_vec(9)
    ) {
        let mut arr_data = [0.0f32; 9];
        arr_data.copy_from_slice(&vals);
        let arr = MatrixDataArray::<3, 3, 9, f32>::new(arr_data);
        let boxed = MatrixDataBoxed::<3, 3, f32>::new(vals.clone());

        let mut arr_result = zero_3x3();
        let mut boxed_result: MatrixDataBoxed<3, 3, f32> = MatrixDataBoxed::new(vec![0.0; 9]);

        arr.mult(&arr, &mut arr_result);
        boxed.mult(&boxed, &mut boxed_result);

        assert_matrix_close(arr_result.as_slice(), boxed_result.as_slice(), EPS);
    }
}
