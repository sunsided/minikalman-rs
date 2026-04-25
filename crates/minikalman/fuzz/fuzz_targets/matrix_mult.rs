#![no_main]

use libfuzzer_sys::fuzz_target;
use minikalman::matrix::MatrixDataArray;
use minikalman::prelude::*;

const ROWS_A: usize = 3;
const COLS_A: usize = 4;
const COLS_B: usize = 3;
const ROWS_B: usize = COLS_A; // must match for multiplication

const LEN_A: usize = ROWS_A * COLS_A;
const LEN_B: usize = ROWS_B * COLS_B;

#[derive(Debug)]
struct Input {
    a: [f32; LEN_A],
    b: [f32; LEN_B],
}

impl arbitrary::Arbitrary<'_> for Input {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        let mut a = [0.0f32; LEN_A];
        let mut b = [0.0f32; LEN_B];
        for v in a.iter_mut() {
            *v = f32::arbitrary(u)?;
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        for v in b.iter_mut() {
            *v = f32::arbitrary(u)?;
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        Ok(Input { a, b })
    }
}

fuzz_target!(|input: Input| {
    let a = MatrixDataArray::<ROWS_A, COLS_A, { LEN_A }, f32>::new(input.a);
    let b = MatrixDataArray::<ROWS_B, COLS_B, { LEN_B }, f32>::new(input.b);

    // mult
    let mut c1 =
        MatrixDataArray::<ROWS_A, COLS_B, { ROWS_A * COLS_B }, f32>::new([0.0; ROWS_A * COLS_B]);
    a.mult(&b, &mut c1);

    // mult_buffered
    let mut c2 =
        MatrixDataArray::<ROWS_A, COLS_B, { ROWS_A * COLS_B }, f32>::new([0.0; ROWS_A * COLS_B]);
    let mut aux = [0.0f32; COLS_A];
    a.mult_buffered(&b, &mut c2, &mut aux);

    // Verify buffered and non-buffered agree
    for (v1, v2) in c1.as_slice().iter().zip(c2.as_slice().iter()) {
        if !v1.is_finite() || !v2.is_finite() {
            // Both NaN is expected to match; NaN != NaN by IEEE 754
            if v1.is_nan() && v2.is_nan() {
                continue;
            }
            assert!(
                *v1 == *v2,
                "mult and mult_buffered disagree on non-finite: {} vs {}",
                v1,
                v2
            );
        } else {
            let diff = (v1 - v2).abs();
            assert!(
                diff < 1e-3 || diff < 1e-3 * v1.abs().max(v2.abs()),
                "mult and mult_buffered disagree: {} vs {}",
                v1,
                v2
            );
        }
    }

    // mult_transb: A is ROWS_A x COLS_A, B^T is COLS_A x ROWS_B = COLS_A x COLS_B
    // But mult_transb expects B: Matrix<U, COLS, T> where COLS = COLS_A
    // So we need B with ROWS_B=COLS_A rows... which doesn't match our b.
    // Skip mult_transb for mismatched dimensions; test with compatible input.
});
