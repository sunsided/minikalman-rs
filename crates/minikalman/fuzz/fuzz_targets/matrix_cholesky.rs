#![no_main]

use libfuzzer_sys::fuzz_target;
use minikalman::matrix::MatrixDataArray;
use minikalman::prelude::*;

const N: usize = 4;

fuzz_target!(|data: [f32; N * N]| {
    // Skip non-finite inputs
    if data.iter().any(|v| !v.is_finite()) {
        return;
    }

    // Build symmetric matrix: A = (M + M^T) / 2 + kI
    let mut m_data = [0.0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            let sym = (data[i * N + j] + data[j * N + i]) * 0.5;
            m_data[i * N + j] = sym;
        }
        // Regularize diagonal
        m_data[i * N + i] += 1.0;
    }

    let mut m = MatrixDataArray::<N, N, { N * N }, f32>::new(m_data);

    // Attempt Cholesky decomposition
    if m.cholesky_decompose_lower() {
        // If successful, attempt inversion
        let mut inv = MatrixDataArray::<N, N, { N * N }, f32>::new([0.0; N * N]);
        m.invert_l_cholesky(&mut inv);

        // Verify output is finite
        for &v in inv.as_slice() {
            assert!(v.is_finite(), "Inverse contains non-finite value");
        }
    }
});
