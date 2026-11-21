//! The innovation covariance `S = H×P⁻×Hᵀ + R` that an observation exposes after a
//! correction, for every number of measurements and every outcome.

#![cfg(all(feature = "std", feature = "alloc"))]
#![forbid(unsafe_code)]

use minikalman::buffers::builder::BufferBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;

/// Number of states of the test filter; larger than every observation below, and smaller than
/// the largest, so both `M < N` and `M > N` are covered.
const N: usize = 4;

/// Returns entry `(i, j)` of a deterministic, well-conditioned symmetric positive definite
/// `n×n` matrix scaled by `scale`.
fn spd(n: usize, i: usize, j: usize, scale: f64) -> f64 {
    let dot: f64 = (0..n)
        .map(|k| ((i * n + k) as f64 * 0.37).sin() * ((j * n + k) as f64 * 0.37).sin())
        .sum();
    scale * (dot + if i == j { 1.0 } else { 0.0 })
}

/// Corrects a 4-state filter with an `M`-measurement observation in `update` form, with the
/// gate `gate` and, if `indefinite`, a negative definite `R` that makes `S` indefinite; with
/// `robust`, through the Student-t correction. Returns the status, the observation's `S` and
/// the expected `H×P⁻×Hᵀ + R` of the prior.
fn correct<const M: usize>(
    update: CovarianceUpdate,
    gate: Option<f64>,
    indefinite: bool,
    robust: bool,
) -> (CorrectionStatus, Vec<f64>, Vec<f64>) {
    let builder = KalmanFilterBuilder::<N, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<M>();
    measurement.set_covariance_update(update);
    measurement.set_gate(gate);

    let p: Vec<f64> = (0..N * N).map(|k| spd(N, k / N, k % N, 1.0)).collect();
    let h: Vec<f64> = (0..M * N).map(|k| ((k as f64) * 0.71).cos()).collect();
    let r_scale = if indefinite { -50.0 } else { 0.5 };
    let r: Vec<f64> = (0..M * M).map(|k| spd(M, k / M, k % M, r_scale)).collect();
    filter
        .estimate_covariance_mut()
        .as_mut_slice()
        .copy_from_slice(&p);
    measurement
        .observation_matrix_mut()
        .as_mut_slice()
        .copy_from_slice(&h);
    measurement
        .measurement_noise_covariance_mut()
        .as_mut_slice()
        .copy_from_slice(&r);
    for i in 0..M {
        measurement
            .measurement_vector_mut()
            .set_at(i, 0, 1.0 + i as f64);
    }

    let mut expected = vec![0.0; M * M];
    for i in 0..M {
        for j in 0..M {
            let hph: f64 = (0..N)
                .map(|k| {
                    (0..N)
                        .map(|l| h[i * N + k] * p[k * N + l] * h[j * N + l])
                        .sum::<f64>()
                })
                .sum();
            expected[i * M + j] = hph + r[i * M + j];
        }
    }

    let status = if robust {
        let mut workspace = BufferBuilder::temp_S_inv::<M>().new();
        let student_t = RobustCorrection {
            weight: RobustWeight::StudentT { dof: 4.0 },
            iterations: 10,
        };
        filter.correct_robust(&mut measurement, &mut workspace, student_t)
    } else {
        filter.correct(&mut measurement)
    };
    let s = measurement.innovation_covariance().as_slice()[..M * M].to_vec();
    (status, s, expected)
}

/// Asserts that `S` holds `H×P⁻×Hᵀ + R` after an applied, a gated and a rejected correction,
/// plain and robust, in both covariance forms.
fn assert_s_is_kept<const M: usize>() {
    for update in [CovarianceUpdate::Standard, CovarianceUpdate::Joseph] {
        for (gate, indefinite, robust, status) in [
            (None, false, false, CorrectionStatus::Applied),
            (Some(0.0), false, false, CorrectionStatus::Gated),
            (None, true, false, CorrectionStatus::NotPositiveDefinite),
            (None, false, true, CorrectionStatus::Applied),
            (Some(0.0), false, true, CorrectionStatus::Gated),
            (None, true, true, CorrectionStatus::NotPositiveDefinite),
        ] {
            let (actual, s, expected) = correct::<M>(update, gate, indefinite, robust);
            let same = actual == status || (status.is_applied() && actual.is_applied());
            assert!(same, "M = {M}, {update:?}, robust {robust}: {actual:?}");
            for (k, (a, e)) in s.iter().zip(&expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-12 * (1.0 + e.abs()),
                    "M = {M}, {update:?}, {status:?}, robust {robust}: S[{k}] = {a}, expected {e}"
                );
            }
        }
    }
}

/// `S` is never left factored, for one to five measurements. An applied robust correction
/// may also report `NotConverged`, which is applied too.
#[test]
fn innovation_covariance_is_kept_after_every_correction() {
    assert_s_is_kept::<1>();
    assert_s_is_kept::<2>();
    assert_s_is_kept::<3>();
    assert_s_is_kept::<4>();
    assert_s_is_kept::<5>();
}
