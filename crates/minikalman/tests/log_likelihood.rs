//! `ln det S` and the Gaussian log-likelihood of a correction, against a direct `f64`
//! computation, and maximum-likelihood fitting of an ARMA(1, 1) model with them.

#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::matrix::Logarithm;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// The number of states of the reference problems.
const N: usize = 6;

/// `ln 2π`.
const LN_TWO_PI: f64 = 1.837_877_066_409_345_5;

/// A deterministic value in `[-1, 1)` for index `i` of sequence `seed`.
fn value(seed: u64, i: usize) -> f64 {
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((i as u64 + 1).wrapping_mul(1_442_695_040_888_963_407));
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// The row-major `n × n` matrix `B×Bᵀ + diagonal×I` with `B` from sequence `seed`, scaled by
/// `scale`.
fn spd(n: usize, seed: u64, diagonal: f64, scale: f64) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n)
                .map(|k| value(seed, i * n + k) * value(seed, j * n + k))
                .sum();
            m[i * n + j] = scale * (dot + if i == j { diagonal } else { 0.0 });
        }
    }
    m
}

/// One correction problem with `m` measurements, row-major in `f64`.
struct Problem {
    /// The number of measurements.
    m: usize,
    /// The prior state (`N` values).
    x: Vec<f64>,
    /// The prior covariance (`N × N`, symmetric positive definite).
    P: Vec<f64>,
    /// The observation matrix (`m × N`).
    H: Vec<f64>,
    /// The measurement noise covariance (`m × m`, symmetric positive definite).
    R: Vec<f64>,
    /// The measurement (`m` values).
    z: Vec<f64>,
}

impl Problem {
    /// A well-conditioned problem; `noise_scale` scales `R`.
    fn new(m: usize, seed: u64, noise_scale: f64) -> Self {
        Self {
            m,
            x: (0..N).map(|i| value(seed, i)).collect(),
            P: spd(N, seed + 1, 1.0, 1.0),
            H: (0..m * N).map(|i| value(seed + 2, i)).collect(),
            R: spd(m, seed + 3, 0.5, noise_scale),
            z: (0..m).map(|i| 3.0 * value(seed + 4, i)).collect(),
        }
    }

    /// Returns `(ln det S, NIS, ℓ)` of the prior innovation, computed directly in `f64` with
    /// a Cholesky factorization of `S = H×P×Hᵀ + R`.
    fn reference(&self) -> (f64, f64, f64) {
        let m = self.m;
        let mut S = self.R.clone();
        for i in 0..m {
            for j in 0..m {
                for a in 0..N {
                    for b in 0..N {
                        S[i * m + j] += self.H[i * N + a] * self.P[a * N + b] * self.H[j * N + b];
                    }
                }
            }
        }

        let mut L = vec![0.0; m * m];
        for j in 0..m {
            let pivot = S[j * m + j] - (0..j).map(|k| L[j * m + k] * L[j * m + k]).sum::<f64>();
            L[j * m + j] = pivot.sqrt();
            for i in (j + 1)..m {
                let sum: f64 = (0..j).map(|k| L[i * m + k] * L[j * m + k]).sum();
                L[i * m + j] = (S[i * m + j] - sum) / L[j * m + j];
            }
        }

        let mut w: Vec<f64> = (0..m)
            .map(|i| self.z[i] - (0..N).map(|k| self.H[i * N + k] * self.x[k]).sum::<f64>())
            .collect();
        for i in 0..m {
            w[i] = (w[i] - (0..i).map(|k| L[i * m + k] * w[k]).sum::<f64>()) / L[i * m + i];
        }

        let log_det = 2.0 * (0..m).map(|i| L[i * m + i].ln()).sum::<f64>();
        let nis = w.iter().map(|w| w * w).sum::<f64>();
        (log_det, nis, -0.5 * (m as f64 * LN_TWO_PI + log_det + nis))
    }
}

/// Converts an `f64` to `T`.
fn t<T: MatrixDataType>(value: f64) -> T {
    T::from_f64(value).unwrap()
}

/// Asserts `|actual − expected| ≤ tolerance×max(1, |expected|)`.
fn assert_close<T: Into<f64>>(what: &str, actual: T, expected: f64, tolerance: f64) {
    let actual = actual.into();
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
        "{what}: {actual} != {expected}"
    );
}

/// How the problem is corrected.
#[derive(Debug, Clone, Copy)]
enum Path {
    /// `RegularKalman::correct`.
    Regular,
    /// `ExtendedKalman::correct_nonlinear` with the linear `h(x) = H×x`.
    Extended,
    /// `RegularKalman::correct_robust` (Student-t, 4 degrees of freedom).
    Robust,
}

/// Runs one correction of `problem` with `M` measurements on the given path and returns the
/// status, `ln det S` and `ℓ` the observation reports.
fn run<const M: usize, T>(
    problem: &Problem,
    path: Path,
    update: CovarianceUpdate,
    gate: Option<f64>,
) -> (CorrectionStatus, T, T)
where
    T: Logarithm + Default + Into<f64>,
{
    match path {
        Path::Regular | Path::Robust => {
            let builder = RegularBuilder::<N, T>::default();
            let mut filter = builder.build();
            let mut measurement = builder.observations().build::<M>();
            copy(filter.state_vector_mut().as_mut_slice(), &problem.x);
            copy(filter.estimate_covariance_mut().as_mut_slice(), &problem.P);
            copy(
                measurement.observation_matrix_mut().as_mut_slice(),
                &problem.H,
            );
            copy(
                measurement
                    .measurement_noise_covariance_mut()
                    .as_mut_slice(),
                &problem.R,
            );
            copy(
                measurement.measurement_vector_mut().as_mut_slice(),
                &problem.z,
            );
            measurement.set_covariance_update(update);
            measurement.set_gate(gate.map(t));
            let status = if let Path::Robust = path {
                let mut workspace = BufferBuilder::temp_S_inv::<M>().new();
                let robust = RobustCorrection {
                    weight: RobustWeight::StudentT { dof: t(4.0) },
                    iterations: 10,
                };
                filter.correct_robust(&mut measurement, &mut workspace, robust)
            } else {
                filter.correct(&mut measurement)
            };
            (
                status,
                measurement.log_det_innovation_covariance(),
                measurement.log_likelihood(),
            )
        }
        Path::Extended => {
            let builder = ExtendedBuilder::<N, T>::default();
            let mut filter = builder.build();
            let mut measurement = builder.observations().build::<M>();
            copy(filter.state_vector_mut().as_mut_slice(), &problem.x);
            copy(filter.estimate_covariance_mut().as_mut_slice(), &problem.P);
            copy(
                measurement.observation_jacobian_matrix_mut().as_mut_slice(),
                &problem.H,
            );
            copy(
                measurement
                    .measurement_noise_covariance_mut()
                    .as_mut_slice(),
                &problem.R,
            );
            copy(
                measurement.measurement_vector_mut().as_mut_slice(),
                &problem.z,
            );
            measurement.set_covariance_update(update);
            measurement.set_gate(gate.map(t));
            let H = problem.H.clone();
            let status = filter.correct_nonlinear(&mut measurement, |x, h| {
                for i in 0..M {
                    let row = (0..N)
                        .map(|k| t::<T>(H[i * N + k]) * x.get_at(k, 0))
                        .sum::<T>();
                    h.set_at(i, 0, row);
                }
            });
            (
                status,
                measurement.log_det_innovation_covariance(),
                measurement.log_likelihood(),
            )
        }
    }
}

/// Copies `source` into the start of `target`, converted to `T`.
fn copy<T: MatrixDataType>(target: &mut [T], source: &[f64]) {
    for (target, &source) in target.iter_mut().zip(source) {
        *target = t(source);
    }
}

/// Checks every path, both covariance forms and a gated correction for `M` measurements.
fn check_all<const M: usize, T>(tolerance: f64)
where
    T: Logarithm + Default + Into<f64>,
{
    for seed in [1, 20, 300] {
        let problem = Problem::new(M, seed, 1.0);
        let (log_det, _, likelihood) = problem.reference();
        for path in [Path::Regular, Path::Extended, Path::Robust] {
            for update in [CovarianceUpdate::Standard, CovarianceUpdate::Joseph] {
                for gate in [None, Some(0.0)] {
                    let what = format!("M = {M}, seed {seed}, {path:?}, {update:?}, gate {gate:?}");
                    let (status, ld, ll) = run::<M, T>(&problem, path, update, gate);
                    let expected: &[CorrectionStatus] = match (gate, path) {
                        (Some(_), _) => &[CorrectionStatus::Gated],
                        (None, Path::Robust) => {
                            &[CorrectionStatus::Applied, CorrectionStatus::NotConverged]
                        }
                        (None, _) => &[CorrectionStatus::Applied],
                    };
                    assert!(expected.contains(&status), "{what}: {status:?}");
                    assert_close(&format!("ln det S, {what}"), ld, log_det, tolerance);
                    assert_close(&format!("ℓ, {what}"), ll, likelihood, tolerance);
                }
            }
        }
    }
}

/// `ln det S` and `ℓ` match the direct computation for one to five measurements, in `f64`
/// and `f32`, for the regular, extended and robust corrections in both covariance forms,
/// applied and gated. One to three measurements take the `L×D×Lᵀ` path, four and five the
/// Cholesky path.
#[test]
fn matches_the_reference_for_one_to_five_measurements() {
    check_all::<1, f64>(1e-12);
    check_all::<2, f64>(1e-12);
    check_all::<3, f64>(1e-12);
    check_all::<4, f64>(1e-12);
    check_all::<5, f64>(1e-12);

    check_all::<1, f32>(1e-5);
    check_all::<2, f32>(1e-5);
    check_all::<3, f32>(1e-5);
    check_all::<4, f32>(1e-5);
    check_all::<5, f32>(1e-5);
}

/// `ln det S` is a sum of logarithms, so it stays finite and accurate in `f32` where
/// `det S` itself (about `10⁴⁵` here) does not fit.
#[test]
fn log_determinant_does_not_need_det_s_to_fit_f32() {
    let problem = Problem::new(5, 7, 1e9);
    let (log_det, _, likelihood) = problem.reference();
    assert!(log_det > f64::from(f32::MAX).ln());

    let (status, ld, ll) = run::<5, f32>(&problem, Path::Regular, CovarianceUpdate::Joseph, None);
    assert_eq!(status, CorrectionStatus::Applied);
    assert_close("ln det S", ld, log_det, 1e-5);
    assert_close("ℓ", ll, likelihood, 1e-5);
}

/// A correction whose `S` is not positive definite reports `NaN`; so does a `NaN` in the
/// innovation, as the NIS does.
#[test]
fn undefined_likelihoods_are_nan() {
    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    // Before the first correction, S is still zero: no density.
    assert!(measurement.log_likelihood().is_nan());

    filter.estimate_covariance_mut().make_identity();
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_identity();
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);
    assert_eq!(filter.correct(&mut measurement), CorrectionStatus::Applied);
    assert!(measurement.log_likelihood().is_finite());

    // S = P + R with R = −2·I is negative definite.
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(-2.0);
    assert_eq!(
        filter.correct(&mut measurement),
        CorrectionStatus::NotPositiveDefinite
    );
    assert!(measurement.log_det_innovation_covariance().is_nan());
    assert!(measurement.log_likelihood().is_nan());

    measurement
        .measurement_noise_covariance_mut()
        .make_identity();
    measurement.measurement_vector_mut().set_at(0, 0, f64::NAN);
    assert_eq!(
        filter.correct(&mut measurement),
        CorrectionStatus::NotPositiveDefinite
    );
    assert!(measurement.log_det_innovation_covariance().is_finite());
    assert!(measurement.log_likelihood().is_nan());
}

/// Simulates `count` samples of the ARMA(1, 1) process `yₖ = φ·yₖ₋₁ + eₖ + θ·eₖ₋₁` with
/// `eₖ ~ N(0, 1)`.
fn arma(phi: f64, theta: f64, count: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 1.0).unwrap();
    let (mut y, mut e) = (0.0, 0.0);
    let mut samples = Vec::with_capacity(count);
    for k in 0..count + 100 {
        let next = noise.sample(&mut rng);
        y = phi * y + next + theta * e;
        e = next;
        if k >= 100 {
            samples.push(y);
        }
    }
    samples
}

/// Returns `Σ ℓₖ` of `samples` under the ARMA(1, 1) model `(φ, θ)` with unit innovation
/// variance, from a Kalman filter in the state-space form `sₖ = [yₖ, θ·eₖ]`:
/// `A = [[φ, 1], [0, 0]]`, `Q = g×gᵀ` with `g = [1, θ]`, `H = [1, 0]` and `R = 0`.
///
/// The filter starts from `x = 0` and `P = I·10`, roughly diffuse, and the first few terms are
/// skipped so that the start does not weigh in.
fn arma_log_likelihood(samples: &[f64], phi: f64, theta: f64) -> f64 {
    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_transition_mut().apply(|a| {
        a.set_at(0, 0, phi);
        a.set_at(0, 1, 1.0);
    });
    filter.direct_process_noise_mut().apply(|q| {
        q.set_at(0, 0, 1.0);
        q.set_symmetric(0, 1, theta);
        q.set_at(1, 1, theta * theta);
    });
    filter.estimate_covariance_mut().make_scalar(10.0);
    measurement.observation_matrix_mut().set_at(0, 0, 1.0);

    let mut sum = 0.0;
    for (k, &y) in samples.iter().enumerate() {
        filter.predict();
        measurement.measurement_vector_mut().set_at(0, 0, y);
        assert_eq!(filter.correct(&mut measurement), CorrectionStatus::Applied);
        if k >= 5 {
            sum += measurement.log_likelihood();
        }
    }
    sum
}

/// Summed over a run, the log-likelihood peaks at the true ARMA(1, 1) parameters on a grid
/// whose step (0.1) is several times the estimator's standard error (about 0.02 for 4000
/// samples).
#[test]
fn arma_likelihood_peaks_at_the_true_parameters() {
    let (phi, theta) = (0.6, 0.3);
    let samples = arma(phi, theta, 4000, 42);

    let mut best = (f64::NEG_INFINITY, 0.0, 0.0);
    for i in 0..=8 {
        for j in 0..=8 {
            let (p, q) = (0.2 + 0.1 * f64::from(i), -0.1 + 0.1 * f64::from(j));
            let likelihood = arma_log_likelihood(&samples, p, q);
            if likelihood > best.0 {
                best = (likelihood, p, q);
            }
        }
    }

    assert!(
        (best.1 - phi).abs() < 1e-9 && (best.2 - theta).abs() < 1e-9,
        "the peak is at φ = {:.1}, θ = {:.1}",
        best.1,
        best.2
    );
}
