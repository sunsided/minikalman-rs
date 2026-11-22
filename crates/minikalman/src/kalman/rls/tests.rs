//! Tests that pin the recursive least-squares estimator to batch least squares, to the
//! regular Kalman filter, and to its forgetting, windup and gating behavior.

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use crate::kalman::CorrectionStatus;
use crate::matrix::Matrix;
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use crate::matrix::MatrixMut;
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use crate::rls::CovarianceForm;
use crate::rls::EstimatorStorage;

/// Both covariance forms, for tests that must hold for each.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const FORMS: [CovarianceForm; 2] = [CovarianceForm::Factored, CovarianceForm::Covariance];

/// A deterministic pseudo-random sequence in `[-1, 1)` (a 64-bit LCG), so the tests need no
/// `rand` and run without `std`.
struct Lcg(u64);

impl Lcg {
    /// Returns the next value in `[-1, 1)`.
    #[allow(clippy::cast_precision_loss)]
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Generates `K` samples `(φ, y)` of `y = φᵀ·θ + v` with `φ = [x, u, 1]` and `|v| ≤ noise`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn dataset<const K: usize>(theta: [f64; 3], noise: f64, seed: u64) -> [([f64; 3], f64); K] {
    let mut rng = Lcg(seed);
    let mut samples = [([0.0; 3], 0.0); K];
    for sample in &mut samples {
        let phi = [rng.next() * 3.0, rng.next(), 1.0];
        let y = phi[0] * theta[0] + phi[1] * theta[1] + phi[2] * theta[2] + noise * rng.next();
        *sample = (phi, y);
    }

    samples
}

/// Solves the 3 × 3 system `m·x = b` by Gaussian elimination with partial pivoting.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn solve3(mut m: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap();
        m.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..3 {
            let f = m[row][col] / m[col][col];
            for c in col..3 {
                m[row][c] -= f * m[col][c];
            }

            b[row] -= f * b[col];
        }
    }

    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let mut v = b[row];
        for c in (row + 1)..3 {
            v -= m[row][c] * x[c];
        }

        x[row] = v / m[row][row];
    }

    x
}

/// Returns `P` of an estimator as a nested array.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn covariance_of<const N: usize>(rls: &crate::rls::BorrowedEstimator<'_, N, f64>) -> [[f64; N]; N] {
    let mut storage = [[0.0; N]; N];
    rls.covariance(&mut crate::matrix::MatrixData::new_mut::<N, N, f64>(
        storage.as_flattened_mut(),
    ));
    storage
}

/// With `μ = 1`, RLS solves the regularized normal equations `(ΦᵀΦ/r + P₀⁻¹)·θ = Φᵀy/r`, and
/// its `P` is the inverse of the information matrix.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
#[allow(clippy::needless_range_loop)]
fn matches_batch_least_squares() {
    let samples = dataset::<40>([0.8, -1.5, 0.3], 0.05, 1);
    let (p0, r) = (100.0, 0.25);

    let mut information = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for i in 0..3 {
        information[i][i] = 1.0 / p0;
    }

    for (phi, y) in &samples {
        for i in 0..3 {
            rhs[i] += phi[i] * y / r;
            for j in 0..3 {
                information[i][j] += phi[i] * phi[j] / r;
            }
        }
    }

    let expected = solve3(information, rhs);

    for form in FORMS {
        let mut storage = EstimatorStorage::<3, f64>::new(0.0);
        let mut rls = storage.build(p0);
        rls.reset_with_form(form, p0);
        rls.set_measurement_variance(r);
        for (phi, y) in &samples {
            assert_eq!(rls.update(phi, *y), CorrectionStatus::Applied);
        }

        for (i, &e) in expected.iter().enumerate() {
            let got = rls.parameters().get(i, 0);
            assert!((got - e).abs() < 1e-12, "{form:?} θ[{i}]: {got} vs {e}");
        }

        // P·(information) = I
        let p = covariance_of(&rls);
        for i in 0..3 {
            for j in 0..3 {
                let product: f64 = (0..3).map(|k| p[i][k] * information[k][j]).sum();
                let identity = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product - identity).abs() < 1e-12,
                    "{form:?} (P·Y)[{i}][{j}]"
                );
            }
        }
    }
}

/// RLS with forgetting `μ` equals a `RegularKalman` with `A = I`, `Q = 0`, `H = φᵀ` that
/// calls `predict_tuned(λ)` with `μ = λ²` before each correction: `θ`, `P` and the NIS.
///
/// Needs exact reciprocals: with only `micromath`, the filter's `1/λ²` and `S⁻¹` are approximate.
#[cfg(any(feature = "std", feature = "libm"))]
#[test]
#[allow(clippy::needless_range_loop)]
fn equals_regular_kalman_with_identity_dynamics() {
    use crate::prelude::*;
    use crate::regular::{FilterStorage, ObservationStorage};

    let lambda = 0.99;
    let (p0, r) = (10.0, 0.5);
    let samples = dataset::<100>([0.5, 2.0, -1.0], 0.1, 7);

    let mut filter_storage = FilterStorage::<3, f64>::new(0.0);
    let mut observation_storage = ObservationStorage::<3, 1, f64>::new(0.0);
    let mut filter = filter_storage.build();
    let mut observation = observation_storage.build();
    filter.state_transition_mut().make_identity();
    filter.estimate_covariance_mut().make_scalar(p0);
    observation
        .measurement_noise_covariance_mut()
        .apply(|m| m.set(0, 0, r));

    for form in FORMS {
        let mut storage = EstimatorStorage::<3, f64>::new(0.0);
        let mut rls = storage.build(p0);
        rls.reset_with_form(form, p0);
        rls.set_forgetting_factor(lambda * lambda);
        rls.set_trace_bound(None);
        rls.set_measurement_variance(r);

        filter.state_vector_mut().apply(|x| x.set_all(0.0));
        filter.estimate_covariance_mut().make_scalar(p0);

        for (phi, y) in &samples {
            filter.predict_tuned(lambda);
            observation.observation_matrix_mut().apply(|h| {
                for (i, &p) in phi.iter().enumerate() {
                    h.set(0, i, p);
                }
            });
            observation
                .measurement_vector_mut()
                .apply(|z| z.set(0, 0, *y));
            assert!(filter.correct(&mut observation).is_applied());
            assert!(rls.update(phi, *y).is_applied());

            let nis = observation.nis();
            assert!(
                (rls.nis() - nis).abs() <= 1e-12 * nis.max(1.0),
                "{form:?} NIS"
            );
        }

        let p = covariance_of(&rls);
        for i in 0..3 {
            let (got, expected) = (rls.parameters().get(i, 0), filter.state_vector().get(i, 0));
            assert!((got - expected).abs() < 1e-12, "{form:?} θ[{i}]");
            for (j, &value) in p[i].iter().enumerate() {
                let expected = filter.estimate_covariance().get(i, j);
                assert!((value - expected).abs() < 1e-12, "{form:?} P[{i}][{j}]");
            }
        }
    }
}

/// Runs `steps` samples of `y = φᵀ·θ` with `φ = [x, u]` and returns the largest parameter
/// error over the last `tail` samples, for a parameter step from `before` to `after` halfway.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn track_step(mu: f64, form: CovarianceForm, steps: usize, tail: usize) -> f64 {
    let (before, after) = ([0.9, 0.5], [0.7, 0.8]);
    let mut rng = Lcg(3);
    let mut storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut rls = storage.build(100.0);
    rls.reset_with_form(form, 100.0);
    rls.set_forgetting_factor(mu);
    rls.set_measurement_variance(1e-4);

    let mut worst: f64 = 0.0;
    for k in 0..steps {
        let theta = if k < steps / 2 { before } else { after };
        let phi = [rng.next(), rng.next()];
        let y = theta[0] * phi[0] + theta[1] * phi[1] + 0.01 * rng.next();
        assert!(rls.update(&phi, y).is_applied());

        if k >= steps - tail {
            for (i, &t) in theta.iter().enumerate() {
                worst = worst.max((rls.parameters().get(i, 0) - t).abs());
            }
        }
    }

    worst
}

/// With `μ < 1`, RLS tracks a step change in the parameters; without forgetting it does not.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn tracks_parameter_step_with_forgetting() {
    for form in FORMS {
        let tracking = track_step(0.95, form, 400, 100);
        let frozen = track_step(1.0, form, 400, 100);
        assert!(tracking < 0.02, "{form:?}: error {tracking} with μ = 0.95");
        assert!(frozen > 0.05, "{form:?}: error {frozen} with μ = 1");
    }
}

/// Without excitation and without the guard, `P` grows as `μ⁻ᵏ`; with the guard (the default)
/// its trace stays at most `tr(P₀)`, and the estimate does not move.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn windup_guard_bounds_the_trace_without_excitation() {
    let mu = 0.9;
    for form in FORMS {
        let mut storage = EstimatorStorage::<2, f64>::new(0.0);
        let mut rls = storage.build(1.0);
        rls.reset_with_form(form, 1.0);
        rls.set_forgetting_factor(mu);
        assert_eq!(rls.trace_bound(), Some(2.0));

        // Excite once, then nothing but zero regressors.
        assert!(rls.update(&[1.0, 1.0], 1.0).is_applied());
        let theta = [rls.parameters().get(0, 0), rls.parameters().get(1, 0)];
        for _ in 0..500 {
            assert!(rls.update(&[0.0, 0.0], 0.0).is_applied());
            assert!(rls.covariance_trace() <= 2.0 * (1.0 + 1e-12), "{form:?}");
        }

        assert!(
            rls.covariance_trace() > 1.99,
            "{form:?}: forgetting reaches the bound"
        );
        assert_eq!(rls.parameters().get(0, 0), theta[0]);
        assert_eq!(rls.parameters().get(1, 0), theta[1]);

        rls.set_trace_bound(None);
        let start = rls.covariance_trace();
        for _ in 0..100 {
            assert!(rls.update(&[0.0, 0.0], 0.0).is_applied());
        }

        let growth = rls.covariance_trace() / start;
        let expected = mu.powi(-100);
        assert!(
            (growth / expected - 1.0).abs() < 1e-9,
            "{form:?}: windup {growth}"
        );
    }
}

/// With excitation in one direction only, the guard bounds the unexcited variance while the
/// excited parameter is still identified, if slowly once forgetting has stalled.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn windup_guard_with_partial_excitation() {
    for form in FORMS {
        let mut storage = EstimatorStorage::<2, f64>::new(0.0);
        let mut rls = storage.build(1.0);
        rls.reset_with_form(form, 1.0);
        rls.set_forgetting_factor(0.95);
        for k in 0..1000 {
            let u = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert!(rls.update(&[u, 0.0], 3.0 * u).is_applied());
            assert!(rls.covariance_trace() <= 2.0 * (1.0 + 1e-12));
        }

        // Once the unexcited variance reaches the bound, forgetting stops for both
        // directions, so the prior's pull on θ₀ fades only as 1/k: the isotropic trade-off.
        let error = (rls.parameters().get(0, 0) - 3.0).abs();
        assert!(error < 2e-3, "{form:?}: error {error}");
        assert_eq!(rls.parameters().get(1, 0), 0.0);
    }
}

/// A gated sample leaves `θ` and `P` bit-identical, forgetting included, and reports its NIS.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn gated_update_leaves_estimate_untouched() {
    for form in FORMS {
        let mut storage = EstimatorStorage::<2, f64>::new(0.0);
        let mut rls = storage.build(10.0);
        rls.reset_with_form(form, 10.0);
        rls.set_forgetting_factor(0.9);
        rls.set_measurement_variance(0.01);
        for k in 0..20 {
            let u = f64::from(k % 5);
            assert!(rls.update(&[u, 1.0], 2.0 * u - 1.0).is_applied());
        }

        rls.set_gate(Some(3.84));
        assert_eq!(rls.gate(), Some(3.84));
        let theta = [rls.parameters().get(0, 0), rls.parameters().get(1, 0)];
        let buffer: [f64; 4] = core::array::from_fn(|i| rls.covariance_buffer().get(i / 2, i % 2));

        assert_eq!(rls.update(&[1.0, 1.0], 100.0), CorrectionStatus::Gated);
        assert!(rls.nis() > 3.84);
        assert_eq!(rls.parameters().get(0, 0), theta[0]);
        assert_eq!(rls.parameters().get(1, 0), theta[1]);
        for (i, &value) in buffer.iter().enumerate() {
            assert_eq!(rls.covariance_buffer().get(i / 2, i % 2), value);
        }

        assert_eq!(rls.update(&[1.0, 1.0], 1.0), CorrectionStatus::Applied);
        assert!(rls.nis() <= 3.84);
    }
}

/// A non-positive `r`, or an indefinite `P` that gives `α ≤ 0`, rejects the sample.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn rejects_non_positive_variances() {
    let mut storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut rls = storage.build(1.0);
    for r in [0.0, -1.0] {
        rls.set_measurement_variance(r);
        assert_eq!(rls.measurement_variance(), r);
        assert_eq!(
            rls.update(&[1.0, 0.0], 1.0),
            CorrectionStatus::NotPositiveDefinite
        );
    }

    rls.reset_with_form(CovarianceForm::Covariance, -10.0);
    rls.set_measurement_variance(1.0);
    assert_eq!(
        rls.update(&[1.0, 0.0], 1.0),
        CorrectionStatus::NotPositiveDefinite
    );
    assert_eq!(rls.parameters().get(0, 0), 0.0);
}

/// Both forms give the same estimate and covariance; reset restores the prior.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
#[allow(clippy::needless_range_loop)]
fn forms_agree_and_reset_restores_the_prior() {
    let samples = dataset::<30>([1.0, 2.0, 3.0], 0.1, 11);
    let mut factored_storage = EstimatorStorage::<3, f64>::new(0.0);
    let mut covariance_storage = EstimatorStorage::<3, f64>::new(0.0);
    let mut factored = factored_storage.build(5.0);
    let mut covariance = covariance_storage.build(5.0);
    covariance.reset_with_form(CovarianceForm::Covariance, 5.0);
    assert_eq!(factored.form(), CovarianceForm::Factored);
    assert_eq!(covariance.form(), CovarianceForm::Covariance);

    for rls in [&mut factored, &mut covariance] {
        rls.set_forgetting_factor(0.97);
        assert_eq!(rls.forgetting_factor(), 0.97);
        for (phi, y) in &samples {
            assert!(rls.update(phi, *y).is_applied());
        }
    }

    let (pf, pc) = (covariance_of(&factored), covariance_of(&covariance));
    for i in 0..3 {
        let (a, b) = (
            factored.parameters().get(i, 0),
            covariance.parameters().get(i, 0),
        );
        assert!((a - b).abs() < 1e-12);
        for j in 0..3 {
            assert!((pf[i][j] - pc[i][j]).abs() < 1e-12);
        }
    }

    let (phi, _) = samples[0];
    assert!((factored.predict(&phi) - covariance.predict(&phi)).abs() < 1e-12);
    assert!((factored.covariance_trace() - covariance.covariance_trace()).abs() < 1e-12);

    factored.parameters_mut().set(0, 0, 4.0);
    factored.reset(2.0);
    assert_eq!(factored.parameters().get(0, 0), 0.0);
    assert_eq!(factored.nis(), 0.0);
    assert_eq!(factored.covariance_trace(), 6.0);
}

/// In `f32`, the factored form keeps `D ≥ 0` through a long, poorly excited run with heavy
/// forgetting, and still identifies the parameters.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn factored_form_stays_definite_in_f32() {
    let mut storage = EstimatorStorage::<3, f32>::new(0.0);
    let mut rls = storage.build(1000.0);
    rls.set_forgetting_factor(0.9);
    rls.set_measurement_variance(1e-6);

    let mut rng = Lcg(5);
    #[allow(clippy::cast_possible_truncation)]
    for k in 0..5000 {
        // Nearly collinear regressors: the third column is almost the first.
        let x = rng.next() as f32;
        let phi = [x, 1.0, x + 1e-4 * rng.next() as f32];
        let y = 0.5 * phi[0] + 0.25 * phi[1] + 0.5 * phi[2];
        assert!(rls.update(&phi, y).is_applied(), "step {k}");
        for j in 0..3 {
            assert!(
                rls.covariance_buffer().get(j, j) >= 0.0,
                "D[{j}] < 0 at step {k}"
            );
        }
    }

    let fit = rls.predict(&[0.3, 1.0, 0.3]);
    assert!((fit - 0.55).abs() < 1e-3, "{fit}");
}

/// From a fixed `p0`, `r` weighs the data against the prior and so changes `θ`, not only `P`:
/// one sample `φ = 1`, `y = 1` from `p0 = 1` gives `θ = p0/(p0 + r)`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn measurement_variance_weighs_data_against_the_prior() {
    for form in FORMS {
        for (r, expected) in [(1.0, 0.5), (4.0, 0.2)] {
            let mut storage = EstimatorStorage::<1, f64>::new(0.0);
            let mut rls = storage.build(1.0);
            rls.reset_with_form(form, 1.0);
            rls.set_measurement_variance(r);
            assert!(rls.update(&[1.0], 1.0).is_applied());

            let theta = rls.parameters().get(0, 0);
            assert!(
                (theta - expected).abs() < 1e-15,
                "{form:?} r = {r}: θ = {theta}"
            );
        }
    }
}

/// In `I16F16`, the factored form identifies a gain and an offset with forgetting, and keeps
/// `D ≥ 0`.
#[cfg(feature = "fixed")]
#[test]
fn identifies_parameters_in_i16f16() {
    use fixed::types::I16F16;

    let mut storage = EstimatorStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut rls = storage.build(I16F16::from_num(10));
    rls.set_forgetting_factor(I16F16::from_num(0.98));
    // P scales with r: the steady-state variances r·(1 − μ)/E[φ²] ≈ 0.04 stay well above the
    // resolution 1.5·10⁻⁵. With r = 0.01 they round to zero and the estimate freezes.
    rls.set_measurement_variance(I16F16::ONE);

    for k in 0..400 {
        let u = I16F16::from_num(k % 9) / I16F16::from_num(4) - I16F16::ONE;
        let gain = if k < 200 { 1.5 } else { 0.5 };
        let y = I16F16::from_num(gain) * u + I16F16::from_num(0.25);
        assert!(rls.update(&[u, I16F16::ONE], y).is_applied());
        for j in 0..2 {
            assert!(rls.covariance_buffer().get(j, j) >= I16F16::ZERO);
        }
    }

    let theta = rls.parameters();
    assert!((theta.get(0, 0) - I16F16::from_num(0.5)).abs() < I16F16::from_num(0.02));
    assert!((theta.get(1, 0) - I16F16::from_num(0.25)).abs() < I16F16::from_num(0.02));
}
