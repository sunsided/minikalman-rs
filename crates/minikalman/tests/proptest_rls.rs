//! Property tests for the recursive least-squares estimator: directional forgetting in both
//! covariance forms, the information it keeps along unexcited directions, and updates that
//! must leave the estimate untouched.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::matrix::MatrixData;
use minikalman::prelude::*;
use minikalman::rls::{
    BorrowedEstimator, CovarianceForm, Directional, EstimatorStorage, Exponential, Forgetting,
};
use proptest::prelude::*;

/// Number of parameters.
const N: usize = 3;

/// One sample `(φ, y)`.
type Sample = ([f64; N], f64);

/// Settings of one run: the forgetting factor `μ`, the measurement variance `r` and `p0`.
#[derive(Debug, Clone, Copy)]
struct Settings {
    /// The forgetting factor.
    mu: f64,
    /// The measurement noise variance.
    r: f64,
    /// The initial variance of every parameter.
    p0: f64,
}

/// Settings across the useful range: memories of 10 to 1000 samples, precise to noisy data,
/// tight to vague priors.
fn settings() -> impl Strategy<Value = Settings> {
    (0.9_f64..0.999, 0.01_f64..10.0, 0.1_f64..1000.0).prop_map(|(mu, r, p0)| Settings { mu, r, p0 })
}

/// Up to 150 samples with `φᵢ, y ∈ [−5, 5]` whose last regressor is zero from a random index
/// on, i.e. goes idle.
fn samples_with_idle_tail() -> impl Strategy<Value = Vec<Sample>> {
    (
        proptest::collection::vec(
            (proptest::array::uniform3(-5.0_f64..5.0), -5.0_f64..5.0),
            20..150,
        ),
        0.0_f64..1.0,
    )
        .prop_map(|(mut samples, idle_from)| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_precision_loss
            )]
            let start = (idle_from * samples.len() as f64) as usize;
            for (phi, _) in samples.iter_mut().skip(start) {
                phi[N - 1] = 0.0;
            }
            samples
        })
}

/// Builds an estimator with the mode `F`, the form `form` and `settings`.
fn build<F: Forgetting>(
    storage: &mut EstimatorStorage<N, f64>,
    form: CovarianceForm,
    settings: Settings,
) -> BorrowedEstimator<'_, N, f64, F> {
    let mut rls = storage.build(settings.p0).with_forgetting::<F>();
    rls.reset_with_form(form, settings.p0);
    rls.set_forgetting_factor(settings.mu);
    rls.set_measurement_variance(settings.r);
    rls
}

/// Returns `P` of an estimator as a nested array.
fn covariance_of<F: Forgetting>(rls: &BorrowedEstimator<'_, N, f64, F>) -> [[f64; N]; N] {
    let mut p = [[0.0; N]; N];
    rls.covariance(&mut MatrixData::new_mut::<N, N, f64>(p.as_flattened_mut()));
    p
}

/// Returns `θ` of an estimator as an array.
fn theta_of<F: Forgetting>(rls: &BorrowedEstimator<'_, N, f64, F>) -> [f64; N] {
    core::array::from_fn(|i| rls.parameters().get(i, 0))
}

/// Returns the information `(P⁻¹)₂₂` along the last parameter, from the cofactor of a 3 × 3
/// `P`.
fn information_along_last(p: &[[f64; N]; N]) -> f64 {
    let cofactor = p[0][0] * p[1][1] - p[0][1] * p[1][0];
    let det = p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
        - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
        + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
    cofactor / det
}

/// Returns the largest absolute entry of `p`.
fn largest(p: &[[f64; N]; N]) -> f64 {
    p.iter().flatten().fold(0.0, |m, x| m.max(x.abs()))
}

proptest! {
    /// With directional forgetting, both covariance forms give the same `θ` and `P`, every
    /// value stays finite, and the factored form keeps `D ≥ 0`, also once a regressor idles.
    #[test]
    fn directional_forms_agree_and_stay_definite(
        settings in settings(),
        samples in samples_with_idle_tail(),
    ) {
        let mut factored_storage = EstimatorStorage::<N, f64>::new(0.0);
        let mut covariance_storage = EstimatorStorage::<N, f64>::new(0.0);
        let mut factored = build::<Directional>(&mut factored_storage, CovarianceForm::Factored, settings);
        let mut covariance = build::<Directional>(&mut covariance_storage, CovarianceForm::Covariance, settings);

        for (k, (phi, y)) in samples.iter().enumerate() {
            prop_assert!(factored.update(phi, *y).is_applied(), "factored step {}", k);
            prop_assert!(covariance.update(phi, *y).is_applied(), "covariance step {}", k);
            for j in 0..N {
                prop_assert!(factored.covariance_buffer().get(j, j) >= 0.0, "D[{}] < 0 at step {}", j, k);
            }
        }

        let (pf, pc) = (covariance_of(&factored), covariance_of(&covariance));
        let scale = largest(&pf).max(largest(&pc));
        for i in 0..N {
            let (a, b) = (factored.parameters().get(i, 0), covariance.parameters().get(i, 0));
            prop_assert!(a.is_finite() && b.is_finite());
            prop_assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "θ[{}]: {} vs {}", i, a, b);
            for j in 0..N {
                prop_assert!(pf[i][j].is_finite());
                prop_assert!((pf[i][j] - pc[i][j]).abs() <= 1e-6 * scale, "P[{}][{}]: {} vs {}", i, j, pf[i][j], pc[i][j]);
            }
        }
    }

    /// Directional forgetting keeps the information `(P⁻¹)₂₂` along a parameter whose
    /// regressor is zero exactly, step by step; exponential forgetting lets it decay by `μ`.
    #[test]
    fn idle_information_is_kept(
        settings in settings(),
        excited in proptest::collection::vec((proptest::array::uniform3(-5.0_f64..5.0), -5.0_f64..5.0), 10..40),
        idle in proptest::collection::vec((proptest::array::uniform2(-5.0_f64..5.0), -5.0_f64..5.0), 1..60),
    ) {
        for form in [CovarianceForm::Factored, CovarianceForm::Covariance] {
            let mut storage = EstimatorStorage::<N, f64>::new(0.0);
            let mut directional = build::<Directional>(&mut storage, form, settings);
            for (phi, y) in &excited {
                prop_assert!(directional.update(phi, *y).is_applied());
            }

            let before = information_along_last(&covariance_of(&directional));
            prop_assume!(before.is_finite() && before > 0.0);
            for ([u, w], y) in &idle {
                prop_assert!(directional.update(&[*u, *w, 0.0], *y).is_applied());
                let now = information_along_last(&covariance_of(&directional));
                prop_assert!((now - before).abs() <= 1e-6 * before, "{:?}: {} became {}", form, before, now);
            }

            // Exponential forgetting without the guard loses it as μᵏ.
            let mut storage = EstimatorStorage::<N, f64>::new(0.0);
            let mut exponential = build::<Exponential>(&mut storage, form, settings);
            exponential.set_trace_bound(None);
            for (phi, y) in &excited {
                prop_assert!(exponential.update(phi, *y).is_applied());
            }

            let before = information_along_last(&covariance_of(&exponential));
            let ([u, w], y) = idle[0];
            prop_assert!(exponential.update(&[u, w, 0.0], y).is_applied());
            let now = information_along_last(&covariance_of(&exponential));
            prop_assert!((now - settings.mu * before).abs() <= 1e-6 * before, "{:?}: {} vs μ·{}", form, now, before);
        }
    }

    /// A gated update leaves `θ` and `P` bit-identical in both modes and forms, and so does a
    /// zero regressor with directional forgetting.
    #[test]
    fn rejected_and_empty_updates_change_nothing(
        settings in settings(),
        samples in proptest::collection::vec((proptest::array::uniform3(-5.0_f64..5.0), -5.0_f64..5.0), 1..40),
        probe in proptest::array::uniform3(-5.0_f64..5.0),
    ) {
        for form in [CovarianceForm::Factored, CovarianceForm::Covariance] {
            let mut storage = EstimatorStorage::<N, f64>::new(0.0);
            let mut exponential = build::<Exponential>(&mut storage, form, settings);
            for (phi, y) in &samples {
                prop_assert!(exponential.update(phi, *y).is_applied());
            }
            assert_rejected_update_is_inert(&mut exponential, &probe)?;

            let mut directional = exponential.with_forgetting::<Directional>();
            assert_rejected_update_is_inert(&mut directional, &probe)?;

            let (p, theta) = (covariance_of(&directional), theta_of(&directional));
            for _ in 0..10 {
                prop_assert!(directional.update(&[0.0; N], 1.0).is_applied());
            }
            prop_assert_eq!(covariance_of(&directional), p, "{:?}", form);
            prop_assert_eq!(theta_of(&directional), theta);
        }
    }
}

/// Asserts that an update far outside a zero gate is gated and changes neither `θ` nor `P`.
fn assert_rejected_update_is_inert<F: Forgetting>(
    rls: &mut BorrowedEstimator<'_, N, f64, F>,
    phi: &[f64; N],
) -> Result<(), TestCaseError> {
    let (p, theta) = (covariance_of(rls), theta_of(rls));
    rls.set_gate(Some(0.0));
    let y = rls.predict(phi) + 100.0;
    prop_assert_eq!(rls.update(phi, y), CorrectionStatus::Gated);
    rls.set_gate(None);
    prop_assert_eq!(covariance_of(rls), p);
    prop_assert_eq!(theta_of(rls), theta);
    Ok(())
}

/// Fixed-point properties of the factored form, the one to use in fixed point: any short run
/// within the ranges of the Kani proofs neither overflows (which panics in debug builds) nor
/// makes `D` negative. The covariance form is left out on purpose: its textbook update rounds
/// `P` indefinite in `I16F16` within a few steps in either mode (see `CovarianceForm`).
#[cfg(feature = "fixed")]
mod fixed_point {
    use super::*;
    use fixed::types::I16F16;

    /// An `I16F16` in `[lo, hi]`, biased towards tiny magnitudes and the ends of the range,
    /// where fixed point loses resolution or overflows.
    fn in_range(lo: f64, hi: f64) -> impl Strategy<Value = I16F16> {
        let (lo, hi) = (
            I16F16::from_num(lo).to_bits(),
            I16F16::from_num(hi).to_bits(),
        );
        prop_oneof![
            lo..=hi,
            (-64_i32..=64).prop_map(move |b| b.clamp(lo, hi)),
            Just(lo),
            Just(hi),
        ]
        .prop_map(I16F16::from_bits)
    }

    /// Settings in the ranges of the Kani proofs.
    fn settings() -> impl Strategy<Value = (I16F16, I16F16, I16F16)> {
        (
            in_range(1.0 / 16.0, 100.0),
            in_range(0.9, 1.0),
            in_range(1.0 / 16.0, 16.0),
        )
    }

    /// Two or three samples with `φ ∈ [−4, 4]²` and `y ∈ [−8, 8]`.
    fn samples() -> impl Strategy<Value = Vec<([I16F16; 2], I16F16)>> {
        proptest::collection::vec(
            (
                [in_range(-4.0, 4.0), in_range(-4.0, 4.0)],
                in_range(-8.0, 8.0),
            ),
            2..=3,
        )
    }

    /// Runs `samples` with the mode `F` in the factored form, asserting `D ≥ 0` after every
    /// step.
    fn run<F: Forgetting>(
        (p0, mu, r): (I16F16, I16F16, I16F16),
        samples: &[([I16F16; 2], I16F16)],
    ) -> Result<(), TestCaseError> {
        let mut storage = EstimatorStorage::<2, I16F16>::new(I16F16::ZERO);
        let mut rls = storage.build(p0).with_forgetting::<F>();
        rls.set_forgetting_factor(mu);
        rls.set_measurement_variance(r);
        for (phi, y) in samples {
            let _ = rls.update(phi, *y);
            for j in 0..2 {
                prop_assert!(
                    rls.covariance_buffer().get(j, j) >= I16F16::ZERO,
                    "{}: D[{}] = {}",
                    core::any::type_name::<F>(),
                    j,
                    rls.covariance_buffer().get(j, j)
                );
            }
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2048))]

        /// Both modes survive short `I16F16` runs in the factored form.
        #[test]
        fn short_runs_do_not_overflow(settings in settings(), samples in samples()) {
            run::<Exponential>(settings, &samples)?;
            run::<Directional>(settings, &samples)?;
        }
    }
}
