//! Online adaptation of the measurement noise covariance `R` from the residuals.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Adaptation with forgetting factor `α` and a negligible floor.
fn adaptation(forgetting: f64) -> Option<NoiseAdaptation<f64>> {
    Some(NoiseAdaptation {
        forgetting,
        floor: 1e-9,
    })
}

/// The status of one correction and `R` afterwards.
type Outcome = (CorrectionStatus, f64);

/// Runs one correction of a scalar regular filter with `x = 0`, `H = 1` and the given `P`
/// and `R`, and returns its [`Outcome`].
fn regular_after_one_reading(
    z: f64,
    p: f64,
    r: f64,
    gate: Option<f64>,
    adaptation: Option<NoiseAdaptation<f64>>,
) -> Outcome {
    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(p);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(r);
    measurement.measurement_vector_mut().set_at(0, 0, z);
    measurement.set_gate(gate);
    measurement.set_noise_adaptation(adaptation);
    assert_eq!(measurement.noise_adaptation(), adaptation);

    let status = filter.correct(&mut measurement);

    (
        status,
        measurement.measurement_noise_covariance().get_at(0, 0),
    )
}

/// Runs one correction of a scalar extended filter with `x = 0`, `H = 1` and the given `P`
/// and `R`, and returns its [`Outcome`].
fn extended_after_one_reading(
    z: f64,
    p: f64,
    r: f64,
    gate: Option<f64>,
    adaptation: Option<NoiseAdaptation<f64>>,
) -> Outcome {
    let builder = ExtendedBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(p);
    measurement
        .observation_jacobian_matrix_mut()
        .make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(r);
    measurement.measurement_vector_mut().set_at(0, 0, z);
    measurement.set_gate(gate);
    measurement.set_noise_adaptation(adaptation);
    assert_eq!(measurement.noise_adaptation(), adaptation);

    let status = filter.correct_nonlinear(&mut measurement, |state, observation| {
        observation.set_at(0, 0, state.get_at(0, 0));
    });

    (
        status,
        measurement.measurement_noise_covariance().get_at(0, 0),
    )
}

/// A one-reading correction of either filter.
type Correct = fn(f64, f64, f64, Option<f64>, Option<NoiseAdaptation<f64>>) -> Outcome;

/// Checks the adaptation of `R` on single readings with `P = 1` and `R = 1`, so `S = 2`.
fn check_single_readings(correct: Correct) {
    // Adaptation is off by default and leaves R untouched.
    let (status, r) = correct(10.0, 1.0, 1.0, None, None);
    assert_eq!((status, r), (CorrectionStatus::Applied, 1.0));

    // A gated correction leaves R untouched.
    let (status, r) = correct(10.0, 1.0, 1.0, Some(6.63), adaptation(0.5));
    assert_eq!((status, r), (CorrectionStatus::Gated, 1.0));

    // A skipped correction (S = 0) leaves R untouched, the floor is not applied either.
    let (status, r) = correct(10.0, 0.0, 0.0, None, adaptation(0.5));
    assert_eq!((status, r), (CorrectionStatus::NotPositiveDefinite, 0.0));

    // The true R is a fixed point: with y² = S, the NIS is 1 and R stays.
    let (status, r) = correct(2f64.sqrt(), 1.0, 1.0, None, adaptation(0.5));
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((r - 1.0).abs() < 1e-12, "R = {r}");

    // A large residual raises R: R × (1 + (1−α) × (R/S) × (NIS − 1)) = 1 + 0.5 × 0.5 × 49.
    let (_, r) = correct(10.0, 1.0, 1.0, None, adaptation(0.5));
    assert!((r - 13.25).abs() < 1e-12, "R = {r}");

    // A zero residual lowers R to 1 − 0.5 × 0.5 = 0.75, unless the floor is higher.
    let (_, r) = correct(0.0, 1.0, 1.0, None, adaptation(0.5));
    assert!((r - 0.75).abs() < 1e-12, "R = {r}");
    let floored = Some(NoiseAdaptation {
        forgetting: 0.5,
        floor: 0.8,
    });
    let (_, r) = correct(0.0, 1.0, 1.0, None, floored);
    assert_eq!(r, 0.8);

    // α > 1 reverses the correction, 1.5 − 0.5 × 25.5 = −11.25; only the floor remains.
    let reversed = Some(NoiseAdaptation {
        forgetting: 1.5,
        floor: 0.8,
    });
    let (_, r) = correct(10.0, 1.0, 1.0, None, reversed);
    assert_eq!(r, 0.8);
}

#[test]
fn regular_adapts_r_on_applied_corrections_only() {
    check_single_readings(regular_after_one_reading);
}

#[test]
fn extended_adapts_r_on_applied_corrections_only() {
    check_single_readings(extended_after_one_reading);
}

/// Root-mean-square error over the second half of the run, and `R` averaged over its last
/// quarter.
type TrackingResult = (f64, f64);

/// Tracks a random walk with `Q = 0.01` whose measurement noise variance steps from `1` to
/// `9` halfway through, starting with `R = 1`, and returns the [`TrackingResult`].
fn track_noise_step(adaptation: Option<NoiseAdaptation<f64>>) -> TrackingResult {
    const STEPS: usize = 8_000;
    let q = 0.01;

    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_transition_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(q);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.set_noise_adaptation(adaptation);

    let mut rng = StdRng::seed_from_u64(61);
    let process = Normal::new(0.0, q.sqrt()).unwrap();
    let quiet = Normal::new(0.0, 1.0).unwrap();
    let noisy = Normal::new(0.0, 3.0).unwrap();

    let mut truth = 0.0;
    let mut squared_error = 0.0;
    let mut r_sum = 0.0;
    for step in 0..STEPS {
        truth += process.sample(&mut rng);
        let sensor = if step < STEPS / 2 { quiet } else { noisy };
        filter.predict();
        measurement
            .measurement_vector_mut()
            .set_at(0, 0, truth + sensor.sample(&mut rng));
        assert!(filter.correct(&mut measurement).is_applied());

        if step >= STEPS / 2 {
            let error = filter.state_vector().get_at(0, 0) - truth;
            squared_error += error * error;
        }

        if step >= STEPS * 3 / 4 {
            r_sum += measurement.measurement_noise_covariance().get_at(0, 0);
        }
    }

    (
        (squared_error / (STEPS / 2) as f64).sqrt(),
        r_sum / (STEPS / 4) as f64,
    )
}

/// When the sensor noise variance steps from `1` to `9`, the adapted filter's `R` follows it
/// and its state error is lower than that of the filter that keeps `R = 1`.
#[test]
fn adapted_r_tracks_a_noise_step_with_lower_error() {
    let (fixed_rmse, fixed_r) = track_noise_step(None);
    let (adapted_rmse, adapted_r) = track_noise_step(adaptation(0.99));

    assert_eq!(fixed_r, 1.0);
    assert!((adapted_r - 9.0).abs() < 1.5, "adapted R = {adapted_r}");
    assert!(
        adapted_rmse < 0.9 * fixed_rmse,
        "adapted RMSE {adapted_rmse} vs fixed RMSE {fixed_rmse}"
    );
}

/// With two correlated measurements, the adapted `R` stays symmetric and positive definite.
#[test]
fn adapted_r_stays_symmetric_positive_definite() {
    const STEPS: usize = 2_000;

    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    filter.state_transition_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().apply(|h| {
        h.set_at(0, 0, 1.0);
        h.set_at(1, 0, 1.0);
        h.set_at(1, 1, 1.0);
    });
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.set_noise_adaptation(adaptation(0.9));

    let mut rng = StdRng::seed_from_u64(62);
    let noise = Normal::new(0.0, 2.0).unwrap();

    for _ in 0..STEPS {
        filter.predict();
        let shared = noise.sample(&mut rng);
        measurement.measurement_vector_mut().apply(|z| {
            z.set_at(0, 0, shared + noise.sample(&mut rng));
            z.set_at(1, 0, shared);
        });
        filter.correct(&mut measurement);

        let r = measurement.measurement_noise_covariance();
        let (a, b, c, d) = (
            r.get_at(0, 0),
            r.get_at(0, 1),
            r.get_at(1, 0),
            r.get_at(1, 1),
        );
        assert_eq!(b, c, "R is not symmetric");
        assert!(a > 0.0 && a * d - b * c > 0.0, "R is not positive definite");
    }
}
