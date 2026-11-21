//! The normalized innovation squared (NIS) of a correction, and gating corrections on it.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Gate for one measurement: the 99% quantile of the chi-square distribution with one
/// degree of freedom.
const GATE: f64 = 6.63;

/// With `x = 0`, `P = 1`, `H = 1` and `R = 1`, a measurement `z` has `S = 2` and
/// `NIS = z² / 2`.
fn expected_nis(z: f64) -> f64 {
    z * z / 2.0
}

/// The state, the variance, the NIS and the status after one correction.
type Outcome = (f64, f64, f64, CorrectionStatus);

/// Runs one correction of a scalar regular filter and returns its [`Outcome`].
fn regular_after_one_reading(z: f64, gate: Option<f64>) -> Outcome {
    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, z);
    measurement.set_gate(gate);

    let status = filter.correct(&mut measurement);

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
        measurement.nis(),
        status,
    )
}

/// Runs one correction of a scalar extended filter and returns its [`Outcome`].
fn extended_after_one_reading(z: f64, gate: Option<f64>) -> Outcome {
    let builder = ExtendedBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement
        .observation_jacobian_matrix_mut()
        .make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, z);
    measurement.set_gate(gate);

    let status = filter.correct_nonlinear(&mut measurement, |state, observation| {
        observation.set_at(0, 0, state.get_at(0, 0));
    });

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
        measurement.nis(),
        status,
    )
}

/// Runs one correction of a scalar unscented filter and returns its [`Outcome`].
fn unscented_after_one_reading(z: f64, gate: Option<f64>) -> Outcome {
    let builder = UnscentedBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, z);
    measurement.set_gate(gate);

    let status = filter.correct_sigma_point(&mut measurement, |sigma, observed| {
        for j in 0..3 {
            observed.set_at(0, j, sigma.get_at(0, j));
        }
    });

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
        measurement.nis(),
        status,
    )
}

/// Checks that an outlier is rejected by the gate and an inlier is applied, for one filter.
fn check_gating(correct: fn(f64, Option<f64>) -> Outcome) {
    // An outlier is rejected: x and P stay as they were, the NIS is still reported.
    let (x, p, nis, status) = correct(100.0, Some(GATE));
    assert_eq!(status, CorrectionStatus::Gated);
    assert_eq!((x, p), (0.0, 1.0));
    assert!((nis - expected_nis(100.0)).abs() < 1e-9, "NIS = {nis}");

    // Without a gate, the same measurement is applied.
    let (x, p, _, status) = correct(100.0, None);
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((x - 50.0).abs() < 1e-9, "x = {x}");
    assert!((p - 0.5).abs() < 1e-9, "P = {p}");

    // A measurement within the gate is applied.
    let (x, p, nis, status) = correct(1.0, Some(GATE));
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((x - 0.5).abs() < 1e-9, "x = {x}");
    assert!((p - 0.5).abs() < 1e-9, "P = {p}");
    assert!((nis - expected_nis(1.0)).abs() < 1e-9, "NIS = {nis}");
}

#[test]
fn regular_gate_rejects_outlier() {
    check_gating(regular_after_one_reading);
}

#[test]
fn extended_gate_rejects_outlier() {
    check_gating(extended_after_one_reading);
}

#[test]
fn unscented_gate_rejects_outlier() {
    check_gating(unscented_after_one_reading);
}

/// Checks the initial NIS and gate of an observation, and that `set_gate` round-trips.
macro_rules! check_defaults {
    ($measurement:expr) => {{
        let mut measurement = $measurement;
        assert_eq!(measurement.nis(), 0.0);
        assert_eq!(measurement.gate(), None);
        measurement.set_gate(Some(GATE));
        assert_eq!(measurement.gate(), Some(GATE));
        measurement.set_gate(None);
        assert_eq!(measurement.gate(), None);
    }};
}

#[test]
fn nis_is_zero_before_the_first_correction_and_gate_is_off_by_default() {
    check_defaults!(
        RegularBuilder::<1, f64>::default()
            .observations()
            .build::<1>()
    );
    check_defaults!(
        ExtendedBuilder::<1, f64>::default()
            .observations()
            .build::<1>()
    );
    check_defaults!(
        UnscentedBuilder::<1, f64>::default()
            .observations()
            .build::<1, 3>()
    );
}

/// A correctly tuned linear filter is consistent: its NIS averages to the number of
/// measurements. Here a random walk with `Q = 0.1` is measured twice per step, with
/// `R = diag(1, 4)`.
#[test]
fn nis_averages_to_number_of_observations_for_a_tuned_filter() {
    const STEPS: usize = 20_000;
    const BURN_IN: usize = 100;
    let q = 0.1;
    let r = [1.0, 4.0];

    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    filter.state_transition_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(q);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().apply(|h| {
        h.set_at(0, 0, 1.0);
        h.set_at(1, 0, 1.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|m| {
        m.set_at(0, 0, r[0]);
        m.set_at(1, 1, r[1]);
    });

    let mut rng = StdRng::seed_from_u64(59);
    let process = Normal::new(0.0, q.sqrt()).unwrap();
    let sensors = [
        Normal::new(0.0, r[0].sqrt()).unwrap(),
        Normal::new(0.0, r[1].sqrt()).unwrap(),
    ];

    let mut truth = 0.0;
    let mut sum = 0.0;
    for step in 0..STEPS {
        truth += process.sample(&mut rng);
        filter.predict();
        measurement.measurement_vector_mut().apply(|z| {
            z.set_at(0, 0, truth + sensors[0].sample(&mut rng));
            z.set_at(1, 0, truth + sensors[1].sample(&mut rng));
        });
        filter.correct(&mut measurement);
        if step >= BURN_IN {
            sum += measurement.nis();
        }
    }

    let mean = sum / (STEPS - BURN_IN) as f64;
    assert!((mean - 2.0).abs() < 0.1, "mean NIS = {mean}");
}

/// Calls `correct` through the [`KalmanFilterUpdate`] trait, as generic code would.
fn correct_via_trait<F, M>(filter: &mut F, measurement: &mut M) -> CorrectionStatus
where
    F: KalmanFilterUpdate<1, f64>,
    M: KalmanFilterObservationCorrectFilter<1, f64>,
{
    filter.correct(measurement)
}

/// The status is also returned through the filter traits and the observations' own
/// `correct` / `correct_nonlinear`, which take the state and covariance directly.
#[test]
fn status_is_returned_through_traits_and_observations() {
    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);
    assert_eq!(
        correct_via_trait(&mut filter, &mut measurement),
        CorrectionStatus::Applied
    );

    let mut x = BufferBuilder::state_vector_x::<1>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<1>().new();
    p.make_scalar(1.0);
    measurement.set_gate(Some(GATE));
    measurement.measurement_vector_mut().set_at(0, 0, 100.0);
    assert_eq!(measurement.correct(&mut x, &mut p), CorrectionStatus::Gated);
    assert_eq!(x.get_at(0, 0), 0.0);

    let mut extended = ExtendedBuilder::<1, f64>::default()
        .observations()
        .build::<1>();
    extended.observation_jacobian_matrix_mut().make_identity();
    extended.measurement_noise_covariance_mut().make_scalar(1.0);
    extended.measurement_vector_mut().set_at(0, 0, 1.0);
    let status = extended.correct_nonlinear(&mut x, &mut p, |state, observation| {
        observation.set_at(0, 0, state.get_at(0, 0));
    });
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((x.get_at(0, 0) - 0.5).abs() < 1e-9);
    assert!(status.is_applied());
}
