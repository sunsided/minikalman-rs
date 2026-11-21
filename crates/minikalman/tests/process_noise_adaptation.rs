//! Online adaptation of the process noise covariance `Q` from the state corrections.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Adaptation with forgetting factor `α` and a zero floor.
fn adaptation(forgetting: f64) -> Option<NoiseAdaptation<f64>> {
    Some(NoiseAdaptation {
        forgetting,
        floor: 0.0,
    })
}

/// The status of one correction, and `x` and `Q` afterwards.
type Outcome = (CorrectionStatus, f64, f64);

/// The scalar setup of a one-reading correction.
#[derive(Clone, Copy)]
struct Reading {
    /// The measurement.
    z: f64,
    /// The prior variance `P`.
    p: f64,
    /// The measurement noise variance `R`.
    r: f64,
    /// The process noise variance `Q` before the correction.
    q: f64,
    /// The innovation gate.
    gate: Option<f64>,
}

/// Runs one correction of a scalar regular filter with `x = 0` and `H = 1`.
fn regular_after_one_reading(
    reading: Reading,
    adaptation: Option<NoiseAdaptation<f64>>,
) -> Outcome {
    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(reading.p);
    filter.direct_process_noise_mut().make_scalar(reading.q);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(reading.r);
    measurement.measurement_vector_mut().set_at(0, 0, reading.z);
    measurement.set_gate(reading.gate);
    filter.set_process_noise_adaptation(adaptation);
    assert_eq!(filter.process_noise_adaptation(), adaptation);

    let status = filter.correct(&mut measurement);

    (
        status,
        filter.state_vector().get_at(0, 0),
        filter.direct_process_noise().get_at(0, 0),
    )
}

/// Runs one correction of a scalar extended filter with `x = 0` and `h(x) = x`.
fn extended_after_one_reading(
    reading: Reading,
    adaptation: Option<NoiseAdaptation<f64>>,
) -> Outcome {
    let builder = ExtendedBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(reading.p);
    filter.direct_process_noise_mut().make_scalar(reading.q);
    measurement
        .observation_jacobian_matrix_mut()
        .make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(reading.r);
    measurement.measurement_vector_mut().set_at(0, 0, reading.z);
    measurement.set_gate(reading.gate);
    filter.set_process_noise_adaptation(adaptation);
    assert_eq!(filter.process_noise_adaptation(), adaptation);

    let status = filter.correct_nonlinear(&mut measurement, |state, observation| {
        observation.set_at(0, 0, state.get_at(0, 0));
    });

    (
        status,
        filter.state_vector().get_at(0, 0),
        filter.direct_process_noise().get_at(0, 0),
    )
}

/// Observes the single state of each of the three sigma points directly, `h(x) = x`.
fn observe_identity(sigma: &impl Matrix<1, 3, f64>, observed: &mut impl MatrixMut<1, 3, f64>) {
    for j in 0..3 {
        observed.set_at(0, j, sigma.get_at(0, j));
    }
}

/// Runs one correction of a scalar unscented filter with `x = 0` and `h(x) = x`, without a
/// prediction, plain or recalibrated.
fn unscented_after_one_reading(
    reading: Reading,
    adaptation: Option<NoiseAdaptation<f64>>,
    recalibrated: bool,
) -> Outcome {
    let builder = UnscentedBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.estimate_covariance_mut().make_scalar(reading.p);
    filter.direct_process_noise_mut().make_scalar(reading.q);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(reading.r);
    measurement.measurement_vector_mut().set_at(0, 0, reading.z);
    measurement.set_gate(reading.gate);
    filter.set_process_noise_adaptation(adaptation);
    assert_eq!(filter.process_noise_adaptation(), adaptation);

    let observe = |sigma: &_, observed: &mut _| observe_identity(sigma, observed);
    let status = if recalibrated {
        filter.correct_sigma_point_recalibrated(&mut measurement, observe)
    } else {
        filter.correct_sigma_point(&mut measurement, observe)
    };

    (
        status,
        filter.state_vector().get_at(0, 0),
        filter.direct_process_noise().get_at(0, 0),
    )
}

/// A one-reading correction of any filter.
type Correct = fn(Reading, Option<NoiseAdaptation<f64>>) -> Outcome;

/// Checks the adaptation of `Q` on single readings with `P = 1` and `R = 1`, so `K = 1/2`.
fn check_single_readings(correct: Correct) {
    let reading = Reading {
        z: 4.0,
        p: 1.0,
        r: 1.0,
        q: 1.0,
        gate: None,
    };

    // Adaptation is off by default and leaves Q bit-identical.
    let (status, x, q) = correct(reading, None);
    assert_eq!((status, q), (CorrectionStatus::Applied, 1.0));
    assert!((x - 2.0).abs() < 1e-12, "x = {x}");

    // Δx = K×y = 2, so Q = 0.5 × 1 + 0.5 × 2² = 2.5.
    let (status, x_adapted, q) = correct(reading, adaptation(0.5));
    assert_eq!(status, CorrectionStatus::Applied);
    assert_eq!(x_adapted, x, "the adaptation changed the correction");
    assert!((q - 2.5).abs() < 1e-12, "Q = {q}");

    // A zero correction lowers Q to α × Q, unless the floor is higher.
    let zero = Reading { z: 0.0, ..reading };
    let (_, _, q) = correct(zero, adaptation(0.5));
    assert!((q - 0.5).abs() < 1e-12, "Q = {q}");
    let floored = Some(NoiseAdaptation {
        forgetting: 0.5,
        floor: 0.8,
    });
    let (_, _, q) = correct(zero, floored);
    assert_eq!(q, 0.8);

    // A gated correction leaves Q untouched.
    let gated = Reading {
        gate: Some(6.63),
        z: 10.0,
        ..reading
    };
    let (status, _, q) = correct(gated, adaptation(0.5));
    assert_eq!((status, q), (CorrectionStatus::Gated, 1.0));

    // A skipped correction (S = 0) leaves Q untouched, the floor is not applied either.
    let singular = Reading {
        p: 0.0,
        r: 0.0,
        q: 0.0,
        ..reading
    };
    let (status, _, q) = correct(singular, floored);
    assert_eq!((status, q), (CorrectionStatus::NotPositiveDefinite, 0.0));
}

#[test]
fn regular_adapts_q_on_applied_corrections_only() {
    check_single_readings(regular_after_one_reading);
}

#[test]
fn extended_adapts_q_on_applied_corrections_only() {
    check_single_readings(extended_after_one_reading);
}

#[test]
fn unscented_adapts_q_on_applied_corrections_only() {
    check_single_readings(|reading, adaptation| {
        unscented_after_one_reading(reading, adaptation, false)
    });
}

#[test]
fn recalibrated_unscented_adapts_q_on_applied_corrections_only() {
    check_single_readings(|reading, adaptation| {
        unscented_after_one_reading(reading, adaptation, true)
    });
}

/// Tracks a random walk with `Q = 0.1`, measured with noise variance `1`, starting from the
/// given `Q`, and returns `Q` averaged over the second half of the run.
fn track_random_walk(initial_q: f64) -> f64 {
    const STEPS: usize = 20_000;
    let q: f64 = 0.1;

    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_transition_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(initial_q);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    filter.set_process_noise_adaptation(adaptation(0.995));

    let mut rng = StdRng::seed_from_u64(89);
    let process = Normal::new(0.0, q.sqrt()).unwrap();
    let sensor = Normal::new(0.0, 1.0).unwrap();

    let mut truth = 0.0;
    let mut q_sum = 0.0;
    for step in 0..STEPS {
        truth += process.sample(&mut rng);
        filter.predict();
        measurement
            .measurement_vector_mut()
            .set_at(0, 0, truth + sensor.sample(&mut rng));
        assert!(filter.correct(&mut measurement).is_applied());

        if step >= STEPS / 2 {
            q_sum += filter.direct_process_noise().get_at(0, 0);
        }
    }

    q_sum / (STEPS / 2) as f64
}

/// Started ten times too high or too low, the adapted `Q` of a random walk ends up near the
/// true `0.1`.
#[test]
fn adapted_q_converges_towards_the_true_q() {
    for initial_q in [1.0, 0.01] {
        let q = track_random_walk(initial_q);
        assert!(
            (0.07..0.14).contains(&q),
            "started at {initial_q}: adapted Q = {q}"
        );
    }
}

/// On a constant velocity model with a position sensor, the adapted `Q` stays symmetric
/// positive semi-definite at every step, and the filter stays consistent.
///
/// The entries of `Q` are not compared with the true `diag(0, 0.01)`: with a scalar sensor
/// and a steady-state gain, every `Δx` points along the same `K`, so the estimate has rank
/// one along `K` instead of the true shape.
#[test]
fn adapted_q_stays_symmetric_positive_semi_definite() {
    const STEPS: usize = 5_000;
    let dt = 0.1;
    let acceleration = 1.0;

    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 1, dt);
    });
    filter.direct_process_noise_mut().make_scalar(1.0);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().set_at(0, 0, 1.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(0.01);
    filter.set_process_noise_adaptation(adaptation(0.99));

    let mut rng = StdRng::seed_from_u64(90);
    let jerk = Normal::new(0.0, acceleration * dt).unwrap();
    let sensor = Normal::new(0.0, 0.1).unwrap();

    let (mut position, mut velocity) = (0.0, 0.0);
    let mut nis_sum = 0.0;
    for step in 0..STEPS {
        position += dt * velocity;
        velocity += jerk.sample(&mut rng);
        filter.predict();
        measurement
            .measurement_vector_mut()
            .set_at(0, 0, position + sensor.sample(&mut rng));
        assert!(filter.correct(&mut measurement).is_applied());

        let q = filter.direct_process_noise();
        let (a, b, c, d) = (
            q.get_at(0, 0),
            q.get_at(0, 1),
            q.get_at(1, 0),
            q.get_at(1, 1),
        );
        assert_eq!(b, c, "Q is not symmetric at step {step}");
        assert!(
            a >= 0.0 && d >= 0.0,
            "Q has a negative variance at step {step}"
        );
        assert!(
            a * d - b * c >= -1e-12 * (a * d).max(1e-300),
            "Q is not positive semi-definite at step {step}: {a} {b} {d}"
        );

        if step >= STEPS / 2 {
            nis_sum += measurement.nis();
        }
    }

    // A consistent filter averages a NIS of 1 per scalar measurement.
    let nis = nis_sum / (STEPS / 2) as f64;
    assert!((0.8..1.25).contains(&nis), "mean NIS = {nis}");
}

/// Enabling and then disabling the adaptation leaves the filter's behavior unchanged.
#[test]
fn disabled_adaptation_leaves_the_filter_unchanged() {
    /// Runs a short random walk and returns the final state, `P` and `Q`.
    fn run(toggle: bool) -> [f64; 3] {
        let builder = RegularBuilder::<1, f64>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<1>();
        filter.state_transition_mut().make_identity();
        filter.direct_process_noise_mut().make_scalar(0.1);
        filter.estimate_covariance_mut().make_scalar(1.0);
        measurement.observation_matrix_mut().make_identity();
        measurement
            .measurement_noise_covariance_mut()
            .make_scalar(1.0);
        if toggle {
            filter.set_process_noise_adaptation(adaptation(0.9));
            filter.set_process_noise_adaptation(None);
        }
        assert_eq!(filter.process_noise_adaptation(), None);

        let mut rng = StdRng::seed_from_u64(91);
        let noise = Normal::new(0.0, 1.0).unwrap();
        for _ in 0..200 {
            filter.predict();
            measurement
                .measurement_vector_mut()
                .set_at(0, 0, noise.sample(&mut rng));
            filter.correct(&mut measurement);
        }

        [
            filter.state_vector().get_at(0, 0),
            filter.estimate_covariance().get_at(0, 0),
            filter.direct_process_noise().get_at(0, 0),
        ]
    }

    assert_eq!(run(true), run(false));
}
