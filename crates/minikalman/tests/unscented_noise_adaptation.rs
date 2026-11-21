//! Online adaptation of the measurement noise covariance `R` of the unscented observation.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;
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

/// Which of the sigma point corrections to run.
#[derive(Clone, Copy, Debug)]
enum Correction {
    /// `correct_sigma_point`.
    Plain,
    /// `correct_sigma_point_recalibrated`.
    Recalibrated,
}

/// Observes the single state of each of the three sigma points directly, `h(x) = x`.
fn observe_identity(sigma: &impl Matrix<1, 3, f64>, observed: &mut impl MatrixMut<1, 3, f64>) {
    for j in 0..3 {
        observed.set_at(0, j, sigma.get_at(0, j));
    }
}

/// Runs one correction of a scalar unscented filter with `x = 0`, `h(x) = x` and the given
/// `P` and `R`, without a prediction, and returns its [`Outcome`].
fn ukf_after_one_reading(
    correction: Correction,
    z: f64,
    p: f64,
    r: f64,
    gate: Option<f64>,
    adaptation: Option<NoiseAdaptation<f64>>,
) -> Outcome {
    let builder = UnscentedBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.estimate_covariance_mut().make_scalar(p);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(r);
    measurement.measurement_vector_mut().set_at(0, 0, z);
    measurement.set_gate(gate);
    measurement.set_noise_adaptation(adaptation);
    assert_eq!(measurement.noise_adaptation(), adaptation);

    let status = match correction {
        Correction::Plain => filter.correct_sigma_point(&mut measurement, |sigma, observed| {
            observe_identity(sigma, observed);
        }),
        Correction::Recalibrated => {
            filter.correct_sigma_point_recalibrated(&mut measurement, |sigma, observed| {
                observe_identity(sigma, observed);
            })
        }
    };

    (
        status,
        measurement.measurement_noise_covariance().get_at(0, 0),
    )
}

/// Checks the adaptation of `R` on single readings with `P = 1` and `R = 1`, so `S = 2`.
///
/// These are the values of the regular and extended observations in `noise_adaptation.rs`.
fn check_single_readings(correction: Correction) {
    let correct =
        |z, p, r, gate, adaptation| ukf_after_one_reading(correction, z, p, r, gate, adaptation);

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
}

#[test]
fn unscented_adapts_r_on_applied_corrections_only() {
    check_single_readings(Correction::Plain);
}

#[test]
fn recalibrated_unscented_adapts_r_on_applied_corrections_only() {
    check_single_readings(Correction::Recalibrated);
}

/// The two-state, two-measurement observation matrix used to compare the filters.
const H: [[f64; 2]; 2] = [[1.0, 0.5], [0.0, 1.0]];

/// Observes `H×x` for each of the five sigma points.
fn observe_linear(sigma: &impl Matrix<2, 5, f64>, observed: &mut impl MatrixMut<2, 5, f64>) {
    for j in 0..5 {
        for (i, row) in H.iter().enumerate() {
            let value = row[0] * sigma.get_at(0, j) + row[1] * sigma.get_at(1, j);
            observed.set_at(i, j, value);
        }
    }
}

/// Writes the correlated prior covariance and measurement noise shared by both filters.
fn set_prior(p: &mut impl MatrixMut<2, 2, f64>, r: &mut impl MatrixMut<2, 2, f64>) {
    p.set_at(0, 0, 2.0);
    p.set_at(0, 1, 0.3);
    p.set_at(1, 0, 0.3);
    p.set_at(1, 1, 1.0);
    r.set_at(0, 0, 0.5);
    r.set_at(0, 1, 0.1);
    r.set_at(1, 0, 0.1);
    r.set_at(1, 1, 0.8);
}

/// Returns the four entries of a 2×2 matrix in row-major order.
fn entries(m: &impl Matrix<2, 2, f64>) -> [f64; 4] {
    [
        m.get_at(0, 0),
        m.get_at(0, 1),
        m.get_at(1, 0),
        m.get_at(1, 1),
    ]
}

/// For a linear observation, the unscented update of `R`, written in terms of `S⁻¹`, equals
/// the regular update with `H×P⁺×Hᵀ`.
#[test]
fn unscented_r_matches_regular_r_for_a_linear_observation() {
    let z = [1.5, -0.7];

    let regular = RegularBuilder::<2, f64>::default();
    let mut regular_filter = regular.build();
    let mut regular_measurement = regular.observations().build::<2>();
    regular_measurement.observation_matrix_mut().apply(|h| {
        for (i, row) in H.iter().enumerate() {
            h.set_at(i, 0, row[0]);
            h.set_at(i, 1, row[1]);
        }
    });

    let unscented = UnscentedBuilder::<2, f64>::default();
    let mut unscented_filter = unscented.build::<5>();
    let mut unscented_measurement = unscented.observations().build::<2, 5>();

    set_prior(
        regular_filter.estimate_covariance_mut(),
        regular_measurement.measurement_noise_covariance_mut(),
    );
    set_prior(
        unscented_filter.estimate_covariance_mut(),
        unscented_measurement.measurement_noise_covariance_mut(),
    );

    for (i, value) in z.iter().enumerate() {
        regular_measurement
            .measurement_vector_mut()
            .set_at(i, 0, *value);
        unscented_measurement
            .measurement_vector_mut()
            .set_at(i, 0, *value);
    }

    regular_measurement.set_noise_adaptation(adaptation(0.7));
    unscented_measurement.set_noise_adaptation(adaptation(0.7));

    assert!(regular_filter
        .correct(&mut regular_measurement)
        .is_applied());
    assert!(unscented_filter
        .correct_sigma_point(&mut unscented_measurement, |sigma, observed| {
            observe_linear(sigma, observed);
        })
        .is_applied());

    let expected = entries(regular_measurement.measurement_noise_covariance());
    let actual = entries(unscented_measurement.measurement_noise_covariance());
    assert_ne!(expected, [0.5, 0.1, 0.1, 0.8], "R did not change");
    assert_eq!(actual[1], actual[2], "R is not symmetric");
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert!(
            (e - a).abs() < 1e-12,
            "regular {expected:?} vs unscented {actual:?}"
        );
    }
}

/// Tracks a random walk with `Q = 0.01`, measured with noise variance `4`, starting with
/// `R = 1`, and returns `R` averaged over the last quarter of the run.
fn track_true_r(correction: Correction) -> f64 {
    const STEPS: usize = 8_000;
    let q = 0.01;

    let builder = UnscentedBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.direct_process_noise_mut().make_scalar(q);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.set_noise_adaptation(adaptation(0.99));

    let mut rng = StdRng::seed_from_u64(89);
    let process = Normal::new(0.0, q.sqrt()).unwrap();
    let sensor = Normal::new(0.0, 2.0).unwrap();

    let mut truth = 0.0;
    let mut r_sum = 0.0;
    for step in 0..STEPS {
        truth += process.sample(&mut rng);
        filter.predict_nonlinear(|_| {});
        measurement
            .measurement_vector_mut()
            .set_at(0, 0, truth + sensor.sample(&mut rng));
        let observe = |sigma: &_, observed: &mut _| observe_identity(sigma, observed);
        let status = match correction {
            Correction::Plain => filter.correct_sigma_point(&mut measurement, observe),
            Correction::Recalibrated => {
                filter.correct_sigma_point_recalibrated(&mut measurement, observe)
            }
        };
        assert!(
            status.is_applied(),
            "{correction:?} at step {step}: {status:?}"
        );

        if step >= STEPS * 3 / 4 {
            r_sum += measurement.measurement_noise_covariance().get_at(0, 0);
        }
    }

    r_sum / (STEPS / 4) as f64
}

/// Starting from `R = 1`, the adapted `R` converges to the true measurement noise `4`.
#[test]
fn unscented_r_converges_to_the_true_r() {
    for correction in [Correction::Plain, Correction::Recalibrated] {
        let r = track_true_r(correction);
        assert!((r - 4.0).abs() < 0.6, "{correction:?}: adapted R = {r}");
    }
}
