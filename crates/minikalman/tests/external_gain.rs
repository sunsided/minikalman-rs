//! Corrections with an externally supplied Kalman gain, compared to regular corrections.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::buffers::builder::BufferBuilder;
use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;

/// Time step of the constant-velocity test system, in seconds.
const DT: f64 = 0.1;

/// Measurement noise variance of the position readings.
const R: f64 = 0.25;

/// Number of steps after which the regular filter's gain is considered converged.
const WARMUP: usize = 500;

/// Absolute tolerance for comparing states and covariances.
const EPS: f64 = 1e-9;

/// A deterministic, noisy position reading at step `t` of a target moving at 2 units/s.
fn position_reading(t: usize) -> f64 {
    let t = t as f64;
    2.0 * DT * t + 0.3 * (1.7 * t).sin()
}

/// Builds a constant-velocity filter (state: position, velocity) and a position measurement.
macro_rules! constant_velocity {
    () => {{
        let builder = RegularBuilder::<2, f64>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<1>();

        filter.state_transition_mut().apply(|a| {
            a.make_identity();
            a.set_at(0, 1, DT);
        });
        filter.direct_process_noise_mut().apply(|q| {
            q.set_at(0, 0, 1e-4);
            q.set_at(1, 1, 1e-3);
        });
        filter.estimate_covariance_mut().make_scalar(10.0);
        measurement.observation_matrix_mut().set_at(0, 0, 1.0);
        measurement
            .measurement_noise_covariance_mut()
            .make_scalar(R);

        (filter, measurement)
    }};
}

/// Asserts that two 2×2 covariance matrices, given as row-major arrays, agree within [`EPS`].
fn assert_close(a: [f64; 4], b: [f64; 4]) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < EPS, "{a:?} != {b:?}");
    }
}

/// Copies the Kalman gain of one observation into another.
fn copy_gain<const STATES: usize>(
    from: &impl KalmanGainMatrix<STATES, 1, f64>,
    to: &mut impl KalmanGainMatrix<STATES, 1, f64>,
) {
    for i in 0..STATES {
        to[i] = from[i];
    }
}

/// Reads a 2×2 matrix into a row-major array.
fn read_2x2(m: &impl Matrix<2, 2, f64>) -> [f64; 4] {
    [m.get(0, 0), m.get(0, 1), m.get(1, 0), m.get(1, 1)]
}

/// Asserts that a 2×2 covariance matrix is symmetric and positive semi-definite.
fn assert_symmetric_psd(p: [f64; 4]) {
    let [a, b, c, d] = p;
    assert!(
        (b - c).abs() <= 1e-9 * (1.0 + b.abs()),
        "not symmetric: {p:?}"
    );
    assert!(a >= 0.0 && d >= 0.0, "negative variance: {p:?}");
    assert!(a * d - b * c >= -1e-12, "not positive semi-definite: {p:?}");
}

/// With the converged gain, an external-gain correction matches a regular correction.
#[test]
fn steady_state_gain_matches_regular_correction() {
    let (mut regular, mut regular_z) = constant_velocity!();
    let (mut external, mut external_z) = constant_velocity!();

    for t in 0..WARMUP {
        for (filter, z) in [
            (&mut regular, &mut regular_z),
            (&mut external, &mut external_z),
        ] {
            filter.predict();
            z.measurement_vector_mut().set_at(0, 0, position_reading(t));
            assert_eq!(filter.correct(z), CorrectionStatus::Applied);
        }
    }

    // Freeze the converged gain on the external filter.
    copy_gain(regular_z.kalman_gain(), external_z.kalman_gain_mut());

    for t in WARMUP..WARMUP + 200 {
        regular.predict();
        external.predict();
        regular_z
            .measurement_vector_mut()
            .set_at(0, 0, position_reading(t));
        external_z
            .measurement_vector_mut()
            .set_at(0, 0, position_reading(t));

        regular.correct(&mut regular_z);
        let status = external.correct_with_gain(&mut external_z);

        assert_eq!(status, CorrectionStatus::Applied);
        assert_close(
            read_2x2(regular.estimate_covariance().as_matrix()),
            read_2x2(external.estimate_covariance().as_matrix()),
        );
        for i in 0..2 {
            let (a, b) = (
                regular.state_vector().get_at(i, 0),
                external.state_vector().get_at(i, 0),
            );
            assert!((a - b).abs() < EPS, "x[{i}]: {a} != {b}");
        }
    }
}

/// With the optimal gain of each step, `correct_with_gain_with` matches `correct_with`,
/// including the normalized innovation.
#[test]
fn optimal_gain_with_fix_matches_regular_correction() {
    let (mut regular, mut regular_z) = constant_velocity!();
    let (mut external, mut external_z) = constant_velocity!();
    let fix = |y: &mut [f64]| y[0] = y[0].clamp(-0.1, 0.1);

    for t in 0..50 {
        regular.predict();
        external.predict();
        regular_z
            .measurement_vector_mut()
            .set_at(0, 0, position_reading(t));
        external_z
            .measurement_vector_mut()
            .set_at(0, 0, position_reading(t));

        regular.correct_with(&mut regular_z, fix);
        copy_gain(regular_z.kalman_gain(), external_z.kalman_gain_mut());
        external.correct_with_gain_with(&mut external_z, fix);

        assert_close(
            read_2x2(regular.estimate_covariance().as_matrix()),
            read_2x2(external.estimate_covariance().as_matrix()),
        );
        assert!(
            (regular.state_vector().get_at(0, 0) - external.state_vector().get_at(0, 0)).abs()
                < EPS
        );
    }
}

/// A suboptimal gain keeps `P` symmetric and positive semi-definite, and matches the
/// Joseph form `(I - K×H)×P×(I - K×H)ᵀ + K×R×Kᵀ` computed by hand.
#[test]
fn suboptimal_gain_keeps_covariance_symmetric_psd() {
    let (mut regular, mut regular_z) = constant_velocity!();
    for t in 0..WARMUP {
        regular.predict();
        regular_z
            .measurement_vector_mut()
            .set_at(0, 0, position_reading(t));
        regular.correct(&mut regular_z);
    }

    let optimal = regular_z.kalman_gain();
    for scale in [0.5, 1.5] {
        let (mut filter, mut z) = constant_velocity!();
        let (k0, k1) = (scale * optimal[0], scale * optimal[1]);
        z.kalman_gain_mut().apply(|k| {
            k.set_at(0, 0, k0);
            k.set_at(1, 0, k1);
        });

        for t in 0..2000 {
            filter.predict();
            let [p00, p01, p10, p11] = read_2x2(filter.estimate_covariance().as_matrix());
            z.measurement_vector_mut().set_at(0, 0, position_reading(t));
            filter.correct_with_gain(&mut z);

            // With H = [1 0], (I - K×H) = [[1 - k0, 0], [-k1, 1]].
            let (a00, a10) = (1.0 - k0, -k1);
            let ap = [a00 * p00, a00 * p01, a10 * p00 + p10, a10 * p01 + p11];
            let expected = [
                ap[0] * a00 + k0 * R * k0,
                ap[0] * a10 + ap[1] + k0 * R * k1,
                ap[2] * a00 + k1 * R * k0,
                ap[2] * a10 + ap[3] + k1 * R * k1,
            ];
            let actual = read_2x2(filter.estimate_covariance().as_matrix());

            assert_symmetric_psd(actual);
            for (e, a) in expected.iter().zip(actual.iter()) {
                assert!(
                    (e - a).abs() <= 1e-9 * (1.0 + e.abs()),
                    "{expected:?} != {actual:?}"
                );
            }
        }
    }
}

/// The observation-level methods and the gain accessors work without a filter.
#[test]
fn observation_correct_with_gain_applies_gain() {
    let builder = RegularBuilder::<1, f64>::default();
    let mut measurement = builder.observations().build::<1>();
    let mut x = BufferBuilder::state_vector_x::<1>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<1>().new();
    p.set_at(0, 0, 1.0);
    measurement.observation_matrix_mut().set_at(0, 0, 1.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 4.0);
    measurement.kalman_gain_mut().set_at(0, 0, 0.25);
    assert_eq!(measurement.kalman_gain().get_at(0, 0), 0.25);

    let status = measurement.correct_with_gain(&mut x, &mut p);

    // x = 0.25 × 4; P = (1 - 0.25)² × 1 + 0.25² × 1.
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((x.get_at(0, 0) - 1.0).abs() < EPS);
    assert!((p.get_at(0, 0) - 0.625).abs() < EPS);
    assert_eq!(measurement.nis(), 0.0, "the NIS is not updated");
}

/// The observation function `h(x) = x²` of the scalar extended test system.
fn square(x: &impl Matrix<1, 1, f64>, y: &mut impl MatrixMut<1, 1, f64>) {
    y.set_at(0, 0, x.get_at(0, 0) * x.get_at(0, 0));
}

/// With the gain of each step, the extended external-gain correction matches
/// `correct_nonlinear` for `h(x) = x²`, with and without normalizing the innovation.
#[test]
fn extended_gain_matches_nonlinear_correction() {
    for with_fix in [false, true] {
        let builder = ExtendedBuilder::<1, f64>::default();
        let mut regular = builder.build();
        let mut regular_z = builder.observations().build::<1>();
        let mut external = builder.build();
        let mut external_z = builder.observations().build::<1>();
        let fix = |y: &mut [f64]| y[0] = y[0].clamp(-0.5, 0.5);

        for (filter, z) in [
            (&mut regular, &mut regular_z),
            (&mut external, &mut external_z),
        ] {
            filter.state_vector_mut().set_at(0, 0, 1.0);
            filter.estimate_covariance_mut().make_scalar(1.0);
            filter.direct_process_noise_mut().make_scalar(1e-3);
            filter.state_transition_jacobian_mut().make_identity();
            z.measurement_noise_covariance_mut().make_scalar(0.1);
        }

        for t in 0..50 {
            let reading = 4.0 + 0.2 * (t as f64).sin();
            for (filter, z) in [
                (&mut regular, &mut regular_z),
                (&mut external, &mut external_z),
            ] {
                filter.predict_nonlinear(|current, next| next[0] = current[0]);
                let jacobian = 2.0 * filter.state_vector().get_at(0, 0);
                z.observation_jacobian_matrix_mut().set_at(0, 0, jacobian);
                z.measurement_vector_mut().set_at(0, 0, reading);
            }

            let status = if with_fix {
                regular.correct_nonlinear_with(&mut regular_z, square, fix);
                copy_gain(regular_z.kalman_gain(), external_z.kalman_gain_mut());
                external.correct_nonlinear_with_gain_with(&mut external_z, square, fix)
            } else {
                regular.correct_nonlinear(&mut regular_z, square);
                copy_gain(regular_z.kalman_gain(), external_z.kalman_gain_mut());
                external.correct_nonlinear_with_gain(&mut external_z, square)
            };

            assert_eq!(status, CorrectionStatus::Applied);
            let (a, b) = (
                regular.state_vector().get_at(0, 0),
                external.state_vector().get_at(0, 0),
            );
            assert!((a - b).abs() < EPS, "x: {a} != {b}");
            let (a, b) = (
                regular.estimate_covariance().get_at(0, 0),
                external.estimate_covariance().get_at(0, 0),
            );
            assert!((a - b).abs() < EPS, "P: {a} != {b}");
        }

        // The estimate converges to √4 = 2.
        assert!((regular.state_vector().get_at(0, 0) - 2.0).abs() < 0.1);
    }
}

/// The observation-level nonlinear method and the gain accessors work without a filter.
#[test]
fn extended_observation_correct_with_gain_applies_gain() {
    let builder = ExtendedBuilder::<1, f64>::default();
    let mut measurement = builder.observations().build::<1>();
    let mut x = BufferBuilder::state_vector_x::<1>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<1>().new();
    x.set_at(0, 0, 1.0);
    p.set_at(0, 0, 1.0);
    measurement
        .observation_jacobian_matrix_mut()
        .set_at(0, 0, 2.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 5.0);
    measurement.kalman_gain_mut().set_at(0, 0, 0.25);
    assert_eq!(measurement.kalman_gain().get_at(0, 0), 0.25);

    let status = measurement.correct_nonlinear_with_gain(&mut x, &mut p, square);

    // y = 5 - 1² = 4, x = 1 + 0.25 × 4; P = (1 - 0.25 × 2)² × 1 + 0.25² × 1.
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((x.get_at(0, 0) - 2.0).abs() < EPS);
    assert!((p.get_at(0, 0) - 0.3125).abs() < EPS);
    assert_eq!(measurement.nis(), 0.0, "the NIS is not updated");
}
