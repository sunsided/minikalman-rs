//! The information filter against the regular filter, and its handling of singular matrices.

#![forbid(unsafe_code)]
#![cfg(all(
    feature = "alloc",
    any(feature = "std", feature = "libm", feature = "micromath")
))]
#![allow(non_snake_case)]

use minikalman::prelude::*;
use minikalman::regular::builder::{InformationFilterType, KalmanFilterBuilder};

/// Absolute tolerance for comparing `f32` results of the two filter forms.
const TOLERANCE: f32 = 1e-3;

/// Sets a constant-velocity state transition with `dt = 0.5`.
fn set_transition(A: &mut impl MatrixMut<2, 2, f32>) {
    A.apply(|a| {
        a.set(0, 0, 1.0);
        a.set(0, 1, 0.5);
        a.set(1, 0, 0.0);
        a.set(1, 1, 1.0);
    });
}

/// Sets some correlated process noise.
fn set_noise(Q: &mut impl MatrixMut<2, 2, f32>) {
    Q.apply(|q| {
        q.set(0, 0, 0.05);
        q.set_symmetric(0, 1, 0.01);
        q.set(1, 1, 0.1);
    });
}

/// Asserts that two buffers of the same length agree to within [`TOLERANCE`].
fn assert_close(what: &str, actual: &[f32], expected: &[f32]) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() < TOLERANCE,
            "{what}[{i}]: {a} != {e} (actual {actual:?}, expected {expected:?})"
        );
    }
}

/// Recovers `x` and `P` from the information filter and compares them to the regular filter.
fn assert_matches(
    what: &str,
    info: &mut InformationFilterType<2, f32>,
    regular: &minikalman::regular::builder::KalmanFilterType<2, f32>,
) {
    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
    assert!(info.state(&mut x, &mut P), "{what}: Y is not invertible");

    assert_close(what, x.as_slice(), regular.state_vector().as_slice());
    assert_close(what, P.as_slice(), regular.estimate_covariance().as_slice());
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt and recip compound over the extra inversions"
)]
fn two_sensors_match_sequential_corrections() {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut regular = builder.build();
    let mut info = builder.build_information();

    // A position sensor that also sees half the velocity.
    let mut first = builder.observations().build::<1>();
    first.observation_matrix_mut().apply(|h| {
        h.set(0, 0, 1.0);
        h.set(0, 1, 0.5);
    });
    first.measurement_noise_covariance_mut().set(0, 0, 0.5);

    // A two-channel sensor with correlated noise.
    let mut second = builder.observations().build::<2>();
    second.observation_matrix_mut().apply(|h| {
        h.set(0, 1, 1.0);
        h.set(1, 0, 1.0);
        h.set(1, 1, 1.0);
    });
    second.measurement_noise_covariance_mut().apply(|r| {
        r.set(0, 0, 1.0);
        r.set_symmetric(0, 1, 0.2);
        r.set(1, 1, 2.0);
    });

    set_transition(regular.state_transition_mut());
    set_noise(regular.direct_process_noise_mut());
    set_transition(info.state_transition_mut());
    set_noise(info.process_noise_mut());
    regular.state_vector_mut().apply(|x| {
        x.set(0, 0, 1.0);
        x.set(1, 0, -0.5);
    });
    regular.estimate_covariance_mut().apply(|p| {
        p.set(0, 0, 4.0);
        p.set_symmetric(0, 1, 0.5);
        p.set(1, 1, 1.0);
    });
    assert!(info.set_state(regular.state_vector(), regular.estimate_covariance()));

    for (step, (z1, z2)) in [(3.0, [0.8, 3.5]), (3.6, [1.1, 4.4])]
        .into_iter()
        .enumerate()
    {
        first.measurement_vector_mut().set(0, 0, z1);
        second.measurement_vector_mut().apply(|z| {
            z.set(0, 0, z2[0]);
            z.set(1, 0, z2[1]);
        });

        regular.correct(&mut first);
        regular.correct(&mut second);
        assert!(info.fuse(&mut first));
        assert!(info.fuse(&mut second));
        assert_matches(&format!("correct {step}"), &mut info, &regular);

        regular.predict();
        assert!(info.predict());
        assert_matches(&format!("predict {step}"), &mut info, &regular);
    }
}

/// Copies the filter's `Y` and `ŷ`, to check that a failed step left them untouched.
fn snapshot(info: &InformationFilterType<2, f32>) -> Vec<f32> {
    let mut values = info.information_matrix().as_slice().to_vec();
    values.extend_from_slice(info.information_vector().as_slice());
    values
}

#[test]
fn fuse_skips_non_positive_definite_noise() {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut info = builder.build_information();
    let mut sensor = builder.observations().build::<1>();
    sensor.observation_matrix_mut().set(0, 0, 1.0);
    sensor.measurement_vector_mut().set(0, 0, 1.0);
    info.information_matrix_mut().make_identity();

    let before = snapshot(&info);
    assert!(!info.fuse(&mut sensor), "R = 0 must be rejected");
    assert_eq!(snapshot(&info), before);
}

#[test]
fn predict_skips_singular_information() {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut info = builder.build_information();
    let mut sensor = builder.observations().build::<1>();
    sensor.observation_matrix_mut().set(0, 0, 1.0);
    sensor.measurement_noise_covariance_mut().set(0, 0, 1.0);
    sensor.measurement_vector_mut().set(0, 0, 2.0);
    set_transition(info.state_transition_mut());
    set_noise(info.process_noise_mut());

    // Only the first state is observed, so Y is singular.
    assert!(info.fuse(&mut sensor));
    let before = snapshot(&info);
    assert!(before.iter().all(|v| v.is_finite()));

    assert!(!info.predict());
    assert_eq!(snapshot(&info), before);
}

#[test]
fn predict_skips_singular_predicted_covariance() {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut info = builder.build_information();

    // Y is invertible, but A = 0 and Q = 0 make P⁻ = 0.
    info.information_matrix_mut().make_identity();
    info.information_vector_mut().apply(|y| {
        y.set(0, 0, 1.0);
        y.set(1, 0, 2.0);
    });

    let before = snapshot(&info);
    assert!(!info.predict());
    assert_eq!(snapshot(&info), before);
}

#[test]
fn conversions_skip_singular_matrices() {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut info = builder.build_information();
    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
    x.set(0, 0, 1.0);

    // Y = 0 cannot be turned into a covariance; P = 0 cannot be turned into information.
    assert!(!info.state(&mut x, &mut P));
    assert_eq!(x.as_slice(), &[1.0, 0.0]);
    assert_eq!(P.as_slice(), &[0.0; 4]);

    info.information_matrix_mut().make_identity();
    let before = snapshot(&info);
    assert!(!info.set_state(&x, &P));
    assert_eq!(snapshot(&info), before);
}
