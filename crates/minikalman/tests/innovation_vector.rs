//! The innovation vector exposed after a correction, as read by learned-gain filters.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;

/// Corrects a scalar regular filter at `x = 1`, `P = R = H = 1` with `z = 4` and returns the
/// innovation and the status.
fn regular_innovation(gate: Option<f64>, with_gain: bool) -> (f64, CorrectionStatus) {
    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_vector_mut().set_at(0, 0, 1.0);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 4.0);
    measurement.set_gate(gate);

    let status = if with_gain {
        measurement.kalman_gain_mut().set_at(0, 0, 0.25);
        filter.correct_with_gain(&mut measurement)
    } else {
        filter.correct(&mut measurement)
    };

    (measurement.innovation_vector().get_at(0, 0), status)
}

/// The innovation of an applied correction is `z − H×x` of the prior state.
#[test]
fn regular_applied() {
    assert_eq!(
        regular_innovation(None, false),
        (3.0, CorrectionStatus::Applied)
    );
}

/// A gated correction still exposes the innovation it was gated on.
#[test]
fn regular_gated() {
    assert_eq!(
        regular_innovation(Some(1.0), false),
        (3.0, CorrectionStatus::Gated)
    );
}

/// The externally supplied gain path writes the innovation as well.
#[test]
fn regular_with_gain() {
    assert_eq!(
        regular_innovation(None, true),
        (3.0, CorrectionStatus::Applied)
    );
}

/// The innovation of an extended correction is `z − h(x)` of the prior state.
#[test]
fn extended_applied() {
    let builder = ExtendedBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_vector_mut().set_at(0, 0, 2.0);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement
        .observation_jacobian_matrix_mut()
        .set_at(0, 0, 4.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 5.0);

    // h(x) = x², so z − h(x) = 5 − 4 = 1.
    let status = filter.correct_nonlinear(&mut measurement, |state, observation| {
        observation.set_at(0, 0, state.get_at(0, 0) * state.get_at(0, 0));
    });

    assert_eq!(status, CorrectionStatus::Applied);
    assert_eq!(measurement.innovation_vector().get_at(0, 0), 1.0);
}
