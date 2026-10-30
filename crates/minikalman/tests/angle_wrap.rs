//! Measurements that wrap around, corrected with and without normalizing the innovation.

#![forbid(unsafe_code)]
#![cfg(feature = "alloc")]

use minikalman::angles;
use minikalman::prelude::*;

/// A compass heading in degrees: the filter believes 359°, the compass reads 1°.
fn compass_after_one_reading(normalize: bool) -> f32 {
    use minikalman::regular::builder::KalmanFilterBuilder;

    let builder = KalmanFilterBuilder::<1, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();

    filter.state_transition_mut().make_identity();
    filter.state_vector_mut().set_at(0, 0, 359.0);
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);

    if normalize {
        filter.correct_with(&mut measurement, |y| y[0] = angles::wrap_180(y[0]));
    } else {
        filter.correct(&mut measurement);
    }

    filter.state_vector().get_at(0, 0)
}

#[test]
fn compass_heading_crosses_north() {
    // Equal variances give a gain of about ½ (less exact with `micromath`). The innovation
    // is +2°, so the estimate moves up across north, to about 360°.
    let heading = compass_after_one_reading(true);
    assert!(heading > 359.5 && heading <= 361.0, "heading = {heading}");
}

#[test]
fn compass_heading_without_normalizing_goes_the_long_way() {
    // Without the fix, the innovation is 1 - 359 = -358° and the estimate swings
    // to about 180°.
    let heading = compass_after_one_reading(false);
    assert!(heading > 150.0 && heading < 210.0, "heading = {heading}");
}

/// A static target at `(-10, 0.1)` seen from the origin, bearing just below +π.
/// The prior is mirrored at `(-10, -0.1)`, bearing just above -π.
fn bearing_target_y_after_one_reading(normalize: bool) -> f64 {
    use minikalman::extended::builder::KalmanFilterBuilder;

    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();

    let (true_x, true_y) = (-10.0_f64, 0.1_f64);
    let (x, y) = (-10.0_f64, -0.1_f64);

    filter.state_vector_mut().apply(|s| {
        s.set_at(0, 0, x);
        s.set_at(1, 0, y);
    });
    filter.estimate_covariance_mut().make_scalar(1.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(0.01);
    measurement
        .measurement_vector_mut()
        .set_at(0, 0, true_y.atan2(true_x));

    let norm_sq = x * x + y * y;
    measurement.observation_jacobian_matrix_mut().apply(|h| {
        h.set_at(0, 0, -y / norm_sq);
        h.set_at(0, 1, x / norm_sq);
    });

    if normalize {
        filter.correct_nonlinear_with(
            &mut measurement,
            |state, observation| observation.set_at(0, 0, state[1].atan2(state[0])),
            |y| y[0] = angles::wrap_pi(y[0]),
        );
    } else {
        filter.correct_nonlinear(&mut measurement, |state, observation| {
            observation.set_at(0, 0, state[1].atan2(state[0]))
        });
    }

    filter.state_vector().get_at(1, 0)
}

#[test]
fn bearing_crosses_the_atan2_discontinuity() {
    // The true bearing differs by about 0.02 rad; the estimate moves towards y = +0.1.
    let y = bearing_target_y_after_one_reading(true);
    assert!(y > -0.1 && y < 0.2, "y = {y}");
}

#[test]
fn bearing_without_normalizing_jumps() {
    // Without the fix, the innovation is about -2π and the estimate is thrown far off.
    let y = bearing_target_y_after_one_reading(false);
    assert!(y.abs() > 10.0, "y = {y}");
}
