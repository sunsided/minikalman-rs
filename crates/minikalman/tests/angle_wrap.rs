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

/// A heading near π whose transition wraps it, so the propagated sigma points land on
/// both sides of ±π. Returns the predicted heading and its variance.
fn ukf_heading_after_predict(normalize: bool) -> (f64, f64) {
    use minikalman::unscented::builder::KalmanFilterBuilder;

    let builder = KalmanFilterBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    filter.set_alpha(1.0);
    filter.state_vector_mut().set_at(0, 0, 3.1);
    filter.estimate_covariance_mut().make_scalar(0.01);
    filter.direct_process_noise_mut().make_scalar(0.0);
    if normalize {
        filter.set_state_residual(|dx| dx[0] = angles::wrap_pi(dx[0]));
    }

    // Turn slowly; keep the heading in (-π, π].
    filter.predict_nonlinear(|next| {
        let heading = next.as_matrix().get_at(0, 0);
        next.as_matrix_mut()
            .set_at(0, 0, angles::wrap_pi(heading + 0.05));
    });

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
    )
}

#[test]
fn ukf_heading_mean_across_pi() {
    // 3.1 + 0.05 = 3.15, which wraps to 3.15 - 2π. The spread stays at about 0.01 (Q = 0);
    // the tolerances allow for `micromath`'s approximate square root.
    let (heading, variance) = ukf_heading_after_predict(true);
    let expected = 3.15 - 2.0 * core::f64::consts::PI;
    assert!((heading - expected).abs() < 1e-3, "heading = {heading}");
    assert!((variance - 0.01).abs() < 0.003, "variance = {variance}");
}

#[test]
fn ukf_heading_mean_without_normalizing_is_wrong() {
    // Sigma points at ≈+3.0 and ≈-3.0 average to about -π/2, with a huge variance.
    let (heading, variance) = ukf_heading_after_predict(false);
    assert!(heading.abs() < 2.0, "heading = {heading}");
    assert!(variance > 1.0, "variance = {variance}");
}

/// The bearing test from above, for the UKF: the prior's sigma points straddle ±π.
/// Returns the corrected `y` and its variance.
fn ukf_bearing_target_after_one_reading(normalize: bool) -> (f64, f64) {
    use minikalman::unscented::builder::KalmanFilterBuilder;

    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut filter = builder.build::<5>();
    let mut measurement = builder.observations().build::<1, 5>();
    filter.set_alpha(1.0);

    let (true_x, true_y) = (-10.0_f64, 0.1_f64);
    filter.state_vector_mut().apply(|s| {
        s.set_at(0, 0, -10.0);
        s.set_at(1, 0, -0.1);
    });
    filter.estimate_covariance_mut().apply(|p| {
        p.make_scalar(1.0);
        p.set_at(1, 1, 0.04);
    });
    filter.direct_process_noise_mut().make_scalar(0.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(0.0001);
    measurement
        .measurement_vector_mut()
        .set_at(0, 0, true_y.atan2(true_x));

    // Static target; the prediction only spreads the sigma points.
    filter.predict_nonlinear(|_| {});

    if normalize {
        filter.correct_sigma_point_with(
            &mut measurement,
            |sigma_pred, observed| {
                for j in 0..5 {
                    observed[j] = sigma_pred[5 + j].atan2(sigma_pred[j]);
                }
            },
            |dz| dz[0] = angles::wrap_pi(dz[0]),
        );
    } else {
        filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
            for j in 0..5 {
                observed[j] = sigma_pred[5 + j].atan2(sigma_pred[j]);
            }
        });
    }

    (
        filter.state_vector().get_at(1, 0),
        filter.estimate_covariance().get_at(1, 1),
    )
}

#[test]
fn ukf_bearing_across_pi() {
    // A bearing with σ = 0.01 rad at range 10 pins y to about ±0.1 around the target.
    let (y, variance) = ukf_bearing_target_after_one_reading(true);
    assert!((y - 0.1).abs() < 0.1, "y = {y}");
    assert!(variance < 0.012, "variance = {variance}");
}

#[test]
fn ukf_bearing_without_normalizing_barely_updates() {
    // Without the fix, the predicted bearing and its spread are off by about 2π, so the
    // innovation covariance is huge and the update is far too weak.
    let (_, normalized) = ukf_bearing_target_after_one_reading(true);
    let (_, variance) = ukf_bearing_target_after_one_reading(false);
    assert!(
        variance > 2.0 * normalized,
        "variance = {variance} vs. {normalized}"
    );
}
