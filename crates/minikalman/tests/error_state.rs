//! The error-state recipe from the `extended` module docs, run on a vector space.
//!
//! On `ℝⁿ` the group product is addition and `Exp` is the identity, so the error-state filter
//! (nominal state outside the filter, error `x = 0` between steps, injection after each
//! correction) must reproduce the plain extended filter's estimate and covariance.

#![forbid(unsafe_code)]
#![cfg(feature = "alloc")]

use minikalman::extended::builder::{KalmanFilterBuilder, KalmanFilterType};
use minikalman::prelude::*;

/// Time step of the constant-velocity model in seconds.
const DELTA_T: f32 = 0.5;

/// Position measurements of the test track.
const MEASUREMENTS: [f32; 8] = [0.4, 1.1, 1.4, 2.2, 2.4, 3.1, 3.4, 4.2];

/// Creates a constant-velocity filter `(position, velocity)` with its noise and Jacobian set.
fn new_filter() -> KalmanFilterType<2, f32> {
    let mut filter = KalmanFilterBuilder::<2, f32>::default().build();

    filter.state_transition_jacobian_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 1, DELTA_T);
    });
    filter.estimate_covariance_mut().make_scalar(4.0);
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter
}

/// Returns the next `(position, velocity)` of the constant-velocity model.
fn transition(position: f32, velocity: f32) -> [f32; 2] {
    [position + DELTA_T * velocity, velocity]
}

/// Runs the error-state recipe and the plain filter on the same track and compares every step.
#[test]
fn error_state_recipe_matches_plain_filter_on_a_vector_space() {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut plain = new_filter();
    let mut error = new_filter();
    let mut plain_fix = builder.observations().build::<1>();
    let mut error_fix = builder.observations().build::<1>();
    let mut nominal = [0.0_f32; 2];

    for fix in [&mut plain_fix, &mut error_fix] {
        fix.observation_jacobian_matrix_mut().set_at(0, 0, 1.0);
        fix.measurement_noise_covariance_mut().make_scalar(0.25);
    }

    for z in MEASUREMENTS {
        plain.predict_nonlinear(|x, next| {
            let [p, v] = transition(x.get_row(0), x.get_row(1));
            next.set_row(0, p);
            next.set_row(1, v);
        });
        plain_fix.measurement_vector_mut().set_row(0, z);
        plain.correct_nonlinear(&mut plain_fix, |x, h| h.set_row(0, x.get_row(0)));

        // Predict: move the nominal state, keep the error at zero.
        nominal = transition(nominal[0], nominal[1]);
        error.predict_nonlinear(|_, next| {
            next.set_row(0, 0.0);
            next.set_row(1, 0.0);
        });

        // Correct: h reads the nominal state, the error is zero.
        error_fix.measurement_vector_mut().set_row(0, z);
        error.correct_nonlinear(&mut error_fix, |_, h| h.set_row(0, nominal[0]));

        // Inject and reset; the reset Jacobian is the identity on a vector space.
        nominal[0] += error.state_vector().get_row(0);
        nominal[1] += error.state_vector().get_row(1);
        error.state_vector_mut().apply(|x| {
            x.set_row(0, 0.0);
            x.set_row(1, 0.0);
        });

        for (i, value) in nominal.iter().enumerate() {
            assert!((plain.state_vector().get_row(i) - value).abs() < 1e-4);
        }

        for r in 0..2 {
            for c in 0..2 {
                let expected = plain.estimate_covariance().get_at(r, c);
                assert!((error.estimate_covariance().get_at(r, c) - expected).abs() < 1e-5);
            }
        }
    }

    assert!(
        (nominal[1] - 1.0).abs() < 0.5,
        "velocity estimate {}",
        nominal[1]
    );
}
