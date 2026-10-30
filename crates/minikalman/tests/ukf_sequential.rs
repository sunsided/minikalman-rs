//! Several UKF corrections between two predictions, e.g. one per sensor.

#![forbid(unsafe_code)]
#![cfg(feature = "alloc")]

use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder;

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs(),
        "{what} = {actual}, expected {expected}"
    );
}

/// Measures a single state `n` times with variance `r` after one prediction and
/// returns the posterior mean and variance.
fn measure_repeatedly(n: usize, z: f64, r: f64) -> (f64, f64) {
    let builder = KalmanFilterBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.set_alpha(1.0);

    filter.state_vector_mut().set_at(0, 0, 0.0);
    filter.estimate_covariance_mut().make_scalar(1.0);
    filter.direct_process_noise_mut().make_scalar(0.0);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(r);
    measurement.measurement_vector_mut().set_at(0, 0, z);

    filter.predict_nonlinear(|_| {});
    for _ in 0..n {
        filter.correct_sigma_point(&mut measurement, |sigma, observed| {
            for k in 0..3 {
                observed[k] = sigma[k];
            }
        });
    }

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
    )
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt and recip are off by up to ~25% here"
)]
fn single_correction_matches_closed_form() {
    // P = 1, R = 1: gain ½, posterior variance ½.
    let (x, p) = measure_repeatedly(1, 2.0, 1.0);
    assert_close(x, 1.0, "x");
    assert_close(p, 0.5, "p");
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt and recip are off by up to ~25% here"
)]
fn sequential_corrections_match_one_combined_measurement() {
    // Three independent measurements with R = 1 each equal one with R = 1/3:
    // posterior variance 1 / (1 + 3) and mean ¾ of the way to z.
    let (x, p) = measure_repeatedly(3, 2.0, 1.0);
    assert_close(p, 0.25, "p");
    assert_close(x, 1.5, "x");
}

#[test]
fn many_sequential_corrections_keep_the_covariance_positive() {
    // Before redrawing the sigma points, the second correction already drove P to 0
    // and the third made it negative.
    let (_, p) = measure_repeatedly(20, 0.0, 1.0);
    assert!(p > 0.0 && p < 1.0, "p = {p}");
    if !cfg!(feature = "micromath") {
        assert_close(p, 1.0 / 21.0, "p");
    }
}
