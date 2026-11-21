//! Reference vectors: the crate against the independent NumPy/SciPy implementation.
//!
//! Each test runs one scenario of [`reference`](crate::reference) and compares every
//! recorded step with the number type's [`Real::REFERENCE`] tolerance.

use crate::reference::{cv, gravity, pendulum, radar};
use crate::scenarios::{self, History, CV, GRAVITY};
use crate::support::{assert_close, Real};

/// Compares the filtered states and covariances of a run with the reference.
fn assert_posterior<T: Real>(what: &str, history: &History, x: &[f64], p: &[f64]) {
    assert_close(
        &format!("{what} {} x_k|k", T::NAME),
        &history.x_post,
        x,
        T::REFERENCE,
    );
    assert_close(
        &format!("{what} {} P_k|k", T::NAME),
        &history.p_post,
        p,
        T::REFERENCE,
    );
}

/// Runs the regular filter on the constant-velocity scenario and compares every step.
fn regular_cv<T: Real>() {
    let history = scenarios::run_regular::<2, 1, T>(&CV);
    assert_close(
        &format!("cv {} x_k|k-1", T::NAME),
        &history.x_prior,
        &cv::X_PRIOR,
        T::REFERENCE,
    );
    assert_close(
        &format!("cv {} P_k|k-1", T::NAME),
        &history.p_prior,
        &cv::P_PRIOR,
        T::REFERENCE,
    );
    assert_posterior::<T>("cv", &history, &cv::X_POST, &cv::P_POST);
}

/// Runs the regular filter on the free-fall scenario.
fn regular_gravity<T: Real>() {
    let history = scenarios::run_regular::<3, 1, T>(&GRAVITY);
    assert_posterior::<T>("gravity", &history, &gravity::X_POST, &gravity::P_POST);
}

/// Runs the extended filter on the radar scenario.
fn extended_radar<T: Real>() {
    let history = scenarios::run_extended_radar::<T>();
    assert_posterior::<T>("radar", &history, &radar::X_POST, &radar::P_POST);
}

/// Runs the unscented filter on the pendulum scenario.
fn unscented_pendulum<T: Real>() {
    let history = scenarios::run_unscented_pendulum::<T>();
    assert_posterior::<T>("pendulum", &history, &pendulum::X_POST, &pendulum::P_POST);
}

/// Runs the information filter on the constant-velocity scenario; it must match the KF reference.
fn information_cv<T: Real>() {
    let history = scenarios::run_information::<2, 1, T>(&CV);
    assert_posterior::<T>("information cv", &history, &cv::X_POST, &cv::P_POST);
}

/// Smooths the constant-velocity run with the RTS smoother.
fn rts_cv<T: Real>() {
    let history = scenarios::run_regular::<2, 1, T>(&CV);
    let (x, p) = scenarios::smooth::<2, T>(&CV, &history);
    assert_close(
        &format!("rts cv {} x_k|N", T::NAME),
        &x,
        &cv::X_SMOOTH,
        T::REFERENCE,
    );
    assert_close(
        &format!("rts cv {} P_k|N", T::NAME),
        &p,
        &cv::P_SMOOTH,
        T::REFERENCE,
    );
}

/// The KF reproduces the constant-velocity reference in `f64`.
#[test]
fn regular_constant_velocity_f64() {
    regular_cv::<f64>();
}

/// The KF reproduces the constant-velocity reference in `f32`.
#[test]
fn regular_constant_velocity_f32() {
    regular_cv::<f32>();
}

/// The KF reproduces the constant-velocity reference in `I32F32`.
#[test]
#[cfg(feature = "fixed")]
fn regular_constant_velocity_i32f32() {
    regular_cv::<fixed::types::I32F32>();
}

/// The KF reproduces the constant-velocity reference in `I16F16`.
#[test]
#[cfg(feature = "fixed")]
fn regular_constant_velocity_i16f16() {
    regular_cv::<fixed::types::I16F16>();
}

/// The KF reproduces the free-fall reference in `f64`.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "micromath's approximate sqrt and recip put P off by up to 0.8 (130 %)"
)]
fn regular_gravity_f64() {
    regular_gravity::<f64>();
}

/// The KF reproduces the free-fall reference in `f32`.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "micromath's approximate sqrt and recip put P off by up to 0.8 (130 %)"
)]
fn regular_gravity_f32() {
    regular_gravity::<f32>();
}

/// The KF reproduces the free-fall reference in `I32F32`.
#[test]
#[cfg(feature = "fixed")]
fn regular_gravity_i32f32() {
    regular_gravity::<fixed::types::I32F32>();
}

/// The EKF reproduces the radar reference in `f64`.
#[test]
fn extended_radar_f64() {
    extended_radar::<f64>();
}

/// The EKF reproduces the radar reference in `f32`.
#[test]
fn extended_radar_f32() {
    extended_radar::<f32>();
}

/// The UKF reproduces the pendulum reference in `f64`.
#[test]
fn unscented_pendulum_f64() {
    unscented_pendulum::<f64>();
}

/// The UKF reproduces the pendulum reference in `f32`.
#[test]
fn unscented_pendulum_f32() {
    unscented_pendulum::<f32>();
}

/// The information filter reproduces the constant-velocity reference in `f64`.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "micromath's approximate recip compounds over the inversions, x off by up to 0.63"
)]
fn information_constant_velocity_f64() {
    information_cv::<f64>();
}

/// The information filter reproduces the constant-velocity reference in `f32`.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "micromath's approximate recip compounds over the inversions, x off by up to 0.63"
)]
fn information_constant_velocity_f32() {
    information_cv::<f32>();
}

/// The RTS smoother reproduces the smoothed constant-velocity reference in `f64`.
#[test]
fn rts_constant_velocity_f64() {
    rts_cv::<f64>();
}

/// The RTS smoother reproduces the smoothed constant-velocity reference in `f32`.
#[test]
fn rts_constant_velocity_f32() {
    rts_cv::<f32>();
}

/// The RTS smoother reproduces the smoothed constant-velocity reference in `I32F32`.
#[test]
#[cfg(feature = "fixed")]
fn rts_constant_velocity_i32f32() {
    rts_cv::<fixed::types::I32F32>();
}
