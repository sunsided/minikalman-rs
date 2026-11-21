//! Invariants: properties every step of every filter must keep, whatever the numbers.
//!
//! * `P` stays symmetric and positive semidefinite.
//! * A correction that is not applied ([`CorrectionStatus::Gated`],
//!   [`CorrectionStatus::NotPositiveDefinite`]) leaves `x` and `P` bit-for-bit unchanged.
//! * Runs are deterministic: the same inputs give bit-identical outputs.

use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;

use crate::reference::{cv, pendulum, radar};
use crate::scenarios::{self, CV, GRAVITY, History};
use crate::support::{Real, assert_covariance, load, read, steps};

/// Asserts that every filtered covariance of a run is symmetric and positive semidefinite.
fn assert_covariances<T: Real>(what: &str, history: &History, n: usize) {
    for (k, p) in steps(&history.p_post, n * n).enumerate() {
        assert_covariance(&format!("{what} {} step {k}", T::NAME), p, n, T::COVARIANCE);
    }
}

/// Checks the covariances of every reference run in number type `T`.
fn covariances_stay_valid<T: Real>() {
    let cv_run = scenarios::run_regular::<2, 1, T>(&CV);
    assert_covariances::<T>("kf cv", &cv_run, 2);
    assert_covariances::<T>(
        "kf gravity",
        &scenarios::run_regular::<3, 1, T>(&GRAVITY),
        3,
    );
    assert_covariances::<T>("ekf radar", &scenarios::run_extended_radar::<T>(), 4);
    assert_covariances::<T>("ukf pendulum", &scenarios::run_unscented_pendulum::<T>(), 2);
    assert_covariances::<T>(
        "information cv",
        &scenarios::run_information::<2, 1, T>(&CV),
        2,
    );

    let (_, smoothed) = scenarios::smooth::<2, T>(&CV, &cv_run);
    let smoothed = History {
        p_post: smoothed,
        ..History::default()
    };
    assert_covariances::<T>("rts cv", &smoothed, 2);
}

/// Every covariance of every scenario stays symmetric positive semidefinite in `f64`.
#[test]
fn covariance_stays_symmetric_positive_semidefinite_f64() {
    covariances_stay_valid::<f64>();
}

/// Every covariance of every scenario stays symmetric positive semidefinite in `f32`.
#[test]
fn covariance_stays_symmetric_positive_semidefinite_f32() {
    covariances_stay_valid::<f32>();
}

/// The KF covariances on `cv` and `gravity` (`I16F16`, `I32F32`) and the RTS covariances on
/// `cv` (`I32F32`) stay symmetric positive semidefinite.
#[test]
#[cfg(feature = "fixed")]
fn covariance_stays_symmetric_positive_semidefinite_fixed() {
    use fixed::types::{I16F16, I32F32};

    let cv_run = scenarios::run_regular::<2, 1, I16F16>(&CV);
    assert_covariances::<I16F16>("kf cv", &cv_run, 2);
    assert_covariances::<I16F16>(
        "kf gravity",
        &scenarios::run_regular::<3, 1, I16F16>(&GRAVITY),
        3,
    );

    let cv_run = scenarios::run_regular::<2, 1, I32F32>(&CV);
    assert_covariances::<I32F32>("kf cv", &cv_run, 2);
    assert_covariances::<I32F32>(
        "kf gravity",
        &scenarios::run_regular::<3, 1, I32F32>(&GRAVITY),
        3,
    );

    let (_, smoothed) = scenarios::smooth::<2, I32F32>(&CV, &cv_run);
    let smoothed = History {
        p_post: smoothed,
        ..History::default()
    };
    assert_covariances::<I32F32>("rts cv", &smoothed, 2);
}

/// The bit patterns of a state and covariance, to prove they did not change.
fn snapshot<const N: usize>(x: &impl Matrix<N, 1, f64>, p: &impl Matrix<N, N, f64>) -> Vec<u64> {
    read(x)
        .into_iter()
        .chain(read(p))
        .map(f64::to_bits)
        .collect()
}

/// A measurement far outside the prediction, and a gate it fails.
const OUTLIER: f64 = 1e3;

/// A gate far below the outlier's NIS.
const GATE: f64 = 9.0;

/// A measurement noise covariance that makes `S = H×P×Hᵀ + R` negative.
const NEGATIVE_R: f64 = -1e6;

/// Gated and non-positive-definite KF corrections leave `x` and `P` bit-identical.
#[test]
fn rejected_regular_corrections_leave_the_estimate_untouched() {
    let (mut filter, mut measurement) = scenarios::regular::<2, 1, f64>(&CV);
    filter.predict();
    let before = snapshot(filter.state_vector(), filter.estimate_covariance());

    measurement.set_gate(Some(GATE));
    measurement
        .measurement_vector_mut()
        .apply(|z| load(z, &[OUTLIER]));
    assert_eq!(filter.correct(&mut measurement), CorrectionStatus::Gated);
    assert_eq!(
        before,
        snapshot(filter.state_vector(), filter.estimate_covariance())
    );

    measurement.set_gate(None);
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, &[NEGATIVE_R]));
    assert_eq!(
        filter.correct(&mut measurement),
        CorrectionStatus::NotPositiveDefinite
    );
    assert_eq!(
        before,
        snapshot(filter.state_vector(), filter.estimate_covariance())
    );
}

/// Gated and non-positive-definite EKF corrections leave `x` and `P` bit-identical.
#[test]
fn rejected_extended_corrections_leave_the_estimate_untouched() {
    let (mut filter, mut measurement) = scenarios::extended_radar::<f64>();
    scenarios::radar_predict(&mut filter);
    let before = snapshot(filter.state_vector(), filter.estimate_covariance());

    measurement.set_gate(Some(GATE));
    let z = [radar::Z[0] + OUTLIER, radar::Z[1]];
    let status = scenarios::radar_correct(&mut filter, &mut measurement, &z);
    assert_eq!(status, CorrectionStatus::Gated);
    assert_eq!(
        before,
        snapshot(filter.state_vector(), filter.estimate_covariance())
    );

    measurement.set_gate(None);
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, &[NEGATIVE_R, 0.0, 0.0, NEGATIVE_R]));
    let status = scenarios::radar_correct(&mut filter, &mut measurement, &radar::Z[..2]);
    assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!(
        before,
        snapshot(filter.state_vector(), filter.estimate_covariance())
    );
}

/// Gated and non-positive-definite UKF corrections leave `x` and `P` bit-identical.
#[test]
fn rejected_unscented_corrections_leave_the_estimate_untouched() {
    let (mut filter, mut measurement) = scenarios::unscented_pendulum::<f64>();
    scenarios::pendulum_predict(&mut filter);
    let before = snapshot(filter.state_vector(), filter.estimate_covariance());

    measurement.set_gate(Some(GATE));
    let z = [pendulum::Z[0] + OUTLIER];
    let status = scenarios::pendulum_correct(&mut filter, &mut measurement, &z);
    assert_eq!(status, CorrectionStatus::Gated);
    assert_eq!(
        before,
        snapshot(filter.state_vector(), filter.estimate_covariance())
    );

    measurement.set_gate(None);
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, &[NEGATIVE_R]));
    let status = scenarios::pendulum_correct(&mut filter, &mut measurement, &pendulum::Z[..1]);
    assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!(
        before,
        snapshot(filter.state_vector(), filter.estimate_covariance())
    );
}

/// A fusion with a non-positive-definite `R` leaves `Y` and `ŷ` bit-identical.
#[test]
fn rejected_information_fusion_leaves_the_estimate_untouched() {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut info = builder.build_information();
    let mut measurement = builder.observations().build::<1>();
    info.information_matrix_mut().apply(|y| load(y, &cv::P0));
    measurement
        .observation_matrix_mut()
        .apply(|h| load(h, &cv::H));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, &[NEGATIVE_R]));
    let before = snapshot(info.information_vector(), info.information_matrix());

    assert!(!info.fuse(&mut measurement));
    assert_eq!(
        before,
        snapshot(info.information_vector(), info.information_matrix())
    );
}

/// Runs every reference scenario in `T` and returns all recorded bits.
fn all_runs<T: Real>() -> Vec<u64> {
    let cv_run = scenarios::run_regular::<2, 1, T>(&CV);
    let (x, p) = scenarios::smooth::<2, T>(&CV, &cv_run);
    [
        cv_run.bits(),
        scenarios::run_regular::<3, 1, T>(&GRAVITY).bits(),
        scenarios::run_extended_radar::<T>().bits(),
        scenarios::run_unscented_pendulum::<T>().bits(),
        scenarios::run_information::<2, 1, T>(&CV).bits(),
        x.into_iter().chain(p).map(f64::to_bits).collect(),
    ]
    .concat()
}

/// Running every scenario twice gives bit-identical results.
#[test]
fn reruns_are_bit_identical() {
    assert_eq!(all_runs::<f64>(), all_runs::<f64>());
    assert_eq!(all_runs::<f32>(), all_runs::<f32>());
    #[cfg(feature = "fixed")]
    {
        let first = scenarios::run_regular::<2, 1, fixed::types::I16F16>(&CV).bits();
        assert_eq!(
            first,
            scenarios::run_regular::<2, 1, fixed::types::I16F16>(&CV).bits()
        );
    }
}
