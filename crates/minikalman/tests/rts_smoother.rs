//! The RTS smoother must beat the filter on a noisy constant-velocity track.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::matrix::MatrixDataArray;
use minikalman::models::motion;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;
use minikalman::smoother::RtsSmoother;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Number of filter steps in the simulated run.
const STEPS: usize = 200;

/// A `[position, velocity]` state vector.
type Vector = MatrixDataArray<2, 1, 2, f32>;

/// A 2×2 covariance or scratch matrix.
type Square = MatrixDataArray<2, 2, 4, f32>;

/// Mean squared position error of `estimates` against `truth`.
fn position_mse(estimates: &[Vector], truth: &[f32]) -> f32 {
    let sum: f32 = estimates
        .iter()
        .zip(truth)
        .map(|(x, t)| (x.get(0, 0) - t).powi(2))
        .sum();

    sum / truth.len() as f32
}

/// Filters a constant-velocity track measured with noisy positions, then smooths it.
#[test]
fn smoothed_mse_is_below_filtered_mse() {
    let mut rng = StdRng::seed_from_u64(62);
    let noise = Normal::new(0.0f32, 1.0).unwrap();

    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();

    let model = motion::constant_velocity(0.1, 0.01);
    filter
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    filter
        .direct_process_noise_mut()
        .apply(|q| model.fill_process_noise(q));
    filter
        .estimate_covariance_mut()
        .apply(|p| p.make_scalar(100.0));
    measurement
        .observation_matrix_mut()
        .apply(|h| h.set(0, 0, 1.0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| r.set(0, 0, 1.0));

    let mut truth = Vec::with_capacity(STEPS);
    let mut x = vec![Vector::new([0.0; 2]); STEPS];
    let mut p = vec![Square::new([0.0; 4]); STEPS];
    let mut x_predicted = x.clone();
    let mut p_predicted = p.clone();

    for k in 0..STEPS {
        let position = 2.0 * 0.1 * k as f32;
        truth.push(position);

        filter.predict();
        filter
            .state_vector()
            .inspect(|v| v.copy_to(&mut x_predicted[k]));
        filter
            .estimate_covariance()
            .inspect(|m| m.copy_to(&mut p_predicted[k]));

        let z = position + noise.sample(&mut rng);
        measurement
            .measurement_vector_mut()
            .apply(|v| v.set(0, 0, z));
        filter.correct(&mut measurement);
        filter.state_vector().inspect(|v| v.copy_to(&mut x[k]));
        filter
            .estimate_covariance()
            .inspect(|m| m.copy_to(&mut p[k]));
    }

    let filtered_mse = position_mse(&x, &truth);

    let mut smoother = RtsSmoother::new(
        Square::new([0.0; 4]),
        Square::new([0.0; 4]),
        Square::new([0.0; 4]),
        Vector::new([0.0; 2]),
    );
    let all_smoothed = filter
        .state_transition()
        .inspect(|a| smoother.smooth(a, &mut x, &mut p, &x_predicted, &p_predicted));
    let smoothed_mse = position_mse(&x, &truth);

    assert!(all_smoothed);
    assert!(
        smoothed_mse < filtered_mse,
        "smoothed {smoothed_mse} vs filtered {filtered_mse}"
    );
}
