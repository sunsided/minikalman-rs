//! UKF example using the builder API (requires `alloc`/`std`).
//!
//! This is the same radar observation scenario as `radar_2d_ukf.rs`, but using
//! the builder pattern instead of `impl_buffer!` macros for buffer construction.
//!
//! Run with: `cargo run --example radar-2d-ukf-builder --features std`

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use rand_distr::{Distribution, Normal};

use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder;

const NUM_STATES: usize = 4; // position (x, y), velocity (x, y)
const NUM_SIGMA: usize = 2 * NUM_STATES + 1; // 9 sigma points
const NUM_OBSERVATIONS: usize = 3; // distance, angle to object and object velocity
const DELTA_T: f32 = 0.1;
const OBSERVE_EVERY: usize = 10;
const RX: f32 = 2.0;
const RY: f32 = 1.0;

fn main() {
    // Build the UKF filter and observation using the builder API.
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Set up the initial state vector.
    filter.state_vector_mut().apply(|vec| {
        vec.set_row(0, 0.0);
        vec.set_row(1, 0.0);
        vec.set_row(2, 0.5);
        vec.set_row(3, 2.0);
    });

    // Set up the initial estimate covariance as an identity matrix.
    filter.estimate_covariance_mut().make_identity();

    // Set up the process noise covariance matrix.
    filter.direct_process_noise_mut().make_scalar(0.1);

    // Set up the measurement noise covariance.
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.5);
        mat.set_at(1, 1, 0.1);
        mat.set_at(2, 2, 0.2);
    });

    // Simulate
    for step in 1..=100 {
        let time = step as f32 * DELTA_T;

        // Perform a nonlinear prediction step using sigma points.
        filter.predict_nonlinear(|next| {
            let px = next.as_matrix().get_at(0, 0);
            let py = next.as_matrix().get_at(1, 0);
            let vx = next.as_matrix().get_at(2, 0);
            let vy = next.as_matrix().get_at(3, 0);

            next.as_matrix_mut().set_at(0, 0, px + vx * DELTA_T);
            next.as_matrix_mut().set_at(1, 0, py + vy * DELTA_T);
            next.as_matrix_mut().set_at(2, 0, vx);
            next.as_matrix_mut().set_at(3, 0, vy);
        });

        if step % OBSERVE_EVERY != 0 {
            print_state(time, &filter, Stage::Prior);
        } else {
            print_state(time, &filter, Stage::PriorAboutToUpdate);

            // Prepare a measurement.
            measurement.measurement_vector_mut().apply(|vec| {
                let mut rng = rand::thread_rng();
                let measurement_noise_pos = Normal::new(0.0, 0.5).unwrap();
                let measurement_noise_angle = Normal::new(0.0, 0.1).unwrap();
                let measurement_noise_vel = Normal::new(0.0, 0.2).unwrap();

                let dist_norm = ((time - RX).powi(2) + (time - RY).powi(2)).sqrt();

                let z = dist_norm;
                let noise_pos = measurement_noise_pos.sample(&mut rng);

                let theta = (time - RY).atan2(time - RX);
                let noise_theta = measurement_noise_angle.sample(&mut rng);

                let v = ((time - RX) + (time - RY)) / dist_norm;
                let noise_v = measurement_noise_vel.sample(&mut rng);

                vec.set_row(0, z + noise_pos);
                vec.set_row(1, theta + noise_theta);
                vec.set_row(2, v + noise_v);
            });

            // Apply nonlinear correction step using sigma points.
            filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
                for j in 0..NUM_SIGMA {
                    let px = sigma_pred[j];
                    let py = sigma_pred[NUM_SIGMA + j];
                    let vx = sigma_pred[2 * NUM_SIGMA + j];
                    let vy = sigma_pred[3 * NUM_SIGMA + j];

                    let dx = px - RX;
                    let dy = py - RY;
                    let norm = (dx * dx + dy * dy).sqrt();
                    observed[j] = norm;
                    observed[NUM_SIGMA + j] = dy.atan2(dx);
                    observed[2 * NUM_SIGMA + j] = (dx * vx + dy * vy) / norm;
                }
            });

            print_state(time, &filter, Stage::Posterior);
        }
    }

    enum Stage {
        Prior,
        PriorAboutToUpdate,
        Posterior,
    }

    fn print_state<T>(time: f32, filter: &T, state: Stage)
    where
        T: UnscentedKalmanFilter<NUM_STATES, NUM_SIGMA, f32>,
    {
        let marker = match state {
            Stage::Prior => ' ',
            Stage::PriorAboutToUpdate => '-',
            Stage::Posterior => '+',
        };

        let state = filter.state_vector();

        let covariances = filter.estimate_covariance();
        let covariances = covariances.as_matrix();

        let std_x = covariances.get_at(0, 0).sqrt();
        let std_y = covariances.get_at(1, 1).sqrt();
        let std_vx = covariances.get_at(2, 2).sqrt();
        let std_vy = covariances.get_at(3, 3).sqrt();

        println!(
        "{} t={:.2} s,  x={:.2} ± {:.4} m\n              y={:.2} ± {:.4} m\n             vx={:2.2} ± {:.4} m/s\n             vy={:2.2} ± {:.4} m/s",
        marker, time, state[0], std_x, state[1], std_y, state[2], std_vx, state[3], std_vy
    );
    }
}
