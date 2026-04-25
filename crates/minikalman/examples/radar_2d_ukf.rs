//! UKF example for radar observations of a moving object.
//!
//! This is a reimplementation of the `radar-2d` EKF example using the Unscented Kalman Filter.
//! Unlike the EKF, the UKF does not require computing Jacobian matrices - it propagates
//! sigma points through the nonlinear functions directly.
//!
//! Run with: `cargo run --example radar-2d-ukf --features std`
//!
//! Compare with the EKF version: `cargo run --example radar-2d --features std`
//! Both should produce approximately the same results.

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use rand_distr::{Distribution, Normal};

use minikalman::prelude::*;
use minikalman::unscented::{UnscentedKalman, UnscentedObservation};

const NUM_STATES: usize = 4; // position (x, y), velocity (x, y)
const NUM_SIGMA: usize = 2 * NUM_STATES + 1; // 9 sigma points
const NUM_OBSERVATIONS: usize = 3; // distance, angle to object and object velocity
const DELTA_T: f32 = 0.1;
const OBSERVE_EVERY: usize = 10;
const RX: f32 = 2.0;
const RY: f32 = 1.0;

fn main() {
    impl_buffer_x!(mut state_vec, NUM_STATES, f32, 0.0);
    impl_buffer_P!(mut cov_P, NUM_STATES, f32, 0.0);
    impl_buffer_Q_direct!(mut noise_Q, NUM_STATES, f32, 0.0);
    impl_buffer_temp_x!(mut predicted_x, NUM_STATES, f32, 0.0);

    impl_buffer_sigma_points!(mut sigma_pts, NUM_STATES, NUM_SIGMA, f32, 0.0);
    impl_buffer_sigma_weights!(mut sigma_w, NUM_SIGMA, f32, 0.0);
    impl_buffer_sigma_predicted!(mut sigma_pred, NUM_STATES, NUM_SIGMA, f32, 0.0);
    impl_buffer_temp_sigma_P!(mut sigma_temp_P, NUM_STATES, f32, 0.0);

    impl_buffer_z!(mut meas_z, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_R!(mut meas_R, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_y!(mut innovation_y, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_S!(mut innov_cov_S, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_K!(mut kalman_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_temp_S_inv!(mut s_inv_temp, NUM_OBSERVATIONS, f32, 0.0);

    impl_buffer_sigma_observed!(mut sigma_obs, NUM_OBSERVATIONS, NUM_SIGMA, f32, 0.0);
    impl_buffer_cross_covariance!(mut cross_cov, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_temp_sigma_P!(mut ukf_temp_P, NUM_STATES, f32, 0.0);

    // Build the UKF filter.
    let mut filter = UnscentedKalman::new(
        state_vec,
        cov_P,
        noise_Q,
        predicted_x,
        sigma_pts,
        sigma_w,
        sigma_pred,
        sigma_temp_P,
        1.0, // alpha: spread parameter
        2.0, // beta: distribution knowledge (2.0 for Gaussian)
        1.0, // kappa: secondary scaling
    );

    // Build the observation.
    let mut measurement = UnscentedObservation::new(
        meas_z,
        meas_R,
        innovation_y,
        innov_cov_S,
        kalman_K,
        s_inv_temp,
        sigma_obs,
        cross_cov,
        ukf_temp_P,
    );

    // Set up the initial state vector.
    filter.state_vector_mut().apply(|vec| {
        vec.set_row(0, 0.0);
        vec.set_row(1, 0.0);
        vec.set_row(2, 0.5); // Underestimate the actual velocity (1 m/s)
        vec.set_row(3, 2.0); // Overestimate the actual velocity (1 m/s)
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
            // No Jacobian computation needed!
            // sigma_pred is row-major: sigma_pred[row * NUM_SIGMA + col]
            filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
                for j in 0..NUM_SIGMA {
                    let px = sigma_pred[0 * NUM_SIGMA + j];
                    let py = sigma_pred[1 * NUM_SIGMA + j];
                    let vx = sigma_pred[2 * NUM_SIGMA + j];
                    let vy = sigma_pred[3 * NUM_SIGMA + j];

                    let dx = px - RX;
                    let dy = py - RY;
                    let norm = (dx * dx + dy * dy).sqrt();
                    observed[0 * NUM_SIGMA + j] = norm;
                    observed[1 * NUM_SIGMA + j] = dy.atan2(dx);
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

fn print_correction_debug<T>(filter: &T)
where
    T: UnscentedKalmanFilter<NUM_STATES, NUM_SIGMA, f32>,
{
    let x = filter.state_vector();
    let P = filter.estimate_covariance();
    let P = P.as_matrix();
    eprintln!(
        "  After correction: x=[{:.4}, {:.4}, {:.4}, {:.4}]",
        x[0], x[1], x[2], x[3]
    );
    eprintln!(
        "  P diag: [{:.4}, {:.4}, {:.4}, {:.4}]",
        P.get_at(0, 0),
        P.get_at(1, 1),
        P.get_at(2, 2),
        P.get_at(3, 3)
    );
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
