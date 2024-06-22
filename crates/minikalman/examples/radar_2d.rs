//! EKF example for radar observations of a moving object.
//!
//! The object follows a simple constant velocity model. Only the distance to the object
//! can be observed through a nonlinear observation function.

#![forbid(unsafe_code)]

use rand_distr::{Distribution, Normal};

use minikalman::builder::extended::KalmanFilterBuilder;
use minikalman::prelude::*;

const NUM_STATES: usize = 4; // position (x, y), velocity (x, y)
const NUM_OBSERVATIONS: usize = 2; // distance and angle to object

#[allow(non_snake_case)]
fn main() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // The time step of our simulation.
    const DELTA_T: f32 = 0.1;

    // Update the filter every N steps.
    const OBSERVE_EVERY: usize = 10;

    // Define the radar position in the global frame.
    let rx = 2.0;
    let ry = 1.0;

    // Set up the initial state vector.
    filter.state_vector_mut().apply(|vec| {
        vec.set_row(0, 0.0);
        vec.set_row(1, 0.0);
        vec.set_row(2, 0.5); // Underestimate the actual velocity (1 m/s)
        vec.set_row(3, 2.0); // Overestimate the actual velocity (1 m/s)
    });

    // Set up the initial estimate covariance as an identity matrix.
    filter.estimate_covariance_mut().make_identity();

    // Set up the process noise covariance matrix as an identity matrix.
    measurement
        .measurement_noise_covariance_mut()
        .make_identity();

    // Set up the measurement noise covariance.
    measurement
        .measurement_noise_covariance_mut()
        .make_identity();

    // Simulate
    for step in 1..=100 {
        let time = step as f32 * DELTA_T;

        // Update the system transition Jacobian matrix.
        filter.state_transition_mut().apply(|mat| {
            mat.make_identity();
            mat.set_at(0, 2, DELTA_T);
            mat.set_at(1, 3, DELTA_T);
        });

        // Perform a nonlinear prediction step.
        filter.predict_nonlinear(|state, next| {
            // Simple constant velocity model.
            next[0] = state[0] + state[2] * DELTA_T;
            next[1] = state[1] + state[3] * DELTA_T;
            next[2] = state[2];
            next[3] = state[3];
        });

        if step % OBSERVE_EVERY != 0 {
            print_state(time, &filter, Stage::Prior);
        } else {
            print_state(time, &filter, Stage::PriorAboutToUpdate);

            // Prepare a measurement.
            measurement.measurement_vector_mut().apply(|vec| {
                // Noise setup.
                let mut rng = rand::thread_rng();
                let measurement_noise_pos = Normal::new(0.0, 0.5).unwrap();
                let measurement_noise_angle = Normal::new(0.0, 0.1).unwrap();

                // Perform a noisy measurement of the (simulated) position.
                let z = ((time - rx).powi(2) + (time - ry).powi(2)).sqrt();
                let noise_pos = measurement_noise_pos.sample(&mut rng);

                // Perform a noisy measurement of the (simulated) angle.
                let theta = (time - ry).atan2(time - rx);
                let noise_theta = measurement_noise_angle.sample(&mut rng);

                vec.set_row(0, z + noise_pos);
                vec.set_row(1, theta + noise_theta);
            });

            // Update the observation Jacobian.
            measurement.observation_matrix_mut().apply(|mat| {
                let x = filter.state_vector().get_row(0);
                let y = filter.state_vector().get_row(1);

                let norm_sq = (x - rx).powi(2) + (y - ry).powi(2);
                let norm = norm_sq.sqrt();
                let dx = x / norm;
                let dy = y / norm;

                mat.set_at(0, 0, dx);
                mat.set_at(0, 1, dy);
                mat.set_at(0, 2, 0.0);
                mat.set_at(0, 3, 0.0);

                mat.set_at(1, 0, -(y - ry) / norm_sq);
                mat.set_at(1, 1, (x - rx) / norm_sq);
                mat.set_at(1, 2, 0.0);
                mat.set_at(1, 3, 0.0);
            });

            // Apply nonlinear correction step.
            filter.correct_nonlinear(&mut measurement, |state, observation| {
                // Transform the state into an observation.
                let x = state.get_row(0);
                let y = state.get_row(1);
                let z = ((x - rx).powi(2) + (y - ry).powi(2)).sqrt();
                let theta = (y - ry).atan2(x - rx);
                observation.set_row(0, z);
                observation.set_row(1, theta);
            });

            print_state(time, &filter, Stage::Posterior);
        }
    }
}

enum Stage {
    Prior,
    PriorAboutToUpdate,
    Posterior,
}

fn print_state<T>(time: f32, filter: &T, state: Stage)
where
    T: ExtendedKalmanFilter<4, f32>,
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
