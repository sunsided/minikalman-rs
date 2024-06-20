//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

use rand_distr::{Distribution, Normal};

use minikalman::builder::KalmanFilterBuilder;
use minikalman::prelude::*;

const NUM_STATES: usize = 4; // position (x, y), velocity (x, y)
const NUM_OBSERVATIONS: usize = 1; // distance to object

#[allow(non_snake_case)]
fn main() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // The time step of our simulation.
    const DELTA_T: f32 = 0.1;

    // Update the filter every N steps.
    const OBSERVE_EVERY: usize = 10;

    // Set up the initial state vector.
    filter.state_vector_mut().apply(|vec| {
        vec.set_row(0, 0.0);
        vec.set_row(1, 0.0);
        vec.set_row(2, 1.0);
        vec.set_row(3, 1.0);
    });

    // Set up the initial estimate covariance as an identity matrix.
    filter.estimate_covariance_mut().make_identity();

    // Set up the process noise covariance matrix as an identity matrix.
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(0.1);

    // Set up the measurement noise covariance.
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_value(0.5); // matrix is 1x1
    });

    // Simulate
    for step in 1..=100 {
        let time = step as f32 * DELTA_T;

        // Update the system transition Jacobian matrix.
        filter.state_transition_mut().apply(|mat| {
            mat.make_identity();
            mat.set(0, 2, DELTA_T);
            mat.set(1, 3, DELTA_T);
        });

        // Perform a nonlinear prediction step.
        filter.predict_nonlinear(|state, next| {
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
                let measurement_noise = Normal::new(0.0, 0.5).unwrap();

                // Perform a noisy measurement of the (simulated) position.
                let z = (time.powi(2) + time.powi(2)).sqrt();
                let noise = measurement_noise.sample(&mut rng);

                vec.set_value(z + noise);
            });

            // Update the observation Jacobian.
            measurement.observation_matrix_mut().apply(|mat| {
                let x = filter.state_vector().get_row(0);
                let y = filter.state_vector().get_row(1);

                let norm = (x.powi(2) + y.powi(2)).sqrt();
                let dx = x / norm;
                let dy = y / norm;

                mat.set_col(0, dx);
                mat.set_col(1, dy);
                mat.set_col(2, 0.0);
                mat.set_col(3, 0.0);
            });

            // Apply nonlinear correction step.
            filter.correct_nonlinear(&mut measurement, |state, observation| {
                // We transform the state into an observation.
                let x = state.get_row(0);
                let y = state.get_row(1);
                let z = (x.powi(2) + y.powi(2)).sqrt();
                observation.set_value(z);
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
    T: KalmanFilter<4, f32>,
{
    let marker = match state {
        Stage::Prior => ' ',
        Stage::PriorAboutToUpdate => '-',
        Stage::Posterior => '+',
    };

    let state = filter.state_vector();

    let covariances = filter.estimate_covariance();
    let covariances = covariances.as_matrix();

    let std_x = covariances.get(0, 0).sqrt();
    let std_y = covariances.get(1, 1).sqrt();
    let std_vx = covariances.get(2, 2).sqrt();
    let std_vy = covariances.get(3, 3).sqrt();

    println!(
        "{} t={:.2} s,  x={:.2} ± {:.2} m\n              x={:.2} ± {:.2} m\n             vx={:2.2} ± {:.2} m/s\n             vy={:2.2} ± {:.2} m/s",
        marker, time, state[0], std_x, state[1], std_y, state[2], std_vx, state[3], std_vy
    );
}
