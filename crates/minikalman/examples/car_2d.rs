//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

use assert_float_eq::*;

use minikalman::builder::KalmanFilterBuilder;
use minikalman::prelude::*;

const NUM_STATES: usize = 3; // position, velocity, acceleration
const NUM_CONTROLS: usize = 1; // acceleration
const NUM_OBSERVATIONS: usize = 1; // position

#[allow(non_snake_case)]
fn main() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut control = builder.controls().build::<NUM_CONTROLS>();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // The time step of our simulation.
    const DELTA_T: f32 = 1.0;

    // Set up the initial state vector.
    filter.state_vector_apply(|state| {
        state[0] = -5.0; // the car is at -5 m
        state[1] = 0.0; // with no velocity
        state[2] = 0.3; // and an acceleration of 0.3 m/s²
    });

    // Set up the state transition matrix.
    filter.state_transition_apply(|mat| {
        // p = p + v×∆t + a×0.5×∆t²
        mat.set(0, 0, 1.0);
        mat.set(0, 1, DELTA_T);
        mat.set(0, 2, 0.5 * DELTA_T * DELTA_T);

        // v = v + a×∆t
        mat.set(1, 0, 0.0);
        mat.set(1, 1, 1.0);
        mat.set(1, 2, DELTA_T);

        // a = a
        mat.set(2, 0, 0.0);
        mat.set(2, 1, 0.0);
        mat.set(2, 2, 1.0);
    });

    // Set up the initial estimate covariance as an identity matrix.
    filter.estimate_covariance_apply(|mat| mat.make_identity());

    // Set up the control matrix.
    control.control_matrix_apply(|mat| {
        mat.set(0, 0, 0.0);
        mat.set(1, 0, 0.0);
        mat.set(2, 0, DELTA_T); // affect acceleration directly
    });

    // Control vector is empty.
    control.control_vector_mut().set_zero();

    // Process noise covariance is identity.
    control.process_noise_covariance_mut().make_identity();

    // Set up the observation matrix.
    measurement.observation_matrix_apply(|mat| {
        mat.set(0, 0, 1.0); // only the first element is set.
    });

    // Set up the process noise covariance matrix as an identity matrix.
    measurement.measurement_noise_covariance_apply(|mat| {
        mat.make_scalar(0.1);
    });

    // Set up the measurement noise covariance.
    measurement.measurement_noise_covariance_apply(|mat| {
        mat.set(0, 0, 1.0); // matrix is 1x1
    });

    println!("Simulate without inputs and observations.");

    // Accelerate the car for 5 seconds.
    for t in 0..10 {
        filter.predict();

        let state = filter.state_vector_ref();
        println!(
            "  t={t:3} s, p={:.2} m, v={:.2} m/s, a={:.2} m/s²",
            state[0], state[1], state[2]
        );
    }

    // The car should now be at approximately 10 m, at 3.3 m/s with an unchanged acceleration.
    {
        let state = filter.state_vector_ref();
        assert_f32_near!(state[0], 10.0);
        assert_f32_near!(state[1], 3.0);
        assert_f32_near!(state[2], 0.3);
    }

    println!("Decelerate and begin to incorporate measurements");

    // Noisy observations.
    const OBSERVATIONS: [f32; 10] = [13.2, 16.5, 20.0, 23.5, 26.8, 29.7, 32.8, 34.3, 35.6, 36.00];

    // The car now begins to brake.
    let ACCELERATION: f32 = -0.1333333; // m/s²
    for t in 10..20 {
        control.control_vector_apply(|vec| vec.set(0, 0, ACCELERATION));

        filter.predict();
        filter.control(&mut control);

        if t % 2 != 0 {
            print_state(t, &filter, State::Prior);
        } else {
            print_state(t, &filter, State::PriorAboutToUpdate);

            measurement.measurement_vector_apply(|measurement| {
                measurement[0] = OBSERVATIONS[t - 10];
            });

            filter.correct(&mut measurement);
            print_state(t, &filter, State::Posterior);
        }
    }

    // The car should now be approximately stopped (but still decelerating).
    {
        let state = filter.state_vector_ref();
        assert!(is_between(state[0], 36.0, 36.3));
        assert!(is_between(state[1], 0.02, 0.04));
        assert!(is_between(state[2], -1.04, -1.03));
    }
}

enum State {
    Prior,
    PriorAboutToUpdate,
    Posterior,
}

fn print_state<T>(time: usize, filter: &T, state: State)
where
    T: KalmanFilter<3, f32>,
{
    let marker = match state {
        State::Prior => ' ',
        State::PriorAboutToUpdate => ' ',
        State::Posterior => '✅',
    };

    let state = filter.state_vector_ref();

    let covariances = filter.estimate_covariance_ref();
    let covariances = covariances.as_matrix();

    let std_p = covariances.get(0, 0).sqrt();
    let std_v = covariances.get(1, 1).sqrt();
    let std_a = covariances.get(2, 2).sqrt();

    println!(
        "{}  t={:3} s, p={:.2} ± {:.2} m\n            v={:2.2} ± {:.2} m/s\n            a={:2.2} ± {:.2} m/s²",
        marker, time, state[0], std_p, state[1], std_v, state[2], std_a
    );
}

fn is_between(value: f32, low: f32, high: f32) -> bool {
    low <= value && value <= high
}
