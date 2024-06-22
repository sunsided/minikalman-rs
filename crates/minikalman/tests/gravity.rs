//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

use minikalman::prelude::*;
use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};

/// Observations.
///
/// MATLAB source:
/// ```matlab
/// s = s + v*T + g*0.5*T^2;
/// v = v + g*T;
/// ```
const REAL_DISTANCE: [f32; 15] = [
    0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
    706.32, 828.94, 961.38,
];

/// Observation noise with variance 0.5
///
/// MATLAB source:
/// ```matlab
/// noise = 0.5^2*randn(15,1);
/// ```
const OBSERVATION_ERROR: [f32; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

const NUM_STATES: usize = 3;
const NUM_OBSERVATIONS: usize = 1;

#[allow(non_snake_case)]
#[test]
fn test_gravity_estimation() {
    // System buffers.
    impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);

    // Observation buffers.
    impl_buffer_z!(mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_H!(mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    impl_buffer_R!(mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_y!(mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_S!(mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);

    // Filter temporaries.
    impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);

    // Observation temporaries.
    impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);

    let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
        gravity_A,
        gravity_x,
        gravity_P,
        gravity_temp_x,
        gravity_temp_P,
    );

    let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
        gravity_H,
        gravity_z,
        gravity_R,
        gravity_y,
        gravity_S,
        gravity_K,
        gravity_temp_S_inv,
        gravity_temp_HP,
        gravity_temp_PHt,
        gravity_temp_KHP,
    );

    // Set initial state.
    initialize_state_vector(filter.state_vector_mut());
    initialize_state_transition_matrix(filter.state_transition_mut());
    initialize_state_covariance_matrix(filter.estimate_covariance_mut());
    initialize_position_measurement_transformation_matrix(measurement.observation_matrix_mut());
    initialize_position_measurement_process_noise_matrix(
        measurement.measurement_noise_covariance_mut(),
    );

    // Filter!
    for t in 0..REAL_DISTANCE.len() {
        // Prediction.
        filter.predict();

        // Measure ...
        let m = REAL_DISTANCE[t] + OBSERVATION_ERROR[t];
        measurement.measurement_vector_mut().apply(|z| z[0] = m);

        // Update.
        filter.correct(&mut measurement);
    }

    // Fetch estimated gravity constant.
    let gravity_x = filter.state_vector();
    let g_estimated = gravity_x[2];
    assert!(g_estimated > 9.0 && g_estimated < 10.0);
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVectorMut<NUM_STATES, f32>) {
    filter.as_matrix_mut().apply(|state| {
        state[0] = 0 as _; // position
        state[1] = 0 as _; // velocity
        state[2] = 6 as _; // acceleration
    });
}

/// Initializes the state transition matrix.
///
/// This sets up the differential equations for
/// ```math
/// s₁ = 1×s₀ + T×v₀ + 0.5×T²×a₀
/// v₁ = 1×v₀ + T×a₀
/// a₁ = 1×a₀
/// ```
fn initialize_state_transition_matrix(filter: &mut impl StateTransitionMatrixMut<NUM_STATES, f32>) {
    filter.as_matrix_mut().apply(|a| {
        // Time constant.
        const T: f32 = 1 as _;

        // Transition of x to s.
        a.set_at(0, 0, 1 as _); // 1
        a.set_at(0, 1, T as _); // T
        a.set_at(0, 2, 0.5 * T * T); // 0.5 × T²

        // Transition of x to v.
        a.set_at(1, 0, 0 as _); // 0
        a.set_at(1, 1, 1 as _); // 1
        a.set_at(1, 2, T as _); // T

        // Transition of b to g.
        a.set_at(2, 0, 0 as _); // 0
        a.set_at(2, 1, 0 as _); // 0
        a.set_at(2, 2, 1 as _); // 1
    });
}

/// Initializes the system covariance matrix.
///
/// This defines how different states (linearly) influence each other
/// over time. In this setup we claim that position, velocity and acceleration
/// are linearly independent.
fn initialize_state_covariance_matrix(filter: &mut impl EstimateCovarianceMatrix<NUM_STATES, f32>) {
    filter.as_matrix_mut().apply(|p| {
        p.set_at(0, 0, 0.1 as _); // var(s)
        p.set_at(0, 1, 0 as _); // cov(s, v)
        p.set_at(0, 2, 0 as _); // cov(s, g)

        p.set_at(1, 1, 1 as _); // var(v)
        p.set_at(1, 2, 0 as _); // cov(v, g)

        p.set_at(2, 2, 1 as _); // var(g)
    });
}

/// Initializes the measurement transformation matrix.
///
/// This matrix describes how a single measurement is obtained from the
/// state vector. In our case, we directly observe one of the states, namely position.
/// ```math
/// z = 1×s + 0×v + 0×a
/// ```
fn initialize_position_measurement_transformation_matrix(
    measurement: &mut impl ObservationMatrixMut<NUM_OBSERVATIONS, NUM_STATES, f32>,
) {
    measurement.as_matrix_mut().apply(|h| {
        h.set_at(0, 0, 1 as _); // z = 1*s
        h.set_at(0, 1, 0 as _); //   + 0*v
        h.set_at(0, 2, 0 as _); //   + 0*g
    });
}

/// Initializes the measurement noise / uncertainty matrix.
///
/// This matrix describes the measurement covariances as well as the
/// individual variation components. It is the measurement counterpart
/// of the state covariance matrix.
fn initialize_position_measurement_process_noise_matrix(
    measurement: &mut impl MeasurementNoiseCovarianceMatrix<NUM_OBSERVATIONS, f32>,
) {
    measurement.as_matrix_mut().apply(|r| {
        r.set_at(0, 0, 0.5 as _); // var(s)
    });
}
