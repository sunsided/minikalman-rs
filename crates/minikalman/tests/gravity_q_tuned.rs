//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

use minikalman::prelude::*;

/// Measurements.
///
/// MATLAB source:
/// ```matlab
/// s = s + v*T + g*0.5*T^2;
/// v = v + g*T;
/// ```
const REAL_DISTANCE: [f64; 15] = [
    0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
    706.32, 828.94, 961.38,
];

/// Measurement noise with variance 0.5
///
/// MATLAB source:
/// ```matlab
/// noise = 0.5^2*randn(15,1);
/// ```
const MEASUREMENT_ERROR: [f64; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

const NUM_STATES: usize = 3;
const NUM_INPUTS: usize = 0;
const NUM_MEASUREMENTS: usize = 1;

#[allow(non_snake_case)]
#[test]
fn test_gravity_estimation_tuned() {
    // System buffers.
    let mut gravity_x = create_buffer_x!(NUM_STATES, f64);
    let mut gravity_A = create_buffer_A!(NUM_STATES, f64);
    let mut gravity_P = create_buffer_P!(NUM_STATES, f64);

    // Input buffers.
    let mut gravity_u = create_buffer_u!(NUM_INPUTS, f64);
    let mut gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS, f64);
    let mut gravity_Q = create_buffer_Q!(NUM_INPUTS, f64);

    // Measurement buffers.
    let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS, f64);
    let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES, f64);
    let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS, f64);
    let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS, f64);
    let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS, f64);
    let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS, f64);

    // Filter temporaries.
    let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES, f64);
    let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES, f64);
    let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS, f64);

    // Measurement temporaries.
    let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS, f64);
    let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES, f64);
    let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS, f64);
    let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES, f64);

    let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f64>(
        SystemMatrixMutBuffer::new(gravity_A),
        StateVectorBuffer::new(gravity_x),
        InputMatrixMutBuffer::new(gravity_B),
        InputVectorBuffer::new(gravity_u),
        SystemCovarianceMatrixBuffer::new(gravity_P),
        InputCovarianceMatrixBuffer::new(gravity_Q),
        StatePredictionVectorBuffer::new(gravity_temp_x),
        TemporaryStateMatrixBuffer::new(gravity_temp_P),
        TemporaryBQMatrixBuffer::new(gravity_temp_BQ),
    );

    let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f64>(
        MeasurementTransformationMatrixMutBuffer::new(gravity_H),
        MeasurementVectorBuffer::new(gravity_z),
        MeasurementProcessNoiseCovarianceMatrixBuffer::new(gravity_R),
        InnovationVectorBuffer::new(gravity_y),
        InnovationResidualCovarianceMatrixBuffer::new(gravity_S),
        KalmanGainMatrixBuffer::new(gravity_K),
        TemporaryResidualCovarianceInvertedMatrixBuffer::new(gravity_temp_S_inv),
        TemporaryHPMatrixBuffer::new(gravity_temp_HP),
        TemporaryPHTMatrixBuffer::new(gravity_temp_PHt),
        TemporaryKHPMatrixBuffer::new(gravity_temp_KHP),
    );

    // Set initial state.
    initialize_state_vector(filter.state_vector_mut());
    initialize_state_transition_matrix(filter.state_transition_mut());
    initialize_state_covariance_matrix(filter.system_covariance_mut());
    initialize_position_measurement_transformation_matrix(
        measurement.measurement_transformation_mut(),
    );
    initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

    const LAMBDA: f64 = 0.5;

    // Filter!
    for t in 0..REAL_DISTANCE.len() {
        // Prediction.
        filter.predict_tuned(LAMBDA);

        // Measure ...
        let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
        measurement.measurement_vector_apply(|z| z[0] = m);

        // Update.
        filter.correct(&mut measurement);
    }

    // Fetch estimated gravity constant.
    let gravity_x = filter.state_vector_ref();
    let g_estimated = gravity_x[2];
    assert!(g_estimated > 9.0 && g_estimated < 10.0);
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVector<NUM_STATES, f64>) {
    filter.apply(|state| {
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
fn initialize_state_transition_matrix(filter: &mut impl SystemMatrixMut<NUM_STATES, f64>) {
    filter.apply(|a| {
        // Time constant.
        const T: f64 = 1 as _;

        // Transition of x to s.
        a.set(0, 0, 1 as _); // 1
        a.set(0, 1, T as _); // T
        a.set(0, 2, 0.5 * T * T); // 0.5 × T²

        // Transition of x to v.
        a.set(1, 0, 0 as _); // 0
        a.set(1, 1, 1 as _); // 1
        a.set(1, 2, T as _); // T

        // Transition of b to g.
        a.set(2, 0, 0 as _); // 0
        a.set(2, 1, 0 as _); // 0
        a.set(2, 2, 1 as _); // 1
    });
}

/// Initializes the system covariance matrix.
///
/// This defines how different states (linearly) influence each other
/// over time. In this setup we claim that position, velocity and acceleration
/// are linearly independent.
fn initialize_state_covariance_matrix(filter: &mut impl SystemCovarianceMatrix<NUM_STATES, f64>) {
    filter.apply(|p| {
        p.set(0, 0, 0.1 as _); // var(s)
        p.set(0, 1, 0 as _); // cov(s, v)
        p.set(0, 2, 0 as _); // cov(s, g)

        p.set(1, 1, 1 as _); // var(v)
        p.set(1, 2, 0 as _); // cov(v, g)

        p.set(2, 2, 1 as _); // var(g)
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
    measurement: &mut impl MeasurementTransformationMatrixMut<NUM_MEASUREMENTS, NUM_STATES, f64>,
) {
    measurement.apply(|h| {
        h.set(0, 0, 1 as _); // z = 1*s
        h.set(0, 1, 0 as _); //   + 0*v
        h.set(0, 2, 0 as _); //   + 0*g
    });
}

/// Initializes the measurement noise / uncertainty matrix.
///
/// This matrix describes the measurement covariances as well as the
/// individual variation components. It is the measurement counterpart
/// of the state covariance matrix.
fn initialize_position_measurement_process_noise_matrix(
    measurement: &mut impl MeasurementProcessNoiseCovarianceMatrix<NUM_MEASUREMENTS, f64>,
) {
    measurement.apply(|r| {
        r.set(0, 0, 0.5 as _); // var(s)
    });
}
