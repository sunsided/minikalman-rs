use core::ptr::addr_of_mut;
use minikalman::buffer_types::{
    InnovationResidualCovarianceMatrixBuffer, InnovationVectorBuffer,
    InputCovarianceMatrixMutBuffer, InputMatrixBuffer, InputMatrixMutBuffer, InputVectorBuffer,
    KalmanGainMatrixBuffer, MeasurementProcessNoiseCovarianceMatrixBuffer,
    MeasurementTransformationMatrixMutBuffer, MeasurementVectorBuffer, StatePredictionVectorBuffer,
    StateVectorBuffer, SystemCovarianceMatrixBuffer, SystemMatrixMutBuffer,
    TemporaryBQMatrixBuffer, TemporaryHPMatrixBuffer, TemporaryKHPMatrixBuffer,
    TemporaryPHTMatrixBuffer, TemporaryResidualCovarianceInvertedMatrixBuffer,
    TemporaryStateMatrixBuffer,
};
use minikalman::{prelude::*, Kalman, MatrixDataOwned, Measurement};

/// Measurements.
const NUM_STATES: usize = 3;
const NUM_INPUTS: usize = 0;
const NUM_MEASUREMENTS: usize = 1;

/// Measurements.
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

/// Measurement noise with variance 0.5
///
/// MATLAB source:
/// ```matlab
/// noise = 0.5^2*randn(15,1);
/// ```
const MEASUREMENT_ERROR: [f32; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
pub fn predict_gravity() -> f32 {
    // System buffers.
    impl_static_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    impl_static_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    impl_static_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);

    // Input buffers.
    impl_static_buffer_u!(mut gravity_u, NUM_INPUTS, f32, 0.0);
    impl_static_buffer_B!(mut gravity_B, NUM_STATES, NUM_INPUTS, f32, 0.0);
    impl_static_buffer_Q!(mut gravity_Q, NUM_INPUTS, f32, 0.0);

    // Measurement buffers.
    impl_static_buffer_Q!(mut gravity_z, NUM_MEASUREMENTS, f32, 0.0);
    impl_static_buffer_H!(mut gravity_H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    impl_static_buffer_R!(mut gravity_R, NUM_MEASUREMENTS, f32, 0.0);
    impl_static_buffer_y!(mut gravity_y, NUM_MEASUREMENTS, f32, 0.0);
    impl_static_buffer_S!(mut gravity_S, NUM_MEASUREMENTS, f32, 0.0);
    impl_static_buffer_K!(mut gravity_K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);

    // Filter temporaries.
    impl_static_buffer_temp_x!(gravity_temp_x, NUM_STATES, f32, 0.0);
    impl_static_buffer_temp_P!(gravity_temp_P, NUM_STATES, f32, 0.0);
    impl_static_buffer_temp_BQ!(gravity_temp_BQ, NUM_STATES, NUM_INPUTS, f32, 0.0);

    // Measurement temporaries.
    impl_static_buffer_temp_S_inv!(gravity_temp_S_inv, NUM_MEASUREMENTS, f32, 0.0);

    // Measurement temporaries.
    impl_static_buffer_temp_HP!(gravity_temp_HP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    impl_static_buffer_temp_PHt!(gravity_temp_PHt, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    impl_static_buffer_temp_KHP!(gravity_temp_KHP, NUM_STATES, f32, 0.0);

    let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
        unsafe { &mut *addr_of_mut!(gravity_A) },
        unsafe { &mut *addr_of_mut!(gravity_x) },
        unsafe { &mut *addr_of_mut!(gravity_B) },
        unsafe { &mut *addr_of_mut!(gravity_u) },
        unsafe { &mut *addr_of_mut!(gravity_P) },
        unsafe { &mut *addr_of_mut!(gravity_Q) },
        unsafe { &mut *addr_of_mut!(gravity_temp_x) },
        unsafe { &mut *addr_of_mut!(gravity_temp_P) },
        unsafe { &mut *addr_of_mut!(gravity_temp_BQ) },
    );

    let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
        unsafe { &mut *addr_of_mut!(gravity_H) },
        unsafe { &mut *addr_of_mut!(gravity_z) },
        unsafe { &mut *addr_of_mut!(gravity_R) },
        unsafe { &mut *addr_of_mut!(gravity_y) },
        unsafe { &mut *addr_of_mut!(gravity_S) },
        unsafe { &mut *addr_of_mut!(gravity_K) },
        unsafe { &mut *addr_of_mut!(gravity_temp_S_inv) },
        unsafe { &mut *addr_of_mut!(gravity_temp_HP) },
        unsafe { &mut *addr_of_mut!(gravity_temp_PHt) },
        unsafe { &mut *addr_of_mut!(gravity_temp_KHP) },
    );

    // Set initial state.
    initialize_state_vector(&mut filter);
    initialize_state_transition_matrix(&mut filter);
    initialize_state_covariance_matrix(&mut filter);
    initialize_position_measurement_transformation_matrix(&mut measurement);
    initialize_position_measurement_process_noise_matrix(&mut measurement);

    // Filter!
    for t in 0..REAL_DISTANCE.len() {
        // Prediction.
        filter.predict();

        // Measure ...
        let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
        measurement.measurement_vector_apply(|z| z[0] = m);

        // Update.
        filter.correct(&mut measurement);
    }

    // Fetch estimated gravity constant.
    unsafe { gravity_x[2] }
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVector<NUM_STATES, f32>) {
    filter.apply(|state| {
        state[0] = 0.0; // position
        state[1] = 0.0; // velocity
        state[2] = 6.0; // acceleration
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
fn initialize_state_transition_matrix(filter: &mut impl SystemMatrixMut<NUM_STATES, f32>) {
    filter.apply(|a| {
        // Time constant.
        const T: f32 = 1.0;

        // Transition of x to s.
        a.set(0, 0, 1.0); // 1
        a.set(0, 1, T as _); // T
        a.set(0, 2, 0.5 * T * T); // 0.5 × T²

        // Transition of x to v.
        a.set(1, 0, 0.0); // 0
        a.set(1, 1, 1.0); // 1
        a.set(1, 2, T as _); // T

        // Transition of b to g.
        a.set(2, 0, 0.0); // 0
        a.set(2, 1, 0.0); // 0
        a.set(2, 2, 1.0); // 1
    });
}

/// Initializes the system covariance matrix.
///
/// This defines how different states (linearly) influence each other
/// over time. In this setup we claim that position, velocity and acceleration
/// are linearly independent.
fn initialize_state_covariance_matrix(filter: &mut impl SystemCovarianceMatrix<NUM_STATES, f32>) {
    filter.apply(|p| {
        p.set(0, 0, 0.1); // var(s)
        p.set(0, 1, 0.0); // cov(s, v)
        p.set(0, 2, 0.0); // cov(s, g)

        p.set(1, 1, 1.0); // var(v)
        p.set(1, 2, 0.0); // cov(v, g)

        p.set(2, 2, 1.0); // var(g)
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
    measurement: &mut impl MeasurementTransformationMatrixMut<NUM_MEASUREMENTS, NUM_STATES, f32>,
) {
    measurement.apply(|h| {
        h.set(0, 0, 1.0); // z = 1*s
        h.set(0, 1, 0.0); //   + 0*v
        h.set(0, 2, 0.0); //   + 0*g
    });
}

/// Initializes the measurement noise / uncertainty matrix.
///
/// This matrix describes the measurement covariances as well as the
/// individual variation components. It is the measurement counterpart
/// of the state covariance matrix.
fn initialize_position_measurement_process_noise_matrix(
    measurement: &mut impl MeasurementProcessNoiseCovarianceMatrix<NUM_MEASUREMENTS, f32>,
) {
    measurement.apply(|r| {
        r.set(0, 0, 0.5); // var(s)
    });
}
