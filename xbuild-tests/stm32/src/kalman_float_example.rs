use minikalman::prelude::{buffer_types::*, *};

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
    impl_buffer_x!(static mut gravity_x, NUM_STATES, f32, 0.0);
    impl_buffer_A!(static mut gravity_A, NUM_STATES, f32, 0.0);
    impl_buffer_P!(static mut gravity_P, NUM_STATES, f32, 0.0);

    // Input buffers.
    impl_buffer_u!(static mut gravity_u, NUM_INPUTS, f32, 0.0);
    impl_buffer_B!(static mut gravity_B, NUM_STATES, NUM_INPUTS, f32, 0.0);
    impl_buffer_Q!(static mut gravity_Q, NUM_INPUTS, f32, 0.0);

    // Measurement buffers.
    impl_buffer_Q!(static mut gravity_z, NUM_MEASUREMENTS, f32, 0.0);
    impl_buffer_H!(static mut gravity_H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    impl_buffer_R!(static mut gravity_R, NUM_MEASUREMENTS, f32, 0.0);
    impl_buffer_y!(static mut gravity_y, NUM_MEASUREMENTS, f32, 0.0);
    impl_buffer_S!(static mut gravity_S, NUM_MEASUREMENTS, f32, 0.0);
    impl_buffer_K!(static mut gravity_K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);

    // Filter temporaries.
    impl_buffer_temp_x!(static mut gravity_temp_x, NUM_STATES, f32, 0.0);
    impl_buffer_temp_P!(static mut gravity_temp_P, NUM_STATES, f32, 0.0);
    impl_buffer_temp_BQ!(static mut gravity_temp_BQ, NUM_STATES, NUM_INPUTS, f32, 0.0);

    // Measurement temporaries.
    impl_buffer_temp_S_inv!(static mut gravity_temp_S_inv, NUM_MEASUREMENTS, f32, 0.0);

    // Measurement temporaries.
    impl_buffer_temp_HP!(static mut gravity_temp_HP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    impl_buffer_temp_PHt!(static mut gravity_temp_PHt, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    impl_buffer_temp_KHP!(static mut gravity_temp_KHP, NUM_STATES, f32, 0.0);

    let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
        SystemMatrixMutBuffer::from(unsafe { gravity_A.as_mut() }),
        StateVectorBuffer::from(unsafe { gravity_x.as_mut() }),
        InputMatrixMutBuffer::from(unsafe { gravity_B.as_mut() }),
        InputVectorBuffer::from(unsafe { gravity_u.as_mut() }),
        SystemCovarianceMatrixBuffer::from(unsafe { gravity_P.as_mut() }),
        InputCovarianceMatrixMutBuffer::from(unsafe { gravity_Q.as_mut() }),
        StatePredictionVectorBuffer::from(unsafe { gravity_temp_x.as_mut() }),
        TemporaryStateMatrixBuffer::from(unsafe { gravity_temp_P.as_mut() }),
        TemporaryBQMatrixBuffer::from(unsafe { gravity_temp_BQ.as_mut() }),
    );

    let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
        MeasurementTransformationMatrixMutBuffer::from(unsafe { gravity_H.as_mut() }),
        MeasurementVectorBuffer::from(unsafe { gravity_z.as_mut() }),
        MeasurementProcessNoiseCovarianceMatrixBuffer::from(unsafe { gravity_R.as_mut() }),
        InnovationVectorBuffer::from(unsafe { gravity_y.as_mut() }),
        InnovationResidualCovarianceMatrixBuffer::from(unsafe { gravity_S.as_mut() }),
        KalmanGainMatrixBuffer::from(unsafe { gravity_K.as_mut() }),
        TemporaryResidualCovarianceInvertedMatrixBuffer::from(unsafe {
            gravity_temp_S_inv.as_mut()
        }),
        TemporaryHPMatrixBuffer::from(unsafe { gravity_temp_HP.as_mut() }),
        TemporaryPHTMatrixBuffer::from(unsafe { gravity_temp_PHt.as_mut() }),
        TemporaryKHPMatrixBuffer::from(unsafe { gravity_temp_KHP.as_mut() }),
    );

    // Set initial state.
    initialize_state_vector(filter.state_vector_mut());
    initialize_state_transition_matrix(filter.state_transition_mut());
    initialize_state_covariance_matrix(filter.system_covariance_mut());
    initialize_position_measurement_transformation_matrix(
        measurement.measurement_transformation_mut(),
    );
    initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

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
