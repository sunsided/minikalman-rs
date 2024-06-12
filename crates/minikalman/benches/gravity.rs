use criterion::{black_box, criterion_group, criterion_main, Criterion};
use minikalman::buffer_types::*;
use minikalman::{BufferBuilder, KalmanBuilder, MeasurementBuilder};

use minikalman_traits::kalman::*;
use minikalman_traits::matrix::*;

const REAL_DISTANCE: [f32; 15] = [
    0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
    706.32, 828.94, 961.38,
];

const MEASUREMENT_ERROR: [f32; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

const NUM_STATES: usize = 3;
// const NUM_INPUTS: usize = 0;
const NUM_MEASUREMENTS: usize = 1;

#[allow(non_snake_case)]
fn criterion_benchmark(c: &mut Criterion) {
    // System buffers.
    let mut gravity_x = BufferBuilder::state_vector_x::<NUM_STATES>().new(0.0_f32);
    let mut gravity_A = BufferBuilder::system_state_transition_A::<NUM_STATES>().new(0.0_f32);
    let mut gravity_P = BufferBuilder::system_covariance_P::<NUM_STATES>().new(0.0_f32);

    // Input buffers.
    // let mut gravity_u = BufferBuilder::input_vector_u::<NUM_INPUTS>().new(0.0_f32);
    // let mut gravity_B = BufferBuilder::input_transition_B::<NUM_STATES, NUM_INPUTS>().new(0.0_f32);
    // let mut gravity_Q = BufferBuilder::input_covariance_Q::<NUM_INPUTS>().new(0.0_f32);

    // Measurement buffers.
    let mut gravity_z = BufferBuilder::measurement_vector_z::<NUM_MEASUREMENTS>().new(0.0_f32);
    let mut gravity_H =
        BufferBuilder::measurement_transformation_H::<NUM_MEASUREMENTS, NUM_STATES>().new(0.0_f32);
    let mut gravity_R = BufferBuilder::measurement_covariance_R::<NUM_MEASUREMENTS>().new(0.0_f32);
    let mut gravity_y = BufferBuilder::innovation_vector_y::<NUM_MEASUREMENTS>().new(0.0_f32);
    let mut gravity_S = BufferBuilder::innovation_covariance_S::<NUM_MEASUREMENTS>().new(0.0_f32);
    let mut gravity_K = BufferBuilder::kalman_gain_K::<NUM_STATES, NUM_MEASUREMENTS>().new(0.0_f32);

    // Filter temporaries.
    let mut gravity_temp_x = BufferBuilder::state_prediction_temp_x::<NUM_STATES>().new(0.0_f32);
    let mut gravity_temp_P = BufferBuilder::temp_system_covariance_P::<NUM_STATES>().new(0.0_f32);
    // let mut gravity_temp_BQ = BufferBuilder::temp_BQ::<NUM_STATES, NUM_INPUTS>().new(0.0_f32);

    // Measurement temporaries.
    let mut gravity_temp_S_inv = BufferBuilder::temp_S_inv::<NUM_MEASUREMENTS>().new(0.0_f32);
    let mut gravity_temp_HP = BufferBuilder::temp_HP::<NUM_MEASUREMENTS, NUM_STATES>().new(0.0_f32);
    let mut gravity_temp_PHt =
        BufferBuilder::temp_PHt::<NUM_STATES, NUM_MEASUREMENTS>().new(0.0_f32);
    let mut gravity_temp_KHP = BufferBuilder::temp_KHP::<NUM_STATES>().new(0.0_f32);

    c.bench_function("filter loop", |bencher| {
        let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
            SystemMatrixMutBuffer::from(gravity_A.as_mut()),
            StateVectorBuffer::from(gravity_x.as_mut()),
            SystemCovarianceMatrixBuffer::from(gravity_P.as_mut()),
            TemporaryStatePredictionVectorBuffer::from(gravity_temp_x.as_mut()),
            TemporaryStateMatrixBuffer::from(gravity_temp_P.as_mut()),
        );

        let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
            MeasurementObservationMatrixMutBuffer::from(gravity_H.as_mut()),
            MeasurementVectorBuffer::from(gravity_z.as_mut()),
            MeasurementProcessNoiseCovarianceMatrixBuffer::from(gravity_R.as_mut()),
            InnovationVectorBuffer::from(gravity_y.as_mut()),
            InnovationResidualCovarianceMatrixBuffer::from(gravity_S.as_mut()),
            KalmanGainMatrixBuffer::from(gravity_K.as_mut()),
            TemporaryResidualCovarianceInvertedMatrixBuffer::from(gravity_temp_S_inv.as_mut()),
            TemporaryHPMatrixBuffer::from(gravity_temp_HP.as_mut()),
            TemporaryPHTMatrixBuffer::from(gravity_temp_PHt.as_mut()),
            TemporaryKHPMatrixBuffer::from(gravity_temp_KHP.as_mut()),
        );

        // Set initial state.
        initialize_state_vector(filter.state_vector_mut());
        initialize_state_transition_matrix(filter.state_transition_mut());
        initialize_state_covariance_matrix(filter.system_covariance_mut());
        initialize_position_measurement_transformation_matrix(
            measurement.measurement_transformation_mut(),
        );
        initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

        bencher.iter(|| {
            for t in 0..REAL_DISTANCE.len() {
                filter.predict();
                measurement.measurement_vector_apply(|z| {
                    z[0] = black_box(REAL_DISTANCE[t]) + black_box(MEASUREMENT_ERROR[t])
                });
                filter.correct(black_box(&mut measurement));
            }
        })
    });

    c.bench_function("predict", |bencher| {
        let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
            SystemMatrixMutBuffer::from(gravity_A.as_mut()),
            StateVectorBuffer::from(gravity_x.as_mut()),
            SystemCovarianceMatrixBuffer::from(gravity_P.as_mut()),
            TemporaryStatePredictionVectorBuffer::from(gravity_temp_x.as_mut()),
            TemporaryStateMatrixBuffer::from(gravity_temp_P.as_mut()),
        );

        let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
            MeasurementObservationMatrixMutBuffer::from(gravity_H.as_mut()),
            MeasurementVectorBuffer::from(gravity_z.as_mut()),
            MeasurementProcessNoiseCovarianceMatrixBuffer::from(gravity_R.as_mut()),
            InnovationVectorBuffer::from(gravity_y.as_mut()),
            InnovationResidualCovarianceMatrixBuffer::from(gravity_S.as_mut()),
            KalmanGainMatrixBuffer::from(gravity_K.as_mut()),
            TemporaryResidualCovarianceInvertedMatrixBuffer::from(gravity_temp_S_inv.as_mut()),
            TemporaryHPMatrixBuffer::from(gravity_temp_HP.as_mut()),
            TemporaryPHTMatrixBuffer::from(gravity_temp_PHt.as_mut()),
            TemporaryKHPMatrixBuffer::from(gravity_temp_KHP.as_mut()),
        );

        // Set initial state.
        initialize_state_vector(filter.state_vector_mut());
        initialize_state_transition_matrix(filter.state_transition_mut());
        initialize_state_covariance_matrix(filter.system_covariance_mut());
        initialize_position_measurement_transformation_matrix(
            measurement.measurement_transformation_mut(),
        );
        initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

        bencher.iter(|| {
            filter.predict();
        })
    });

    c.bench_function("update", |bencher| {
        let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
            SystemMatrixMutBuffer::from(gravity_A.as_mut()),
            StateVectorBuffer::from(gravity_x.as_mut()),
            SystemCovarianceMatrixBuffer::from(gravity_P.as_mut()),
            TemporaryStatePredictionVectorBuffer::from(gravity_temp_x.as_mut()),
            TemporaryStateMatrixBuffer::from(gravity_temp_P.as_mut()),
        );

        let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
            MeasurementObservationMatrixMutBuffer::from(gravity_H.as_mut()),
            MeasurementVectorBuffer::from(gravity_z.as_mut()),
            MeasurementProcessNoiseCovarianceMatrixBuffer::from(gravity_R.as_mut()),
            InnovationVectorBuffer::from(gravity_y.as_mut()),
            InnovationResidualCovarianceMatrixBuffer::from(gravity_S.as_mut()),
            KalmanGainMatrixBuffer::from(gravity_K.as_mut()),
            TemporaryResidualCovarianceInvertedMatrixBuffer::from(gravity_temp_S_inv.as_mut()),
            TemporaryHPMatrixBuffer::from(gravity_temp_HP.as_mut()),
            TemporaryPHTMatrixBuffer::from(gravity_temp_PHt.as_mut()),
            TemporaryKHPMatrixBuffer::from(gravity_temp_KHP.as_mut()),
        );

        // Set initial state.
        initialize_state_vector(filter.state_vector_mut());
        initialize_state_transition_matrix(filter.state_transition_mut());
        initialize_state_covariance_matrix(filter.system_covariance_mut());
        initialize_position_measurement_transformation_matrix(
            measurement.measurement_transformation_mut(),
        );
        initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

        measurement.measurement_vector_apply(|z| {
            z[0] = black_box(REAL_DISTANCE[0]) + black_box(MEASUREMENT_ERROR[0])
        });

        bencher.iter(|| {
            filter.correct(black_box(&mut measurement));
        })
    });
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVectorMut<NUM_STATES, f32>) {
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
fn initialize_state_transition_matrix(filter: &mut impl SystemMatrixMut<NUM_STATES, f32>) {
    filter.apply(|a| {
        // Time constant.
        const T: f32 = 1 as _;

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
fn initialize_state_covariance_matrix(filter: &mut impl SystemCovarianceMatrix<NUM_STATES, f32>) {
    filter.apply(|p| {
        p.set_symmetric(0, 0, 0.1 as _); // var(s)
        p.set_symmetric(0, 1, 0 as _); // cov(s, v)
        p.set_symmetric(0, 2, 0 as _); // cov(s, g)

        p.set_symmetric(1, 1, 1 as _); // var(v)
        p.set_symmetric(1, 2, 0 as _); // cov(v, g)

        p.set_symmetric(2, 2, 1 as _); // var(g)
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
    measurement: &mut impl MeasurementObservationMatrixMut<NUM_MEASUREMENTS, NUM_STATES, f32>,
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
    measurement: &mut impl MeasurementProcessNoiseCovarianceMatrix<NUM_MEASUREMENTS, f32>,
) {
    measurement.apply(|r| {
        r.set_symmetric(0, 0, 0.5 as _); // var(s)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
