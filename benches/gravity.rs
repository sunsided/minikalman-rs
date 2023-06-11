use criterion::{black_box, criterion_group, criterion_main, Criterion};

use minikalman::{
    create_buffer_A, create_buffer_B, create_buffer_H, create_buffer_K, create_buffer_P,
    create_buffer_Q, create_buffer_R, create_buffer_S, create_buffer_temp_BQ,
    create_buffer_temp_HP, create_buffer_temp_KHP, create_buffer_temp_P, create_buffer_temp_PHt,
    create_buffer_temp_S_inv, create_buffer_temp_x, create_buffer_u, create_buffer_x,
    create_buffer_y, create_buffer_z, matrix_data_t, Kalman, Measurement,
};
use stdint::uint_fast8_t;

const REAL_DISTANCE: [matrix_data_t; 15] = [
    0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
    706.32, 828.94, 961.38,
];

const MEASUREMENT_ERROR: [matrix_data_t; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

#[allow(non_snake_case)]
fn criterion_benchmark(c: &mut Criterion) {
    const NUM_STATES: uint_fast8_t = 3;
    const NUM_INPUTS: uint_fast8_t = 0;
    const NUM_MEASUREMENTS: uint_fast8_t = 1;

    // System buffers.
    let mut gravity_x = create_buffer_x!(NUM_STATES);
    let mut gravity_A = create_buffer_A!(NUM_STATES);
    let mut gravity_P = create_buffer_P!(NUM_STATES);

    // Input buffers.
    let mut gravity_u = create_buffer_u!(0);
    let mut gravity_B = create_buffer_B!(0, 0);
    let mut gravity_Q = create_buffer_Q!(0);

    // Measurement buffers.
    let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);

    // Filter temporaries.
    let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);

    // Measurement temporaries.
    let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);

    c.bench_function("filter loop", |bencher| {
        let mut filter = Kalman::new_from_buffers(
            NUM_STATES,
            NUM_INPUTS,
            &mut gravity_A,
            &mut gravity_x,
            &mut gravity_B,
            &mut gravity_u,
            &mut gravity_P,
            &mut gravity_Q,
            &mut gravity_temp_x,
            &mut gravity_temp_P,
            &mut gravity_temp_BQ,
        );

        let mut measurement = Measurement::new(
            NUM_STATES,
            NUM_MEASUREMENTS,
            &mut gravity_H,
            &mut gravity_z,
            &mut gravity_R,
            &mut gravity_y,
            &mut gravity_S,
            &mut gravity_K,
            &mut gravity_temp_S_inv,
            &mut gravity_temp_HP,
            &mut gravity_temp_PHt,
            &mut gravity_temp_KHP,
        );

        // Set initial state.
        initialize_state_vector(&mut filter);
        initialize_state_transition_matrix(&mut filter);
        initialize_state_covariance_matrix(&mut filter);
        initialize_position_measurement_transformation_matrix(&mut measurement);
        initialize_position_measurement_process_noise_matrix(&mut measurement);

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
        let mut filter = Kalman::new_from_buffers(
            NUM_STATES,
            NUM_INPUTS,
            &mut gravity_A,
            &mut gravity_x,
            &mut gravity_B,
            &mut gravity_u,
            &mut gravity_P,
            &mut gravity_Q,
            &mut gravity_temp_x,
            &mut gravity_temp_P,
            &mut gravity_temp_BQ,
        );

        let mut measurement = Measurement::new(
            NUM_STATES,
            NUM_MEASUREMENTS,
            &mut gravity_H,
            &mut gravity_z,
            &mut gravity_R,
            &mut gravity_y,
            &mut gravity_S,
            &mut gravity_K,
            &mut gravity_temp_S_inv,
            &mut gravity_temp_HP,
            &mut gravity_temp_PHt,
            &mut gravity_temp_KHP,
        );

        // Set initial state.
        initialize_state_vector(&mut filter);
        initialize_state_transition_matrix(&mut filter);
        initialize_state_covariance_matrix(&mut filter);
        initialize_position_measurement_transformation_matrix(&mut measurement);
        initialize_position_measurement_process_noise_matrix(&mut measurement);

        bencher.iter(|| {
            filter.predict();
        })
    });

    c.bench_function("update", |bencher| {
        let mut filter = Kalman::new_from_buffers(
            NUM_STATES,
            NUM_INPUTS,
            &mut gravity_A,
            &mut gravity_x,
            &mut gravity_B,
            &mut gravity_u,
            &mut gravity_P,
            &mut gravity_Q,
            &mut gravity_temp_x,
            &mut gravity_temp_P,
            &mut gravity_temp_BQ,
        );

        let mut measurement = Measurement::new(
            NUM_STATES,
            NUM_MEASUREMENTS,
            &mut gravity_H,
            &mut gravity_z,
            &mut gravity_R,
            &mut gravity_y,
            &mut gravity_S,
            &mut gravity_K,
            &mut gravity_temp_S_inv,
            &mut gravity_temp_HP,
            &mut gravity_temp_PHt,
            &mut gravity_temp_KHP,
        );

        // Set initial state.
        initialize_state_vector(&mut filter);
        initialize_state_transition_matrix(&mut filter);
        initialize_state_covariance_matrix(&mut filter);
        initialize_position_measurement_transformation_matrix(&mut measurement);
        initialize_position_measurement_process_noise_matrix(&mut measurement);

        measurement.measurement_vector_apply(|z| {
            z[0] = black_box(REAL_DISTANCE[0]) + black_box(MEASUREMENT_ERROR[0])
        });

        bencher.iter(|| {
            filter.correct(black_box(&mut measurement));
        })
    });
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut Kalman) {
    filter.state_vector_apply(|state| {
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
fn initialize_state_transition_matrix(filter: &mut Kalman) {
    filter.state_transition_apply(|a| {
        // Time constant.
        const T: matrix_data_t = 1 as _;

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
/// linearly are linearly independent.
fn initialize_state_covariance_matrix(filter: &mut Kalman) {
    filter.system_covariance_apply(|p| {
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
fn initialize_position_measurement_transformation_matrix(measurement: &mut Measurement) {
    measurement.measurement_transformation_apply(|h| {
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
fn initialize_position_measurement_process_noise_matrix(measurement: &mut Measurement) {
    measurement.process_noise_apply(|r| {
        r.set(0, 0, 0.5 as _); // var(s)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
