use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fixed::types::I16F16;
use lazy_static::lazy_static;

use minikalman::{
    create_buffer_A, create_buffer_B, create_buffer_H, create_buffer_K, create_buffer_P,
    create_buffer_Q, create_buffer_R, create_buffer_S, create_buffer_temp_BQ,
    create_buffer_temp_HP, create_buffer_temp_KHP, create_buffer_temp_P, create_buffer_temp_PHt,
    create_buffer_temp_S_inv, create_buffer_temp_x, create_buffer_u, create_buffer_x,
    create_buffer_y, create_buffer_z, Kalman, Measurement,
};

lazy_static! {
    /// Measurements.
    ///
    /// MATLAB source:
    /// ```matlab
    /// s = s + v*T + g*0.5*T^2;
    /// v = v + g*T;
    /// ```
    static ref REAL_DISTANCE: [I16F16; 15] = [
        I16F16::from_num(0.0),
        I16F16::from_num(4.905),
        I16F16::from_num(19.62),
        I16F16::from_num(44.145),
        I16F16::from_num(78.48),
        I16F16::from_num(122.63),
        I16F16::from_num(176.58),
        I16F16::from_num(240.35),
        I16F16::from_num(313.92),
        I16F16::from_num(397.31),
        I16F16::from_num(490.5),
        I16F16::from_num(593.51),
        I16F16::from_num(706.32),
        I16F16::from_num(828.94),
        I16F16::from_num(961.38),
    ];

    /// Measurement noise with variance 0.5
    ///
    /// MATLAB source:
    /// ```matlab
    /// noise = 0.5^2*randn(15,1);
    /// ```
    static ref MEASUREMENT_ERROR: [I16F16; 15] = [
        I16F16::from_num(0.13442),
        I16F16::from_num(0.45847),
        I16F16::from_num(-0.56471),
        I16F16::from_num(0.21554),
        I16F16::from_num(0.079691),
        I16F16::from_num(-0.32692),
        I16F16::from_num(-0.1084),
        I16F16::from_num(0.085656),
        I16F16::from_num(0.8946),
        I16F16::from_num(0.69236),
        I16F16::from_num(-0.33747),
        I16F16::from_num(0.75873),
        I16F16::from_num(0.18135),
        I16F16::from_num(-0.015764),
        I16F16::from_num(0.17869),
    ];
}

const NUM_STATES: usize = 3;
const NUM_INPUTS: usize = 0;
const NUM_MEASUREMENTS: usize = 1;

#[allow(non_snake_case)]
fn criterion_benchmark(c: &mut Criterion) {
    // System buffers.
    let mut gravity_x = create_buffer_x!(NUM_STATES, I16F16, I16F16::ZERO);
    let mut gravity_A = create_buffer_A!(NUM_STATES, I16F16, I16F16::ZERO);
    let mut gravity_P = create_buffer_P!(NUM_STATES, I16F16, I16F16::ZERO);

    // Input buffers.
    let mut gravity_u = create_buffer_u!(0, I16F16, I16F16::ZERO);
    let mut gravity_B = create_buffer_B!(0, 0, I16F16, I16F16::ZERO);
    let mut gravity_Q = create_buffer_Q!(0, I16F16, I16F16::ZERO);

    // Measurement buffers.
    let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS, I16F16, I16F16::ZERO);
    let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES, I16F16, I16F16::ZERO);
    let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS, I16F16, I16F16::ZERO);
    let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS, I16F16, I16F16::ZERO);
    let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS, I16F16, I16F16::ZERO);
    let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS, I16F16, I16F16::ZERO);

    // Filter temporaries.
    let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES, I16F16, I16F16::ZERO);
    let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES, I16F16, I16F16::ZERO);
    let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS, I16F16, I16F16::ZERO);

    // Measurement temporaries.
    let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS, I16F16, I16F16::ZERO);
    let mut gravity_temp_HP =
        create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES, I16F16, I16F16::ZERO);
    let mut gravity_temp_PHt =
        create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS, I16F16, I16F16::ZERO);
    let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES, I16F16, I16F16::ZERO);

    c.bench_function("filter loop (fixed-point)", |bencher| {
        let mut filter = Kalman::<NUM_STATES, NUM_INPUTS, I16F16>::new_direct(
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

        let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS, I16F16>::new_direct(
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

    c.bench_function("predict (fixed-point)", |bencher| {
        let mut filter = Kalman::<NUM_STATES, NUM_INPUTS, I16F16>::new_direct(
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

        let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS, I16F16>::new_direct(
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

    c.bench_function("update (fixed-point)", |bencher| {
        let mut filter = Kalman::<NUM_STATES, NUM_INPUTS, I16F16>::new_direct(
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

        let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS, I16F16>::new_direct(
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
fn initialize_state_vector(filter: &mut Kalman<'_, NUM_STATES, NUM_INPUTS, I16F16>) {
    filter.state_vector_apply(|state| {
        state[0] = I16F16::ZERO; // position
        state[1] = I16F16::ZERO; // velocity
        state[2] = I16F16::from_num(6.0); // acceleration
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
fn initialize_state_transition_matrix(filter: &mut Kalman<'_, NUM_STATES, NUM_INPUTS, I16F16>) {
    filter.state_transition_apply(|a| {
        // Time constant.
        const T: I16F16 = I16F16::ONE;

        // Transition of x to s.
        a.set(0, 0, I16F16::ONE); // 1
        a.set(0, 1, T as _); // T
        a.set(0, 2, I16F16::from_num(0.5) * T * T); // 0.5 × T²

        // Transition of x to v.
        a.set(1, 0, I16F16::ZERO); // 0
        a.set(1, 1, I16F16::ONE); // 1
        a.set(1, 2, T as _); // T

        // Transition of b to g.
        a.set(2, 0, I16F16::ZERO); // 0
        a.set(2, 1, I16F16::ZERO); // 0
        a.set(2, 2, I16F16::ONE); // 1
    });
}

/// Initializes the system covariance matrix.
///
/// This defines how different states (linearly) influence each other
/// over time. In this setup we claim that position, velocity and acceleration
/// are linearly independent.
fn initialize_state_covariance_matrix(filter: &mut Kalman<'_, NUM_STATES, NUM_INPUTS, I16F16>) {
    filter.system_covariance_apply(|p| {
        p.set(0, 0, I16F16::from_num(0.1)); // var(s)
        p.set(0, 1, I16F16::ZERO); // cov(s, v)
        p.set(0, 2, I16F16::ZERO); // cov(s, g)

        p.set(1, 1, I16F16::ONE); // var(v)
        p.set(1, 2, I16F16::ZERO); // cov(v, g)

        p.set(2, 2, I16F16::ONE); // var(g)
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
    measurement: &mut Measurement<'_, NUM_STATES, NUM_MEASUREMENTS, I16F16>,
) {
    measurement.measurement_transformation_apply(|h| {
        h.set(0, 0, I16F16::ONE); // z = 1*s
        h.set(0, 1, I16F16::ZERO); //   + 0*v
        h.set(0, 2, I16F16::ZERO); //   + 0*g
    });
}

/// Initializes the measurement noise / uncertainty matrix.
///
/// This matrix describes the measurement covariances as well as the
/// individual variation components. It is the measurement counterpart
/// of the state covariance matrix.
fn initialize_position_measurement_process_noise_matrix(
    measurement: &mut Measurement<'_, NUM_STATES, NUM_MEASUREMENTS, I16F16>,
) {
    measurement.process_noise_apply(|r| {
        r.set(0, 0, I16F16::from_num(0.5)); // var(s)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
