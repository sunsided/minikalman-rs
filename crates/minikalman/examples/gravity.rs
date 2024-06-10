//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

#[cfg(feature = "std")]
#[allow(unused)]
use colored::Colorize;

use rand_distr::{Distribution, Normal};

use minikalman::prelude::*;

const NUM_STATES: usize = 3; // height, upwards velocity, upwards acceleration
const NUM_INPUTS: usize = 1; // constant velocity
const NUM_MEASUREMENTS: usize = 1; // position

#[allow(non_snake_case)]
fn main() {
    // System buffers.
    let gravity_x = BufferBuilder::state_vector_x::<NUM_STATES>().new(0.0_f32);
    let gravity_A = BufferBuilder::system_state_transition_A::<NUM_STATES>().new(0.0_f32);
    let gravity_P = BufferBuilder::system_covariance_P::<NUM_STATES>().new(0.0_f32);

    // Input buffers.
    let gravity_u = BufferBuilder::input_vector_u::<NUM_INPUTS>().new(0.0_f32);
    let gravity_B = BufferBuilder::input_transition_B::<NUM_STATES, NUM_INPUTS>().new(0.0_f32);
    let gravity_Q = BufferBuilder::input_covariance_Q::<NUM_INPUTS>().new(0.0_f32);

    // Measurement buffers.
    let gravity_z = BufferBuilder::measurement_vector_z::<NUM_MEASUREMENTS>().new(0.0_f32);
    let gravity_H =
        BufferBuilder::measurement_transformation_H::<NUM_MEASUREMENTS, NUM_STATES>().new(0.0_f32);
    let gravity_R = BufferBuilder::measurement_covariance_R::<NUM_MEASUREMENTS>().new(0.0_f32);
    let gravity_y = BufferBuilder::innovation_vector_y::<NUM_MEASUREMENTS>().new(0.0_f32);
    let gravity_S = BufferBuilder::innovation_covariance_S::<NUM_MEASUREMENTS>().new(0.0_f32);
    let gravity_K = BufferBuilder::kalman_gain_K::<NUM_STATES, NUM_MEASUREMENTS>().new(0.0_f32);

    // Filter temporaries.
    let gravity_temp_x = BufferBuilder::state_prediction_temp_x::<NUM_STATES>().new(0.0_f32);
    let gravity_temp_P = BufferBuilder::temp_system_covariance_P::<NUM_STATES>().new(0.0_f32);
    let gravity_temp_BQ = BufferBuilder::temp_BQ::<NUM_STATES, NUM_INPUTS>().new(0.0_f32);

    // Measurement temporaries.
    let gravity_temp_S_inv = BufferBuilder::temp_S_inv::<NUM_MEASUREMENTS>().new(0.0_f32);
    let gravity_temp_HP = BufferBuilder::temp_HP::<NUM_MEASUREMENTS, NUM_STATES>().new(0.0_f32);
    let gravity_temp_PHt = BufferBuilder::temp_PHt::<NUM_STATES, NUM_MEASUREMENTS>().new(0.0_f32);
    let gravity_temp_KHP = BufferBuilder::temp_KHP::<NUM_STATES>().new(0.0_f32);

    let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
        gravity_A,
        gravity_x,
        gravity_P,
        gravity_temp_x,
        gravity_temp_P,
    );

    let mut input = InputBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
        gravity_B,
        gravity_u,
        gravity_Q,
        gravity_temp_BQ,
    );

    let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
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
    initialize_state_covariance_matrix(filter.system_covariance_mut());

    // Set up inputs.
    initialize_input_vector(input.input_vector_mut());
    initialize_input_matrix(input.input_transition_mut());
    initialize_input_covariance_matrix(input.input_covariance_mut());

    // Set up measurements.
    initialize_position_measurement_transformation_matrix(
        measurement.measurement_transformation_mut(),
    );
    initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

    // Generate the data.
    let measurements = generate_values(100);
    let measurement_noise = generate_error(100);

    // Filter!
    for (t, (m, err)) in measurements
        .iter()
        .copied()
        .zip(measurement_noise)
        .enumerate()
    {
        // Update prediction and apply the inputs.
        filter.predict();
        filter.input(&mut input);
        print_state_prediction(t, filter.state_vector_ref());

        // Measure ...
        measurement.measurement_vector_apply(|z| z[0] = m + err);
        print_measurement(t, m, err);

        // Update.
        filter.correct(&mut measurement);
        print_state_correction(t, filter.state_vector_ref());
    }

    // Fetch estimated gravity constant.
    let gravity_x = filter.state_vector_ref();
    let g_estimated = gravity_x[2];
    assert!(g_estimated > 9.0 && g_estimated < 10.0);
}

fn generate_values(n: usize) -> Vec<f32> {
    let g = 9.81; // acceleration due to gravity
    let mut s = 0.0; // initial displacement
    let mut v = 0.0; // initial velocity
    let delta_t = 1.0; // time duration (1 second)

    let mut values = vec![0.0; n]; // vector to store the generated values

    for i in 0..n {
        s = s + v * delta_t + g * 0.5 * delta_t * delta_t;
        v = v + g * delta_t;
        values[i] = s;
    }

    values // return the generated values
}

/// Generate measurement error with variance 0.5
/// MATLAB source: noise = 0.5^2*randn(15,1);
fn generate_error(n: usize) -> Vec<f32> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::thread_rng();

    let mut error = vec![0.0; n]; // vector to store the generated errors

    for i in 0..n {
        error[i] = 0.5 * 0.5 * normal.sample(&mut rng);
    }

    error // return the generated errors
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVector<NUM_STATES, f32>) {
    filter.apply(|state| {
        state[0] = 0 as _; // position
        state[1] = 0 as _; // velocity
        state[2] = 6 as _; // acceleration (guess)
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
        p.set(0, 0, 0.1 as _); // var(s)
        p.set(0, 1, 0 as _); // cov(s, v)
        p.set(0, 2, 0 as _); // cov(s, g)

        p.set(1, 1, 1 as _); // var(v)
        p.set(1, 2, 0 as _); // cov(v, g)

        p.set(2, 2, 1 as _); // var(g)
    });
}

/// Initializes the input vector.
fn initialize_input_vector(filter: &mut impl InputVectorMut<NUM_INPUTS, f32>) {
    filter.apply(|state| {
        state[0] = 0.0 as _; // acceleration
    });
}

/// Initializes the input transformation matrix.
fn initialize_input_matrix(filter: &mut impl InputMatrixMut<NUM_STATES, NUM_INPUTS, f32>) {
    filter.apply(|mat| {
        // Time constant.
        const T: f32 = 1 as _;

        mat[0] = 0.0;
        mat[1] = 0.0;
        mat[2] = 1.0;
    });
}

/// Initializes the input covariance.
fn initialize_input_covariance_matrix(filter: &mut impl InputCovarianceMatrixMut<NUM_INPUTS, f32>) {
    filter.apply(|mat| {
        mat[0] = 1.0; // :)
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
        r.set(0, 0, 0.5 as _); // var(s)
    });
}

/// Print the state prediction. Will do nothing on `no_std` features.
#[allow(unused)]
fn print_state_prediction<T>(t: usize, x: T)
where
    T: AsRef<[f32]>,
{
    let x = x.as_ref();
    #[cfg(feature = "std")]
    println!(
        "At t = {}, predicted state: s = {}, v = {}, a = {}",
        format!("{}", t).bright_white(),
        format!("{} m", x[0]).magenta(),
        format!("{} m/s", x[1]).magenta(),
        format!("{} m/s²", x[2]).magenta(),
    );
}

/// Print the measurement corrected state. Will do nothing on `no_std` features.
#[allow(unused)]
fn print_state_correction<T>(t: usize, x: T)
where
    T: AsRef<[f32]>,
{
    let x = x.as_ref();
    #[cfg(feature = "std")]
    println!(
        "At t = {}, corrected state: s = {}, v = {}, a = {}",
        format!("{}", t).bright_white(),
        format!("{} m", x[0]).yellow(),
        format!("{} m/s", x[1]).yellow(),
        format!("{} m/s²", x[2]).yellow(),
    );
}

/// Print the current measurement. Will do nothing on `no_std` features.
#[allow(unused)]
fn print_measurement(t: usize, real: f32, error: f32) {
    #[cfg(feature = "std")]
    println!(
        "At t = {}, measurement: s = {}, noise ε = {}",
        format!("{}", t).bright_white(),
        format!("{} m", real).green(),
        format!("{} m", error).blue()
    );
}
