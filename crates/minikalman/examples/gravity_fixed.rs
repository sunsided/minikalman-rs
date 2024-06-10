//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

#[cfg(feature = "std")]
#[allow(unused)]
use colored::Colorize;
use lazy_static::lazy_static;

use minikalman::prelude::{fixed::I16F16, *};
use minikalman_traits::kalman::*;
use minikalman_traits::matrix::*;

const NUM_STATES: usize = 3;
const NUM_INPUTS: usize = 0;
const NUM_MEASUREMENTS: usize = 1;

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

#[allow(non_snake_case)]
fn main() {
    // System buffers.
    let gravity_x = BufferBuilder::state_vector_x::<NUM_STATES>().new(I16F16::ZERO);
    let gravity_A = BufferBuilder::system_state_transition_A::<NUM_STATES>().new(I16F16::ZERO);
    let gravity_P = BufferBuilder::system_covariance_P::<NUM_STATES>().new(I16F16::ZERO);

    // Input buffers.
    let gravity_u = BufferBuilder::input_vector_u::<NUM_INPUTS>().new(I16F16::ZERO);
    let gravity_B = BufferBuilder::input_transition_B::<NUM_STATES, NUM_INPUTS>().new(I16F16::ZERO);
    let gravity_Q = BufferBuilder::input_covariance_Q::<NUM_INPUTS>().new(I16F16::ZERO);

    // Measurement buffers.
    let gravity_z = BufferBuilder::measurement_vector_z::<NUM_MEASUREMENTS>().new(I16F16::ZERO);
    let gravity_H = BufferBuilder::measurement_transformation_H::<NUM_MEASUREMENTS, NUM_STATES>()
        .new(I16F16::ZERO);
    let gravity_R = BufferBuilder::measurement_covariance_R::<NUM_MEASUREMENTS>().new(I16F16::ZERO);
    let gravity_y = BufferBuilder::innovation_vector_y::<NUM_MEASUREMENTS>().new(I16F16::ZERO);
    let gravity_S = BufferBuilder::innovation_covariance_S::<NUM_MEASUREMENTS>().new(I16F16::ZERO);
    let gravity_K =
        BufferBuilder::kalman_gain_K::<NUM_STATES, NUM_MEASUREMENTS>().new(I16F16::ZERO);

    // Filter temporaries.
    let gravity_temp_x = BufferBuilder::state_prediction_temp_x::<NUM_STATES>().new(I16F16::ZERO);
    let gravity_temp_P = BufferBuilder::temp_system_covariance_P::<NUM_STATES>().new(I16F16::ZERO);
    let gravity_temp_BQ = BufferBuilder::temp_BQ::<NUM_STATES, NUM_INPUTS>().new(I16F16::ZERO);

    // Measurement temporaries.
    let gravity_temp_S_inv = BufferBuilder::temp_S_inv::<NUM_MEASUREMENTS>().new(I16F16::ZERO);
    let gravity_temp_HP =
        BufferBuilder::temp_HP::<NUM_MEASUREMENTS, NUM_STATES>().new(I16F16::ZERO);
    let gravity_temp_PHt =
        BufferBuilder::temp_PHt::<NUM_STATES, NUM_MEASUREMENTS>().new(I16F16::ZERO);
    let gravity_temp_KHP = BufferBuilder::temp_KHP::<NUM_STATES>().new(I16F16::ZERO);

    let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, I16F16>(
        gravity_A,
        gravity_x,
        gravity_B,
        gravity_u,
        gravity_P,
        gravity_Q,
        gravity_temp_x,
        gravity_temp_P,
        gravity_temp_BQ,
    );

    let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, I16F16>(
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
    initialize_position_measurement_transformation_matrix(
        measurement.measurement_transformation_mut(),
    );
    initialize_position_measurement_process_noise_matrix(measurement.process_noise_mut());

    // Filter!
    for t in 0..REAL_DISTANCE.len() {
        // Prediction.
        filter.predict();
        print_state_prediction(t, filter.state_vector_ref());

        // Measure ...
        let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
        measurement.measurement_vector_apply(|z| z[0] = m);
        print_measurement(t);

        // Update.
        filter.correct(&mut measurement);
        print_state_correction(t, filter.state_vector_ref());
    }

    // Fetch estimated gravity constant.
    let gravity_x = filter.state_vector_ref();
    let g_estimated = gravity_x[2];
    assert!(g_estimated > 9.0 && g_estimated < 10.0);
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVector<NUM_STATES, I16F16>) {
    filter.apply(|state| {
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
fn initialize_state_transition_matrix(filter: &mut impl SystemMatrixMut<NUM_STATES, I16F16>) {
    filter.apply(|a| {
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
fn initialize_state_covariance_matrix(
    filter: &mut impl SystemCovarianceMatrix<NUM_STATES, I16F16>,
) {
    filter.apply(|p| {
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
    measurement: &mut impl MeasurementTransformationMatrixMut<NUM_MEASUREMENTS, NUM_STATES, I16F16>,
) {
    measurement.apply(|h| {
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
    measurement: &mut impl MeasurementProcessNoiseCovarianceMatrix<NUM_MEASUREMENTS, I16F16>,
) {
    measurement.apply(|r| {
        r.set(0, 0, I16F16::from_num(0.5)); // var(s)
    });
}

/// Print the state prediction. Will do nothing on `no_std` features.
#[allow(unused)]
fn print_state_prediction<T>(t: usize, x: T)
where
    T: AsRef<[I16F16]>,
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
    T: AsRef<[I16F16]>,
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
fn print_measurement(t: usize) {
    #[cfg(feature = "std")]
    println!(
        "At t = {}, measurement: s = {}, noise ε = {}",
        format!("{}", t).bright_white(),
        format!("{} m", REAL_DISTANCE[t]).green(),
        format!("{} m", MEASUREMENT_ERROR[t]).blue()
    );
}
