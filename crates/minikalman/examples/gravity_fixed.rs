//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

#[cfg(feature = "std")]
#[allow(unused)]
use colored::Colorize;
use lazy_static::lazy_static;
use minikalman::regular::builder::KalmanFilterBuilder;

use minikalman::prelude::*;

const NUM_STATES: usize = 3;
const NUM_OBSERVATIONS: usize = 1;

lazy_static! {
    /// Observations.
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

    /// Observation noise with variance 0.5
    ///
    /// MATLAB source:
    /// ```matlab
    /// noise = 0.5^2*randn(15,1);
    /// ```
    static ref OBSERVATION_ERROR: [I16F16; 15] = [
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
    let builder = KalmanFilterBuilder::<NUM_STATES, I16F16>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

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
        print_state_prediction(t, filter.state_vector());

        // Measure ...
        let m = REAL_DISTANCE[t] + OBSERVATION_ERROR[t];
        measurement.measurement_vector_mut().apply(|z| z[0] = m);
        print_measurement(t);

        // Update.
        filter.correct(&mut measurement);
        print_state_correction(t, filter.state_vector());
    }

    // Fetch estimated gravity constant.
    let gravity_x = filter.state_vector();
    let g_estimated = gravity_x[2];
    assert!(g_estimated > 9.0 && g_estimated < 10.0);
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVectorMut<NUM_STATES, I16F16>) {
    filter.as_matrix_mut().apply(|state| {
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
fn initialize_state_transition_matrix(
    filter: &mut impl StateTransitionMatrixMut<NUM_STATES, I16F16>,
) {
    filter.as_matrix_mut().apply(|a| {
        // Time constant.
        const T: I16F16 = I16F16::ONE;

        // Transition of x to s.
        a.set_at(0, 0, I16F16::ONE); // 1
        a.set_at(0, 1, T as _); // T
        a.set_at(0, 2, I16F16::from_num(0.5) * T * T); // 0.5 × T²

        // Transition of x to v.
        a.set_at(1, 0, I16F16::ZERO); // 0
        a.set_at(1, 1, I16F16::ONE); // 1
        a.set_at(1, 2, T as _); // T

        // Transition of b to g.
        a.set_at(2, 0, I16F16::ZERO); // 0
        a.set_at(2, 1, I16F16::ZERO); // 0
        a.set_at(2, 2, I16F16::ONE); // 1
    });
}

/// Initializes the system covariance matrix.
///
/// This defines how different states (linearly) influence each other
/// over time. In this setup we claim that position, velocity and acceleration
/// are linearly independent.
fn initialize_state_covariance_matrix(
    filter: &mut impl EstimateCovarianceMatrix<NUM_STATES, I16F16>,
) {
    filter.as_matrix_mut().apply(|p| {
        p.set_at(0, 0, I16F16::from_num(0.1)); // var(s)
        p.set_at(0, 1, I16F16::ZERO); // cov(s, v)
        p.set_at(0, 2, I16F16::ZERO); // cov(s, g)

        p.set_at(1, 1, I16F16::ONE); // var(v)
        p.set_at(1, 2, I16F16::ZERO); // cov(v, g)

        p.set_at(2, 2, I16F16::ONE); // var(g)
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
    measurement: &mut impl ObservationMatrixMut<NUM_OBSERVATIONS, NUM_STATES, I16F16>,
) {
    measurement.as_matrix_mut().apply(|h| {
        h.set_at(0, 0, I16F16::ONE); // z = 1*s
        h.set_at(0, 1, I16F16::ZERO); //   + 0*v
        h.set_at(0, 2, I16F16::ZERO); //   + 0*g
    });
}

/// Initializes the measurement noise / uncertainty matrix.
///
/// This matrix describes the measurement covariances as well as the
/// individual variation components. It is the measurement counterpart
/// of the state covariance matrix.
fn initialize_position_measurement_process_noise_matrix(
    measurement: &mut impl MeasurementNoiseCovarianceMatrix<NUM_OBSERVATIONS, I16F16>,
) {
    measurement.as_matrix_mut().apply(|r| {
        r.set_at(0, 0, I16F16::from_num(0.5)); // var(s)
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
        format!("{} m", OBSERVATION_ERROR[t]).blue()
    );
}
