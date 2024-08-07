//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

#[cfg(feature = "std")]
#[allow(unused)]
use colored::Colorize;

use minikalman::regular::builder::KalmanFilterBuilder;
use rand_distr::{Distribution, Normal};

use minikalman::prelude::*;

const NUM_STATES: usize = 3; // height, upwards velocity, upwards acceleration
const NUM_CONTROLS: usize = 1; // constant velocity
const NUM_OBSERVATIONS: usize = 1; // position

#[allow(non_snake_case)]
fn main() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut control = builder.controls().build::<NUM_CONTROLS>();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // Set initial state.
    initialize_state_vector(filter.state_vector_mut());
    initialize_state_transition_matrix(filter.state_transition_mut());
    initialize_state_covariance_matrix(filter.estimate_covariance_mut());

    // Set up controls.
    initialize_control_vector(control.control_vector_mut());
    initialize_control_matrix(control.control_matrix_mut());
    initialize_control_covariance_matrix(control.process_noise_covariance_mut());

    // Set up measurements.
    initialize_position_measurement_transformation_matrix(measurement.observation_matrix_mut());
    initialize_position_measurement_process_noise_matrix(
        measurement.measurement_noise_covariance_mut(),
    );

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
        // Update prediction and apply the controls.
        filter.predict();
        filter.control(&mut control);
        print_state_prediction(t, filter.state_vector());

        // Measure ...
        measurement
            .measurement_vector_mut()
            .apply(|z| z[0] = m + err);
        print_measurement(t, m, err);

        // Update.
        filter.correct(&mut measurement);
        print_state_correction(t, filter.state_vector());
    }

    // Fetch estimated gravity constant.
    let gravity_x = filter.state_vector();
    let g_estimated = gravity_x.as_matrix().get_at(0, 2);
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
    let normal = Normal::new(0.0, 0.1).unwrap();
    let mut rng = rand::thread_rng();

    let mut error = vec![0.0; n]; // vector to store the generated errors

    for i in 0..n {
        error[i] = 0.5 * 0.5 * normal.sample(&mut rng);
    }

    error // return the generated errors
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVectorMut<NUM_STATES, f32>) {
    filter.as_matrix_mut().apply(|state| {
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

/// Initializes the control vector.
fn initialize_control_vector(filter: &mut impl ControlVectorMut<NUM_CONTROLS, f32>) {
    filter.as_matrix_mut().apply(|state| {
        state[0] = 0.0 as _; // acceleration
    });
}

/// Initializes the control transformation matrix.
fn initialize_control_matrix(filter: &mut impl ControlMatrixMut<NUM_STATES, NUM_CONTROLS, f32>) {
    filter.as_matrix_mut().apply(|mat| {
        mat[0] = 0.0;
        mat[1] = 0.0;
        mat[2] = 1.0;
    });
}

/// Initializes the control covariance.
fn initialize_control_covariance_matrix(
    filter: &mut impl ControlProcessNoiseCovarianceMatrixMut<NUM_CONTROLS, f32>,
) {
    filter.as_matrix_mut().apply(|mat| {
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
