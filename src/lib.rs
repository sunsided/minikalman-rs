//! # Kalman Filters for Embedded Targets (in Rust)
//!
//! This is the Rust port of the [kalman-clib](https://github.com/sunsided/kalman-clib/) library,
//! a microcontroller targeted Kalman filter implementation. Uses [`micromath`](https://docs.rs/micromath)
//! for square root calculations on `no_std`.
//!
//! This implementation uses statically allocated buffers for all matrix operations. Due to lack
//! of `const` generics for array allocations in Rust, this crate also provides helper macros
//! to create the required arrays (see e.g. [`create_buffer_A`]).

// Enable `no_std` if the `no_std` crate feature is enabled.
#![cfg_attr(feature = "no_std", no_std)]
// Forbid unsafe code unless the `unsafe` crate feature is explicitly enabled.
#![cfg_attr(not(feature = "unsafe"), forbid(unsafe_code))]

extern crate alloc;

mod kalman;
mod matrix;
mod matrix_ops;
mod matrix_owned;
mod measurement;
mod scratch;

pub use matrix::matrix_data_t;
pub use matrix::Matrix;

pub use crate::kalman::Kalman;
pub use crate::matrix_ops::{MatrixBase, MatrixOps};
pub use crate::measurement::Measurement;
pub use crate::scratch::ScratchBuffer;

/// Creates a buffer fitting the state transition matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_A {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the state covariance matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_P {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the state vector (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_x {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * 1) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the input vector (`num_inputs` × `1`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_u {
    ( $num_inputs:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_INPUTS_: uint_fast16_t = ($num_inputs) as uint_fast16_t;
        const COUNT: usize = (NUM_INPUTS_ * 1) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the input transition matrix (`num_states` × `num_inputs`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_inputs` - The number of inputs.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_B {
    ( $num_states:expr, $num_inputs:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const NUM_INPUTS_: uint_fast16_t = ($num_inputs) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_INPUTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_Q {
    ( $num_inputs:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_INPUTS_: uint_fast16_t = ($num_inputs) as uint_fast16_t;
        const COUNT: usize = (NUM_INPUTS_ * NUM_INPUTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary P matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_P_temp {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary B×Q matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_BQ_temp {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_z {
    ( $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_MEASUREMENTS_ * 1) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_H {
    ( $num_measurements:expr, $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_R {
    ( $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_y {
    ( $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_MEASUREMENTS_ * 1) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the innovation covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_S {
    ( $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_K {
    ( $num_states:expr, $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_x {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * 1) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary P buffer (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_P {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary B×Q buffer (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_inputs` - The number of inputs.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_BQ {
    ( $num_states:expr, $num_inputs:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const NUM_INPUTS_: uint_fast16_t = ($num_inputs) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_INPUTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_S_inv {
    ( $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_HP {
    ( $num_measurements:expr, $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary P×H^-1 buffer (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_PHt {
    ( $num_states:expr, $num_measurements:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const NUM_MEASUREMENTS_: uint_fast16_t = ($num_measurements) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_KHP {
    ( $num_states:expr ) => {{
        use minikalman::matrix_data_t;
        use stdint::{uint_fast16_t, uint_fast8_t};

        const NUM_STATES_: uint_fast16_t = ($num_states) as uint_fast16_t;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as matrix_data_t; COUNT]
    }};
}
