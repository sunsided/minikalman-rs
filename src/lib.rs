//! # Kalman Filters for Embedded Targets (in Rust)
//!
//! This is the Rust port of the [kalman-clib](https://github.com/sunsided/kalman-clib/) library,
//! a microcontroller targeted Kalman filter implementation. Uses [`micromath`](https://docs.rs/micromath)
//! for square root calculations on `no_std`.
//!
//! This implementation uses statically allocated buffers for all matrix operations. Due to lack
//! of `const` generics for array allocations in Rust, this crate also provides helper macros
//! to create the required arrays (see e.g. [`create_buffer_A`]).

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
// Enable `no_std` if the `no_std` crate feature is enabled.
#![cfg_attr(feature = "no_std", no_std)]
// Forbid unsafe code unless the `unsafe` crate feature is explicitly enabled.
#![cfg_attr(not(feature = "unsafe"), forbid(unsafe_code))]

extern crate alloc;

mod kalman;
mod matrix;
mod matrix_ops;
mod measurement;
mod types;

pub use matrix::Matrix;

pub use crate::kalman::Kalman;
pub use crate::matrix_ops::{MatrixBase, MatrixOps};
pub use crate::measurement::Measurement;
pub use crate::types::*;

/// Creates a buffer fitting the state transition matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
///
/// // System buffers.
/// let mut gravity_x = create_buffer_x!(NUM_STATES);
/// let mut gravity_A = create_buffer_A!(NUM_STATES);
/// let mut gravity_P = create_buffer_P!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_A {
    ( $num_states:expr ) => {
        (create_buffer_A!($num_states, f32))
    };
    ( $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the state covariance matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
///
/// // System buffers.
/// let mut gravity_x = create_buffer_x!(NUM_STATES);
/// let mut gravity_A = create_buffer_A!(NUM_STATES);
/// let mut gravity_P = create_buffer_P!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_P {
    ( $num_states:expr ) => {
        (create_buffer_P!($num_states, f32))
    };
    ( $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the state vector (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
///
/// // System buffers.
/// let mut gravity_x = create_buffer_x!(NUM_STATES);
/// let mut gravity_A = create_buffer_A!(NUM_STATES);
/// let mut gravity_P = create_buffer_P!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_x {
    ( $num_states:expr ) => {
        (create_buffer_x![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * 1) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the input vector (`num_inputs` × `1`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Input buffers.
/// let mut gravity_u = create_buffer_u!(NUM_INPUTS);
/// let mut gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
/// let mut gravity_Q = create_buffer_Q!(NUM_INPUTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_u {
    ( $num_inputs:expr ) => {
        (create_buffer_u![$num_inputs, f32])
    };
    ( $num_inputs:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        const COUNT: usize = (NUM_INPUTS_ * 1) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the input transition matrix (`num_states` × `num_inputs`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_inputs` - The number of inputs.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Input buffers.
/// let mut gravity_u = create_buffer_u!(NUM_INPUTS);
/// let mut gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
/// let mut gravity_Q = create_buffer_Q!(NUM_INPUTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_B {
    ( $num_states:expr, $num_inputs:expr ) => {
        (create_buffer_B![$num_states, $num_inputs, f32])
    };
    ( $num_states:expr, $num_inputs:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_INPUTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Input buffers.
/// let mut gravity_u = create_buffer_u!(NUM_INPUTS);
/// let mut gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
/// let mut gravity_Q = create_buffer_Q!(NUM_INPUTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_Q {
    ( $num_states:expr ) => {
        (create_buffer_Q![$num_states, f32])
    };
    ( $num_inputs:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        const COUNT: usize = (NUM_INPUTS_ * NUM_INPUTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_z {
    ( $num_states:expr ) => {
        (create_buffer_z![$num_states, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_MEASUREMENTS_ * 1) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_H {
    ( $num_measurements:expr, $num_states:expr ) => {
        (create_buffer_H![$num_measurements, $num_states, f32])
    };
    ( $num_measurements:expr, $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_STATES_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_R {
    ( $num_measurements:expr ) => {
        (create_buffer_R![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_y {
    ( $num_measurements:expr ) => {
        (create_buffer_y![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_MEASUREMENTS_ * 1) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the innovation covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_S {
    ( $num_measurements:expr ) => {
        (create_buffer_S![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_K {
    ( $num_states:expr, $num_measurements:expr ) => {
        (create_buffer_K![$num_states, $num_measurements, f32])
    };
    ( $num_states:expr, $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Filter temporaries.
/// let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
/// let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
/// let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_x {
    ( $num_states:expr ) => {
        (create_buffer_temp_x![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * 1) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary P matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Filter temporaries.
/// let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
/// let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
/// let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_P {
    ( $num_states:expr ) => {
        (create_buffer_temp_P![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary B×Q matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_inputs` - The number of inputs.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Filter temporaries.
/// let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
/// let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
/// let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_BQ {
    ( $num_states:expr, $num_inputs:expr ) => {
        (create_buffer_temp_BQ![$num_states, $num_inputs, f32])
    };
    ( $num_states:expr, $num_inputs:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_INPUTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_S_inv {
    ( $num_measurements:expr ) => {
        (create_buffer_temp_S_inv![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_HP {
    ( $num_measurements:expr, $num_states:expr ) => {
        (create_buffer_temp_HP![$num_measurements, $num_states, f32])
    };
    ( $num_measurements:expr, $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary P×H^-1 buffer (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_PHt {
    ( $num_states:expr, $num_measurements:expr ) => {
        (create_buffer_temp_PHt![$num_states, $num_measurements, f32])
    };
    ( $num_states:expr, $num_measurements:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_MEASUREMENTS_) as usize;
        [0.0 as $t; COUNT]
    }};
}

/// Creates a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_KHP {
    ( $num_states:expr ) => {
        (create_buffer_temp_KHP![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const COUNT: usize = (NUM_STATES_ * NUM_STATES_) as usize;
        [0.0 as $t; COUNT]
    }};
}
