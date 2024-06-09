//! # Kalman Filters for Embedded Targets (in Rust)
//!
//! This is the Rust port of the [kalman-clib](https://github.com/sunsided/kalman-clib/) library,
//! a microcontroller targeted Kalman filter implementation. Uses [`micromath`](https://docs.rs/micromath)
//! for square root calculations on `no_std`.
//!
//! This implementation uses statically allocated buffers for all matrix operations. Due to lack
//! of `const` generics for array allocations in Rust, this crate also provides helper macros
//! to create the required arrays (see e.g. [`create_buffer_A`]).
//!
//! ## Crate Features
//!
//! * `std` - Disabled by default. Disables the `no_std` configuration attribute (enabling `std` support).
//! * `libm` - Enables libm support.
//! * `float` - Enables some in-built support for `f32` and `f64` support.
//! * `fixed` - Enables fixed-point support via the [fixed](https://crates.io/crates/fixed) crate.
//! * `unsafe` - Enables some unsafe pointer operations. Disabled by default; when turned off,
//!              compiles the crate as `#![forbid(unsafe)]`.

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
// Enable `no_std` if the `no_std` crate feature is enabled.
#![cfg_attr(not(feature = "std"), no_std)]
// Forbid unsafe code unless the `unsafe` crate feature is explicitly enabled.
#![cfg_attr(not(feature = "unsafe"), forbid(unsafe_code))]
// Attempt to disable allocations.
#![forbid(box_pointers)]

pub mod buffer_types;
mod filter_traits;
mod kalman;
mod matrix_traits;
mod matrix_types;
mod measurement;
mod static_macros;
mod types;

pub use crate::kalman::{Kalman, KalmanBuilder};
pub use crate::matrix_types::*;
pub use crate::measurement::{Measurement, MeasurementBuilder};
pub use crate::types::*;

/// Re-export `num_traits`.
pub use num_traits;

/// Exports all macros and common types.
pub mod prelude {
    pub use crate::buffer_types;
    pub use crate::filter_traits::*;
    pub use crate::kalman::{Kalman, KalmanBuilder};
    pub use crate::matrix_traits::*;
    pub use crate::matrix_types::{IntoInnerData, MatrixData};
    pub use crate::measurement::{Measurement, MeasurementBuilder};
    pub use crate::types::*;

    pub use crate::{
        impl_static_buffer_A, impl_static_buffer_B, impl_static_buffer_H, impl_static_buffer_K,
        impl_static_buffer_P, impl_static_buffer_Q, impl_static_buffer_R, impl_static_buffer_S,
        impl_static_buffer_temp_BQ, impl_static_buffer_temp_HP, impl_static_buffer_temp_KHP,
        impl_static_buffer_temp_P, impl_static_buffer_temp_PHt, impl_static_buffer_temp_S_inv,
        impl_static_buffer_temp_x, impl_static_buffer_u, impl_static_buffer_x,
        impl_static_buffer_y, impl_static_buffer_z,
    };

    pub use crate::{
        create_buffer_A, create_buffer_B, create_buffer_H, create_buffer_K, create_buffer_P,
        create_buffer_Q, create_buffer_R, create_buffer_S, create_buffer_u, create_buffer_x,
        create_buffer_y, create_buffer_z,
    };
    pub use crate::{
        create_buffer_temp_BQ, create_buffer_temp_HP, create_buffer_temp_KHP, create_buffer_temp_P,
        create_buffer_temp_PHt, create_buffer_temp_S_inv, create_buffer_temp_x,
    };
    pub use crate::{
        size_buffer_A, size_buffer_B, size_buffer_H, size_buffer_K, size_buffer_P, size_buffer_Q,
        size_buffer_R, size_buffer_S, size_buffer_u, size_buffer_x, size_buffer_y, size_buffer_z,
    };
    pub use crate::{
        size_buffer_temp_BQ, size_buffer_temp_HP, size_buffer_temp_KHP, size_buffer_temp_P,
        size_buffer_temp_PHt, size_buffer_temp_S_inv, size_buffer_temp_x,
    };

    #[cfg_attr(docsrs, doc(cfg(feature = "fixed")))]
    #[cfg(feature = "fixed")]
    pub mod fixed {
        pub use fixed::types::{I16F16, I32F32};
    }
}

/// Sizes a buffer fitting the state transition matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_A!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_A {
    ( $num_states:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Creates a buffer fitting the state transition matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
///
/// // System buffers.
/// let gravity_x = create_buffer_x!(NUM_STATES);
/// let gravity_A = create_buffer_A!(NUM_STATES);
/// let gravity_P = create_buffer_P!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// let gravity_A = create_buffer_A!(NUM_STATES, f64);
/// let gravity_A = create_buffer_A!(NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_A {
    ( $num_states:expr ) => {
        (create_buffer_A!($num_states, f32))
    };
    ( $num_states:expr, $t:ty ) => {{
        create_buffer_A!($num_states, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_A!($num_states);
        $crate::buffer_types::SystemMatrixMutBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_states,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the state covariance matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_P!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_P {
    ( $num_states:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Creates a buffer fitting the state covariance matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
///
/// // System buffers.
/// let gravity_x = create_buffer_x!(NUM_STATES);
/// let gravity_A = create_buffer_A!(NUM_STATES);
/// let gravity_P = create_buffer_P!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// let gravity_P = create_buffer_P!(NUM_STATES, f64);
/// let gravity_P = create_buffer_P!(NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_P {
    ( $num_states:expr ) => {
        (create_buffer_P!($num_states, f32))
    };
    ( $num_states:expr, $t:ty ) => {{
        create_buffer_P!($num_states, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_P!($num_states);
        $crate::buffer_types::SystemCovarianceMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_states,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the state vector (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_x!(NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_x {
    ( $num_states:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * 1) as usize
    }};
}

/// Creates a buffer fitting the state vector (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
///
/// // System buffers.
/// let gravity_x = create_buffer_x!(NUM_STATES);
/// let gravity_A = create_buffer_A!(NUM_STATES);
/// let gravity_P = create_buffer_P!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// let gravity_x = create_buffer_x!(NUM_STATES, f64);
/// let gravity_x = create_buffer_x!(NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_x {
    ( $num_states:expr ) => {
        (create_buffer_x![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        create_buffer_x!($num_states, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_x!($num_states);
        $crate::buffer_types::StateVectorBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            1,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the input vector (`num_inputs` × `1`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_INPUTS: usize = 1;
/// assert_eq!(size_buffer_u!(NUM_INPUTS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_u {
    ( $num_inputs:expr ) => {{
        use $crate::FastUInt16;

        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        (NUM_INPUTS_ * 1) as usize
    }};
}

/// Creates a buffer fitting the input vector (`num_inputs` × `1`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Input buffers.
/// let gravity_u = create_buffer_u!(NUM_INPUTS);
/// let gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
/// let gravity_Q = create_buffer_Q!(NUM_INPUTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_INPUTS: usize = 1;
/// let gravity_u = create_buffer_u!(NUM_INPUTS, f64);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_u {
    ( $num_inputs:expr ) => {
        (create_buffer_u![$num_inputs, f32])
    };
    ( $num_inputs:expr, $t:ty ) => {{
        create_buffer_u!($num_inputs, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_inputs:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_u!($num_inputs);
        $crate::buffer_types::InputVectorBuffer::new($crate::MatrixData::new_owned::<
            $num_inputs,
            1,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the input transition matrix (`num_states` × `num_inputs`).
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
/// assert_eq!(size_buffer_B!(NUM_STATES, NUM_INPUTS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_B {
    ( $num_states:expr, $num_inputs:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        (NUM_STATES_ * NUM_INPUTS_) as usize
    }};
}

/// Creates a buffer fitting the input transition matrix (`num_states` × `num_inputs`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_inputs` - The number of inputs.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Input buffers.
/// let gravity_u = create_buffer_u!(NUM_INPUTS);
/// let gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
/// let gravity_Q = create_buffer_Q!(NUM_INPUTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_INPUTS: usize = 1;
/// let gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS, f64);
/// let gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_B {
    ( $num_states:expr, $num_inputs:expr ) => {
        (create_buffer_B![$num_states, $num_inputs, f32])
    };
    ( $num_states:expr, $num_inputs:expr, $t:ty ) => {{
        create_buffer_B!(
            $num_states,
            $num_inputs,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_states:expr, $num_inputs:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_B!($num_states, $num_inputs);
        $crate::buffer_types::InputMatrixMutBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_inputs,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_INPUTS: usize = 1;
/// assert_eq!(size_buffer_Q!(NUM_INPUTS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_Q {
    ( $num_inputs:expr ) => {{
        use $crate::FastUInt16;

        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        (NUM_INPUTS_ * NUM_INPUTS_) as usize
    }};
}

/// Creates a buffer fitting the input covariance matrix (`num_inputs` × `num_inputs`).
///
/// ## Arguments
/// * `num_inputs` - The number of states describing the system.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Input buffers.
/// let gravity_u = create_buffer_u!(NUM_INPUTS);
/// let gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
/// let gravity_Q = create_buffer_Q!(NUM_INPUTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_INPUTS: usize = 1;
/// let gravity_Q = create_buffer_Q!(NUM_INPUTS, f64);
/// let gravity_Q = create_buffer_Q!(NUM_INPUTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_Q {
    ( $num_inputs:expr ) => {
        (create_buffer_Q![$num_inputs, f32])
    };
    ( $num_inputs:expr, $t:ty ) => {{
        create_buffer_Q!($num_inputs, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_inputs:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_Q!($num_inputs);
        $crate::buffer_types::InputCovarianceMatrixMutBuffer::new($crate::MatrixData::new_owned::<
            $num_inputs,
            $num_inputs,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_MEASUREMENTS: usize = 1;
/// assert_eq!(size_buffer_z!(NUM_MEASUREMENTS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_z {
    ( $num_measurements:expr ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_MEASUREMENTS_ * 1) as usize
    }};
}

/// Creates a buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS, f64);
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_z {
    ( $num_measurements:expr ) => {
        (create_buffer_z![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        create_buffer_z!(
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_z!($num_measurements);
        $crate::buffer_types::MeasurementVectorBuffer::new($crate::MatrixData::new_owned::<
            $num_measurements,
            1,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
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
/// assert_eq!(size_buffer_H!(NUM_MEASUREMENTS, NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_H {
    ( $num_measurements:expr, $num_states:expr ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_MEASUREMENTS_ * NUM_STATES_) as usize
    }};
}

/// Creates a buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES, f64);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_H {
    ( $num_measurements:expr, $num_states:expr ) => {
        (create_buffer_H![$num_measurements, $num_states, f32])
    };
    ( $num_measurements:expr, $num_states:expr, $t:ty ) => {{
        create_buffer_H!(
            $num_measurements,
            $num_states,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_H!($num_measurements, $num_states);
        $crate::buffer_types::MeasurementTransformationMatrixMutBuffer::new(
            $crate::MatrixData::new_owned::<$num_measurements, $num_states, COUNT, $t>(
                [($value) as $t; COUNT],
            ),
        )
    }};
}

/// Sizes a buffer fitting the measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_MEASUREMENTS: usize = 1;
/// assert_eq!(size_buffer_R!(NUM_MEASUREMENTS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_R {
    ( $num_measurements:expr ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize
    }};
}

/// Creates a buffer fitting the measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS, f64);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_R {
    ( $num_measurements:expr ) => {
        (create_buffer_R![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        create_buffer_R!(
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_R!($num_measurements);
        $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer::new(
            $crate::MatrixData::new_owned::<$num_measurements, $num_measurements, COUNT, $t>(
                [($value) as $t; COUNT],
            ),
        )
    }};
}

/// Sizes a buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_MEASUREMENTS: usize = 1;
/// assert_eq!(size_buffer_y!(NUM_MEASUREMENTS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_y {
    ( $num_measurements:expr ) => {{
        use $crate::FastUInt16;

        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_MEASUREMENTS_ * 1) as usize
    }};
}

/// Creates a buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS, f64);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_y {
    ( $num_measurements:expr ) => {
        (create_buffer_y![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        create_buffer_y!(
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_y!($num_measurements);
        $crate::buffer_types::InnovationVectorBuffer::new($crate::MatrixData::new_owned::<
            $num_measurements,
            1,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the innovation covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_MEASUREMENTS: usize = 1;
/// assert_eq!(size_buffer_S!(NUM_MEASUREMENTS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_S {
    ( $num_measurements:expr ) => {{
        use $crate::FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize
    }};
}

/// Creates a buffer fitting the innovation covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS, f64);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_S {
    ( $num_measurements:expr ) => {
        (create_buffer_S![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        create_buffer_S!(
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_S!($num_measurements);
        $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer::new(
            $crate::MatrixData::new_owned::<$num_measurements, $num_measurements, COUNT, $t>(
                [($value) as $t; COUNT],
            ),
        )
    }};
}

/// Sizes a buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
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
/// assert_eq!(size_buffer_K!(NUM_STATES, NUM_MEASUREMENTS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_K {
    ( $num_states:expr, $num_measurements:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_STATES_ * NUM_MEASUREMENTS_) as usize
    }};
}

/// Creates a buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement buffers.
/// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
/// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
/// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
/// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS, f64);
/// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_K {
    ( $num_states:expr, $num_measurements:expr ) => {
        (create_buffer_K![$num_states, $num_measurements, f32])
    };
    ( $num_states:expr, $num_measurements:expr, $t:ty ) => {{
        create_buffer_K!(
            $num_states,
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_states:expr, $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_K!($num_states, $num_measurements);
        $crate::buffer_types::KalmanGainMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_measurements,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_temp_x!(NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_x {
    ( $num_states:expr ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * 1) as usize
    }};
}

/// Creates a buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Filter temporaries.
/// let gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
/// let gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
/// let gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_INPUTS: usize = 1;
/// let gravity_temp_x = create_buffer_temp_x!(NUM_STATES, f64);
/// let gravity_temp_x = create_buffer_temp_x!(NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_x {
    ( $num_states:expr ) => {
        (create_buffer_temp_x![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        create_buffer_temp_x!($num_states, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_x!($num_states);
        $crate::buffer_types::StatePredictionVectorBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            1,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the temporary P matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_temp_P!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_P {
    ( $num_states:expr ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Creates a buffer fitting the temporary P matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Filter temporaries.
/// let gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
/// let gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
/// let gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_INPUTS: usize = 1;
/// let gravity_temp_P = create_buffer_temp_P!(NUM_STATES, f64);
/// let gravity_temp_P = create_buffer_temp_P!(NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_P {
    ( $num_states:expr ) => {
        (create_buffer_temp_P![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        create_buffer_temp_P!($num_states, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_P!($num_states);
        $crate::buffer_types::TemporaryStateMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_states,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the temporary B×Q matrix (`num_states` × `num_inputs`).
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
/// assert_eq!(size_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_BQ {
    ( $num_states:expr, $num_inputs:expr ) => {{
        use $crate::FastUInt16;

        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_INPUTS_: FastUInt16 = ($num_inputs) as FastUInt16;
        (NUM_STATES_ * NUM_INPUTS_) as usize
    }};
}

/// Creates a buffer fitting the temporary B×Q matrix (`num_states` × `num_inputs`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_inputs` - The number of inputs.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 1;
///
/// // Filter temporaries.
/// let gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
/// let gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
/// let gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_INPUTS: usize = 1;
/// let gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS, f64);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_BQ {
    ( $num_states:expr, $num_inputs:expr ) => {
        (create_buffer_temp_BQ![$num_states, $num_inputs, f32])
    };
    ( $num_states:expr, $num_inputs:expr, $t:ty ) => {{
        create_buffer_temp_BQ!(
            $num_states,
            $num_inputs,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_states:expr, $num_inputs:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_BQ!($num_states, $num_inputs);
        $crate::buffer_types::TemporaryBQMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_inputs,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_MEASUREMENTS: usize = 1;
/// assert_eq!(size_buffer_temp_S_inv!(NUM_MEASUREMENTS), 1);
/// ```
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_S_inv {
    ( $num_measurements:expr ) => {{
        use $crate::FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_MEASUREMENTS_ * NUM_MEASUREMENTS_) as usize
    }};
}

/// Creates a buffer fitting the temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS, f64);
/// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS, f64, 0.0);
/// ```
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_S_inv {
    ( $num_measurements:expr ) => {
        (create_buffer_temp_S_inv![$num_measurements, f32])
    };
    ( $num_measurements:expr, $t:ty ) => {{
        create_buffer_temp_S_inv!(
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_S_inv!($num_measurements);
        $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer::new(
            $crate::MatrixData::new_owned::<$num_measurements, $num_measurements, COUNT, $t>(
                [($value) as $t; COUNT],
            ),
        )
    }};
}

/// Sizes a buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
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
/// assert_eq!(size_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_HP {
    ( $num_measurements:expr, $num_states:expr) => {{
        use $crate::FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * NUM_MEASUREMENTS_) as usize
    }};
}

/// Creates a buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES, f64);
/// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES, f64, 0.0);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_HP {
    ( $num_measurements:expr, $num_states:expr ) => {
        (create_buffer_temp_HP![$num_measurements, $num_states, f32])
    };
    ( $num_measurements:expr, $num_states:expr, $t:ty ) => {{
        create_buffer_temp_HP!(
            $num_measurements,
            $num_states,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_measurements:expr, $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_HP!($num_measurements, $num_states);
        $crate::buffer_types::TemporaryHPMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_measurements,
            $num_states,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the temporary P×H^-1 buffer (`num_states` × `num_measurements`).
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
/// assert_eq!(size_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_PHt {
    ( $num_states:expr, $num_measurements:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        const NUM_MEASUREMENTS_: FastUInt16 = ($num_measurements) as FastUInt16;
        (NUM_STATES_ * NUM_MEASUREMENTS_) as usize
    }};
}

/// Creates a buffer fitting the temporary P×H^-1 buffer (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS, f64);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_PHt {
    ( $num_states:expr, $num_measurements:expr ) => {
        (create_buffer_temp_PHt![$num_states, $num_measurements, f32])
    };
    ( $num_states:expr, $num_measurements:expr, $t:ty ) => {{
        create_buffer_temp_PHt!(
            $num_states,
            $num_measurements,
            $t,
            <$t as $crate::num_traits::ConstZero>::ZERO
        )
    }};
    ( $num_states:expr, $num_measurements:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_PHt!($num_states, $num_measurements);
        $crate::buffer_types::TemporaryPHTMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_measurements,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}

/// Sizes a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_temp_KHP!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_KHP {
    ( $num_states:expr ) => {{
        use $crate::FastUInt16;
        const NUM_STATES_: FastUInt16 = ($num_states) as FastUInt16;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Creates a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - Optional: The data type. Defaults to `f32`.
/// * `value` - Optional: The constant value to initialize the field with. Defaults to [`ZERO`](num_traits::ConstZero::ZERO).
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 1;
///
/// // Measurement temporaries.
/// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
/// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
/// let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
/// let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
/// ```
///
/// or with explicit types:
///
/// ```
/// # use minikalman::*;
/// # const NUM_STATES: usize = 3;
/// # const NUM_MEASUREMENTS: usize = 1;
/// let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES, f64);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! create_buffer_temp_KHP {
    ( $num_states:expr ) => {
        (create_buffer_temp_KHP![$num_states, f32])
    };
    ( $num_states:expr, $t:ty ) => {{
        create_buffer_temp_KHP!($num_states, $t, <$t as $crate::num_traits::ConstZero>::ZERO)
    }};
    ( $num_states:expr, $t:ty, $value:expr ) => {{
        const COUNT: usize = $crate::size_buffer_temp_KHP!($num_states);
        $crate::buffer_types::TemporaryKHPMatrixBuffer::new($crate::MatrixData::new_owned::<
            $num_states,
            $num_states,
            COUNT,
            $t,
        >([($value) as $t; COUNT]))
    }};
}
