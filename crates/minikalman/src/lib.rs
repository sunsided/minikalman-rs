//! # Kalman Filters for Embedded Targets
//!
//! This is the Rust port of the [kalman-clib](https://github.com/sunsided/kalman-clib/) library,
//! a microcontroller targeted Kalman filter implementation. It can use [`micromath`](https://docs.rs/micromath)
//! for square root and reciprocal calculations on `no_std`; [`libm`](https://docs.rs/libm) is supported as well.
//!
//! This implementation uses statically allocated buffers for all matrix operations. Due to lack
//! of `const` generics for array allocations in Rust, this crate also provides helper macros
//! to create the required arrays (see e.g. [`impl_buffer_A`]).
//!
//! If allocation is available (via `std` or `alloc` crate features), the [`KalmanFilterBuilder`](builder::KalmanFilterBuilder) can be
//! used to quickly create a [`Kalman`] filter instance with all necessary buffers, alongside
//! [`Control`] and [`Observation`] instances.
//!
//! ## Crate Features
//!
//! * `std` - Disabled by default. Disables the `no_std` configuration attribute (enabling `std` support).
//! * `alloc` - Enables allocation support for builder types.
//! * `nalgebra` - Enables [nalgebra](https://crates.io/crates/nalgebra) support.
//! * `libm` - Enables [libm](https://crates.io/crates/libm) support.
//! * `micromath` - Enables [micromath](https://crates.io/crates/micromath) support.
//! * `fixed` - Enables fixed-point support via the [fixed](https://crates.io/crates/fixed) crate.
//! * `unsafe` - Enables some unsafe pointer operations. Disabled by default; when turned off,
//!              compiles the crate as `#![forbid(unsafe)]`.
//!
//! ## Example
//!
//! On `std` or `alloc` crates, the [`KalmanFilterBuilder`](builder::KalmanFilterBuilder) is enabled. An overly simplified example
//! for setting up and operating the Kalman Filter could look like this:
//!
//! ```no_run
//! use minikalman::builder::KalmanFilterBuilder;
//! use minikalman::prelude::MatrixMut;
//!
//! const NUM_STATES: usize = 3;
//! const NUM_CONTROLS: usize = 2;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
//! let mut filter = builder.build();
//! let mut control = builder.controls().build::<NUM_CONTROLS>();
//! let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
//!
//! // Set up the system dynamics, control matrices, observation matrices, ...
//!
//! // Filter!
//! loop {
//!     // Update your control vector(s).
//!     control.control_vector_mut().apply(|u| {
//!         u[0] = 0.0;
//!         u[1] = 1.0;
//!     });
//!
//!     // Update your measurement vectors.
//!     measurement.measurement_vector_mut().apply(|z| {
//!         z[0] = 42.0;
//!     });
//!
//!     // Update prediction (without controls).
//!     filter.predict();
//!
//!     // Apply any controls to the prediction.
//!     filter.control(&mut control);
//!
//!     // Apply any measurements.
//!     filter.correct(&mut measurement);
//!
//!     // Access the state
//!     let state = filter.state_vector();
//!     let covariance = filter.estimate_covariance();
//! }
//! ```
//!
//! ## Extended Kalman Filters
//!
//! The general setup remains the same, however the `predict` and `correct` methods are
//! replaced with their nonlinear counterparts:
//!
//! ```no_run
//! use minikalman::builder::KalmanFilterBuilder;
//! use minikalman::prelude::MatrixMut;
//!
//! const NUM_STATES: usize = 3;
//! const NUM_CONTROLS: usize = 2;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
//! let mut filter = builder.build();
//! let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
//!
//! // Set up the system dynamics, control matrices, observation matrices, ...
//!
//! // Filter!
//! loop {
//!     // Obtain the control values.
//!     let control_value = 1.0;
//!
//!     // Update prediction using nonlinear transfer function.
//!     filter.predict_nonlinear(|current, next| {
//!         next[0] = current[0] * current[0];
//!         next[1] = current[1].sin() * control_value;
//!     });
//!
//!     // Update your measurement vectors.
//!     measurement.measurement_vector_mut().apply(|z| {
//!         z[0] = 42.0;
//!     });
//!
//!     // Apply any measurements using a nonlinear measurement function.
//!     filter.correct_nonlinear(&mut measurement, |state, observation| {
//!         observation[0] = state[0].cos() + state[1].sin();
//!     });
//!
//!     // Access the state
//!     let state = filter.state_vector();
//!     let covariance = filter.estimate_covariance();
//! }
//! ```
//!
//! ## `no_std` Example
//!
//! Systems without the liberty of heap allocation may make use of the provided helper macros
//! to wire up new types. This comes at the cost of potentially confusing IDEs due to recursive
//! macro expansion, so buyer beware. In the example below, types are set up as arrays bound
//! to `static mut` variables.
//!
//! ```
//! # #![allow(non_snake_case)]
//! # #![allow(non_upper_case_globals)]
//! use minikalman::buffers::types::*;
//! use minikalman::prelude::*;
//!
//! const NUM_STATES: usize = 3;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! // System buffers.
//! impl_buffer_x!(static mut gravity_x, NUM_STATES, f32, 0.0);
//! impl_buffer_A!(static mut gravity_A, NUM_STATES, f32, 0.0);
//! impl_buffer_P!(static mut gravity_P, NUM_STATES, f32, 0.0);
//!
//! // Observation buffers.
//! impl_buffer_Q!(static mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_H!(static mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
//! impl_buffer_R!(static mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_y!(static mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_S!(static mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_K!(static mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
//!
//! // Filter temporaries.
//! impl_buffer_temp_x!(static mut gravity_temp_x, NUM_STATES, f32, 0.0);
//! impl_buffer_temp_P!(static mut gravity_temp_P, NUM_STATES, f32, 0.0);
//!
//! // Observation temporaries.
//! impl_buffer_temp_S_inv!(static mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
//!
//! // Observation temporaries.
//! impl_buffer_temp_HP!(static mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
//! impl_buffer_temp_PHt!(static mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_temp_KHP!(static mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
//!
//! let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
//!     StateTransitionMatrixMutBuffer::from(unsafe { gravity_A.as_mut_slice() }),
//!     StateVectorBuffer::from(unsafe { gravity_x.as_mut_slice() }),
//!     EstimateCovarianceMatrixBuffer::from(unsafe { gravity_P.as_mut_slice() }),
//!     PredictedStateEstimateVectorBuffer::from(unsafe { gravity_temp_x.as_mut_slice() }),
//!     TemporaryStateMatrixBuffer::from(unsafe { gravity_temp_P.as_mut_slice() }),
//! );
//!
//! let mut measurement = ObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
//!     ObservationMatrixMutBuffer::from(unsafe { gravity_H.as_mut_slice() }),
//!     MeasurementVectorBuffer::from(unsafe { gravity_z.as_mut_slice() }),
//!     MeasurementNoiseCovarianceMatrixBuffer::from(unsafe { gravity_R.as_mut_slice() }),
//!     InnovationVectorBuffer::from(unsafe { gravity_y.as_mut_slice() }),
//!     InnovationCovarianceMatrixBuffer::from(unsafe { gravity_S.as_mut_slice() }),
//!     KalmanGainMatrixBuffer::from(unsafe { gravity_K.as_mut_slice() }),
//!     TemporaryResidualCovarianceInvertedMatrixBuffer::from(unsafe {
//!         gravity_temp_S_inv.as_mut_slice()
//!     }),
//!     TemporaryHPMatrixBuffer::from(unsafe { gravity_temp_HP.as_mut_slice() }),
//!     TemporaryPHTMatrixBuffer::from(unsafe { gravity_temp_PHt.as_mut_slice() }),
//!     TemporaryKHPMatrixBuffer::from(unsafe { gravity_temp_KHP.as_mut_slice() }),
//! );
//! ```
//!
//! After that, the `filter` and `measurement` variables can be used similar to the example above.

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
// Enable `no_std` if the `no_std` crate feature is enabled.
#![cfg_attr(not(feature = "std"), no_std)]
// Forbid unsafe code unless the `unsafe` crate feature is explicitly enabled.
#![cfg_attr(not(feature = "unsafe"), forbid(unsafe_code))]
// Attempt to disable allocations.
#![cfg_attr(not(feature = "alloc"), forbid(box_pointers))]

#[cfg(any(feature = "std", feature = "alloc"))]
extern crate alloc;
extern crate core;

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
mod kalman_builder;

pub mod buffers;
mod controls;
mod kalman;
pub mod matrix;
mod observations;
mod static_macros;

#[cfg(test)]
mod test_dummies;

#[cfg(all(test, feature = "alloc"))]
mod test_filter;

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
pub use crate::buffers::builder::BufferBuilder;
pub use crate::controls::{Control, ControlBuilder};
pub use crate::kalman::{Kalman, KalmanBuilder};
pub use crate::observations::{Observation, ObservationBuilder};

/// Re-export `num_traits`.
pub use num_traits;

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
pub mod builder {
    pub use crate::kalman_builder::{
        KalmanFilterBuilder, KalmanFilterControlBuilder, KalmanFilterObservationBuilder,
    };
}

/// Exports all macros and common types.
pub mod prelude {
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub use crate::buffers::builder::*;

    pub use crate::controls::{Control, ControlBuilder};
    pub use crate::kalman::{Kalman, KalmanBuilder};
    pub use crate::observations::{Observation, ObservationBuilder};

    pub use crate::kalman::*;
    pub use crate::matrix::*;

    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub use crate::kalman_builder::{
        KalmanFilterControlType, KalmanFilterObservationType, KalmanFilterType,
    };

    pub use crate::{
        impl_buffer_A, impl_buffer_B, impl_buffer_H, impl_buffer_K, impl_buffer_P, impl_buffer_Q,
        impl_buffer_R, impl_buffer_S, impl_buffer_temp_BQ, impl_buffer_temp_HP,
        impl_buffer_temp_KHP, impl_buffer_temp_P, impl_buffer_temp_PHt, impl_buffer_temp_S_inv,
        impl_buffer_temp_x, impl_buffer_u, impl_buffer_x, impl_buffer_y, impl_buffer_z,
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
    pub use fixed::types::{I16F16, I32F32};
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
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
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
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
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
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the control vector (`num_controls` × `1`).
///
/// ## Arguments
/// * `num_controls` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_u!(NUM_CONTROLS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_u {
    ( $num_controls:expr ) => {{
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_CONTROLS_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the control transition matrix (`num_states` × `num_controls`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_controls` - The number of controls.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_B!(NUM_STATES, NUM_CONTROLS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_B {
    ( $num_states:expr, $num_controls:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_STATES_ * NUM_CONTROLS_) as usize
    }};
}

/// Sizes a buffer fitting the control covariance matrix (`num_controls` × `num_controls`).
///
/// ## Arguments
/// * `num_controls` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_Q!(NUM_CONTROLS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_Q {
    ( $num_controls:expr ) => {{
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_CONTROLS_ * NUM_CONTROLS_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_z!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_z {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * 1) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_H!(NUM_OBSERVATIONS, NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_H {
    ( $num_measurements:expr, $num_states:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_OBSERVATIONS_ * NUM_STATES_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_R!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_R {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * NUM_OBSERVATIONS_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_y!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_y {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * 1) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_S!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_S {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * NUM_OBSERVATIONS_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_K!(NUM_STATES, NUM_OBSERVATIONS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_K {
    ( $num_states:expr, $num_measurements:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_STATES_ * NUM_OBSERVATIONS_) as usize
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
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * 1) as usize
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
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the temporary B×Q matrix (`num_states` × `num_controls`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_controls` - The number of controls.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_temp_BQ!(NUM_STATES, NUM_CONTROLS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_BQ {
    ( $num_states:expr, $num_controls:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_STATES_ * NUM_CONTROLS_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_temp_S_inv!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_S_inv {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * NUM_OBSERVATIONS_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_temp_HP!(NUM_OBSERVATIONS, NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_HP {
    ( $num_measurements:expr, $num_states:expr) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_OBSERVATIONS_) as usize
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
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_temp_PHt!(NUM_STATES, NUM_OBSERVATIONS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_PHt {
    ( $num_states:expr, $num_measurements:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_STATES_ * NUM_OBSERVATIONS_) as usize
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
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}
