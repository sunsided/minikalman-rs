/// Creates a static buffer fitting the state vector x (`num_states` × `1`).
///
/// This will create a [`StateVectorBuffer`](crate::buffers::types::StateVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This vector represents the state estimate. It contains the predicted values of the system's
/// state variables at a given time step. The state vector \( x \) is updated at each time step
/// based on the system dynamics, control inputs, and measurements. It provides the best estimate
/// of the current state of the system, combining prior knowledge with new information from
/// observations to minimize the estimation error.
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_x!(static mut X, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(X.len(), 3);
///     assert_eq!(X[0], 0.0_f32);
/// }
/// ```
///
/// Or create it as a regular variable:
///
/// ```
/// # use minikalman::prelude::*;
/// # const NUM_STATES: usize = 3;
/// impl_buffer_x!(x, NUM_STATES, f32, 0.0);
/// impl_buffer_x!(mut x, NUM_STATES, f32, 0.0);
/// impl_buffer_x!(let mut x, NUM_STATES, f32, 0.0);
///
/// assert_eq!(x.len(), 3);
/// assert_eq!(x[0], 0.0_f32);
/// ```
#[macro_export]
macro_rules! impl_buffer_x {
    (mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_x!($vec_name, $num_states, $t, $init, let mut)
    };
    ($vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_x!($vec_name, $num_states, $t, $init, let)
    };
    (let mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_x!($vec_name, $num_states, $t, $init, let mut)
    };
    (let $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_x!($vec_name, $num_states, $t, $init, let)
    };
    (static mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_x!($vec_name, $num_states, $t, $init, static mut)
    };
    (static $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_x!($vec_name, $num_states, $t, $init, static)
    };
    ($vec_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $vec_name: $crate::buffers::types::StateVectorBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffers::types::StateVectorBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square state transition matrix A (`num_states` × `num_states`).
///
/// This will create a [`SystemMatrixMutBuffer`](crate::buffers::types::StateTransitionMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Regular Kalman Filters
/// This matrix represents the state transition model. It defines how the state
/// evolves from one time step to the next in the absence of process noise and control inputs.
/// The matrix \( A \) is used to predict the next state based on the current state,
/// encapsulating the system dynamics and their influence on state progression.
///
/// ## Extended Kalman Filters
/// This matrix represents the state transition model in the context of the Extended Kalman Filter (EKF).
/// It defines how the state evolves from one time step to the next in the absence of process noise and control inputs.
/// In the EKF, the matrix \( A \) is the Jacobian of the state transition function with respect to the state,
/// evaluated at the current state estimate. This Jacobian matrix linearizes the non-linear state transition
/// function around the current estimate, allowing the EKF to predict the next state based on the current state
/// while accounting for the non-linear dynamics of the system.
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_A!(static mut A, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(A.len(), 9);
///     assert_eq!(A[0], 0.0_f32);
/// }
/// ```
///
/// Or create it as a regular variable:
///
/// ```
/// # use minikalman::prelude::*;
/// # const NUM_STATES: usize = 3;
/// impl_buffer_A!(let mut A, NUM_STATES, f32, 0.0);
///
/// assert_eq!(A.len(), 9);
/// assert_eq!(A[0], 0.0_f32);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_A {
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_A!($mat_name, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_A!($mat_name, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_A!($mat_name, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_A!($mat_name, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_A!($mat_name, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_A!($mat_name, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::StateTransitionMatrixMutBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffers::types::StateTransitionMatrixMutBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the square estimate covariance matrix P (`num_states` × `num_states`).
///
/// This will create a [`EstimateCovarianceMatrixBuffer`](crate::buffers::types::EstimateCovarianceMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the estimate covariance. It quantifies the uncertainty in
/// the state estimate, providing a measure of how much the state estimate is expected
/// to vary. This matrix offers a measure of confidence in the estimate by indicating
/// the degree of variability and uncertainty associated with the predicted state.
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_P!(static mut P, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(P.len(), 9);
///     assert_eq!(P[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_P {
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_P!($mat_name, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_P!($mat_name, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_P!($mat_name, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_P!($mat_name, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_P!($mat_name, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_P!($mat_name, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::EstimateCovarianceMatrixBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffers::types::EstimateCovarianceMatrixBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the square direct process noise covariance matrix Q (`num_states` × `num_states`).
///
/// This will create a [`DirectProcessNoiseCovarianceMatrixMutBuffer`](crate::buffers::types::DirectProcessNoiseCovarianceMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the direct process noise covariance. It quantifies the
/// uncertainty introduced by inherent system dynamics and external disturbances,
/// providing a measure of how much the true state is expected to deviate from the
/// predicted state due to these process variations.
///
/// ## Arguments
/// * `num_controls` - The number of controls to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_CONTROLS: usize = 2;
/// impl_buffer_Q_direct!(static mut Q, NUM_CONTROLS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Q.len(), 4);
///     assert_eq!(Q[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_Q_direct {
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_direct!($mat_name, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_direct!($mat_name, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_direct!($mat_name, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_direct!($mat_name, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_direct!($mat_name, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_direct!($mat_name, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::DirectProcessNoiseCovarianceMatrixMutBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffers::types::DirectProcessNoiseCovarianceMatrixMutBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Sizes a static buffer fitting the control vector u (`num_controls` × `1`).
///
/// This will create a [`ControlVectorBuffer`](crate::buffers::types::ControlVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This vector represents the control input. It contains the values of the external inputs
/// applied to the system at a given time step. The control vector \( u \) influences the state
/// transition, allowing the Kalman Filter to account for known control actions when predicting
/// the next state. By incorporating the effects of these control inputs, the filter provides
/// a more accurate and realistic estimate of the system's state.
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_CONTROLS: usize = 2;
/// impl_buffer_u!(static mut U, NUM_CONTROLS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(U.len(), 2);
///     assert_eq!(U[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_buffer_u {
    (mut $vec_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_controls, $t, $init, let mut)
    };
    ($vec_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_controls, $t, $init, let)
    };
    (let mut $vec_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_controls, $t, $init, let mut)
    };
    (let $vec_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_controls, $t, $init, let)
    };
    (static mut $vec_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_controls, $t, $init, static mut)
    };
    (static $vec_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_controls, $t, $init, static)
    };
    ($vec_name:ident, $num_controls:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $vec_name: $crate::buffers::types::ControlVectorBuffer<
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_controls, 1, { $num_controls * 1 }, $t>,
        > = $crate::buffers::types::ControlVectorBuffer::<
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_controls, 1, { $num_controls * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_controls * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the control matrix B (`num_states` × `num_controls`).
///
/// This will create a [`ControlMatrixMutBuffer`](crate::buffers::types::ControlMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the control input model. It defines how the control inputs
/// influence the state evolution from one time step to the next. The matrix \( B \)
/// is used to incorporate the effect of control inputs into the state transition,
/// allowing the model to account for external controls applied to the system.
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_controls` - The number of controls to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_CONTROLS: usize = 2;
/// impl_buffer_B!(static mut B, NUM_STATES, NUM_CONTROLS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(B.len(), 6);
///     assert_eq!(B[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_B {
    (mut $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_controls, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_controls, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_controls, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_controls, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_controls, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_controls, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::ControlMatrixMutBuffer<
            $num_states,
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_controls, { $num_states * $num_controls }, $t>,
        > = $crate::buffers::types::ControlMatrixMutBuffer::<
            $num_states,
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_controls, { $num_states * $num_controls }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_controls }],
        ));
    };
}

/// Creates a static buffer fitting the square control process noise covariance matrix Q (`num_controls` × `num_controls`).
///
/// This will create a [`ProcessNoiseCovarianceMatrixMutBuffer`](crate::buffers::types::ControlProcessNoiseCovarianceMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the control process noise covariance. It quantifies the
/// uncertainty introduced by the control inputs, reflecting how much the true state
/// is expected to deviate from the predicted state due to noise and variations
/// in the control process. The matrix is used as B×Q×Bᵀ, where B
/// represents the control input model, and Q is the process noise covariance (this matrix).
///
/// ## Arguments
/// * `num_controls` - The number of controls to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_CONTROLS: usize = 2;
/// impl_buffer_Q_control!(static mut Q, NUM_CONTROLS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Q.len(), 4);
///     assert_eq!(Q[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_Q_control {
    (mut $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_control!($mat_name, $num_controls, $t, $init, let mut)
    };
    ($mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_control!($mat_name, $num_controls, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_control!($mat_name, $num_controls, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_control!($mat_name, $num_controls, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_control!($mat_name, $num_controls, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q_control!($mat_name, $num_controls, $t, $init, static)
    };
    ($mat_name:ident, $num_controls:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::ControlProcessNoiseCovarianceMatrixMutBuffer<
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_controls, $num_controls, { $num_controls * $num_controls }, $t>,
        > = $crate::buffers::types::ControlProcessNoiseCovarianceMatrixMutBuffer::<
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_controls, $num_controls, { $num_controls * $num_controls }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_controls * $num_controls }],
        ));
    };
}

/// Creates a static buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// This will create a [`ObservationVectorBuffer`](crate::buffers::types::MeasurementVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_z!(static mut Z, NUM_OBSERVATIONS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Z.len(), 5);
///     assert_eq!(Z[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_buffer_z {
    (mut $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_z!($vec_name, $num_measurements, $t, $init, let mut)
    };
    ($vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_z!($vec_name, $num_measurements, $t, $init, let)
    };
    (let mut $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_z!($vec_name, $num_measurements, $t, $init, let mut)
    };
    (let $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_z!($vec_name, $num_measurements, $t, $init, let)
    };
    (static mut $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_z!($vec_name, $num_measurements, $t, $init, static mut)
    };
    (static $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_z!($vec_name, $num_measurements, $t, $init, static)
    };
    ($vec_name:ident, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $vec_name: $crate::buffers::types::MeasurementVectorBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffers::types::MeasurementVectorBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the observation matrix H (`num_measurements` × `num_states`).
///
/// This will create a [`ObservationMatrixMutBuffer`](crate::buffers::types::ObservationMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Regular Kalman Filters
/// This matrix represents the observation model. It defines the relationship between
/// the state and the measurements obtained from the system. The matrix \( H \) is used
/// to map the predicted state into the measurement space, allowing the Kalman filter
/// to compare the predicted measurements with the actual measurements for updating the state estimate.
///
/// ## Extended Kalman Filters
/// This matrix represents the observation model in the context of the Extended Kalman Filter (EKF).
/// It defines the relationship between the state and the measurements obtained from the system.
/// In the EKF, the matrix \( H \) is the Jacobian of the measurement function with respect to the state,
/// evaluated at the current state estimate. This Jacobian matrix linearizes the non-linear measurement
/// function around the current estimate, allowing the EKF to map the predicted state into the measurement
/// space for comparison with the actual measurements during the update step.
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states of the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_H!(static mut H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(H.len(), 15);
///     assert_eq!(H[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_H {
    (mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_H!($mat_name, $num_measurements, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_H!($mat_name, $num_measurements, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_H!($mat_name, $num_measurements, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_H!($mat_name, $num_measurements, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_H!($mat_name, $num_measurements, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_H!($mat_name, $num_measurements, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::ObservationMatrixMutBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffers::types::ObservationMatrixMutBuffer::<
            $num_measurements,
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the square measurement noise covariance matrix (`num_measurements` × `num_measurements`).
///
/// This will create a [`MeasurementNoiseCovarianceMatrixBuffer`](crate::buffers::types::MeasurementNoiseCovarianceMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the measurement noise covariance. It quantifies the uncertainty
/// associated with the measurements obtained from the system. The matrix \( R \) provides
/// a measure of the expected variability in the measurement noise, reflecting the accuracy
/// and reliability of the sensor or measurement device. This matrix is used in the Kalman
/// filter to weigh the influence of the actual measurements during the update step,
/// balancing it against the predicted state estimate.
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_R!(static mut R, NUM_OBSERVATIONS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(R.len(), 25);
///     assert_eq!(R[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_R {
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_R!($mat_name, $num_measurements, $t, $init, let mut)
    };
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_R!($mat_name, $num_measurements, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_R!($mat_name, $num_measurements, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_R!($mat_name, $num_measurements, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_R!($mat_name, $num_measurements, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_R!($mat_name, $num_measurements, $t, $init, static)
    };
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::MeasurementNoiseCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffers::types::MeasurementNoiseCovarianceMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the innovation (or measurement residual) vector y (`num_measurements` × `1`).
///
/// This will create a [`InnovationVectorBuffer`](crate::buffers::types::InnovationVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This vector represents the innovation (or residual). It is the difference between the actual
/// measurement and the predicted measurement based on the current state estimate. The innovation
/// vector \( y \) quantifies the discrepancy between observed data and the filter's predictions,
/// providing a measure of the new information gained from the measurements. This vector is used
/// to update the state estimate, ensuring that the Kalman Filter corrects for any deviations
/// between the predicted and actual observations, thus refining the state estimation.
///
/// Some implementations may choose to use it as a temporary observation buffer, e.g. during
/// Extended Kalman Filter measurement updates.
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_y!(static mut Y, NUM_OBSERVATIONS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Y.len(), 5);
///     assert_eq!(Y[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_buffer_y {
    (mut $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_y!($vec_name, $num_measurements, $t, $init, let mut)
    };
    ($vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_y!($vec_name, $num_measurements, $t, $init, let)
    };
    (let mut $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_y!($vec_name, $num_measurements, $t, $init, let mut)
    };
    (let $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_y!($vec_name, $num_measurements, $t, $init, let)
    };
    (static mut $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_y!($vec_name, $num_measurements, $t, $init, static mut)
    };
    (static $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_y!($vec_name, $num_measurements, $t, $init, static)
    };
    ($vec_name:ident, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $vec_name: $crate::buffers::types::InnovationVectorBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffers::types::InnovationVectorBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square innovation (residual) covariance matrix S (`num_measurements` × `num_measurements`).
///
/// This will create a [`InnovationResidualCovarianceMatrixBuffer`](crate::buffers::types::InnovationCovarianceMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the innovation (residual) covariance. It quantifies the
/// uncertainty in the difference between the predicted measurement and the actual measurement.
/// The innovation covariance matrix provides a measure of how much the innovation (residual)
/// is expected to vary, reflecting the combined effects of measurement noise and
/// the uncertainty in the state estimate.
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_S!(static mut S, NUM_OBSERVATIONS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(S.len(), 25);
///     assert_eq!(S[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_S {
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_S!($mat_name, $num_measurements, $t, $init, let mut)
    };
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_S!($mat_name, $num_measurements, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_S!($mat_name, $num_measurements, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_S!($mat_name, $num_measurements, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_S!($mat_name, $num_measurements, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_S!($mat_name, $num_measurements, $t, $init, static)
    };
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::InnovationCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffers::types::InnovationCovarianceMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
///
/// This will create a [`KalmanGainMatrixBuffer`](crate::buffers::types::KalmanGainMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the Kalman gain. It determines how much the state estimate should
/// be adjusted based on the difference between the predicted measurements and the actual measurements.
/// The matrix \( K \) balances the uncertainty in the state estimate with the uncertainty in the
/// measurements, providing an optimal weight for incorporating new measurement information into the
/// state estimate. By minimizing the estimation error covariance, the Kalman gain ensures that the
/// updated state estimate is as accurate as possible given the available data.
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_K!(static mut K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(K.len(), 15);
///     assert_eq!(K[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_K {
    (mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_K!($mat_name, $num_states, $num_measurements, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_K!($mat_name, $num_states, $num_measurements, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_K!($mat_name, $num_states, $num_measurements, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_K!($mat_name, $num_states, $num_measurements, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_K!($mat_name, $num_states, $num_measurements, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_K!($mat_name, $num_states, $num_measurements, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::KalmanGainMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffers::types::KalmanGainMatrixBuffer::<
            $num_states,
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// This will create a [`StatePredictionVectorBuffer`](crate::buffers::types::PredictedStateEstimateVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_temp_x!(static mut TX, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TX.len(), 3);
///     assert_eq!(TX[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_buffer_temp_x {
    (mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_x!($vec_name, $num_states, $t, $init, let mut)
    };
    ($vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_x!($vec_name, $num_states, $t, $init, let)
    };
    (let mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_x!($vec_name, $num_states, $t, $init, let mut)
    };
    (let $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_x!($vec_name, $num_states, $t, $init, let)
    };
    (static mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_x!($vec_name, $num_states, $t, $init, static mut)
    };
    (static $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_x!($vec_name, $num_states, $t, $init, static)
    };
    ($vec_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $vec_name: $crate::buffers::types::PredictedStateEstimateVectorBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffers::types::PredictedStateEstimateVectorBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square temporary P matrix (`num_states` × `num_states`).
///
/// This will create a [`TemporaryStateMatrixBuffer`](crate::buffers::types::TemporaryStateMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_temp_P!(static mut TP, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TP.len(), 9);
///     assert_eq!(TP[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_P {
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_P!($mat_name, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_P!($mat_name, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_P!($mat_name, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_P!($mat_name, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_P!($mat_name, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_P!($mat_name, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::TemporaryStateMatrixBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffers::types::TemporaryStateMatrixBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the temporary B×Q matrix (`num_states` × `num_controls`).
///
/// This will create a [`TemporaryBQMatrixBuffer`](crate::buffers::types::TemporaryBQMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the product of the control input model and the process noise covariance, \( B \cdot Q \).
/// It quantifies the influence of the process noise on the state evolution when control inputs are applied.
/// The resulting matrix captures the combined effect of control input dynamics and inherent system noise,
/// providing an intermediate step in calculating the control process noise contribution to the state
/// covariance update. This product helps to incorporate the uncertainty due to control actions into the
/// overall state estimation process.
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_controls` - The number of controls to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_CONTROLS: usize = 2;
/// impl_buffer_temp_BQ!(static mut TBQ, NUM_STATES, NUM_CONTROLS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TBQ.len(), 6);
///     assert_eq!(TBQ[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_BQ {
    (mut $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_controls, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_controls, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_controls, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_controls, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_controls, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_controls, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $num_controls:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::TemporaryBQMatrixBuffer<
            $num_states,
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_controls, { $num_states * $num_controls }, $t>,
        > = $crate::buffers::types::TemporaryBQMatrixBuffer::<
            $num_states,
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_controls, { $num_states * $num_controls }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_controls }],
        ));
    };
}

/// Creates a static buffer fitting the square temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// This will create a [`TemporaryResidualCovarianceInvertedMatrixBuffer`](crate::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the inverse of the innovation (residual) covariance matrix, \( S^{-1} \).
/// It quantifies the weight given to the innovation (residual) in the update step of the Kalman Filter.
/// By inverting the innovation covariance matrix, \( S^{-1} \) provides a measure of the certainty
/// of the innovation, allowing the Kalman gain to optimally adjust the state estimate based on
/// the difference between the predicted and actual measurements. This inverse matrix ensures that
/// the filter accurately balances the contributions of the state prediction and the measurement
/// update in minimizing the overall estimation error.
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_temp_S_inv!(static mut TSINV, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TSINV.len(), 9);
///     assert_eq!(TSINV[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_S_inv {
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_S_inv!($mat_name, $num_measurements, $t, $init, let mut)
    };
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_S_inv!($mat_name, $num_measurements, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_S_inv!($mat_name, $num_measurements, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_S_inv!($mat_name, $num_measurements, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_S_inv!($mat_name, $num_measurements, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_S_inv!($mat_name, $num_measurements, $t, $init, static)
    };
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// This will create a [`TemporaryHPMatrixBuffer`](crate::buffers::types::TemporaryHPMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the product of the observation model and the estimate covariance, \( H×P \).
/// It quantifies how the uncertainty in the state estimate propagates into the measurement space.
/// The resulting matrix captures the influence of the current state uncertainty on the predicted
/// measurements, providing an intermediate step in calculating the innovation (residual) covariance matrix.
/// This product helps to incorporate the effects of state estimation uncertainty into the measurement
/// update process, ensuring that the Kalman Filter accurately adjusts the state estimate based on the
/// observed data.
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_temp_HP!(static mut THP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(THP.len(), 15);
///     assert_eq!(THP[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_HP {
    (mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_HP!($mat_name, $num_measurements, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_HP!($mat_name, $num_measurements, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_HP!($mat_name, $num_measurements, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_HP!($mat_name, $num_measurements, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_HP!($mat_name, $num_measurements, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_HP!($mat_name, $num_measurements, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::TemporaryHPMatrixBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffers::types::TemporaryHPMatrixBuffer::<
            $num_measurements,
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * $num_states }],
        ));
    };
}

/// Creates a buffer fitting the temporary P×Hᵀ buffer (`num_states` × `num_measurements`).
///
/// This will create a [`TemporaryPHTMatrixBuffer`](crate::buffers::types::TemporaryPHTMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the product of the estimate covariance and the transpose of the observation model, \( P×Hᵀ \).
/// It quantifies how the uncertainty in the state estimate influences the relationship between the state and the measurements.
/// The resulting matrix captures the effect of the current state uncertainty on the measurement update,
/// providing an intermediate step in calculating the Kalman gain. This product helps to incorporate the
/// variability of the state estimate into the measurement update process, ensuring that the Kalman Filter
/// accurately balances the contributions of the state prediction and the actual measurements in the state
/// estimation.
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 5;
/// impl_buffer_temp_PHt!(static mut TPHT, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TPHT.len(), 15);
///     assert_eq!(TPHT[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_PHt {
    (mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_PHt!($mat_name, $num_states, $num_measurements, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_PHt!($mat_name, $num_states, $num_measurements, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_PHt!($mat_name, $num_states, $num_measurements, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_PHt!($mat_name, $num_states, $num_measurements, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_PHt!($mat_name, $num_states, $num_measurements, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_PHt!($mat_name, $num_states, $num_measurements, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::TemporaryPHTMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffers::types::TemporaryPHTMatrixBuffer::<
            $num_states,
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_measurements }],
        ));
    };
}

/// Creates a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_states`).
///
/// This will create a [`TemporaryKHPMatrixBuffer`](crate::buffers::types::TemporaryKHPMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// This matrix represents the product of the Kalman gain and the product of the observation model and the estimate covariance, \( K×(H×P) \).
/// It quantifies the adjustment applied to the state estimate covariance during the measurement update step.
/// The resulting matrix captures the influence of the Kalman gain on the uncertainty of the state estimate,
/// providing an intermediate step in updating the estimate covariance matrix. This product helps to incorporate
/// the reduction in state estimation uncertainty achieved through the measurement update, ensuring that the Kalman Filter
/// accurately reflects the improved confidence in the state estimate after incorporating the observed measurements.
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_buffer_temp_KHP!(static mut TKHP, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TKHP.len(), 9);
///     assert_eq!(TKHP[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_KHP {
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_KHP!($mat_name, $num_states, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_KHP!($mat_name, $num_states, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_KHP!($mat_name, $num_states, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_KHP!($mat_name, $num_states, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_KHP!($mat_name, $num_states, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_KHP!($mat_name, $num_states, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::TemporaryKHPMatrixBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffers::types::TemporaryKHPMatrixBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}
