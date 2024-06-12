/// Creates a static buffer fitting the state vector x (`num_states` × `1`).
///
/// This will create a [`StateVectorBuffer`](crate::buffers::types::StateVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
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
/// This will create a [`SystemCovarianceMatrixBuffer`](crate::buffers::types::EstimateCovarianceMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
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

/// Sizes a static buffer fitting the control vector u (`num_controls` × `1`).
///
/// This will create a [`ControlVectorBuffer`](crate::buffers::types::ControlVectorBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
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

/// Creates a static buffer fitting the square process noise covariance matrix Q (`num_controls` × `num_controls`).
///
/// This will create a [`ProcessNoiseCovarianceMatrixMutBuffer`](crate::buffers::types::ProcessNoiseCovarianceMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
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
/// impl_buffer_Q!(static mut Q, NUM_CONTROLS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Q.len(), 4);
///     assert_eq!(Q[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_Q {
    (mut $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_controls, $t, $init, let mut)
    };
    ($mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_controls, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_controls, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_controls, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_controls, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_controls:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_controls, $t, $init, static)
    };
    ($mat_name:ident, $num_controls:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffers::types::ProcessNoiseCovarianceMatrixMutBuffer<
            $num_controls,
            $t,
            $crate::matrix::MatrixDataArray<$num_controls, $num_controls, { $num_controls * $num_controls }, $t>,
        > = $crate::buffers::types::ProcessNoiseCovarianceMatrixMutBuffer::<
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
/// This will create a [`ObservationTransformationMatrixMutBuffer`](crate::buffers::types::ObservationMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
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
