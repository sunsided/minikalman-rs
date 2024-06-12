/// Creates a static buffer fitting the state vector (`num_states` × `1`).
///
/// This will create a [`StateVectorBuffer`](crate::buffer_types::StateVectorBuffer)
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
        $($keywords)* $vec_name: $crate::buffer_types::StateVectorBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffer_types::StateVectorBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square state transition matrix (`num_states` × `num_states`).
///
/// This will create a [`SystemMatrixMutBuffer`](crate::buffer_types::SystemMatrixMutBuffer)
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
        $($keywords)* $mat_name: $crate::buffer_types::SystemMatrixMutBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::SystemMatrixMutBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the square state covariance matrix (`num_states` × `num_states`).
///
/// This will create a [`SystemCovarianceMatrixBuffer`](crate::buffer_types::SystemCovarianceMatrixBuffer)
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
        $($keywords)* $mat_name: $crate::buffer_types::SystemCovarianceMatrixBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::SystemCovarianceMatrixBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Sizes a static buffer fitting the input vector (`num_inputs` × `1`).
///
/// This will create a [`InputVectorBuffer`](crate::buffer_types::InputVectorBuffer)
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
/// const NUM_INPUTS: usize = 2;
/// impl_buffer_u!(static mut U, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(U.len(), 2);
///     assert_eq!(U[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_buffer_u {
    (mut $vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_inputs, $t, $init, let mut)
    };
    ($vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_inputs, $t, $init, let)
    };
    (let mut $vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_inputs, $t, $init, let mut)
    };
    (let $vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_inputs, $t, $init, let)
    };
    (static mut $vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_inputs, $t, $init, static mut)
    };
    (static $vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_u!($vec_name, $num_inputs, $t, $init, static)
    };
    ($vec_name:ident, $num_inputs:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $vec_name: $crate::buffer_types::InputVectorBuffer<
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_inputs, 1, { $num_inputs * 1 }, $t>,
        > = $crate::buffer_types::InputVectorBuffer::<
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_inputs, 1, { $num_inputs * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_inputs * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the input transition matrix (`num_states` × `num_inputs`).
///
/// This will create a [`InputMatrixMutBuffer`](crate::buffer_types::InputMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_inputs` - The number of inputs to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 2;
/// impl_buffer_B!(static mut B, NUM_STATES, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(B.len(), 6);
///     assert_eq!(B[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_B {
    (mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_inputs, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_inputs, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_inputs, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_inputs, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_inputs, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_B!($mat_name, $num_states, $num_inputs, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffer_types::InputMatrixMutBuffer<
            $num_states,
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        > = $crate::buffer_types::InputMatrixMutBuffer::<
            $num_states,
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_inputs }],
        ));
    };
}

/// Creates a static buffer fitting the square input covariance matrix (`num_inputs` × `num_inputs`).
///
/// This will create a [`InputCovarianceMatrixMutBuffer`](crate::buffer_types::InputCovarianceMatrixMutBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Arguments
/// * `num_inputs` - The number of inputs to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_INPUTS: usize = 2;
/// impl_buffer_Q!(static mut Q, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Q.len(), 4);
///     assert_eq!(Q[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_Q {
    (mut $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_inputs, $t, $init, let mut)
    };
    ($mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_inputs, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_inputs, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_inputs, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_inputs, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_Q!($mat_name, $num_inputs, $t, $init, static)
    };
    ($mat_name:ident, $num_inputs:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffer_types::InputCovarianceMatrixMutBuffer<
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_inputs, $num_inputs, { $num_inputs * $num_inputs }, $t>,
        > = $crate::buffer_types::InputCovarianceMatrixMutBuffer::<
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_inputs, $num_inputs, { $num_inputs * $num_inputs }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_inputs * $num_inputs }],
        ));
    };
}

/// Creates a static buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// This will create a [`MeasurementVectorBuffer`](crate::buffer_types::MeasurementVectorBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_z!(static mut Z, NUM_MEASUREMENTS, f32, 0.0);
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
        $($keywords)* $vec_name: $crate::buffer_types::MeasurementVectorBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffer_types::MeasurementVectorBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
///
/// This will create a [`MeasurementTransformationMatrixMutBuffer`](crate::buffer_types::MeasurementObservationMatrixMutBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_H!(static mut H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
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
        $($keywords)* $mat_name: $crate::buffer_types::MeasurementObservationMatrixMutBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffer_types::MeasurementObservationMatrixMutBuffer::<
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

/// Creates a static buffer fitting the square measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// This will create a [`MeasurementProcessNoiseCovarianceMatrixBuffer`](crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_R!(static mut R, NUM_MEASUREMENTS, f32, 0.0);
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
        $($keywords)* $mat_name: $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer::<
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

/// Creates a static buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// This will create a [`InnovationVectorBuffer`](crate::buffer_types::InnovationVectorBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_y!(static mut Y, NUM_MEASUREMENTS, f32, 0.0);
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
        $($keywords)* $vec_name: $crate::buffer_types::InnovationVectorBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffer_types::InnovationVectorBuffer::<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square innovation (residual) covariance matrix (`num_measurements` × `num_measurements`).
///
/// This will create a [`InnovationResidualCovarianceMatrixBuffer`](crate::buffer_types::InnovationResidualCovarianceMatrixBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_S!(static mut S, NUM_MEASUREMENTS, f32, 0.0);
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
        $($keywords)* $mat_name: $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer::<
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
/// This will create a [`KalmanGainMatrixBuffer`](crate::buffer_types::KalmanGainMatrixBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_K!(static mut K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
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
        $($keywords)* $mat_name: $crate::buffer_types::KalmanGainMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::KalmanGainMatrixBuffer::<
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
/// This will create a [`StatePredictionVectorBuffer`](crate::buffer_types::TemporaryStatePredictionVectorBuffer)
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
        $($keywords)* $vec_name: $crate::buffer_types::TemporaryStatePredictionVectorBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffer_types::TemporaryStatePredictionVectorBuffer::<
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
/// This will create a [`TemporaryStateMatrixBuffer`](crate::buffer_types::TemporaryStateMatrixBuffer)
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
        $($keywords)* $mat_name: $crate::buffer_types::TemporaryStateMatrixBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::TemporaryStateMatrixBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the temporary B×Q matrix (`num_states` × `num_inputs`).
///
/// This will create a [`TemporaryBQMatrixBuffer`](crate::buffer_types::TemporaryBQMatrixBuffer)
/// backed by a [`MatrixDataArray`](crate::matrix::MatrixDataArray).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_inputs` - The number of inputs to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// You can generate a `static mut` binding:
///
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 2;
/// impl_buffer_temp_BQ!(static mut TBQ, NUM_STATES, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TBQ.len(), 6);
///     assert_eq!(TBQ[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_buffer_temp_BQ {
    (mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_inputs, $t, $init, let mut)
    };
    ($mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_inputs, $t, $init, let)
    };
    (let mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_inputs, $t, $init, let mut)
    };
    (let $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_inputs, $t, $init, let)
    };
    (static mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_inputs, $t, $init, static mut)
    };
    (static $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        $crate::impl_buffer_temp_BQ!($mat_name, $num_states, $num_inputs, $t, $init, static)
    };
    ($mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr, $($keywords:tt)+) => {
        $($keywords)* $mat_name: $crate::buffer_types::TemporaryBQMatrixBuffer<
            $num_states,
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        > = $crate::buffer_types::TemporaryBQMatrixBuffer::<
            $num_states,
            $num_inputs,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_inputs }],
        ));
    };
}

/// Creates a static buffer fitting the square temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// This will create a [`TemporaryResidualCovarianceInvertedMatrixBuffer`](crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer)
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
        $($keywords)* $mat_name: $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer<
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer::<
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
/// This will create a [`TemporaryHPMatrixBuffer`](crate::buffer_types::TemporaryHPMatrixBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_temp_HP!(static mut THP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
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
        $($keywords)* $mat_name: $crate::buffer_types::TemporaryHPMatrixBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffer_types::TemporaryHPMatrixBuffer::<
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
/// This will create a [`TemporaryPHTMatrixBuffer`](crate::buffer_types::TemporaryPHTMatrixBuffer)
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
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_buffer_temp_PHt!(static mut TPHT, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
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
        $($keywords)* $mat_name: $crate::buffer_types::TemporaryPHTMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::matrix::MatrixDataArray<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::TemporaryPHTMatrixBuffer::<
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
/// This will create a [`TemporaryKHPMatrixBuffer`](crate::buffer_types::TemporaryKHPMatrixBuffer)
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
        $($keywords)* $mat_name: $crate::buffer_types::TemporaryKHPMatrixBuffer<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::TemporaryKHPMatrixBuffer::<
            $num_states,
            $t,
            $crate::matrix::MatrixDataArray<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::matrix::MatrixDataArray::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}
