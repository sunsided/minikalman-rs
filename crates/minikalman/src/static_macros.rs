/// Creates a static buffer fitting the state covariance matrix (`num_states` × `num_states`).
///
/// This will create a [`StateVectorBuffer`](crate::buffer_types::StateVectorBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_x!(mut X, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(X.len(), 3);
///     assert_eq!(X[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_static_buffer_x {
    ($vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static $vec_name: $crate::buffer_types::StateVectorBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffer_types::StateVectorBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
    (mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static mut $vec_name: $crate::buffer_types::StateVectorBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffer_types::StateVectorBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square state transition matrix (`num_states` × `num_states`).
///
/// This will create a [`SystemMatrixBuffer`](crate::buffer_types::SystemMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_A!(mut A, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(A.len(), 9);
///     assert_eq!(A[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_A {
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::SystemMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::SystemMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::SystemMatrixMutBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::SystemMatrixMutBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the square state covariance matrix (`num_states` × `num_states`).
///
/// This will create a [`SystemCovarianceMatrixBuffer`](crate::buffer_types::SystemCovarianceMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_P!(mut P, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(P.len(), 9);
///     assert_eq!(P[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_P {
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::SystemCovarianceMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::SystemCovarianceMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::SystemCovarianceMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::SystemCovarianceMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Sizes a static buffer fitting the input vector (`num_inputs` × `1`).
///
/// This will create a [`InputVectorBuffer`](crate::buffer_types::InputVectorBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_INPUTS: usize = 2;
/// impl_static_buffer_u!(mut U, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(U.len(), 2);
///     assert_eq!(U[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_static_buffer_u {
    ($vec_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        static $vec_name: $crate::buffer_types::InputVectorBuffer<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, 1, { $num_inputs * 1 }, $t>,
        > = $crate::buffer_types::InputVectorBuffer::<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, 1, { $num_inputs * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_inputs * 1 }],
        ));
    };
    (mut $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::InputVectorBuffer<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, 1, { $num_inputs * 1 }, $t>,
        > = $crate::buffer_types::InputVectorBuffer::<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, 1, { $num_inputs * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_inputs * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the input transition matrix (`num_states` × `num_inputs`).
///
/// This will create a [`InputMatrixMutBuffer`](crate::buffer_types::InputMatrixMutBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_inputs` - The number of inputs to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 2;
/// impl_static_buffer_B!(mut B, NUM_STATES, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(B.len(), 6);
///     assert_eq!(B[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_B {
    ($mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::InputMatrixMutBuffer<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        > = $crate::buffer_types::InputMatrixMutBuffer::<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_inputs }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::InputMatrixMutBuffer<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        > = $crate::buffer_types::InputMatrixMutBuffer::<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_inputs }],
        ));
    };
}

/// Creates a static buffer fitting the square input covariance matrix (`num_inputs` × `num_inputs`).
///
/// This will create a [`InputCovarianceMatrixMutBuffer`](crate::buffer_types::InputCovarianceMatrixMutBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_inputs` - The number of inputs to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_INPUTS: usize = 2;
/// impl_static_buffer_Q!(mut Q, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Q.len(), 4);
///     assert_eq!(Q[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_Q {
    ($mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::InputCovarianceMatrixMutBuffer<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, $num_inputs, { $num_inputs * $num_inputs }, $t>,
        > = $crate::buffer_types::InputCovarianceMatrixMutBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_inputs, $num_inputs, { $num_inputs * $num_inputs }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_inputs * $num_inputs }],
        ));
    };
    (mut $mat_name:ident, $num_inputs:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::InputCovarianceMatrixMutBuffer<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, $num_inputs, { $num_inputs * $num_inputs }, $t>,
        > = $crate::buffer_types::InputCovarianceMatrixMutBuffer::<
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_inputs, $num_inputs, { $num_inputs * $num_inputs }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_inputs * $num_inputs }],
        ));
    };
}

/// Creates a static buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// This will create a [`MeasurementVectorBuffer`](crate::buffer_types::MeasurementVectorBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_z!(mut Z, NUM_MEASUREMENTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Z.len(), 5);
///     assert_eq!(Z[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_static_buffer_z {
    ($modifier:tt $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static $modifier $vec_name: $crate::buffer_types::MeasurementVectorBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffer_types::MeasurementVectorBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
    (mut $modifier:tt $vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $modifier $vec_name: $crate::buffer_types::MeasurementVectorBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffer_types::MeasurementVectorBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
///
/// This will create a [`MeasurementTransformationMatrixMutBuffer`](crate::buffer_types::MeasurementTransformationMatrixMutBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states of the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_H!(mut H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(H.len(), 15);
///     assert_eq!(H[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_H {
    ($mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::MeasurementTransformationMatrixMutBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffer_types::MeasurementTransformationMatrixMutBuffer::<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_states }],
        ));
    };
    (mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::MeasurementTransformationMatrixMutBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffer_types::MeasurementTransformationMatrixMutBuffer::<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the square measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// This will create a [`MeasurementProcessNoiseCovarianceMatrixBuffer`](crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_R!(mut R, NUM_MEASUREMENTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(R.len(), 25);
///     assert_eq!(R[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_R {
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::MeasurementProcessNoiseCovarianceMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// This will create a [`InnovationVectorBuffer`](crate::buffer_types::InnovationVectorBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_y!(mut Y, NUM_MEASUREMENTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(Y.len(), 5);
///     assert_eq!(Y[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_static_buffer_y {
    ($vec_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static $vec_name: $crate::buffer_types::InnovationVectorBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffer_types::InnovationVectorBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::InnovationVectorBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        > = $crate::buffer_types::InnovationVectorBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<$num_measurements, 1, { $num_measurements * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square innovation (residual) covariance matrix (`num_measurements` × `num_measurements`).
///
/// This will create a [`InnovationResidualCovarianceMatrixBuffer`](crate::buffer_types::InnovationResidualCovarianceMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_S!(mut S, NUM_MEASUREMENTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(S.len(), 25);
///     assert_eq!(S[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_S {
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::InnovationResidualCovarianceMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
///
/// This will create a [`InnovationResidualCovarianceMatrixBuffer`](crate::buffer_types::InnovationResidualCovarianceMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_K!(mut K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(K.len(), 15);
///     assert_eq!(K[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_K {
    ($mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::KalmanGainMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::KalmanGainMatrixBuffer::<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_measurements }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::KalmanGainMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::KalmanGainMatrixBuffer::<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// This will create a [`StatePredictionVectorBuffer`](crate::buffer_types::StatePredictionVectorBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_temp_x!(mut TX, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TX.len(), 3);
///     assert_eq!(TX[0], 0.0_f32);
/// }
/// ```
#[macro_export]
macro_rules! impl_static_buffer_temp_x {
    ($vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static $vec_name: $crate::buffer_types::StatePredictionVectorBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffer_types::StatePredictionVectorBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
    (mut $vec_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static mut $vec_name: $crate::buffer_types::StatePredictionVectorBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        > = $crate::buffer_types::StatePredictionVectorBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, 1, { $num_states * 1 }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * 1 }],
        ));
    };
}

/// Creates a static buffer fitting the square temporary P matrix (`num_states` × `num_states`).
///
/// This will create a [`TemporaryStateMatrixBuffer`](crate::buffer_types::TemporaryStateMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_temp_P!(mut TP, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TP.len(), 9);
///     assert_eq!(TP[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_temp_P {
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::TemporaryStateMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::TemporaryStateMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::TemporaryStateMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::TemporaryStateMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}

/// Creates a static buffer fitting the temporary B×Q matrix (`num_states` × `num_inputs`).
///
/// This will create a [`TemporaryBQMatrixBuffer`](crate::buffer_types::TemporaryBQMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_inputs` - The number of inputs to the system.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_INPUTS: usize = 2;
/// impl_static_buffer_temp_BQ!(mut TBQ, NUM_STATES, NUM_INPUTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TBQ.len(), 6);
///     assert_eq!(TBQ[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_temp_BQ {
    ($mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::TemporaryBQMatrixBuffer<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        > = $crate::buffer_types::TemporaryBQMatrixBuffer::<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_inputs }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $num_inputs:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::TemporaryBQMatrixBuffer<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        > = $crate::buffer_types::TemporaryBQMatrixBuffer::<
            $num_states,
            $num_inputs,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_inputs, { $num_states * $num_inputs }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_inputs }],
        ));
    };
}

/// Creates a static buffer fitting the square temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// This will create a [`TemporaryBQMatrixBuffer`](crate::buffer_types::TemporaryBQMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_temp_S_inv!(mut TSINV, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TSINV.len(), 9);
///     assert_eq!(TSINV[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_temp_S_inv {
    ($mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
    (mut $mat_name:ident, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $mat_name:
            $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer<
                $num_measurements,
                $t,
                $crate::MatrixDataOwned<
                    $num_measurements,
                    $num_measurements,
                    { $num_measurements * $num_measurements },
                    $t,
                >,
            > = $crate::buffer_types::TemporaryResidualCovarianceInvertedMatrixBuffer::<
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_measurements,
                { $num_measurements * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_measurements }],
        ));
    };
}

/// Creates a static buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// This will create a [`TemporaryHPMatrixBuffer`](crate::buffer_types::TemporaryHPMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_temp_HP!(mut THP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(THP.len(), 15);
///     assert_eq!(THP[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_temp_HP {
    ($mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::TemporaryHPMatrixBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffer_types::TemporaryHPMatrixBuffer::<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_states }],
        ));
    };
    (mut $mat_name:ident, $num_measurements:expr, $num_states:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::TemporaryHPMatrixBuffer<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        > = $crate::buffer_types::TemporaryHPMatrixBuffer::<
            $num_measurements,
            $num_states,
            $t,
            $crate::MatrixDataOwned<
                $num_measurements,
                $num_states,
                { $num_measurements * $num_states },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_measurements * $num_states }],
        ));
    };
}

/// Creates a buffer fitting the temporary P×Hᵀ buffer (`num_states` × `num_measurements`).
///
/// This will create a [`TemporaryPHTMatrixBuffer`](crate::buffer_types::TemporaryPHTMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// const NUM_MEASUREMENTS: usize = 5;
/// impl_static_buffer_temp_PHt!(mut TPHT, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TPHT.len(), 15);
///     assert_eq!(TPHT[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_temp_PHt {
    ($mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::TemporaryPHTMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::TemporaryPHTMatrixBuffer::<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_measurements }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $num_measurements:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::TemporaryPHTMatrixBuffer<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        > = $crate::buffer_types::TemporaryPHTMatrixBuffer::<
            $num_states,
            $num_measurements,
            $t,
            $crate::MatrixDataOwned<
                $num_states,
                $num_measurements,
                { $num_states * $num_measurements },
                $t,
            >,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_measurements }],
        ));
    };
}

/// Creates a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_states`).
///
/// This will create a [`TemporaryKHPMatrixBuffer`](crate::buffer_types::TemporaryKHPMatrixBuffer)
/// backed by a [`MatrixDataOwned`](crate::MatrixDataOwned).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `t` - The data type.
/// * `init` - The default value to initialize the buffer with.
///
/// ## Example
/// ```
/// # use minikalman::prelude::*;
/// const NUM_STATES: usize = 3;
/// impl_static_buffer_temp_KHP!(mut TKHP, NUM_STATES, f32, 0.0);
///
/// unsafe {
///     assert_eq!(TKHP.len(), 9);
///     assert_eq!(TKHP[0], 0.0_f32);
/// }
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! impl_static_buffer_temp_KHP {
    ($mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static $mat_name: $crate::buffer_types::TemporaryKHPMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::TemporaryKHPMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
    (mut $mat_name:ident, $num_states:expr, $t:ty, $init:expr) => {
        static mut $mat_name: $crate::buffer_types::TemporaryKHPMatrixBuffer<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        > = $crate::buffer_types::TemporaryKHPMatrixBuffer::<
            $num_states,
            $t,
            $crate::MatrixDataOwned<$num_states, $num_states, { $num_states * $num_states }, $t>,
        >::new($crate::MatrixDataOwned::new_unchecked(
            [$init; { $num_states * $num_states }],
        ));
    };
}
