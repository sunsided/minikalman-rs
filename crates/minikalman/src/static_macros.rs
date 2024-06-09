/// Sizes a buffer fitting the state vector (`num_states` × `1`).
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

/// Creates a buffer fitting the state transition matrix (`num_states` × `num_states`).
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

/// Creates a buffer fitting the state covariance matrix (`num_states` × `num_states`).
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

/// Sizes a buffer fitting the input vector (`num_inputs` × `1`).
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

/// Creates a buffer fitting the input transition matrix (`num_states` × `num_inputs`).
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

/// Creates a buffer fitting the input covariance matrix (`num_inputs` × `num_inputs`).
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

/// Creates a buffer fitting the measurement vector z (`num_measurements` × `1`).
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

/// Creates a buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
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

/// Creates a buffer fitting the measurement uncertainty matrix (`num_measurements` × `num_measurements`).
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

/// Creates a buffer fitting the innovation vector (`num_measurements` × `1`).
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

/// Creates a buffer fitting the innovation covariance matrix (`num_measurements` × `num_measurements`).
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

/// Creates a buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
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

/// Creates a buffer fitting the temporary x predictions (`num_states` × `1`).
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

/// Creates a buffer fitting the temporary P matrix (`num_states` × `num_states`).
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

/// Creates a buffer fitting the temporary B×Q matrix (`num_states` × `num_inputs`).
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

/// Creates a buffer fitting the temporary S-inverted (`num_measurements` × `num_measurements`).
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
