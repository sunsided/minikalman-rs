use crate::prelude::*;
use minikalman_traits::matrix::{MatrixData, MatrixDataBoxed, MatrixDataOwned};

/// A builder for buffers.

pub struct BufferBuilder;

impl BufferBuilder {
    pub fn state_vector_x<const STATES: usize>() -> StateVectorBufferBuilder<STATES> {
        StateVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn system_state_transition_A<const STATES: usize>(
    ) -> StateTransitionMatrixBufferBuilder<STATES> {
        StateTransitionMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn system_covariance_P<const STATES: usize>() -> SystemCovarianceMatrixBufferBuilder<STATES>
    {
        SystemCovarianceMatrixBufferBuilder
    }

    pub fn input_vector_u<const INPUTS: usize>() -> InputVectorBufferBuilder<INPUTS> {
        InputVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn input_transition_B<const STATES: usize, const INPUTS: usize>(
    ) -> InputTransitionMatrixBufferBuilder<STATES, INPUTS> {
        InputTransitionMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn input_covariance_Q<const INPUTS: usize>() -> InputCovarianceMatrixBufferBuilder<INPUTS> {
        InputCovarianceMatrixBufferBuilder
    }

    pub fn measurement_vector_z<const MEASUREMENTS: usize>(
    ) -> MeasurementVectorBufferBuilder<MEASUREMENTS> {
        MeasurementVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn measurement_transformation_H<const MEASUREMENTS: usize, const STATES: usize>(
    ) -> MeasurementTransformationMatrixBufferBuilder<MEASUREMENTS, STATES> {
        MeasurementTransformationMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn measurement_covariance_R<const MEASUREMENTS: usize>(
    ) -> MeasurementProcessNoiseCovarianceMatrixBufferBuilder<MEASUREMENTS> {
        MeasurementProcessNoiseCovarianceMatrixBufferBuilder
    }

    pub fn innovation_vector_y<const MEASUREMENTS: usize>(
    ) -> InnovationVectorBufferBuilder<MEASUREMENTS> {
        InnovationVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn innovation_covariance_S<const MEASUREMENTS: usize>(
    ) -> InnovationResidualCovarianceMatrixBufferBuilder<MEASUREMENTS> {
        InnovationResidualCovarianceMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn kalman_gain_K<const STATES: usize, const MEASUREMENTS: usize>(
    ) -> KalmanGainMatrixBufferBuilder<STATES, MEASUREMENTS> {
        KalmanGainMatrixBufferBuilder
    }

    pub fn state_prediction_temp_x<const STATES: usize>(
    ) -> StatePredictionVectorBufferBuilder<STATES> {
        StatePredictionVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_system_covariance_P<const STATES: usize>(
    ) -> TemporarySystemCovarianceMatrixBufferBuilder<STATES> {
        TemporarySystemCovarianceMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_BQ<const STATES: usize, const INPUTS: usize>(
    ) -> TemporaryBQMatrixBufferBuilder<STATES, INPUTS> {
        TemporaryBQMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_S_inv<const MEASUREMENTS: usize>() -> TemporarySInvMatrixBufferBuilder<MEASUREMENTS>
    {
        TemporarySInvMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_HP<const MEASUREMENTS: usize, const STATES: usize>(
    ) -> TemporaryHPMatrixBufferBuilder<MEASUREMENTS, STATES> {
        TemporaryHPMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_PHt<const STATES: usize, const MEASUREMENTS: usize>(
    ) -> TemporaryPHtMatrixBufferBuilder<STATES, MEASUREMENTS> {
        TemporaryPHtMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_KHP<const STATES: usize>() -> TemporaryKHPMatrixBufferBuilder<STATES> {
        TemporaryKHPMatrixBufferBuilder
    }
}

/// A builder for state vectors (`num_states` × `1`).
pub struct StateVectorBufferBuilder<const STATES: usize>;

/// A builder for state transition matrices (`num_states` × `num_states`).
pub struct StateTransitionMatrixBufferBuilder<const STATES: usize>;

/// A builder for system covariance matrices (`num_states` × `num_states`).
pub struct SystemCovarianceMatrixBufferBuilder<const STATES: usize>;

/// A builder for input vectors (`num_inputs` × `1`).
pub struct InputVectorBufferBuilder<const INPUTS: usize>;

/// A builder for input transition matrices (`num_states` × `num_inputs`).
pub struct InputTransitionMatrixBufferBuilder<const STATES: usize, const INPUTS: usize>;

/// A builder for input covariance matrices (`num_inputs` × `num_inputs`).
pub struct InputCovarianceMatrixBufferBuilder<const INPUTS: usize>;

/// A builder for measurement vectors (`num_measurements` × `1`).
pub struct MeasurementVectorBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for input transition matrices (`num_measurements` × `num_inputs`).
pub struct MeasurementTransformationMatrixBufferBuilder<
    const MEASUREMENT: usize,
    const STATES: usize,
>;

/// A builder for innovation vectors (`num_measurements` × `1`).
pub struct InnovationVectorBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for measurement noise covariance matrices (`num_measurements` × `num_measurements`).
pub struct MeasurementProcessNoiseCovarianceMatrixBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for measurement innovation (residual) covariance matrices (`num_measurements` × `num_measurements`).
pub struct InnovationResidualCovarianceMatrixBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for Kalman Gain matrices (`num_states` × `num_measurements`).
pub struct KalmanGainMatrixBufferBuilder<const STATES: usize, const MEASUREMENTS: usize>;

/// A builder for temporary state prediction vectors (`num_states` × `1`).
pub struct StatePredictionVectorBufferBuilder<const STATES: usize>;

/// A builder for temporary system covariance matrices (`num_states` × `num_states`).
pub struct TemporarySystemCovarianceMatrixBufferBuilder<const STATES: usize>;

/// A builder for temporary matrices (`num_states` × `num_inputs`).
pub struct TemporaryBQMatrixBufferBuilder<const STATES: usize, const INPUTS: usize>;

/// A builder for temporary matrices (`num_measurements` × `num_measurements`).
pub struct TemporarySInvMatrixBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for temporary matrices (`num_measurements` × `num_inputs`).
pub struct TemporaryHPMatrixBufferBuilder<const MEASUREMENT: usize, const STATES: usize>;

/// A builder for temporary matrices (`num_states` × `num_measurements`).
pub struct TemporaryPHtMatrixBufferBuilder<const STATES: usize, const MEASUREMENTS: usize>;

/// A builder for temporary K×(H×P) sized matrices (`num_states` × `num_states`).
pub struct TemporaryKHPMatrixBufferBuilder<const STATES: usize>;

impl<const STATES: usize> StateVectorBufferBuilder<STATES> {
    /// Builds a new [`StateVectorBufferBuilder`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::state_vector_x::<3>().new(0.0);
    ///
    /// let buffer: StateVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> StateVectorBuffer<STATES, T, MatrixDataOwned<STATES, 1, STATES, T>>
    where
        T: Copy,
    {
        StateVectorBuffer::<STATES, T, MatrixDataOwned<STATES, 1, STATES, T>>::new(
            MatrixData::new_owned::<STATES, 1, STATES, T>([init; STATES]),
        )
    }
}

impl<const STATES: usize> StateTransitionMatrixBufferBuilder<STATES> {
    /// Builds a new [`SystemMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::system_state_transition_A::<3>().new(0.0);
    ///
    /// let buffer: SystemMatrixMutBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> SystemMatrixMutBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>
    where
        T: Copy,
    {
        SystemMatrixMutBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}

impl<const STATES: usize> SystemCovarianceMatrixBufferBuilder<STATES> {
    /// Builds a new [`SystemCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::system_covariance_P::<3>().new(0.0);
    ///
    /// let buffer: SystemCovarianceMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> SystemCovarianceMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>
    where
        T: Copy,
    {
        SystemCovarianceMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}

impl<const INPUTS: usize> InputVectorBufferBuilder<INPUTS> {
    /// Builds a new [`InputVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::input_vector_u::<2>().new(0.0);
    ///
    /// let buffer: InputVectorBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> InputVectorBuffer<INPUTS, T, MatrixDataOwned<INPUTS, 1, INPUTS, T>>
    where
        T: Copy,
    {
        InputVectorBuffer::<INPUTS, T, MatrixDataOwned<INPUTS, 1, INPUTS, T>>::new(
            MatrixData::new_owned::<INPUTS, 1, INPUTS, T>([init; INPUTS]),
        )
    }
}

impl<const STATES: usize, const INPUTS: usize> InputTransitionMatrixBufferBuilder<STATES, INPUTS> {
    /// Builds a new [`InputMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::input_transition_B::<3, 2>().new(0.0);
    ///
    /// let buffer: InputMatrixMutBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> InputMatrixMutBuffer<STATES, INPUTS, T, MatrixDataBoxed<STATES, INPUTS, T>>
    where
        T: Copy,
    {
        InputMatrixMutBuffer::<STATES, INPUTS, T, MatrixDataBoxed<STATES, INPUTS, T>>::new(
            MatrixData::new_boxed::<STATES, INPUTS, T, _>(vec![init; STATES * INPUTS]),
        )
    }
}

impl<const INPUTS: usize> InputCovarianceMatrixBufferBuilder<INPUTS> {
    /// Builds a new [`InputCovarianceMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::input_covariance_Q::<2>().new(0.0);
    ///
    /// let buffer: InputCovarianceMatrixMutBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 4);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> InputCovarianceMatrixMutBuffer<INPUTS, T, MatrixDataBoxed<INPUTS, INPUTS, T>>
    where
        T: Copy,
    {
        InputCovarianceMatrixMutBuffer::<INPUTS, T, MatrixDataBoxed<INPUTS, INPUTS, T>>::new(
            MatrixData::new_boxed::<INPUTS, INPUTS, T, _>(vec![init; INPUTS * INPUTS]),
        )
    }
}

impl<const MEASUREMENTS: usize> MeasurementVectorBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`MeasurementVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::measurement_vector_z::<5>().new(0.0);
    ///
    /// let buffer: MeasurementVectorBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 5);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> MeasurementVectorBuffer<MEASUREMENTS, T, MatrixDataOwned<MEASUREMENTS, 1, MEASUREMENTS, T>>
    where
        T: Copy,
    {
        MeasurementVectorBuffer::<MEASUREMENTS, T, MatrixDataOwned<MEASUREMENTS, 1, MEASUREMENTS, T>>::new(
            MatrixData::new_owned::<MEASUREMENTS, 1, MEASUREMENTS, T>([init; MEASUREMENTS]),
        )
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize>
    MeasurementTransformationMatrixBufferBuilder<MEASUREMENTS, STATES>
{
    /// Builds a new [`MeasurementTransformationMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::measurement_transformation_H::<5, 3>().new(0.0);
    ///
    /// let buffer: MeasurementTransformationMatrixMutBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> MeasurementTransformationMatrixMutBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataBoxed<MEASUREMENTS, STATES, T>,
    >
    where
        T: Copy,
    {
        MeasurementTransformationMatrixMutBuffer::<
            MEASUREMENTS,
            STATES,
            T,
            MatrixDataBoxed<MEASUREMENTS, STATES, T>,
        >::new(MatrixData::new_boxed::<MEASUREMENTS, STATES, T, _>(
            vec![init; MEASUREMENTS * STATES],
        ))
    }
}

impl<const MEASUREMENTS: usize> MeasurementProcessNoiseCovarianceMatrixBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`MeasurementProcessNoiseCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::measurement_covariance_R::<5>().new(0.0);
    ///
    /// let buffer: MeasurementProcessNoiseCovarianceMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> MeasurementProcessNoiseCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
    >
    where
        T: Copy,
    {
        MeasurementProcessNoiseCovarianceMatrixBuffer::<
            MEASUREMENTS,
            T,
            MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
        >::new(MatrixData::new_boxed::<MEASUREMENTS, MEASUREMENTS, T, _>(
            vec![init; MEASUREMENTS * MEASUREMENTS],
        ))
    }
}

impl<const MEASUREMENTS: usize> InnovationVectorBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`InnovationVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::innovation_vector_y::<5>().new(0.0);
    ///
    /// let buffer: InnovationVectorBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 5);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> InnovationVectorBuffer<MEASUREMENTS, T, MatrixDataOwned<MEASUREMENTS, 1, MEASUREMENTS, T>>
    where
        T: Copy,
    {
        InnovationVectorBuffer::<MEASUREMENTS, T, MatrixDataOwned<MEASUREMENTS, 1, MEASUREMENTS, T>>::new(
            MatrixData::new_owned::<MEASUREMENTS, 1, MEASUREMENTS, T>([init; MEASUREMENTS]),
        )
    }
}

impl<const MEASUREMENTS: usize> InnovationResidualCovarianceMatrixBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`InnovationResidualCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::innovation_covariance_S::<5>().new(0.0);
    ///
    /// let buffer: InnovationResidualCovarianceMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> InnovationResidualCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
    >
    where
        T: Copy,
    {
        InnovationResidualCovarianceMatrixBuffer::<
            MEASUREMENTS,
            T,
            MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
        >::new(MatrixData::new_boxed::<MEASUREMENTS, MEASUREMENTS, T, _>(
            vec![init; MEASUREMENTS * MEASUREMENTS],
        ))
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize>
    KalmanGainMatrixBufferBuilder<STATES, MEASUREMENTS>
{
    /// Builds a new [`KalmanGainMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::kalman_gain_K::<3, 5>().new(0.0);
    ///
    /// let buffer: KalmanGainMatrixBuffer<3, 5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, MatrixDataBoxed<STATES, MEASUREMENTS, T>>
    where
        T: Copy,
    {
        KalmanGainMatrixBuffer::<
            STATES,
            MEASUREMENTS,
            T,
            MatrixDataBoxed<STATES, MEASUREMENTS, T>,
        >::new(MatrixData::new_boxed::<STATES, MEASUREMENTS, T, _>(
            vec![init; STATES * MEASUREMENTS],
        ))
    }
}

impl<const STATES: usize> StatePredictionVectorBufferBuilder<STATES> {
    /// Builds a new [`StatePredictionVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::state_prediction_temp_x::<3>().new(0.0);
    ///
    /// let buffer: StatePredictionVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> StatePredictionVectorBuffer<STATES, T, MatrixDataOwned<STATES, 1, STATES, T>>
    where
        T: Copy,
    {
        StatePredictionVectorBuffer::<STATES, T, MatrixDataOwned<STATES, 1, STATES, T>>::new(
            MatrixData::new_owned::<STATES, 1, STATES, T>([init; STATES]),
        )
    }
}

impl<const STATES: usize> TemporarySystemCovarianceMatrixBufferBuilder<STATES> {
    /// Builds a new [`TemporaryStateMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_system_covariance_P::<3>().new(0.0);
    ///
    /// let buffer: TemporaryStateMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> TemporaryStateMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>
    where
        T: Copy,
    {
        TemporaryStateMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}

impl<const STATES: usize, const INPUTS: usize> TemporaryBQMatrixBufferBuilder<STATES, INPUTS> {
    /// Builds a new [`TemporaryBQMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_BQ::<3, 2>().new(0.0);
    ///
    /// let buffer: TemporaryBQMatrixBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> TemporaryBQMatrixBuffer<STATES, INPUTS, T, MatrixDataBoxed<STATES, INPUTS, T>>
    where
        T: Copy,
    {
        TemporaryBQMatrixBuffer::<STATES, INPUTS, T, MatrixDataBoxed<STATES, INPUTS, T>>::new(
            MatrixData::new_boxed::<STATES, INPUTS, T, _>(vec![init; STATES * INPUTS]),
        )
    }
}

impl<const MEASUREMENTS: usize> TemporarySInvMatrixBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`TemporaryResidualCovarianceInvertedMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_S_inv::<5>().new(0.0);
    ///
    /// let buffer: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> TemporaryResidualCovarianceInvertedMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
    >
    where
        T: Copy,
    {
        TemporaryResidualCovarianceInvertedMatrixBuffer::<
            MEASUREMENTS,
            T,
            MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
        >::new(MatrixData::new_boxed::<MEASUREMENTS, MEASUREMENTS, T, _>(
            vec![init; MEASUREMENTS * MEASUREMENTS],
        ))
    }
}

impl<const MEASUREMENTS: usize, const STATES: usize>
    TemporaryHPMatrixBufferBuilder<MEASUREMENTS, STATES>
{
    /// Builds a new [`TemporaryHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_HP::<5, 3>().new(0.0);
    ///
    /// let buffer: TemporaryHPMatrixBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, MatrixDataBoxed<MEASUREMENTS, STATES, T>>
    where
        T: Copy,
    {
        TemporaryHPMatrixBuffer::<
            MEASUREMENTS,
            STATES,
            T,
            MatrixDataBoxed<MEASUREMENTS, STATES, T>,
        >::new(MatrixData::new_boxed::<MEASUREMENTS, STATES, T, _>(
            vec![init; MEASUREMENTS * STATES],
        ))
    }
}

impl<const STATES: usize, const MEASUREMENTS: usize>
    TemporaryPHtMatrixBufferBuilder<STATES, MEASUREMENTS>
{
    /// Builds a new [`TemporaryPHTMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_PHt::<3, 5>().new(0.0);
    ///
    /// let buffer: TemporaryPHTMatrixBuffer<3, 5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, MatrixDataBoxed<STATES, MEASUREMENTS, T>>
    where
        T: Copy,
    {
        TemporaryPHTMatrixBuffer::<
            STATES,
            MEASUREMENTS,
            T,
            MatrixDataBoxed<STATES, MEASUREMENTS, T>,
        >::new(MatrixData::new_boxed::<STATES, MEASUREMENTS, T, _>(
            vec![init; STATES * MEASUREMENTS],
        ))
    }
}

impl<const STATES: usize> TemporaryKHPMatrixBufferBuilder<STATES> {
    /// Builds a new [`TemporaryKHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::temp_KHP::<3>().new(0.0);
    ///
    /// let buffer: TemporaryKHPMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> TemporaryKHPMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>
    where
        T: Copy,
    {
        TemporaryKHPMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}
