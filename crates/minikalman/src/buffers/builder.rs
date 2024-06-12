use crate::buffers::types::*;
use crate::matrix::{MatrixData, MatrixDataArray, MatrixDataBoxed};

// TODO: Provide Kalman builder that returns impl KalmanFilter, or export a type alias e.g. via associated type.

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

    pub fn control_vector_u<const CONTROLS: usize>() -> ControlVectorBufferBuilder<CONTROLS> {
        ControlVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn control_transition_B<const STATES: usize, const CONTROLS: usize>(
    ) -> ControlTransitionMatrixBufferBuilder<STATES, CONTROLS> {
        ControlTransitionMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn control_covariance_Q<const CONTROLS: usize>(
    ) -> ControlCovarianceMatrixBufferBuilder<CONTROLS> {
        ControlCovarianceMatrixBufferBuilder
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
    pub fn temp_BQ<const STATES: usize, const CONTROLS: usize>(
    ) -> TemporaryBQMatrixBufferBuilder<STATES, CONTROLS> {
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

/// A builder for control vectors (`num_controls` × `1`).
pub struct ControlVectorBufferBuilder<const CONTROLS: usize>;

/// A builder for control transition matrices (`num_states` × `num_controls`).
pub struct ControlTransitionMatrixBufferBuilder<const STATES: usize, const CONTROLS: usize>;

/// A builder for control covariance matrices (`num_controls` × `num_controls`).
pub struct ControlCovarianceMatrixBufferBuilder<const CONTROLS: usize>;

/// A builder for measurement vectors (`num_measurements` × `1`).
pub struct MeasurementVectorBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for control transition matrices (`num_measurements` × `num_controls`).
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

/// A builder for temporary matrices (`num_states` × `num_controls`).
pub struct TemporaryBQMatrixBufferBuilder<const STATES: usize, const CONTROLS: usize>;

/// A builder for temporary matrices (`num_measurements` × `num_measurements`).
pub struct TemporarySInvMatrixBufferBuilder<const MEASUREMENTS: usize>;

/// A builder for temporary matrices (`num_measurements` × `num_controls`).
pub struct TemporaryHPMatrixBufferBuilder<const MEASUREMENT: usize, const STATES: usize>;

/// A builder for temporary matrices (`num_states` × `num_measurements`).
pub struct TemporaryPHtMatrixBufferBuilder<const STATES: usize, const MEASUREMENTS: usize>;

/// A builder for temporary K×(H×P) sized matrices (`num_states` × `num_states`).
pub struct TemporaryKHPMatrixBufferBuilder<const STATES: usize>;

/// The type of owned state vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type StateVectorBufferOwnedType<const STATES: usize, T> =
    StateVectorBuffer<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>;

impl<const STATES: usize> StateVectorBufferBuilder<STATES> {
    /// Builds a new [`StateVectorBufferBuilder`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::StateVectorBuffer;
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
    pub fn new<T>(&self, init: T) -> StateVectorBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        StateVectorBuffer::<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>::new(
            MatrixData::new_array::<STATES, 1, STATES, T>([init; STATES]),
        )
    }
}

/// The type of owned state transition matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type SystemMatrixMutBufferOwnedType<const STATES: usize, T> =
    SystemMatrixMutBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>;

impl<const STATES: usize> StateTransitionMatrixBufferBuilder<STATES> {
    /// Builds a new [`SystemMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::BufferBuilder;
    /// use minikalman::buffers::types::SystemMatrixMutBuffer;
    ///
    /// let buffer  = BufferBuilder::system_state_transition_A::<3>().new(0.0);
    ///
    /// let buffer: SystemMatrixMutBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self, init: T) -> SystemMatrixMutBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        SystemMatrixMutBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}

/// The type of owned state transition matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type SystemCovarianceMatrixBufferOwnedType<const STATES: usize, T> =
    SystemCovarianceMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>;

impl<const STATES: usize> SystemCovarianceMatrixBufferBuilder<STATES> {
    /// Builds a new [`SystemCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::SystemCovarianceMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> SystemCovarianceMatrixBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        SystemCovarianceMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}

/// The type of owned control vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ControlVectorBufferOwnedType<const STATES: usize, T> =
    ControlVectorBuffer<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>;

impl<const CONTROLS: usize> ControlVectorBufferBuilder<CONTROLS> {
    /// Builds a new [`ControlVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ControlVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::control_vector_u::<2>().new(0.0);
    ///
    /// let buffer: ControlVectorBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> ControlVectorBuffer<CONTROLS, T, MatrixDataArray<CONTROLS, 1, CONTROLS, T>>
    where
        T: Copy,
    {
        ControlVectorBuffer::<CONTROLS, T, MatrixDataArray<CONTROLS, 1, CONTROLS, T>>::new(
            MatrixData::new_array::<CONTROLS, 1, CONTROLS, T>([init; CONTROLS]),
        )
    }
}

/// The type of owned control matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ControlMatrixBufferOwnedType<const STATES: usize, const CONTROLS: usize, T> =
    ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>;

impl<const STATES: usize, const CONTROLS: usize>
    ControlTransitionMatrixBufferBuilder<STATES, CONTROLS>
{
    /// Builds a new [`ControlMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ControlMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::control_transition_B::<3, 2>().new(0.0);
    ///
    /// let buffer: ControlMatrixMutBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>
    where
        T: Copy,
    {
        ControlMatrixMutBuffer::<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>::new(
            MatrixData::new_boxed::<STATES, CONTROLS, T, _>(vec![init; STATES * CONTROLS]),
        )
    }
}

/// The type of owned control matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ControlCovarianceMatrixBufferOwnedType<const CONTROLS: usize, T> =
    ControlCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>;

impl<const CONTROLS: usize> ControlCovarianceMatrixBufferBuilder<CONTROLS> {
    /// Builds a new [`ControlCovarianceMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ControlCovarianceMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::control_covariance_Q::<2>().new(0.0);
    ///
    /// let buffer: ControlCovarianceMatrixMutBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 4);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
        init: T,
    ) -> ControlCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>
    where
        T: Copy,
    {
        ControlCovarianceMatrixMutBuffer::<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>::new(
            MatrixData::new_boxed::<CONTROLS, CONTROLS, T, _>(vec![init; CONTROLS * CONTROLS]),
        )
    }
}

/// The type of observation vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ObservationVectorBufferOwnedType<const MEASUREMENTS: usize, T> =
    MeasurementVectorBuffer<MEASUREMENTS, T, MatrixDataArray<MEASUREMENTS, 1, MEASUREMENTS, T>>;

impl<const MEASUREMENTS: usize> MeasurementVectorBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`MeasurementVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementVectorBuffer;
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
    pub fn new<T>(&self, init: T) -> ObservationVectorBufferOwnedType<MEASUREMENTS, T>
    where
        T: Copy,
    {
        MeasurementVectorBuffer::<MEASUREMENTS, T, MatrixDataArray<MEASUREMENTS, 1, MEASUREMENTS, T>>::new(
            MatrixData::new_array::<MEASUREMENTS, 1, MEASUREMENTS, T>([init; MEASUREMENTS]),
        )
    }
}

/// The type of observation matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ObservationMatrixBufferOwnedType<const MEASUREMENTS: usize, const STATES: usize, T> =
    MeasurementObservationMatrixMutBuffer<
        MEASUREMENTS,
        STATES,
        T,
        MatrixDataBoxed<MEASUREMENTS, STATES, T>,
    >;

impl<const MEASUREMENTS: usize, const STATES: usize>
    MeasurementTransformationMatrixBufferBuilder<MEASUREMENTS, STATES>
{
    /// Builds a new [`MeasurementObservationMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementObservationMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::measurement_transformation_H::<5, 3>().new(0.0);
    ///
    /// let buffer: MeasurementObservationMatrixMutBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self, init: T) -> ObservationMatrixBufferOwnedType<MEASUREMENTS, STATES, T>
    where
        T: Copy,
    {
        MeasurementObservationMatrixMutBuffer::<
            MEASUREMENTS,
            STATES,
            T,
            MatrixDataBoxed<MEASUREMENTS, STATES, T>,
        >::new(MatrixData::new_boxed::<MEASUREMENTS, STATES, T, _>(
            vec![init; MEASUREMENTS * STATES],
        ))
    }
}

/// The type of observation / process noise covariance matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ObservationCovarianceBufferOwnedType<const MEASUREMENTS: usize, T> =
    MeasurementProcessNoiseCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
    >;

impl<const MEASUREMENTS: usize> MeasurementProcessNoiseCovarianceMatrixBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`MeasurementProcessNoiseCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementProcessNoiseCovarianceMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> ObservationCovarianceBufferOwnedType<MEASUREMENTS, T>
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

/// The type of innovation vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type InnovationVectorBufferOwnedType<const MEASUREMENTS: usize, T> =
    InnovationVectorBuffer<MEASUREMENTS, T, MatrixDataArray<MEASUREMENTS, 1, MEASUREMENTS, T>>;

impl<const MEASUREMENTS: usize> InnovationVectorBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`InnovationVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::InnovationVectorBuffer;
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
    pub fn new<T>(&self, init: T) -> InnovationVectorBufferOwnedType<MEASUREMENTS, T>
    where
        T: Copy,
    {
        InnovationVectorBuffer::<MEASUREMENTS, T, MatrixDataArray<MEASUREMENTS, 1, MEASUREMENTS, T>>::new(
            MatrixData::new_array::<MEASUREMENTS, 1, MEASUREMENTS, T>([init; MEASUREMENTS]),
        )
    }
}

/// The type of innovation residual covariance matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type InnovationResidualCovarianceMatrixBufferOwnedType<const MEASUREMENTS: usize, T> =
    InnovationResidualCovarianceMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
    >;

impl<const MEASUREMENTS: usize> InnovationResidualCovarianceMatrixBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`InnovationResidualCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::InnovationResidualCovarianceMatrixBuffer;
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
    ) -> InnovationResidualCovarianceMatrixBufferOwnedType<MEASUREMENTS, T>
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

/// The type of innovation residual covariance matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type KalmanGainMatrixBufferOwnedType<const STATES: usize, const MEASUREMENTS: usize, T> =
    KalmanGainMatrixBuffer<STATES, MEASUREMENTS, T, MatrixDataBoxed<STATES, MEASUREMENTS, T>>;

impl<const STATES: usize, const MEASUREMENTS: usize>
    KalmanGainMatrixBufferBuilder<STATES, MEASUREMENTS>
{
    /// Builds a new [`KalmanGainMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::KalmanGainMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> KalmanGainMatrixBufferOwnedType<STATES, MEASUREMENTS, T>
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

/// The type of owned state vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryStatePredictionVectorBufferOwnedType<const STATES: usize, T> =
    TemporaryStatePredictionVectorBuffer<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>;

impl<const STATES: usize> StatePredictionVectorBufferBuilder<STATES> {
    /// Builds a new [`TemporaryStatePredictionVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryStatePredictionVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::state_prediction_temp_x::<3>().new(0.0);
    ///
    /// let buffer: TemporaryStatePredictionVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self, init: T) -> TemporaryStatePredictionVectorBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        TemporaryStatePredictionVectorBuffer::<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>::new(
            MatrixData::new_array::<STATES, 1, STATES, T>([init; STATES]),
        )
    }
}

/// The type of owned state transition matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryStateMatrixBufferOwnedType<const STATES: usize, T> =
    TemporaryStateMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>;

impl<const STATES: usize> TemporarySystemCovarianceMatrixBufferBuilder<STATES> {
    /// Builds a new [`TemporaryStateMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryStateMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> TemporaryStateMatrixBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        TemporaryStateMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}

/// The type of temporary B×Q matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryBQMatrixBufferOwnedType<const STATES: usize, const CONTROLS: usize, T> =
    TemporaryBQMatrixBuffer<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>;

impl<const STATES: usize, const CONTROLS: usize> TemporaryBQMatrixBufferBuilder<STATES, CONTROLS> {
    /// Builds a new [`TemporaryBQMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryBQMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> TemporaryBQMatrixBufferOwnedType<STATES, CONTROLS, T>
    where
        T: Copy,
    {
        TemporaryBQMatrixBuffer::<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>::new(
            MatrixData::new_boxed::<STATES, CONTROLS, T, _>(vec![init; STATES * CONTROLS]),
        )
    }
}

/// The type of temporary S-inverted matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporarySInvertedMatrixBufferOwnedType<const MEASUREMENTS: usize, T> =
    TemporaryResidualCovarianceInvertedMatrixBuffer<
        MEASUREMENTS,
        T,
        MatrixDataBoxed<MEASUREMENTS, MEASUREMENTS, T>,
    >;

impl<const MEASUREMENTS: usize> TemporarySInvMatrixBufferBuilder<MEASUREMENTS> {
    /// Builds a new [`TemporaryResidualCovarianceInvertedMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> TemporarySInvertedMatrixBufferOwnedType<MEASUREMENTS, T>
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

/// The type of temporary H×P matrix buffers, tandem to [`TemporaryPHtMatrixBufferOwnedType`].
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryHPMatrixBufferOwnedType<const MEASUREMENTS: usize, const STATES: usize, T> =
    TemporaryHPMatrixBuffer<MEASUREMENTS, STATES, T, MatrixDataBoxed<MEASUREMENTS, STATES, T>>;

impl<const MEASUREMENTS: usize, const STATES: usize>
    TemporaryHPMatrixBufferBuilder<MEASUREMENTS, STATES>
{
    /// Builds a new [`TemporaryHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryHPMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> TemporaryHPMatrixBufferOwnedType<MEASUREMENTS, STATES, T>
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

/// The type of temporary P×Hᵀ matrix buffers, tandem to [`TemporaryHPMatrixBufferOwnedType`].
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryPHtMatrixBufferOwnedType<const STATES: usize, const MEASUREMENTS: usize, T> =
    TemporaryPHTMatrixBuffer<STATES, MEASUREMENTS, T, MatrixDataBoxed<STATES, MEASUREMENTS, T>>;

impl<const STATES: usize, const MEASUREMENTS: usize>
    TemporaryPHtMatrixBufferBuilder<STATES, MEASUREMENTS>
{
    /// Builds a new [`TemporaryPHTMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryPHTMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> TemporaryPHtMatrixBufferOwnedType<STATES, MEASUREMENTS, T>
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

/// The type of temporary K×H×P matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryKHPMatrixBufferOwnedType<const STATES: usize, T> =
    TemporaryKHPMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>;

impl<const STATES: usize> TemporaryKHPMatrixBufferBuilder<STATES> {
    /// Builds a new [`TemporaryKHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryKHPMatrixBuffer;
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
    pub fn new<T>(&self, init: T) -> TemporaryKHPMatrixBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        TemporaryKHPMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(vec![init; STATES * STATES]),
        )
    }
}
