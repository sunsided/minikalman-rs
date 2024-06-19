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
    #[doc(alias = "system_state_transition_A")]
    pub fn system_matrix_A<const STATES: usize>() -> StateTransitionMatrixBufferBuilder<STATES> {
        StateTransitionMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    #[doc(alias = "system_covariance_P")]
    pub fn estimate_covariance_P<const STATES: usize>(
    ) -> EstimateCovarianceMatrixBufferBuilder<STATES> {
        EstimateCovarianceMatrixBufferBuilder
    }

    pub fn control_vector_u<const CONTROLS: usize>() -> ControlVectorBufferBuilder<CONTROLS> {
        ControlVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn control_matrix_B<const STATES: usize, const CONTROLS: usize>(
    ) -> ControlTransitionMatrixBufferBuilder<STATES, CONTROLS> {
        ControlTransitionMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    #[doc(alias = "control_covariance_Q")]
    pub fn process_noise_covariance_Q<const CONTROLS: usize>(
    ) -> ProcessNoiseCovarianceMatrixBufferBuilder<CONTROLS> {
        ProcessNoiseCovarianceMatrixBufferBuilder
    }

    #[doc(alias = "observation_vector_z")]
    pub fn measurement_vector_z<const OBSERVATIONS: usize>(
    ) -> ObservationVectorBufferBuilder<OBSERVATIONS> {
        ObservationVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    #[doc(alias = "measurement_transformation_H")]
    pub fn observation_matrix_H<const OBSERVATIONS: usize, const STATES: usize>(
    ) -> ObservationMatrixBufferBuilder<OBSERVATIONS, STATES> {
        ObservationMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn observation_covariance_R<const OBSERVATIONS: usize>(
    ) -> MeasurementNoiseCovarianceMatrixBufferBuilder<OBSERVATIONS> {
        MeasurementNoiseCovarianceMatrixBufferBuilder
    }

    pub fn innovation_vector_y<const OBSERVATIONS: usize>(
    ) -> InnovationVectorBufferBuilder<OBSERVATIONS> {
        InnovationVectorBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn innovation_covariance_S<const OBSERVATIONS: usize>(
    ) -> InnovationResidualCovarianceMatrixBufferBuilder<OBSERVATIONS> {
        InnovationResidualCovarianceMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn kalman_gain_K<const STATES: usize, const OBSERVATIONS: usize>(
    ) -> KalmanGainMatrixBufferBuilder<STATES, OBSERVATIONS> {
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
    pub fn temp_S_inv<const OBSERVATIONS: usize>() -> TemporarySInvMatrixBufferBuilder<OBSERVATIONS>
    {
        TemporarySInvMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_HP<const OBSERVATIONS: usize, const STATES: usize>(
    ) -> TemporaryHPMatrixBufferBuilder<OBSERVATIONS, STATES> {
        TemporaryHPMatrixBufferBuilder
    }

    #[allow(non_snake_case)]
    pub fn temp_PHt<const STATES: usize, const OBSERVATIONS: usize>(
    ) -> TemporaryPHtMatrixBufferBuilder<STATES, OBSERVATIONS> {
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
#[doc(alias = "SystemCovarianceMatrixBufferBuilder")]
pub struct EstimateCovarianceMatrixBufferBuilder<const STATES: usize>;

/// A builder for control vectors (`num_controls` × `1`).
pub struct ControlVectorBufferBuilder<const CONTROLS: usize>;

/// A builder for control transition matrices (`num_states` × `num_controls`).
pub struct ControlTransitionMatrixBufferBuilder<const STATES: usize, const CONTROLS: usize>;

/// A builder for control covariance matrices (`num_controls` × `num_controls`).
#[doc(alias = "ControlCovarianceMatrixBufferBuilder")]
pub struct ProcessNoiseCovarianceMatrixBufferBuilder<const CONTROLS: usize>;

/// A builder for measurement vectors (`num_measurements` × `1`).
pub struct ObservationVectorBufferBuilder<const OBSERVATIONS: usize>;

/// A builder for control transition matrices (`num_measurements` × `num_controls`).
pub struct ObservationMatrixBufferBuilder<const OBSERVATION: usize, const STATES: usize>;

/// A builder for innovation vectors (`num_measurements` × `1`).
pub struct InnovationVectorBufferBuilder<const OBSERVATIONS: usize>;

/// A builder for measurement noise covariance matrices (`num_measurements` × `num_measurements`).
pub struct MeasurementNoiseCovarianceMatrixBufferBuilder<const OBSERVATIONS: usize>;

/// A builder for measurement innovation (residual) covariance matrices (`num_measurements` × `num_measurements`).
pub struct InnovationResidualCovarianceMatrixBufferBuilder<const OBSERVATIONS: usize>;

/// A builder for Kalman Gain matrices (`num_states` × `num_measurements`).
pub struct KalmanGainMatrixBufferBuilder<const STATES: usize, const OBSERVATIONS: usize>;

/// A builder for temporary state prediction vectors (`num_states` × `1`).
pub struct StatePredictionVectorBufferBuilder<const STATES: usize>;

/// A builder for temporary system covariance matrices (`num_states` × `num_states`).
pub struct TemporarySystemCovarianceMatrixBufferBuilder<const STATES: usize>;

/// A builder for temporary matrices (`num_states` × `num_controls`).
pub struct TemporaryBQMatrixBufferBuilder<const STATES: usize, const CONTROLS: usize>;

/// A builder for temporary matrices (`num_measurements` × `num_measurements`).
pub struct TemporarySInvMatrixBufferBuilder<const OBSERVATIONS: usize>;

/// A builder for temporary matrices (`num_measurements` × `num_controls`).
pub struct TemporaryHPMatrixBufferBuilder<const OBSERVATION: usize, const STATES: usize>;

/// A builder for temporary matrices (`num_states` × `num_measurements`).
pub struct TemporaryPHtMatrixBufferBuilder<const STATES: usize, const OBSERVATIONS: usize>;

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
    /// let buffer = BufferBuilder::state_vector_x::<3>().new();
    ///
    /// let buffer: StateVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> StateVectorBufferOwnedType<STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`StateVectorBufferBuilder`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::StateVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::state_vector_x::<3>().new_with(0.0);
    ///
    /// let buffer: StateVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> StateVectorBufferOwnedType<STATES, T>
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
    StateTransitionMatrixMutBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>;

impl<const STATES: usize> StateTransitionMatrixBufferBuilder<STATES> {
    /// Builds a new [`StateTransitionMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::BufferBuilder;
    /// use minikalman::buffers::types::StateTransitionMatrixMutBuffer;
    ///
    /// let buffer  = BufferBuilder::system_matrix_A::<3>().new();
    ///
    /// let buffer: StateTransitionMatrixMutBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> SystemMatrixMutBufferOwnedType<STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`StateTransitionMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::BufferBuilder;
    /// use minikalman::buffers::types::StateTransitionMatrixMutBuffer;
    ///
    /// let buffer  = BufferBuilder::system_matrix_A::<3>().new_with(0.0);
    ///
    /// let buffer: StateTransitionMatrixMutBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> SystemMatrixMutBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        StateTransitionMatrixMutBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(alloc::vec![init; STATES * STATES]),
        )
    }
}

/// The type of owned state transition matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[doc(alias = "SystemCovarianceMatrixBufferOwnedType")]
pub type EstimateCovarianceMatrixBufferOwnedType<const STATES: usize, T> =
    EstimateCovarianceMatrixBuffer<STATES, T, MatrixDataBoxed<STATES, STATES, T>>;

impl<const STATES: usize> EstimateCovarianceMatrixBufferBuilder<STATES> {
    /// Builds a new [`EstimateCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::EstimateCovarianceMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::estimate_covariance_P::<3>().new();
    ///
    /// let buffer: EstimateCovarianceMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> EstimateCovarianceMatrixBufferOwnedType<STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`EstimateCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::EstimateCovarianceMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::estimate_covariance_P::<3>().new_with(0.0);
    ///
    /// let buffer: EstimateCovarianceMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> EstimateCovarianceMatrixBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        EstimateCovarianceMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(alloc::vec![init; STATES * STATES]),
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
    /// let buffer = BufferBuilder::control_vector_u::<2>().new();
    ///
    /// let buffer: ControlVectorBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
    ) -> ControlVectorBuffer<CONTROLS, T, MatrixDataArray<CONTROLS, 1, CONTROLS, T>>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`ControlVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ControlVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::control_vector_u::<2>().new_with(0.0);
    ///
    /// let buffer: ControlVectorBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 2);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(
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
    /// let buffer  = BufferBuilder::control_matrix_B::<3, 2>().new();
    ///
    /// let buffer: ControlMatrixMutBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
    ) -> ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`ControlMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ControlMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::control_matrix_B::<3, 2>().new_with(0.0);
    ///
    /// let buffer: ControlMatrixMutBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(
        &self,
        init: T,
    ) -> ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>
    where
        T: Copy,
    {
        ControlMatrixMutBuffer::<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>::new(
            MatrixData::new_boxed::<STATES, CONTROLS, T, _>(alloc::vec![init; STATES * CONTROLS]),
        )
    }
}

/// The type of owned control matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ControlCovarianceMatrixBufferOwnedType<const CONTROLS: usize, T> =
    ProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>;

impl<const CONTROLS: usize> ProcessNoiseCovarianceMatrixBufferBuilder<CONTROLS> {
    /// Builds a new [`ProcessNoiseCovarianceMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ProcessNoiseCovarianceMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::process_noise_covariance_Q::<2>().new();
    ///
    /// let buffer: ProcessNoiseCovarianceMatrixMutBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 4);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(
        &self,
    ) -> ProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`ProcessNoiseCovarianceMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ProcessNoiseCovarianceMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::process_noise_covariance_Q::<2>().new_with(0.0);
    ///
    /// let buffer: ProcessNoiseCovarianceMatrixMutBuffer<2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 4);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(
        &self,
        init: T,
    ) -> ProcessNoiseCovarianceMatrixMutBuffer<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>
    where
        T: Copy,
    {
        ProcessNoiseCovarianceMatrixMutBuffer::<CONTROLS, T, MatrixDataBoxed<CONTROLS, CONTROLS, T>>::new(
            MatrixData::new_boxed::<CONTROLS, CONTROLS, T, _>(alloc::vec![init; CONTROLS * CONTROLS]),
        )
    }
}

/// The type of observation vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ObservationVectorBufferOwnedType<const OBSERVATIONS: usize, T> =
    MeasurementVectorBuffer<OBSERVATIONS, T, MatrixDataArray<OBSERVATIONS, 1, OBSERVATIONS, T>>;

impl<const OBSERVATIONS: usize> ObservationVectorBufferBuilder<OBSERVATIONS> {
    /// Builds a new [`MeasurementVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::measurement_vector_z::<5>().new();
    ///
    /// let buffer: MeasurementVectorBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 5);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> ObservationVectorBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`MeasurementVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::measurement_vector_z::<5>().new_with(0.0);
    ///
    /// let buffer: MeasurementVectorBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 5);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> ObservationVectorBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy,
    {
        MeasurementVectorBuffer::<OBSERVATIONS, T, MatrixDataArray<OBSERVATIONS, 1, OBSERVATIONS, T>>::new(
            MatrixData::new_array::<OBSERVATIONS, 1, OBSERVATIONS, T>([init; OBSERVATIONS]),
        )
    }
}

/// The type of observation matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type ObservationMatrixBufferOwnedType<const OBSERVATIONS: usize, const STATES: usize, T> =
    ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, MatrixDataBoxed<OBSERVATIONS, STATES, T>>;

impl<const OBSERVATIONS: usize, const STATES: usize>
    ObservationMatrixBufferBuilder<OBSERVATIONS, STATES>
{
    /// Builds a new [`ObservationMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ObservationMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::observation_matrix_H::<5, 3>().new();
    ///
    /// let buffer: ObservationMatrixMutBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> ObservationMatrixBufferOwnedType<OBSERVATIONS, STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`ObservationMatrixMutBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::ObservationMatrixMutBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::observation_matrix_H::<5, 3>().new_with(0.0);
    ///
    /// let buffer: ObservationMatrixMutBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> ObservationMatrixBufferOwnedType<OBSERVATIONS, STATES, T>
    where
        T: Copy,
    {
        ObservationMatrixMutBuffer::<
            OBSERVATIONS,
            STATES,
            T,
            MatrixDataBoxed<OBSERVATIONS, STATES, T>,
        >::new(MatrixData::new_boxed::<OBSERVATIONS, STATES, T, _>(
            alloc::vec![init; OBSERVATIONS * STATES],
        ))
    }
}

/// The type of observation / process noise covariance matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type MeasurementNoiseCovarianceBufferOwnedType<const OBSERVATIONS: usize, T> =
    MeasurementNoiseCovarianceMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataBoxed<OBSERVATIONS, OBSERVATIONS, T>,
    >;

impl<const OBSERVATIONS: usize> MeasurementNoiseCovarianceMatrixBufferBuilder<OBSERVATIONS> {
    /// Builds a new [`MeasurementNoiseCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementNoiseCovarianceMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::observation_covariance_R::<5>().new();
    ///
    /// let buffer: MeasurementNoiseCovarianceMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> MeasurementNoiseCovarianceBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`MeasurementNoiseCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::MeasurementNoiseCovarianceMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::observation_covariance_R::<5>().new_with(0.0);
    ///
    /// let buffer: MeasurementNoiseCovarianceMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> MeasurementNoiseCovarianceBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy,
    {
        MeasurementNoiseCovarianceMatrixBuffer::<
            OBSERVATIONS,
            T,
            MatrixDataBoxed<OBSERVATIONS, OBSERVATIONS, T>,
        >::new(MatrixData::new_boxed::<OBSERVATIONS, OBSERVATIONS, T, _>(
            alloc::vec![init; OBSERVATIONS * OBSERVATIONS],
        ))
    }
}

/// The type of innovation vector buffers. In Extended Kalman Filters, this buffer
/// also acts as a temporary target for nonlinear measurement transformations.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type InnovationVectorBufferOwnedType<const OBSERVATIONS: usize, T> =
    InnovationVectorBuffer<OBSERVATIONS, T, MatrixDataArray<OBSERVATIONS, 1, OBSERVATIONS, T>>;

impl<const OBSERVATIONS: usize> InnovationVectorBufferBuilder<OBSERVATIONS> {
    /// Builds a new [`InnovationVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::InnovationVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::innovation_vector_y::<5>().new();
    ///
    /// let buffer: InnovationVectorBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 5);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> InnovationVectorBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`InnovationVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::InnovationVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::innovation_vector_y::<5>().new_with(0.0);
    ///
    /// let buffer: InnovationVectorBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 5);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> InnovationVectorBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy,
    {
        InnovationVectorBuffer::<OBSERVATIONS, T, MatrixDataArray<OBSERVATIONS, 1, OBSERVATIONS, T>>::new(
            MatrixData::new_array::<OBSERVATIONS, 1, OBSERVATIONS, T>([init; OBSERVATIONS]),
        )
    }
}

/// The type of innovation residual covariance matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type InnovationResidualCovarianceMatrixBufferOwnedType<const OBSERVATIONS: usize, T> =
    InnovationCovarianceMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataBoxed<OBSERVATIONS, OBSERVATIONS, T>,
    >;

impl<const OBSERVATIONS: usize> InnovationResidualCovarianceMatrixBufferBuilder<OBSERVATIONS> {
    /// Builds a new [`InnovationCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::InnovationCovarianceMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::innovation_covariance_S::<5>().new();
    ///
    /// let buffer: InnovationCovarianceMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> InnovationResidualCovarianceMatrixBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`InnovationCovarianceMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::InnovationCovarianceMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::innovation_covariance_S::<5>().new_with(0.0);
    ///
    /// let buffer: InnovationCovarianceMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(
        &self,
        init: T,
    ) -> InnovationResidualCovarianceMatrixBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy,
    {
        InnovationCovarianceMatrixBuffer::<
            OBSERVATIONS,
            T,
            MatrixDataBoxed<OBSERVATIONS, OBSERVATIONS, T>,
        >::new(MatrixData::new_boxed::<OBSERVATIONS, OBSERVATIONS, T, _>(
            alloc::vec![init; OBSERVATIONS * OBSERVATIONS],
        ))
    }
}

/// The type of innovation residual covariance matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type KalmanGainMatrixBufferOwnedType<const STATES: usize, const OBSERVATIONS: usize, T> =
    KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, MatrixDataBoxed<STATES, OBSERVATIONS, T>>;

impl<const STATES: usize, const OBSERVATIONS: usize>
    KalmanGainMatrixBufferBuilder<STATES, OBSERVATIONS>
{
    /// Builds a new [`KalmanGainMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::KalmanGainMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::kalman_gain_K::<3, 5>().new();
    ///
    /// let buffer: KalmanGainMatrixBuffer<3, 5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`KalmanGainMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::KalmanGainMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::kalman_gain_K::<3, 5>().new_with(0.0);
    ///
    /// let buffer: KalmanGainMatrixBuffer<3, 5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>
    where
        T: Copy,
    {
        KalmanGainMatrixBuffer::<
            STATES,
            OBSERVATIONS,
            T,
            MatrixDataBoxed<STATES, OBSERVATIONS, T>,
        >::new(MatrixData::new_boxed::<STATES, OBSERVATIONS, T, _>(
            alloc::vec![init; STATES * OBSERVATIONS],
        ))
    }
}

/// The type of owned state vector buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryStatePredictionVectorBufferOwnedType<const STATES: usize, T> =
    PredictedStateEstimateVectorBuffer<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>;

impl<const STATES: usize> StatePredictionVectorBufferBuilder<STATES> {
    /// Builds a new [`PredictedStateEstimateVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::PredictedStateEstimateVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::state_prediction_temp_x::<3>().new();
    ///
    /// let buffer: PredictedStateEstimateVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporaryStatePredictionVectorBufferOwnedType<STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`PredictedStateEstimateVectorBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::PredictedStateEstimateVectorBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::state_prediction_temp_x::<3>().new_with(0.0);
    ///
    /// let buffer: PredictedStateEstimateVectorBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporaryStatePredictionVectorBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        PredictedStateEstimateVectorBuffer::<STATES, T, MatrixDataArray<STATES, 1, STATES, T>>::new(
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
    /// let buffer = BufferBuilder::temp_system_covariance_P::<3>().new();
    ///
    /// let buffer: TemporaryStateMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporaryStateMatrixBufferOwnedType<STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`TemporaryStateMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryStateMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_system_covariance_P::<3>().new_with(0.0);
    ///
    /// let buffer: TemporaryStateMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporaryStateMatrixBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        TemporaryStateMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(alloc::vec![init; STATES * STATES]),
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
    /// let buffer = BufferBuilder::temp_BQ::<3, 2>().new();
    ///
    /// let buffer: TemporaryBQMatrixBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporaryBQMatrixBufferOwnedType<STATES, CONTROLS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`TemporaryBQMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryBQMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_BQ::<3, 2>().new_with(0.0);
    ///
    /// let buffer: TemporaryBQMatrixBuffer<3, 2, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 6);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporaryBQMatrixBufferOwnedType<STATES, CONTROLS, T>
    where
        T: Copy,
    {
        TemporaryBQMatrixBuffer::<STATES, CONTROLS, T, MatrixDataBoxed<STATES, CONTROLS, T>>::new(
            MatrixData::new_boxed::<STATES, CONTROLS, T, _>(alloc::vec![init; STATES * CONTROLS]),
        )
    }
}

/// The type of temporary S-inverted matrix buffers.
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporarySInvertedMatrixBufferOwnedType<const OBSERVATIONS: usize, T> =
    TemporaryResidualCovarianceInvertedMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataBoxed<OBSERVATIONS, OBSERVATIONS, T>,
    >;

impl<const OBSERVATIONS: usize> TemporarySInvMatrixBufferBuilder<OBSERVATIONS> {
    /// Builds a new [`TemporaryResidualCovarianceInvertedMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_S_inv::<5>().new();
    ///
    /// let buffer: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporarySInvertedMatrixBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`TemporaryResidualCovarianceInvertedMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryResidualCovarianceInvertedMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_S_inv::<5>().new_with(0.0);
    ///
    /// let buffer: TemporaryResidualCovarianceInvertedMatrixBuffer<5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 25);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporarySInvertedMatrixBufferOwnedType<OBSERVATIONS, T>
    where
        T: Copy,
    {
        TemporaryResidualCovarianceInvertedMatrixBuffer::<
            OBSERVATIONS,
            T,
            MatrixDataBoxed<OBSERVATIONS, OBSERVATIONS, T>,
        >::new(MatrixData::new_boxed::<OBSERVATIONS, OBSERVATIONS, T, _>(
            alloc::vec![init; OBSERVATIONS * OBSERVATIONS],
        ))
    }
}

/// The type of temporary H×P matrix buffers, tandem to [`TemporaryPHtMatrixBufferOwnedType`].
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryHPMatrixBufferOwnedType<const OBSERVATIONS: usize, const STATES: usize, T> =
    TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, MatrixDataBoxed<OBSERVATIONS, STATES, T>>;

impl<const OBSERVATIONS: usize, const STATES: usize>
    TemporaryHPMatrixBufferBuilder<OBSERVATIONS, STATES>
{
    /// Builds a new [`TemporaryHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryHPMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_HP::<5, 3>().new();
    ///
    /// let buffer: TemporaryHPMatrixBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporaryHPMatrixBufferOwnedType<OBSERVATIONS, STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`TemporaryHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryHPMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_HP::<5, 3>().new_with(0.0);
    ///
    /// let buffer: TemporaryHPMatrixBuffer<5, 3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporaryHPMatrixBufferOwnedType<OBSERVATIONS, STATES, T>
    where
        T: Copy,
    {
        TemporaryHPMatrixBuffer::<
            OBSERVATIONS,
            STATES,
            T,
            MatrixDataBoxed<OBSERVATIONS, STATES, T>,
        >::new(MatrixData::new_boxed::<OBSERVATIONS, STATES, T, _>(
            alloc::vec![init; OBSERVATIONS * STATES],
        ))
    }
}

/// The type of temporary P×Hᵀ matrix buffers, tandem to [`TemporaryHPMatrixBufferOwnedType`].
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub type TemporaryPHtMatrixBufferOwnedType<const STATES: usize, const OBSERVATIONS: usize, T> =
    TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, MatrixDataBoxed<STATES, OBSERVATIONS, T>>;

impl<const STATES: usize, const OBSERVATIONS: usize>
    TemporaryPHtMatrixBufferBuilder<STATES, OBSERVATIONS>
{
    /// Builds a new [`TemporaryPHTMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryPHTMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_PHt::<3, 5>().new();
    ///
    /// let buffer: TemporaryPHTMatrixBuffer<3, 5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporaryPHtMatrixBufferOwnedType<STATES, OBSERVATIONS, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`TemporaryPHTMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryPHTMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer = BufferBuilder::temp_PHt::<3, 5>().new_with(0.0);
    ///
    /// let buffer: TemporaryPHTMatrixBuffer<3, 5, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 15);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporaryPHtMatrixBufferOwnedType<STATES, OBSERVATIONS, T>
    where
        T: Copy,
    {
        TemporaryPHTMatrixBuffer::<
            STATES,
            OBSERVATIONS,
            T,
            MatrixDataBoxed<STATES, OBSERVATIONS, T>,
        >::new(MatrixData::new_boxed::<STATES, OBSERVATIONS, T, _>(
            alloc::vec![init; STATES * OBSERVATIONS],
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
    /// let buffer  = BufferBuilder::temp_KHP::<3>().new();
    ///
    /// let buffer: TemporaryKHPMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new<T>(&self) -> TemporaryKHPMatrixBufferOwnedType<STATES, T>
    where
        T: Copy + Default,
    {
        self.new_with(T::default())
    }

    /// Builds a new [`TemporaryKHPMatrixBuffer`] that owns its data.
    ///
    /// ## Example
    /// ```
    /// use minikalman::buffers::types::TemporaryKHPMatrixBuffer;
    /// use minikalman::prelude::*;
    ///
    /// let buffer  = BufferBuilder::temp_KHP::<3>().new_with(0.0);
    ///
    /// let buffer: TemporaryKHPMatrixBuffer<3, f32, _> = buffer;
    /// assert_eq!(buffer.len(), 9);
    /// ```
    #[allow(clippy::new_ret_no_self, clippy::wrong_self_convention)]
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub fn new_with<T>(&self, init: T) -> TemporaryKHPMatrixBufferOwnedType<STATES, T>
    where
        T: Copy,
    {
        TemporaryKHPMatrixBuffer::<STATES, T, MatrixDataBoxed<STATES, STATES, T>>::new(
            MatrixData::new_boxed::<STATES, STATES, T, _>(alloc::vec![init; STATES * STATES]),
        )
    }
}
