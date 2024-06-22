use core::marker::PhantomData;

use crate::matrix::MatrixDataType;

use crate::buffers::builder::*;

use crate::regular::{
    Control, ControlBuilder, RegularKalman, RegularKalmanBuilder, RegularObservation,
    RegularObservationBuilder,
};

/// A simple builder for [`RegularKalman`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Control`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterControlBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`RegularObservation`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterObservationBuilder<const STATES: usize, T>(PhantomData<T>);

impl<const STATES: usize, T> Default for KalmanFilterBuilder<STATES, T> {
    fn default() -> Self {
        KalmanFilterBuilder::new()
    }
}

/// The type of Kalman filters with owned buffers.
///
/// See also the [`KalmanFilter`](crate::kalman::KalmanFilter) trait.
pub type KalmanFilterType<const STATES: usize, T> = RegularKalman<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    EstimateCovarianceMatrixBufferOwnedType<STATES, T>,
    TemporaryStatePredictionVectorBufferOwnedType<STATES, T>,
    TemporaryStateMatrixBufferOwnedType<STATES, T>,
>;

impl<const STATES: usize, T> KalmanFilterBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    /// const NUM_OBSERVATIONS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut filter = builder.build();
    /// let mut control = builder.controls().build::<NUM_CONTROLS>();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    /// ```
    ///
    /// See also [`KalmanFilterControlBuilder`] and [`KalmanFilterObservationBuilder`] for further information.
    pub fn build(&self) -> KalmanFilterType<STATES, T>
    where
        T: MatrixDataType + Default,
    {
        // System buffers.
        let state_vector = BufferBuilder::state_vector_x::<STATES>().new();
        let system_matrix = BufferBuilder::system_matrix_A::<STATES>().new();
        let system_covariance = BufferBuilder::estimate_covariance_P::<STATES>().new();

        // Filter temporaries.
        let temp_x = BufferBuilder::state_prediction_temp_x::<STATES>().new();
        let temp_p = BufferBuilder::temp_system_covariance_P::<STATES>().new();

        RegularKalmanBuilder::new::<STATES, T>(
            system_matrix,
            state_vector,
            system_covariance,
            temp_x,
            temp_p,
        )
    }

    /// Convenience function to return a [`KalmanFilterControlBuilder`].
    pub fn controls(&self) -> KalmanFilterControlBuilder<STATES, T> {
        Default::default()
    }

    /// Convenience function to return a [`KalmanFilterObservationBuilder`].
    pub fn observations(&self) -> KalmanFilterObservationBuilder<STATES, T> {
        Default::default()
    }
}

impl<const STATES: usize, T> Default for KalmanFilterControlBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The type of Kalman filter controls with owned buffers.
///
/// See also the [`KalmanFilterControl`](crate::kalman::KalmanFilterControl) trait.
pub type KalmanFilterControlType<const STATES: usize, const CONTROLS: usize, T> = Control<
    STATES,
    CONTROLS,
    T,
    ControlMatrixBufferOwnedType<STATES, CONTROLS, T>,
    ControlVectorBufferOwnedType<CONTROLS, T>,
    ControlCovarianceMatrixBufferOwnedType<CONTROLS, T>,
    TemporaryBQMatrixBufferOwnedType<STATES, CONTROLS, T>,
>;

impl<const STATES: usize, T> KalmanFilterControlBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterControlBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter control input using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut control = builder.controls().build::<NUM_CONTROLS>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterObservationBuilder`] for further information.
    pub fn build<const CONTROLS: usize>(&self) -> KalmanFilterControlType<STATES, CONTROLS, T>
    where
        T: MatrixDataType + Default,
    {
        // Control buffers.
        let control_vector = BufferBuilder::control_vector_u::<CONTROLS>().new();
        let control_matrix = BufferBuilder::control_matrix_B::<STATES, CONTROLS>().new();
        let control_covariance = BufferBuilder::process_noise_covariance_Q::<CONTROLS>().new();

        // Control temporaries.
        let temp_bq = BufferBuilder::temp_BQ::<STATES, CONTROLS>().new();

        ControlBuilder::new::<STATES, CONTROLS, T>(
            control_matrix,
            control_vector,
            control_covariance,
            temp_bq,
        )
    }
}

impl<const STATES: usize, T> Default for KalmanFilterObservationBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The type of Kalman filter measurement / observation with owned buffers.
///
/// See also the [`KalmanFilterObservation`](crate::kalman::KalmanFilterObservation) trait.
pub type KalmanFilterObservationType<const STATES: usize, const OBSERVATIONS: usize, T> =
    RegularObservation<
        STATES,
        OBSERVATIONS,
        T,
        ObservationMatrixBufferOwnedType<OBSERVATIONS, STATES, T>,
        ObservationVectorBufferOwnedType<OBSERVATIONS, T>,
        MeasurementNoiseCovarianceBufferOwnedType<OBSERVATIONS, T>,
        InnovationVectorBufferOwnedType<OBSERVATIONS, T>,
        InnovationResidualCovarianceMatrixBufferOwnedType<OBSERVATIONS, T>,
        KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
        TemporarySInvertedMatrixBufferOwnedType<OBSERVATIONS, T>,
        TemporaryHPMatrixBufferOwnedType<OBSERVATIONS, STATES, T>,
        TemporaryPHtMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
        TemporaryKHPMatrixBufferOwnedType<STATES, T>,
    >;

impl<const STATES: usize, T> KalmanFilterObservationBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterObservationBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter measurements using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_OBSERVATIONS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterControlBuilder`] for further information.
    pub fn build<const OBSERVATIONS: usize>(
        &self,
    ) -> KalmanFilterObservationType<STATES, OBSERVATIONS, T>
    where
        T: MatrixDataType + Default,
    {
        // Observation buffers.
        let measurement_vector = BufferBuilder::measurement_vector_z::<OBSERVATIONS>().new();
        let observation_matrix =
            BufferBuilder::observation_matrix_H::<OBSERVATIONS, STATES>().new();
        let observation_covariance =
            BufferBuilder::observation_covariance_R::<OBSERVATIONS>().new();
        let innovation_vector = BufferBuilder::innovation_vector_y::<OBSERVATIONS>().new();
        let residual_covariance_matrix =
            BufferBuilder::innovation_covariance_S::<OBSERVATIONS>().new();
        let kalman_gain = BufferBuilder::kalman_gain_K::<STATES, OBSERVATIONS>().new();

        // Observation temporaries.
        let temp_s_inverted = BufferBuilder::temp_S_inv::<OBSERVATIONS>().new();
        let temp_hp = BufferBuilder::temp_HP::<OBSERVATIONS, STATES>().new();
        let temp_pht = BufferBuilder::temp_PHt::<STATES, OBSERVATIONS>().new();
        let temp_khp = BufferBuilder::temp_KHP::<STATES>().new();

        RegularObservationBuilder::new::<STATES, OBSERVATIONS, T>(
            observation_matrix,
            measurement_vector,
            observation_covariance,
            innovation_vector,
            residual_covariance_matrix,
            kalman_gain,
            temp_s_inverted,
            temp_hp,
            temp_pht,
            temp_khp,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kalman::{KalmanFilter, KalmanFilterControl, KalmanFilterObservation};

    const NUM_STATES: usize = 3; // height, upwards velocity, upwards acceleration
    const NUM_CONTROLS: usize = 1; // constant velocity
    const NUM_OBSERVATIONS: usize = 1; // position

    fn accept_filter<F, T>(_filter: F)
    where
        F: KalmanFilter<NUM_STATES, T>,
    {
    }

    fn accept_control<I, T>(_control: I)
    where
        I: KalmanFilterControl<NUM_STATES, NUM_CONTROLS, T>,
    {
    }

    fn accept_observation<M, T>(_measurement: M)
    where
        M: KalmanFilterObservation<NUM_STATES, NUM_OBSERVATIONS, T>,
    {
    }

    #[test]
    fn kalman_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let filter = builder.build();
        assert_eq!(filter.states(), NUM_STATES);
        accept_filter(filter);
    }

    #[test]
    fn control_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let control = builder.controls().build::<NUM_CONTROLS>();
        assert_eq!(control.states(), NUM_STATES);
        assert_eq!(control.controls(), NUM_CONTROLS);
        accept_control(control);
    }

    #[test]
    fn measurement_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let measurement = builder.observations().build::<NUM_OBSERVATIONS>();
        assert_eq!(measurement.states(), NUM_STATES);
        assert_eq!(measurement.measurements(), NUM_OBSERVATIONS);
        accept_observation(measurement);
    }
}
