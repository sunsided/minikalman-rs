use core::marker::PhantomData;

use crate::matrix::MatrixDataType;

use crate::buffers::builder::*;
use crate::extended::{
    ExtendedKalman, ExtendedKalmanBuilder, ExtendedObservationBuilder, Observation,
};

/// A simple builder for [`ExtendedKalman`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Observation`] instances.
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
pub type KalmanFilterType<const STATES: usize, T> = ExtendedKalman<
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
    /// use minikalman::extended::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    /// const NUM_OBSERVATIONS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut filter = builder.build();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    /// ```
    ///
    /// See also [`KalmanFilterObservationBuilder`] for further information.
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

        ExtendedKalmanBuilder::new::<STATES, T>(
            system_matrix,
            state_vector,
            system_covariance,
            temp_x,
            temp_p,
        )
    }

    /// Convenience function to return a [`KalmanFilterObservationBuilder`].
    pub fn observations(&self) -> KalmanFilterObservationBuilder<STATES, T> {
        Default::default()
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
    Observation<
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
    /// use minikalman::extended::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_OBSERVATIONS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] for further information.
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

        ExtendedObservationBuilder::new::<STATES, OBSERVATIONS, T>(
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
    use crate::kalman::{ExtendedKalmanFilter, KalmanFilterObservation};

    const NUM_STATES: usize = 3; // height, upwards velocity, upwards acceleration
    const NUM_OBSERVATIONS: usize = 1; // position

    fn accept_filter<F, T>(_filter: F)
    where
        F: ExtendedKalmanFilter<NUM_STATES, T>,
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
    fn measurement_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let measurement = builder.observations().build::<NUM_OBSERVATIONS>();
        assert_eq!(measurement.states(), NUM_STATES);
        assert_eq!(measurement.measurements(), NUM_OBSERVATIONS);
        accept_observation(measurement);
    }
}
