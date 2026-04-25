use core::marker::PhantomData;

use crate::matrix::MatrixDataType;

use crate::buffers::builder::*;
use crate::unscented::{UnscentedKalman, UnscentedObservation};

/// A simple builder for [`UnscentedKalman`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`UnscentedObservation`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterObservationBuilder<const STATES: usize, T>(PhantomData<T>);

impl<const STATES: usize, T> Default for KalmanFilterBuilder<STATES, T> {
    fn default() -> Self {
        KalmanFilterBuilder::new()
    }
}

/// The type of Kalman filters with owned buffers.
///
/// See also the [`UnscentedKalmanFilter`](crate::kalman::UnscentedKalmanFilter) trait.
pub type KalmanFilterType<const STATES: usize, const NUM_SIGMA: usize, T> = UnscentedKalman<
    STATES,
    NUM_SIGMA,
    T,
    StateVectorBufferOwnedType<STATES, T>,
    EstimateCovarianceMatrixBufferOwnedType<STATES, T>,
    DirectProcessNoiseCovarianceMatrixBufferOwnedType<STATES, T>,
    TemporaryStatePredictionVectorBufferOwnedType<STATES, T>,
    SigmaPointMatrixBufferOwnedType<STATES, NUM_SIGMA, T>,
    SigmaWeightsVectorBufferOwnedType<NUM_SIGMA, T>,
    SigmaPropagatedMatrixBufferOwnedType<STATES, NUM_SIGMA, T>,
    TempSigmaPMatrixBufferOwnedType<STATES, T>,
>;

impl<const STATES: usize, T> KalmanFilterBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new UKF filter using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::unscented::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut filter = builder.build::<NUM_SIGMA>();
    /// ```
    ///
    /// See also [`KalmanFilterObservationBuilder`] for further information.
    #[allow(non_snake_case)]
    pub fn build<const NUM_SIGMA: usize>(&self) -> KalmanFilterType<STATES, NUM_SIGMA, T>
    where
        T: MatrixDataType
            + Default
            + core::ops::Add<Output = T>
            + core::ops::Mul<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Div<Output = T>
            + num_traits::FromPrimitive
            + PartialOrd,
    {
        // System buffers.
        let state_vector = BufferBuilder::state_vector_x::<STATES>().new();
        let system_covariance = BufferBuilder::estimate_covariance_P::<STATES>().new();
        let process_noise = BufferBuilder::direct_process_noise_covariance_Q::<STATES>().new();

        // Filter temporaries.
        let predicted_x = BufferBuilder::state_prediction_temp_x::<STATES>().new();

        // UKF-specific buffers.
        let sigma_points = BufferBuilder::sigma_point_matrix::<STATES, NUM_SIGMA>().new();
        let sigma_weights = BufferBuilder::sigma_weights_vector::<NUM_SIGMA>().new();
        let sigma_propagated = BufferBuilder::sigma_propagated_matrix::<STATES, NUM_SIGMA>().new();
        let temp_sigma_P = BufferBuilder::temp_sigma_P::<STATES>().new();

        UnscentedKalman::new(
            state_vector,
            system_covariance,
            process_noise,
            predicted_x,
            sigma_points,
            sigma_weights,
            sigma_propagated,
            temp_sigma_P,
            T::one(),
            T::from_usize(2).unwrap_or(T::one()),
            T::one(),
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

/// The type of Kalman filter observation with owned buffers for UKF.
///
/// See also the [`UnscentedObservation`](crate::unscented::UnscentedObservation) type.
pub type KalmanFilterObservationType<
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
> = UnscentedObservation<
    STATES,
    OBSERVATIONS,
    NUM_SIGMA,
    T,
    SigmaObservedMatrixBufferOwnedType<OBSERVATIONS, NUM_SIGMA, T>,
    CrossCovarianceMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
    ObservationVectorBufferOwnedType<OBSERVATIONS, T>,
    MeasurementNoiseCovarianceBufferOwnedType<OBSERVATIONS, T>,
    InnovationVectorBufferOwnedType<OBSERVATIONS, T>,
    InnovationResidualCovarianceMatrixBufferOwnedType<OBSERVATIONS, T>,
    KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
    TemporarySInvertedMatrixBufferOwnedType<OBSERVATIONS, T>,
    TempSigmaPMatrixBufferOwnedType<STATES, T>,
>;

impl<const STATES: usize, T> KalmanFilterObservationBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterObservationBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new UKF observation using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::unscented::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
    /// const NUM_OBSERVATIONS: usize = 2;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS, NUM_SIGMA>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] for further information.
    #[allow(non_snake_case)]
    pub fn build<const OBSERVATIONS: usize, const NUM_SIGMA: usize>(
        &self,
    ) -> KalmanFilterObservationType<STATES, OBSERVATIONS, NUM_SIGMA, T>
    where
        T: MatrixDataType + Default,
    {
        // UKF-specific observation buffers.
        let sigma_observed =
            BufferBuilder::sigma_observed_matrix::<OBSERVATIONS, NUM_SIGMA>().new();
        let cross_covariance =
            BufferBuilder::cross_covariance_matrix::<STATES, OBSERVATIONS>().new();

        // Standard observation buffers.
        let measurement_vector = BufferBuilder::measurement_vector_z::<OBSERVATIONS>().new();
        let observation_covariance =
            BufferBuilder::observation_covariance_R::<OBSERVATIONS>().new();
        let innovation_vector = BufferBuilder::innovation_vector_y::<OBSERVATIONS>().new();
        let residual_covariance_matrix =
            BufferBuilder::innovation_covariance_S::<OBSERVATIONS>().new();
        let kalman_gain = BufferBuilder::kalman_gain_K::<STATES, OBSERVATIONS>().new();
        let temp_s_inverted = BufferBuilder::temp_S_inv::<OBSERVATIONS>().new();
        let temp_P = BufferBuilder::temp_sigma_P::<STATES>().new();

        UnscentedObservation::new(
            measurement_vector,
            observation_covariance,
            innovation_vector,
            residual_covariance_matrix,
            kalman_gain,
            temp_s_inverted,
            sigma_observed,
            cross_covariance,
            temp_P,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kalman::UnscentedKalmanFilter;

    const NUM_STATES: usize = 3;
    const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
    const NUM_OBSERVATIONS: usize = 2;

    fn accept_filter<F, T>(_filter: F)
    where
        F: UnscentedKalmanFilter<NUM_STATES, NUM_SIGMA, T>,
    {
    }

    #[test]
    fn ukf_kalman_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let filter = builder.build::<NUM_SIGMA>();
        assert_eq!(filter.states(), NUM_STATES);
        assert_eq!(filter.num_sigma_points(), NUM_SIGMA);
        accept_filter(filter);
    }

    #[test]
    fn ukf_measurement_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let measurement = builder
            .observations()
            .build::<NUM_OBSERVATIONS, NUM_SIGMA>();
        assert_eq!(measurement.states(), NUM_STATES);
        assert_eq!(measurement.observations(), NUM_OBSERVATIONS);
    }
}
