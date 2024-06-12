use core::marker::PhantomData;

use crate::matrix::MatrixDataType;

use crate::buffers::builder::*;
use crate::controls::{Control, ControlBuilder};
use crate::{BufferBuilder, Kalman, KalmanBuilder, Measurement, MeasurementBuilder};

/// A simple builder for [`Kalman`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Control`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterControlBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Measurement`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterMeasurementBuilder<const STATES: usize, T>(PhantomData<T>);

impl<const STATES: usize, T> Default for KalmanFilterBuilder<STATES, T> {
    fn default() -> Self {
        KalmanFilterBuilder::new()
    }
}

/// The type of Kalman filters with owned buffers.
///
/// See also the [`KalmanFilter`](minikalman::kalman::KalmanFilter) trait.
pub type KalmanFilterType<const STATES: usize, T> = Kalman<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    SystemCovarianceMatrixBufferOwnedType<STATES, T>,
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
    /// use minikalman::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    /// const NUM_MEASUREMENTS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut filter = builder.build();
    /// let mut control = builder.controls().build::<NUM_CONTROLS>();
    /// let mut measurement = builder.measurements().build::<NUM_MEASUREMENTS>();
    /// ```
    ///
    /// See also [`KalmanFilterControlBuilder`] and [`KalmanFilterMeasurementBuilder`] for further information.
    pub fn build(&self) -> KalmanFilterType<STATES, T>
    where
        T: MatrixDataType,
    {
        // The initialization value.
        let zero = T::zero();

        // System buffers.
        let state_vector = BufferBuilder::state_vector_x::<STATES>().new(zero);
        let system_matrix = BufferBuilder::system_state_transition_A::<STATES>().new(zero);
        let system_covariance = BufferBuilder::system_covariance_P::<STATES>().new(zero);

        // Filter temporaries.
        let temp_x = BufferBuilder::state_prediction_temp_x::<STATES>().new(zero);
        let temp_p = BufferBuilder::temp_system_covariance_P::<STATES>().new(zero);

        KalmanBuilder::new::<STATES, T>(
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

    /// Convenience function to return a [`KalmanFilterMeasurementBuilder`].
    pub fn measurements(&self) -> KalmanFilterMeasurementBuilder<STATES, T> {
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
/// See also the [`KalmanFilterControl`](minikalman::kalman::KalmanFilterControl) trait.
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

    /// Builds a new Kalman filter controls using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut control = builder.controls().build::<NUM_CONTROLS>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterMeasurementBuilder`] for further information.
    pub fn build<const CONTROLS: usize>(&self) -> KalmanFilterControlType<STATES, CONTROLS, T>
    where
        T: MatrixDataType,
    {
        // The initialization value.
        let zero = T::zero();

        // Control buffers.
        let control_vector = BufferBuilder::control_vector_u::<CONTROLS>().new(zero);
        let control_transition =
            BufferBuilder::control_transition_B::<STATES, CONTROLS>().new(zero);
        let control_covariance = BufferBuilder::control_covariance_Q::<CONTROLS>().new(zero);

        // Control temporaries.
        let temp_bq = BufferBuilder::temp_BQ::<STATES, CONTROLS>().new(zero);

        ControlBuilder::new::<STATES, CONTROLS, T>(
            control_transition,
            control_vector,
            control_covariance,
            temp_bq,
        )
    }
}

impl<const STATES: usize, T> Default for KalmanFilterMeasurementBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The type of Kalman filter measurements with owned buffers.
///
/// See also the [`KalmanFilterMeasurement`](minikalman::kalman::KalmanFilterMeasurement) trait.
pub type KalmanFilterMeasurementType<const STATES: usize, const MEASUREMENTS: usize, T> =
    Measurement<
        STATES,
        MEASUREMENTS,
        T,
        ObservationMatrixBufferOwnedType<MEASUREMENTS, STATES, T>,
        ObservationVectorBufferOwnedType<MEASUREMENTS, T>,
        ObservationCovarianceBufferOwnedType<MEASUREMENTS, T>,
        InnovationVectorBufferOwnedType<MEASUREMENTS, T>,
        InnovationResidualCovarianceMatrixBufferOwnedType<MEASUREMENTS, T>,
        KalmanGainMatrixBufferOwnedType<STATES, MEASUREMENTS, T>,
        TemporarySInvertedMatrixBufferOwnedType<MEASUREMENTS, T>,
        TemporaryHPMatrixBufferOwnedType<MEASUREMENTS, STATES, T>,
        TemporaryPHtMatrixBufferOwnedType<STATES, MEASUREMENTS, T>,
        TemporaryKHPMatrixBufferOwnedType<STATES, T>,
    >;

impl<const STATES: usize, T> KalmanFilterMeasurementBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterMeasurementBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter measurements using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_MEASUREMENTS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut measurement = builder.measurements().build::<NUM_MEASUREMENTS>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterControlBuilder`] for further information.
    pub fn build<const MEASUREMENTS: usize>(
        &self,
    ) -> KalmanFilterMeasurementType<STATES, MEASUREMENTS, T>
    where
        T: MatrixDataType,
    {
        // The initialization value.
        let zero = T::zero();

        // Measurement buffers.
        let measurement_vector = BufferBuilder::measurement_vector_z::<MEASUREMENTS>().new(zero);
        let observation_matrix =
            BufferBuilder::measurement_transformation_H::<MEASUREMENTS, STATES>().new(zero);
        let observation_covariance =
            BufferBuilder::measurement_covariance_R::<MEASUREMENTS>().new(zero);
        let innovation_vector = BufferBuilder::innovation_vector_y::<MEASUREMENTS>().new(zero);
        let residual_covariance_matrix =
            BufferBuilder::innovation_covariance_S::<MEASUREMENTS>().new(zero);
        let kalman_gain = BufferBuilder::kalman_gain_K::<STATES, MEASUREMENTS>().new(zero);

        // Measurement temporaries.
        let temp_s_inverted = BufferBuilder::temp_S_inv::<MEASUREMENTS>().new(zero);
        let temp_hp = BufferBuilder::temp_HP::<MEASUREMENTS, STATES>().new(zero);
        let temp_pht = BufferBuilder::temp_PHt::<STATES, MEASUREMENTS>().new(zero);
        let temp_khp = BufferBuilder::temp_KHP::<STATES>().new(zero);

        MeasurementBuilder::new::<STATES, MEASUREMENTS, T>(
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
    use crate::kalman::{KalmanFilter, KalmanFilterControl, KalmanFilterMeasurement};

    const NUM_STATES: usize = 3; // height, upwards velocity, upwards acceleration
    const NUM_CONTROLS: usize = 1; // constant velocity
    const NUM_MEASUREMENTS: usize = 1; // position

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

    fn accept_measurement<M, T>(_measurement: M)
    where
        M: KalmanFilterMeasurement<NUM_STATES, NUM_MEASUREMENTS, T>,
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
        let measurement = builder.measurements().build::<NUM_MEASUREMENTS>();
        assert_eq!(measurement.states(), NUM_STATES);
        assert_eq!(measurement.measurements(), NUM_CONTROLS);
        accept_measurement(measurement);
    }
}
