use core::marker::PhantomData;

use minikalman_traits::kalman::{
    InnovationVector, InputCovarianceMatrixMut, InputMatrixMut, InputVectorMut, KalmanGainMatrix,
    MeasurementObservationMatrixMut, MeasurementProcessNoiseCovarianceMatrix, MeasurementVectorMut,
    ResidualCovarianceMatrix, TemporaryBQMatrix, TemporaryHPMatrix, TemporaryKHPMatrix,
    TemporaryPHTMatrix, TemporaryResidualCovarianceInvertedMatrix,
};
use minikalman_traits::matrix::MatrixDataType;

use crate::buffer_builder::{
    StatePredictionVectorBufferOwnedType, StateVectorBufferOwnedType,
    SystemCovarianceMatrixBufferOwnedType, SystemMatrixMutBufferOwnedType,
    TemporarySystemCovarianceMatrixBufferOwnedType,
};
use crate::inputs::{Input, InputBuilder};
use crate::{BufferBuilder, Kalman, KalmanBuilder, Measurement, MeasurementBuilder};

/// A simple builder for [`Kalman`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Input`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterInputBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Measurement`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterMeasurementBuilder<const STATES: usize, T>(PhantomData<T>);

impl<const STATES: usize, T> Default for KalmanFilterBuilder<STATES, T> {
    fn default() -> Self {
        KalmanFilterBuilder(PhantomData)
    }
}

/// The type of Kalman filters with owned buffers.
pub type KalmanFilterType<const STATES: usize, T> = Kalman<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    SystemCovarianceMatrixBufferOwnedType<STATES, T>,
    StatePredictionVectorBufferOwnedType<STATES, T>,
    TemporarySystemCovarianceMatrixBufferOwnedType<STATES, T>,
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
    /// const NUM_INPUTS: usize = 2;
    /// const NUM_MEASUREMENTS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut filter = builder.build();
    /// let mut input = builder.inputs().build::<NUM_INPUTS>();
    /// let mut measurement = builder.measurements().build::<NUM_MEASUREMENTS>();
    /// ```
    ///
    /// See also [`KalmanFilterInputBuilder`] and [`KalmanFilterMeasurementBuilder`] for further information.
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

    /// Convenience function to return a [`KalmanFilterInputBuilder`].
    pub fn inputs(&self) -> KalmanFilterInputBuilder<STATES, T> {
        Default::default()
    }

    /// Convenience function to return a [`KalmanFilterMeasurementBuilder`].
    pub fn measurements(&self) -> KalmanFilterMeasurementBuilder<STATES, T> {
        Default::default()
    }
}

impl<const STATES: usize, T> Default for KalmanFilterInputBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const STATES: usize, T> KalmanFilterInputBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterInputBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter inputs using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// use minikalman::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_INPUTS: usize = 2;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut input = builder.inputs().build::<NUM_INPUTS>();
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterMeasurementBuilder`] for further information.
    pub fn build<const INPUTS: usize>(
        &self,
    ) -> Input<
        STATES,
        INPUTS,
        T,
        impl InputMatrixMut<STATES, INPUTS, T>,
        impl InputVectorMut<INPUTS, T>,
        impl InputCovarianceMatrixMut<INPUTS, T>,
        impl TemporaryBQMatrix<STATES, INPUTS, T>,
    >
    where
        T: MatrixDataType,
    {
        // The initialization value.
        let zero = T::zero();

        // Input buffers.
        let input_vector = BufferBuilder::input_vector_u::<INPUTS>().new(zero);
        let input_transition = BufferBuilder::input_transition_B::<STATES, INPUTS>().new(zero);
        let input_covariance = BufferBuilder::input_covariance_Q::<INPUTS>().new(zero);

        // Input temporaries.
        let temp_bq = BufferBuilder::temp_BQ::<STATES, INPUTS>().new(zero);

        InputBuilder::new::<STATES, INPUTS, T>(
            input_transition,
            input_vector,
            input_covariance,
            temp_bq,
        )
    }
}

impl<const STATES: usize, T> Default for KalmanFilterMeasurementBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

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
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterInputBuilder`] for further information.
    pub fn build<const MEASUREMENTS: usize>(
        &self,
    ) -> Measurement<
        STATES,
        MEASUREMENTS,
        T,
        impl MeasurementObservationMatrixMut<MEASUREMENTS, STATES, T>,
        impl MeasurementVectorMut<MEASUREMENTS, T>,
        impl MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
        impl InnovationVector<MEASUREMENTS, T>,
        impl ResidualCovarianceMatrix<MEASUREMENTS, T>,
        impl KalmanGainMatrix<STATES, MEASUREMENTS, T>,
        impl TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>,
        impl TemporaryHPMatrix<MEASUREMENTS, STATES, T>,
        impl TemporaryPHTMatrix<STATES, MEASUREMENTS, T>,
        impl TemporaryKHPMatrix<STATES, T>,
    >
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
