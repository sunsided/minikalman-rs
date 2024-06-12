use std::marker::PhantomData;

use minikalman_traits::kalman::{
    InnovationVector, InputCovarianceMatrixMut, InputMatrixMut, InputVectorMut, KalmanGainMatrix,
    MeasurementObservationMatrixMut, MeasurementProcessNoiseCovarianceMatrix, MeasurementVectorMut,
    ResidualCovarianceMatrix, StatePredictionVector, StateVector, SystemCovarianceMatrix,
    SystemMatrixMut, TemporaryBQMatrix, TemporaryHPMatrix, TemporaryKHPMatrix, TemporaryPHTMatrix,
    TemporaryResidualCovarianceInvertedMatrix, TemporaryStateMatrix,
};
use minikalman_traits::matrix::MatrixDataType;

use crate::inputs::{Input, InputBuilder};
use crate::{BufferBuilder, Kalman, KalmanBuilder, Measurement, MeasurementBuilder};

#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

#[derive(Copy, Clone)]
pub struct KalmanFilterMeasurementBuilder<const STATES: usize, T>(PhantomData<T>);

#[derive(Copy, Clone)]
pub struct KalmanFilterInputBuilder<const STATES: usize, T>(PhantomData<T>);

impl<const STATES: usize, T> Default for KalmanFilterBuilder<STATES, T> {
    fn default() -> Self {
        KalmanFilterBuilder(PhantomData)
    }
}

impl<const STATES: usize, T> KalmanFilterBuilder<STATES, T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    pub fn build(
        &self,
    ) -> Kalman<
        STATES,
        T,
        impl SystemMatrixMut<STATES, T>,
        impl StateVector<STATES, T>,
        impl SystemCovarianceMatrix<STATES, T>,
        impl StatePredictionVector<STATES, T>,
        impl TemporaryStateMatrix<STATES, T>,
    >
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

    pub fn measurements(&self) -> KalmanFilterMeasurementBuilder<STATES, T> {
        Default::default()
    }

    pub fn inputs(&self) -> KalmanFilterInputBuilder<STATES, T> {
        Default::default()
    }
}

impl<const STATES: usize, T> Default for KalmanFilterInputBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const STATES: usize, T> KalmanFilterInputBuilder<STATES, T> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

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
    pub fn new() -> Self {
        Self(PhantomData)
    }

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
