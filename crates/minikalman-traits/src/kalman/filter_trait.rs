use crate::kalman::{
    InputCovarianceMatrix, InputCovarianceMatrixMut, InputMatrix, InputMatrixMut, InputVector,
    InputVectorMut, MeasurementProcessNoiseCovarianceMatrix, MeasurementTransformationMatrix,
    MeasurementTransformationMatrixMut, MeasurementVector, MeasurementVectorMut, StateVector,
    SystemCovarianceMatrix, SystemMatrix, SystemMatrixMut,
};

pub trait KalmanFilter<const STATES: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterStateVectorMut<STATES, T>
    + KalmanFilterStateTransition<STATES, T>
    + KalmanFilterSystemCovarianceMut<STATES, T>
    + KalmanFilterPredict<STATES, T>
    + KalmanFilterUpdate<STATES, T>
{
}

pub trait KalmanFilterInput<const STATES: usize, const INPUTS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumInputs<INPUTS>
    + KalmanFilterInputVectorMut<INPUTS, T>
    + KalmanFilterInputTransition<STATES, INPUTS, T>
    + KalmanFilterInputCovarianceMut<INPUTS, T>
{
}

pub trait KalmanFilterMeasurement<const STATES: usize, const MEASUREMENTS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumMeasurements<MEASUREMENTS>
    + KalmanFilterMeasurementVectorMut<MEASUREMENTS, T>
    + KalmanFilterMeasurementTransformation<STATES, MEASUREMENTS, T>
    + KalmanFilterMeasurementProcessNoiseMut<MEASUREMENTS, T>
{
}

/// Auto-implementation of [`KalmanFilter`] for types that implement all necessary traits.
impl<const STATES: usize, T, Filter> KalmanFilter<STATES, T> for Filter where
    Filter: KalmanFilterNumStates<STATES>
        + KalmanFilterStateVectorMut<STATES, T>
        + KalmanFilterStateTransition<STATES, T>
        + KalmanFilterSystemCovarianceMut<STATES, T>
        + KalmanFilterPredict<STATES, T>
        + KalmanFilterUpdate<STATES, T>
{
}

/// Auto-implementation of [`KalmanFilterInput`] for types that implement all necessary traits.
impl<const STATES: usize, const INPUTS: usize, T, Input> KalmanFilterInput<STATES, INPUTS, T>
    for Input
where
    Input: KalmanFilterNumStates<STATES>
        + KalmanFilterNumInputs<INPUTS>
        + KalmanFilterInputVectorMut<INPUTS, T>
        + KalmanFilterInputTransition<STATES, INPUTS, T>
        + KalmanFilterInputCovarianceMut<INPUTS, T>,
{
}

/// Auto-implementation of [`KalmanFilterMeasurement`] for types that implement all necessary traits.
impl<const STATES: usize, const MEASUREMENTS: usize, T, Measurement>
    KalmanFilterMeasurement<STATES, MEASUREMENTS, T> for Measurement
where
    Measurement: KalmanFilterNumStates<STATES>
        + KalmanFilterNumMeasurements<MEASUREMENTS>
        + KalmanFilterMeasurementVectorMut<MEASUREMENTS, T>
        + KalmanFilterMeasurementTransformation<STATES, MEASUREMENTS, T>
        + KalmanFilterMeasurementProcessNoiseMut<MEASUREMENTS, T>
        + KalmanFilterMeasurementCorrect<STATES, T>,
{
}

pub trait KalmanFilterNumStates<const STATES: usize> {
    /// The number of states.
    const NUM_STATES: usize = STATES;

    /// Returns the number of states.
    fn states(&self) -> usize {
        STATES
    }
}

pub trait KalmanFilterPredict<const STATES: usize, T> {
    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the input covariance and variables are already set in the filter structure.
    #[doc(alias = "kalman_predict")]
    fn predict(&mut self);
}

pub trait KalmanFilterUpdate<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    fn correct<const MEASUREMENTS: usize, M>(&mut self, measurement: &mut M)
    where
        M: KalmanFilterMeasurementCorrect<STATES, T> + KalmanFilterNumMeasurements<MEASUREMENTS>;
}

pub trait KalmanFilterMeasurementCorrect<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    #[allow(non_snake_case)]
    fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVector<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>;
}

pub trait KalmanFilterStateVector<const STATES: usize, T> {
    type StateVector: StateVector<STATES, T>;

    /// Gets a reference to the state vector x.
    fn state_vector_ref(&self) -> &Self::StateVector;
}

pub trait KalmanFilterStateVectorMut<const STATES: usize, T> {
    type StateVectorMut: StateVector<STATES, T>;

    /// Gets a reference to the state vector x.
    #[doc(alias = "kalman_get_state_vector")]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut;

    /// Applies a function to the state vector x.
    #[inline(always)]
    fn state_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::StateVectorMut),
    {
        f(self.state_vector_mut())
    }
}

pub trait KalmanFilterStateTransition<const STATES: usize, T> {
    type SystemMatrix: SystemMatrix<STATES, T>;

    /// Gets a reference to the state transition matrix A.
    fn state_transition_ref(&self) -> &Self::SystemMatrix;
}

pub trait KalmanFilterStateTransitionMut<const STATES: usize, T>:
    KalmanFilterStateTransition<STATES, T>
{
    type SystemMatrixMut: SystemMatrixMut<STATES, T>;

    /// Gets a reference to the state transition matrix A.
    #[doc(alias = "kalman_get_state_transition")]
    fn state_transition_mut(&mut self) -> &mut Self::SystemMatrixMut;

    /// Applies a function to the state transition matrix A.
    #[inline(always)]
    fn state_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::SystemMatrixMut),
    {
        f(self.state_transition_mut())
    }
}

pub trait KalmanFilterSystemCovariance<const STATES: usize, T> {
    type SystemCovarianceMatrix: SystemCovarianceMatrix<STATES, T>;

    /// Gets a reference to the system covariance matrix P.
    fn system_covariance_ref(&self) -> &Self::SystemCovarianceMatrix;
}

pub trait KalmanFilterSystemCovarianceMut<const STATES: usize, T>:
    KalmanFilterSystemCovariance<STATES, T>
{
    type SystemCovarianceMatrixMut: SystemCovarianceMatrix<STATES, T>;

    /// Gets a mutable reference to the system covariance matrix P.
    #[doc(alias = "kalman_get_system_covariance")]
    fn system_covariance_mut(&mut self) -> &mut Self::SystemCovarianceMatrixMut;

    /// Applies a function to the system covariance matrix P.
    #[inline(always)]
    fn system_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::SystemCovarianceMatrixMut),
    {
        f(self.system_covariance_mut())
    }
}

pub trait KalmanFilterNumInputs<const INPUTS: usize> {
    /// The number of inputs.
    const NUM_INPUTS: usize = INPUTS;

    /// Returns the number of inputs.
    fn inputs(&self) -> usize {
        INPUTS
    }
}

pub trait KalmanFilterInputVector<const INPUTS: usize, T> {
    type InputVector: InputVector<INPUTS, T>;

    /// Gets a reference to the input vector u.
    fn input_vector_ref(&self) -> &Self::InputVector;
}

pub trait KalmanFilterInputVectorMut<const INPUTS: usize, T>:
    KalmanFilterInputVector<INPUTS, T>
{
    type InputVectorMut: InputVectorMut<INPUTS, T>;

    /// Gets a mutable reference to the input vector u.
    #[doc(alias = "kalman_get_input_vector")]
    fn input_vector_mut(&mut self) -> &mut Self::InputVectorMut;

    /// Applies a function to the input vector u.
    #[inline(always)]
    fn input_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::InputVectorMut),
    {
        f(self.input_vector_mut())
    }
}

pub trait KalmanFilterInputTransition<const STATES: usize, const INPUTS: usize, T> {
    type InputTransitionMatrix: InputMatrix<STATES, INPUTS, T>;

    /// Gets a reference to the input transition matrix B.
    fn input_transition_ref(&self) -> &Self::InputTransitionMatrix;
}

pub trait KalmanFilterInputTransitionMut<const STATES: usize, const INPUTS: usize, T>:
    KalmanFilterInputTransition<STATES, INPUTS, T>
{
    type InputTransitionMatrixMut: InputMatrixMut<STATES, INPUTS, T>;

    /// Gets a mutable reference to the input transition matrix B.
    #[doc(alias = "kalman_get_input_transition")]
    fn input_transition_mut(&mut self) -> &mut Self::InputTransitionMatrixMut;

    /// Applies a function to the input transition matrix B.
    #[inline(always)]
    fn input_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::InputTransitionMatrixMut),
    {
        f(self.input_transition_mut())
    }
}

pub trait KalmanFilterInputCovariance<const INPUTS: usize, T> {
    type InputCovarianceMatrix: InputCovarianceMatrix<INPUTS, T>;

    /// Gets a reference to the input covariance matrix Q.
    fn input_covariance_ref(&self) -> &Self::InputCovarianceMatrix;
}

pub trait KalmanFilterInputCovarianceMut<const INPUTS: usize, T>:
    KalmanFilterInputCovariance<INPUTS, T>
{
    type InputCovarianceMatrixMut: InputCovarianceMatrixMut<INPUTS, T>;

    /// Gets a mutable reference to the input covariance matrix Q.
    #[doc(alias = "kalman_get_input_covariance")]
    fn input_covariance_mut(&mut self) -> &mut Self::InputCovarianceMatrixMut;

    /// Applies a function to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    fn input_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::InputCovarianceMatrixMut),
    {
        f(self.input_covariance_mut())
    }
}

pub trait KalmanFilterNumMeasurements<const MEASUREMENTS: usize> {
    /// The number of measurements.
    const NUM_MEASUREMENTS: usize = MEASUREMENTS;

    /// Returns the number of inputs.
    fn measurements(&self) -> usize {
        MEASUREMENTS
    }
}

pub trait KalmanFilterMeasurementVector<const MEASUREMENTS: usize, T> {
    type MeasurementVector: MeasurementVector<MEASUREMENTS, T>;

    /// Gets a reference to the measurement vector z.
    fn measurement_vector_ref(&self) -> &Self::MeasurementVector;
}

pub trait KalmanFilterMeasurementVectorMut<const MEASUREMENTS: usize, T>:
    KalmanFilterMeasurementVector<MEASUREMENTS, T>
{
    type MeasurementVectorMut: MeasurementVectorMut<MEASUREMENTS, T>;

    /// Gets a mutable reference to the measurement vector z.
    #[doc(alias = "kalman_get_measurement_vector")]
    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut;

    #[inline(always)]
    fn measurement_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::MeasurementVectorMut),
    {
        f(self.measurement_vector_mut())
    }
}

pub trait KalmanFilterMeasurementTransformation<const STATES: usize, const MEASUREMENTS: usize, T> {
    type MeasurementTransformationMatrix: MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>;

    /// Gets a reference to the measurement transformation matrix H.
    fn measurement_transformation_ref(&self) -> &Self::MeasurementTransformationMatrix;
}

pub trait KalmanFilterMeasurementTransformationMut<
    const STATES: usize,
    const MEASUREMENTS: usize,
    T,
>: KalmanFilterMeasurementTransformation<STATES, MEASUREMENTS, T>
{
    type MeasurementTransformationMatrixMut: MeasurementTransformationMatrixMut<
        MEASUREMENTS,
        STATES,
        T,
    >;

    /// Gets a mutable reference to the measurement transformation matrix H.
    #[doc(alias = "kalman_get_measurement_transformation")]
    fn measurement_transformation_mut(&mut self) -> &mut Self::MeasurementTransformationMatrixMut;

    /// Applies a function to the measurement transformation matrix H.
    #[inline(always)]
    fn measurement_transformation_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::MeasurementTransformationMatrixMut),
    {
        f(self.measurement_transformation_mut())
    }
}

pub trait KalmanFilterMeasurementProcessNoise<const MEASUREMENTS: usize, T> {
    type MeasurementProcessNoiseMatrix: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>;

    /// Gets a reference to the process noise matrix R.
    fn process_noise_ref(&self) -> &Self::MeasurementProcessNoiseMatrix;
}

pub trait KalmanFilterMeasurementProcessNoiseMut<const MEASUREMENTS: usize, T>:
    KalmanFilterMeasurementProcessNoise<MEASUREMENTS, T>
{
    type MeasurementProcessNoiseMatrixMut: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>;

    /// Gets a mutable reference to the process noise matrix R.
    #[doc(alias = "kalman_get_process_noise")]
    fn process_noise_mut(&mut self) -> &mut Self::MeasurementProcessNoiseMatrixMut;

    /// Applies a function to the process noise matrix R.
    #[inline(always)]
    fn process_noise_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::MeasurementProcessNoiseMatrixMut),
    {
        f(self.process_noise_mut())
    }
}
