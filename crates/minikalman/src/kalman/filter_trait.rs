use crate::kalman::{
    ControlCovarianceMatrix, ControlCovarianceMatrixMut, ControlMatrix, ControlMatrixMut,
    ControlVector, ControlVectorMut, MeasurementObservationMatrix, MeasurementObservationMatrixMut,
    MeasurementProcessNoiseCovarianceMatrix, MeasurementVector, MeasurementVectorMut, StateVector,
    StateVectorMut, SystemCovarianceMatrix, SystemMatrix, SystemMatrixMut,
};

pub trait KalmanFilter<const STATES: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterStateVectorMut<STATES, T>
    + KalmanFilterStateTransition<STATES, T>
    + KalmanFilterSystemCovarianceMut<STATES, T>
    + KalmanFilterPredict<STATES, T>
    + KalmanFilterApplyControl<STATES, T>
    + KalmanFilterUpdate<STATES, T>
{
}

pub trait KalmanFilterControl<const STATES: usize, const CONTROLS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumControls<CONTROLS>
    + KalmanFilterControlVectorMut<CONTROLS, T>
    + KalmanFilterControlTransition<STATES, CONTROLS, T>
    + KalmanFilterControlCovarianceMut<CONTROLS, T>
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
        + KalmanFilterApplyControl<STATES, T>
        + KalmanFilterUpdate<STATES, T>
{
}

/// Auto-implementation of [`KalmanFilterControl`] for types that implement all necessary traits.
impl<const STATES: usize, const CONTROLS: usize, T, Control>
    KalmanFilterControl<STATES, CONTROLS, T> for Control
where
    Control: KalmanFilterNumStates<STATES>
        + KalmanFilterNumControls<CONTROLS>
        + KalmanFilterControlVectorMut<CONTROLS, T>
        + KalmanFilterControlTransition<STATES, CONTROLS, T>
        + KalmanFilterControlCovarianceMut<CONTROLS, T>
        + KalmanFilterControlApplyToFilter<STATES, T>,
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
        + KalmanFilterMeasurementCorrectFilter<STATES, T>,
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
    /// This call assumes that the control covariance and variables are already set in the filter structure.
    #[doc(alias = "kalman_predict")]
    fn predict(&mut self);
}

pub trait KalmanFilterApplyControl<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    fn control<const CONTROLS: usize, I>(&mut self, control: &mut I)
    where
        I: KalmanFilterControlApplyToFilter<STATES, T> + KalmanFilterNumControls<CONTROLS>;
}

pub trait KalmanFilterUpdate<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    fn correct<const MEASUREMENTS: usize, M>(&mut self, measurement: &mut M)
    where
        M: KalmanFilterMeasurementCorrectFilter<STATES, T>
            + KalmanFilterNumMeasurements<MEASUREMENTS>;
}

pub trait KalmanFilterStateVector<const STATES: usize, T> {
    type StateVector: StateVector<STATES, T>;

    /// Gets a reference to the state vector x.
    fn state_vector_ref(&self) -> &Self::StateVector;
}

pub trait KalmanFilterStateVectorMut<const STATES: usize, T>:
    KalmanFilterStateVector<STATES, T>
{
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

pub trait KalmanFilterNumControls<const CONTROLS: usize> {
    /// The number of controls.
    const NUM_CONTROLS: usize = CONTROLS;

    /// Returns the number of controls.
    fn controls(&self) -> usize {
        CONTROLS
    }
}

pub trait KalmanFilterControlApplyToFilter<const STATES: usize, T> {
    /// Applies an control to the state.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    #[allow(non_snake_case)]
    fn apply_to<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>;
}

pub trait KalmanFilterControlVector<const CONTROLS: usize, T> {
    type ControlVector: ControlVector<CONTROLS, T>;

    /// Gets a reference to the control vector u.
    fn control_vector_ref(&self) -> &Self::ControlVector;
}

pub trait KalmanFilterControlVectorMut<const CONTROLS: usize, T>:
    KalmanFilterControlVector<CONTROLS, T>
{
    type ControlVectorMut: ControlVectorMut<CONTROLS, T>;

    /// Gets a mutable reference to the control vector u.
    #[doc(alias = "kalman_get_control_vector")]
    fn control_vector_mut(&mut self) -> &mut Self::ControlVectorMut;

    /// Applies a function to the control vector u.
    #[inline(always)]
    fn control_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::ControlVectorMut),
    {
        f(self.control_vector_mut())
    }
}

pub trait KalmanFilterControlTransition<const STATES: usize, const CONTROLS: usize, T> {
    type ControlTransitionMatrix: ControlMatrix<STATES, CONTROLS, T>;

    /// Gets a reference to the control transition matrix B.
    fn control_transition_ref(&self) -> &Self::ControlTransitionMatrix;
}

pub trait KalmanFilterControlTransitionMut<const STATES: usize, const CONTROLS: usize, T>:
    KalmanFilterControlTransition<STATES, CONTROLS, T>
{
    type ControlTransitionMatrixMut: ControlMatrixMut<STATES, CONTROLS, T>;

    /// Gets a mutable reference to the control transition matrix B.
    #[doc(alias = "kalman_get_control_transition")]
    fn control_transition_mut(&mut self) -> &mut Self::ControlTransitionMatrixMut;

    /// Applies a function to the control transition matrix B.
    #[inline(always)]
    fn control_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::ControlTransitionMatrixMut),
    {
        f(self.control_transition_mut())
    }
}

pub trait KalmanFilterControlCovariance<const CONTROLS: usize, T> {
    type ControlCovarianceMatrix: ControlCovarianceMatrix<CONTROLS, T>;

    /// Gets a reference to the control covariance matrix Q.
    fn control_covariance_ref(&self) -> &Self::ControlCovarianceMatrix;
}

pub trait KalmanFilterControlCovarianceMut<const CONTROLS: usize, T>:
    KalmanFilterControlCovariance<CONTROLS, T>
{
    type ControlCovarianceMatrixMut: ControlCovarianceMatrixMut<CONTROLS, T>;

    /// Gets a mutable reference to the control covariance matrix Q.
    #[doc(alias = "kalman_get_control_covariance")]
    fn control_covariance_mut(&mut self) -> &mut Self::ControlCovarianceMatrixMut;

    /// Applies a function to the control covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_control_covariance")]
    fn control_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::ControlCovarianceMatrixMut),
    {
        f(self.control_covariance_mut())
    }
}

pub trait KalmanFilterNumMeasurements<const MEASUREMENTS: usize> {
    /// The number of measurements.
    const NUM_MEASUREMENTS: usize = MEASUREMENTS;

    /// Returns the number of controls.
    fn measurements(&self) -> usize {
        MEASUREMENTS
    }
}

pub trait KalmanFilterMeasurementCorrectFilter<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    #[allow(non_snake_case)]
    fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>;
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
    type MeasurementTransformationMatrix: MeasurementObservationMatrix<MEASUREMENTS, STATES, T>;

    /// Gets a reference to the measurement transformation matrix H.
    fn measurement_transformation_ref(&self) -> &Self::MeasurementTransformationMatrix;
}

pub trait KalmanFilterMeasurementTransformationMut<
    const STATES: usize,
    const MEASUREMENTS: usize,
    T,
>: KalmanFilterMeasurementTransformation<STATES, MEASUREMENTS, T>
{
    type MeasurementTransformationMatrixMut: MeasurementObservationMatrixMut<
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
