use crate::kalman::*;

/// A Kalman filter.
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

/// A Kalman filter control input.
pub trait KalmanFilterControl<const STATES: usize, const CONTROLS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumControls<CONTROLS>
    + KalmanFilterControlVectorMut<CONTROLS, T>
    + KalmanFilterControlTransition<STATES, CONTROLS, T>
    + KalmanFilterControlCovarianceMut<CONTROLS, T>
{
}

/// A Kalman filter observation or measurement.
pub trait KalmanFilterObservation<const STATES: usize, const OBSERVATIONS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumObservations<OBSERVATIONS>
    + KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    + KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    + KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
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

/// Auto-implementation of [`KalmanFilterObservation`] for types that implement all necessary traits.
impl<const STATES: usize, const OBSERVATIONS: usize, T, Observation>
    KalmanFilterObservation<STATES, OBSERVATIONS, T> for Observation
where
    Observation: KalmanFilterNumStates<STATES>
        + KalmanFilterNumObservations<OBSERVATIONS>
        + KalmanFilterObservationVectorMut<OBSERVATIONS, T>
        + KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
        + KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
        + KalmanFilterObservationCorrectFilter<STATES, T>,
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
    fn control<I>(&mut self, control: &mut I)
    where
        I: KalmanFilterControlApplyToFilter<STATES, T>;
}

pub trait KalmanFilterUpdate<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    fn correct<M>(&mut self, measurement: &mut M)
    where
        M: KalmanFilterObservationCorrectFilter<STATES, T>;
}

pub trait KalmanFilterStateVector<const STATES: usize, T> {
    type StateVector: StateVector<STATES, T>;

    /// Gets a reference to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time.
    /// It contains all the necessary information to describe the system's current situation.
    fn state_vector(&self) -> &Self::StateVector;
}

pub trait KalmanFilterStateVectorMut<const STATES: usize, T>:
    KalmanFilterStateVector<STATES, T>
{
    type StateVectorMut: StateVectorMut<STATES, T>;

    /// Gets a reference to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time.
    /// It contains all the necessary information to describe the system's current situation.
    #[doc(alias = "kalman_get_state_vector")]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut;
}

pub trait KalmanFilterStateTransition<const STATES: usize, T> {
    type StateTransitionMatrix: StateTransitionMatrix<STATES, T>;

    /// Gets a reference to the state transition matrix A/F.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    fn state_transition(&self) -> &Self::StateTransitionMatrix;
}

pub trait KalmanFilterStateTransitionMut<const STATES: usize, T>:
    KalmanFilterStateTransition<STATES, T>
{
    type StateTransitionMatrixMut: StateTransitionMatrixMut<STATES, T>;

    /// Gets a reference to the state transition matrix A/.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[doc(alias = "kalman_get_state_transition")]
    fn state_transition_mut(&mut self) -> &mut Self::StateTransitionMatrixMut;
}

pub trait KalmanFilterSystemCovariance<const STATES: usize, T> {
    type EstimateCovarianceMatrix: EstimateCovarianceMatrix<STATES, T>;

    /// Gets a reference to the estimate covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[doc(alias = "system_covariance")]
    fn estimate_covariance(&self) -> &Self::EstimateCovarianceMatrix;
}

pub trait KalmanFilterSystemCovarianceMut<const STATES: usize, T>:
    KalmanFilterSystemCovariance<STATES, T>
{
    type EstimateCovarianceMatrixMut: EstimateCovarianceMatrix<STATES, T>;

    /// Gets a mutable reference to the estimate covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[doc(alias = "kalman_get_system_covariance")]
    #[doc(alias = "system_covariance_mut")]
    fn estimate_covariance_mut(&mut self) -> &mut Self::EstimateCovarianceMatrixMut;
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
    /// Applies a control to the state.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    #[allow(non_snake_case)]
    fn apply_to<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>;
}

pub trait KalmanFilterControlVector<const CONTROLS: usize, T> {
    type ControlVector: ControlVector<CONTROLS, T>;

    /// Gets a reference to the control vector u.
    ///
    /// The control vector contains the external inputs to the system that can influence its state.
    /// These inputs might include forces, accelerations, or other actuations applied to the system.
    fn control_vector(&self) -> &Self::ControlVector;
}

pub trait KalmanFilterControlVectorMut<const CONTROLS: usize, T>:
    KalmanFilterControlVector<CONTROLS, T>
{
    type ControlVectorMut: ControlVectorMut<CONTROLS, T>;

    /// Gets a mutable reference to the control vector u.
    ///
    /// The control vector contains the external inputs to the system that can influence its state.
    /// These inputs might include forces, accelerations, or other actuations applied to the system.
    #[doc(alias = "kalman_get_control_vector")]
    fn control_vector_mut(&mut self) -> &mut Self::ControlVectorMut;
}

pub trait KalmanFilterControlTransition<const STATES: usize, const CONTROLS: usize, T> {
    type ControlTransitionMatrix: ControlMatrix<STATES, CONTROLS, T>;

    /// Gets a reference to the control transition matrix B.
    ///
    /// This matrix maps the control inputs to the state space, allowing the control vector to
    /// influence the state transition. It quantifies how the control inputs affect the state change.
    fn control_matrix(&self) -> &Self::ControlTransitionMatrix;
}

pub trait KalmanFilterControlTransitionMut<const STATES: usize, const CONTROLS: usize, T>:
    KalmanFilterControlTransition<STATES, CONTROLS, T>
{
    type ControlTransitionMatrixMut: ControlMatrixMut<STATES, CONTROLS, T>;

    /// Gets a mutable reference to the control transition matrix B.
    ///
    /// This matrix maps the control inputs to the state space, allowing the control vector to
    /// influence the state transition. It quantifies how the control inputs affect the state change.
    #[doc(alias = "kalman_get_control_matrix")]
    fn control_matrix_mut(&mut self) -> &mut Self::ControlTransitionMatrixMut;
}

#[doc(alias = "KalmanFilterControlCovariance")]
pub trait KalmanFilterProcessNoiseCovariance<const CONTROLS: usize, T> {
    type ProcessNoiseCovarianceMatrix: ProcessNoiseCovarianceMatrix<CONTROLS, T>;

    /// Gets a reference to the control covariance matrix Q.
    ///
    /// This matrix represents the uncertainty in the state transition process, accounting for the
    /// randomness and inaccuracies in the model. It quantifies the expected variability in the
    /// state transition.
    #[doc(alias = "control_covariance")]
    fn process_noise_covariance(&self) -> &Self::ProcessNoiseCovarianceMatrix;
}

pub trait KalmanFilterControlCovarianceMut<const CONTROLS: usize, T>:
    KalmanFilterProcessNoiseCovariance<CONTROLS, T>
{
    type ProcessNoiseCovarianceMatrixMut: ProcessNoiseCovarianceMatrixMut<CONTROLS, T>;

    /// Gets a mutable reference to the control covariance matrix Q.
    ///
    /// This matrix represents the uncertainty in the state transition process, accounting for the
    /// randomness and inaccuracies in the model. It quantifies the expected variability in the
    /// state transition.
    #[doc(alias = "kalman_get_control_covariance")]
    #[doc(alias = "control_covariance_mut")]
    fn process_noise_covariance_mut(&mut self) -> &mut Self::ProcessNoiseCovarianceMatrixMut;
}

pub trait KalmanFilterNumObservations<const OBSERVATIONS: usize> {
    /// The number of measurements.
    const NUM_OBSERVATIONS: usize = OBSERVATIONS;

    /// Returns the number of controls.
    fn observations(&self) -> usize {
        OBSERVATIONS
    }
}

pub trait KalmanFilterObservationCorrectFilter<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    #[allow(non_snake_case)]
    fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>;
}

pub trait KalmanFilterMeasurementVector<const OBSERVATIONS: usize, T> {
    type MeasurementVector: MeasurementVector<OBSERVATIONS, T>;

    /// Gets a reference to the measurement vector z.
    ///
    /// The measurement vector represents the observed measurements from the system.
    /// These measurements are typically taken from sensors and are used to update the state estimate.
    fn measurement_vector(&self) -> &Self::MeasurementVector;
}

pub trait KalmanFilterObservationVectorMut<const OBSERVATIONS: usize, T>:
    KalmanFilterMeasurementVector<OBSERVATIONS, T>
{
    type MeasurementVectorMut: MeasurementVectorMut<OBSERVATIONS, T>;

    /// Gets a mutable reference to the measurement vector z.
    ///
    /// The measurement vector represents the observed measurements from the system.
    /// These measurements are typically taken from sensors and are used to update the state estimate.
    #[doc(alias = "kalman_get_measurement_vector")]
    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut;
}

pub trait KalmanFilterObservationTransformation<const STATES: usize, const OBSERVATIONS: usize, T> {
    type ObservationTransformationMatrix: ObservationMatrix<OBSERVATIONS, STATES, T>;

    /// Gets a reference to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    fn observation_matrix(&self) -> &Self::ObservationTransformationMatrix;
}

pub trait KalmanFilterObservationTransformationMut<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>: KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
{
    type ObservationTransformationMatrixMut: ObservationMatrixMut<OBSERVATIONS, STATES, T>;

    /// Gets a mutable reference to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[doc(alias = "kalman_get_measurement_transformation")]
    #[doc(alias = "measurement_transformation_mut")]
    fn observation_matrix_mut(&mut self) -> &mut Self::ObservationTransformationMatrixMut;
}

pub trait KalmanFilterMeasurementNoiseCovariance<const OBSERVATIONS: usize, T> {
    type MeasurementNoiseCovarianceMatrix: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>;

    /// Gets a reference to the process noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    fn measurement_noise_covariance(&self) -> &Self::MeasurementNoiseCovarianceMatrix;
}

pub trait KalmanFilterMeasurementNoiseCovarianceMut<const OBSERVATIONS: usize, T>:
    KalmanFilterMeasurementNoiseCovariance<OBSERVATIONS, T>
{
    type MeasurementNoiseCovarianceMatrixMut: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>;

    /// Gets a mutable reference to the process noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[doc(alias = "kalman_get_process_noise")]
    fn measurement_noise_covariance_mut(
        &mut self,
    ) -> &mut Self::MeasurementNoiseCovarianceMatrixMut;
}
