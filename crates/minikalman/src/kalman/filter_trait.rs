use crate::kalman::*;
use crate::prelude::AsMatrixMut;

/// A Kalman Filter.
pub trait KalmanFilter<const STATES: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterStateVectorMut<STATES, T>
    + KalmanFilterStateTransition<STATES, T>
    + KalmanFilterEstimateCovarianceMut<STATES, T>
    + KalmanFilterPredict<STATES, T>
    + KalmanFilterApplyControl<STATES, T>
    + KalmanFilterUpdate<STATES, T>
{
}

/// An Extended Kalman Filter.
pub trait ExtendedKalmanFilter<const STATES: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterStateVectorMut<STATES, T>
    + ExtendedKalmanFilterStateTransition<STATES, T>
    + KalmanFilterEstimateCovarianceMut<STATES, T>
    + KalmanFilterNonlinearPredict<STATES, T>
    + KalmanFilterNonlinearUpdate<STATES, T>
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

/// A Kalman Filter observation or measurement.
pub trait KalmanFilterObservation<const STATES: usize, const OBSERVATIONS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumObservations<OBSERVATIONS>
    + KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    + KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    + KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
    + KalmanFilterObservationCorrectFilter<STATES, T>
{
}

/// An Extended Kalman Filter observation or measurement.
pub trait ExtendedKalmanFilterObservation<const STATES: usize, const OBSERVATIONS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumObservations<OBSERVATIONS>
    + KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    + KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
    + ExtendedKalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    + KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>
{
}

/// Auto-implementation of [`KalmanFilter`] for types that implement all necessary traits.
impl<const STATES: usize, T, Filter> KalmanFilter<STATES, T> for Filter where
    Filter: KalmanFilterNumStates<STATES>
        + KalmanFilterStateVectorMut<STATES, T>
        + KalmanFilterStateTransition<STATES, T>
        + KalmanFilterEstimateCovarianceMut<STATES, T>
        + KalmanFilterPredict<STATES, T>
        + KalmanFilterApplyControl<STATES, T>
        + KalmanFilterUpdate<STATES, T>
{
}

/// Auto-implementation of [`ExtendedKalmanFilter`] for types that implement all necessary traits.
impl<const STATES: usize, T, Filter> ExtendedKalmanFilter<STATES, T> for Filter where
    Filter: KalmanFilterNumStates<STATES>
        + KalmanFilterStateVectorMut<STATES, T>
        + ExtendedKalmanFilterStateTransition<STATES, T>
        + KalmanFilterEstimateCovarianceMut<STATES, T>
        + KalmanFilterNonlinearPredict<STATES, T>
        + KalmanFilterNonlinearUpdate<STATES, T>
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

/// Auto-implementation of [`ExtendedKalmanFilterObservation`] for types that implement all necessary traits.
impl<const STATES: usize, const OBSERVATIONS: usize, T, Observation>
    ExtendedKalmanFilterObservation<STATES, OBSERVATIONS, T> for Observation
where
    Observation: KalmanFilterNumStates<STATES>
        + KalmanFilterNumObservations<OBSERVATIONS>
        + KalmanFilterObservationVectorMut<OBSERVATIONS, T>
        + KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
        + ExtendedKalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
        + KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>,
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

pub trait KalmanFilterNonlinearPredict<const STATES: usize, T>:
    KalmanFilterStateVectorMut<STATES, T>
{
    /// The type of observation vector to fill.
    type NextStateVector: AsMatrixMut<STATES, 1, T>;

    /// Performs the nonlinear time update / prediction step.
    ///
    /// ## Extended Kalman Filter
    /// This function assumes the state transition Jacobian was set up correctly using
    /// [`state_transition`](KalmanFilterStateTransition::state_transition).
    fn predict_nonlinear<F>(&mut self, state_transition: F)
    where
        F: FnMut(
            &<Self as KalmanFilterStateVectorMut<STATES, T>>::StateVectorMut,
            &mut Self::NextStateVector,
        );
}

pub trait KalmanFilterApplyControl<const STATES: usize, T> {
    /// Performs the measurement update step.
    ///
    /// ## Extended Kalman Filters
    /// In an Extended Kalman Filter, this method is meaningless. Use the
    /// [`predict_nonlinear`](KalmanFilterNonlinearPredict::predict_nonlinear) function set instead.
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

pub trait KalmanFilterNonlinearUpdate<const STATES: usize, T>:
    KalmanFilterStateVectorMut<STATES, T>
{
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    fn correct_nonlinear<M, F, const OBSERVATIONS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
    ) where
        M: KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>,
        F: FnMut(
            &<Self as KalmanFilterStateVectorMut<STATES, T>>::StateVectorMut,
            &mut M::ObservationVector,
        );
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

/// Provides access to the state transition matrix.
///
/// ## (Regular) Kalman Filters
/// This matrix describes how the state vector evolves from one time step to the next in the
/// absence of control inputs. It defines the relationship between the previous state and the
/// current state, accounting for the inherent dynamics of the system.
pub trait KalmanFilterStateTransition<const STATES: usize, T> {
    type StateTransitionMatrix: StateTransitionMatrix<STATES, T>;

    /// Gets a reference to the state transition matrix A/F.
    ///
    /// ## (Regular) Kalman Filters
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    fn state_transition(&self) -> &Self::StateTransitionMatrix;
}

pub trait KalmanFilterStateTransitionMut<const STATES: usize, T>:
    KalmanFilterStateTransition<STATES, T>
{
    type StateTransitionMatrixMut: StateTransitionMatrixMut<STATES, T>;

    /// Gets a reference to the state transition matrix A/Fn
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[doc(alias = "kalman_get_state_transition")]
    fn state_transition_mut(&mut self) -> &mut Self::StateTransitionMatrixMut;
}

/// Provides access to the Jacobian of the state transition matrix.
///
/// ## Extended Kalman Filters
/// In Extended Kalman Filters, this matrix is treated as the Jacobian of the state
/// transition matrix, i.e. the derivative of the state transition matrix with respect
/// to the state vector.
pub trait ExtendedKalmanFilterStateTransition<const STATES: usize, T> {
    type StateTransitionMatrix: StateTransitionMatrix<STATES, T>;

    /// Gets a reference to the Jacobian of the state transition matrix A/F.
    ///
    /// ## Extended Kalman Filters
    /// When predicting using [`predict_nonlinear`](KalmanFilterNonlinearPredict::predict_nonlinear),
    /// this matrix is treated as the Jacobian of the state transition matrix, i.e. the derivative
    /// of the state transition matrix with respect to the state vector.
    fn state_transition_jacobian(&self) -> &Self::StateTransitionMatrix;
}

pub trait ExtendedKalmanFilterStateTransitionMut<const STATES: usize, T>:
    ExtendedKalmanFilterStateTransition<STATES, T>
{
    type StateTransitionMatrixMut: StateTransitionMatrixMut<STATES, T>;

    /// Gets a reference to the Jacobian of the state transition matrix A/F.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    ///
    /// ## Extended Kalman Filters
    /// When predicting using [`predict_nonlinear`](KalmanFilterNonlinearPredict::predict_nonlinear),
    /// this matrix is treated as the Jacobian of the state transition matrix, i.e. the derivative
    /// of the state transition matrix with respect to the state vector.
    fn state_transition_jacobian_mut(&mut self) -> &mut Self::StateTransitionMatrixMut;
}

pub trait KalmanFilterEstimateCovariance<const STATES: usize, T> {
    type EstimateCovarianceMatrix: EstimateCovarianceMatrix<STATES, T>;

    /// Gets a reference to the estimate covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[doc(alias = "system_covariance")]
    fn estimate_covariance(&self) -> &Self::EstimateCovarianceMatrix;
}

pub trait KalmanFilterEstimateCovarianceMut<const STATES: usize, T>:
    KalmanFilterEstimateCovariance<STATES, T>
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

/// The observation function for a regular Kalman Filter.
///
/// ## Extended Kalman Filters
/// See [`KalmanFilterNonlinearObservationCorrectFilter`].
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

/// The observation function for an Extended Kalman Filter.
pub trait KalmanFilterNonlinearObservationCorrectFilter<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>
{
    /// The type of observation vector to fill.
    type ObservationVector: AsMatrixMut<OBSERVATIONS, 1, T>;

    /// Performs the nonlinear measurement update step for Extended Kalman Filters.
    ///
    /// ## Extended Kalman Filters
    /// This function expects the observation transformation Jacobian to be set up correctly
    /// using e.g. [`observation_matrix_mut`](KalmanFilterObservationTransformationMut::observation_matrix_mut).
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observations` - The nonlinear observation function.
    #[allow(non_snake_case)]
    fn correct_nonlinear<X, P, F>(&mut self, x: &mut X, P: &mut P, observation: F)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector);
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

pub trait ExtendedKalmanFilterObservationTransformation<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>
{
    type ObservationTransformationJacobianMatrix: ObservationMatrix<OBSERVATIONS, STATES, T>;

    /// Gets a reference to the Jacobian of the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    ///
    /// ## Extended Kalman Filters
    /// When correcting using [`correct_nonlinear`](KalmanFilterNonlinearObservationCorrectFilter::correct_nonlinear),
    /// this matrix is treated as the Jacobian of the observation matrix, i.e. the derivative of
    /// the measurement function with respect to the state vector.
    fn observation_jacobian_matrix(&self) -> &Self::ObservationTransformationJacobianMatrix;
}

pub trait ExtendedKalmanFilterObservationTransformationMut<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>: ExtendedKalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
{
    type ObservationTransformationJacobianMatrixMut: ObservationMatrixMut<OBSERVATIONS, STATES, T>;

    /// Gets a mutable reference to the Jacobian of the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    ///
    /// ## Extended Kalman Filters
    /// When correcting using [`correct_nonlinear`](KalmanFilterNonlinearObservationCorrectFilter::correct_nonlinear),
    /// this matrix is treated as the Jacobian of the observation matrix, i.e. the derivative of
    /// the measurement function with respect to the state vector.
    #[doc(alias = "kalman_get_measurement_transformation")]
    #[doc(alias = "measurement_transformation_mut")]
    fn observation_jacobian_matrix_mut(
        &mut self,
    ) -> &mut Self::ObservationTransformationJacobianMatrixMut;
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
