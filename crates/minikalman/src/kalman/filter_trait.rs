use crate::kalman::*;
use crate::prelude::{AsMatrix, AsMatrixMut};

/// The outcome of a correction step.
///
/// A correction that is not [applied](Self::is_applied) leaves the state vector and the
/// estimate covariance unchanged. Only [`Applied`](Self::Applied) and
/// [`NotConverged`](Self::NotConverged) count as applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CorrectionStatus {
    /// The measurement was applied to the state and the estimate covariance.
    Applied,
    /// The innovation covariance `S` was not positive definite, so it could not be inverted
    /// and the measurement was skipped.
    NotPositiveDefinite,
    /// The normalized innovation squared exceeded the observation's gate, so the measurement
    /// was rejected as an outlier.
    Gated,
    /// A recalibrated correction (e.g.
    /// [`correct_nonlinear_recalibrated`](crate::extended::ExtendedKalman::correct_nonlinear_recalibrated))
    /// would have increased the trace of the estimate covariance, so the measurement was backed
    /// out and the prior state and covariance were kept.
    ///
    /// This follows the back-out step of Jiang, Shi and Moura, "Mitigating Overconfidence in
    /// Nonlinear Kalman Filters via Covariance Recalibration", Automatica (2026),
    /// [arXiv:2407.05717](https://arxiv.org/abs/2407.05717): a measurement that makes the
    /// recalibrated estimate less certain than the prediction is taken as a sign that the
    /// linearization at the prior was poor.
    BackedOut,
    /// An iterative correction (e.g.
    /// [`correct_predictively_oriented_vector`](crate::regular::RegularObservation::correct_predictively_oriented_vector))
    /// stopped before it converged: it ran out of its iteration budget, or its line search
    /// stalled at the working precision of the number type (e.g. huge outliers in `f32`). The
    /// measurement **was applied** with the last iterate, which is a valid posterior with a
    /// positive definite covariance, but it is not the exact solution.
    /// [`is_applied`](Self::is_applied) returns `true`.
    NotConverged,
}

impl CorrectionStatus {
    /// Returns `true` if the measurement was applied, i.e. for [`Applied`](Self::Applied) and
    /// [`NotConverged`](Self::NotConverged).
    #[inline(always)]
    pub const fn is_applied(self) -> bool {
        matches!(self, Self::Applied | Self::NotConverged)
    }
}

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

/// An Unscented Kalman Filter.
///
/// This trait is implemented automatically for types satisfying all required sub-traits.
/// See [`UnscentedKalman`](crate::unscented::UnscentedKalman) for the concrete implementation.
pub trait UnscentedKalmanFilter<const STATES: usize, const NUM_SIGMA: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterStateVectorMut<STATES, T>
    + KalmanFilterEstimateCovarianceMut<STATES, T>
    + KalmanFilterSigmaPointPredict<STATES, T>
    + KalmanFilterSigmaPointCorrect<STATES, NUM_SIGMA, T>
    + KalmanFilterUnscentedParams<T>
{
}

/// Auto-implementation of [`UnscentedKalmanFilter`] for types that implement all necessary traits.
impl<const STATES: usize, const NUM_SIGMA: usize, T, Filter>
    UnscentedKalmanFilter<STATES, NUM_SIGMA, T> for Filter
where
    Filter: KalmanFilterNumStates<STATES>
        + KalmanFilterStateVectorMut<STATES, T>
        + KalmanFilterEstimateCovarianceMut<STATES, T>
        + KalmanFilterSigmaPointPredict<STATES, T>
        + KalmanFilterSigmaPointCorrect<STATES, NUM_SIGMA, T>
        + KalmanFilterUnscentedParams<T>,
{
}

/// A Kalman filter control input.
pub trait KalmanFilterControl<const STATES: usize, const CONTROLS: usize, T>:
    KalmanFilterNumStates<STATES>
    + KalmanFilterNumControls<CONTROLS>
    + KalmanFilterControlVectorMut<CONTROLS, T>
    + KalmanFilterControlTransition<STATES, CONTROLS, T>
    + KalmanFilterControlProcessNoiseMut<CONTROLS, T>
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
        + KalmanFilterControlProcessNoiseMut<CONTROLS, T>
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
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[inline(always)]
    fn correct<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
    {
        self.correct_with(measurement, |_| {})
    }

    /// Performs the measurement update step, normalizing the innovation before it is applied.
    ///
    /// See [`KalmanFilterObservationCorrectFilter::correct_with`].
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    /// * `fix` - Normalizes the innovation `y = z - H×x`, e.g. by wrapping angles.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    fn correct_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
        N: FnMut(&mut [T]);
}

pub trait KalmanFilterNonlinearUpdate<const STATES: usize, T>:
    KalmanFilterStateVectorMut<STATES, T>
{
    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[inline(always)]
    fn correct_nonlinear<M, F, const OBSERVATIONS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
    ) -> CorrectionStatus
    where
        M: KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>,
        F: FnMut(
            &<Self as KalmanFilterStateVectorMut<STATES, T>>::StateVectorMut,
            &mut M::ObservationVector,
        ),
    {
        self.correct_nonlinear_with(measurement, observation, |_| {})
    }

    /// Performs the measurement update step, normalizing the innovation before it is applied.
    ///
    /// See [`KalmanFilterNonlinearObservationCorrectFilter::correct_nonlinear_with`].
    ///
    /// ## Arguments
    /// * `measurement` - The measurement to update the state prediction with.
    /// * `observation` - The nonlinear observation function.
    /// * `fix` - Normalizes the innovation `y = z - h(x)`, e.g. by wrapping angles.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    fn correct_nonlinear_with<M, F, N, const OBSERVATIONS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        M: KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>,
        F: FnMut(
            &<Self as KalmanFilterStateVectorMut<STATES, T>>::StateVectorMut,
            &mut M::ObservationVector,
        ),
        N: FnMut(&mut [T]);
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
    /// This matrix represents the estimate covariance. It quantifies the uncertainty in
    /// the state estimate, providing a measure of how much the state estimate is expected
    /// to vary. This matrix offers a measure of confidence in the estimate by indicating
    /// the degree of variability and uncertainty associated with the predicted state.
    #[doc(alias = "system_covariance")]
    fn estimate_covariance(&self) -> &Self::EstimateCovarianceMatrix;
}

pub trait KalmanFilterEstimateCovarianceMut<const STATES: usize, T>:
    KalmanFilterEstimateCovariance<STATES, T>
{
    type EstimateCovarianceMatrixMut: EstimateCovarianceMatrix<STATES, T>;

    /// Gets a mutable reference to the estimate covariance matrix P.
    ///
    /// This matrix represents the estimate covariance. It quantifies the uncertainty in
    /// the state estimate, providing a measure of how much the state estimate is expected
    /// to vary. This matrix offers a measure of confidence in the estimate by indicating
    /// the degree of variability and uncertainty associated with the predicted state.
    #[doc(alias = "kalman_get_system_covariance")]
    #[doc(alias = "system_covariance_mut")]
    fn estimate_covariance_mut(&mut self) -> &mut Self::EstimateCovarianceMatrixMut;
}

pub trait KalmanFilterDirectProcessNoiseCovariance<const CONTROLS: usize, T> {
    type ProcessNoiseCovarianceMatrix: DirectProcessNoiseCovarianceMatrix<CONTROLS, T>;

    /// Gets a reference to the direct process noise matrix Q.
    ///
    /// This matrix represents the direct process noise covariance. It quantifies the
    /// uncertainty introduced by inherent system dynamics and external disturbances,
    /// providing a measure of how much the true state is expected to deviate from the
    /// predicted state due to these process variations.
    fn direct_process_noise(&self) -> &Self::ProcessNoiseCovarianceMatrix;
}

pub trait KalmanFilterDirectProcessNoiseMut<const CONTROLS: usize, T>:
    KalmanFilterDirectProcessNoiseCovariance<CONTROLS, T>
{
    type ProcessNoiseCovarianceMatrixMut: DirectProcessNoiseCovarianceMatrixMut<CONTROLS, T>;

    /// Gets a mutable reference to the direct process noise matrix Q.
    ///
    /// This matrix represents the direct process noise covariance. It quantifies the
    /// uncertainty introduced by inherent system dynamics and external disturbances,
    /// providing a measure of how much the true state is expected to deviate from the
    /// predicted state due to these process variations.
    fn direct_process_noise_mut(&mut self) -> &mut Self::ProcessNoiseCovarianceMatrixMut;
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
    /// This matrix represents the control input model. It defines how the control inputs
    /// influence the state evolution from one time step to the next. The matrix \( B \)
    /// is used to incorporate the effect of control inputs into the state transition,
    /// allowing the model to account for external controls applied to the system.
    fn control_matrix(&self) -> &Self::ControlTransitionMatrix;
}

pub trait KalmanFilterControlTransitionMut<const STATES: usize, const CONTROLS: usize, T>:
    KalmanFilterControlTransition<STATES, CONTROLS, T>
{
    type ControlTransitionMatrixMut: ControlMatrixMut<STATES, CONTROLS, T>;

    /// Gets a mutable reference to the control transition matrix B.
    ///
    /// This matrix represents the control input model. It defines how the control inputs
    /// influence the state evolution from one time step to the next. The matrix \( B \)
    /// is used to incorporate the effect of control inputs into the state transition,
    /// allowing the model to account for external controls applied to the system.
    #[doc(alias = "kalman_get_control_matrix")]
    fn control_matrix_mut(&mut self) -> &mut Self::ControlTransitionMatrixMut;
}

#[doc(alias = "KalmanFilterControlCovariance")]
pub trait KalmanFilterControlProcessNoiseCovariance<const CONTROLS: usize, T> {
    type ProcessNoiseCovarianceMatrix: ControlProcessNoiseCovarianceMatrix<CONTROLS, T>;

    /// Gets a reference to the control process noise matrix Q.
    ///
    /// This matrix represents the control process noise covariance. It quantifies the
    /// uncertainty introduced by the control inputs, reflecting how much the true state
    /// is expected to deviate from the predicted state due to noise and variations
    /// in the control process. The matrix is used as B×Q×Bᵀ, where B
    /// represents the control input model, and Q is the process noise covariance (this matrix).
    #[doc(alias = "control_covariance")]
    fn process_noise_covariance(&self) -> &Self::ProcessNoiseCovarianceMatrix;
}

pub trait KalmanFilterControlProcessNoiseMut<const CONTROLS: usize, T>:
    KalmanFilterControlProcessNoiseCovariance<CONTROLS, T>
{
    type ProcessNoiseCovarianceMatrixMut: ControlProcessNoiseCovarianceMatrixMut<CONTROLS, T>;

    /// Gets a mutable reference to the control process noise matrix Q.
    ///
    /// This matrix represents the control process noise covariance. It quantifies the
    /// uncertainty introduced by the control inputs, reflecting how much the true state
    /// is expected to deviate from the predicted state due to noise and variations
    /// in the control process. The matrix is used as B×Q×Bᵀ, where B
    /// represents the control input model, and Q is the process noise covariance (this matrix).
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
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct<X, P>(&mut self, x: &mut X, P: &mut P) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.correct_with(x, P, |_| {})
    }

    /// Performs the measurement update step, normalizing the innovation before it is applied.
    ///
    /// The innovation `y = z - H×x` is computed by plain subtraction. For measurements that
    /// wrap around, such as angles, the difference can come out a full turn too large;
    /// `fix` receives `y` as a slice (one entry per measurement) and can wrap it back,
    /// e.g. with [`angles::wrap_pi`](crate::angles::wrap_pi).
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `fix` - Normalizes the innovation vector in place.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_with<X, P, N>(&mut self, x: &mut X, P: &mut P, fix: N) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        N: FnMut(&mut [T]);
}

/// The predictively oriented (PrO) observation function for a regular Kalman Filter.
///
/// See [`RegularObservation::correct_predictively_oriented`](crate::regular::RegularObservation::correct_predictively_oriented)
/// for the objective, its cost, and when it helps.
pub trait KalmanFilterObservationCorrectPredictivelyOriented<const STATES: usize, T> {
    /// Performs the predictively oriented measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_predictively_oriented<X, P>(
        &mut self,
        x: &mut X,
        P: &mut P,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>;
}

/// The predictively oriented (PrO) observation function for a regular Kalman Filter with
/// vector observations.
///
/// See [`RegularObservation::correct_predictively_oriented_vector`](crate::regular::RegularObservation::correct_predictively_oriented_vector)
/// for the iteration, its cost, and the workspace.
pub trait KalmanFilterObservationCorrectPredictivelyOrientedVector<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>
{
    /// Performs the predictively oriented measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `workspace` - Three `OBSERVATIONS × OBSERVATIONS` scratch matrices.
    /// * `max_iterations` - The budget of accepted steps of the iteration.
    ///
    /// Returns whether the measurement was applied and the iteration converged; see
    /// [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_predictively_oriented_vector<X, P, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        workspace: &mut crate::regular::PredictivelyOrientedWorkspace<W>,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>;
}

/// The outlier-robust (Student-t) observation function for a regular Kalman Filter.
///
/// See [`RegularObservation::correct_robust`](crate::regular::RegularObservation::correct_robust)
/// for the update, its cost, and how it relates to the predictively oriented correction.
pub trait KalmanFilterObservationCorrectRobust<const STATES: usize, const OBSERVATIONS: usize, T> {
    /// Performs the robust measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `workspace` - One `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget.
    ///
    /// Returns whether the measurement was applied and the weight converged; see
    /// [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_robust<X, P, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    {
        self.correct_robust_with(x, P, workspace, robust, |_| {})
    }

    /// Performs the robust measurement update step, normalizing the innovation before it is
    /// applied.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `workspace` - One `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget.
    /// * `fix` - Normalizes the innovation vector in place.
    ///
    /// Returns whether the measurement was applied and the weight converged; see
    /// [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_robust_with<X, P, W, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
        N: FnMut(&mut [T]);
}

/// The outlier-robust (Student-t) observation function for an Extended Kalman Filter.
///
/// See [`ExtendedObservation::correct_nonlinear_robust`](crate::extended::ExtendedObservation::correct_nonlinear_robust)
/// for the update. Unlike the extended predictively oriented correction, it supports any
/// number of measurements.
pub trait KalmanFilterNonlinearObservationCorrectRobust<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>: KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>
{
    /// Performs the robust nonlinear measurement update step.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `workspace` - One `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget.
    ///
    /// Returns whether the measurement was applied and the weight converged; see
    /// [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_nonlinear_robust<X, P, F, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    {
        self.correct_nonlinear_robust_with(x, P, observation, |_| {}, workspace, robust)
    }

    /// Performs the robust nonlinear measurement update step, normalizing the innovation
    /// before it is applied.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `fix` - Normalizes the innovation vector in place.
    /// * `workspace` - One `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget.
    ///
    /// Returns whether the measurement was applied and the weight converged; see
    /// [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_nonlinear_robust_with<X, P, F, N, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]),
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>;
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
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_nonlinear<X, P, F>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
    {
        self.correct_nonlinear_with(x, P, observation, |_| {})
    }

    /// Performs the nonlinear measurement update step, normalizing the innovation before
    /// it is applied.
    ///
    /// The innovation `y = z - h(x)` is computed by plain subtraction. For measurements that
    /// wrap around, such as bearings, the difference can come out a full turn too large;
    /// `fix` receives `y` as a slice (one entry per measurement) and can wrap it back,
    /// e.g. with [`angles::wrap_pi`](crate::angles::wrap_pi).
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `fix` - Normalizes the innovation vector in place.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_nonlinear_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]);
}

/// A linear observation that can be applied with an externally supplied Kalman gain.
///
/// Instead of computing `K = P×Hᵀ×S⁻¹`, the correction uses the gain already stored in the
/// observation's `K` buffer, e.g. a precomputed steady-state gain or a learned one. `P` is
/// updated with the Joseph form, so it stays symmetric and positive semi-definite for any `K`.
///
/// Since `S` is never inverted, the [gate](crate::regular::RegularObservation::set_gate) is not
/// evaluated and the [NIS](crate::regular::RegularObservation::nis) is not updated.
///
/// See [`KalmanFilterNonlinearObservationCorrectWithGain`] for the nonlinear variant.
pub trait KalmanFilterObservationCorrectWithGain<const STATES: usize, T> {
    /// Performs the measurement update step with the gain stored in the observation.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_with_gain<X, P>(&mut self, x: &mut X, P: &mut P) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.correct_with_gain_with(x, P, |_| {})
    }

    /// Performs the measurement update step with the gain stored in the observation,
    /// normalizing the innovation `y = z - H×x` before it is applied.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `fix` - Normalizes the innovation vector in place, e.g. by wrapping angles.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[allow(non_snake_case)]
    fn correct_with_gain_with<X, P, N>(&mut self, x: &mut X, P: &mut P, fix: N) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        N: FnMut(&mut [T]);
}

/// A nonlinear (Extended Kalman Filter) observation that can be applied with an externally
/// supplied Kalman gain.
///
/// Like [`KalmanFilterObservationCorrectWithGain`], but the innovation is `y = z - h(x)` and
/// the observation matrix `H` holds the Jacobian of `h`, which is used for the Joseph-form
/// covariance update.
pub trait KalmanFilterNonlinearObservationCorrectWithGain<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>: KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>
{
    /// Performs the nonlinear measurement update step with the gain stored in the observation.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_nonlinear_with_gain<X, P, F>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
    {
        self.correct_nonlinear_with_gain_with(x, P, observation, |_| {})
    }

    /// Performs the nonlinear measurement update step with the gain stored in the observation,
    /// normalizing the innovation `y = z - h(x)` before it is applied.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `fix` - Normalizes the innovation vector in place, e.g. by wrapping angles.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[allow(non_snake_case)]
    fn correct_nonlinear_with_gain_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]);
}

/// Recalibrated nonlinear measurement update for Extended Kalman Filters.
///
#[doc = diagram!("extended-recalibrated.mmd")]
///
/// Implements the covariance recalibration of Jiang, Shi and Moura, "Mitigating
/// Overconfidence in Nonlinear Kalman Filters via Covariance Recalibration", Automatica 194
/// (2026) 113295, [arXiv:2407.05717](https://arxiv.org/abs/2407.05717), Algorithm 2:
///
/// 1. The gain `K` and the state `x⁺ = x + K×y` are computed at the prior, exactly as in
///    [`correct_nonlinear`](KalmanFilterNonlinearObservationCorrectFilter::correct_nonlinear).
/// 2. The observation is linearized again at `x⁺`, giving `H₂`, and the covariance is
///    recalibrated with the Joseph form `P⁺ = P − K×H₂×P − (K×H₂×P)ᵀ + K×(H₂×P×H₂ᵀ + R)×Kᵀ`.
/// 3. If `trace(P⁺) > trace(P)`, the measurement is backed out: `x` and `P` keep their prior
///    values and [`CorrectionStatus::BackedOut`] is returned.
///
/// The conventional update takes `P` from the linearization at the prior, which makes the
/// filter overconfident when the prior is far from the truth. For a linear observation,
/// `H₂ = H` and the result equals the conventional update.
pub trait KalmanFilterNonlinearObservationCorrectRecalibrated<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
>: KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>
{
    /// The type of the observation Jacobian `H` that the `jacobian` callback fills.
    type ObservationJacobian: ObservationMatrixMut<OBSERVATIONS, STATES, T>;

    /// Performs the recalibrated nonlinear measurement update step.
    ///
    /// The observation Jacobian must be set up at the prior `x`, as for
    /// [`correct_nonlinear`](KalmanFilterNonlinearObservationCorrectFilter::correct_nonlinear).
    /// `jacobian` is then called once with the updated state `x⁺` and overwrites the Jacobian
    /// with `H₂`; it is not called if the measurement is skipped before `x⁺` exists. Gating
    /// uses the normalized innovation squared at the prior.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `jacobian` - Writes the observation Jacobian at the given state.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_nonlinear_recalibrated<X, P, F, J>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        J: FnMut(&X, &mut Self::ObservationJacobian),
    {
        self.correct_nonlinear_recalibrated_with(x, P, observation, jacobian, |_| {})
    }

    /// Performs the recalibrated nonlinear measurement update step, normalizing the
    /// innovation before it is applied.
    ///
    /// See [`correct_nonlinear_recalibrated`](Self::correct_nonlinear_recalibrated) and
    /// [`correct_nonlinear_with`](KalmanFilterNonlinearObservationCorrectFilter::correct_nonlinear_with).
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `jacobian` - Writes the observation Jacobian at the given state.
    /// * `fix` - Normalizes the innovation vector in place.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_nonlinear_recalibrated_with<X, P, F, J, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        J: FnMut(&X, &mut Self::ObservationJacobian),
        N: FnMut(&mut [T]);
}

/// The predictively oriented (PrO) observation function for an Extended Kalman Filter.
///
/// See [`ExtendedObservation::correct_nonlinear_predictively_oriented`](crate::extended::ExtendedObservation::correct_nonlinear_predictively_oriented)
/// and [`ExtendedObservation::correct_nonlinear_predictively_oriented_iterated`](crate::extended::ExtendedObservation::correct_nonlinear_predictively_oriented_iterated)
/// for the algorithm and its limits. Only scalar observations support it.
pub trait KalmanFilterNonlinearObservationCorrectPredictivelyOriented<const STATES: usize, T>:
    KalmanFilterNonlinearObservationCorrectFilter<STATES, 1, T>
{
    /// The type of the observation Jacobian `H` that the `jacobian` callback fills.
    type ObservationJacobian: ObservationMatrixMut<1, STATES, T>;

    /// Performs the PrO measurement update step with the Jacobian set up at the prior.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_nonlinear_predictively_oriented<X, P, F>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
    {
        self.correct_nonlinear_predictively_oriented_with(x, P, observation, |_| {}, max_iterations)
    }

    /// Performs the PrO measurement update step with the Jacobian set up at the prior,
    /// normalizing the innovation before it is applied.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `fix` - Normalizes the innovation vector in place.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    fn correct_nonlinear_predictively_oriented_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]);

    /// Performs the PrO measurement update step, relinearizing the observation at the PrO
    /// mean for `relinearizations` rounds.
    ///
    /// The Jacobian must be set up at the prior; `jacobian` overwrites it at every new
    /// linearization point.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `jacobian` - Writes the observation Jacobian at the given state.
    /// * `relinearizations` - The number of relinearization rounds.
    /// * `max_iterations` - The budget of Newton steps for each stationarity condition.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case)]
    #[inline(always)]
    fn correct_nonlinear_predictively_oriented_iterated<X, P, F, J>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
        relinearizations: usize,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        J: FnMut(&X, &mut Self::ObservationJacobian),
    {
        self.correct_nonlinear_predictively_oriented_iterated_with(
            x,
            P,
            observation,
            jacobian,
            |_| {},
            relinearizations,
            max_iterations,
        )
    }

    /// Performs the PrO measurement update step, relinearizing the observation at the PrO
    /// mean for `relinearizations` rounds and normalizing every innovation before it is used.
    ///
    /// ## Arguments
    /// * `x` - The state vector.
    /// * `P` - The system covariance matrix.
    /// * `observation` - The nonlinear observation function.
    /// * `jacobian` - Writes the observation Jacobian at the given state.
    /// * `fix` - Normalizes the innovation vector in place.
    /// * `relinearizations` - The number of relinearization rounds.
    /// * `max_iterations` - The budget of Newton steps for each stationarity condition.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn correct_nonlinear_predictively_oriented_iterated_with<X, P, F, J, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
        fix: N,
        relinearizations: usize,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        J: FnMut(&X, &mut Self::ObservationJacobian),
        N: FnMut(&mut [T]);
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

/// UKF parameters (alpha, beta, kappa).
///
/// They set the spread of the sigma points and their weights, which the prediction and the
/// correction both use:
#[doc = diagram!("unscented-loop.mmd", "sigma")]
///
/// See [Choosing α, β and κ](crate::unscented::UnscentedKalman#choosing-α-β-and-κ).
pub trait KalmanFilterUnscentedParams<T> {
    /// Spread parameter α (builder default 1.0).
    fn alpha(&self) -> T;

    /// Distribution knowledge parameter β (builder default 2.0, optimal for a Gaussian).
    fn beta(&self) -> T;

    /// Secondary scaling parameter κ (builder default 1.0).
    fn kappa(&self) -> T;

    /// Computed lambda: alpha^2 * (n + kappa) - n.
    fn lambda(&self, n: usize) -> T;
}

/// Mutable UKF parameters.
pub trait KalmanFilterUnscentedParamsMut<T>: KalmanFilterUnscentedParams<T> {
    fn set_alpha(&mut self, alpha: T);
    fn set_beta(&mut self, beta: T);
    fn set_kappa(&mut self, kappa: T);

    /// Sets the function that normalizes differences between states, e.g. sigma point
    /// minus mean.
    ///
    /// Use this when a state wraps around, such as a heading: the function receives the
    /// difference as a slice (one entry per state) and can wrap it back into range, e.g.
    /// with [`angles::wrap_pi`](crate::angles::wrap_pi). It applies to the state means and
    /// residuals in both the prediction and the correction step.
    fn set_state_residual(&mut self, fix: fn(&mut [T]));
}

/// Nonlinear prediction using sigma points (Unscented Kalman Filter).
pub trait KalmanFilterSigmaPointPredict<const STATES: usize, T>:
    KalmanFilterStateVectorMut<STATES, T>
{
    /// The type of the next state vector (used for sigma point propagation).
    type NextStateVector: AsMatrixMut<STATES, 1, T>;

    /// Performs the nonlinear time update / prediction step.
    ///
    #[doc = diagram!("unscented-loop.mmd", "sigma", "propagate", "predict")]
    ///
    /// The closure receives each sigma point and should modify it in-place
    /// to produce the predicted sigma point.
    fn predict_sigma_point<F>(&mut self, state_transition: F)
    where
        F: FnMut(&mut Self::NextStateVector);
}

/// Nonlinear correction using sigma points (Unscented Kalman Filter).
pub trait KalmanFilterSigmaPointCorrect<const STATES: usize, const NUM_SIGMA: usize, T>:
    KalmanFilterStateVectorMut<STATES, T>
{
    /// The type for propagated sigma points.
    type SigmaPropagated: AsMatrix<STATES, NUM_SIGMA, T>;

    /// Performs the nonlinear correction step using sigma points.
    ///
    #[doc = diagram!("unscented-correct.mmd")]
    ///
    /// The sigma points of the last prediction are passed to `observation`, which writes
    /// their images under `h`. A second correction before the next prediction redraws them
    /// from the corrected estimate instead.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[inline(always)]
    fn correct_sigma_point<M, F, const OBSERVATIONS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBSERVATIONS, NUM_SIGMA, T>,
        F: FnMut(&Self::SigmaPropagated, &mut M::ObservedSigmaPoints),
    {
        self.correct_sigma_point_with(measurement, observation, |_| {})
    }

    /// Performs the nonlinear correction step using sigma points, normalizing differences
    /// between measurements before they are used.
    ///
    #[doc = diagram!("unscented-loop.mmd", "observe", "correct")]
    ///
    /// `fix` receives each measurement difference (observed sigma point minus predicted
    /// measurement, and measurement minus predicted measurement) as a slice, one entry per
    /// measurement, and can wrap it back into range, e.g. with
    /// [`angles::wrap_pi`](crate::angles::wrap_pi) for a bearing. The predicted measurement is
    /// averaged relative to the centre sigma point, so values on either side of the wrap point
    /// average correctly.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    fn correct_sigma_point_with<M, F, N, const OBSERVATIONS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBSERVATIONS, NUM_SIGMA, T>,
        F: FnMut(&Self::SigmaPropagated, &mut M::ObservedSigmaPoints),
        N: FnMut(&mut [T]);
}

/// Observation correct filter for Unscented Kalman Filter.
pub trait KalmanFilterUnscentedObservationCorrectFilter<
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
>
{
    /// The type for observed sigma points.
    type ObservedSigmaPoints: AsMatrixMut<OBSERVATIONS, NUM_SIGMA, T>;

    /// Performs the correction given pre-populated observed sigma points.
    ///
    /// The closure receives predicted sigma points and is responsible for
    /// populating the observed sigma points before correction proceeds.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[inline(always)]
    fn correct_with_observed<X, P, SP, F, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_predicted: &SP,
        weights: &W,
        lambda: T,
        observation: F,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        F: FnMut(&SP, &mut Self::ObservedSigmaPoints),
    {
        self.correct_with_observed_residuals(
            x,
            P,
            sigma_predicted,
            weights,
            lambda,
            observation,
            |_| {},
            |_| {},
        )
    }

    /// Performs the correction given pre-populated observed sigma points, normalizing
    /// state and measurement differences before they are used.
    ///
    /// * `fix_state` normalizes differences between states (one entry per state).
    /// * `fix_measurement` normalizes differences between measurements (one entry per
    ///   measurement).
    ///
    /// Means are taken relative to the centre sigma point, so values on either side of a
    /// wrap point average correctly.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn correct_with_observed_residuals<X, P, SP, F, W, NX, NZ>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_predicted: &SP,
        weights: &W,
        lambda: T,
        observation: F,
        fix_state: NX,
        fix_measurement: NZ,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        F: FnMut(&SP, &mut Self::ObservedSigmaPoints),
        NX: FnMut(&mut [T]),
        NZ: FnMut(&mut [T]);
}

/// Outlier-robust (Student-t) sigma point correction for Unscented Kalman Filters.
///
/// See [`UnscentedKalman::correct_sigma_point_robust`](crate::unscented::UnscentedKalman::correct_sigma_point_robust)
/// for the update and how it differs from Piché et al.
pub trait KalmanFilterUnscentedObservationCorrectRobust<
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
>: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBSERVATIONS, NUM_SIGMA, T>
{
    /// Performs the robust correction given the sigma points, normalizing state and
    /// measurement differences before they are used.
    ///
    /// `observation`, `fix_state` and `fix_measurement` are used as in
    /// [`correct_with_observed_residuals`](KalmanFilterUnscentedObservationCorrectFilter::correct_with_observed_residuals);
    /// `workspace` is one `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    ///
    /// Returns whether the measurement was applied and the weight converged; see
    /// [`CorrectionStatus`].
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn correct_with_observed_robust_residuals<X, P, SP, F, W, NX, NZ, WS>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_predicted: &SP,
        weights: &W,
        lambda: T,
        observation: F,
        fix_state: NX,
        fix_measurement: NZ,
        workspace: &mut WS,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        F: FnMut(&SP, &mut Self::ObservedSigmaPoints),
        NX: FnMut(&mut [T]),
        NZ: FnMut(&mut [T]),
        WS: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>;
}

/// Recalibrated sigma point correction for Unscented Kalman Filters.
///
#[doc = diagram!("unscented-correct.mmd", "recalibrate")]
///
/// Implements the covariance recalibration of Jiang, Shi and Moura, "Mitigating
/// Overconfidence in Nonlinear Kalman Filters via Covariance Recalibration", Automatica 194
/// (2026) 113295, [arXiv:2407.05717](https://arxiv.org/abs/2407.05717), Eq. 22–23, for the
/// unscented transform:
///
/// 1. The gain `K` and the innovation `y` are computed from the sigma points as in
///    [`correct_with_observed_residuals`](KalmanFilterUnscentedObservationCorrectFilter::correct_with_observed_residuals).
/// 2. The sigma points are shifted by `K×y`, which places them around `x⁺ = x + K×y` with the
///    prior spread, and observed again. This gives `S₂ = Pzz₂ + R` and `Pxz₂`, and the
///    covariance becomes `P⁺ = P + K×S₂×Kᵀ − Pxz₂×Kᵀ − K×Pxz₂ᵀ`.
/// 3. If `trace(P⁺) > trace(P)`, the measurement is backed out: `x` and `P` keep their prior
///    values and [`CorrectionStatus::BackedOut`] is returned.
pub trait KalmanFilterUnscentedObservationCorrectRecalibrated<
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
>: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBSERVATIONS, NUM_SIGMA, T>
{
    /// Performs the recalibrated correction, normalizing state and measurement differences
    /// before they are used.
    ///
    /// `observation` runs twice if the first pass is applied: once on the sigma points and
    /// once on the shifted sigma points. Gating uses the normalized innovation squared of the
    /// first pass. Unless the measurement is skipped before the second pass,
    /// `sigma_predicted` is left shifted by `K×y` on return, so it must be regenerated before
    /// it is used again.
    ///
    /// Returns whether the measurement was applied; see [`CorrectionStatus`].
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn correct_with_observed_recalibrated_residuals<X, P, SP, F, W, NX, NZ>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_predicted: &mut SP,
        weights: &W,
        lambda: T,
        observation: F,
        fix_state: NX,
        fix_measurement: NZ,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrixMut<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        F: FnMut(&SP, &mut Self::ObservedSigmaPoints),
        NX: FnMut(&mut [T]),
        NZ: FnMut(&mut [T]);
}
