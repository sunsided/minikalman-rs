//! # Regular Kalman Filter

use crate::kalman::*;
use crate::matrix::MatrixDataType;
use crate::prelude::Matrix;
use core::marker::PhantomData;

/// A builder for a [`RegularKalman`] filter instances.
#[allow(clippy::type_complexity)]
pub struct RegularKalmanBuilder<A, X, P, Q, PX, TempP> {
    _phantom: (
        PhantomData<A>,
        PhantomData<X>,
        PhantomData<P>,
        PhantomData<Q>,
        PhantomData<PX>,
        PhantomData<TempP>,
    ),
}

impl<A, X, P, Q, PX, TempP> RegularKalmanBuilder<A, X, P, Q, PX, TempP> {
    /// Initializes a Kalman filter instance.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `P` - The state covariance matrix (`STATES` × `STATES`).
    /// * `Q` - The direct process noise matrix (`STATES` × `STATES`).
    /// * `predictedX` - The temporary vector for predicted states (`STATES` × `1`).
    /// * `temp_P` - The temporary vector for P calculation (`STATES` × `STATES`).
    ///
    /// ## Example
    ///
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// use minikalman::regular::RegularKalmanBuilder;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_CONTROLS: usize = 0;
    /// # const NUM_OBSERVATIONS: usize = 1;
    /// // System buffers.
    /// impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// impl_buffer_Q_direct!(mut gravity_Q, NUM_STATES, f32, 0.0);
    ///
    /// // Filter temporaries.
    /// impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    ///
    /// let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    ///     gravity_A,
    ///     gravity_x,
    ///     gravity_P,
    ///     gravity_Q,
    ///     gravity_temp_x,
    ///     gravity_temp_P,
    ///  );
    /// ```
    #[allow(non_snake_case, clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub fn new<const STATES: usize, T>(
        A: A,
        x: X,
        P: P,
        Q: Q,
        predicted_x: PX,
        temp_P: TempP,
    ) -> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
    where
        T: MatrixDataType,
        A: StateTransitionMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
        PX: PredictedStateEstimateVector<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
    {
        RegularKalman::<STATES, T, _, _, _, _, _, _> {
            x,
            A,
            P,
            Q,
            predicted_x,
            temp_P,
            _phantom: Default::default(),
        }
    }
}

/// Kalman Filter structure.  See [`RegularKalmanBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct RegularKalman<const STATES: usize, T, A, X, P, Q, PX, TempP> {
    /// State vector.
    x: X,

    /// System matrix. In Extended Kalman Filters, the Jacobian of the system matrix.
    ///
    /// See also [`P`].
    A: A,

    /// Estimation covariance matrix.
    ///
    /// See also [`A`].
    P: P,

    /// Direct process noise matrix.
    ///
    /// See also [`P`].
    Q: Q,

    /// x-sized temporary vector.
    predicted_x: PX,

    /// P-Sized temporary matrix (number of states × number of states).
    ///
    /// The backing field for this temporary MAY be aliased with temporary BQ.
    temp_P: TempP,

    _phantom: PhantomData<T>,
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP>
    RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
{
    /// Returns the number of states.
    pub const fn states(&self) -> usize {
        STATES
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    X: StateVector<STATES, T>,
{
    /// Gets a reference to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    pub fn state_vector(&self) -> &X {
        &self.x
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    X: StateVectorMut<STATES, T>,
{
    /// Gets a reference to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_vector")]
    pub fn state_vector_mut(&mut self) -> &mut X {
        &mut self.x
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    A: StateTransitionMatrix<STATES, T>,
{
    /// Gets a reference to the state transition matrix A/F, or its Jacobian
    ///
    /// ## (Regular) Kalman Filters
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix")]
    #[doc(alias = "system_jacobian_matrix")]
    pub fn state_transition(&self) -> &A {
        &self.A
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    A: StateTransitionMatrixMut<STATES, T>,
{
    /// Gets a reference to the state transition matrix A/F, or its Jacobian.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_mut")]
    #[doc(alias = "system_jacobian_matrix_mut")]
    #[doc(alias = "kalman_get_state_transition")]
    pub fn state_transition_mut(&mut self) -> &mut A {
        &mut self.A
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    /// Gets a reference to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance")]
    pub fn estimate_covariance(&self) -> &P {
        &self.P
    }

    /// Gets a mutable reference to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance_mut")]
    #[doc(alias = "kalman_get_system_covariance")]
    pub fn estimate_covariance_mut(&mut self) -> &mut P {
        &mut self.P
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP>
    RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
{
    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the control covariance and variables are already set in the filter structure.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::prelude::*;
    /// use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_CONTROLS: usize = 0;
    /// # const NUM_OBSERVATIONS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_P,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #  );
    /// #
    /// # // Observation buffers.
    /// # impl_buffer_z!(mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_H!(mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_R!(mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_y!(mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_S!(mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// #
    /// # // Observation temporaries.
    /// # impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
    /// #     gravity_H,
    /// #     gravity_z,
    /// #     gravity_R,
    /// #     gravity_y,
    /// #     gravity_S,
    /// #     gravity_K,
    /// #     gravity_temp_S_inv,
    /// #     gravity_temp_HP,
    /// #     gravity_temp_PHt,
    /// #     gravity_temp_KHP,
    /// # );
    /// #
    /// # const REAL_DISTANCE: &[f32] = &[0.0, 0.0, 0.0];
    /// # const OBSERVATION_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict();
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + OBSERVATION_ERROR[t];
    ///     measurement.measurement_vector_mut().apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    #[doc(alias = "kalman_predict")]
    pub fn predict(&mut self)
    where
        X: StateVectorMut<STATES, T>,
        A: StateTransitionMatrix<STATES, T>,
        PX: PredictedStateEstimateVector<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and control
        //* P = A*P*Aᵀ
        self.predict_P();
    }

    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the control covariance and variables are already set in the filter structure.
    ///
    /// ## Arguments
    /// * `lambda` - The estimation covariance scaling factor (0 < `lambda` <= 1) to forcibly reduce prediction certainty. Smaller values mean larger uncertainty.
    ///
    /// ## Tuning Factor (lambda)
    /// In general, a process noise component is factored into the filter's state estimation
    /// covariance matrix update. Since it can be difficult to create a correct process noise
    /// matrix, this function incorporates a scaling factor of 1/λ² into the update process,
    /// where a value of 1.0 resembles no change.
    /// Smaller values correspond to a higher uncertainty increase.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::prelude::*;
    /// use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_CONTROLS: usize = 0;
    /// # const NUM_OBSERVATIONS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_P,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #  );
    /// #
    /// # // Observation buffers.
    /// # impl_buffer_z!(mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_H!(mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_R!(mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_y!(mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_S!(mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// #
    /// # // Observation temporaries.
    /// # impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
    /// #     gravity_H,
    /// #     gravity_z,
    /// #     gravity_R,
    /// #     gravity_y,
    /// #     gravity_S,
    /// #     gravity_K,
    /// #     gravity_temp_S_inv,
    /// #     gravity_temp_HP,
    /// #     gravity_temp_PHt,
    /// #     gravity_temp_KHP,
    /// # );
    /// #
    /// # const REAL_DISTANCE: &[f32] = &[0.0, 0.0, 0.0];
    /// # const OBSERVATION_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// const LAMBDA: f32 = 0.97;
    ///
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict_tuned(LAMBDA);
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + OBSERVATION_ERROR[t];
    ///     measurement.measurement_vector_mut().apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    #[doc(alias = "kalman_predict_tuned")]
    pub fn predict_tuned(&mut self, lambda: T)
    where
        X: StateVectorMut<STATES, T>,
        A: StateTransitionMatrix<STATES, T>,
        PX: PredictedStateEstimateVector<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // Predict next state using system dynamics
        // x = A*x
        self.predict_x();

        // Predict next covariance using system dynamics and control
        // P = A*P*Aᵀ * 1/lambda^2
        self.predict_P_tuned(lambda);
    }

    /// Performs the time update / prediction step of only the state vector
    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_x")]
    fn predict_x(&mut self)
    where
        X: StateVectorMut<STATES, T>,
        A: StateTransitionMatrix<STATES, T>,
        PX: PredictedStateEstimateVector<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let x = self.x.as_matrix_mut();

        // temporaries
        let x_predicted = self.predicted_x.as_matrix_mut();

        // Predict next state using system dynamics
        // x = A*x

        A.mult_rowvector(x, x_predicted);
        x_predicted.copy(x);
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_P")]
    fn predict_P(&mut self)
    where
        A: StateTransitionMatrix<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let P = self.P.as_matrix_mut();

        // temporaries
        let P_temp = self.temp_P.as_matrix_mut();

        // Predict next covariance using system dynamics (without control)

        // P = A*P*Aᵀ
        A.mult(P, P_temp); // temp = A*P
        P_temp.mult_transb(A, P); // P = temp*Aᵀ
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_P_tuned(&mut self, lambda: T)
    where
        A: StateTransitionMatrix<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let P = self.P.as_matrix_mut();

        // temporaries
        let P_temp = self.temp_P.as_matrix_mut();

        // Predict next covariance using system dynamics (without control)
        // P = A*P*Aᵀ * 1/lambda^2

        // lambda = 1/lambda^2
        let lambda = lambda.mul(lambda).recip(); // TODO: This should be precalculated, e.g. using set_lambda(...);

        // P = A*P*A'
        A.mult(P, P_temp); // temp = A*P
        P_temp.multscale_transb(A, lambda, P); // P = temp*A' * 1/(lambda^2)
    }

    /// Applies a control input.
    #[inline(always)]
    pub fn control<I>(&mut self, control: &mut I)
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        I: KalmanFilterControlApplyToFilter<STATES, T>,
    {
        control.apply_to(&mut self.x, &mut self.P)
    }

    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::prelude::*;
    /// use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_CONTROLS: usize = 0;
    /// # const NUM_OBSERVATIONS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_P,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #  );
    /// #
    /// # // Observation buffers.
    /// # impl_buffer_z!(mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_H!(mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_R!(mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_y!(mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_S!(mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// #
    /// # // Observation temporaries.
    /// # impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// # impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
    /// #     gravity_H,
    /// #     gravity_z,
    /// #     gravity_R,
    /// #     gravity_y,
    /// #     gravity_S,
    /// #     gravity_K,
    /// #     gravity_temp_S_inv,
    /// #     gravity_temp_HP,
    /// #     gravity_temp_PHt,
    /// #     gravity_temp_KHP,
    /// # );
    /// #
    /// # const REAL_DISTANCE: &[f32] = &[0.0, 0.0, 0.0];
    /// # const OBSERVATION_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict();
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + OBSERVATION_ERROR[t];
    ///     measurement.measurement_vector_mut().apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    pub fn correct<M>(&mut self, measurement: &mut M)
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
    {
        measurement.correct(&mut self.x, &mut self.P);
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterNumStates<STATES>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
{
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterStateVector<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    X: StateVector<STATES, T>,
{
    type StateVector = X;

    #[inline(always)]
    fn state_vector(&self) -> &Self::StateVector {
        self.state_vector()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterStateVectorMut<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    X: StateVectorMut<STATES, T>,
{
    type StateVectorMut = X;

    #[inline(always)]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut {
        self.state_vector_mut()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterStateTransition<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    A: StateTransitionMatrix<STATES, T>,
{
    type StateTransitionMatrix = A;

    #[inline(always)]
    fn state_transition(&self) -> &Self::StateTransitionMatrix {
        self.state_transition()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterStateTransitionMut<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    A: StateTransitionMatrixMut<STATES, T>,
{
    type StateTransitionMatrixMut = A;

    #[inline(always)]
    fn state_transition_mut(&mut self) -> &mut Self::StateTransitionMatrixMut {
        self.state_transition_mut()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterEstimateCovariance<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    type EstimateCovarianceMatrix = P;

    #[inline(always)]
    fn estimate_covariance(&self) -> &Self::EstimateCovarianceMatrix {
        self.estimate_covariance()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterEstimateCovarianceMut<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    type EstimateCovarianceMatrixMut = P;

    #[inline(always)]
    fn estimate_covariance_mut(&mut self) -> &mut Self::EstimateCovarianceMatrixMut {
        self.estimate_covariance_mut()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: DirectProcessNoiseCovarianceMatrix<STATES, T>,
{
    /// Gets a reference to the direct process noise matrix Q.
    ///
    /// This matrix represents the process noise covariance. It quantifies the uncertainty
    /// introduced by the model dynamics and process variations, providing a measure of
    /// how much the true state is expected to deviate from the predicted state due to
    /// inherent system noise and external influences.
    #[inline(always)]
    pub fn direct_process_noise(&self) -> &Q {
        &self.Q
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
{
    /// Gets a mutable reference to the direct process noise matrix Q.
    ///
    /// This matrix represents the process noise covariance. It quantifies the uncertainty
    /// introduced by the model dynamics and process variations, providing a measure of
    /// how much the true state is expected to deviate from the predicted state due to
    /// inherent system noise and external influences.
    #[inline(always)]
    pub fn direct_process_noise_mut(&mut self) -> &mut Q {
        &mut self.Q
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterPredict<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    X: StateVectorMut<STATES, T>,
    A: StateTransitionMatrix<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    P: EstimateCovarianceMatrix<STATES, T>,
    TempP: TemporaryStateMatrix<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn predict(&mut self) {
        self.predict()
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterUpdate<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn correct<M>(&mut self, measurement: &mut M)
    where
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
    {
        self.correct(measurement)
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterApplyControl<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn control<I>(&mut self, control: &mut I)
    where
        I: KalmanFilterControlApplyToFilter<STATES, T>,
    {
        self.control(control)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, MatrixMut};
    use crate::test_dummies::make_dummy_filter;

    fn trait_impl<const STATES: usize, T, K>(mut filter: K) -> K
    where
        K: KalmanFilter<STATES, T> + KalmanFilterStateTransitionMut<STATES, T>,
    {
        assert_eq!(filter.states(), STATES);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = filter.state_vector();
        let _vec = filter.state_vector_mut();
        let _ = filter.state_vector().as_matrix().inspect(|_vec| test_fn());
        let _ = filter
            .state_vector_mut()
            .as_matrix()
            .inspect(|_vec| test_fn_mut());
        filter
            .state_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn());
        filter
            .state_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn_mut());

        let _mat = filter.state_transition();
        let _mat = filter.state_transition_mut();
        let _ = filter
            .state_transition()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = filter
            .state_transition_mut()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        filter
            .state_transition_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        filter
            .state_transition_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        let _mat = filter.estimate_covariance();
        let _mat = filter.estimate_covariance_mut();
        let _ = filter
            .estimate_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = filter
            .estimate_covariance_mut()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        filter
            .estimate_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        filter
            .estimate_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        filter.predict();

        filter
    }

    #[test]
    fn builder_simple() {
        let filter = make_dummy_filter();

        let mut filter = trait_impl(filter);
        assert_eq!(filter.states(), 3);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = filter.state_vector();
        let _vec = filter.state_vector_mut();
        let _ = filter.state_vector().as_matrix().inspect(|_vec| test_fn());
        let _ = filter
            .state_vector_mut()
            .as_matrix()
            .inspect(|_vec| test_fn_mut());
        filter
            .state_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn());
        filter
            .state_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn_mut());

        let _mat = filter.state_transition();
        let _mat = filter.state_transition_mut();
        let _ = filter
            .state_transition()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = filter
            .state_transition_mut()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        filter
            .state_transition_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        filter
            .state_transition_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        let _mat = filter.estimate_covariance();
        let _mat = filter.estimate_covariance_mut();
        let _ = filter
            .estimate_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = filter
            .estimate_covariance_mut()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        filter
            .estimate_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        filter
            .estimate_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        filter.predict();
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn it_works() {
        use crate::prelude::*;
        use assert_float_eq::*;

        let mut example = crate::test_filter::create_test_filter(1.0);

        // The estimate covariance still is scalar.
        assert!(example
            .filter
            .estimate_covariance()
            .inspect(|mat| (0..3).into_iter().all(|i| { mat.get_at(i, i) == 0.1 })));

        // Since our initial state is zero, any number of prediction steps keeps the filter unchanged.
        for _ in 0..10 {
            example.filter.predict();
        }

        // All states are zero.
        assert!(example
            .filter
            .state_vector()
            .as_ref()
            .iter()
            .all(|&x| x == 0.0));

        // The estimate covariance has changed.
        example.filter.estimate_covariance().inspect(|mat| {
            assert_f32_near!(mat.get_at(0, 0), 260.1);
            assert_f32_near!(mat.get_at(1, 1), 10.1);
            assert_f32_near!(mat.get_at(2, 2), 0.1);
        });

        // The measurement is zero.
        example
            .measurement
            .measurement_vector_mut()
            .set_at(0, 0, 0.0);

        // Apply a measurement of the unchanged state.
        example.filter.correct(&mut example.measurement);

        // All states are still zero.
        assert!(example
            .filter
            .state_vector()
            .as_ref()
            .iter()
            .all(|&x| x == 0.0));

        // The estimate covariance has improved.
        example.filter.estimate_covariance().inspect(|mat| {
            assert!(mat.get_at(0, 0) < 1.0);
            assert!(mat.get_at(1, 1) < 0.2);
            assert!(mat.get_at(2, 2) < 0.01);
        });

        // Set an input.
        example.control.control_vector_mut().set_at(0, 0, 1.0);

        // Predict and apply an input.
        example.filter.predict();
        example.filter.control(&mut example.control);

        // All states are still zero.
        example.filter.state_vector().inspect(|vec| {
            assert_eq!(
                vec.get_at(0, 0),
                0.5,
                "incorrect position after control input"
            );
            assert_eq!(
                vec.get_at(1, 0),
                1.0,
                "incorrect velocity after control input"
            );
            assert_eq!(
                vec.get_at(2, 0),
                1.0,
                "incorrect acceleration after control input"
            );
        });

        // Predict without input.
        example.filter.predict();

        // All states are still zero.
        example.filter.state_vector().inspect(|vec| {
            assert_eq!(vec.get_at(0, 0), 2.0, "incorrect position");
            assert_eq!(vec.get_at(1, 0), 2.0, "incorrect velocity");
            assert_eq!(vec.get_at(2, 0), 1.0, "incorrect acceleration");
        });

        // The estimate covariance has worsened.
        example.filter.estimate_covariance().inspect(|mat| {
            assert!(mat.get_at(0, 0) > 6.2);
            assert!(mat.get_at(1, 1) > 4.2);
            assert!(mat.get_at(2, 2) > 1.0);
        });

        // Set a new measurement
        example.measurement.measurement_vector_mut().apply(|vec| {
            vec.set_at(0, 0, 2.0);
            vec.set_at(1, 0, (2.0 + 2.0 + 1.0) / 3.0);
        });

        // Apply a measurement of the state.
        example.filter.correct(&mut example.measurement);

        // The estimate covariance has improved.
        example.filter.estimate_covariance().inspect(|mat| {
            assert!(mat.get_at(0, 0) < 1.0);
            assert!(mat.get_at(1, 1) < 1.0);
            assert!(mat.get_at(2, 2) < 0.4);
        });
    }
}
