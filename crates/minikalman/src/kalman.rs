use core::marker::PhantomData;

mod filter_trait;
mod matrix_types;

use crate::matrix::{Matrix, MatrixDataType};
pub use filter_trait::*;
pub use matrix_types::*;

/// A builder for a [`Kalman`] filter instances.
#[allow(clippy::type_complexity)]
pub struct KalmanBuilder<A, X, P, PX, TempP> {
    _phantom: (
        PhantomData<A>,
        PhantomData<X>,
        PhantomData<P>,
        PhantomData<PX>,
        PhantomData<TempP>,
    ),
}

impl<A, X, P, PX, TempP> KalmanBuilder<A, X, P, PX, TempP> {
    /// Initializes a Kalman filter instance.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `B` - The control transition matrix (`STATES` × `CONTROLS`).
    /// * `u` - The control vector (`CONTROLS` × `1`).
    /// * `P` - The state covariance matrix (`STATES` × `STATES`).
    /// * `Q` - The control covariance matrix (`CONTROLS` × `CONTROLS`).
    /// * `predictedX` - The temporary vector for predicted states (`STATES` × `1`).
    /// * `temp_P` - The temporary vector for P calculation (`STATES` × `STATES`).
    /// * `temp_BQ` - The temporary vector for B×Q calculation (`STATES` × `CONTROLS`).
    ///
    /// ## Example
    ///
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_CONTROLS: usize = 0;
    /// # const NUM_OBSERVATIONS: usize = 1;
    /// // System buffers.
    /// impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    ///
    /// // Filter temporaries.
    /// impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    ///
    /// let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
    ///     gravity_A,
    ///     gravity_x,
    ///     gravity_P,
    ///     gravity_temp_x,
    ///     gravity_temp_P,
    ///  );
    /// ```
    #[allow(non_snake_case, clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub fn new<const STATES: usize, T>(
        A: A,
        x: X,
        P: P,
        predicted_x: PX,
        temp_P: TempP,
    ) -> Kalman<STATES, T, A, X, P, PX, TempP>
    where
        T: MatrixDataType,
        A: StateTransitionMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        PX: PredictedStateEstimateVector<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
    {
        Kalman::<STATES, T, _, _, _, _, _> {
            x,
            A,
            P,
            predicted_x,
            temp_P,
            _phantom: Default::default(),
        }
    }
}

/// Kalman Filter structure.  See [`KalmanBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Kalman<const STATES: usize, T, A, X, P, PX, TempP> {
    /// State vector.
    x: X,

    /// System matrix.
    ///
    /// See also [`P`].
    A: A,

    /// System covariance matrix.
    ///
    /// See also [`A`].
    P: P,

    /// x-sized temporary vector.
    predicted_x: PX,

    /// P-Sized temporary matrix (number of states × number of states).
    ///
    /// The backing field for this temporary MAY be aliased with temporary BQ.
    temp_P: TempP,

    _phantom: PhantomData<T>,
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP> {
    /// Returns the number of states.
    pub const fn states(&self) -> usize {
        STATES
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
where
    X: StateVector<STATES, T>,
{
    /// Gets a reference to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    pub fn state_vector_ref(&self) -> &X {
        &self.x
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
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

    /// Applies a function to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    pub fn state_vector_apply<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&mut X) -> O,
    {
        f(&mut self.x)
    }

    /// Applies a function to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    pub fn state_vector_apply_mut<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut X) -> O,
    {
        f(&mut self.x)
    }

    /// Applies a function to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    pub fn state_vector_inspect<F, O>(&self, f: F) -> O
    where
        F: Fn(&X) -> O,
    {
        f(&self.x)
    }

    /// Applies a function to the state vector x.
    ///
    /// The state vector represents the internal state of the system at a given time. It contains
    /// all the necessary information to describe the system's current situation.
    #[inline(always)]
    pub fn state_vector_inspect_mut<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&X) -> O,
    {
        f(&self.x)
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
where
    A: StateTransitionMatrix<STATES, T>,
{
    /// Gets a reference to the state transition matrix A/F.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_ref")]
    pub fn state_transition_ref(&self) -> &A {
        &self.A
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
where
    A: StateTransitionMatrixMut<STATES, T>,
{
    /// Gets a reference to the state transition matrix A/F.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_mut")]
    #[doc(alias = "kalman_get_state_transition")]
    pub fn state_transition_mut(&mut self) -> &mut A {
        &mut self.A
    }

    /// Applies a function to the state transition matrix A.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_apply")]
    pub fn state_transition_apply<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&mut A) -> O,
    {
        f(&mut self.A)
    }

    /// Applies a function to the state transition matrix A.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_apply_mut")]
    pub fn state_transition_apply_mut<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut A) -> O,
    {
        f(&mut self.A)
    }

    /// Applies a function to the state transition matrix A.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_inspect")]
    pub fn state_transition_inspect<F, O>(&self, f: F) -> O
    where
        F: Fn(&A) -> O,
    {
        f(&self.A)
    }

    /// Applies a function to the state transition matrix A.
    ///
    /// This matrix describes how the state vector evolves from one time step to the next in the
    /// absence of control inputs. It defines the relationship between the previous state and the
    /// current state, accounting for the inherent dynamics of the system.
    #[inline(always)]
    #[doc(alias = "system_matrix_inspect_mut")]
    pub fn state_transition_inspect_mut<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&A) -> O,
    {
        f(&self.A)
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    /// Gets a reference to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance_ref")]
    pub fn estimate_covariance_ref(&self) -> &P {
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

    /// Applies a function to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance_apply")]
    pub fn estimate_covariance_apply<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&mut P) -> O,
    {
        f(&mut self.P)
    }

    /// Applies a function to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance_apply")]
    pub fn estimate_covariance_apply_mut<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut P) -> O,
    {
        f(&mut self.P)
    }

    /// Applies a function to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance_inspect")]
    pub fn estimate_covariance_inspect<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&P) -> O,
    {
        f(&self.P)
    }

    /// Applies a function to the system covariance matrix P.
    ///
    /// This matrix represents the uncertainty in the state estimate. It quantifies how much the
    /// state estimate is expected to vary, providing a measure of confidence in the estimate.
    #[inline(always)]
    #[doc(alias = "system_covariance_inspect")]
    pub fn estimate_covariance_inspect_mut<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&P) -> O,
    {
        f(&self.P)
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP> {
    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the control covariance and variables are already set in the filter structure.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
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
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
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
    /// # let mut measurement = ObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
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
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
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

        // TODO: Add control support
        //       P = P + B*Q*Bᵀ
    }

    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the control covariance and variables are already set in the filter structure.
    ///
    /// ## Arguments
    /// * `lambda` - Lambda factor (0 < `lambda` <= 1) to forcibly reduce prediction certainty. Smaller values mean larger uncertainty.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
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
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
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
    /// # let mut measurement = ObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
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
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
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
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and control
        //* P = A*P*Aᵀ * 1/lambda^2
        self.predict_P_tuned(lambda);

        // TODO: Add control support
        //       P = P + B*Q*Bᵀ
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
    /// # use minikalman::*;
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
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(
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
    /// # let mut measurement = ObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
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
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
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

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterNumStates<STATES>
    for Kalman<STATES, T, A, X, P, PX, TempP>
{
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterStateVector<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    X: StateVector<STATES, T>,
{
    type StateVector = X;

    #[inline(always)]
    fn state_vector_ref(&self) -> &Self::StateVector {
        self.state_vector_ref()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterStateVectorMut<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    X: StateVectorMut<STATES, T>,
{
    type StateVectorMut = X;

    #[inline(always)]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut {
        self.state_vector_mut()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterStateTransition<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    A: StateTransitionMatrix<STATES, T>,
{
    type StateTransitionMatrix = A;

    #[inline(always)]
    fn state_transition_ref(&self) -> &Self::StateTransitionMatrix {
        self.state_transition_ref()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterStateTransitionMut<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    A: StateTransitionMatrixMut<STATES, T>,
{
    type StateTransitionMatrixMut = A;

    #[inline(always)]
    fn state_transition_mut(&mut self) -> &mut Self::StateTransitionMatrixMut {
        self.state_transition_mut()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterSystemCovariance<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    type EstimateCovarianceMatrix = P;

    #[inline(always)]
    fn estimate_covariance_ref(&self) -> &Self::EstimateCovarianceMatrix {
        self.estimate_covariance_ref()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterSystemCovarianceMut<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    type EstimateCovarianceMatrixMut = P;

    #[inline(always)]
    fn estimate_covariance_mut(&mut self) -> &mut Self::EstimateCovarianceMatrixMut {
        self.estimate_covariance_mut()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterPredict<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
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

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterUpdate<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
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

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterApplyControl<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
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
    use crate::test_dummies::{Dummy, DummyMatrix};

    fn trait_impl<const STATES: usize, T, K>(mut filter: K) -> K
    where
        K: KalmanFilter<STATES, T> + KalmanFilterStateTransitionMut<STATES, T>,
    {
        assert_eq!(filter.states(), STATES);

        let test_fn = || {};

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
        };

        let _vec = filter.state_vector_ref();
        let _vec = filter.state_vector_mut();
        filter.state_vector_inspect(|_vec| test_fn());
        filter.state_vector_inspect_mut(|_vec| test_fn_mut());
        filter.state_vector_apply(|_vec| test_fn());
        filter.state_vector_apply_mut(|_vec| test_fn_mut());

        let _mat = filter.state_transition_ref();
        let _mat = filter.state_transition_mut();
        filter.state_transition_inspect(|_mat| test_fn());
        filter.state_transition_inspect_mut(|_mat| test_fn_mut());
        filter.state_transition_apply(|_mat| test_fn());
        filter.state_transition_apply_mut(|_mat| test_fn_mut());

        let _mat = filter.estimate_covariance_ref();
        let _mat = filter.estimate_covariance_mut();
        filter.estimate_covariance_inspect(|_mat| test_fn());
        filter.estimate_covariance_inspect_mut(|_mat| test_fn_mut());
        filter.estimate_covariance_apply(|_mat| test_fn());
        filter.estimate_covariance_apply_mut(|_mat| test_fn_mut());

        filter
    }

    #[test]
    fn builder_simple() {
        let filter = KalmanBuilder::new::<3, f32>(
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
        );

        let mut filter = trait_impl(filter);

        let test_fn = || {};

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
        };

        let _vec = filter.state_vector_ref();
        let _vec = filter.state_vector_mut();
        filter.state_vector_inspect(|_vec| test_fn());
        filter.state_vector_inspect_mut(|_vec| test_fn_mut());
        filter.state_vector_apply(|_vec| test_fn());
        filter.state_vector_apply_mut(|_vec| test_fn_mut());

        let _mat = filter.state_transition_ref();
        let _mat = filter.state_transition_mut();
        filter.state_transition_inspect(|_mat| test_fn());
        filter.state_transition_inspect_mut(|_mat| test_fn_mut());
        filter.state_transition_apply(|_mat| test_fn());
        filter.state_transition_apply_mut(|_mat| test_fn_mut());

        let _mat = filter.estimate_covariance_ref();
        let _mat = filter.estimate_covariance_mut();
        filter.estimate_covariance_inspect(|_mat| test_fn());
        filter.estimate_covariance_inspect_mut(|_mat| test_fn_mut());
        filter.estimate_covariance_apply(|_mat| test_fn());
        filter.estimate_covariance_apply_mut(|_mat| test_fn_mut());
    }

    impl<const STATES: usize, T> StateVector<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<const STATES: usize, T> StateVectorMut<STATES, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;
        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, T> StateTransitionMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<const STATES: usize, T> StateTransitionMatrixMut<STATES, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, T> EstimateCovarianceMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, T> PredictedStateEstimateVector<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, T> TemporaryStateMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    #[test]
    #[cfg(feature = "alloc")]
    fn it_works() {
        use crate::prelude::*;
        use assert_float_eq::*;

        let mut example = crate::test_filter::create_test_filter(1.0);

        // The estimate covariance still is scalar.
        assert!(example.filter.estimate_covariance_inspect(|mat| (0..3)
            .into_iter()
            .all(|i| { mat.get(i, i) == 0.1 })));

        // Since our initial state is zero, any number of prediction steps keeps the filter unchanged.
        for _ in 0..10 {
            example.filter.predict();
        }

        // All states are zero.
        assert!(example
            .filter
            .state_vector_ref()
            .as_ref()
            .iter()
            .all(|&x| x == 0.0));

        // The estimate covariance has changed.
        example.filter.estimate_covariance_inspect(|mat| {
            assert_f32_near!(mat.get(0, 0), 260.1);
            assert_f32_near!(mat.get(1, 1), 10.1);
            assert_f32_near!(mat.get(2, 2), 0.1);
        });

        // The measurement is zero.
        example.measurement.measurement_vector_mut().set(0, 0, 0.0);

        // Apply a measurement of the unchanged state.
        example.filter.correct(&mut example.measurement);

        // All states are still zero.
        assert!(example
            .filter
            .state_vector_ref()
            .as_ref()
            .iter()
            .all(|&x| x == 0.0));

        // The estimate covariance has improved.
        example.filter.estimate_covariance_inspect(|mat| {
            assert_f32_near!(mat.get(0, 0), 0.85736084);
            assert_f32_near!(mat.get(1, 1), 0.12626839);
            assert_f32_near!(mat.get(2, 2), 0.0040448904);
        });

        // Set an input.
        example.control.control_vector_mut().set(0, 0, 1.0);

        // Predict and apply an input.
        example.filter.predict();
        example.filter.control(&mut example.control);

        // All states are still zero.
        example.filter.state_vector_inspect(|vec| {
            assert_eq!(vec.get(0, 0), 0.5, "incorrect position after control input");
            assert_eq!(vec.get(1, 0), 1.0, "incorrect velocity after control input");
            assert_eq!(
                vec.get(2, 0),
                1.0,
                "incorrect acceleration after control input"
            );
        });

        // Predict without input.
        example.filter.predict();

        // All states are still zero.
        example.filter.state_vector_inspect(|vec| {
            assert_eq!(vec.get(0, 0), 2.0, "incorrect position");
            assert_eq!(vec.get(1, 0), 2.0, "incorrect velocity");
            assert_eq!(vec.get(2, 0), 1.0, "incorrect acceleration");
        });

        // The estimate covariance has worsened.
        example.filter.estimate_covariance_inspect(|mat| {
            assert_f32_near!(mat.get(0, 0), 6.226019);
            assert_f32_near!(mat.get(1, 1), 4.229596);
            assert_f32_near!(mat.get(2, 2), 1.0040449);
        });

        // Set a new measurement
        example.measurement.measurement_vector_apply(|vec| {
            vec.set(0, 0, 2.0);
            vec.set(1, 0, (2.0 + 2.0 + 1.0) / 3.0);
        });

        // Apply a measurement of the state.
        example.filter.correct(&mut example.measurement);

        // The estimate covariance has improved.
        example.filter.estimate_covariance_inspect(|mat| {
            assert_f32_near!(mat.get(0, 0), 0.6483326);
            assert_f32_near!(mat.get(1, 1), 0.8424177);
            assert_f32_near!(mat.get(2, 2), 0.27818835);
        });
    }
}
