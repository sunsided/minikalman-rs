use core::marker::PhantomData;
use minikalman_traits::kalman::*;
use minikalman_traits::matrix::*;

use crate::measurement::Measurement;

/// A builder for a [`Kalman`] filter instances.
#[allow(clippy::type_complexity)]
pub struct KalmanBuilder<A, X, B, U, P, Q, PX, TempP, TempBQ> {
    _phantom: (
        PhantomData<A>,
        PhantomData<X>,
        PhantomData<B>,
        PhantomData<U>,
        PhantomData<P>,
        PhantomData<Q>,
        PhantomData<PX>,
        PhantomData<TempP>,
        PhantomData<TempBQ>,
    ),
}

impl<A, X, B, U, P, Q, PX, TempP, TempBQ> KalmanBuilder<A, X, B, U, P, Q, PX, TempP, TempBQ> {
    /// Initializes a Kalman filter instance.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `B` - The input transition matrix (`STATES` × `INPUTS`).
    /// * `u` - The input vector (`INPUTS` × `1`).
    /// * `P` - The state covariance matrix (`STATES` × `STATES`).
    /// * `Q` - The input covariance matrix (`INPUTS` × `INPUTS`).
    /// * `predictedX` - The temporary vector for predicted states (`STATES` × `1`).
    /// * `temp_P` - The temporary vector for P calculation (`STATES` × `STATES`).
    /// * `temp_BQ` - The temporary vector for B×Q calculation (`STATES` × `INPUTS`).
    ///
    /// ## Example
    ///
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// // System buffers.
    /// impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    ///
    /// // Input buffers.
    /// impl_buffer_u!(mut gravity_u, NUM_INPUTS, f32, 0.0);
    /// impl_buffer_B!(mut gravity_B, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// impl_buffer_Q!(mut gravity_Q, NUM_INPUTS, f32, 0.0);
    ///
    /// // Filter temporaries.
    /// impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// impl_buffer_temp_BQ!(mut gravity_temp_BQ, NUM_STATES, NUM_INPUTS, f32, 0.0);
    ///
    /// let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
    ///     gravity_A,
    ///     gravity_x,
    ///     gravity_B,
    ///     gravity_u,
    ///     gravity_P,
    ///     gravity_Q,
    ///     gravity_temp_x,
    ///     gravity_temp_P,
    ///     gravity_temp_BQ,
    ///  );
    /// ```
    #[allow(non_snake_case, clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub fn new<const STATES: usize, const INPUTS: usize, T>(
        A: A,
        x: X,
        B: B,
        u: U,
        P: P,
        Q: Q,
        predicted_x: PX,
        temp_P: TempP,
        temp_BQ: TempBQ,
    ) -> Kalman<STATES, INPUTS, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
    where
        T: MatrixDataType,
        A: SystemMatrix<STATES, T>,
        X: StateVector<STATES, T>,
        B: InputMatrix<STATES, INPUTS, T>,
        U: InputVector<INPUTS, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        Q: InputCovarianceMatrix<INPUTS, T>,
        PX: StatePredictionVector<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
    {
        Kalman::<STATES, INPUTS, T, _, _, _, _, _, _, _, _, _> {
            x,
            A,
            P,
            u,
            B,
            Q,
            predicted_x,
            temp_P,
            temp_BQ,
            _phantom: Default::default(),
        }
    }
}

/// Kalman Filter structure.  See [`KalmanBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Kalman<const STATES: usize, const INPUTS: usize, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
{
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

    /// Input vector.
    u: U,

    /// Input matrix.
    ///
    /// See also [`Q`].
    B: B,

    /// Input covariance matrix.
    ///
    /// See also [`B`].
    Q: Q,

    /// x-sized temporary vector.
    predicted_x: PX,

    /// P-Sized temporary matrix (number of states × number of states).
    ///
    /// The backing field for this temporary MAY be aliased with temporary BQ.
    temp_P: TempP,

    /// B×Q-sized temporary matrix (number of states × number of inputs).
    ///
    /// The backing field for this temporary MAY be aliased with temporary P.
    temp_BQ: TempBQ,

    _phantom: PhantomData<T>,
}

impl<const STATES: usize, const INPUTS: usize, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
{
    /// Returns the number of states.
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Returns the number of inputs.
    pub const fn inputs(&self) -> usize {
        INPUTS
    }

    /// Gets a reference to the state vector x.
    #[inline(always)]
    pub fn state_vector_ref(&self) -> &X {
        &self.x
    }

    /// Gets a reference to the state vector x.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_vector")]
    pub fn state_vector_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Applies a function to the state vector x.
    #[inline(always)]
    pub fn state_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut X),
    {
        f(&mut self.x)
    }

    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_ref(&self) -> &A {
        &self.A
    }

    /// Gets a reference to the system covariance matrix P.
    #[inline(always)]
    pub fn system_covariance_ref(&self) -> &P {
        &self.P
    }

    /// Gets a mutable reference to the system covariance matrix P.
    #[inline(always)]
    #[doc(alias = "kalman_get_system_covariance")]
    pub fn system_covariance_mut(&mut self) -> &mut P {
        &mut self.P
    }

    /// Applies a function to the system covariance matrix P.
    #[inline(always)]
    pub fn system_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut P),
    {
        f(&mut self.P)
    }

    /// Gets a reference to the input vector u.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_vector")]
    pub fn input_vector_ref(&self) -> &U {
        &self.u
    }

    /// Gets a reference to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_ref(&self) -> &B {
        &self.B
    }

    /// Gets a reference to the input covariance matrix Q.
    #[inline(always)]
    pub fn input_covariance_ref(&self) -> &Q {
        &self.Q
    }
}

impl<const STATES: usize, const INPUTS: usize, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
where
    A: SystemMatrixMut<STATES, T>,
{
    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_transition")]
    pub fn state_transition_mut(&mut self) -> &mut A {
        &mut self.A
    }

    /// Applies a function to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut A),
    {
        f(&mut self.A)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
where
    U: InputVectorMut<STATES, T>,
{
    /// Gets a mutable reference to the input vector u.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_vector")]
    pub fn input_vector_mut(&mut self) -> &mut U {
        &mut self.u
    }

    /// Applies a function to the input vector u.
    #[inline(always)]
    pub fn input_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut U),
    {
        f(&mut self.u)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
where
    B: InputMatrixMut<STATES, INPUTS, T>,
{
    /// Gets a mutable reference to the input transition matrix B.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_transition")]
    pub fn input_transition_mut(&mut self) -> &mut B {
        &mut self.B
    }

    /// Applies a function to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut B),
    {
        f(&mut self.B)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
where
    Q: InputCovarianceMatrixMut<INPUTS, T>,
{
    /// Gets a mutable reference to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    pub fn input_covariance_mut(&mut self) -> &mut Q {
        &mut self.Q
    }

    /// Applies a function to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    pub fn input_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Q),
    {
        f(&mut self.Q)
    }
}

impl<const STATES: usize, const INPUTS: usize, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, A, X, B, U, P, Q, PX, TempP, TempBQ>
{
    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the input covariance and variables are already set in the filter structure.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Input buffers.
    /// # impl_buffer_u!(mut gravity_u, NUM_INPUTS, f32, 0.0);
    /// # impl_buffer_B!(mut gravity_B, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// # impl_buffer_Q!(mut gravity_Q, NUM_INPUTS, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_BQ!(mut gravity_temp_BQ, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// #
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_B,
    /// #     gravity_u,
    /// #     gravity_P,
    /// #     gravity_Q,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #     gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # impl_buffer_z!(mut gravity_z, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_H!(mut gravity_H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_R!(mut gravity_R, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_y!(mut gravity_y, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_S!(mut gravity_S, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// #
    /// # // Measurement temporaries.
    /// # impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
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
    /// # const MEASUREMENT_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict();
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    #[doc(alias = "kalman_predict")]
    pub fn predict(&mut self)
    where
        X: StateVector<STATES, T>,
        A: SystemMatrix<STATES, T>,
        PX: StatePredictionVector<STATES, T>,
        B: InputMatrix<STATES, INPUTS, T>,
        Q: InputCovarianceMatrix<INPUTS, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
        T: MatrixDataType,
    {
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and input
        //* P = A*P*A' + B*Q*B'
        self.predict_Q();
    }

    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the input covariance and variables are already set in the filter structure.
    ///
    /// ## Arguments
    /// * `lambda` - Lambda factor (0 < `lambda` <= 1) to forcibly reduce prediction certainty. Smaller values mean larger uncertainty.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Input buffers.
    /// # impl_buffer_u!(mut gravity_u, NUM_INPUTS, f32, 0.0);
    /// # impl_buffer_B!(mut gravity_B, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// # impl_buffer_Q!(mut gravity_Q, NUM_INPUTS, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_BQ!(mut gravity_temp_BQ, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// #
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_B,
    /// #     gravity_u,
    /// #     gravity_P,
    /// #     gravity_Q,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #     gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # impl_buffer_z!(mut gravity_z, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_H!(mut gravity_H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_R!(mut gravity_R, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_y!(mut gravity_y, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_S!(mut gravity_S, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// #
    /// # // Measurement temporaries.
    /// # impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
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
    /// # const MEASUREMENT_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// const LAMBDA: f32 = 0.97;
    ///
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict_tuned(LAMBDA);
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    #[doc(alias = "kalman_predict_tuned")]
    pub fn predict_tuned(&mut self, lambda: T)
    where
        X: StateVector<STATES, T>,
        A: SystemMatrix<STATES, T>,
        PX: StatePredictionVector<STATES, T>,
        B: InputMatrix<STATES, INPUTS, T>,
        Q: InputCovarianceMatrix<INPUTS, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
        T: MatrixDataType,
    {
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and input
        //* P = A*P*A' * 1/lambda^2 + B*Q*B'
        self.predict_Q_tuned(lambda);
    }

    /// Performs the time update / prediction step of only the state vector
    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_x")]
    fn predict_x(&mut self)
    where
        X: StateVector<STATES, T>,
        A: SystemMatrix<STATES, T>,
        PX: StatePredictionVector<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let x = self.x.as_matrix_mut();

        // temporaries
        let x_predicted = self.predicted_x.as_matrix_mut();

        //* Predict next state using system dynamics
        //* x = A*x

        A.mult_rowvector(x, x_predicted);
        x_predicted.copy(x);

        // TODO: Add missing input
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_Q(&mut self)
    where
        A: SystemMatrix<STATES, T>,
        B: InputMatrix<STATES, INPUTS, T>,
        Q: InputCovarianceMatrix<INPUTS, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let B = self.B.as_matrix();
        let Q = self.Q.as_matrix();
        let P = self.P.as_matrix_mut();

        // temporaries
        let P_temp = self.temp_P.as_matrix_mut();
        let BQ_temp = self.temp_BQ.as_matrix_mut();

        //* Predict next covariance using system dynamics and input
        //* P = A*P*A' + B*Q*B'

        // P = A*P*A'
        A.mult(P, P_temp); // temp = A*P
        P_temp.mult_transb(A, P); // P = temp*A'

        // P = P + B*Q*B'
        if !B.is_empty() {
            B.mult(Q, BQ_temp); // temp = B*Q
            BQ_temp.multadd_transb(B, P); // P += temp*B'
        }
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_Q_tuned(&mut self, lambda: T)
    where
        A: SystemMatrix<STATES, T>,
        B: InputMatrix<STATES, INPUTS, T>,
        Q: InputCovarianceMatrix<INPUTS, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let B = self.B.as_matrix();
        let Q = self.Q.as_matrix();
        let P = self.P.as_matrix_mut();

        // temporaries
        let P_temp = self.temp_P.as_matrix_mut();
        let BQ_temp = self.temp_BQ.as_matrix_mut();

        //* Predict next covariance using system dynamics and input
        //* P = A*P*A' * 1/lambda^2 + B*Q*B'

        // lambda = 1/lambda^2
        let lambda = lambda.mul(lambda).recip(); // TODO: This should be precalculated, e.g. using set_lambda(...);

        // P = A*P*A'
        A.mult(P, P_temp); // temp = A*P
        P_temp.multscale_transb(A, lambda, P); // P = temp*A' * 1/(lambda^2)

        // P = P + B*Q*B'
        if !B.is_empty() {
            B.mult(Q, BQ_temp); // temp = B*Q
            BQ_temp.multadd_transb(B, P); // P += temp*B'
        }
    }

    /// Performs the measurement update step.
    ///
    /// ## Arguments
    /// * `kfm` - The measurement.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Input buffers.
    /// # impl_buffer_u!(mut gravity_u, NUM_INPUTS, f32, 0.0);
    /// # impl_buffer_B!(mut gravity_B, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// # impl_buffer_Q!(mut gravity_Q, NUM_INPUTS, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_BQ!(mut gravity_temp_BQ, NUM_STATES, NUM_INPUTS, f32, 0.0);
    /// #
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_B,
    /// #     gravity_u,
    /// #     gravity_P,
    /// #     gravity_Q,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #     gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # impl_buffer_z!(mut gravity_z, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_H!(mut gravity_H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_R!(mut gravity_R, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_y!(mut gravity_y, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_S!(mut gravity_S, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// #
    /// # // Measurement temporaries.
    /// # impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// # impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
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
    /// # const MEASUREMENT_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict();
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    #[allow(non_snake_case)]
    pub fn correct<
        const MEASUREMENTS: usize,
        Z,
        H,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempHP,
        TempPHt,
        TempKHP,
    >(
        &mut self,
        kfm: &mut Measurement<
            STATES,
            MEASUREMENTS,
            T,
            H,
            Z,
            R,
            Y,
            S,
            K,
            TempSInv,
            TempHP,
            TempPHt,
            TempKHP,
        >,
    ) where
        P: SystemCovarianceMatrix<STATES, T>,
        X: StateVector<STATES, T>,
        H: MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>,
        K: KalmanGainMatrix<STATES, MEASUREMENTS, T>,
        S: ResidualCovarianceMatrix<MEASUREMENTS, T>,
        R: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
        Y: InnovationVector<MEASUREMENTS, T>,
        Z: MeasurementVector<MEASUREMENTS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>,
        TempHP: TemporaryHPMatrix<MEASUREMENTS, STATES, T>,

        TempPHt: TemporaryPHTMatrix<STATES, MEASUREMENTS, T>,
        TempKHP: TemporaryKHPMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let P = self.P.as_matrix_mut();
        let x = self.x.as_matrix_mut();

        let H = kfm.H.as_matrix();
        let K = kfm.K.as_matrix_mut();
        let S = kfm.S.as_matrix_mut();
        let R = kfm.R.as_matrix_mut();
        let y = kfm.y.as_matrix_mut();
        let z = kfm.z.as_matrix();

        // temporaries
        let S_inv = kfm.temp_S_inv.as_matrix_mut();
        let temp_HP = kfm.temp_HP.as_matrix_mut();
        let temp_KHP = kfm.temp_KHP.as_matrix_mut();
        let temp_PHt = kfm.temp_PHt.as_matrix_mut();

        //* Calculate innovation and residual covariance
        //* y = z - H*x
        //* S = H*P*H' + R

        // y = z - H*x
        H.mult_rowvector(x, y);
        z.sub_inplace_b(y);

        // S = H*P*H' + R
        H.mult(P, temp_HP); // temp = H*P
        temp_HP.mult_transb(H, S); // S = temp*H'
        S.add_inplace_a(R); // S += R

        //* Calculate Kalman gain
        //* K = P*H' * S^-1

        // K = P*H' * S^-1
        S.cholesky_decompose_lower();
        S.invert_l_cholesky(S_inv); // S_inv = S^-1
                                    // NOTE that to allow aliasing of Sinv and temp_PHt, a copy must be performed here
        P.mult_transb(H, temp_PHt); // temp = P*H'
        temp_PHt.mult(S_inv, K); // K = temp*Sinv

        //* Correct state prediction
        //* x = x + K*y

        // x = x + K*y
        K.multadd_rowvector(y, x);

        //* Correct state covariances
        //* P = (I-K*H) * P
        //*   = P - K*(H*P)

        // P = P - K*(H*P)
        H.mult(P, temp_HP); // temp_HP = H*P
        K.mult(temp_HP, temp_KHP); // temp_KHP = K*temp_HP
        P.sub_inplace_a(temp_KHP); // P -= temp_KHP
    }
}

#[cfg(test)]
mod tests {
    use core::ops::{Index, IndexMut};

    use minikalman_traits::matrix::{Matrix, MatrixMut};

    use super::*;

    #[test]
    fn builder_simple() {
        let _filter = KalmanBuilder::new::<3, 0, f32>(
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
        );
    }

    #[derive(Default)]
    struct Dummy<T>(DummyMatrix<T>, PhantomData<T>);

    #[derive(Default)]
    struct DummyMatrix<T>(PhantomData<T>);

    impl<const STATES: usize, T> StateVector<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, T> SystemMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<const STATES: usize, T> SystemMatrixMut<STATES, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, T> SystemCovarianceMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const INPUTS: usize, T> InputVector<INPUTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<const INPUTS: usize, T> InputVectorMut<INPUTS, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, const INPUTS: usize, T> InputMatrix<STATES, INPUTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<const STATES: usize, const INPUTS: usize, T> InputMatrixMut<STATES, INPUTS, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const INPUTS: usize, T> InputCovarianceMatrix<INPUTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<const INPUTS: usize, T> InputCovarianceMatrixMut<INPUTS, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }
    impl<const STATES: usize, T> StatePredictionVector<STATES, T> for Dummy<T> {
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
    impl<const STATES: usize, const INPUTS: usize, T> TemporaryBQMatrix<STATES, INPUTS, T>
        for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<T> AsRef<[T]> for DummyMatrix<T> {
        fn as_ref(&self) -> &[T] {
            todo!()
        }
    }

    impl<T> AsMut<[T]> for DummyMatrix<T> {
        fn as_mut(&mut self) -> &mut [T] {
            todo!()
        }
    }

    impl<T> Index<usize> for DummyMatrix<T> {
        type Output = T;

        fn index(&self, _index: usize) -> &Self::Output {
            todo!()
        }
    }

    impl<T> IndexMut<usize> for DummyMatrix<T> {
        fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
            todo!()
        }
    }

    impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix<T> {}
    impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T> for DummyMatrix<T> {}
}
