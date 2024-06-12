use core::marker::PhantomData;

use minikalman_traits::kalman::*;
use minikalman_traits::matrix::*;

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
        A: SystemMatrix<STATES, T>,
        X: StateVector<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        PX: StatePredictionVector<STATES, T>,
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
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
where
    A: SystemMatrix<STATES, T>,
{
    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_ref(&self) -> &A {
        &self.A
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
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

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: SystemCovarianceMatrix<STATES, T>,
{
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
}

impl<const STATES: usize, T, A, X, P, PX, TempP> Kalman<STATES, T, A, X, P, PX, TempP> {
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
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and input
        //* P = A*P*Aᵀ
        self.predict_P();

        // TODO: Add input support
        //       P = P + B*Q*Bᵀ
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
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and input
        //* P = A*P*Aᵀ * 1/lambda^2
        self.predict_P_tuned(lambda);

        // TODO: Add input support
        //       P = P + B*Q*Bᵀ
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

        // Predict next state using system dynamics
        // x = A*x

        A.mult_rowvector(x, x_predicted);
        x_predicted.copy(x);
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_P")]
    fn predict_P(&mut self)
    where
        A: SystemMatrix<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let P = self.P.as_matrix_mut();

        // temporaries
        let P_temp = self.temp_P.as_matrix_mut();

        // Predict next covariance using system dynamics (without input)

        // P = A*P*Aᵀ
        A.mult(P, P_temp); // temp = A*P
        P_temp.mult_transb(A, P); // P = temp*Aᵀ
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_P_tuned(&mut self, lambda: T)
    where
        A: SystemMatrix<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = self.A.as_matrix();
        let P = self.P.as_matrix_mut();

        // temporaries
        let P_temp = self.temp_P.as_matrix_mut();

        // Predict next covariance using system dynamics (without input)
        // P = A*P*Aᵀ * 1/lambda^2

        // lambda = 1/lambda^2
        let lambda = lambda.mul(lambda).recip(); // TODO: This should be precalculated, e.g. using set_lambda(...);

        // P = A*P*A'
        A.mult(P, P_temp); // temp = A*P
        P_temp.multscale_transb(A, lambda, P); // P = temp*A' * 1/(lambda^2)
    }

    #[inline(always)]
    pub fn input<const INPUTS: usize, I>(&mut self, input: &mut I)
    where
        P: SystemCovarianceMatrix<STATES, T>,
        X: StateVector<STATES, T>,
        T: MatrixDataType,
        I: KalmanFilterInputApplyToFilter<STATES, T> + KalmanFilterNumInputs<INPUTS>,
    {
        input.apply_to(&mut self.x, &mut self.P)
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
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
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
    pub fn correct<const MEASUREMENTS: usize, M>(&mut self, measurement: &mut M)
    where
        P: SystemCovarianceMatrix<STATES, T>,
        X: StateVector<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterMeasurementCorrectFilter<STATES, T>
            + KalmanFilterNumMeasurements<MEASUREMENTS>,
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
    X: StateVector<STATES, T>,
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
    A: SystemMatrix<STATES, T>,
{
    type SystemMatrix = A;

    #[inline(always)]
    fn state_transition_ref(&self) -> &Self::SystemMatrix {
        self.state_transition_ref()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterStateTransitionMut<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    A: SystemMatrixMut<STATES, T>,
{
    type SystemMatrixMut = A;

    #[inline(always)]
    fn state_transition_mut(&mut self) -> &mut Self::SystemMatrixMut {
        self.state_transition_mut()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterSystemCovariance<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: SystemCovarianceMatrix<STATES, T>,
{
    type SystemCovarianceMatrix = P;

    #[inline(always)]
    fn system_covariance_ref(&self) -> &Self::SystemCovarianceMatrix {
        self.system_covariance_ref()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterSystemCovarianceMut<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: SystemCovarianceMatrix<STATES, T>,
{
    type SystemCovarianceMatrixMut = P;

    #[inline(always)]
    fn system_covariance_mut(&mut self) -> &mut Self::SystemCovarianceMatrixMut {
        self.system_covariance_mut()
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterPredict<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    X: StateVector<STATES, T>,
    A: SystemMatrix<STATES, T>,
    PX: StatePredictionVector<STATES, T>,
    P: SystemCovarianceMatrix<STATES, T>,
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
    P: SystemCovarianceMatrix<STATES, T>,
    X: StateVector<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn correct<const MEASUREMENTS: usize, M>(&mut self, measurement: &mut M)
    where
        M: KalmanFilterMeasurementCorrectFilter<STATES, T>
            + KalmanFilterNumMeasurements<MEASUREMENTS>,
    {
        self.correct(measurement)
    }
}

impl<const STATES: usize, T, A, X, P, PX, TempP> KalmanFilterApplyInput<STATES, T>
    for Kalman<STATES, T, A, X, P, PX, TempP>
where
    P: SystemCovarianceMatrix<STATES, T>,
    X: StateVector<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn input<const INPUTS: usize, I>(&mut self, input: &mut I)
    where
        I: KalmanFilterInputApplyToFilter<STATES, T> + KalmanFilterNumInputs<INPUTS>,
    {
        self.input(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dummies::{Dummy, DummyMatrix};

    #[test]
    fn builder_simple() {
        let _filter = KalmanBuilder::new::<3, f32>(
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
        );
    }

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
}
