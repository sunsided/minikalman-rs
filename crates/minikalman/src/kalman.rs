use crate::measurement::Measurement;
use crate::{MatrixData, MatrixDataType};

/// Kalman Filter structure.
#[allow(non_snake_case, unused)]
pub struct Kalman<'a, const STATES: usize, const INPUTS: usize, T = f32> {
    /// State vector.
    x: MatrixData<'a, STATES, 1, T>,
    /// System matrix.
    ///
    /// See also [`P`].
    A: MatrixData<'a, STATES, STATES, T>,
    /// System covariance matrix.
    ///
    /// See also [`A`].
    P: MatrixData<'a, STATES, STATES, T>,
    /// Input vector.
    u: MatrixData<'a, INPUTS, 1, T>,
    /// Input matrix.
    ///
    /// See also [`Q`].
    B: MatrixData<'a, STATES, INPUTS, T>,
    /// Input covariance matrix.
    ///
    /// See also [`B`].
    Q: MatrixData<'a, INPUTS, INPUTS, T>,

    /// Temporary storage.
    temporary: KalmanTemporary<'a, STATES, INPUTS, T>,
}

#[allow(non_snake_case)]
struct KalmanTemporary<'a, const STATES: usize, const INPUTS: usize, T = f32> {
    /// x-sized temporary vector.
    predicted_x: MatrixData<'a, STATES, 1, T>,
    /// P-Sized temporary matrix (number of states × number of states).
    ///
    /// The backing field for this temporary MAY be aliased with temporary BQ.
    P: MatrixData<'a, STATES, STATES, T>,
    /// B×Q-sized temporary matrix (number of states × number of inputs).
    ///
    /// The backing field for this temporary MAY be aliased with temporary P.
    BQ: MatrixData<'a, STATES, INPUTS, T>,
}

impl<'a, const STATES: usize, const INPUTS: usize, T> Kalman<'a, STATES, INPUTS, T> {
    /// The number of states.
    const NUM_STATES: usize = STATES;

    /// The number of inputs.
    const NUM_INPUTS: usize = INPUTS;

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
    /// let mut gravity_x = create_buffer_x!(NUM_STATES);
    /// let mut gravity_A = create_buffer_A!(NUM_STATES);
    /// let mut gravity_P = create_buffer_P!(NUM_STATES);
    ///
    /// // Input buffers.
    /// let mut gravity_u = create_buffer_u!(0);
    /// let mut gravity_B = create_buffer_B!(0, 0);
    /// let mut gravity_Q = create_buffer_Q!(0);
    ///
    /// // Filter temporaries.
    /// let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    /// let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    /// let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
    ///
    /// let mut filter = Kalman::<NUM_STATES, NUM_INPUTS>::new_direct(
    ///     &mut gravity_A,
    ///     &mut gravity_x,
    ///     &mut gravity_B,
    ///     &mut gravity_u,
    ///     &mut gravity_P,
    ///     &mut gravity_Q,
    ///     &mut gravity_temp_x,
    ///     &mut gravity_temp_P,
    ///     &mut gravity_temp_BQ,
    ///  );
    /// ```
    ///
    /// See also [`Measurement::new_direct`] for setting up measurement buffers.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc(alias = "kalman_filter_initialize")]
    pub fn new_direct(
        A: &'a mut [T],
        x: &'a mut [T],
        B: &'a mut [T],
        u: &'a mut [T],
        P: &'a mut [T],
        Q: &'a mut [T],
        predictedX: &'a mut [T],
        temp_P: &'a mut [T],
        temp_BQ: &'a mut [T],
    ) -> Self {
        Self::new(
            MatrixData::<STATES, STATES, T>::new(A),
            MatrixData::<STATES, 1, T>::new(x),
            MatrixData::<STATES, INPUTS, T>::new(B),
            MatrixData::<INPUTS, 1, T>::new(u),
            MatrixData::<STATES, STATES, T>::new(P),
            MatrixData::<INPUTS, INPUTS, T>::new(Q),
            MatrixData::<STATES, 1, T>::new(predictedX),
            MatrixData::<STATES, STATES, T>::new(temp_P),
            MatrixData::<STATES, INPUTS, T>::new(temp_BQ),
        )
    }

    /// See [`Kalman::new_direct`] instead.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[deprecated(since = "0.2.2")]
    pub fn new_from_buffers(
        A: &'a mut [T],
        x: &'a mut [T],
        B: &'a mut [T],
        u: &'a mut [T],
        P: &'a mut [T],
        Q: &'a mut [T],
        predictedX: &'a mut [T],
        temp_P: &'a mut [T],
        temp_BQ: &'a mut [T],
    ) -> Self {
        Self::new_direct(A, x, B, u, P, Q, predictedX, temp_P, temp_BQ)
    }

    /// Initializes a Kalman filter instance.
    ///
    /// ## Arguments
    /// * `num_states` - The number of states tracked by this filter.
    /// * `num_inputs` - The number of inputs available to the filter.
    /// * `A` - The state transition matrix (`num_states` × `num_states`).
    /// * `x` - The state vector (`num_states` × `1`).
    /// * `B` - The input transition matrix (`num_states` × `num_inputs`).
    /// * `u` - The input vector (`num_inputs` × `1`).
    /// * `P` - The state covariance matrix (`num_states` × `num_states`).
    /// * `Q` - The input covariance matrix (`num_inputs` × `num_inputs`).
    /// * `predictedX` - The temporary vector for predicted states (`num_states` × `1`).
    /// * `temp_P` - The temporary vector for P calculation (`num_states` × `num_states`).
    /// * `temp_BQ` - The temporary vector for B×Q calculation (`num_states` × `num_inputs`).
    ///
    /// ## Example
    ///
    /// See [`Kalman::new_direct`] for an example.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub fn new(
        A: MatrixData<'a, STATES, STATES, T>,
        x: MatrixData<'a, STATES, 1, T>,
        B: MatrixData<'a, STATES, INPUTS, T>,
        u: MatrixData<'a, INPUTS, 1, T>,
        P: MatrixData<'a, STATES, STATES, T>,
        Q: MatrixData<'a, INPUTS, INPUTS, T>,
        predictedX: MatrixData<'a, STATES, 1, T>,
        temp_P: MatrixData<'a, STATES, STATES, T>,
        temp_BQ: MatrixData<'a, STATES, INPUTS, T>,
    ) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            use crate::FastUInt8;
            debug_assert_eq!(
                A.rows(),
                STATES as FastUInt8,
                "The state transition matrix A requires {} rows and {} columns (i.e. states × states)",
                STATES,
                STATES
            );
            debug_assert_eq!(
                A.cols(),
                STATES as FastUInt8,
                "The state transition matrix A requires {} rows and {} columns (i.e. states × states)",
                STATES,
                STATES
            );

            debug_assert_eq!(
                P.rows(),
                STATES as FastUInt8,
                "The system covariance matrix P requires {} rows and {} columns (i.e. states × states)",
                STATES,
                STATES
            );
            debug_assert_eq!(
                P.cols(),
                STATES as FastUInt8,
                "The system covariance matrix P requires {} rows and {} columns (i.e. states × states)",
                STATES,
                STATES
            );

            debug_assert_eq!(
                x.rows(),
                STATES as FastUInt8,
                "The state vector x requires {} rows and 1 column (i.e. states × 1)",
                STATES
            );
            debug_assert_eq!(
                x.cols(),
                1,
                "The state vector x requires {} rows and 1 column (i.e. states × 1)",
                STATES
            );

            debug_assert_eq!(
                B.rows(),
                STATES as FastUInt8,
                "The input transition matrix B requires {} rows and {} columns (i.e. states × inputs)",
                STATES,
                INPUTS
            );
            debug_assert_eq!(
                B.cols(),
                INPUTS as FastUInt8,
                "The input transition matrix B requires {} rows and {} columns (i.e. states × inputs)",
                STATES,
                INPUTS
            );

            debug_assert_eq!(
                Q.rows(),
                INPUTS as FastUInt8,
                "The input covariance matrix Q requires {} rows and {} columns (i.e. inputs × inputs)",
                INPUTS,
                INPUTS
            );
            debug_assert_eq!(
                Q.cols(),
                INPUTS as FastUInt8,
                "The input covariance matrix Q requires {} rows and {} columns (i.e. inputs × inputs)",
                INPUTS,
                INPUTS
            );

            debug_assert_eq!(
                u.rows(),
                INPUTS as FastUInt8,
                "The input vector u requires {} rows and 1 column (i.e. inputs × 1)",
                INPUTS
            );
            debug_assert_eq!(
                u.cols(),
                1,
                "The input vector u requires {} rows and 1 column (i.e. inputs × 1)",
                INPUTS
            );

            debug_assert_eq!(
                predictedX.rows(),
                STATES as FastUInt8,
                "The temporary state prediction vector requires {} rows and 1 column (i.e. states × 1)",
                STATES
            );
            debug_assert_eq!(
                predictedX.cols(),
                1,
                "The temporary state prediction vector requires {} rows and 1 column (i.e. states × 1)",
                STATES
            );

            debug_assert_eq!(
                temp_P.rows(), STATES as FastUInt8,
                "The temporary system covariance matrix requires {} rows and {} columns (i.e. states × states)",
                STATES, STATES
            );
            debug_assert_eq!(
                temp_P.cols(), STATES as FastUInt8,
                "The temporary system covariance matrix requires {} rows and {} columns (i.e. states × states)",
                STATES, STATES
            );

            debug_assert_eq!(
                temp_BQ.rows(),
                STATES as FastUInt8,
                "The temporary B×Q matrix requires {} rows and {} columns (i.e. states × inputs)",
                STATES,
                INPUTS
            );
            debug_assert_eq!(
                temp_BQ.cols(),
                INPUTS as FastUInt8,
                "The temporary B×Q matrix requires {} rows and {} columns (i.e. states × inputs)",
                STATES,
                INPUTS
            );
        }

        Self {
            A,
            P,
            x,
            B,
            Q,
            u,
            temporary: KalmanTemporary {
                predicted_x: predictedX,
                P: temp_P,
                BQ: temp_BQ,
            },
        }
    }

    /// Returns the number of states.
    pub const fn states(&self) -> usize {
        Self::NUM_STATES
    }

    /// Returns the number of inputs.
    pub const fn inputs(&self) -> usize {
        Self::NUM_INPUTS
    }

    /// Gets a reference to the state vector x.
    #[inline(always)]
    pub fn state_vector_ref(&self) -> &MatrixData<'_, STATES, 1, T> {
        &self.x
    }

    /// Gets a reference to the state vector x.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_vector")]
    pub fn state_vector_mut<'b: 'a>(&'b mut self) -> &'b mut MatrixData<'a, STATES, 1, T> {
        &mut self.x
    }

    /// Applies a function to the state vector x.
    #[inline(always)]
    pub fn state_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MatrixData<'a, STATES, 1, T>),
    {
        f(&mut self.x)
    }

    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_ref(&self) -> &MatrixData<'_, STATES, STATES, T> {
        &self.A
    }

    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_transition")]
    pub fn state_transition_mut(&'a mut self) -> &mut MatrixData<'_, STATES, STATES, T> {
        &mut self.A
    }

    /// Applies a function to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MatrixData<'a, STATES, STATES, T>),
    {
        f(&mut self.A)
    }

    /// Gets a reference to the system covariance matrix P.
    #[inline(always)]
    pub fn system_covariance_ref(&self) -> &MatrixData<'_, STATES, STATES, T> {
        &self.P
    }

    /// Gets a mutable reference to the system covariance matrix P.
    #[inline(always)]
    #[doc(alias = "kalman_get_system_covariance")]
    pub fn system_covariance_mut(&'a mut self) -> &'a mut MatrixData<'_, STATES, STATES, T> {
        &mut self.P
    }

    /// Applies a function to the system covariance matrix P.
    #[inline(always)]
    pub fn system_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MatrixData<'a, STATES, STATES, T>),
    {
        f(&mut self.P)
    }

    /// Gets a reference to the input vector u.
    #[inline(always)]
    pub fn input_vector_ref(&self) -> &MatrixData<'_, INPUTS, 1, T> {
        &self.u
    }

    /// Gets a mutable reference to the input vector u.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_vector")]
    pub fn input_vector_mut(&'a mut self) -> &'a mut MatrixData<'_, INPUTS, 1, T> {
        &mut self.u
    }

    /// Applies a function to the input vector u.
    #[inline(always)]
    pub fn input_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MatrixData<'a, INPUTS, 1, T>),
    {
        f(&mut self.u)
    }

    /// Gets a reference to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_ref(&self) -> &MatrixData<'a, STATES, INPUTS, T> {
        &self.B
    }

    /// Gets a mutable reference to the input transition matrix B.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_transition")]
    pub fn input_transition_mut(&'a mut self) -> &'a mut MatrixData<'_, STATES, INPUTS, T> {
        &mut self.B
    }

    /// Applies a function to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MatrixData<'a, STATES, INPUTS, T>),
    {
        f(&mut self.B)
    }

    /// Gets a reference to the input covariance matrix Q.
    #[inline(always)]
    pub fn input_covariance_ref(&self) -> &MatrixData<'_, INPUTS, INPUTS, T> {
        &self.Q
    }

    /// Gets a mutable reference to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    pub fn input_covariance_mut(&'a mut self) -> &'a mut MatrixData<'_, INPUTS, INPUTS, T> {
        &mut self.Q
    }

    /// Applies a function to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    pub fn input_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut MatrixData<'a, INPUTS, INPUTS, T>),
    {
        f(&mut self.Q)
    }

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
    /// # let mut gravity_x = create_buffer_x!(NUM_STATES);
    /// # let mut gravity_A = create_buffer_A!(NUM_STATES);
    /// # let mut gravity_P = create_buffer_P!(NUM_STATES);
    /// #
    /// # // Input buffers.
    /// # let mut gravity_u = create_buffer_u!(0);
    /// # let mut gravity_B = create_buffer_B!(0, 0);
    /// # let mut gravity_Q = create_buffer_Q!(0);
    /// #
    /// # // Filter temporaries.
    /// # let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    /// # let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    /// # let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
    /// #
    /// # let mut filter = Kalman::<NUM_STATES, NUM_INPUTS>::new_direct(
    /// #     &mut gravity_A,
    /// #     &mut gravity_x,
    /// #     &mut gravity_B,
    /// #     &mut gravity_u,
    /// #     &mut gravity_P,
    /// #     &mut gravity_Q,
    /// #     &mut gravity_temp_x,
    /// #     &mut gravity_temp_P,
    /// #     &mut gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// # let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// # let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// # let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// # let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    /// #
    /// # // Measurement temporaries.
    /// # let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// # let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// # let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    /// #
    /// # let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS>::new_direct(
    /// #     &mut gravity_H,
    /// #     &mut gravity_z,
    /// #     &mut gravity_R,
    /// #     &mut gravity_y,
    /// #     &mut gravity_S,
    /// #     &mut gravity_K,
    /// #     &mut gravity_temp_S_inv,
    /// #     &mut gravity_temp_HP,
    /// #     &mut gravity_temp_PHt,
    /// #     &mut gravity_temp_KHP,
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
    /// # let mut gravity_x = create_buffer_x!(NUM_STATES);
    /// # let mut gravity_A = create_buffer_A!(NUM_STATES);
    /// # let mut gravity_P = create_buffer_P!(NUM_STATES);
    /// #
    /// # // Input buffers.
    /// # let mut gravity_u = create_buffer_u!(0);
    /// # let mut gravity_B = create_buffer_B!(0, 0);
    /// # let mut gravity_Q = create_buffer_Q!(0);
    /// #
    /// # // Filter temporaries.
    /// # let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    /// # let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    /// # let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
    /// #
    /// # let mut filter = Kalman::<NUM_STATES, NUM_INPUTS>::new_direct(
    /// #     &mut gravity_A,
    /// #     &mut gravity_x,
    /// #     &mut gravity_B,
    /// #     &mut gravity_u,
    /// #     &mut gravity_P,
    /// #     &mut gravity_Q,
    /// #     &mut gravity_temp_x,
    /// #     &mut gravity_temp_P,
    /// #     &mut gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// # let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// # let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// # let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// # let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    /// #
    /// # // Measurement temporaries.
    /// # let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// # let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// # let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    /// #
    /// # let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS>::new_direct(
    /// #     &mut gravity_H,
    /// #     &mut gravity_z,
    /// #     &mut gravity_R,
    /// #     &mut gravity_y,
    /// #     &mut gravity_S,
    /// #     &mut gravity_K,
    /// #     &mut gravity_temp_S_inv,
    /// #     &mut gravity_temp_HP,
    /// #     &mut gravity_temp_PHt,
    /// #     &mut gravity_temp_KHP,
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
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = &self.A;
        let x = &mut self.x;

        // temporaries
        let x_predicted = &mut self.temporary.predicted_x;

        //* Predict next state using system dynamics
        //* x = A*x

        A.mult_rowvector(x, x_predicted);
        x_predicted.copy(x);
    }

    /// Performs the time update / prediction step of only the state covariance matrix
    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_Q(&mut self)
    where
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = &self.A;
        let B = &self.B;
        let Q = &self.Q;
        let P = &mut self.P;

        // temporaries
        let P_temp = &mut self.temporary.P;
        let BQ_temp = &mut self.temporary.BQ;

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

    /// Performs the time update / prediction step of only the state covariance matrix
    ///
    /// ## Arguments
    /// * `lambda` - A tuning parameter.
    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_Q_tuned(&mut self, lambda: T)
    where
        T: MatrixDataType,
    {
        // matrices and vectors
        let A = &self.A;
        let B = &self.B;
        let Q = &self.Q;
        let P = &mut self.P;

        // temporaries
        let P_temp = &mut self.temporary.P;
        let BQ_temp = &mut self.temporary.BQ;

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
    /// # let mut gravity_x = create_buffer_x!(NUM_STATES);
    /// # let mut gravity_A = create_buffer_A!(NUM_STATES);
    /// # let mut gravity_P = create_buffer_P!(NUM_STATES);
    /// #
    /// # // Input buffers.
    /// # let mut gravity_u = create_buffer_u!(0);
    /// # let mut gravity_B = create_buffer_B!(0, 0);
    /// # let mut gravity_Q = create_buffer_Q!(0);
    /// #
    /// # // Filter temporaries.
    /// # let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    /// # let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    /// # let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
    /// #
    /// # let mut filter = Kalman::<NUM_STATES, NUM_INPUTS>::new_direct(
    /// #     &mut gravity_A,
    /// #     &mut gravity_x,
    /// #     &mut gravity_B,
    /// #     &mut gravity_u,
    /// #     &mut gravity_P,
    /// #     &mut gravity_Q,
    /// #     &mut gravity_temp_x,
    /// #     &mut gravity_temp_P,
    /// #     &mut gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// # let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// # let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// # let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// # let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    /// #
    /// # // Measurement temporaries.
    /// # let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// # let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// # let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    /// #
    /// # let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS>::new_direct(
    /// #     &mut gravity_H,
    /// #     &mut gravity_z,
    /// #     &mut gravity_R,
    /// #     &mut gravity_y,
    /// #     &mut gravity_S,
    /// #     &mut gravity_K,
    /// #     &mut gravity_temp_S_inv,
    /// #     &mut gravity_temp_HP,
    /// #     &mut gravity_temp_PHt,
    /// #     &mut gravity_temp_KHP,
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
    #[doc(alias = "kalman_predict_Q")]
    pub fn correct<const M: usize>(&mut self, kfm: &mut Measurement<'a, STATES, M, T>)
    where
        T: MatrixDataType,
    {
        // matrices and vectors
        let P = &mut self.P;
        let x = &mut self.x;

        let H = &kfm.H;
        let K = &mut kfm.K;
        let S = &mut kfm.S;
        let R = &mut kfm.R;
        let y = &mut kfm.y;
        let z = &kfm.z;

        // temporaries
        let S_inv = &mut kfm.temporary.S_inv;
        let temp_HP = &mut kfm.temporary.HP;
        let temp_KHP = &mut kfm.temporary.KHP;
        let temp_PHt = &mut kfm.temporary.PHt;

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
