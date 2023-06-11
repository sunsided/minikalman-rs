use crate::measurement::Measurement;
use crate::{matrix_data_t, Matrix};
use stdint::uint_fast8_t;

/// Kalman Filter structure.
#[allow(non_snake_case, unused)]
pub struct Kalman<'a> {
    /// The number of states.
    num_states: uint_fast8_t,
    /// The number of inputs.
    num_inputs: uint_fast8_t,
    /// State vector.
    x: Matrix<'a>,
    /// System matrix.
    ///
    /// See also [`P`].
    A: Matrix<'a>,
    /// System covariance matrix.
    ///
    /// See also [`A`].
    P: Matrix<'a>,
    /// Input vector.
    u: Matrix<'a>,
    /// Input matrix.
    ///
    /// See also [`Q`].
    B: Matrix<'a>,
    /// Input covariance matrix.
    ///
    /// See also [`B`].
    Q: Matrix<'a>,

    /// Temporary storage.
    temporary: KalmanTemporary<'a>,
}

#[allow(non_snake_case)]
struct KalmanTemporary<'a> {
    /// x-sized temporary vector.
    predicted_x: Matrix<'a>,
    /// P-Sized temporary matrix (number of states × number of states).
    ///
    /// The backing field for this temporary MAY be aliased with temporary BQ.
    P: Matrix<'a>,
    /// B×Q-sized temporary matrix (number of states × number of inputs).
    ///
    /// The backing field for this temporary MAY be aliased with temporary P.
    BQ: Matrix<'a>,
}

impl<'a> Kalman<'a> {
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
    #[allow(non_snake_case)]
    #[doc(alias = "kalman_filter_initialize")]
    pub fn new(
        num_states: uint_fast8_t,
        num_inputs: uint_fast8_t,
        A: &'a mut [matrix_data_t],
        x: &'a mut [matrix_data_t],
        B: &'a mut [matrix_data_t],
        u: &'a mut [matrix_data_t],
        P: &'a mut [matrix_data_t],
        Q: &'a mut [matrix_data_t],
        predictedX: &'a mut [matrix_data_t],
        temp_P: &'a mut [matrix_data_t],
        temp_BQ: &'a mut [matrix_data_t],
    ) -> Self {
        Self {
            num_states,
            num_inputs,
            A: Matrix::new(num_states, num_states, A),
            P: Matrix::new(num_states, num_states, P),
            x: Matrix::new(num_states, 1, x),
            B: Matrix::new(num_states, num_inputs, B),
            Q: Matrix::new(num_inputs, num_inputs, Q),
            u: Matrix::new(num_inputs, 1, u),
            temporary: KalmanTemporary {
                predicted_x: Matrix::new(num_states, 1, predictedX),
                P: Matrix::new(num_states, num_states, temp_P),
                BQ: Matrix::new(num_states, num_inputs, temp_BQ),
            },
        }
    }

    /// Gets a reference to the state vector x.
    #[inline(always)]
    pub fn state_vector_ref(&self) -> &Matrix {
        &self.x
    }

    /// Gets a reference to the state vector x.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_vector")]
    pub fn state_vector_mut<'b: 'a>(&'b mut self) -> &'b mut Matrix<'a> {
        &mut self.x
    }

    /// Applies a function to the state vector x.
    #[inline(always)]
    pub fn state_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        f(&mut self.x)
    }

    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_ref(&self) -> &Matrix {
        &self.A
    }

    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    #[doc(alias = "kalman_get_state_transition")]
    pub fn state_transition_mut(&'a mut self) -> &mut Matrix {
        &mut self.A
    }

    /// Applies a function to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        f(&mut self.A)
    }

    /// Gets a reference to the system covariance matrix P.
    #[inline(always)]
    pub fn system_covariance_ref(&self) -> &Matrix {
        &self.P
    }

    /// Gets a mutable reference to the system covariance matrix P.
    #[inline(always)]
    #[doc(alias = "kalman_get_system_covariance")]
    pub fn system_covariance_mut(&'a mut self) -> &'a mut Matrix {
        &mut self.P
    }

    /// Applies a function to the system covariance matrix P.
    #[inline(always)]
    pub fn system_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        f(&mut self.P)
    }

    /// Gets a reference to the input vector u.
    #[inline(always)]
    pub fn input_vector_ref(&self) -> &Matrix {
        &self.u
    }

    /// Gets a mutable reference to the input vector u.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_vector")]
    pub fn input_vector_mut(&'a mut self) -> &'a mut Matrix {
        &mut self.u
    }

    /// Applies a function to the input vector u.
    #[inline(always)]
    pub fn input_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        f(&mut self.u)
    }

    /// Gets a reference to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_ref(&self) -> &Matrix {
        &self.B
    }

    /// Gets a mutable reference to the input transition matrix B.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_transition")]
    pub fn input_transition_mut(&'a mut self) -> &'a mut Matrix {
        &mut self.B
    }

    /// Applies a function to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        f(&mut self.B)
    }

    /// Gets a reference to the input covariance matrix Q.
    #[inline(always)]
    pub fn input_covariance_ref(&self) -> &Matrix {
        &self.Q
    }

    /// Gets a mutable reference to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    pub fn input_covariance_mut(&'a mut self) -> &'a mut Matrix {
        &mut self.Q
    }

    /// Applies a function to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
    pub fn input_covariance_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        f(&mut self.Q)
    }

    /// Performs the time update / prediction step.
    ///
    /// This call assumes that the input covariance and variables are already set in the filter structure.
    #[doc(alias = "kalman_predict")]
    pub fn predict(&mut self) {
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
    #[doc(alias = "kalman_predict_tuned")]
    pub fn predict_tuned(&mut self, lambda: matrix_data_t) {
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
    fn predict_x(&mut self) {
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
    fn predict_Q(&mut self) {
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
    fn predict_Q_tuned(&mut self, lambda: matrix_data_t) {
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
        let lambda = 1.0 / (lambda * lambda); // TODO: This should be precalculated, e.g. using set_lambda(...);

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
    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    pub fn correct(&mut self, kfm: &mut Measurement<'a>) {
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
