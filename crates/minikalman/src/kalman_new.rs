use crate::measurement_new::Measurement;
use crate::more_matrix_traits::{Matrix, MatrixMut, SquareMatrix};
use crate::more_traits::{
    InnovationVector, InputCovarianceMatrix, InputCovarianceMatrixMut, InputMatrix, InputMatrixMut,
    InputVector, InputVectorMut, KalmanGainMatrix, MeasurementTransformationMatrix,
    MeasurementVector, ProcessNoiseCovarianceMatrix, ResidualCovarianceMatrix,
    StatePredictionVector, StateVector, SystemCovarianceMatrix, SystemMatrix, SystemMatrixMut,
    TemporaryBQMatrix, TemporaryHPMatrix, TemporaryKHPMatrix, TemporaryPHTMatrix,
    TemporaryResidualCovarianceInvertedMatrix, TemporaryStateMatrix,
};
use crate::{FastUInt8, MatrixDataType};
use core::marker::PhantomData;

/// A builder for a Kalman filter.
pub struct KalmanBuilder<X, A, P, U, B, Q, PX, TempP, TempBQ> {
    _phantom: (
        PhantomData<X>,
        PhantomData<A>,
        PhantomData<P>,
        PhantomData<U>,
        PhantomData<B>,
        PhantomData<Q>,
        PhantomData<PX>,
        PhantomData<TempP>,
        PhantomData<TempBQ>,
    ),
}

impl<X, A, P, U, B, Q, PX, TempP, TempBQ> KalmanBuilder<X, A, P, U, B, Q, PX, TempP, TempBQ> {
    #[allow(non_snake_case, unused)]
    pub fn new<const STATES: usize, const INPUTS: usize, T>(
        x: X,
        A: A,
        P: P,
        u: U,
        B: B,
        Q: Q,
        predicted_x: PX,
        temp_P: TempP,
        temp_BQ: TempBQ,
    ) -> Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    where
        T: MatrixDataType,
        X: StateVector<STATES, T>,
        A: SystemMatrix<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
        U: InputVector<INPUTS, T>,
        B: InputMatrix<STATES, INPUTS, T>,
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
            _phantom: PhantomData::default(),
        }
    }
}

/// Kalman Filter structure.
#[allow(non_snake_case, unused)]
pub struct Kalman<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
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

impl<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
{
    /// Returns the number of states.
    pub const fn states(&self) -> FastUInt8 {
        STATES as _
    }

    /// Returns the number of inputs.
    pub const fn inputs(&self) -> FastUInt8 {
        INPUTS as _
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

impl<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
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

impl<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
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

impl<const STATES: usize, const INPUTS: usize, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
    Kalman<STATES, INPUTS, T, X, A, P, U, B, Q, PX, TempP, TempBQ>
{
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
        let A = &self.A.as_matrix();
        let x = &mut self.x.as_matrix_mut();

        // temporaries
        let x_predicted = &mut self.predicted_x.as_matrix_mut();

        //* Predict next state using system dynamics
        //* x = A*x

        A.mult_rowvector(x, x_predicted);
        x_predicted.copy(x);
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
        let A = &self.A.as_matrix();
        let B = &self.B.as_matrix();
        let Q = &self.Q.as_matrix();
        let P = &mut self.P.as_matrix_mut();

        // temporaries
        let P_temp = &mut self.temp_P.as_matrix_mut();
        let BQ_temp = &mut self.temp_BQ.as_matrix_mut();

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
        let A = &self.A.as_matrix();
        let B = &self.B.as_matrix();
        let Q = &self.Q.as_matrix();
        let P = &mut self.P.as_matrix_mut();

        // temporaries
        let P_temp = &mut self.temp_P.as_matrix_mut();
        let BQ_temp = &mut self.temp_BQ.as_matrix_mut();

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

    #[allow(non_snake_case)]
    fn correct<const MEASUREMENTS: usize, Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt>(
        &mut self,
        kfm: &mut Measurement<
            STATES,
            MEASUREMENTS,
            T,
            Z,
            H,
            R,
            Y,
            S,
            K,
            TempSInv,
            TempHP,
            TempKHP,
            TempPHt,
        >,
    ) where
        P: SystemCovarianceMatrix<STATES, T>,
        X: StateVector<STATES, T>,
        H: MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>,
        K: KalmanGainMatrix<STATES, MEASUREMENTS, T>,
        S: ResidualCovarianceMatrix<MEASUREMENTS, T>,
        R: ProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
        Y: InnovationVector<MEASUREMENTS, T>,
        Z: MeasurementVector<MEASUREMENTS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>,
        TempHP: TemporaryHPMatrix<MEASUREMENTS, STATES, T>,
        TempKHP: TemporaryKHPMatrix<STATES, T>,
        TempPHt: TemporaryPHTMatrix<STATES, MEASUREMENTS, T>,
        T: MatrixDataType,
    {
        // matrices and vectors
        let P = &mut self.P.as_matrix_mut();
        let x = &mut self.x.as_matrix_mut();

        let H = &kfm.H.as_matrix();
        let K = &mut kfm.K.as_matrix_mut();
        let S = &mut kfm.S.as_matrix_mut();
        let R = &mut kfm.R.as_matrix_mut();
        let y = &mut kfm.y.as_matrix_mut();
        let z = &kfm.z.as_matrix();

        // temporaries
        let S_inv = &mut kfm.temp_S_inv.as_matrix_mut();
        let temp_HP = &mut kfm.temp_HP.as_matrix_mut();
        let temp_KHP = &mut kfm.temp_KHP.as_matrix_mut();
        let temp_PHt = &mut kfm.temp_PHt.as_matrix_mut();

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
    use super::*;
    use crate::more_matrix_traits::MatrixMut;

    #[test]
    fn builder_simple() {
        let filter = KalmanBuilder::new::<3, 0, f32>(
            Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy,
        );
    }

    struct Dummy;
    struct DummyMatrix;

    impl<const STATES: usize, T> StateVector<STATES, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<STATES, 1, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, 1, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, T> SystemMatrix<STATES, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<STATES, STATES, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, T> SystemMatrixMut<STATES, T> for Dummy {
        fn as_matrix_mut(&self) -> impl MatrixMut<STATES, STATES, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, T> SystemCovarianceMatrix<STATES, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<STATES, STATES, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&self) -> impl MatrixMut<STATES, STATES, T> {
            DummyMatrix
        }
    }
    impl<const INPUTS: usize, T> InputVector<INPUTS, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<INPUTS, 1, T> {
            DummyMatrix
        }
    }
    impl<const INPUTS: usize, T> InputVectorMut<INPUTS, T> for Dummy {
        fn as_matrix_mut(&self) -> impl MatrixMut<INPUTS, 1, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, const INPUTS: usize, T> InputMatrix<STATES, INPUTS, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<STATES, INPUTS, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, const INPUTS: usize, T> InputMatrixMut<STATES, INPUTS, T> for Dummy {
        fn as_matrix_mut(&self) -> impl MatrixMut<STATES, INPUTS, T> {
            DummyMatrix
        }
    }
    impl<const INPUTS: usize, T> InputCovarianceMatrix<INPUTS, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<INPUTS, INPUTS, T> {
            DummyMatrix
        }
    }
    impl<const INPUTS: usize, T> InputCovarianceMatrixMut<INPUTS, T> for Dummy {
        fn as_matrix_mut(&self) -> impl MatrixMut<INPUTS, INPUTS, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, T> StatePredictionVector<STATES, T> for Dummy {
        fn as_matrix(&mut self) -> impl Matrix<STATES, 1, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, 1, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, T> TemporaryStateMatrix<STATES, T> for Dummy {
        fn as_matrix(&mut self) -> impl Matrix<STATES, STATES, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, STATES, T> {
            DummyMatrix
        }
    }
    impl<const STATES: usize, const INPUTS: usize, T> TemporaryBQMatrix<STATES, INPUTS, T> for Dummy {
        fn as_matrix(&mut self) -> impl Matrix<STATES, INPUTS, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, INPUTS, T> {
            DummyMatrix
        }
    }

    impl<T> AsRef<[T]> for DummyMatrix {
        fn as_ref(&self) -> &[T] {
            todo!()
        }
    }

    impl<T> AsMut<[T]> for DummyMatrix {
        fn as_mut(&mut self) -> &mut [T] {
            todo!()
        }
    }

    impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix {}
    impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T> for DummyMatrix {}
}
