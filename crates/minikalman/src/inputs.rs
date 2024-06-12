use core::marker::PhantomData;
use minikalman_traits::kalman::*;
use minikalman_traits::matrix::*;

/// A builder for a [`Input`] filter instances.
#[allow(clippy::type_complexity)]
pub struct InputBuilder<B, U, Q, TempBQ> {
    _phantom: (
        PhantomData<B>,
        PhantomData<U>,
        PhantomData<Q>,
        PhantomData<TempBQ>,
    ),
}

impl<B, U, Q, TempBQ> InputBuilder<B, U, Q, TempBQ> {
    /// Initializes a Kalman filter input instance.
    ///
    /// ## Arguments
    /// * `B` - The input transition matrix (`STATES` × `INPUTS`).
    /// * `u` - The input vector (`INPUTS` × `1`).
    /// * `Q` - The input covariance matrix (`INPUTS` × `INPUTS`).
    /// * `temp_BQ` - The temporary vector for B×Q calculation (`STATES` × `INPUTS`).
    #[allow(non_snake_case, clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub fn new<const STATES: usize, const INPUTS: usize, T>(
        B: B,
        u: U,
        Q: Q,
        temp_BQ: TempBQ,
    ) -> Input<STATES, INPUTS, T, B, U, Q, TempBQ>
    where
        T: MatrixDataType,
        B: InputMatrix<STATES, INPUTS, T>,
        U: InputVector<INPUTS, T>,
        Q: InputCovarianceMatrix<INPUTS, T>,
        TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
    {
        Input::<STATES, INPUTS, T, _, _, _, _> {
            B,
            u,
            Q,
            temp_BQ,
            _phantom: Default::default(),
        }
    }
}

/// Input Filter structure.  See [`InputBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Input<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ> {
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

    /// B×Q-sized temporary matrix (number of states × number of inputs).
    ///
    /// The backing field for this temporary MAY be aliased with temporary P.
    temp_BQ: TempBQ,

    _phantom: PhantomData<T>,
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
{
    /// Returns the number of states.
    #[allow(unused)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Returns the number of inputs.
    #[allow(unused)]
    pub const fn inputs(&self) -> usize {
        INPUTS
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    U: InputVector<INPUTS, T>,
{
    /// Gets a reference to the input vector u.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_vector")]
    pub fn input_vector_ref(&self) -> &U {
        &self.u
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    U: InputVectorMut<INPUTS, T>,
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

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    B: InputMatrix<STATES, INPUTS, T>,
{
    /// Gets a reference to the input transition matrix B.
    #[inline(always)]
    pub fn input_transition_ref(&self) -> &B {
        &self.B
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
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

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    Q: InputCovarianceMatrix<INPUTS, T>,
{
    /// Gets a reference to the input covariance matrix Q.
    #[inline(always)]
    pub fn input_covariance_ref(&self) -> &Q {
        &self.Q
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
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

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    U: InputVector<INPUTS, T>,
    B: InputMatrix<STATES, INPUTS, T>,
    Q: InputCovarianceMatrix<INPUTS, T>,
    TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
    T: MatrixDataType,
{
    /// Applies a correction step to the provided state vector and covariance matrix.
    #[allow(non_snake_case)]
    pub fn apply_input<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVector<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
    {
        // matrices and vectors
        let P = P.as_matrix_mut();
        let x = x.as_matrix_mut();

        // matrices and vectors
        let u = self.u.as_matrix();
        let B = self.B.as_matrix();
        let Q = self.Q.as_matrix();

        if u.is_empty() || B.is_empty() {
            return;
        }

        // temporaries
        let BQ_temp = self.temp_BQ.as_matrix_mut();

        // Incorporate input with state
        // x = x + B*u
        B.multadd_rowvector(u, x);

        // P = P + B*Q*Bᵀ
        B.mult(Q, BQ_temp); // temp = B*Q
        BQ_temp.multadd_transb(B, P); // P += temp*Bᵀ
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ> KalmanFilterNumStates<STATES>
    for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
{
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ> KalmanFilterNumInputs<STATES>
    for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
{
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputVector<INPUTS, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    U: InputVector<INPUTS, T>,
{
    type InputVector = U;

    fn input_vector_ref(&self) -> &Self::InputVector {
        self.input_vector_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputVectorMut<INPUTS, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    U: InputVectorMut<INPUTS, T>,
{
    type InputVectorMut = U;

    fn input_vector_mut(&mut self) -> &mut Self::InputVectorMut {
        self.input_vector_mut()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputTransition<STATES, INPUTS, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    B: InputMatrix<STATES, INPUTS, T>,
{
    type InputTransitionMatrix = B;

    fn input_transition_ref(&self) -> &Self::InputTransitionMatrix {
        self.input_transition_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputTransitionMut<STATES, INPUTS, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    B: InputMatrixMut<STATES, INPUTS, T>,
{
    type InputTransitionMatrixMut = B;

    fn input_transition_mut(&mut self) -> &mut Self::InputTransitionMatrixMut {
        self.input_transition_mut()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputCovariance<INPUTS, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    Q: InputCovarianceMatrix<INPUTS, T>,
{
    type InputCovarianceMatrix = Q;

    fn input_covariance_ref(&self) -> &Self::InputCovarianceMatrix {
        self.input_covariance_ref()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputCovarianceMut<INPUTS, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    Q: InputCovarianceMatrixMut<INPUTS, T>,
{
    type InputCovarianceMatrixMut = Q;

    fn input_covariance_mut(&mut self) -> &mut Self::InputCovarianceMatrixMut {
        self.input_covariance_mut()
    }
}

impl<const STATES: usize, const INPUTS: usize, T, B, U, Q, TempBQ>
    KalmanFilterInputApplyToFilter<STATES, T> for Input<STATES, INPUTS, T, B, U, Q, TempBQ>
where
    U: InputVector<INPUTS, T>,
    B: InputMatrix<STATES, INPUTS, T>,
    Q: InputCovarianceMatrix<INPUTS, T>,
    TempBQ: TemporaryBQMatrix<STATES, INPUTS, T>,
    T: MatrixDataType,
{
    #[allow(non_snake_case)]
    fn apply_to<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVector<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
    {
        self.apply_input(x, P)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dummies::{Dummy, DummyMatrix};

    #[allow(non_snake_case)]
    #[test]
    #[cfg(feature = "alloc")]
    fn input_only() {
        use assert_float_eq::*;
        use minikalman_traits::matrix::MatrixMut;

        use crate::prelude::{BufferBuilder, KalmanBuilder};

        const NUM_STATES: usize = 4;
        const NUM_INPUTS: usize = 3;

        // System buffers.
        let x = BufferBuilder::state_vector_x::<NUM_STATES>().new(0.0_f32);
        let A = BufferBuilder::system_state_transition_A::<NUM_STATES>().new(0.0_f32);
        let P = BufferBuilder::system_covariance_P::<NUM_STATES>().new(0.0_f32);

        // Input buffers.
        let u = BufferBuilder::input_vector_u::<NUM_INPUTS>().new(0.0_f32);
        let B = BufferBuilder::input_transition_B::<NUM_STATES, NUM_INPUTS>().new(0.0_f32);
        let Q = BufferBuilder::input_covariance_Q::<NUM_INPUTS>().new(0.0_f32);

        // Filter temporaries.
        let temp_x = BufferBuilder::state_prediction_temp_x::<NUM_STATES>().new(0.0_f32);
        let temp_P = BufferBuilder::temp_system_covariance_P::<NUM_STATES>().new(0.0_f32);

        // Input temporaries
        let temp_BQ = BufferBuilder::temp_BQ::<NUM_STATES, NUM_INPUTS>().new(0.0_f32);

        let mut filter = KalmanBuilder::new::<NUM_STATES, f32>(A, x, P, temp_x, temp_P);
        let mut input = InputBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(B, u, Q, temp_BQ);

        // State transition is identity.
        filter.state_transition_apply(|mat| {
            mat[0 * NUM_STATES] = 1.0;
            mat[0 * NUM_STATES + 1] = 1.0;
            mat[0 * NUM_STATES + 2] = 1.0;
            mat[0 * NUM_STATES + 3] = 1.0;

            mat[NUM_STATES + 1] = 1.0;
            mat[2 * NUM_STATES + 2] = 1.0;
            mat[3 * NUM_STATES + 3] = 1.0;
        });

        // State covariance is identity.
        filter.system_covariance_apply(|mat| {
            mat[0 * NUM_STATES] = 1.0;
            mat[NUM_STATES + 1] = 1.0;
            mat[2 * NUM_STATES + 2] = 1.0;
            mat[3 * NUM_STATES + 3] = 1.0;
        });

        // Input applies linearly to state.
        input.input_transition_apply(|mat| {
            mat[NUM_INPUTS] = 1.0;
            mat[2 * NUM_INPUTS + 1] = 1.0;
            mat[3 * NUM_INPUTS + 2] = 1.0;
        });

        // Input covariance is identity.
        input.input_covariance_apply(|mat| {
            mat[0 * NUM_INPUTS] = 1.0;
            mat[NUM_INPUTS + 1] = 1.0;
            mat[2 * NUM_INPUTS + 2] = 1.0;
        });

        // Define some test input vector.
        input.input_vector_apply(|vec| {
            vec.set(0, 0, 0.1);
            vec.set(1, 0, 1.0);
            vec.set(2, 0, 10.0);
        });

        // Sanity checks.
        assert_eq!(filter.states(), 4);
        assert_eq!(input.states(), 4);
        assert_eq!(input.inputs(), 3);

        // First round, state vector is empty.
        let state = filter.state_vector_ref().as_ref();
        assert_f32_near!(state[0], 0.0);
        assert_f32_near!(state[1], 0.0);
        assert_f32_near!(state[2], 0.0);
        assert_f32_near!(state[3], 0.0);

        // Predict one step - no inputs, so no changes.
        filter.predict();
        let state = filter.state_vector_ref().as_ref();
        assert_f32_near!(state[0], 0.0);
        assert_f32_near!(state[1], 0.0);
        assert_f32_near!(state[2], 0.0);
        assert_f32_near!(state[3], 0.0);

        // Predict one step (with inputs).
        filter.predict();
        filter.input(&mut input);
        let state = filter.state_vector_ref().as_ref();
        assert_f32_near!(state[0], 0.0);
        assert_f32_near!(state[1], 0.1);
        assert_f32_near!(state[2], 1.0);
        assert_f32_near!(state[3], 10.0);

        // Predict another step (with inputs).
        filter.predict();
        filter.input(&mut input);
        let state = filter.state_vector_ref().as_ref();
        assert_f32_near!(state[0], 11.1);
        assert_f32_near!(state[1], 0.2);
        assert_f32_near!(state[2], 2.0);
        assert_f32_near!(state[3], 20.0);
    }

    #[test]
    fn builder_simple() {
        let _filter = InputBuilder::new::<3, 2, f32>(
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
        );
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
}
