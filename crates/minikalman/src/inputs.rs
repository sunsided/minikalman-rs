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
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Returns the number of inputs.
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

#[cfg(test)]
mod tests {
    use core::ops::{Index, IndexMut};

    use minikalman_traits::matrix::{Matrix, MatrixMut};

    use super::*;

    #[test]
    fn builder_simple() {
        let _filter = InputBuilder::new::<3, 2, f32>(
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
