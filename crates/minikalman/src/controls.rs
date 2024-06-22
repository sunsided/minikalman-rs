//! # Control inputs for Regular Kalman Filters.

use crate::kalman::*;
use crate::matrix::*;
use core::marker::PhantomData;

/// A builder for a [`Control`] filter instances.
#[allow(clippy::type_complexity)]
pub struct ControlBuilder<B, U, Q, TempBQ> {
    _phantom: (
        PhantomData<B>,
        PhantomData<U>,
        PhantomData<Q>,
        PhantomData<TempBQ>,
    ),
}

impl<B, U, Q, TempBQ> ControlBuilder<B, U, Q, TempBQ> {
    /// Initializes a Kalman filter control instance.
    ///
    /// ## Arguments
    /// * `B` - The control transition matrix (`STATES` × `CONTROLS`).
    /// * `u` - The control vector (`CONTROLS` × `1`).
    /// * `Q` - The control covariance matrix (`CONTROLS` × `CONTROLS`).
    /// * `temp_BQ` - The temporary vector for B×Q calculation (`STATES` × `CONTROLS`).
    #[allow(non_snake_case, clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub fn new<const STATES: usize, const CONTROLS: usize, T>(
        B: B,
        u: U,
        Q: Q,
        temp_BQ: TempBQ,
    ) -> Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
    where
        T: MatrixDataType,
        B: ControlMatrix<STATES, CONTROLS, T>,
        U: ControlVector<CONTROLS, T>,
        Q: ProcessNoiseCovarianceMatrix<CONTROLS, T>,
        TempBQ: TemporaryBQMatrix<STATES, CONTROLS, T>,
    {
        Control::<STATES, CONTROLS, T, _, _, _, _> {
            B,
            u,
            Q,
            temp_BQ,
            _phantom: Default::default(),
        }
    }
}

/// Control Filter structure.  See [`ControlBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Control<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ> {
    /// Control vector.
    u: U,

    /// Control matrix.
    ///
    /// See also [`Q`].
    B: B,

    /// Control covariance matrix.
    ///
    /// See also [`B`].
    Q: Q,

    /// B×Q-sized temporary matrix (number of states × number of controls).
    ///
    /// The backing field for this temporary MAY be aliased with temporary P.
    temp_BQ: TempBQ,

    _phantom: PhantomData<T>,
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
{
    /// Returns the number of states.
    #[allow(unused)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Returns the number of controls.
    #[allow(unused)]
    pub const fn controls(&self) -> usize {
        CONTROLS
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    U: ControlVector<CONTROLS, T>,
{
    /// Gets a reference to the control vector u.
    ///
    /// The control vector contains the external inputs to the system that can influence its state.
    /// These inputs might include forces, accelerations, or other actuations applied to the system.
    #[inline(always)]
    #[doc(alias = "kalman_get_control_vector")]
    pub fn control_vector(&self) -> &U {
        &self.u
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    U: ControlVectorMut<CONTROLS, T>,
{
    /// Gets a mutable reference to the control vector u.
    ///
    /// The control vector contains the external inputs to the system that can influence its state.
    /// These inputs might include forces, accelerations, or other actuations applied to the system.
    #[inline(always)]
    #[doc(alias = "kalman_get_control_vector")]
    pub fn control_vector_mut(&mut self) -> &mut U {
        &mut self.u
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    B: ControlMatrix<STATES, CONTROLS, T>,
{
    /// Gets a reference to the control transition matrix B.
    ///
    /// This matrix maps the control inputs to the state space, allowing the control vector to
    /// influence the state transition. It quantifies how the control inputs affect the state change.
    #[inline(always)]
    pub fn control_matrix(&self) -> &B {
        &self.B
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    B: ControlMatrixMut<STATES, CONTROLS, T>,
{
    /// Gets a mutable reference to the control transition matrix B.
    ///
    /// This matrix maps the control inputs to the state space, allowing the control vector to
    /// influence the state transition. It quantifies how the control inputs affect the state change.
    #[inline(always)]
    #[doc(alias = "kalman_get_control_matrix")]
    pub fn control_matrix_mut(&mut self) -> &mut B {
        &mut self.B
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    Q: ProcessNoiseCovarianceMatrix<CONTROLS, T>,
{
    /// Gets a reference to the control covariance matrix Q.
    ///
    /// This matrix represents the uncertainty in the state transition process, accounting for the
    /// randomness and inaccuracies in the model. It quantifies the expected variability in the
    /// state transition.
    #[inline(always)]
    #[doc(alias = "control_covariance")]
    pub fn process_noise_covariance(&self) -> &Q {
        &self.Q
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    Q: ProcessNoiseCovarianceMatrixMut<CONTROLS, T>,
{
    /// Gets a mutable reference to the control covariance matrix Q.
    ///
    /// This matrix represents the uncertainty in the state transition process, accounting for the
    /// randomness and inaccuracies in the model. It quantifies the expected variability in the
    /// state transition.
    #[inline(always)]
    #[doc(alias = "kalman_get_control_covariance")]
    #[doc(alias = "control_covariance_mut")]
    pub fn process_noise_covariance_mut(&mut self) -> &mut Q {
        &mut self.Q
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    U: ControlVector<CONTROLS, T>,
    B: ControlMatrix<STATES, CONTROLS, T>,
    Q: ProcessNoiseCovarianceMatrix<CONTROLS, T>,
    TempBQ: TemporaryBQMatrix<STATES, CONTROLS, T>,
    T: MatrixDataType,
{
    /// Applies a correction step to the provided state vector and covariance matrix.
    #[allow(non_snake_case)]
    pub fn apply_control<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
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

        // Incorporate control with state
        // x = x + B*u
        B.multadd_rowvector(u, x);

        // P = P + B*Q*Bᵀ
        B.mult(Q, BQ_temp); // temp = B*Q
        BQ_temp.multadd_transb(B, P); // P += temp*Bᵀ
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ> KalmanFilterNumStates<STATES>
    for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
{
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterNumControls<CONTROLS> for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
{
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterControlVector<CONTROLS, T> for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    U: ControlVector<CONTROLS, T>,
{
    type ControlVector = U;

    fn control_vector(&self) -> &Self::ControlVector {
        self.control_vector()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterControlVectorMut<CONTROLS, T> for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    U: ControlVectorMut<CONTROLS, T>,
{
    type ControlVectorMut = U;

    fn control_vector_mut(&mut self) -> &mut Self::ControlVectorMut {
        self.control_vector_mut()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterControlTransition<STATES, CONTROLS, T>
    for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    B: ControlMatrix<STATES, CONTROLS, T>,
{
    type ControlTransitionMatrix = B;

    fn control_matrix(&self) -> &Self::ControlTransitionMatrix {
        self.control_matrix()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterControlTransitionMut<STATES, CONTROLS, T>
    for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    B: ControlMatrixMut<STATES, CONTROLS, T>,
{
    type ControlTransitionMatrixMut = B;

    fn control_matrix_mut(&mut self) -> &mut Self::ControlTransitionMatrixMut {
        self.control_matrix_mut()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterProcessNoiseCovariance<CONTROLS, T>
    for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    Q: ProcessNoiseCovarianceMatrix<CONTROLS, T>,
{
    type ProcessNoiseCovarianceMatrix = Q;

    fn process_noise_covariance(&self) -> &Self::ProcessNoiseCovarianceMatrix {
        self.process_noise_covariance()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterControlCovarianceMut<CONTROLS, T> for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    Q: ProcessNoiseCovarianceMatrixMut<CONTROLS, T>,
{
    type ProcessNoiseCovarianceMatrixMut = Q;

    fn process_noise_covariance_mut(&mut self) -> &mut Self::ProcessNoiseCovarianceMatrixMut {
        self.process_noise_covariance_mut()
    }
}

impl<const STATES: usize, const CONTROLS: usize, T, B, U, Q, TempBQ>
    KalmanFilterControlApplyToFilter<STATES, T> for Control<STATES, CONTROLS, T, B, U, Q, TempBQ>
where
    U: ControlVector<CONTROLS, T>,
    B: ControlMatrix<STATES, CONTROLS, T>,
    Q: ProcessNoiseCovarianceMatrix<CONTROLS, T>,
    TempBQ: TemporaryBQMatrix<STATES, CONTROLS, T>,
    T: MatrixDataType,
{
    #[allow(non_snake_case)]
    fn apply_to<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.apply_control(x, P)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dummies::make_dummy_control;

    #[test]
    #[cfg(feature = "alloc")]
    fn test_mut_apply() {
        use crate::regular::builder::KalmanFilterBuilder;

        let builder = KalmanFilterBuilder::<3, f32>::default();
        let mut filter = builder.build();
        let mut control = builder.controls().build::<1>();

        filter.predict();
        filter.control(&mut control);
    }

    #[allow(non_snake_case)]
    #[test]
    #[cfg(feature = "alloc")]
    fn control_only() {
        use crate::matrix::MatrixMut;
        use assert_float_eq::*;

        use crate::prelude::{regular::RegularKalmanBuilder, BufferBuilder};

        const NUM_STATES: usize = 4;
        const NUM_CONTROLS: usize = 3;

        // System buffers.
        let x = BufferBuilder::state_vector_x::<NUM_STATES>().new();
        let A = BufferBuilder::system_matrix_A::<NUM_STATES>().new();
        let P = BufferBuilder::estimate_covariance_P::<NUM_STATES>().new();

        // Control buffers.
        let u = BufferBuilder::control_vector_u::<NUM_CONTROLS>().new();
        let B = BufferBuilder::control_matrix_B::<NUM_STATES, NUM_CONTROLS>().new();
        let Q = BufferBuilder::process_noise_covariance_Q::<NUM_CONTROLS>().new();

        // Filter temporaries.
        let temp_x = BufferBuilder::state_prediction_temp_x::<NUM_STATES>().new();
        let temp_P = BufferBuilder::temp_system_covariance_P::<NUM_STATES>().new();

        // Control temporaries
        let temp_BQ = BufferBuilder::temp_BQ::<NUM_STATES, NUM_CONTROLS>().new();

        let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(A, x, P, temp_x, temp_P);
        let mut control = ControlBuilder::new::<NUM_STATES, NUM_CONTROLS, f32>(B, u, Q, temp_BQ);

        // State transition is identity.
        filter.state_transition_mut().apply(|mat| {
            mat[0] = 1.0;
            mat[1] = 1.0;
            mat[2] = 1.0;
            mat[3] = 1.0;

            mat[NUM_STATES + 1] = 1.0;
            mat[2 * NUM_STATES + 2] = 1.0;
            mat[3 * NUM_STATES + 3] = 1.0;
        });

        // State covariance is identity.
        filter.estimate_covariance_mut().apply(|mat| {
            mat[0] = 1.0;
            mat[NUM_STATES + 1] = 1.0;
            mat[2 * NUM_STATES + 2] = 1.0;
            mat[3 * NUM_STATES + 3] = 1.0;
        });

        // Control applies linearly to state.
        control.control_matrix_mut().apply(|mat| {
            mat[NUM_CONTROLS] = 1.0;
            mat[2 * NUM_CONTROLS + 1] = 1.0;
            mat[3 * NUM_CONTROLS + 2] = 1.0;
        });

        // Control covariance is identity.
        control.process_noise_covariance_mut().apply(|mat| {
            mat[0] = 1.0;
            mat[NUM_CONTROLS + 1] = 1.0;
            mat[2 * NUM_CONTROLS + 2] = 1.0;
        });

        // Define some test control vector.
        control.control_vector_mut().apply(|vec| {
            vec.set_at(0, 0, 0.1);
            vec.set_at(1, 0, 1.0);
            vec.set_at(2, 0, 10.0);
        });

        // Sanity checks.
        assert_eq!(filter.states(), 4);
        assert_eq!(control.states(), 4);
        assert_eq!(control.controls(), 3);

        // First round, state vector is empty.
        let state = filter.state_vector().as_ref();
        assert_f32_near!(state[0], 0.0);
        assert_f32_near!(state[1], 0.0);
        assert_f32_near!(state[2], 0.0);
        assert_f32_near!(state[3], 0.0);

        // Predict one step - no controls, so no changes.
        filter.predict();
        let state = filter.state_vector().as_ref();
        assert_f32_near!(state[0], 0.0);
        assert_f32_near!(state[1], 0.0);
        assert_f32_near!(state[2], 0.0);
        assert_f32_near!(state[3], 0.0);

        // Predict one step (with controls).
        filter.predict();
        filter.control(&mut control);
        let state = filter.state_vector().as_ref();
        assert_f32_near!(state[0], 0.0);
        assert_f32_near!(state[1], 0.1);
        assert_f32_near!(state[2], 1.0);
        assert_f32_near!(state[3], 10.0);

        // Predict another step (with controls).
        filter.predict();
        filter.control(&mut control);
        let state = filter.state_vector().as_ref();
        assert_f32_near!(state[0], 11.1);
        assert_f32_near!(state[1], 0.2);
        assert_f32_near!(state[2], 2.0);
        assert_f32_near!(state[3], 20.0);
    }

    fn trait_impl<const STATES: usize, const CONTROLS: usize, T, M>(mut control: M) -> M
    where
        M: KalmanFilterControl<STATES, CONTROLS, T>
            + KalmanFilterControlTransitionMut<STATES, CONTROLS, T>,
    {
        assert_eq!(control.states(), STATES);
        assert_eq!(control.controls(), CONTROLS);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = control.control_vector();
        let _vec = control.control_vector_mut();
        let _ = control
            .control_vector()
            .as_matrix()
            .inspect(|_vec| test_fn());
        let _ = control
            .control_vector()
            .as_matrix()
            .inspect(|_vec| test_fn_mut());
        control
            .control_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn());
        control
            .control_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn_mut());

        let _mat = control.control_matrix();
        let _mat = control.control_matrix_mut();
        let _ = control
            .control_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = control
            .control_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        control
            .control_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        control
            .control_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        let _mat = control.process_noise_covariance();
        let _mat = control.process_noise_covariance_mut();
        let _ = control
            .process_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = control
            .process_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        control
            .process_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        control
            .process_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        control
    }

    #[test]
    fn builder_simple() {
        let control = make_dummy_control();

        let mut control = trait_impl(control);
        assert_eq!(control.states(), 3);
        assert_eq!(control.controls(), 2);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = control.control_vector();
        let _vec = control.control_vector_mut();
        let _ = control
            .control_vector()
            .as_matrix()
            .inspect(|_vec| test_fn());
        let _ = control
            .control_vector()
            .as_matrix()
            .inspect(|_vec| test_fn_mut());
        control
            .control_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn());
        control
            .control_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn_mut());

        let _mat = control.control_matrix();
        let _mat = control.control_matrix_mut();
        let _ = control
            .control_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = control
            .control_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        control
            .control_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        control
            .control_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        let _mat = control.process_noise_covariance();
        let _mat = control.process_noise_covariance_mut();
        let _ = control
            .process_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = control
            .process_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        control
            .process_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        control
            .process_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());
    }
}
