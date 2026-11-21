//! # Regular Kalman Filter

use crate::kalman::process_noise::{correct_adapting_process_noise, ProcessNoiseAdapter};
use crate::kalman::*;
use crate::matrix::MatrixDataType;
use crate::observations::NoiseAdaptation;
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
            process_noise_adaptation: None,
            _phantom: Default::default(),
        }
    }
}

/// Kalman Filter structure.  See [`RegularKalmanBuilder`] for construction.
///
#[doc = diagram!("regular-loop.mmd")]
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

    /// Online adaptation of `Q` from the state corrections; `None` (the default) disables it.
    ///
    /// See [`set_process_noise_adaptation`](Self::set_process_noise_adaptation).
    process_noise_adaptation: Option<ProcessNoiseAdapter<STATES, T, Q>>,

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
    #[doc = diagram!("regular-loop.mmd", "predict")]
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
    /// # impl_buffer_Q_direct!(mut gravity_Q, NUM_STATES, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_P,
    /// #     gravity_Q,
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
        Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        //* Predict next state using system dynamics
        //* x = A*x
        self.predict_x();

        //* Predict next covariance using system dynamics and control
        //* P = A*P*Aᵀ + Q
        self.predict_P();
    }

    /// Performs the time update / prediction step.
    ///
    #[doc = diagram!("regular-loop.mmd", "predict")]
    /// Unlike the plain [`predict`](Self::predict) shown above, this scales the propagated
    /// covariance by `1/λ²` before adding `Q`: `P ← A·P·Aᵀ/λ² + Q`.
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
    /// # impl_buffer_Q_direct!(mut gravity_Q, NUM_STATES, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_P,
    /// #     gravity_Q,
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
        Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        // Predict next state using system dynamics
        // x = A*x
        self.predict_x();

        // Predict next covariance using system dynamics and control
        // P = A*P*Aᵀ * 1/lambda^2 + Q
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
        x_predicted.copy_to(x);
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_P")]
    fn predict_P(&mut self)
    where
        A: StateTransitionMatrix<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        crate::kalman::predict_covariance(
            self.A.as_matrix(),
            self.P.as_matrix_mut(),
            self.Q.as_matrix(),
            self.temp_P.as_matrix_mut(),
            None,
        );
    }

    #[allow(non_snake_case)]
    #[doc(alias = "kalman_predict_Q")]
    fn predict_P_tuned(&mut self, lambda: T)
    where
        A: StateTransitionMatrix<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
        TempP: TemporaryStateMatrix<STATES, T>,
        T: MatrixDataType,
    {
        crate::kalman::predict_covariance(
            self.A.as_matrix(),
            self.P.as_matrix_mut(),
            self.Q.as_matrix(),
            self.temp_P.as_matrix_mut(),
            Some(lambda),
        );
    }

    /// Applies a control input.
    ///
    #[doc = diagram!("regular-loop.mmd", "control")]
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
    #[doc = diagram!("regular-correct.mmd")]
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
    /// # impl_buffer_Q_direct!(mut gravity_Q, NUM_STATES, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// #
    /// # let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_P,
    /// #     gravity_Q,
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
    pub fn correct<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
    {
        correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| measurement.correct(x, &mut self.P),
        )
    }

    /// Performs the measurement update step, normalizing the innovation before it is applied.
    ///
    #[doc = diagram!("regular-correct.mmd")]
    ///
    /// The innovation `y = z - H×x` is computed by plain subtraction. For measurements that
    /// wrap around, such as a compass heading, `fix` can wrap each entry back into range:
    ///
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// # use minikalman::prelude::*;
    /// # use minikalman::regular::builder::KalmanFilterBuilder;
    /// use minikalman::angles;
    ///
    /// # let builder = KalmanFilterBuilder::<1, f32>::default();
    /// # let mut filter = builder.build();
    /// # let mut measurement = builder.observations().build::<1>();
    /// # filter.state_transition_mut().make_identity();
    /// # measurement.observation_matrix_mut().make_identity();
    /// # measurement.measurement_noise_covariance_mut().make_identity();
    /// // The filter believes the heading is 359°; the compass reads 1°.
    /// filter.state_vector_mut().set_at(0, 0, 359.0);
    /// filter.estimate_covariance_mut().make_identity();
    /// measurement.measurement_vector_mut().set_at(0, 0, 1.0);
    ///
    /// filter.correct_with(&mut measurement, |y| y[0] = angles::wrap_180(y[0]));
    ///
    /// // The estimate moves about +1° across north, not 179° the long way round.
    /// let heading = filter.state_vector().get_at(0, 0);
    /// assert!(heading > 359.5 && heading <= 361.0);
    /// # }
    /// ```
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    /// * `fix` - Normalizes the innovation vector in place.
    pub fn correct_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
        N: FnMut(&mut [T]),
    {
        correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| measurement.correct_with(x, &mut self.P, fix),
        )
    }

    /// Performs the measurement update step with the externally supplied Kalman gain
    /// stored in the measurement, instead of computing it.
    ///
    #[doc = diagram!("regular-loop.mmd", "correct")]
    ///
    /// See [`KalmanFilterObservationCorrectWithGain`]: `P` is updated with the Joseph form,
    /// the gate is not evaluated and the NIS is not updated.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement, with its gain set via `kalman_gain_mut`.
    pub fn correct_with_gain<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectWithGain<STATES, T>,
    {
        measurement.correct_with_gain(&mut self.x, &mut self.P)
    }

    /// Performs the measurement update step with the externally supplied Kalman gain
    /// stored in the measurement, normalizing the innovation before it is applied.
    ///
    #[doc = diagram!("regular-loop.mmd", "correct")]
    ///
    /// See [`correct_with_gain`](Self::correct_with_gain).
    ///
    /// ## Arguments
    /// * `measurement` - The measurement, with its gain set via `kalman_gain_mut`.
    /// * `fix` - Normalizes the innovation vector `y = z - H×x` in place.
    pub fn correct_with_gain_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectWithGain<STATES, T>,
        N: FnMut(&mut [T]),
    {
        measurement.correct_with_gain_with(&mut self.x, &mut self.P, fix)
    }

    /// Performs a predictively oriented (PrO) measurement update step.
    ///
    #[doc = diagram!("regular-loop.mmd", "correct")]
    ///
    /// More conservative than [`correct`](Self::correct) under large innovations, which makes
    /// it robust to heavy-tailed noise and outliers at a small cost under Gaussian noise. See
    /// [`RegularObservation::correct_predictively_oriented`](crate::regular::RegularObservation::correct_predictively_oriented)
    /// for the objective, its cost, and when it helps. Only scalar observations support it;
    /// use [`correct_predictively_oriented_vector`](Self::correct_predictively_oriented_vector)
    /// for vector observations.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition, e.g. `20`.
    pub fn correct_predictively_oriented<M>(
        &mut self,
        measurement: &mut M,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectPredictivelyOriented<STATES, T>,
    {
        measurement.correct_predictively_oriented(&mut self.x, &mut self.P, max_iterations)
    }

    /// Performs a predictively oriented (PrO) measurement update step for an observation with
    /// any number of measurements.
    ///
    /// See
    /// [`RegularObservation::correct_predictively_oriented_vector`](crate::regular::RegularObservation::correct_predictively_oriented_vector)
    /// for the iteration, its cost and its convergence.
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    /// * `workspace` - Three `OBSERVATIONS × OBSERVATIONS` scratch matrices; see
    ///   [`PredictivelyOrientedWorkspace`](crate::regular::PredictivelyOrientedWorkspace).
    /// * `max_iterations` - The budget of accepted steps of the iteration, e.g. `50`.
    ///
    /// ## Example
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f64>::default();
    /// let mut filter = builder.build();
    /// let mut measurement = builder.observations().build::<2>();
    /// let mut workspace = PredictivelyOrientedWorkspace::new(
    ///     BufferBuilder::temp_S_inv::<2>().new(),
    ///     BufferBuilder::temp_S_inv::<2>().new(),
    ///     BufferBuilder::temp_S_inv::<2>().new(),
    /// );
    ///
    /// filter.estimate_covariance_mut().make_identity();
    /// measurement.observation_matrix_mut().make_identity();
    /// measurement.measurement_noise_covariance_mut().make_identity();
    /// measurement.measurement_vector_mut().set_at(0, 0, 3.0);
    ///
    /// let status = filter.correct_predictively_oriented_vector(&mut measurement, &mut workspace, 50);
    /// assert!(status.is_applied());
    /// # }
    /// ```
    pub fn correct_predictively_oriented_vector<const OBSERVATIONS: usize, M, W>(
        &mut self,
        measurement: &mut M,
        workspace: &mut crate::regular::PredictivelyOrientedWorkspace<W>,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectPredictivelyOrientedVector<STATES, OBSERVATIONS, T>,
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    {
        measurement.correct_predictively_oriented_vector(
            &mut self.x,
            &mut self.P,
            workspace,
            max_iterations,
        )
    }

    /// Performs an outlier-robust measurement update step with Student-t measurement noise.
    ///
    #[doc = diagram!("regular-loop.mmd", "correct")]
    ///
    /// A Kalman update with the noise covariance `R/w`, where the weight `w` is fitted to the
    /// innovation by a few variational Bayes iterations (Piché, Särkkä, Hartikainen, IEEE
    /// MLSP 2012). Outliers get a small weight and move the state little; `P` never grows past
    /// the prior. See
    /// [`RegularObservation::correct_robust_with`](crate::regular::RegularObservation::correct_robust_with)
    /// for the update, its status, and when to prefer it over the
    /// [predictively oriented](Self::correct_predictively_oriented_vector) correction.
    ///
    /// The [process noise adaptation](Self::set_process_noise_adaptation) runs on applied
    /// corrections, including [`CorrectionStatus::NotConverged`].
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    /// * `workspace` - One `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget, e.g. `20`.
    ///
    /// ## Example
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f64>::default();
    /// let mut filter = builder.build();
    /// let mut measurement = builder.observations().build::<2>();
    /// let mut workspace = BufferBuilder::temp_S_inv::<2>().new();
    ///
    /// filter.estimate_covariance_mut().make_identity();
    /// measurement.observation_matrix_mut().make_identity();
    /// measurement.measurement_noise_covariance_mut().make_identity();
    /// measurement.measurement_vector_mut().set_at(0, 0, 50.0);
    ///
    /// let robust = RobustCorrection {
    ///     weight: RobustWeight::StudentT { dof: 4.0 },
    ///     iterations: 20,
    /// };
    /// let status = filter.correct_robust(&mut measurement, &mut workspace, robust);
    /// assert_eq!(status, CorrectionStatus::Applied);
    ///
    /// // The 50σ outlier is discounted: a Kalman update would move the state to 25.
    /// assert!(measurement.robust_weight() < 0.01);
    /// assert!(filter.state_vector().get_at(0, 0) < 1.0);
    /// # }
    /// ```
    pub fn correct_robust<const OBSERVATIONS: usize, M, W>(
        &mut self,
        measurement: &mut M,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectRobust<STATES, OBSERVATIONS, T>,
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    {
        correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| measurement.correct_robust(x, &mut self.P, workspace, robust),
        )
    }

    /// Performs an outlier-robust measurement update step with Student-t measurement noise,
    /// normalizing the innovation before it is applied.
    ///
    /// See [`correct_robust`](Self::correct_robust) and [`correct_with`](Self::correct_with).
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    /// * `workspace` - One `OBSERVATIONS × OBSERVATIONS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget.
    /// * `fix` - Normalizes the innovation vector in place.
    pub fn correct_robust_with<const OBSERVATIONS: usize, M, W, N>(
        &mut self,
        measurement: &mut M,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
        fix: N,
    ) -> CorrectionStatus
    where
        P: EstimateCovarianceMatrix<STATES, T>,
        X: StateVectorMut<STATES, T>,
        T: MatrixDataType,
        M: KalmanFilterObservationCorrectRobust<STATES, OBSERVATIONS, T>,
        W: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
        N: FnMut(&mut [T]),
    {
        correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| measurement.correct_robust_with(x, &mut self.P, workspace, robust, fix),
        )
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
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
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
    Q: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
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

    /// Returns the online adaptation of the process noise `Q`, if enabled with
    /// [`set_process_noise_adaptation`](Self::set_process_noise_adaptation).
    #[inline(always)]
    pub fn process_noise_adaptation(&self) -> Option<NoiseAdaptation<T>>
    where
        T: MatrixDataType,
    {
        self.process_noise_adaptation
            .map(|adapter| adapter.settings())
    }

    /// Enables or disables the online adaptation of the process noise `Q`.
    ///
    /// With `Some`, every applied [`correct`](Self::correct) or [`correct_with`](Self::correct_with)
    /// replaces `Q` by `α×Q + (1−α)×Δx×Δxᵀ` from the state correction `Δx = x⁺ − x⁻`, then
    /// raises every diagonal entry to at least the floor; `Q` stays symmetric positive
    /// semi-definite. The estimate is unbiased only for a steady state with `A×P⁺×Aᵀ = P⁺`,
    /// e.g. a random walk; otherwise it is biased, and with a steady-state gain it has rank
    /// one along `K`. Corrections with an external gain and predictively oriented
    /// corrections do not adapt `Q`. `None` (the default) leaves `Q` untouched.
    ///
    /// See [`NoiseAdaptation`] for the forgetting factor, the method and its limits.
    #[inline(always)]
    pub fn set_process_noise_adaptation(&mut self, adaptation: Option<NoiseAdaptation<T>>)
    where
        T: MatrixDataType,
    {
        self.process_noise_adaptation = adaptation.map(ProcessNoiseAdapter::new);
    }
}

impl<const STATES: usize, T, A, X, P, Q, PX, TempP> KalmanFilterPredict<STATES, T>
    for RegularKalman<STATES, T, A, X, P, Q, PX, TempP>
where
    X: StateVectorMut<STATES, T>,
    A: StateTransitionMatrix<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    P: EstimateCovarianceMatrix<STATES, T>,
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
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
    fn correct_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFilter<STATES, T>,
        N: FnMut(&mut [T]),
    {
        self.correct_with(measurement, fix)
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
