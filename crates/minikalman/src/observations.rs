use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

/// A builder for a Kalman filter [`Observation`] instances.
#[allow(clippy::type_complexity)]
pub struct ObservationBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP> {
    _phantom: (
        PhantomData<Z>,
        PhantomData<H>,
        PhantomData<R>,
        PhantomData<Y>,
        PhantomData<S>,
        PhantomData<K>,
        PhantomData<TempSInv>,
        PhantomData<TempHP>,
        PhantomData<TempPHt>,
        PhantomData<TempKHP>,
    ),
}

impl<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    ObservationBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
    /// Initializes a measurement.
    ///
    /// ## Arguments
    /// * `num_states` - The number of states tracked by the filter.
    /// * `num_measurements` - The number of measurements available to the filter.
    /// * `H` - The measurement transformation matrix (`num_measurements` × `num_states`).
    /// * `z` - The measurement vector (`num_measurements` × `1`).
    /// * `R` - The process noise / measurement uncertainty (`num_measurements` × `num_measurements`).
    /// * `y` - The innovation (`num_measurements` × `1`).
    /// * `S` - The residual covariance (`num_measurements` × `num_measurements`).
    /// * `K` - The Kalman gain (`num_states` × `num_measurements`).
    /// * `S_inv` - The temporary vector for predicted states (`num_states` × `1`).
    /// * `temp_HP` - The temporary matrix for H×P calculation (`num_measurements` × `num_states`).
    /// * `temp_PHt` - The temporary matrix for P×H' calculation (`num_states` × `num_measurements`).
    /// * `temp_KHP` - The temporary matrix for K×H×P calculation (`num_states` × `num_states`).
    /// ## Example
    ///
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_CONTROLS: usize = 0;
    /// # const NUM_OBSERVATIONS: usize = 1;
    /// // Observation buffers.
    /// impl_buffer_z!(mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
    /// impl_buffer_H!(mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// impl_buffer_R!(mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
    /// impl_buffer_y!(mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
    /// impl_buffer_S!(mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
    /// impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    ///
    /// // Observation temporaries.
    /// impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
    /// impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
    /// impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
    /// impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    ///
    /// let mut measurement = ObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
    ///     gravity_H,
    ///     gravity_z,
    ///     gravity_R,
    ///     gravity_y,
    ///     gravity_S,
    ///     gravity_K,
    ///     gravity_temp_S_inv,
    ///     gravity_temp_HP,
    ///     gravity_temp_PHt,
    ///     gravity_temp_KHP,
    /// );
    /// ```
    ///
    /// See also [`KalmanBuilder::new`](KalmanBuilder::new) for setting up the Kalman filter itself.
    #[allow(non_snake_case, clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub fn new<const STATES: usize, const OBSERVATIONS: usize, T>(
        H: H,
        z: Z,
        R: R,
        y: Y,
        S: S,
        K: K,
        temp_S_inv: TempSInv,
        temp_HP: TempHP,
        temp_PHt: TempPHt,
        temp_KHP: TempKHP,
    ) -> Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    where
        T: MatrixDataType,
        Z: MeasurementVector<OBSERVATIONS, T>,
        H: ObservationMatrix<OBSERVATIONS, STATES, T>,
        R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
        Y: InnovationVector<OBSERVATIONS, T>,
        S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
        K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
        TempHP: TemporaryHPMatrix<OBSERVATIONS, STATES, T>,
        TempPHt: TemporaryPHTMatrix<STATES, OBSERVATIONS, T>,
        TempKHP: TemporaryKHPMatrix<STATES, T>,
    {
        Observation::<STATES, OBSERVATIONS, T, _, _, _, _, _, _, _, _, _, _> {
            z,
            H,
            R,
            y,
            S,
            K,
            temp_S_inv,
            temp_HP,
            temp_PHt,
            temp_KHP,
            _phantom: Default::default(),
        }
    }
}

/// Kalman Filter measurement structure. See [`ObservationBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Observation<
    const STATES: usize,
    const OBSERVATIONS: usize,
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
> {
    /// Observation vector.
    pub(crate) z: Z,

    /// Observation transformation matrix.
    ///
    /// See also [`R`].
    pub(crate) H: H,

    /// Process noise covariance matrix.
    ///
    /// See also [`A`].
    pub(crate) R: R,

    /// Innovation vector.
    pub(crate) y: Y,

    /// Residual covariance matrix.
    pub(crate) S: S,

    /// Kalman gain matrix.
    pub(crate) K: K,

    /// S-Sized temporary matrix  (number of measurements × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_KHP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`tmp_HP`] (if it is not aliased with [`temp_PHt`]).
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_PHt`].
    pub(crate) temp_S_inv: TempSInv,

    /// H-Sized temporary matrix  (number of measurements × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_KHP`].
    pub(crate) temp_HP: TempHP,

    /// P×H'-Sized (H'-Sized) temporary matrix  (number of states × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`tmp_HP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_KHP`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_S_inv`].
    pub(crate) temp_PHt: TempPHt,

    /// P-Sized temporary matrix  (number of states × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`tmp_HP`].
    pub(crate) temp_KHP: TempKHP,

    _phantom: PhantomData<T>,
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
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
    > Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
    /// Returns then number of measurements.
    #[inline(always)]
    pub const fn measurements(&self) -> usize {
        OBSERVATIONS
    }

    /// Returns then number of states.
    #[inline(always)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Gets a reference to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_ref(&self) -> &Z {
        &self.z
    }

    /// Gets a mutable reference to the measurement vector z.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_vector")]
    pub fn measurement_vector_mut(&mut self) -> &mut Z {
        &mut self.z
    }

    /// Applies a function to the measurement vector z.
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
    /// # // Control buffers.
    /// # impl_buffer_u!(mut gravity_u, NUM_CONTROLS, f32, 0.0);
    /// # impl_buffer_B!(mut gravity_B, NUM_STATES, NUM_CONTROLS, f32, 0.0);
    /// # impl_buffer_Q!(mut gravity_Q, NUM_CONTROLS, f32, 0.0);
    /// #
    /// # // Filter temporaries.
    /// # impl_buffer_temp_x!(mut gravity_temp_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_P!(mut gravity_temp_P, NUM_STATES, f32, 0.0);
    /// # impl_buffer_temp_BQ!(mut gravity_temp_BQ, NUM_STATES, NUM_CONTROLS, f32, 0.0);
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
    #[inline(always)]
    pub fn measurement_vector_apply<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&mut Z) -> O,
    {
        f(&mut self.z)
    }

    /// Applies a function to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_apply_mut<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut Z) -> O,
    {
        f(&mut self.z)
    }

    /// Applies a function to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_inspect<F, O>(&self, f: F) -> O
    where
        F: Fn(&Z) -> O,
    {
        f(&self.z)
    }

    /// Applies a function to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_inspect_mut<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&Z) -> O,
    {
        f(&self.z)
    }

    /// Gets a reference to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    pub fn observation_matrix_ref(&self) -> &H {
        &self.H
    }

    /// Applies a function to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    pub fn observation_matrix_inspect<F, O>(&self, f: F) -> O
    where
        F: Fn(&H) -> O,
    {
        f(&self.H)
    }

    /// Applies a function to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    pub fn observation_matrix_inspect_mut<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&H) -> O,
    {
        f(&self.H)
    }

    /// Gets a reference to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    pub fn measurement_noise_ref(&self) -> &R {
        &self.R
    }

    /// Gets a mutable reference to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_noise")]
    pub fn measurement_noise_mut(&mut self) -> &mut R {
        &mut self.R
    }

    /// Applies a function to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    pub fn measurement_noise_apply<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut R) -> O,
    {
        f(&mut self.R)
    }

    /// Applies a function to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    pub fn measurement_noise_apply_mut<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&mut R) -> O,
    {
        f(&mut self.R)
    }

    /// Applies a function to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    pub fn measurement_noise_inspect<F, O>(&self, f: F) -> O
    where
        F: Fn(&R) -> O,
    {
        f(&self.R)
    }

    /// Applies a function to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    pub fn measurement_noise_inspect_mut<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&R) -> O,
    {
        f(&self.R)
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
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
    > Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    Z: MeasurementVector<OBSERVATIONS, T>,
    TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    TempHP: TemporaryHPMatrix<OBSERVATIONS, STATES, T>,
    TempPHt: TemporaryPHTMatrix<STATES, OBSERVATIONS, T>,
    TempKHP: TemporaryKHPMatrix<STATES, T>,
    T: MatrixDataType,
{
    /// Applies a correction step to the provided state vector and covariance matrix.
    #[allow(non_snake_case)]
    pub fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        // matrices and vectors
        let P = P.as_matrix_mut();
        let x = x.as_matrix_mut();

        let H = self.H.as_matrix();
        let K = self.K.as_matrix_mut();
        let S = self.S.as_matrix_mut();
        let R = self.R.as_matrix_mut();
        let y = self.y.as_matrix_mut();
        let z = self.z.as_matrix();

        // temporaries
        let S_inv = self.temp_S_inv.as_matrix_mut();
        let temp_HP = self.temp_HP.as_matrix_mut();
        let temp_KHP = self.temp_KHP.as_matrix_mut();
        let temp_PHt = self.temp_PHt.as_matrix_mut();

        //* Calculate innovation and residual covariance
        //* y = z - H*x
        //* S = H*P*Hᵀ + R

        // y = z - H*x
        H.mult_rowvector(x, y);
        z.sub_inplace_b(y);

        // S = H*P*H' + R
        H.mult(P, temp_HP); // temp = H*P
        temp_HP.mult_transb(H, S); // S = temp*Hᵀ
        S.add_inplace_a(R); // S += R

        //* Calculate Kalman gain
        //* K = P*Hᵀ * S^-1

        // K = P*Hᵀ * S^-1
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

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
{
    /// Gets a mutable reference to the observation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_transformation")]
    pub fn observation_matrix_mut(&mut self) -> &mut H {
        &mut self.H
    }

    /// Applies a function to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    pub fn observation_matrix_apply<F, O>(&mut self, f: F) -> O
    where
        F: Fn(&mut H) -> O,
    {
        f(&mut self.H)
    }

    /// Applies a function to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    pub fn observation_matrix_apply_mut<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut H) -> O,
    {
        f(&mut self.H)
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterNumStates<STATES>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterNumObservations<OBSERVATIONS>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterMeasurementVector<OBSERVATIONS, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    Z: MeasurementVector<OBSERVATIONS, T>,
{
    type MeasurementVector = Z;

    fn measurement_vector_ref(&self) -> &Self::MeasurementVector {
        self.measurement_vector_ref()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
{
    type MeasurementVectorMut = Z;

    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut {
        self.measurement_vector_mut()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationMatrix = H;

    fn observation_matrix_ref(&self) -> &Self::ObservationTransformationMatrix {
        self.observation_matrix_ref()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterObservationTransformationMut<STATES, OBSERVATIONS, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationMatrixMut = H;

    fn observation_matrix_mut(&mut self) -> &mut Self::ObservationTransformationMatrixMut {
        self.observation_matrix_mut()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterMeasurementNoiseCovariance<OBSERVATIONS, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    type MeasurementNoiseCovarianceMatrix = R;

    fn measurement_noise_covariance_ref(&self) -> &Self::MeasurementNoiseCovarianceMatrix {
        self.measurement_noise_ref()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    type MeasurementNoiseCovarianceMatrixMut = R;

    fn measurement_noise_covariance_mut(
        &mut self,
    ) -> &mut Self::MeasurementNoiseCovarianceMatrixMut {
        self.measurement_noise_mut()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        T,
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
    > KalmanFilterObservationCorrectFilter<STATES, T>
    for Observation<STATES, OBSERVATIONS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    Z: MeasurementVector<OBSERVATIONS, T>,
    TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    TempHP: TemporaryHPMatrix<OBSERVATIONS, STATES, T>,
    TempPHt: TemporaryPHTMatrix<STATES, OBSERVATIONS, T>,
    TempKHP: TemporaryKHPMatrix<STATES, T>,
    T: MatrixDataType,
{
    #[allow(non_snake_case)]
    fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.correct(x, P)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_dummies::{Dummy, DummyMatrix};

    use super::*;

    #[test]
    #[cfg(feature = "alloc")]
    fn test_apply() {
        use crate::builder::KalmanFilterBuilder;

        let builder = KalmanFilterBuilder::<3, f32>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<5>();

        filter.predict();
        filter.correct(&mut measurement);
    }

    fn trait_impl<const STATES: usize, const OBSERVATIONS: usize, T, M>(mut measurement: M) -> M
    where
        M: KalmanFilterObservation<STATES, OBSERVATIONS, T>
            + KalmanFilterObservationTransformationMut<STATES, OBSERVATIONS, T>,
    {
        assert_eq!(measurement.states(), STATES);
        assert_eq!(measurement.observations(), OBSERVATIONS);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = measurement.measurement_vector_ref();
        let _vec = measurement.measurement_vector_mut();
        measurement.measurement_vector_inspect(|_vec| test_fn());
        measurement.measurement_vector_inspect_mut(|_vec| test_fn_mut());
        measurement.measurement_vector_apply(|_vec| test_fn());
        measurement.measurement_vector_apply_mut(|_vec| test_fn_mut());

        let _mat = measurement.observation_matrix_ref();
        let _mat = measurement.observation_matrix_mut();
        measurement.observation_matrix_inspect(|_mat| test_fn());
        measurement.observation_matrix_inspect_mut(|_mat| test_fn_mut());
        measurement.observation_matrix_apply(|_mat| test_fn());
        measurement.observation_matrix_apply_mut(|_mat| test_fn_mut());

        let _mat = measurement.measurement_noise_covariance_ref();
        let _mat = measurement.measurement_noise_covariance_mut();
        measurement.measurement_noise_covariance_inspect(|_mat| test_fn());
        measurement.measurement_noise_covariance_inspect_mut(|_mat| test_fn_mut());
        measurement.measurement_noise_covariance_apply(|_mat| test_fn());
        measurement.measurement_noise_covariance_apply_mut(|_mat| test_fn_mut());

        measurement
    }

    #[test]
    fn builder_simple() {
        let measurement = ObservationBuilder::new::<3, 1, f32>(
            Dummy::default(),
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

        let mut measurement = trait_impl(measurement);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = measurement.measurement_vector_ref();
        let _vec = measurement.measurement_vector_mut();
        measurement.measurement_vector_inspect(|_vec| test_fn());
        measurement.measurement_vector_inspect_mut(|_vec| test_fn_mut());
        measurement.measurement_vector_apply(|_vec| test_fn());
        measurement.measurement_vector_apply_mut(|_vec| test_fn_mut());

        let _mat = measurement.observation_matrix_ref();
        let _mat = measurement.observation_matrix_mut();
        measurement.observation_matrix_inspect(|_mat| test_fn());
        measurement.observation_matrix_inspect_mut(|_mat| test_fn_mut());
        measurement.observation_matrix_apply(|_mat| test_fn());
        measurement.observation_matrix_apply_mut(|_mat| test_fn_mut());

        let _mat = measurement.measurement_noise_covariance_ref();
        let _mat = measurement.measurement_noise_covariance_mut();
        measurement.measurement_noise_covariance_inspect(|_mat| test_fn());
        measurement.measurement_noise_covariance_inspect_mut(|_mat| test_fn_mut());
        measurement.measurement_noise_covariance_apply(|_mat| test_fn());
        measurement.measurement_noise_covariance_apply_mut(|_mat| test_fn_mut());
    }

    impl<const STATES: usize, T> MeasurementVector<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<const STATES: usize, T> MeasurementVectorMut<STATES, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const OBSERVATIONS: usize, const STATES: usize, T>
        ObservationMatrix<OBSERVATIONS, STATES, T> for Dummy<T>
    {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<const OBSERVATIONS: usize, const STATES: usize, T>
        ObservationMatrixMut<OBSERVATIONS, STATES, T> for Dummy<T>
    {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const OBSERVATIONS: usize, T> MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::TargetMut {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const OBSERVATIONS: usize, T> InnovationVector<OBSERVATIONS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const OBSERVATIONS: usize, T> InnovationCovarianceMatrix<OBSERVATIONS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, const OBSERVATIONS: usize, T>
        KalmanGainMatrix<STATES, OBSERVATIONS, T> for Dummy<T>
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

    impl<const OBSERVATIONS: usize, T> TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>
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

    impl<const OBSERVATIONS: usize, const STATES: usize, T>
        TemporaryHPMatrix<OBSERVATIONS, STATES, T> for Dummy<T>
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

    impl<const STATES: usize, T> TemporaryKHPMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, const OBSERVATIONS: usize, T>
        TemporaryPHTMatrix<STATES, OBSERVATIONS, T> for Dummy<T>
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
