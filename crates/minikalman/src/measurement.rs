use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

/// A builder for a Kalman filter [`Measurement`] instances.
#[allow(clippy::type_complexity)]
pub struct MeasurementBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP> {
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
    MeasurementBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
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
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// // Measurement buffers.
    /// impl_buffer_z!(mut gravity_z, NUM_MEASUREMENTS, f32, 0.0);
    /// impl_buffer_H!(mut gravity_H, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// impl_buffer_R!(mut gravity_R, NUM_MEASUREMENTS, f32, 0.0);
    /// impl_buffer_y!(mut gravity_y, NUM_MEASUREMENTS, f32, 0.0);
    /// impl_buffer_S!(mut gravity_S, NUM_MEASUREMENTS, f32, 0.0);
    /// impl_buffer_K!(mut gravity_K, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    ///
    /// // Measurement temporaries.
    /// impl_buffer_temp_S_inv!(mut gravity_temp_S_inv, NUM_MEASUREMENTS, f32, 0.0);
    /// impl_buffer_temp_HP!(mut gravity_temp_HP, NUM_MEASUREMENTS, NUM_STATES, f32, 0.0);
    /// impl_buffer_temp_PHt!(mut gravity_temp_PHt, NUM_STATES, NUM_MEASUREMENTS, f32, 0.0);
    /// impl_buffer_temp_KHP!(mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
    ///
    /// let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
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
    pub fn new<const STATES: usize, const MEASUREMENTS: usize, T>(
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
    ) -> Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    where
        T: MatrixDataType,
        Z: MeasurementVector<MEASUREMENTS, T>,
        H: MeasurementObservationMatrix<MEASUREMENTS, STATES, T>,
        R: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
        Y: InnovationVector<MEASUREMENTS, T>,
        S: ResidualCovarianceMatrix<MEASUREMENTS, T>,
        K: KalmanGainMatrix<STATES, MEASUREMENTS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>,
        TempHP: TemporaryHPMatrix<MEASUREMENTS, STATES, T>,
        TempPHt: TemporaryPHTMatrix<STATES, MEASUREMENTS, T>,
        TempKHP: TemporaryKHPMatrix<STATES, T>,
    {
        Measurement::<STATES, MEASUREMENTS, T, _, _, _, _, _, _, _, _, _, _> {
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

/// Kalman Filter measurement structure. See [`MeasurementBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Measurement<
    const STATES: usize,
    const MEASUREMENTS: usize,
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
    /// Measurement vector.
    pub(crate) z: Z,

    /// Measurement transformation matrix.
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
        const MEASUREMENTS: usize,
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
    > Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
    /// Returns then number of measurements.
    #[inline(always)]
    pub const fn measurements(&self) -> usize {
        MEASUREMENTS
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
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// # // System buffers.
    /// # impl_buffer_x!(mut gravity_x, NUM_STATES, f32, 0.0);
    /// # impl_buffer_A!(mut gravity_A, NUM_STATES, f32, 0.0);
    /// # impl_buffer_P!(mut gravity_P, NUM_STATES, f32, 0.0);
    /// #
    /// # // Input buffers.
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
    #[inline(always)]
    pub fn measurement_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Z),
    {
        f(&mut self.z)
    }

    /// Gets a reference to the measurement transformation matrix H.
    #[inline(always)]
    pub fn measurement_transformation_ref(&self) -> &H {
        &self.H
    }

    /// Gets a reference to the process noise matrix R.
    #[inline(always)]
    pub fn process_noise_ref(&self) -> &R {
        &self.R
    }

    /// Gets a mutable reference to the process noise matrix R.
    #[inline(always)]
    #[doc(alias = "kalman_get_process_noise")]
    pub fn process_noise_mut(&mut self) -> &mut R {
        &mut self.R
    }

    /// Applies a function to the process noise matrix R.
    #[inline(always)]
    pub fn process_noise_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut R),
    {
        f(&mut self.R)
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: MeasurementObservationMatrix<MEASUREMENTS, STATES, T>,
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
    /// Applies a correction step to the provided state vector and covariance matrix.
    #[allow(non_snake_case)]
    pub fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
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
        const MEASUREMENTS: usize,
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
    > Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: MeasurementObservationMatrixMut<MEASUREMENTS, STATES, T>,
{
    /// Gets a mutable reference to the measurement transformation matrix H.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_transformation")]
    pub fn measurement_transformation_mut(&mut self) -> &mut H {
        &mut self.H
    }

    /// Applies a function to the measurement transformation matrix H.
    #[inline(always)]
    pub fn measurement_transformation_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut H),
    {
        f(&mut self.H)
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterNumMeasurements<MEASUREMENTS>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementVector<MEASUREMENTS, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    Z: MeasurementVector<MEASUREMENTS, T>,
{
    type MeasurementVector = Z;

    fn measurement_vector_ref(&self) -> &Self::MeasurementVector {
        self.measurement_vector_ref()
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementVectorMut<MEASUREMENTS, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    Z: MeasurementVectorMut<MEASUREMENTS, T>,
{
    type MeasurementVectorMut = Z;

    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut {
        self.measurement_vector_mut()
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementTransformation<STATES, MEASUREMENTS, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: MeasurementObservationMatrix<MEASUREMENTS, STATES, T>,
{
    type MeasurementTransformationMatrix = H;

    fn measurement_transformation_ref(&self) -> &Self::MeasurementTransformationMatrix {
        self.measurement_transformation_ref()
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementTransformationMut<STATES, MEASUREMENTS, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: MeasurementObservationMatrixMut<MEASUREMENTS, STATES, T>,
{
    type MeasurementTransformationMatrixMut = H;

    fn measurement_transformation_mut(&mut self) -> &mut Self::MeasurementTransformationMatrixMut {
        self.measurement_transformation_mut()
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementProcessNoise<MEASUREMENTS, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    R: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
{
    type MeasurementProcessNoiseMatrix = R;

    fn process_noise_ref(&self) -> &Self::MeasurementProcessNoiseMatrix {
        self.process_noise_ref()
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementProcessNoiseMut<MEASUREMENTS, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    R: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
{
    type MeasurementProcessNoiseMatrixMut = R;

    fn process_noise_mut(&mut self) -> &mut Self::MeasurementProcessNoiseMatrixMut {
        self.process_noise_mut()
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
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
    > KalmanFilterMeasurementCorrectFilter<STATES, T>
    for Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: MeasurementObservationMatrix<MEASUREMENTS, STATES, T>,
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
    #[allow(non_snake_case)]
    fn correct<X, P>(&mut self, x: &mut X, P: &mut P)
    where
        X: StateVectorMut<STATES, T>,
        P: SystemCovarianceMatrix<STATES, T>,
    {
        self.correct(x, P)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_dummies::{Dummy, DummyMatrix};

    use super::*;

    fn trait_impl<const STATES: usize, const MEASUREMENTS: usize, T, M>(_measurement: M)
    where
        M: KalmanFilterMeasurement<STATES, MEASUREMENTS, T>,
    {
    }

    #[test]
    fn builder_simple() {
        let measurement = MeasurementBuilder::new::<3, 1, f32>(
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

        trait_impl(measurement);
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

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        MeasurementObservationMatrix<MEASUREMENTS, STATES, T> for Dummy<T>
    {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        MeasurementObservationMatrixMut<MEASUREMENTS, STATES, T> for Dummy<T>
    {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>
        for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::TargetMut {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> InnovationVector<MEASUREMENTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> ResidualCovarianceMatrix<MEASUREMENTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, const MEASUREMENTS: usize, T>
        KalmanGainMatrix<STATES, MEASUREMENTS, T> for Dummy<T>
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

    impl<const MEASUREMENTS: usize, T> TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>
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

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        TemporaryHPMatrix<MEASUREMENTS, STATES, T> for Dummy<T>
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

    impl<const STATES: usize, const MEASUREMENTS: usize, T>
        TemporaryPHTMatrix<STATES, MEASUREMENTS, T> for Dummy<T>
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
