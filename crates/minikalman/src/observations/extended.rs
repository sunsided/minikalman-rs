//! # Observations for Extended Kalman Filters.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, RowMajorSequentialDataMut};
use crate::observations::NoiseAdaptation;

/// A builder for a Kalman filter [`ExtendedObservation`] instances.
#[allow(clippy::type_complexity)]
pub struct ExtendedObservationBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP> {
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
    ExtendedObservationBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
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
    /// use minikalman::extended::ExtendedObservationBuilder;
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
    /// let mut measurement = ExtendedObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
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
    /// See also [`RegularKalmanBuilder::new`](regular::RegularKalmanBuilder::new) for setting up the Kalman filter itself.
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
    ) -> ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
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
        ExtendedObservation::<STATES, OBSERVATIONS, T, _, _, _, _, _, _, _, _, _, _> {
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
            nis: T::zero(),
            gate: None,
            noise_adaptation: None,
        }
    }
}

/// Kalman Filter measurement structure. See [`ExtendedObservationBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct ExtendedObservation<
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

    /// Normalized innovation squared `yᵀ×S⁻¹×y` of the last correction.
    pub(crate) nis: T,

    /// Upper bound on the normalized innovation squared above which a correction is skipped.
    pub(crate) gate: Option<T>,

    /// Online adaptation of `R` after applied corrections; `None` disables it.
    pub(crate) noise_adaptation: Option<NoiseAdaptation<T>>,
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
    >
    ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
{
    /// Returns the normalized innovation squared `NIS = yᵀ×S⁻¹×y` of the last correction.
    ///
    /// For a consistent filter, the NIS averages to `OBSERVATIONS` over time; a larger average
    /// indicates an overconfident filter. The value is updated by every correction whose
    /// innovation covariance `S` is positive definite, including corrections rejected by the
    /// [gate](Self::set_gate), and is `0` before the first such correction.
    /// Corrections with an external gain ([`correct_nonlinear_with_gain`](Self::correct_nonlinear_with_gain)) do not invert `S` and leave
    /// the NIS unchanged.
    ///
    /// With fixed-point types, large innovations can overflow `yᵀ×S⁻¹×y`; choose a type with
    /// enough integer bits for the expected outliers.
    #[inline(always)]
    pub fn nis(&self) -> T
    where
        T: Copy,
    {
        self.nis
    }

    /// Returns the gate set by [`set_gate`](Self::set_gate).
    #[inline(always)]
    pub fn gate(&self) -> Option<T>
    where
        T: Copy,
    {
        self.gate
    }

    /// Sets an innovation gate: a correction whose [normalized innovation squared](Self::nis)
    /// exceeds `threshold` is skipped and leaves the state and covariance unchanged.
    ///
    /// A common choice is a chi-square quantile for `OBSERVATIONS` degrees of freedom,
    /// e.g. `6.63` (one measurement) or `9.21` (two measurements) to reject about 1% of
    /// measurements of a consistent filter. `None` (the default) disables the gate.
    ///
    /// The gate compares the NIS after it has been computed in `T`, so it does not protect
    /// against overflow: with fixed-point types, an outlier large enough to overflow
    /// `yᵀ×S⁻¹×y` panics in debug builds and wraps in release builds, where the wrapped value
    /// may pass the gate. Choose a type with enough integer bits for the expected outliers.
    #[inline(always)]
    pub fn set_gate(&mut self, threshold: Option<T>) {
        self.gate = threshold;
    }

    /// Returns the measurement noise adaptation set by
    /// [`set_noise_adaptation`](Self::set_noise_adaptation).
    #[inline(always)]
    pub fn noise_adaptation(&self) -> Option<NoiseAdaptation<T>>
    where
        T: Copy,
    {
        self.noise_adaptation
    }

    /// Enables online adaptation of the measurement noise covariance `R` from the residuals
    /// of the corrections; `None` (the default) disables it.
    ///
    /// After every correction applied by `correct_nonlinear` and `correct_nonlinear_with`, `R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)` with the residual
    /// `ε = z − H×x⁺`; gated and skipped corrections leave `R` unchanged. See
    /// [`NoiseAdaptation`] for the method, the choice of `α` and when the estimate diverges.
    ///
    /// Only those corrections adapt `R`: `correct_nonlinear_with_gain*` and `correct_nonlinear_recalibrated*` leave `R` and its floor untouched, also when
    /// they return [`CorrectionStatus::Applied`].
    #[inline(always)]
    pub fn set_noise_adaptation(&mut self, adaptation: Option<NoiseAdaptation<T>>) {
        self.noise_adaptation = adaptation;
    }

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
    pub fn measurement_vector(&self) -> &Z {
        &self.z
    }

    /// Gets a mutable reference to the measurement vector z.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_vector")]
    pub fn measurement_vector_mut(&mut self) -> &mut Z {
        &mut self.z
    }

    /// Gets a reference to the Jacobian of the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    ///
    /// ## Extended Kalman Filters
    /// In Extended Kalman Filters, this matrix is treated as the Jacobian of the observation matrix,
    /// i.e. the derivative of the measurement function with respect to the state vector.
    #[inline(always)]
    pub fn observation_jacobian_matrix(&self) -> &H {
        &self.H
    }

    /// Gets a reference to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    pub fn measurement_noise_covariance(&self) -> &R {
        &self.R
    }

    /// Gets a mutable reference to the measurement noise matrix R.
    ///
    /// This matrix represents the uncertainty in the measurements, accounting for sensor noise and
    /// inaccuracies. It quantifies the expected variability in the measurement process.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_noise")]
    pub fn measurement_noise_covariance_mut(&mut self) -> &mut R {
        &mut self.R
    }

    /// Gets a reference to the Kalman gain matrix K.
    ///
    /// After a regular correction, this holds the gain `K = P×Hᵀ×S⁻¹` that was applied.
    #[inline(always)]
    pub fn kalman_gain(&self) -> &K {
        &self.K
    }

    /// Gets a mutable reference to the Kalman gain matrix K.
    ///
    /// Set the gain here before calling [`correct_nonlinear_with_gain`](Self::correct_nonlinear_with_gain). Note that a regular
    /// correction overwrites K with the gain it computes.
    #[inline(always)]
    pub fn kalman_gain_mut(&mut self) -> &mut K {
        &mut self.K
    }

    /// Gets a reference to the innovation vector y.
    ///
    /// Every correction writes the innovation `z − h(x)` of the prior state here (after the
    /// normalization, for the `_with` variants) before it applies, gates or rejects the update,
    /// so it is also available when the update is skipped. This is the input a learned-gain
    /// filter computes its gain from.
    #[inline(always)]
    pub fn innovation_vector(&self) -> &Y {
        &self.y
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
    >
    ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
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
    /// Applies a nonlinear correction step to the provided state vector and covariance matrix.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    ///   This function uses the innovation vector as a temporary storage, even though
    ///   the innovation itself will only be calculated afterward.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear<X, P, F>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
    {
        self.correct_nonlinear_with(x, P, observation, |_| {})
    }

    /// Applies a nonlinear correction step to the provided state vector and covariance matrix,
    /// normalizing the innovation before it is applied.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `fix` - Normalizes the innovation `y = z - h(x)` in place, e.g. by wrapping angles.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        mut observation: F,
        mut fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        N: FnMut(&mut [T]),
    {
        // y = h(x)
        observation(x, &mut self.y);

        let y = self.y.as_matrix_mut();
        let z = self.z.as_matrix();

        // Calculate innovation:
        // y = z - h(x)
        z.sub_inplace_b(y);
        fix(y.as_mut_slice());

        // Perform the remaining update.
        self.update_with_innovation(x, P)
    }

    /// Applies a nonlinear correction step with the externally supplied Kalman gain stored in K.
    ///
    /// Set the gain with [`kalman_gain_mut`](Self::kalman_gain_mut) and the Jacobian of the
    /// observation function in the observation matrix first. `P` is updated with the Joseph
    /// form, so it stays symmetric and positive semi-definite for any gain. `S` is not
    /// inverted: the [gate](Self::set_gate) is not evaluated and the [NIS](Self::nis) is not
    /// updated.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_with_gain<X, P, F>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
    {
        self.correct_nonlinear_with_gain_with(x, P, observation, |_| {})
    }

    /// Applies a nonlinear correction step with the externally supplied Kalman gain stored in K,
    /// normalizing the innovation before it is applied.
    ///
    /// See [`correct_nonlinear_with_gain`](Self::correct_nonlinear_with_gain).
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `fix` - Normalizes the innovation `y = z - h(x)` in place, e.g. by wrapping angles.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_with_gain_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        mut observation: F,
        mut fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        N: FnMut(&mut [T]),
    {
        // y = z - h(x)
        observation(x, &mut self.y);
        let y = self.y.as_matrix_mut();
        self.z.as_matrix().sub_inplace_b(y);
        fix(y.as_mut_slice());

        crate::observations::update_with_gain(
            x.as_matrix_mut(),
            P.as_matrix_mut(),
            self.H.as_matrix(),
            self.R.as_matrix(),
            self.y.as_matrix(),
            self.K.as_matrix(),
            self.S.as_matrix_mut(),
            self.temp_HP.as_matrix_mut(),
            self.temp_PHt.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
        )
    }

    /// Applies a recalibrated nonlinear correction step to the provided state vector and
    /// covariance matrix.
    ///
    /// See [`KalmanFilterNonlinearObservationCorrectRecalibrated`] for the algorithm. The
    /// observation Jacobian must be set up at the prior `x`; `jacobian` overwrites it with the
    /// Jacobian at the updated state, so it holds that one on return.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `jacobian` - Writes the observation Jacobian at the given (updated) state.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_recalibrated<X, P, F, J>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        J: FnMut(&X, &mut H),
        H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
    {
        self.correct_nonlinear_recalibrated_with(x, P, observation, jacobian, |_| {})
    }

    /// Applies a recalibrated nonlinear correction step to the provided state vector and
    /// covariance matrix, normalizing the innovation before it is applied.
    ///
    /// See [`correct_nonlinear_recalibrated`](Self::correct_nonlinear_recalibrated) and
    /// [`correct_nonlinear_with`](Self::correct_nonlinear_with).
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `jacobian` - Writes the observation Jacobian at the given (updated) state.
    /// * `fix` - Normalizes the innovation `y = z - h(x)` in place, e.g. by wrapping angles.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_recalibrated_with<X, P, F, J, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        mut observation: F,
        mut jacobian: J,
        mut fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        J: FnMut(&X, &mut H),
        N: FnMut(&mut [T]),
        H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
    {
        // y = z - h(x)
        observation(x, &mut self.y);
        self.z.as_matrix().sub_inplace_b(self.y.as_matrix_mut());
        fix(self.y.as_matrix_mut().as_mut_slice());

        // The update is skipped, leaving x and P unchanged, if S cannot be inverted or
        // the innovation is gated.
        let status = crate::observations::update_state_for_recalibration(
            x.as_matrix_mut(),
            P.as_matrix(),
            self.H.as_matrix(),
            self.R.as_matrix(),
            self.y.as_matrix(),
            self.K.as_matrix_mut(),
            self.S.as_matrix_mut(),
            self.temp_S_inv.as_matrix_mut(),
            self.temp_HP.as_matrix_mut(),
            self.temp_PHt.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
            &mut self.nis,
            self.gate,
        );
        if !status.is_applied() {
            return status;
        }

        // H₂ = ∂h/∂x at x⁺
        jacobian(x, &mut self.H);

        crate::observations::recalibrate_covariance(
            x.as_matrix_mut(),
            P.as_matrix_mut(),
            self.H.as_matrix(),
            self.R.as_matrix(),
            self.K.as_matrix(),
            self.S.as_matrix_mut(),
            self.temp_HP.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
        )
    }

    /// Performs the update step. Assumes that the innovation vector is already calculated correctly,
    /// either by means of [`correct`](Self::correct) or [`correct_nonlinear`](Self::correct_nonlinear).
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    #[allow(non_snake_case)]
    fn update_with_innovation<X, P>(&mut self, x: &mut X, P: &mut P) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        // The update is skipped, leaving x and P unchanged, if S cannot be inverted or
        // the innovation is gated.
        crate::observations::update_with_innovation(
            x.as_matrix_mut(),
            P.as_matrix_mut(),
            self.H.as_matrix(),
            self.R.as_matrix_mut(),
            self.y.as_matrix(),
            self.K.as_matrix_mut(),
            self.S.as_matrix_mut(),
            self.temp_S_inv.as_matrix_mut(),
            self.temp_HP.as_matrix_mut(),
            self.temp_PHt.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
            &mut self.nis,
            self.gate,
            self.noise_adaptation,
        )
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
    >
    ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
{
    /// Gets a mutable reference to Jacobian of the observation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    ///
    /// ## Extended Kalman Filters
    /// In Extended Kalman Filters, this matrix is treated as the Jacobian of the observation matrix,
    /// i.e. the derivative of the measurement function with respect to the state vector.
    #[inline(always)]
    pub fn observation_jacobian_matrix_mut(&mut self) -> &mut H {
        &mut self.H
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
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
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
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
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
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    Z: MeasurementVector<OBSERVATIONS, T>,
{
    type MeasurementVector = Z;

    #[inline(always)]
    fn measurement_vector(&self) -> &Self::MeasurementVector {
        self.measurement_vector()
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
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
{
    type MeasurementVectorMut = Z;

    #[inline(always)]
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
    > ExtendedKalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationJacobianMatrix = H;

    #[inline(always)]
    fn observation_jacobian_matrix(&self) -> &Self::ObservationTransformationJacobianMatrix {
        self.observation_jacobian_matrix()
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
    > ExtendedKalmanFilterObservationTransformationMut<STATES, OBSERVATIONS, T>
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationJacobianMatrixMut = H;

    #[inline(always)]
    fn observation_jacobian_matrix_mut(
        &mut self,
    ) -> &mut Self::ObservationTransformationJacobianMatrixMut {
        self.observation_jacobian_matrix_mut()
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
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    type MeasurementNoiseCovarianceMatrix = R;

    #[inline(always)]
    fn measurement_noise_covariance(&self) -> &Self::MeasurementNoiseCovarianceMatrix {
        self.measurement_noise_covariance()
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
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    type MeasurementNoiseCovarianceMatrixMut = R;

    #[inline(always)]
    fn measurement_noise_covariance_mut(
        &mut self,
    ) -> &mut Self::MeasurementNoiseCovarianceMatrixMut {
        self.measurement_noise_covariance_mut()
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
    > KalmanFilterNonlinearObservationCorrectFilter<STATES, OBSERVATIONS, T>
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
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
    type ObservationVector = Y;

    #[inline(always)]
    #[allow(non_snake_case)]
    fn correct_nonlinear_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]),
    {
        self.correct_nonlinear_with(x, P, observation, fix)
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
    > KalmanFilterNonlinearObservationCorrectWithGain<STATES, OBSERVATIONS, T>
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
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
    #[inline(always)]
    #[allow(non_snake_case)]
    fn correct_nonlinear_with_gain_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]),
    {
        self.correct_nonlinear_with_gain_with(x, P, observation, fix)
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
    > KalmanFilterNonlinearObservationCorrectRecalibrated<STATES, OBSERVATIONS, T>
    for ExtendedObservation<
        STATES,
        OBSERVATIONS,
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
    >
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
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
    type ObservationJacobian = H;

    #[inline(always)]
    #[allow(non_snake_case)]
    fn correct_nonlinear_recalibrated_with<X, P, F, J, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
        fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        J: FnMut(&X, &mut Self::ObservationJacobian),
        N: FnMut(&mut [T]),
    {
        self.correct_nonlinear_recalibrated_with(x, P, observation, jacobian, fix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, MatrixMut};
    use crate::test_dummies::make_dummy_observation_ekf;

    #[test]
    #[cfg(feature = "alloc")]
    fn test_mut_apply() {
        use crate::regular::builder::KalmanFilterBuilder;

        let builder = KalmanFilterBuilder::<3, f32>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<5>();

        filter.predict();
        filter.correct(&mut measurement);
    }

    fn trait_impl<const STATES: usize, const OBSERVATIONS: usize, T, M>(mut measurement: M) -> M
    where
        M: ExtendedKalmanFilterObservation<STATES, OBSERVATIONS, T>
            + ExtendedKalmanFilterObservationTransformationMut<STATES, OBSERVATIONS, T>,
    {
        assert_eq!(measurement.states(), STATES);
        assert_eq!(measurement.observations(), OBSERVATIONS);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = measurement.measurement_vector();
        let _vec = measurement.measurement_vector_mut();
        measurement
            .measurement_vector()
            .as_matrix()
            .inspect(|_vec| test_fn());
        measurement
            .measurement_vector()
            .as_matrix()
            .inspect(|_vec| test_fn_mut());
        measurement
            .measurement_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn());
        measurement
            .measurement_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn_mut());

        let _mat = measurement.observation_jacobian_matrix();
        let _mat = measurement.observation_jacobian_matrix_mut();
        let _ = measurement
            .observation_jacobian_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = measurement
            .observation_jacobian_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        measurement
            .observation_jacobian_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        measurement
            .observation_jacobian_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        let _mat = measurement.measurement_noise_covariance();
        let _mat = measurement.measurement_noise_covariance_mut();
        let _ = measurement
            .measurement_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = measurement
            .measurement_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        measurement
            .measurement_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        measurement
            .measurement_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        measurement
    }

    #[test]
    fn builder_simple() {
        let measurement = make_dummy_observation_ekf();

        let mut measurement = trait_impl(measurement);

        assert_eq!(measurement.states(), 3);
        assert_eq!(measurement.observations(), 1);

        let test_fn = || 42;

        let mut temp = 0;
        let mut test_fn_mut = || {
            temp += 0;
            42
        };

        let _vec = measurement.measurement_vector();
        let _vec = measurement.measurement_vector_mut();
        let _ = measurement
            .measurement_vector()
            .as_matrix()
            .inspect(|_vec| test_fn());
        let _ = measurement
            .measurement_vector()
            .as_matrix()
            .inspect(|_vec| test_fn_mut());
        measurement
            .measurement_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn());
        measurement
            .measurement_vector_mut()
            .as_matrix_mut()
            .apply(|_vec| test_fn_mut());

        let _mat = measurement.observation_jacobian_matrix();
        let _mat = measurement.observation_jacobian_matrix_mut();
        let _ = measurement
            .observation_jacobian_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = measurement
            .observation_jacobian_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        measurement
            .observation_jacobian_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        measurement
            .observation_jacobian_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());

        let _mat = measurement.measurement_noise_covariance();
        let _mat = measurement.measurement_noise_covariance_mut();
        let _ = measurement
            .measurement_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = measurement
            .measurement_noise_covariance()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        measurement
            .measurement_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        measurement
            .measurement_noise_covariance_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn_mut());
    }
}
