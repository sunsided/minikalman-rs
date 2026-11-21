//! # Observations for Extended Kalman Filters.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, RowMajorSequentialDataMut};
use crate::observations::{CovarianceUpdate, NoiseAdaptation};

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
            covariance_update: CovarianceUpdate::Joseph,
        }
    }
}

/// Kalman Filter measurement structure. See [`ExtendedObservationBuilder`] for construction.
///
#[doc = diagram!("extended-correct.mmd")]
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

    /// How the corrections with the Kalman gain update `P`.
    pub(crate) covariance_update: CovarianceUpdate,
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
    #[doc = diagram!("extended-correct.mmd", "gate")]
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
    #[doc = diagram!("extended-correct.mmd", "adapt")]
    ///
    /// After every correction applied by `correct_nonlinear` and `correct_nonlinear_with`,
    /// `R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)` with the residual `ε = z − H×x⁺`. See
    /// [`NoiseAdaptation`] for the method, the choice of `α` and when the estimate diverges.
    ///
    /// `correct_nonlinear_with_gain*` and `correct_nonlinear_recalibrated*` adapt `R` with the
    /// innovation form `R ← R + (1−α)×(y×yᵀ − S)` instead, because their gain is not, in
    /// general, the optimal one for the covariance they produce, which biases the residual
    /// form. For the recalibrated correction, `S` uses the Jacobian at the updated state, and
    /// strong nonlinearity inflates `R` more than in `correct_nonlinear`. A candidate that is
    /// not positive definite is discarded.
    ///
    /// `correct_nonlinear_predictively_oriented*` leaves `R` and its floor untouched, also when it
    /// returns [`CorrectionStatus::Applied`]: its inflated posterior covariance is not the error
    /// covariance of the state, so neither form gives a consistent estimate.
    ///
    /// Only corrections that return [`CorrectionStatus::Applied`] adapt `R`; gated, skipped and
    /// backed-out corrections leave it unchanged.
    #[inline(always)]
    pub fn set_noise_adaptation(&mut self, adaptation: Option<NoiseAdaptation<T>>) {
        self.noise_adaptation = adaptation;
    }

    /// Returns the covariance update set by
    /// [`set_covariance_update`](Self::set_covariance_update).
    #[inline(always)]
    pub fn covariance_update(&self) -> CovarianceUpdate {
        self.covariance_update
    }

    /// Sets how `correct_nonlinear` and `correct_nonlinear_with` update the estimate covariance `P`;
    /// [`CovarianceUpdate::Joseph`] (the default) uses the Joseph form, which keeps `P` exactly
    /// symmetric at the cost of forming `S` again and two more matrix products;
    /// [`CovarianceUpdate::Standard`] the cheaper `P ← P − K×H×P`. See [`CovarianceUpdate`] for
    /// what it does and does not fix.
    ///
    /// The external-gain, recalibrated and predictively oriented corrections ignore this setting.
    #[inline(always)]
    pub fn set_covariance_update(&mut self, update: CovarianceUpdate) {
        self.covariance_update = update;
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
            self.R.as_matrix_mut(),
            self.y.as_matrix(),
            self.K.as_matrix(),
            self.S.as_matrix_mut(),
            self.temp_S_inv.as_matrix_mut(),
            self.temp_HP.as_matrix_mut(),
            self.temp_PHt.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
            self.noise_adaptation,
        )
    }

    /// Applies a recalibrated nonlinear correction step to the provided state vector and
    /// covariance matrix.
    ///
    #[doc = diagram!("extended-recalibrated.mmd")]
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

        let status = crate::observations::recalibrate_covariance(
            x.as_matrix_mut(),
            P.as_matrix_mut(),
            self.H.as_matrix(),
            self.R.as_matrix(),
            self.K.as_matrix(),
            self.S.as_matrix_mut(),
            self.temp_HP.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
        );

        // S holds S₂ = H₂×P⁻×H₂ᵀ + R; the residual form would be biased, see NoiseAdaptation.
        if let Some(adaptation) = self.noise_adaptation.filter(|_| status.is_applied()) {
            crate::observations::adapt_measurement_noise_from_innovation(
                adaptation,
                self.R.as_matrix_mut(),
                self.y.as_matrix(),
                self.S.as_matrix(),
                self.temp_S_inv.as_matrix_mut(),
            );
        }

        status
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
            self.covariance_update,
        )
    }
}

impl<const STATES: usize, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    ExtendedObservation<STATES, 1, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrix<1, STATES, T>,
    K: KalmanGainMatrix<STATES, 1, T>,
    S: InnovationCovarianceMatrix<1, T>,
    R: MeasurementNoiseCovarianceMatrix<1, T>,
    Y: InnovationVector<1, T>,
    Z: MeasurementVector<1, T>,
    TempPHt: TemporaryPHTMatrix<STATES, 1, T>,
    TempKHP: TemporaryKHPMatrix<STATES, T>,
    T: MatrixDataType,
{
    /// Applies a predictively oriented (PrO) correction step with the observation linearized at
    /// the prior.
    ///
    /// This is the PrO update of
    /// [`RegularObservation::correct_predictively_oriented`](crate::regular::RegularObservation::correct_predictively_oriented)
    /// with the Jacobian in the observation matrix as `H` and the nonlinear innovation
    /// `y = z − h(x̃)`, i.e. the PrO counterpart of [`correct_nonlinear`](Self::correct_nonlinear).
    /// See there for the objective, its cost, and when it helps. As for the regular PrO update,
    /// it is limited to scalar observations, needs `H×P×Hᵀ > 0` and `R > 0`, updates the
    /// [NIS](Self::nis) of the prior innovation, honors the [gate](Self::set_gate), and does not
    /// adapt `R`.
    ///
    /// The Jacobian must be set up at the prior `x`. The papers relinearize `h` at the PrO mean
    /// instead, see
    /// [`correct_nonlinear_predictively_oriented_iterated`](Self::correct_nonlinear_predictively_oriented_iterated).
    ///
    /// ## Limits of the single linearization
    /// After an outlier, the PrO covariance can grow past the prior, and the next corrections
    /// then linearize `h` over a wide region. Where `h` is ambiguous, this can lock the filter
    /// onto the wrong branch: in a range-only tracking simulation with Student-t noise whose
    /// target passed the sensor (vanishing Jacobian, mirror-symmetric range), one of 20 runs
    /// flipped to the mirror track, where neither the EKF nor the relinearized variant did.
    /// Prefer the relinearized variant for strongly nonlinear or ambiguous `h`.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_predictively_oriented<X, P, F>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
    {
        self.correct_nonlinear_predictively_oriented_with(x, P, observation, |_| {}, max_iterations)
    }

    /// Applies a predictively oriented (PrO) correction step with the observation linearized at
    /// the prior, normalizing the innovation before it is applied.
    ///
    /// See [`correct_nonlinear_predictively_oriented`](Self::correct_nonlinear_predictively_oriented)
    /// and [`correct_nonlinear_with`](Self::correct_nonlinear_with).
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `fix` - Normalizes the innovation `y = z - h(x)` in place, e.g. by wrapping angles.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_predictively_oriented_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        mut observation: F,
        mut fix: N,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        N: FnMut(&mut [T]),
    {
        self.nonlinear_innovation(x, &mut observation, &mut fix);
        self.update_predictively_oriented(x, P, self.gate, max_iterations)
    }

    /// Applies a predictively oriented (PrO) correction step, relinearizing the observation at
    /// the PrO mean in iterated-EKF style.
    ///
    /// Shen, Duran-Martin and Oates ([arXiv:2606.03230](https://arxiv.org/abs/2606.03230),
    /// Section 2.4) linearize `h` at the current PrO mean `m` instead of the prior `x̃`,
    /// analogous to the iterated EKF:
    /// `h(x) ≈ h(m) + H_m×(x − m)`. The linearized problem is a regular PrO update with the
    /// Jacobian `H_m` and the prior innovation
    ///
    /// ```text
    /// v = z − h(m) − H_m×(x̃ − m),
    /// ```
    ///
    /// whose mean is the next `m`. Starting from `m = x̃` with the Jacobian set up at the prior,
    /// each of the `relinearizations` rounds moves `m` to the PrO mean of the current
    /// linearization and calls `jacobian` and `observation` there. The final linearization
    /// gives the posterior, so `P` is only written once and stays positive definite as in
    /// [`correct_nonlinear_predictively_oriented`](Self::correct_nonlinear_predictively_oriented).
    /// `relinearizations = 0` is exactly that method.
    ///
    /// ## Limits
    /// - The rounds are a fixed budget without a convergence test. Like the iterated EKF, the
    ///   relinearization is a Gauss-Newton-style fixed-point iteration that is not guaranteed to
    ///   converge for strongly nonlinear `h`. For mildly nonlinear `h` (e.g. a range far from
    ///   the sensor) a few rounds change the result little; reaching the fixed point to
    ///   working precision can take tens of rounds.
    /// - The papers alternate relinearization with an L-BFGS solve of the full objective; here
    ///   each round solves the linearized problem exactly with the scalar Newton iteration.
    /// - Gating and the [NIS](Self::nis) use the prior innovation `z − h(x̃)` and the Jacobian
    ///   at the prior. The checks `H×P×Hᵀ > 0` and `R > 0` are repeated for every
    ///   linearization; if one fails, `x` and `P` are left unchanged and
    ///   [`CorrectionStatus::NotPositiveDefinite`] is returned.
    /// - The observation Jacobian holds the one at the last linearization point on return, so
    ///   set it up again before the next correction.
    ///
    /// ## Cost
    /// Per round, one evaluation each of `jacobian` and `observation`, a `P×Hᵀ` product
    /// (`O(STATES²)`), and a scalar Newton solve; `P` is only updated once. The prior state is
    /// kept in the Kalman gain buffer `K` during the rounds.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `jacobian` - Writes the observation Jacobian at the given state.
    /// * `relinearizations` - The number of relinearization rounds.
    /// * `max_iterations` - The budget of Newton steps for each stationarity condition.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_nonlinear_predictively_oriented_iterated<X, P, F, J>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
        relinearizations: usize,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        J: FnMut(&X, &mut H),
        H: ObservationMatrixMut<1, STATES, T>,
    {
        self.correct_nonlinear_predictively_oriented_iterated_with(
            x,
            P,
            observation,
            jacobian,
            |_| {},
            relinearizations,
            max_iterations,
        )
    }

    /// Applies a predictively oriented (PrO) correction step, relinearizing the observation at
    /// the PrO mean, and normalizing every innovation before it is used.
    ///
    /// See
    /// [`correct_nonlinear_predictively_oriented_iterated`](Self::correct_nonlinear_predictively_oriented_iterated)
    /// and [`correct_nonlinear_with`](Self::correct_nonlinear_with). `fix` is applied to
    /// `z − h(m)` before the linearization term is subtracted.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `observation` - The observation function; takes the current state and provides observations.
    /// * `jacobian` - Writes the observation Jacobian at the given state.
    /// * `fix` - Normalizes the innovation `y = z - h(m)` in place, e.g. by wrapping angles.
    /// * `relinearizations` - The number of relinearization rounds.
    /// * `max_iterations` - The budget of Newton steps for each stationarity condition.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub fn correct_nonlinear_predictively_oriented_iterated_with<X, P, F, J, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        mut observation: F,
        mut jacobian: J,
        mut fix: N,
        relinearizations: usize,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Y),
        J: FnMut(&X, &mut H),
        N: FnMut(&mut [T]),
        H: ObservationMatrixMut<1, STATES, T>,
    {
        self.nonlinear_innovation(x, &mut observation, &mut fix);
        if relinearizations == 0 {
            return self.update_predictively_oriented(x, P, self.gate, max_iterations);
        }

        // The prior x̃ is kept in K until the final update overwrites it.
        self.K.as_matrix_mut().copy_from(x.as_matrix_mut());
        for round in 0..relinearizations {
            let Some((g, r, v)) = self.project_prior(P) else {
                x.as_matrix_mut().copy_from(self.K.as_matrix_mut());
                return CorrectionStatus::NotPositiveDefinite;
            };

            if round == 0 {
                self.nis = v * v / (g + r);
                if self.gate.is_some_and(|gate| self.nis > gate) {
                    return CorrectionStatus::Gated;
                }
            }

            self.move_to_pro_mean(x, g, r, v, max_iterations);
            jacobian(x, &mut self.H);
            self.nonlinear_innovation(x, &mut observation, &mut fix);
            self.linearize_innovation(x);
        }

        // The final update starts from x̃; it must neither gate again nor overwrite the NIS of
        // the prior innovation.
        x.as_matrix_mut().copy_from(self.K.as_matrix_mut());
        let nis = self.nis;
        let status = self.update_predictively_oriented(x, P, None, max_iterations);
        self.nis = nis;
        status
    }

    /// Computes the nonlinear innovation `y = z − h(x)` and normalizes it with `fix`.
    ///
    /// The innovation buffer serves as the output of `observation` first.
    fn nonlinear_innovation<X, F, N>(&mut self, x: &X, observation: &mut F, fix: &mut N)
    where
        X: StateVectorMut<STATES, T>,
        F: FnMut(&X, &mut Y),
        N: FnMut(&mut [T]),
    {
        observation(x, &mut self.y);
        let y = self.y.as_matrix_mut();
        self.z.as_matrix().sub_inplace_b(y);
        fix(y.as_mut_slice());
    }

    /// Runs the scalar PrO update of the current linearization from the innovation in `y`.
    ///
    /// See `observations::predictively_oriented::update_scalar`; `gate` overrides the
    /// observation's own gate.
    #[allow(non_snake_case)]
    fn update_predictively_oriented<X, P>(
        &mut self,
        x: &mut X,
        P: &mut P,
        gate: Option<T>,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        crate::observations::predictively_oriented::update_scalar(
            x.as_matrix_mut(),
            P.as_matrix_mut(),
            self.H.as_matrix(),
            self.R.as_matrix(),
            self.y.as_matrix(),
            self.K.as_matrix_mut(),
            self.S.as_matrix_mut(),
            self.temp_PHt.as_matrix_mut(),
            self.temp_KHP.as_matrix_mut(),
            &mut self.nis,
            gate,
            max_iterations,
        )
    }

    /// Forms `P̃×Hᵀ` in `temp_PHt` and returns `(g, r, v)` with `g = H×P̃×Hᵀ`, the noise
    /// variance `r` and the innovation `v` in `y`.
    ///
    /// Returns `None` unless both `g` and `r` are strictly positive (this also rejects NaN),
    /// as the PrO stationarity condition requires.
    #[allow(non_snake_case)]
    fn project_prior<P>(&mut self, P: &P) -> Option<(T, T, T)>
    where
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        let H = self.H.as_matrix();
        let PHt = self.temp_PHt.as_matrix_mut();
        P.as_matrix().mult_transb(H, PHt);
        let g = (0..STATES).fold(T::zero(), |sum, i| sum + H.get(0, i) * PHt.get(i, 0));
        let r = self.R.as_matrix().get(0, 0);
        let v = self.y.as_matrix().get(0, 0);

        let positive = g > T::zero() && r > T::zero();
        positive.then_some((g, r, v))
    }

    /// Moves `x` to the PrO mean `x̃ + P̃×Hᵀ × v/(c+r+g)` of the current linearization, with
    /// the prior `x̃` in `K` and `P̃×Hᵀ` in `temp_PHt` (see [`Self::project_prior`]).
    #[allow(non_snake_case)]
    fn move_to_pro_mean<X>(&mut self, x: &mut X, g: T, r: T, v: T, max_iterations: usize)
    where
        X: StateVectorMut<STATES, T>,
    {
        let c =
            crate::observations::predictively_oriented::solve_stationarity(g, r, v, max_iterations);
        let scale = v / (c + r + g);

        let x = x.as_matrix_mut();
        let prior = self.K.as_matrix();
        let PHt = self.temp_PHt.as_matrix();
        for i in 0..STATES {
            x.set(i, 0, prior.get(i, 0) + PHt.get(i, 0) * scale);
        }
    }

    /// Turns the innovation `z − h(m)` in `y` into the prior innovation of the linearization
    /// at `m = x`, `z − h(m) − H_m×(x̃ − m)`, with the prior `x̃` in `K`.
    #[allow(non_snake_case)]
    fn linearize_innovation<X>(&mut self, x: &X)
    where
        X: StateVectorMut<STATES, T>,
    {
        let x = x.as_matrix();
        let prior = self.K.as_matrix();
        let H = self.H.as_matrix();
        let offset = (0..STATES).fold(T::zero(), |sum, i| {
            sum + H.get(0, i) * (prior.get(i, 0) - x.get(i, 0))
        });

        let y = self.y.as_matrix_mut();
        y.set(0, 0, y.get(0, 0) - offset);
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

impl<const STATES: usize, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    KalmanFilterNonlinearObservationCorrectPredictivelyOriented<STATES, T>
    for ExtendedObservation<STATES, 1, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: ObservationMatrixMut<1, STATES, T>,
    K: KalmanGainMatrix<STATES, 1, T>,
    S: InnovationCovarianceMatrix<1, T>,
    R: MeasurementNoiseCovarianceMatrix<1, T>,
    Y: InnovationVector<1, T>,
    Z: MeasurementVector<1, T>,
    TempSInv: TemporaryResidualCovarianceInvertedMatrix<1, T>,
    TempHP: TemporaryHPMatrix<1, STATES, T>,
    TempPHt: TemporaryPHTMatrix<STATES, 1, T>,
    TempKHP: TemporaryKHPMatrix<STATES, T>,
    T: MatrixDataType,
{
    type ObservationJacobian = H;

    #[inline(always)]
    #[allow(non_snake_case)]
    fn correct_nonlinear_predictively_oriented_with<X, P, F, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        fix: N,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        N: FnMut(&mut [T]),
    {
        self.correct_nonlinear_predictively_oriented_with(x, P, observation, fix, max_iterations)
    }

    #[inline(always)]
    #[allow(non_snake_case)]
    fn correct_nonlinear_predictively_oriented_iterated_with<X, P, F, J, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        observation: F,
        jacobian: J,
        fix: N,
        relinearizations: usize,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        F: FnMut(&X, &mut Self::ObservationVector),
        J: FnMut(&X, &mut Self::ObservationJacobian),
        N: FnMut(&mut [T]),
    {
        self.correct_nonlinear_predictively_oriented_iterated_with(
            x,
            P,
            observation,
            jacobian,
            fix,
            relinearizations,
            max_iterations,
        )
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
