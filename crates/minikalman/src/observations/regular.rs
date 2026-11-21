//! # Observations for Regular Kalman Filters.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, RowMajorSequentialDataMut};
use crate::observations::NoiseAdaptation;

/// A builder for a Kalman filter [`RegularObservation`] instances.
#[allow(clippy::type_complexity)]
pub struct RegularObservationBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP> {
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
    RegularObservationBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
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
    /// use minikalman::regular::RegularObservationBuilder;
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
    /// let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
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
    /// See also [`KalmanBuilder::new`](regular::RegularKalmanBuilder::new) for setting up the Kalman filter itself.
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
    ) -> RegularObservation<
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
        RegularObservation::<STATES, OBSERVATIONS, T, _, _, _, _, _, _, _, _, _, _> {
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

/// Kalman Filter measurement structure. See [`RegularObservationBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct RegularObservation<
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
    RegularObservation<
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
    /// Corrections with an external gain ([`correct_with_gain`](Self::correct_with_gain)) do not invert `S` and leave
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
    /// After every correction applied by `correct` and `correct_with`,
    /// `R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)` with the residual `ε = z − H×x⁺`. See
    /// [`NoiseAdaptation`] for the method, the choice of `α` and when the estimate diverges.
    ///
    /// `correct_with_gain*` adapts `R` with the innovation form `R ← R + (1−α)×(y×yᵀ − S)`
    /// instead, because the residual form is biased for any gain but the optimal one. A
    /// candidate that is not positive definite is discarded. These corrections have no gate,
    /// so each of them adapts `R`.
    ///
    /// `correct_predictively_oriented` leaves `R` and its floor untouched, also when it returns
    /// [`CorrectionStatus::Applied`]: its inflated posterior covariance is not the error
    /// covariance of the state, so neither form gives a consistent estimate.
    ///
    /// Gated and skipped corrections leave `R` unchanged.
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

    /// Gets a reference to the measurement transformation matrix H.
    ///
    /// This matrix maps the state vector into the measurement space, relating the state of the
    /// system to the observations or measurements. It defines how each state component contributes
    /// to the measurement.
    #[inline(always)]
    pub fn observation_matrix(&self) -> &H {
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
    /// Set the gain here before calling [`correct_with_gain`](Self::correct_with_gain). Note that a regular
    /// correction overwrites K with the gain it computes.
    #[inline(always)]
    pub fn kalman_gain_mut(&mut self) -> &mut K {
        &mut self.K
    }

    /// Gets a reference to the innovation vector y.
    ///
    /// Every correction writes the innovation `z − H×x` of the prior state here (after the
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
    RegularObservation<
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
    /// Applies a correction step to the provided state vector and covariance matrix.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct<X, P>(&mut self, x: &mut X, P: &mut P) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.correct_with(x, P, |_| {})
    }

    /// Applies a correction step to the provided state vector and covariance matrix,
    /// normalizing the innovation before it is applied.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `fix` - Normalizes the innovation `y = z - H×x` in place, e.g. by wrapping angles.
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_with<X, P, N>(&mut self, x: &mut X, P: &mut P, mut fix: N) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        N: FnMut(&mut [T]),
    {
        // matrices and vectors
        let H = self.H.as_matrix();
        let y = self.y.as_matrix_mut();
        let z = self.z.as_matrix();

        // Calculate innovation and residual covariance
        // y = z - H*x
        H.mult_rowvector(x.as_matrix(), y);
        z.sub_inplace_b(y);
        fix(y.as_mut_slice());

        // Perform the remaining update.
        self.update_with_innovation(x, P)
    }

    /// Applies a correction step with the externally supplied Kalman gain stored in K.
    ///
    /// Set the gain with [`kalman_gain_mut`](Self::kalman_gain_mut) first, e.g. a precomputed
    /// steady-state gain. `P` is updated with the Joseph form, so it stays symmetric and
    /// positive semi-definite for any gain. `S` is not inverted: the [gate](Self::set_gate) is
    /// not evaluated and the [NIS](Self::nis) is not updated.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_with_gain<X, P>(&mut self, x: &mut X, P: &mut P) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.correct_with_gain_with(x, P, |_| {})
    }

    /// Applies a correction step with the externally supplied Kalman gain stored in K,
    /// normalizing the innovation before it is applied.
    ///
    /// See [`correct_with_gain`](Self::correct_with_gain).
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `fix` - Normalizes the innovation `y = z - H×x` in place, e.g. by wrapping angles.
    ///
    /// Always returns [`CorrectionStatus::Applied`].
    #[inline]
    #[allow(non_snake_case)]
    pub fn correct_with_gain_with<X, P, N>(
        &mut self,
        x: &mut X,
        P: &mut P,
        mut fix: N,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        N: FnMut(&mut [T]),
    {
        // y = z - H*x
        let y = self.y.as_matrix_mut();
        self.H.as_matrix().mult_rowvector(x.as_matrix(), y);
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
    RegularObservation<
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
}

impl<const STATES: usize, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    RegularObservation<STATES, 1, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
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
    /// Applies a predictively oriented (PrO) correction step to the provided state vector and
    /// covariance matrix.
    ///
    /// Instead of the Bayesian posterior, the PrO update (Shen, Duran-Martin, Oates,
    /// [arXiv:2606.03230](https://arxiv.org/abs/2606.03230); unscented variant: Kean, Shen,
    /// Oates, [arXiv:2610.14773](https://arxiv.org/abs/2610.14773)) picks the Gaussian
    /// `q = N(x, P)` that minimizes
    ///
    /// ```text
    /// −log ∫ p(z|x) q(x) dx + KL(q ‖ N(x̃, P̃))
    /// ```
    ///
    /// i.e. the one that best predicts the measurement while staying close to the prior.
    /// The result is still Kalman-shaped, but the covariance is inflated in the direction of
    /// `H` whenever the innovation is larger than predicted; `P` can even grow past the prior.
    /// Large outliers therefore move the state much less than in [`correct`](Self::correct).
    ///
    /// ## When to use it
    /// - **Helps** under heavy-tailed measurement noise or outliers. In a 1-D tracking
    ///   simulation with Student-t noise (2.01 degrees of freedom) the mean squared error
    ///   drops by a factor of 3 or more compared to [`correct`](Self::correct).
    /// - **Costs** under truly Gaussian noise: the same simulation shows about 8% higher mean
    ///   squared error, because the update is more conservative than the optimal Kalman update.
    ///   Even with a zero innovation, the covariance shrinks less than in a Kalman update.
    ///
    /// ## Cost
    /// One Kalman-sized update plus a scalar Newton iteration for the stationarity condition,
    /// typically 3 to 6 steps (up to about 15 for badly scaled problems with `H×P×Hᵀ` and `R`
    /// eight orders of magnitude apart). No buffers are needed beyond the observation's own.
    /// The iteration converges monotonically from below, so every iterate is a valid posterior;
    /// if `max_iterations` runs out (`0` skips the iteration entirely), the covariance is
    /// inflated less than the exact PrO solution. A budget of 20 is ample for `f32` and `f64`.
    /// With `micromath`, whose square root is approximate, the start point can be poorer and
    /// badly scaled problems may need a larger budget.
    ///
    /// ## Scalar observations only
    /// This method exists for observations with a single measurement. Applying it row by row
    /// to a vector observation is **not** the joint PrO update: the update does not compose
    /// over measurements.
    ///
    /// ## Status
    /// Like [`correct`](Self::correct), this updates the [NIS](Self::nis) (of the prior
    /// innovation) and honors the [gate](Self::set_gate). Unlike [`correct`](Self::correct),
    /// which only needs `H×P×Hᵀ + R > 0`, it requires both `H×P×Hᵀ > 0` and `R > 0`, since
    /// the PrO objective needs measurement noise and an informative measurement. Otherwise it
    /// returns [`CorrectionStatus::NotPositiveDefinite`] and leaves `x` and `P` unchanged.
    ///
    /// With fixed-point types, large innovations can overflow intermediate values; choose a
    /// type with enough integer bits for the expected outliers.
    ///
    /// ## Arguments
    /// * `x` - The current state vector.
    /// * `P` - The current state estimate covariance matrix.
    /// * `max_iterations` - The budget of Newton steps for the stationarity condition.
    #[allow(non_snake_case)]
    pub fn correct_predictively_oriented<X, P>(
        &mut self,
        x: &mut X,
        P: &mut P,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        // y = z - H*x
        let y = self.y.as_matrix_mut();
        self.H.as_matrix().mult_rowvector(x.as_matrix(), y);
        self.z.as_matrix().sub_inplace_b(y);

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
            self.gate,
            max_iterations,
        )
    }
}

impl<const STATES: usize, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    KalmanFilterObservationCorrectPredictivelyOriented<STATES, T>
    for RegularObservation<STATES, 1, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
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
    #[inline(always)]
    #[allow(non_snake_case)]
    fn correct_predictively_oriented<X, P>(
        &mut self,
        x: &mut X,
        P: &mut P,
        max_iterations: usize,
    ) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.correct_predictively_oriented(x, P, max_iterations)
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
    for RegularObservation<
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
    for RegularObservation<
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
    for RegularObservation<
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
    for RegularObservation<
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
    > KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    for RegularObservation<
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
    type ObservationTransformationMatrix = H;

    #[inline(always)]
    fn observation_matrix(&self) -> &Self::ObservationTransformationMatrix {
        self.observation_matrix()
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
    for RegularObservation<
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
    type ObservationTransformationMatrixMut = H;

    #[inline(always)]
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
    for RegularObservation<
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
    for RegularObservation<
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
    > KalmanFilterObservationCorrectFilter<STATES, T>
    for RegularObservation<
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
    fn correct_with<X, P, N>(&mut self, x: &mut X, P: &mut P, fix: N) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        N: FnMut(&mut [T]),
    {
        self.correct_with(x, P, fix)
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
    > KalmanFilterObservationCorrectWithGain<STATES, T>
    for RegularObservation<
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
    fn correct_with_gain_with<X, P, N>(&mut self, x: &mut X, P: &mut P, fix: N) -> CorrectionStatus
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        N: FnMut(&mut [T]),
    {
        self.correct_with_gain_with(x, P, fix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, MatrixMut};
    use crate::test_dummies::make_dummy_observation;

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

        let _mat = measurement.observation_matrix();
        let _mat = measurement.observation_matrix_mut();
        let _ = measurement
            .observation_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = measurement
            .observation_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        measurement
            .observation_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        measurement
            .observation_matrix_mut()
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
        let measurement = make_dummy_observation();

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

        let _mat = measurement.observation_matrix();
        let _mat = measurement.observation_matrix_mut();
        let _ = measurement
            .observation_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn());
        let _ = measurement
            .observation_matrix()
            .as_matrix()
            .inspect(|_mat| test_fn_mut());
        measurement
            .observation_matrix_mut()
            .as_matrix_mut()
            .apply(|_mat| test_fn());
        measurement
            .observation_matrix_mut()
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
