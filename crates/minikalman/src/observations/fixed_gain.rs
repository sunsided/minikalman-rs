//! # Fixed-gain observations for the Luenberger observer.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, RowMajorSequentialDataMut, SquareMatrix};
use crate::observations::normalized_innovation_squared;

/// A measurement with a fixed gain for the [`LuenbergerObserver`](crate::regular::LuenbergerObserver).
///
/// Holds only the observation matrix `H`, the measurement `z`, the innovation `y` and the gain:
/// `OBSERVATIONS·STATES·2 + OBSERVATIONS·2` scalars. There is no `R`, `S` or covariance
/// temporary, so the correction costs `2·OBSERVATIONS·STATES` multiply-adds, with no square
/// root, division or data-dependent branch.
///
/// The type parameter `F` is the [`GainForm`] of the gain: [`Current`] (the default) for a
/// Kalman-style gain `K`, e.g. from
/// [`RegularKalman::settle_gain`](crate::regular::RegularKalman::settle_gain), or [`Predictor`]
/// for a textbook observer gain `L`, e.g. from `place` or python-control `dlqe`. A
/// [`LuenbergerObserver`](crate::regular::LuenbergerObserver) of one form only accepts
/// observations of the same form, so a gain cannot run in the wrong recursion; see
/// [the gain convention trap](crate::regular::LuenbergerObserver#the-gain-convention-trap).
///
/// ## Innovation gate
///
/// The last type parameter `G` is the optional gate. By default it is [`NoGate`], which costs
/// nothing. [`with_gate`](Self::with_gate) adds a [`FixedGainGate`]: a stored steady-state
/// `S∞⁻¹` (`OBSERVATIONS²` scalars) that rejects outliers by their normalized innovation
/// squared `NIS = yᵀ·S∞⁻¹·y`, for `OBSERVATIONS² + OBSERVATIONS` more multiply-adds per
/// correction. See [`set_gate`](Self::set_gate) for when this matches the Kalman filter's gate
/// and when it does not.
#[allow(non_snake_case)]
pub struct FixedGainObservation<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
    H,
    Z,
    Y,
    K,
    F = Current,
    G = NoGate,
> {
    /// Observation matrix `H` (`OBSERVATIONS × STATES`).
    H: H,

    /// Measurement vector `z` (`OBSERVATIONS × 1`).
    z: Z,

    /// Innovation vector `y = z − H·x̂` of the last correction (`OBSERVATIONS × 1`).
    y: Y,

    /// The fixed gain (`STATES × OBSERVATIONS`): `K` or `L`, as the form `F` says.
    K: K,

    /// The innovation gate: [`NoGate`] or a [`FixedGainGate`].
    gate: G,

    /// Marks the scalar type and the [`GainForm`].
    _phantom: PhantomData<(T, F)>,
}

/// The absent innovation gate of a [`FixedGainObservation`]: every correction is applied.
///
/// This is the default, so an observation without a gate stores no `S∞⁻¹`.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct NoGate;

/// The innovation gate of a [`FixedGainObservation`]: a stored `S∞⁻¹`, the threshold and the
/// last normalized innovation squared.
///
/// Add it with [`FixedGainObservation::with_gate`]; its methods are on the observation.
#[allow(non_snake_case)]
pub struct FixedGainGate<const OBSERVATIONS: usize, T, SInv> {
    /// The inverse steady-state innovation covariance `S∞⁻¹` (`OBSERVATIONS × OBSERVATIONS`).
    S_inv: SInv,

    /// The NIS threshold; `None` computes the NIS without rejecting anything.
    threshold: Option<T>,

    /// `yᵀ·S∞⁻¹·y` of the last correction; `0` before the first one.
    nis: T,
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F>
    FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, NoGate>
where
    F: GainForm,
{
    /// Creates a fixed-gain observation of the [`GainForm`] `F` from its buffers.
    ///
    /// `K` is taken to hold a gain in the form `F`; set one with [`set_gain`](Self::set_gain).
    ///
    /// ## Arguments
    /// * `H` - The observation matrix (`OBSERVATIONS` × `STATES`).
    /// * `z` - The measurement vector (`OBSERVATIONS` × `1`).
    /// * `y` - The innovation vector (`OBSERVATIONS` × `1`).
    /// * `K` - The gain (`STATES` × `OBSERVATIONS`) in the form `F`.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// use minikalman::prelude::*;
    /// use minikalman::regular::FixedGainObservation;
    ///
    /// impl_buffer_H!(mut H, 1, 2, f32, 0.0);
    /// impl_buffer_z!(mut z, 1, f32, 0.0);
    /// impl_buffer_y!(mut y, 1, f32, 0.0);
    /// impl_buffer_K!(mut K, 2, 1, f32, 0.0);
    ///
    /// let mut position = FixedGainObservation::<2, 1, f32, _, _, _, _, Current>::new(H, z, y, K);
    /// position.observation_matrix_mut().set_at(0, 0, 1.0);
    /// position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.181201, 0.180975]));
    /// ```
    #[allow(non_snake_case)]
    pub fn new(H: H, z: Z, y: Y, K: K) -> Self {
        Self {
            H,
            z,
            y,
            K,
            gate: NoGate,
            _phantom: PhantomData,
        }
    }

    /// Adds an innovation gate with the `S∞⁻¹` buffer `S_inv`, taking its current contents
    /// as `S∞⁻¹`.
    ///
    /// The gate starts disabled (threshold `None`) with a NIS of `0`. Fill the buffer with
    /// [`set_innovation_covariance`](FixedGainObservation::set_innovation_covariance) or
    /// [`set_innovation_covariance_inverse`](FixedGainObservation::set_innovation_covariance_inverse),
    /// then set a threshold with [`set_gate`](FixedGainObservation::set_gate). A zeroed buffer
    /// gives a NIS of `0`, which passes every gate.
    ///
    /// ## Arguments
    /// * `S_inv` - The buffer for `S∞⁻¹` (`OBSERVATIONS` × `OBSERVATIONS`).
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::FixedGainObservation;
    ///
    /// impl_buffer_H!(mut H, 1, 2, f32, 0.0);
    /// impl_buffer_z!(mut z, 1, f32, 0.0);
    /// impl_buffer_y!(mut y, 1, f32, 0.0);
    /// impl_buffer_K!(mut K, 2, 1, f32, 0.0);
    /// impl_buffer_S!(mut S_inv, 1, f32, 0.0);
    ///
    /// let mut position =
    ///     FixedGainObservation::<2, 1, f32, _, _, _, _, Current>::new(H, z, y, K).with_gate(S_inv);
    /// position.set_innovation_covariance_inverse(&MatrixData::new_array::<1, 1, 1, f32>([1.0 / 0.3053]));
    /// position.set_gate(Some(6.63));
    /// # }
    /// ```
    #[allow(non_snake_case)]
    pub fn with_gate<SInv>(
        self,
        S_inv: SInv,
    ) -> FixedGainObservation<
        STATES,
        OBSERVATIONS,
        T,
        H,
        Z,
        Y,
        K,
        F,
        FixedGainGate<OBSERVATIONS, T, SInv>,
    >
    where
        T: MatrixDataType,
    {
        FixedGainObservation {
            H: self.H,
            z: self.z,
            y: self.y,
            K: self.K,
            gate: FixedGainGate {
                S_inv,
                threshold: None,
                nis: T::zero(),
            },
            _phantom: PhantomData,
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
{
    /// Returns the number of states.
    #[inline(always)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Returns the number of measurements.
    #[inline(always)]
    pub const fn measurements(&self) -> usize {
        OBSERVATIONS
    }

    /// Gets a reference to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector(&self) -> &Z {
        &self.z
    }

    /// Gets a mutable reference to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_mut(&mut self) -> &mut Z {
        &mut self.z
    }

    /// Gets a reference to the observation matrix H.
    #[inline(always)]
    pub fn observation_matrix(&self) -> &H {
        &self.H
    }

    /// Gets a mutable reference to the observation matrix H.
    #[inline(always)]
    pub fn observation_matrix_mut(&mut self) -> &mut H {
        &mut self.H
    }

    /// Gets a reference to the innovation vector `y = z − H·x̂` of the last correction.
    #[inline(always)]
    pub fn innovation_vector(&self) -> &Y {
        &self.y
    }

    /// Gets a reference to the stored gain: a current-form `K` or a predictor-form `L`, as the
    /// type parameter `F` says.
    #[inline(always)]
    pub fn gain(&self) -> &K {
        &self.K
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
where
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    T: Copy,
{
    /// Stores the gain, written in the [`GainForm`] `F` of this observation.
    ///
    /// * For [`Current`], this is `K` of `x̂⁺ = x̂⁻ + K·(z − H·x̂⁻)`, the crate's convention and
    ///   the Kalman gain: copy it from
    ///   [`RegularObservation::kalman_gain`](crate::regular::RegularObservation::kalman_gain)
    ///   after [`RegularKalman::settle_gain`](crate::regular::RegularKalman::settle_gain), or
    ///   from MATLAB `kalman`'s `Mx`. The error dynamics are `(I − K·H)·A`. A pole placement
    ///   gives `K` directly when designed on the pair `(A, H·A)`, e.g.
    ///   `K = place(A.T, (H @ A).T, poles).T`, because `(I − K·H)·A = A − K·(H·A)`.
    /// * For [`Predictor`], this is `L` of `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)`, which
    ///   `place(A', C')'`, `acker` and python-control `dlqe` return. The error dynamics are
    ///   `A − L·H`, the same poles as the current form with `K` where `L = A·K`.
    ///
    /// ## Arguments
    /// * `gain` - The gain (`STATES` × `OBSERVATIONS`).
    pub fn set_gain<M>(&mut self, gain: &M)
    where
        M: Matrix<STATES, OBSERVATIONS, T>,
    {
        gain.copy_to(self.K.as_matrix_mut());
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, SInv>
    FixedGainObservation<
        STATES,
        OBSERVATIONS,
        T,
        H,
        Z,
        Y,
        K,
        F,
        FixedGainGate<OBSERVATIONS, T, SInv>,
    >
where
    SInv: InnovationCovarianceMatrix<OBSERVATIONS, T>,
    T: MatrixDataType,
{
    /// Returns the normalized innovation squared `NIS = yᵀ·S∞⁻¹·y` of the last correction.
    ///
    /// Every correction updates it, including one the [gate](Self::set_gate) rejects. It is
    /// `0` before the first correction. Near steady state it equals the NIS of
    /// [`RegularObservation::nis`](crate::regular::RegularObservation::nis) on the same data, so
    /// it averages to `OBSERVATIONS` for a consistent design.
    #[inline(always)]
    pub fn nis(&self) -> T {
        self.gate.nis
    }

    /// Returns the threshold set by [`set_gate`](Self::set_gate).
    #[inline(always)]
    pub fn gate(&self) -> Option<T> {
        self.gate.threshold
    }

    /// Sets the innovation gate: a correction whose [NIS](Self::nis) exceeds `threshold` is
    /// skipped with [`CorrectionStatus::Gated`] and leaves the state unchanged.
    ///
    /// The threshold has the convention of
    /// [`RegularObservation::set_gate`](crate::regular::RegularObservation::set_gate): a
    /// chi-square quantile for `OBSERVATIONS` degrees of freedom, e.g. `6.63` (one measurement)
    /// or `9.21` (two measurements) to reject about 1% of the measurements of a consistent
    /// filter. `None` (the default) disables the gate; the NIS is still computed.
    ///
    /// In the [`Predictor`] form, a gated correction leaves no pending correction behind, so the
    /// next [`predict`](crate::regular::LuenbergerObserver::predict) is a pure model step.
    ///
    /// ## Start-up and lock-out
    ///
    /// `S∞` is the innovation covariance of the *settled* filter. While the estimate is still
    /// converging, e.g. after a wrong initial `x̂`, the true innovations are larger than `S∞`
    /// says, so the gate rejects good measurements. Unlike a Kalman filter, whose `S` grows
    /// with every skipped correction until measurements pass again, a fixed `S∞` never grows:
    /// once the estimate has drifted away, the gate can reject every measurement from then on.
    /// Guard against this by
    ///
    /// * seeding `x̂` from the first measurement and keeping the gate off (`None`) for the first
    ///   steps, about the settling time of the error dynamics `(I − K·H)·A`,
    /// * a larger threshold during start-up, and
    /// * disabling the gate, or re-seeding `x̂`, after several consecutive rejections.
    ///
    /// The gate compares the NIS after it has been computed in `T`, so it does not protect
    /// against overflow: with fixed-point types, an outlier large enough to overflow
    /// `yᵀ·S∞⁻¹·y` panics in debug builds and wraps in release builds, where the wrapped value
    /// may pass the gate. Choose a type with enough integer bits for the expected outliers.
    #[inline(always)]
    pub fn set_gate(&mut self, threshold: Option<T>) {
        self.gate.threshold = threshold;
    }

    /// Gets a reference to the stored inverse innovation covariance `S∞⁻¹`.
    #[inline(always)]
    pub fn innovation_covariance_inverse(&self) -> &SInv {
        &self.gate.S_inv
    }

    /// Stores a precomputed inverse innovation covariance `S∞⁻¹`, e.g. one designed offline
    /// together with the gain.
    ///
    /// ## Arguments
    /// * `S_inv` - The symmetric positive definite `S∞⁻¹` (`OBSERVATIONS` × `OBSERVATIONS`).
    #[allow(non_snake_case)]
    pub fn set_innovation_covariance_inverse<M>(&mut self, S_inv: &M)
    where
        M: Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    {
        S_inv.copy_to(self.gate.S_inv.as_matrix_mut());
    }

    /// Inverts the steady-state innovation covariance `S∞` with a Cholesky factorization and
    /// stores `S∞⁻¹`.
    ///
    /// After [`RegularKalman::settle_gain`](crate::regular::RegularKalman::settle_gain), the
    /// design observation's
    /// [`innovation_covariance_mut`](crate::regular::RegularObservation::innovation_covariance_mut)
    /// holds the unfactored `S∞ = H·P⁻∞·Hᵀ + R`; pass it here. Its temporary `S⁻¹` is not
    /// valid after the loop, since it may share memory with other temporaries.
    ///
    /// `S` is factored in place: afterwards, its lower triangle holds the Cholesky factor of
    /// `S∞`. Costs `OBSERVATIONS` square roots and reciprocals, once.
    ///
    /// ## Arguments
    /// * `S` - The symmetric `S∞` (`OBSERVATIONS` × `OBSERVATIONS`); replaced by its factor.
    ///
    /// ## Errors
    /// Returns [`CorrectionStatus::NotPositiveDefinite`] if `S` is not positive definite; the
    /// stored `S∞⁻¹` is then unchanged, and `S` holds intermediate values.
    ///
    /// ## Example
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// use minikalman::models::motion;
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f32>::default();
    /// let model = motion::constant_velocity(0.1, 1.0);
    ///
    /// // Design K∞ and S∞ with Q and R.
    /// let mut design = builder.build();
    /// let mut position = builder.observations().build::<1>();
    /// design.state_transition_mut().apply(|a| model.fill_transition(a));
    /// design.direct_process_noise_mut().apply(|q| model.fill_process_noise(q));
    /// design.estimate_covariance_mut().make_identity();
    /// position.observation_matrix_mut().set_at(0, 0, 1.0);
    /// position.measurement_noise_covariance_mut().set_at(0, 0, 0.25);
    /// design.settle_gain(&mut position, 1e-6, 10_000).expect("detectable model");
    ///
    /// // Copy K∞, invert S∞ ≈ 0.3053, and gate at the 99% chi-square quantile.
    /// let mut observer = builder.build_observer();
    /// let mut fixed = builder.observations().build_fixed_gain_gated::<1, Current>();
    /// observer.state_transition_mut().apply(|a| model.fill_transition(a));
    /// fixed.observation_matrix_mut().set_at(0, 0, 1.0);
    /// fixed.set_gain(position.kalman_gain());
    /// fixed
    ///     .set_innovation_covariance(position.innovation_covariance_mut().as_matrix_mut())
    ///     .expect("S∞ is positive definite");
    /// fixed.set_gate(Some(6.63));
    ///
    /// // A 5 m jump has a NIS of about 82 and is rejected.
    /// observer.predict();
    /// fixed.measurement_vector_mut().set_at(0, 0, 5.0);
    /// assert_eq!(observer.correct(&mut fixed), CorrectionStatus::Gated);
    /// assert!(fixed.nis() > 80.0);
    /// assert_eq!(observer.state_vector().get_at(0, 0), 0.0);
    /// # }
    /// ```
    #[allow(non_snake_case)]
    pub fn set_innovation_covariance<M>(&mut self, S: &mut M) -> Result<(), CorrectionStatus>
    where
        M: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    {
        if !S.cholesky_decompose_lower() {
            return Err(CorrectionStatus::NotPositiveDefinite);
        }

        S.invert_l_cholesky(self.gate.S_inv.as_matrix_mut());
        Ok(())
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
    Z: MeasurementVector<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    T: MatrixDataType,
{
    /// Forms `y = z − H·x` and normalizes it in place with `fix`; shared by both gate kinds.
    fn form_innovation<X, N>(&mut self, x: &X, mut fix: N)
    where
        X: Matrix<STATES, 1, T>,
        N: FnMut(&mut [T]),
    {
        let y = self.y.as_matrix_mut();
        self.H.as_matrix().mult_rowvector(x, y);
        self.z.as_matrix().sub_inplace_b(y);
        fix(y.as_mut_slice());
    }

    /// Adds `K·y` to `target`; shared by both gate kinds.
    #[inline(always)]
    fn add_gain<V>(&self, target: &mut V)
    where
        V: MatrixMut<STATES, 1, T>,
    {
        self.K
            .as_matrix()
            .multadd_rowvector(self.y.as_matrix(), target);
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F>
    KalmanFilterObservationCorrectFixedGain<STATES, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, NoGate>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
    Z: MeasurementVector<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    F: GainForm,
    T: MatrixDataType,
{
    type Form = F;

    #[inline(always)]
    fn compute_innovation<X, N>(&mut self, x: &X, fix: N)
    where
        X: Matrix<STATES, 1, T>,
        N: FnMut(&mut [T]),
    {
        self.form_innovation(x, fix);
    }

    #[inline(always)]
    fn add_gain_times_innovation<V>(&self, target: &mut V)
    where
        V: MatrixMut<STATES, 1, T>,
    {
        self.add_gain(target);
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, SInv>
    KalmanFilterObservationCorrectFixedGain<STATES, T>
    for FixedGainObservation<
        STATES,
        OBSERVATIONS,
        T,
        H,
        Z,
        Y,
        K,
        F,
        FixedGainGate<OBSERVATIONS, T, SInv>,
    >
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
    Z: MeasurementVector<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    SInv: InnovationCovarianceMatrix<OBSERVATIONS, T>,
    F: GainForm,
    T: MatrixDataType,
{
    type Form = F;

    #[inline(always)]
    fn compute_innovation<X, N>(&mut self, x: &X, fix: N)
    where
        X: Matrix<STATES, 1, T>,
        N: FnMut(&mut [T]),
    {
        self.form_innovation(x, fix);
    }

    fn is_gated(&mut self) -> bool {
        let gate = &mut self.gate;
        gate.nis = normalized_innovation_squared(self.y.as_matrix(), gate.S_inv.as_matrix());
        gate.threshold.is_some_and(|threshold| gate.nis > threshold)
    }

    #[inline(always)]
    fn add_gain_times_innovation<V>(&self, target: &mut V)
    where
        V: MatrixMut<STATES, 1, T>,
    {
        self.add_gain(target);
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    KalmanFilterNumStates<STATES>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    KalmanFilterNumObservations<OBSERVATIONS>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    KalmanFilterMeasurementVector<OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
where
    Z: MeasurementVector<OBSERVATIONS, T>,
{
    type MeasurementVector = Z;

    #[inline(always)]
    fn measurement_vector(&self) -> &Self::MeasurementVector {
        self.measurement_vector()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
where
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
{
    type MeasurementVectorMut = Z;

    #[inline(always)]
    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut {
        self.measurement_vector_mut()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationMatrix = H;

    #[inline(always)]
    fn observation_matrix(&self) -> &Self::ObservationTransformationMatrix {
        self.observation_matrix()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K, F, G>
    KalmanFilterObservationTransformationMut<STATES, OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K, F, G>
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationMatrixMut = H;

    #[inline(always)]
    fn observation_matrix_mut(&mut self) -> &mut Self::ObservationTransformationMatrixMut {
        self.observation_matrix_mut()
    }
}
