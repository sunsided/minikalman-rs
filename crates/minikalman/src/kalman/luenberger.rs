//! # Fixed-gain (Luenberger) state observer

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType};
use core::marker::PhantomData;

/// A fixed-gain (Luenberger) state observer: the steady-state Kalman filter without a
/// covariance.
///
/// It holds only the state transition `A`, the state `x̂` and one temporary vector, `STATES² +
/// 2·STATES` scalars. There is no `P`, `Q` or `R`; the gain lives in a
/// [`FixedGainObservation`](crate::regular::FixedGainObservation).
///
/// ## Tune trust, or tune dynamics
///
/// A Kalman filter and a Luenberger observer run the same loop: predict with the model, then
/// correct with a gain times the innovation. They differ in where the gain comes from.
///
/// * **Kalman: tune trust.** `Q` and `R` say how far the model and the sensor are trusted,
///   and the filter computes the optimal gain at every step. It carries `P` for that.
/// * **Luenberger: tune dynamics.** The gain is fixed and sets how fast the estimation error
///   dies out, i.e. the poles of the error dynamics `(I − K·H)·A`.
///
/// They meet in the middle: on a time-invariant model, the Kalman gain converges to a constant
/// `K∞`, and a Luenberger observer with `K∞` *is* the steady-state Kalman filter.
/// [`RegularKalman::settle_gain`](crate::regular::RegularKalman::settle_gain) computes it from
/// `Q` and `R`, so the design can stay in terms of trust while the running estimator is cheap.
///
#[doc = diagram!("luenberger-loop.mmd")]
///
/// Use a fixed gain when the model and the sample rate are fixed, the start-up transient does
/// not matter (or a few Kalman steps can be afforded before switching), and the CPU, RAM or
/// fixed-point budget is tight. Keep the Kalman filter when `dt` varies, sensors drop out
/// irregularly, `Q` or `R` change, or `P` is needed for smoothing or fusion decisions. A
/// fixed gain is optimal only for the measurement schedule it was designed for. Outliers can
/// still be gated with the stored steady-state `S∞⁻¹` of a
/// [`FixedGainGate`](crate::regular::FixedGainGate), which matches the Kalman filter's gate
/// near steady state only; see
/// [`set_gate`](crate::regular::FixedGainObservation::set_gate).
///
/// ## Cost
///
/// Per step with `n` states and `m` measurements: `n²` multiply-adds to predict and `2·n·m`
/// to correct, with no square root, no division and no data-dependent branch, so every step
/// takes the same time. An innovation [gate](crate::regular::FixedGainGate) adds `m² + m`
/// multiply-adds and one comparison, and a rejected step skips the gain (and, in the predictor
/// form, the time update), so it is shorter. The regular filter adds `2·n³` for `P` in the prediction, and a
/// Cholesky factorization with `m` square roots and reciprocals, the gain and the covariance
/// update in the correction.
///
/// ## Designing the gain
///
/// * **From `Q` and `R`**, on the host or at boot: build a [`RegularKalman`](crate::regular::RegularKalman) and a
///   [`RegularObservation`](crate::regular::RegularObservation) with the same `A` and `H`,
///   call [`settle_gain`](crate::regular::RegularKalman::settle_gain) and copy the observation's
///   [`kalman_gain`](crate::regular::RegularObservation::kalman_gain) into
///   [`set_gain_current`](crate::regular::FixedGainObservation::set_gain_current). The
///   covariance buffers can be dropped afterwards. See the example below.
/// * **Offline**, with SciPy, python-control or MATLAB, pasted in as constants. Mind the
///   convention below.
/// * **From continuous-time poles** `s`: the discrete poles are `z = e^{s·dt}`.
///
/// The reverse direction, from poles or a gain back to a `Q` and `R`, has no unique answer
/// (`c·Q, c·R` give the same gain for any `c > 0`) and often none at all. Pole placement
/// (Ackermann, `place`) is not part of the crate: it needs linear algebra the crate does not
/// have, and the offline tools do it well.
///
/// ## The gain convention trap
///
/// Observer gains come in two forms (see [`GainForm`]), and both are common:
///
/// | Form | Recursion | Calls per step | Setter |
/// |---|---|---|---|
/// | current | `x̂⁻ = A·x̂⁺ + B·u`, `x̂⁺ = x̂⁻ + K·(z − H·x̂⁻)` | [`predict`](Self::predict), [`control`](Self::control), [`correct`](Self::correct) | [`set_gain_current`](crate::regular::FixedGainObservation::set_gain_current) |
/// | predictor | `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)` | [`predict_from_measurement`](Self::predict_from_measurement), then [`control`](Self::control) | [`set_gain_predictor`](crate::regular::FixedGainObservation::set_gain_predictor) |
///
/// In both forms, a step without a measurement is a plain [`predict`](Self::predict). The
/// predictor step already contains `A·x̂`, so never call `predict` in the same step; it takes
/// one observation per step (stack several sensors into one).
///
/// The two gains are related by `L = A·K` and give the same poles, but **they are not
/// interchangeable**. `place(A', C')'`, `acker` and python-control `dlqe` return `L`; this
/// crate's [`settle_gain`](crate::regular::RegularKalman::settle_gain) and MATLAB `kalman`'s `Mx` return `K`.
/// The observer refuses a gain in the other form with
/// [`CorrectionStatus::GainFormMismatch`] instead of silently running the wrong dynamics.
///
/// A worked example with the constant-velocity model `A = [[1, 0.1], [0, 1]]`, `H = [1, 0]`:
/// placing both poles at `0` (deadbeat) with `place` gives the predictor gain `L = [2, 10]ᵀ`,
/// with `(A − L·H)² = 0`. Used as a current-form gain instead, `(I − L·H)·A` has the
/// characteristic polynomial `λ² + λ − 1`, i.e. the poles `0.618` and `−1.618`: the observer
/// diverges. The matching current-form gain is `K = A⁻¹·L = [1, 10]ᵀ` ("the position is the
/// measurement, the velocity its finite difference"). For a singular `A`, `A⁻¹·L` does not
/// exist; place the poles on the pair `(A, H·A)` instead, which gives `K` directly, since
/// `(I − K·H)·A = A − K·(H·A)`: `K = place(A.T, (H @ A).T, poles).T`.
///
/// ## Fixed point
///
/// Compute the gain in `f32` or `f64` and convert it, e.g. with `I16F16::from_num`; do not
/// iterate [`settle_gain`](crate::regular::RegularKalman::settle_gain) in `I16F16`, whose small `Q` entries
/// round to a few LSB. The observer itself runs well in `I16F16`: each rounding error is
/// filtered by the error dynamics, so the state stays within about `LSB / (1 − ρ)` of a
/// floating-point run, with the spectral radius `ρ` of `(I − K·H)·A`. Slow observers
/// (`ρ → 1`) amplify the rounding accordingly. Large start-up innovations times large gains
/// can overflow, so seed `x̂` from the first measurement.
///
/// ## Example
///
/// Design with `Q` and `R` once, then run the fixed gain:
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use minikalman::models::motion;
/// use minikalman::prelude::*;
/// use minikalman::regular::builder::KalmanFilterBuilder;
///
/// let builder = KalmanFilterBuilder::<2, f32>::default();
/// let model = motion::constant_velocity(0.1, 1.0);
///
/// // Design: iterate the Kalman filter's covariance recursion until K settles.
/// let mut design = builder.build();
/// let mut position = builder.observations().build::<1>();
/// design.state_transition_mut().apply(|a| model.fill_transition(a));
/// design.direct_process_noise_mut().apply(|q| model.fill_process_noise(q));
/// design.estimate_covariance_mut().make_identity();
/// position.observation_matrix_mut().set_at(0, 0, 1.0);
/// position.measurement_noise_covariance_mut().set_at(0, 0, 0.25);
/// design.settle_gain(&mut position, 1e-6, 10_000).expect("detectable model");
///
/// // Run: no P, Q or R.
/// let mut observer = builder.build_observer();
/// let mut fixed = builder.observations().build_fixed_gain::<1>();
/// observer.state_transition_mut().apply(|a| model.fill_transition(a));
/// fixed.observation_matrix_mut().set_at(0, 0, 1.0);
/// fixed.set_gain_current(position.kalman_gain());
///
/// for step in 0..100 {
///     observer.predict();
///     fixed.measurement_vector_mut().set_at(0, 0, 0.2 * step as f32);
///     assert!(observer.correct(&mut fixed).is_applied());
/// }
///
/// // The velocity estimate settles at 0.2 per 0.1 s.
/// assert!((observer.state_vector().get_at(1, 0) - 2.0).abs() < 0.05);
/// # }
/// ```
#[doc(alias = "steady_state_kalman")]
#[doc(alias = "fixed_gain_observer")]
#[allow(non_snake_case)]
pub struct LuenbergerObserver<const STATES: usize, T, A, X, PX> {
    /// State transition matrix `A` (`STATES × STATES`).
    A: A,

    /// State estimate `x̂` (`STATES × 1`).
    x: X,

    /// Scratch for `A·x̂` (`STATES × 1`).
    predicted_x: PX,

    /// Marks the scalar type.
    _phantom: PhantomData<T>,
}

impl<const STATES: usize, T, A, X, PX> LuenbergerObserver<STATES, T, A, X, PX> {
    /// Creates an observer from its buffers.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `predicted_x` - A temporary vector (`STATES` × `1`).
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// use minikalman::prelude::*;
    /// use minikalman::regular::LuenbergerObserver;
    ///
    /// impl_buffer_A!(mut A, 2, f32, 0.0);
    /// impl_buffer_x!(mut x, 2, f32, 0.0);
    /// impl_buffer_temp_x!(mut temp_x, 2, f32, 0.0);
    ///
    /// let mut observer = LuenbergerObserver::<2, f32, _, _, _>::new(A, x, temp_x);
    /// observer.state_transition_mut().make_identity();
    /// observer.predict();
    /// ```
    #[allow(non_snake_case)]
    pub fn new(A: A, x: X, predicted_x: PX) -> Self {
        Self {
            A,
            x,
            predicted_x,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of states.
    #[inline(always)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Gets a reference to the state vector x̂.
    #[inline(always)]
    pub fn state_vector(&self) -> &X {
        &self.x
    }

    /// Gets a mutable reference to the state vector x̂, e.g. to seed it from a first
    /// measurement.
    #[inline(always)]
    pub fn state_vector_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Gets a reference to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition(&self) -> &A {
        &self.A
    }

    /// Gets a mutable reference to the state transition matrix A.
    #[inline(always)]
    pub fn state_transition_mut(&mut self) -> &mut A {
        &mut self.A
    }
}

impl<const STATES: usize, T, A, X, PX> LuenbergerObserver<STATES, T, A, X, PX>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    T: MatrixDataType,
{
    /// Performs the time update `x̂ ← A·x̂`.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "predict")]
    ///
    /// In the current form, every step starts with it; in either form, it is the whole step
    /// when no measurement arrives. Costs `STATES²` multiply-adds.
    pub fn predict(&mut self) {
        let x = self.x.as_matrix_mut();
        let x_predicted = self.predicted_x.as_matrix_mut();
        self.A.as_matrix().mult_rowvector(x, x_predicted);
        x_predicted.copy_to(x);
    }

    /// Applies a control input to the state only: `x̂ ← x̂ + B·u`.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "control")]
    ///
    /// Takes the regular [`Control`](crate::regular::Control); its process noise `Q` and its
    /// temporary are not used. In the current form, apply it between [`predict`](Self::predict)
    /// and [`correct`](Self::correct), as for the regular filter. In the predictor form, apply
    /// `B·uₖ` **after** [`predict_from_measurement`](Self::predict_from_measurement): applied
    /// before, the innovation would be formed from `x̂ₖ + B·uₖ` and `B·uₖ` would pass through `A`.
    pub fn control<I>(&mut self, control: &mut I)
    where
        I: KalmanFilterControlApplyToState<STATES, T>,
    {
        control.apply_to_state(&mut self.x);
    }

    /// Performs the current-form measurement update `x̂ ← x̂ + K·(z − H·x̂)`.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "correct")]
    ///
    /// Call it after [`predict`](Self::predict). Costs `2·STATES·OBSERVATIONS` multiply-adds.
    /// Several observations can be applied in sequence, each with the gain designed for that
    /// schedule.
    ///
    /// Returns [`CorrectionStatus::Applied`], [`CorrectionStatus::Gated`] (state unchanged) if
    /// the observation's [gate](crate::regular::FixedGainObservation::set_gate) rejects the
    /// innovation, or [`CorrectionStatus::GainFormMismatch`] (state unchanged) if the
    /// observation holds a [predictor-form](GainForm::Predictor) gain.
    #[must_use = "a gated measurement or a gain-form mismatch skips the measurement"]
    pub fn correct<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T>,
    {
        self.correct_with(measurement, |_| {})
    }

    /// Performs the current-form measurement update, normalizing the innovation before it is
    /// applied, e.g. to wrap an angle.
    ///
    /// See [`correct`](Self::correct).
    ///
    /// ## Arguments
    /// * `measurement` - The observation with a current-form gain.
    /// * `fix` - Normalizes the innovation `y = z − H·x̂` in place.
    #[must_use = "a gated measurement or a gain-form mismatch skips the measurement"]
    pub fn correct_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T>,
        N: FnMut(&mut [T]),
    {
        if measurement.gain_form() != GainForm::Current {
            return CorrectionStatus::GainFormMismatch;
        }

        let x = self.x.as_matrix_mut();
        measurement.compute_innovation(&*x, fix);
        if measurement.is_gated() {
            return CorrectionStatus::Gated;
        }

        measurement.add_gain_times_innovation(x);
        CorrectionStatus::Applied
    }

    /// Performs a whole predictor-form step `x̂ ← A·x̂ + L·(z − H·x̂)`: the innovation is formed
    /// from the current estimate, then the estimate is propagated.
    ///
    /// This is the textbook Luenberger recursion that `place`, `acker` and python-control
    /// `dlqe` gains are designed for. It already contains the time update, so do not call
    /// [`predict`](Self::predict) in the same step. Afterwards, the state is the one-step
    /// prediction `x̂ₖ₊₁|ₖ`, which equals `A·x̂⁺ₖ` of a current-form run with `K = A⁻¹·L`.
    /// Costs `STATES² + 2·STATES·OBSERVATIONS` multiply-adds.
    ///
    /// Returns [`CorrectionStatus::Applied`], [`CorrectionStatus::Gated`] if the observation's
    /// [gate](crate::regular::FixedGainObservation::set_gate) rejects the innovation, or
    /// [`CorrectionStatus::GainFormMismatch`] if the observation holds a
    /// [current-form](GainForm::Current) gain. In both rejections the state is unchanged and
    /// the time update has not run; call [`predict`](Self::predict) to advance the step
    /// without the measurement.
    #[must_use = "a gated measurement or a gain-form mismatch skips the measurement"]
    pub fn predict_from_measurement<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T>,
    {
        self.predict_from_measurement_with(measurement, |_| {})
    }

    /// Performs a whole predictor-form step, normalizing the innovation before it is applied.
    ///
    /// See [`predict_from_measurement`](Self::predict_from_measurement).
    ///
    /// ## Arguments
    /// * `measurement` - The observation with a predictor-form gain.
    /// * `fix` - Normalizes the innovation `y = z − H·x̂` in place.
    #[must_use = "a gated measurement or a gain-form mismatch skips the measurement"]
    pub fn predict_from_measurement_with<M, N>(
        &mut self,
        measurement: &mut M,
        fix: N,
    ) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T>,
        N: FnMut(&mut [T]),
    {
        if measurement.gain_form() != GainForm::Predictor {
            return CorrectionStatus::GainFormMismatch;
        }

        let x = self.x.as_matrix_mut();
        let x_predicted = self.predicted_x.as_matrix_mut();
        measurement.compute_innovation(&*x, fix);
        if measurement.is_gated() {
            return CorrectionStatus::Gated;
        }

        self.A.as_matrix().mult_rowvector(x, x_predicted);
        measurement.add_gain_times_innovation(x_predicted);
        x_predicted.copy_to(x);
        CorrectionStatus::Applied
    }
}

impl<const STATES: usize, T, A, X, PX> KalmanFilterNumStates<STATES>
    for LuenbergerObserver<STATES, T, A, X, PX>
{
}

impl<const STATES: usize, T, A, X, PX> KalmanFilterStateVector<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX>
where
    X: StateVector<STATES, T>,
{
    type StateVector = X;

    #[inline(always)]
    fn state_vector(&self) -> &Self::StateVector {
        self.state_vector()
    }
}

impl<const STATES: usize, T, A, X, PX> KalmanFilterStateVectorMut<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX>
where
    X: StateVectorMut<STATES, T>,
{
    type StateVectorMut = X;

    #[inline(always)]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut {
        self.state_vector_mut()
    }
}

impl<const STATES: usize, T, A, X, PX> KalmanFilterStateTransition<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX>
where
    A: StateTransitionMatrix<STATES, T>,
{
    type StateTransitionMatrix = A;

    #[inline(always)]
    fn state_transition(&self) -> &Self::StateTransitionMatrix {
        self.state_transition()
    }
}

impl<const STATES: usize, T, A, X, PX> KalmanFilterStateTransitionMut<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX>
where
    A: StateTransitionMatrixMut<STATES, T>,
{
    type StateTransitionMatrixMut = A;

    #[inline(always)]
    fn state_transition_mut(&mut self) -> &mut Self::StateTransitionMatrixMut {
        self.state_transition_mut()
    }
}

impl<const STATES: usize, T, A, X, PX> KalmanFilterPredict<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn predict(&mut self) {
        self.predict()
    }
}
