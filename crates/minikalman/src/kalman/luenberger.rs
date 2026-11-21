//! # Fixed-gain (Luenberger) state observer

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};
use core::marker::PhantomData;

/// A fixed-gain (Luenberger) state observer: the steady-state Kalman filter without a
/// covariance.
///
/// It holds only the state transition `A`, the state `x̂` and one temporary vector, `STATES² +
/// 2·STATES` scalars, plus one more vector in the [`Predictor`] form. There is no `P`, `Q` or
/// `R`; the gain lives in a [`FixedGainObservation`](crate::regular::FixedGainObservation).
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
/// ## One step: predict, control, correct
///
/// The time update and the measurement update are separate calls in both gain forms, and
/// nothing predicts implicitly:
///
/// * [`predict`](Self::predict) is the time update. It returns a [`TimeUpdate`] handle through
///   which control inputs are applied, e.g.
///   `observer.predict().control(&mut throttle).control(&mut steering);`. Controls can only be
///   applied there, right after the prediction and before any correction; the borrow checker
///   enforces the order. A plain `observer.predict();` is a step without input.
/// * [`correct`](Self::correct) is the measurement update, once per observation. A step
///   without a measurement is a plain `predict`.
///
/// ## Cost
///
/// Per step with `n` states and `m` measurements: `n²` multiply-adds to predict and `2·n·m`
/// to correct, with no square root, no division and no data-dependent branch, so every step
/// takes the same time. The [`Predictor`] form adds `n` additions to the prediction. An
/// innovation [gate](crate::regular::FixedGainGate) adds `m² + m` multiply-adds and one
/// comparison, and a rejected correction skips the gain, so it is shorter. The regular filter
/// adds `2·n³` for `P` in the prediction, and a Cholesky factorization with `m` square roots
/// and reciprocals, the gain and the covariance update in the correction.
///
/// ## Designing the gain
///
/// * **From `Q` and `R`**, on the host or at boot: build a [`RegularKalman`](crate::regular::RegularKalman) and a
///   [`RegularObservation`](crate::regular::RegularObservation) with the same `A` and `H`,
///   call [`settle_gain`](crate::regular::RegularKalman::settle_gain) and copy the observation's
///   [`kalman_gain`](crate::regular::RegularObservation::kalman_gain) into a [`Current`]-form
///   observation with [`set_gain`](crate::regular::FixedGainObservation::set_gain). The
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
/// | Form | Recursion | Observer | Observation |
/// |---|---|---|---|
/// | [`Current`] | `x̂⁻ = A·x̂⁺ + B·u`, `x̂⁺ = x̂⁻ + K·(z − H·x̂⁻)` | [`new`](Self::new) | `FixedGainObservation<…, Current>` |
/// | [`Predictor`] | `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)` | [`new_predictor`](LuenbergerObserver::new_predictor) | `FixedGainObservation<…, Predictor>` |
///
/// The two gains are related by `L = A·K` and give the same poles, but **they are not
/// interchangeable**. `place(A', C')'`, `acker` and python-control `dlqe` return `L`; this
/// crate's [`settle_gain`](crate::regular::RegularKalman::settle_gain) and MATLAB `kalman`'s `Mx` return `K`.
/// The form is part of the observer's and the observation's types, so running a gain in the
/// other recursion does not compile:
///
/// ```compile_fail
/// # #[cfg(feature = "alloc")] {
/// use minikalman::prelude::*;
/// use minikalman::regular::builder::KalmanFilterBuilder;
///
/// let builder = KalmanFilterBuilder::<2, f32>::default();
/// let mut observer = builder.build_observer();
/// let mut deadbeat = builder.observations().build_fixed_gain::<1, Predictor>();
/// let _ = observer.correct(&mut deadbeat); // a predictor-form gain in the current form
/// # }
/// # #[cfg(not(feature = "alloc"))]
/// # compile_error!("the example needs alloc");
/// ```
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
/// ## The predictor form: deferred correction
///
/// In the [`Predictor`] form, [`correct`](LuenbergerObserver::correct) forms the innovation
/// `yₖ = zₖ − H·x̂ₖ` from the current state, gates it, and adds `L·yₖ` to a pending
/// correction, **without touching `x̂`**. The next [`predict`](LuenbergerObserver::predict)
/// applies `x̂ ← A·x̂ + L·yₖ` and clears it. So after `correct`, the
/// [`state_vector`](Self::state_vector) is still the prior `x̂ₖ|ₖ₋₁`; the measurement shows up
/// after `predict`, as `x̂ₖ₊₁|ₖ`, which equals `A·x̂⁺ₖ` of a current-form run with `K = A⁻¹·L`.
/// Several corrections in one step add their terms, each evaluated at the same prior: for a
/// block-diagonal `L`, this is the update of the stacked observation. A gated correction leaves
/// no term, so the next `predict` is a pure model step.
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
/// let mut fixed = builder.observations().build_fixed_gain::<1, Current>();
/// observer.state_transition_mut().apply(|a| model.fill_transition(a));
/// fixed.observation_matrix_mut().set_at(0, 0, 1.0);
/// fixed.set_gain(position.kalman_gain());
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
pub struct LuenbergerObserver<const STATES: usize, T, A, X, PX, F = Current> {
    /// State transition matrix `A` (`STATES × STATES`).
    A: A,

    /// State estimate `x̂` (`STATES × 1`).
    x: X,

    /// Scratch for `A·x̂` (`STATES × 1`).
    predicted_x: PX,

    /// The gain form: [`Current`], or a [`PendingCorrection`] for the [`Predictor`] form.
    form: F,

    /// Marks the scalar type.
    _phantom: PhantomData<T>,
}

/// The [`Predictor`]-form state of a [`LuenbergerObserver`]: the correction `Σ L·y` that the
/// corrections of this step deferred to the next [`predict`](LuenbergerObserver::predict).
///
/// Holds one `STATES × 1` vector; zero after every prediction.
pub struct PendingCorrection<C>(C);

/// The time update of a [`LuenbergerObserver`] step, through which control inputs are applied.
///
/// Returned by [`LuenbergerObserver::predict`]. It borrows the observer, so controls can only be
/// applied right after the prediction and before any correction:
///
/// ```compile_fail
/// # #[cfg(feature = "alloc")] {
/// use minikalman::prelude::*;
/// use minikalman::regular::builder::KalmanFilterBuilder;
///
/// let builder = KalmanFilterBuilder::<2, f32>::default();
/// let mut observer = builder.build_observer();
/// let mut position = builder.observations().build_fixed_gain::<1, Current>();
/// let mut throttle = builder.controls().build::<1>();
///
/// let step = observer.predict();
/// let _ = observer.correct(&mut position);
/// step.control(&mut throttle); // too late: the step has been corrected
/// # }
/// # #[cfg(not(feature = "alloc"))]
/// # compile_error!("the example needs alloc");
/// ```
///
/// Dropping it without a control, as in a plain `observer.predict();`, is a step without
/// input.
pub struct TimeUpdate<'a, const STATES: usize, T, X> {
    /// The state estimate the controls move.
    x: &'a mut X,

    /// Marks the scalar type.
    _phantom: PhantomData<T>,
}

impl<const STATES: usize, T, X> TimeUpdate<'_, STATES, T, X>
where
    X: StateVectorMut<STATES, T>,
{
    /// Applies a control input to the state only: `x̂ ← x̂ + B·u`.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "control")]
    ///
    /// Takes the regular [`Control`](crate::regular::Control); its process noise `Q` and its
    /// temporary are not used. Chain several distinct controls, e.g. throttle and steering, for
    /// `x̂ ← A·x̂ + B₁·u₁ + B₂·u₂`, as several observations correct one step. Applying the same
    /// control twice in one step adds `B·u` twice, which is a misuse.
    ///
    /// In the [`Predictor`] form, the prediction has already added the pending correction, so
    /// this gives `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·yₖ`.
    ///
    /// ## Arguments
    /// * `control` - The control input.
    ///
    /// ## Example
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f32>::default();
    /// let mut observer = builder.build_observer();
    /// let mut throttle = builder.controls().build::<1>();
    /// let mut wind = builder.controls().build::<1>();
    /// observer.state_transition_mut().make_identity();
    /// throttle.control_matrix_mut().set_at(1, 0, 0.1);
    /// throttle.control_vector_mut().set_at(0, 0, 2.0);
    /// wind.control_matrix_mut().set_at(1, 0, 0.1);
    /// wind.control_vector_mut().set_at(0, 0, -0.5);
    ///
    /// observer.predict().control(&mut throttle).control(&mut wind);
    /// assert!((observer.state_vector().get_at(1, 0) - 0.15).abs() < 1e-6);
    /// # }
    /// ```
    pub fn control<I>(self, control: &mut I) -> Self
    where
        I: KalmanFilterControlApplyToState<STATES, T>,
    {
        control.apply_to_state(self.x);
        self
    }
}

impl<const STATES: usize, T, A, X, PX> LuenbergerObserver<STATES, T, A, X, PX, Current> {
    /// Creates a [`Current`]-form observer from its buffers.
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
            form: Current,
            _phantom: PhantomData,
        }
    }
}

impl<const STATES: usize, T, A, X, PX, C>
    LuenbergerObserver<STATES, T, A, X, PX, PendingCorrection<C>>
where
    C: StateVectorMut<STATES, T>,
{
    /// Creates a [`Predictor`]-form observer from its buffers.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `predicted_x` - A temporary vector (`STATES` × `1`).
    /// * `pending` - The pending correction (`STATES` × `1`); must start at zero.
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
    /// impl_buffer_x!(mut pending, 2, f32, 0.0);
    ///
    /// let mut observer = LuenbergerObserver::<2, f32, _, _, _, _>::new_predictor(A, x, temp_x, pending);
    /// observer.state_transition_mut().make_identity();
    /// observer.predict();
    /// ```
    #[allow(non_snake_case)]
    pub fn new_predictor(A: A, x: X, predicted_x: PX, pending: C) -> Self {
        Self {
            A,
            x,
            predicted_x,
            form: PendingCorrection(pending),
            _phantom: PhantomData,
        }
    }

    /// Gets a reference to the pending correction `Σ L·y` that the next
    /// [`predict`](Self::predict) adds; zero after every prediction.
    #[inline(always)]
    pub fn pending_correction(&self) -> &C {
        &self.form.0
    }
}

impl<const STATES: usize, T, A, X, PX, F> LuenbergerObserver<STATES, T, A, X, PX, F> {
    /// Returns the number of states.
    #[inline(always)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Gets a reference to the state vector x̂.
    ///
    /// In the [`Predictor`] form, this is the prior until the next [`predict`](Self::predict)
    /// applies the pending correction.
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

impl<const STATES: usize, T, A, X, PX, F> LuenbergerObserver<STATES, T, A, X, PX, F>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    T: MatrixDataType,
{
    /// Forms `A·x̂` in the temporary, without writing `x̂`.
    fn propagate(&mut self) {
        let x = self.x.as_matrix();
        let x_predicted = self.predicted_x.as_matrix_mut();
        self.A.as_matrix().mult_rowvector(x, x_predicted);
    }

    /// Copies the temporary into `x̂` and returns the control handle for this step.
    fn finish_prediction(&mut self) -> TimeUpdate<'_, STATES, T, X> {
        self.predicted_x.as_matrix().copy_to(self.x.as_matrix_mut());
        TimeUpdate {
            x: &mut self.x,
            _phantom: PhantomData,
        }
    }

    /// Forms the innovation of `measurement` at the current state and gates it; returns
    /// `true` if the gain is to be applied.
    fn innovation_passes<M, N>(&mut self, measurement: &mut M, fix: N) -> bool
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T>,
        N: FnMut(&mut [T]),
    {
        measurement.compute_innovation(self.x.as_matrix(), fix);
        !measurement.is_gated()
    }
}

impl<const STATES: usize, T, A, X, PX> LuenbergerObserver<STATES, T, A, X, PX, Current>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    T: MatrixDataType,
{
    /// Performs the time update `x̂ ← A·x̂` and returns the handle to apply controls with.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "predict")]
    ///
    /// Every step starts with it; it is the whole step when no measurement arrives. Apply
    /// control inputs through the returned [`TimeUpdate`], e.g.
    /// `observer.predict().control(&mut throttle);`. Costs `STATES²` multiply-adds.
    pub fn predict(&mut self) -> TimeUpdate<'_, STATES, T, X> {
        self.propagate();
        self.finish_prediction()
    }

    /// Performs the current-form measurement update `x̂ ← x̂ + K·(z − H·x̂)`.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "correct")]
    ///
    /// Call it after [`predict`](Self::predict) (and its controls). Costs
    /// `2·STATES·OBSERVATIONS` multiply-adds. Several observations can be applied in sequence,
    /// each with the gain designed for that schedule and each seeing the state the previous
    /// one left.
    ///
    /// Returns [`CorrectionStatus::Applied`], or [`CorrectionStatus::Gated`] (state unchanged)
    /// if the observation's [gate](crate::regular::FixedGainObservation::set_gate) rejects the
    /// innovation.
    #[must_use = "a gated measurement is skipped"]
    pub fn correct<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T, Form = Current>,
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
    #[must_use = "a gated measurement is skipped"]
    pub fn correct_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T, Form = Current>,
        N: FnMut(&mut [T]),
    {
        if !self.innovation_passes(measurement, fix) {
            return CorrectionStatus::Gated;
        }

        measurement.add_gain_times_innovation(self.x.as_matrix_mut());
        CorrectionStatus::Applied
    }
}

impl<const STATES: usize, T, A, X, PX, C>
    LuenbergerObserver<STATES, T, A, X, PX, PendingCorrection<C>>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    C: StateVectorMut<STATES, T>,
    T: MatrixDataType,
{
    /// Performs the predictor-form time update `x̂ ← A·x̂ + Σ L·y`, clears the pending
    /// correction, and returns the handle to apply controls with.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "predict")]
    ///
    /// The pending correction holds the `L·y` terms of the [`correct`](Self::correct) calls
    /// since the last prediction; without any, this is the pure model step `x̂ ← A·x̂`. Apply
    /// `B·uₖ` through the returned [`TimeUpdate`]. Costs `STATES²` multiply-adds and
    /// `STATES` additions.
    pub fn predict(&mut self) -> TimeUpdate<'_, STATES, T, X> {
        self.propagate();
        let pending = self.form.0.as_matrix_mut();
        self.predicted_x.as_matrix_mut().add_inplace_a(&*pending);
        pending.set_zero();
        self.finish_prediction()
    }

    /// Performs the predictor-form measurement update: forms `y = z − H·x̂` from the current
    /// state and defers `L·y` to the next [`predict`](Self::predict), **without touching
    /// `x̂`**.
    ///
    #[doc = diagram!("luenberger-loop.mmd", "correct")]
    ///
    /// After it, the [`state_vector`](Self::state_vector) is still the prior `x̂ₖ|ₖ₋₁`; the
    /// measurement shows up after the next `predict`. Several corrections per step add their
    /// terms, each evaluated at the same prior; for a block-diagonal `L` this equals the update
    /// of the stacked observation. Costs `2·STATES·OBSERVATIONS` multiply-adds.
    ///
    /// Returns [`CorrectionStatus::Applied`], or [`CorrectionStatus::Gated`] if the
    /// observation's [gate](crate::regular::FixedGainObservation::set_gate) rejects the
    /// innovation; a gated measurement leaves no pending term.
    #[must_use = "a gated measurement is skipped"]
    pub fn correct<M>(&mut self, measurement: &mut M) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T, Form = Predictor>,
    {
        self.correct_with(measurement, |_| {})
    }

    /// Performs the predictor-form measurement update, normalizing the innovation before it is
    /// deferred, e.g. to wrap an angle.
    ///
    /// See [`correct`](Self::correct).
    ///
    /// ## Arguments
    /// * `measurement` - The observation with a predictor-form gain.
    /// * `fix` - Normalizes the innovation `y = z − H·x̂` in place.
    #[must_use = "a gated measurement is skipped"]
    pub fn correct_with<M, N>(&mut self, measurement: &mut M, fix: N) -> CorrectionStatus
    where
        M: KalmanFilterObservationCorrectFixedGain<STATES, T, Form = Predictor>,
        N: FnMut(&mut [T]),
    {
        if !self.innovation_passes(measurement, fix) {
            return CorrectionStatus::Gated;
        }

        measurement.add_gain_times_innovation(self.form.0.as_matrix_mut());
        CorrectionStatus::Applied
    }
}

impl<const STATES: usize, T, A, X, PX, F> KalmanFilterNumStates<STATES>
    for LuenbergerObserver<STATES, T, A, X, PX, F>
{
}

impl<const STATES: usize, T, A, X, PX, F> KalmanFilterStateVector<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX, F>
where
    X: StateVector<STATES, T>,
{
    type StateVector = X;

    #[inline(always)]
    fn state_vector(&self) -> &Self::StateVector {
        self.state_vector()
    }
}

impl<const STATES: usize, T, A, X, PX, F> KalmanFilterStateVectorMut<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX, F>
where
    X: StateVectorMut<STATES, T>,
{
    type StateVectorMut = X;

    #[inline(always)]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut {
        self.state_vector_mut()
    }
}

impl<const STATES: usize, T, A, X, PX, F> KalmanFilterStateTransition<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX, F>
where
    A: StateTransitionMatrix<STATES, T>,
{
    type StateTransitionMatrix = A;

    #[inline(always)]
    fn state_transition(&self) -> &Self::StateTransitionMatrix {
        self.state_transition()
    }
}

impl<const STATES: usize, T, A, X, PX, F> KalmanFilterStateTransitionMut<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX, F>
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
    for LuenbergerObserver<STATES, T, A, X, PX, Current>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn predict(&mut self) {
        Self::predict(self);
    }
}

impl<const STATES: usize, T, A, X, PX, C> KalmanFilterPredict<STATES, T>
    for LuenbergerObserver<STATES, T, A, X, PX, PendingCorrection<C>>
where
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    C: StateVectorMut<STATES, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn predict(&mut self) {
        Self::predict(self);
    }
}
