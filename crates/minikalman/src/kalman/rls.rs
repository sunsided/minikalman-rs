//! # Recursive Least Squares
//!
//! An online estimator of a parameter vector `θ` that enters a scalar output linearly,
//! `yₖ = φₖᵀ·θ + vₖ`, with exponential forgetting and a covariance windup guard. See
//! [`RecursiveLeastSquares`] for the math.

use crate::kalman::CorrectionStatus;
use crate::kalman::ud::{bierman_update, rank_one_update, reconstruct_ud};
use crate::matrix::{MatrixData, MatrixDataType, MatrixMut};
use core::marker::PhantomData;

/// How a [`RecursiveLeastSquares`] estimator keeps its covariance `P`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CovarianceForm {
    /// `P = U·D·Uᵀ`, updated with Bierman's scalar update (the default).
    ///
    /// `P` stays symmetric and positive semi-definite by construction, whatever rounding does,
    /// and no square root is taken. Costs about `1.5·N²` multiplications per update.
    #[default]
    Factored,
    /// `P` itself, updated with the textbook `P ← P⁻ − K·φᵀ·P⁻` (symmetrized).
    ///
    /// Slightly cheaper, but rounding can make `P` indefinite on long `f32` or fixed-point runs
    /// with poor excitation. Prefer [`Factored`](Self::Factored) there.
    Covariance,
}

/// How a [`RecursiveLeastSquares`] estimator forgets old samples when `μ < 1`.
///
/// Independent of the [`CovarianceForm`]: both modes work with both forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Forgetting {
    /// `P⁻ = P/μₖ` in every direction, with the [windup guard](RecursiveLeastSquares#covariance-windup)
    /// (the default).
    ///
    /// Tracks every parameter at the rate `μ`, but lets `P` wind up in directions that `φ`
    /// does not excite; the guard bounds that at the cost of stalling forgetting everywhere.
    #[default]
    Exponential,
    /// Forgets only along the current regressor `φ`, so `P` stays bounded in unexcited
    /// directions; see [directional forgetting](RecursiveLeastSquares#directional-forgetting).
    ///
    /// The windup guard is not used. Costs one more rank-one update, about `N²`
    /// multiplications.
    Directional,
}

/// Recursive least squares (RLS): an online estimator of model parameters.
///
#[doc = diagram!("rls-update.mmd")]
/// Estimates the parameter vector `θ` (`N` × `1`) of a model whose scalar output is linear in
/// it, `yₖ = φₖᵀ·θ + vₖ`, from one sample `(φₖ, yₖ)` at a time. The regressor `φₖ` holds known
/// signals, e.g. `φ = [xₖ, uₖ]` for the first-order plant `xₖ₊₁ = a·xₖ + b·uₖ` with `θ = [a, b]`.
/// Each [`update`](Self::update) costs `O(N²)` and inverts no matrix.
///
/// ## RLS is a Kalman filter
///
/// RLS is the Kalman filter with the state `θ`, `A = I`, `Q = 0`, the time-varying observation
/// row `H = φₖᵀ` and the measurement noise variance `r`. With the forgetting factor
/// `0 < μ ≤ 1`, one step is
///
/// ```text
/// P⁻ = P/μ                                  forget (the Kalman prediction)
/// e  = y − φᵀ·θ,   α = φᵀ·P⁻·φ + r          innovation and its variance
/// K  = P⁻·φ/α,   θ ← θ + K·e,   P ← P⁻ − K·φᵀ·P⁻
/// ```
///
/// which minimizes `Σᵢ μ^(k−i)·(yᵢ − φᵢᵀ·θ)²/r + μ^(k+1)·(θ − θ₀)ᵀ·P₀⁻¹·(θ − θ₀)`, i.e.
/// `Pₖ₊₁⁻¹ = μ·Pₖ⁻¹ + φₖ·φₖᵀ/r` (Lai and Bernstein, "Generalized Forgetting Recursive Least
/// Squares: Stability and Robustness Guarantees", [arXiv:2308.04259](https://arxiv.org/abs/2308.04259),
/// eqs. (37)–(39), there with `r = 1`). The gain is often written `K = P·φ/(μ·r + φᵀ·P·φ)`,
/// which is the same. With `μ = 1` and a large `P₀`, `θ` approaches the batch least-squares
/// solution of the normal equations.
///
/// **Forgetting and `predict_tuned`.**
/// [`RegularKalman::predict_tuned(λ)`](crate::regular::RegularKalman::predict_tuned) scales `P`
/// by `1/λ²`, so a `RegularKalman` with `A = I`, `Q = 0`, `H = φᵀ`, `R = r` that calls
/// `predict_tuned(λ)` before each correction is this estimator with `μ = λ²`, that is,
/// `predict_tuned(√μ)`, as long as the [windup guard](#covariance-windup) is off or inactive
/// (`tr(P) ≤ μ·c`); the guard raises the factor above `μ`. `μ = 0.98` weights a sample from 50 steps ago by `0.98⁵⁰ ≈ 0.36`.
/// A random-walk `Q` (a [`RegularKalman`](crate::regular::RegularKalman) with `A = I`) is the
/// alternative for drifting parameters: it adds a fixed uncertainty per step instead of
/// scaling `P`.
///
/// ## Numerics
///
/// The default [`CovarianceForm::Factored`] keeps `P = U·D·Uᵀ` and applies Bierman's scalar
/// update, the same code as [`UdKalman::correct`](crate::ud::UdKalman::correct); forgetting
/// is `D ← D/μ` with `U` unchanged. `P` stays positive semi-definite by construction, also in
/// `f32` and `I16F16`. [`CovarianceForm::Covariance`] keeps `P` directly.
///
/// **Fixed point.** Forgetting raises a variance `D` by `D·(1 − μ)/μ`, which is lost to
/// rounding unless it is at least one step of the type (`2⁻¹⁶ ≈ 1.5·10⁻⁵` for `I16F16`). Keep
/// the variances above about `2⁻¹⁶·μ/(1 − μ)`: 50 steps (`7.5·10⁻⁴`) at `μ = 0.98`, 1000 steps
/// (`0.015`) at `μ = 0.999`. Below that, forgetting stops and the estimate freezes; a variance
/// that rounds to zero stays zero. `P` scales with `r` and settles near `r·(1 − μ)/E[φ²]` per
/// parameter, so scale `r` up if needed. Keep `α = φᵀ·P⁻·φ + r` and `p0` well below the range.
/// [Directional forgetting](#directional-forgetting) raises a variance `Pⱼⱼ` by at most
/// `Pⱼⱼ·(1 − μ)/μ` (since `gⱼ² ≤ Pⱼⱼ·s`), so the same resolution bound applies to the excited
/// variances, and it cannot overflow where exponential forgetting does not. Its weight
/// `(1 − μ)/(μ·s)` grows as `s` shrinks; a regressor whose `s` rounds to zero skips the step.
///
/// ## Covariance windup
///
/// With `μ < 1`, `P` grows as `μ⁻ᵏ` in every direction that `φ` does not excite, e.g. while
/// a plant sits at rest with `u = 0`. When excitation returns, the huge gain makes `θ` jump
/// ("covariance windup"). The guard, on by default, bounds the trace of `P`: each step uses
///
/// ```text
/// μₖ = clamp(tr(P)/c, μ, 1),   so that   tr(P⁻) = tr(P)/μₖ ≤ max(c, tr(P))
/// ```
///
/// Since an update never increases the trace, `tr(P) ≤ max(c, tr(P₀))` for all `k`. While
/// `tr(P) ≤ μ·c`, forgetting runs at the full rate `μ`; above that it slows down, and it stops
/// at `c`. This is variable-rate forgetting with `μ ≤ μₖ ≤ 1`, the class of eqs. (42)–(44) of
/// Lai and Bernstein. The bound defaults to `tr(P₀) = N·p0`, so the first update runs without
/// forgetting; once the data has brought the trace below `c`, forgetting resumes at a reduced
/// rate, and at the full rate `μ` below `μ·c`. Change it with
/// [`set_trace_bound`](Self::set_trace_bound), or pass `None` to disable the guard.
///
/// **Trade-off.** The bound is on the trace, not per direction. While one direction is not
/// excited, its variance grows until the trace reaches `c`, and then forgetting **stops for
/// all directions**: the excited ones no longer track changes (as with `μ = 1`), and the prior
/// `P₀` fades only as `1/k` instead of `μᵏ`. The default `c = N·p0` with a large `p0` also
/// lets an unexcited variance climb back to its startup value. Set `c` deliberately, e.g. to
/// the largest parameter variance you accept times `N`.
/// [Directional forgetting](#directional-forgetting) avoids both the windup and the stall.
///
/// ## Directional forgetting
///
/// With [`Forgetting::Directional`], each step forgets only the information that lies along
/// the current regressor (Kulhavý and Kárný, "Tracking of Slowly Varying Parameters by
/// Directional Forgetting", 9th IFAC World Congress, 1984; Bittanti, Bolzern and Campi,
/// "Convergence and Exponential Convergence of Identification Algorithms with Directional
/// Forgetting Factor", Automatica 26(5), 1990):
///
/// ```text
/// s  = φᵀ·P·φ,   g = P·φ
/// P⁻ = P + (1 − μ)/μ · g·gᵀ/s              i.e.   P⁻⁻¹ = P⁻¹ − (1 − μ)·φ·φᵀ/s
/// ```
///
/// The information `P⁻¹` shrinks by the factor `μ` along `φ` and is kept in every direction
/// that `φ` does not excite, so `P` cannot wind up there and needs no guard: an idle
/// regressor keeps the variance it had, and its parameter does not jump when excitation
/// returns. Since `φᵀ·P⁻·φ = s/μ`, the innovation variance, the NIS and the gate are the same
/// as with exponential forgetting; the correction then runs unchanged. A step with `s = 0`
/// (e.g. `φ = 0`) forgets nothing. In the [`Factored`](CovarianceForm::Factored) form the
/// inflation is an Agee–Turner rank-one update of `U` and `D` with a positive weight, so
/// `D ≥ 0` is kept by construction, without square roots or re-factorization.
///
/// **Tracking rate.** A direction forgets only on the samples that excite it. With one
/// excited direction, this tracks like exponential forgetting at the same `μ`. When `k`
/// independent directions take turns (or share the excitation), each is forgotten about
/// `1/k` as often, so tracking is about `k` times slower; `μ ≈ 1 − k·(1 − μₑ)` matches the
/// memory of an exponential factor `μₑ`.
///
/// **Limitation.** Information in a direction that is excited only rarely is still added on
/// each such sample but forgotten only along `φ`, so its variance can become small and that
/// parameter slow to follow later changes. Bounding the information matrix as well (e.g.
/// Cao and Schwartz, "A Directional Forgetting Algorithm Based on the Decomposition of the
/// Information Matrix", Automatica 36(11), 2000) is not implemented.
///
/// ## Gate and NIS
///
/// The [NIS](Self::nis) is `e²/α`. With a [gate](Self::set_gate), an update whose NIS exceeds
/// it returns [`CorrectionStatus::Gated`]; a gated or rejected update leaves `θ` and `P`
/// unchanged, forgetting included, like an observation's correction.
///
/// ## Buffers
///
/// `Theta` holds `θ` (`N` × `1`), `P` the covariance or its UD factors (`N` × `N`, see
/// [`CovarianceForm`]), and `TempF` and `TempK` (`N` × `1` each) are scratch space. Use
/// [`EstimatorStorage`](crate::rls::EstimatorStorage) to hold them without allocation.
///
/// ## Example
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
/// use minikalman::prelude::*;
/// use minikalman::rls::EstimatorStorage;
///
/// // y = 2·u + 1, estimated as θ = [gain, offset] with φ = [u, 1].
/// let mut storage = EstimatorStorage::<2, f64>::new(0.0);
/// let mut rls = storage.build(1000.0);
/// rls.set_forgetting_factor(0.99);
///
/// for k in 0..50 {
///     let u = (k % 7) as f64;
///     assert!(rls.update(&[u, 1.0], 2.0 * u + 1.0).is_applied());
/// }
///
/// let theta = rls.parameters();
/// assert!((theta.get_row(0) - 2.0).abs() < 1e-3);
/// assert!((theta.get_row(1) - 1.0).abs() < 1e-3);
/// # }
/// ```
#[allow(non_snake_case)]
pub struct RecursiveLeastSquares<const N: usize, T, Theta, P, TempF, TempK> {
    /// The parameter estimate `θ` (`N` × `1`).
    theta: Theta,

    /// The covariance `P` (`N` × `N`), or its UD factors in the layout of
    /// [`UdKalman`](crate::ud::UdKalman): `D` on the diagonal, the strictly upper part of `U`
    /// above it, zeros below.
    P: P,

    /// Scratch `N` × `1`: `Uᵀ·φ` or `P·φ`.
    temp_f: TempF,

    /// Scratch `N` × `1`: the gain.
    temp_k: TempK,

    /// How [`Self::P`] is stored.
    form: CovarianceForm,

    /// How the estimator forgets.
    mode: Forgetting,

    /// The forgetting factor `μ`, `0 < μ ≤ 1`; `1` disables forgetting.
    forgetting: T,

    /// The trace bound `c` of the windup guard, or `None` if the guard is off.
    trace_bound: Option<T>,

    /// The measurement noise variance `r > 0`.
    r: T,

    /// The NIS gate, or `None` to accept every sample.
    gate: Option<T>,

    /// The NIS of the last update, also of a gated one.
    nis: T,

    /// Binds the element type.
    _phantom: PhantomData<T>,
}

impl<const N: usize, T, Theta, P, TempF, TempK> RecursiveLeastSquares<N, T, Theta, P, TempF, TempK>
where
    T: MatrixDataType,
    Theta: MatrixMut<N, 1, T>,
    P: MatrixMut<N, N, T>,
    TempF: MatrixMut<N, 1, T>,
    TempK: MatrixMut<N, 1, T>,
{
    /// Initializes an estimator from caller-provided buffers and resets it to `θ = 0` and
    /// `P = p0·I`.
    ///
    /// Starts with the [`Factored`](CovarianceForm::Factored) form, no forgetting (`μ = 1`),
    /// `r = 1`, no gate and the windup guard at `c = tr(P₀) = N·p0`. A large `p0` (e.g.
    /// `1000` for parameters of order one) means little prior knowledge. Set a known initial
    /// guess with [`parameters_mut`](Self::parameters_mut).
    ///
    /// ## Arguments
    /// * `theta` - The parameter vector `θ` (`N` × `1`).
    /// * `P` - The covariance or its UD factors (`N` × `N`).
    /// * `temp_f`, `temp_k` - Scratch vectors (`N` × `1`).
    /// * `p0` - The initial variance of every parameter.
    #[allow(non_snake_case)]
    pub fn new(theta: Theta, P: P, temp_f: TempF, temp_k: TempK, p0: T) -> Self {
        let mut rls = Self {
            theta,
            P,
            temp_f,
            temp_k,
            form: CovarianceForm::Factored,
            mode: Forgetting::Exponential,
            forgetting: T::one(),
            trace_bound: None,
            r: T::one(),
            gate: None,
            nis: T::zero(),
            _phantom: PhantomData,
        };

        rls.reset(p0);
        rls.trace_bound = Some(rls.covariance_trace());
        rls
    }

    /// Resets the estimate to `θ = 0` and `P = p0·I`, keeping the form and all settings.
    pub fn reset(&mut self, p0: T) {
        for i in 0..N {
            self.theta.set(i, 0, T::zero());
            for j in 0..N {
                self.P.set(i, j, if i == j { p0 } else { T::zero() });
            }
        }

        self.nis = T::zero();
    }

    /// Switches to the covariance form `form` and resets the estimate as
    /// [`reset`](Self::reset) does. The windup guard's bound is kept.
    pub fn reset_with_form(&mut self, form: CovarianceForm, p0: T) {
        self.form = form;
        self.reset(p0);
    }

    /// Returns how the covariance is stored.
    pub fn form(&self) -> CovarianceForm {
        self.form
    }

    /// Returns the parameter estimate `θ` (`N` × `1`).
    pub fn parameters(&self) -> &Theta {
        &self.theta
    }

    /// Returns the parameter estimate `θ` (`N` × `1`) mutably, e.g. to set an initial guess.
    pub fn parameters_mut(&mut self) -> &mut Theta {
        &mut self.theta
    }

    /// Returns the raw covariance buffer: `P` in the
    /// [`Covariance`](CovarianceForm::Covariance) form, its UD factors in the
    /// [`Factored`](CovarianceForm::Factored) form. Use [`covariance`](Self::covariance) for `P`.
    pub fn covariance_buffer(&self) -> &P {
        &self.P
    }

    /// Writes the covariance `P` into `out`, recovering it from the factors if needed.
    pub fn covariance(&self, out: &mut impl MatrixMut<N, N, T>) {
        match self.form {
            CovarianceForm::Factored => reconstruct_ud(&self.P, out),
            CovarianceForm::Covariance => self.P.copy_to(out),
        }
    }

    /// Returns the trace of `P`, the sum of the parameter variances. Costs `O(N²)` in the
    /// factored form.
    pub fn covariance_trace(&self) -> T {
        let mut trace = T::zero();
        for k in 0..N {
            let mut column = T::one();
            if self.form == CovarianceForm::Factored {
                for i in 0..k {
                    column += self.P.get(i, k) * self.P.get(i, k);
                }
            }

            trace += self.P.get(k, k) * column;
        }

        trace
    }

    /// Returns the forgetting factor `μ`.
    pub fn forgetting_factor(&self) -> T {
        self.forgetting
    }

    /// Sets the forgetting factor `0 < μ ≤ 1`; `1` (the default) disables forgetting.
    ///
    /// The memory is about `1/(1 − μ)` samples. Typical values are `0.95` to `0.999`. The
    /// equivalent [`predict_tuned`](crate::regular::RegularKalman::predict_tuned) factor is
    /// `λ = √μ`.
    pub fn set_forgetting_factor(&mut self, mu: T) {
        self.forgetting = mu;
    }

    /// Returns how the estimator forgets.
    pub fn forgetting_mode(&self) -> Forgetting {
        self.mode
    }

    /// Sets how the estimator forgets (default [`Forgetting::Exponential`]). Takes effect with
    /// the next update and keeps `θ` and `P`.
    pub fn set_forgetting_mode(&mut self, mode: Forgetting) {
        self.mode = mode;
    }

    /// Returns the trace bound `c` of the windup guard, or `None` if the guard is off.
    pub fn trace_bound(&self) -> Option<T> {
        self.trace_bound
    }

    /// Sets the trace bound `c` of the windup guard, or turns the guard off with `None`.
    ///
    /// Forgetting slows down as `tr(P)` approaches `c` and stops at `c`. A bound below the
    /// current trace stops forgetting until updates have brought the trace below it. Only used
    /// with [`Forgetting::Exponential`].
    pub fn set_trace_bound(&mut self, bound: Option<T>) {
        self.trace_bound = bound;
    }

    /// Returns the measurement noise variance `r`.
    pub fn measurement_variance(&self) -> T {
        self.r
    }

    /// Sets the measurement noise variance `r > 0` (default `1`).
    ///
    /// `r` weighs the data against the prior `P₀`: from a fixed `p0` it also changes `θ` (with
    /// `N = 1`, `p0 = 1`, `φ = 1`, `y = 1`, `μ = 1`, `r = 1` gives `θ = 0.5` and `r = 4` gives
    /// `θ = 0.2`). For a constant `r`, it only rescales `P` once `P` has settled, i.e. the prior
    /// has faded. It also scales the [NIS](Self::nis) and thus the gate, and weighs the samples
    /// against each other when it changes between them (weighted least squares).
    pub fn set_measurement_variance(&mut self, r: T) {
        self.r = r;
    }

    /// Returns the NIS gate, or `None` if every sample is accepted.
    pub fn gate(&self) -> Option<T> {
        self.gate
    }

    /// Sets the NIS gate: an update whose NIS `e²/α` exceeds it is rejected as an outlier.
    /// For one output, `3.84` (`χ²₁` at 95 %) or `6.63` (99 %) are common.
    pub fn set_gate(&mut self, gate: Option<T>) {
        self.gate = gate;
    }

    /// Returns the normalized innovation squared `e²/α` of the last update, also of a gated
    /// one (`0` after a reset).
    pub fn nis(&self) -> T {
        self.nis
    }

    /// Returns the model output `φᵀ·θ` of the current estimate for the regressor `phi`.
    pub fn predict(&self, phi: &[T; N]) -> T {
        let mut sum = T::zero();
        for (i, &p) in phi.iter().enumerate() {
            sum += p * self.theta.get(i, 0);
        }

        sum
    }

    /// Updates the estimate with the sample `y = φᵀ·θ + v`.
    ///
    #[doc = diagram!("rls-update.mmd", "predict", "innovation", "correct")]
    /// Forgets with the guarded factor `μₖ` (or along `φ` with [`Forgetting::Directional`]),
    /// then applies the scalar Kalman update; see the [type documentation](Self) for the
    /// equations. Costs `O(N²)`.
    ///
    /// ## Status
    /// Returns [`CorrectionStatus::NotPositiveDefinite`] if `r ≤ 0` or the innovation
    /// variance `α ≤ 0`, and [`CorrectionStatus::Gated`] if the NIS exceeds the
    /// [gate](Self::set_gate). Both leave `θ` and `P` unchanged.
    pub fn update(&mut self, phi: &[T; N], y: T) -> CorrectionStatus {
        if self.r <= T::zero() {
            return CorrectionStatus::NotPositiveDefinite;
        }

        let mu = match self.mode {
            Forgetting::Exponential => self.effective_forgetting(),
            Forgetting::Directional => self.forgetting,
        };
        let s = self.regressor_variance(phi);
        let alpha = s / mu + self.r;
        if alpha <= T::zero() {
            return CorrectionStatus::NotPositiveDefinite;
        }

        let e = y - self.predict(phi);
        self.nis = e * (e / alpha);
        if self.gate.is_some_and(|gate| self.nis > gate) {
            return CorrectionStatus::Gated;
        }

        match (self.mode, self.form) {
            (Forgetting::Exponential, CovarianceForm::Factored) => {
                self.update_factored(phi, mu, e);
            }
            (Forgetting::Exponential, CovarianceForm::Covariance) => {
                self.update_covariance(mu, mu, e, alpha);
            }
            (Forgetting::Directional, CovarianceForm::Factored) => {
                if let Some(c) = Self::directional_weight(mu, s) {
                    self.forget_along_factored(c);
                }
                self.update_factored(phi, T::one(), e);
            }
            (Forgetting::Directional, CovarianceForm::Covariance) => {
                if let Some(c) = Self::directional_weight(mu, s) {
                    self.forget_along_covariance(c);
                }
                // P⁻·φ = g + (1 − μ)/μ·g = g/μ, and P⁻ is already in P.
                self.update_covariance(mu, T::one(), e, alpha);
            }
        }

        CorrectionStatus::Applied
    }

    /// Returns the forgetting factor of this step, `μₖ = clamp(tr(P)/c, μ, 1)` with the
    /// windup guard, or `μ` without it.
    fn effective_forgetting(&self) -> T {
        let mu = self.forgetting;
        let Some(bound) = self.trace_bound else {
            return mu;
        };

        let trace = self.covariance_trace();
        if trace >= bound {
            return T::one();
        }

        let ratio = trace / bound;
        if ratio > mu { ratio } else { mu }
    }

    /// Returns `φᵀ·P·φ` of the current (not yet forgotten) `P`, and leaves `Uᵀ·φ` (factored)
    /// or `P·φ` (covariance) in [`Self::temp_f`].
    fn regressor_variance(&mut self, phi: &[T; N]) -> T {
        let mut variance = T::zero();
        match self.form {
            CovarianceForm::Factored => {
                for j in 0..N {
                    let mut f = phi[j];
                    for (i, &p) in phi.iter().enumerate().take(j) {
                        f += self.P.get(i, j) * p;
                    }

                    self.temp_f.set(j, 0, f);
                    variance += self.P.get(j, j) * (f * f);
                }
            }
            CovarianceForm::Covariance => {
                for i in 0..N {
                    let mut g = T::zero();
                    for (j, &p) in phi.iter().enumerate() {
                        g += self.P.get(i, j) * p;
                    }

                    self.temp_f.set(i, 0, g);
                    variance += phi[i] * g;
                }
            }
        }

        variance
    }

    /// Returns the weight `(1 − μ)/(μ·s)` of directional forgetting, or `None` if there is
    /// nothing to forget (`μ ≥ 1` or `s ≤ 0`).
    fn directional_weight(mu: T, s: T) -> Option<T> {
        if mu >= T::one() || s <= T::zero() {
            return None;
        }

        Some((T::one() - mu) / mu / s)
    }

    /// Applies `P ← P + c·g·gᵀ` with `g = P·φ = U·(D·f)` to the UD factors, with `f = Uᵀ·φ` in
    /// [`Self::temp_f`]. Uses [`Self::temp_k`] for `g`.
    fn forget_along_factored(&mut self, c: T) {
        for i in 0..N {
            let mut g = self.P.get(i, i) * self.temp_f.get(i, 0);
            for k in (i + 1)..N {
                g += self.P.get(i, k) * (self.P.get(k, k) * self.temp_f.get(k, 0));
            }

            self.temp_k.set(i, 0, g);
        }

        rank_one_update(&mut self.P, c, &mut self.temp_k);
    }

    /// Applies `P ← P + c·g·gᵀ` with `g = P·φ` in [`Self::temp_f`], keeping `P` symmetric.
    fn forget_along_covariance(&mut self, c: T) {
        for i in 0..N {
            let cg = c * self.temp_f.get(i, 0);
            for j in i..N {
                let p = self.P.get(i, j) + cg * self.temp_f.get(j, 0);
                self.P.set(i, j, p);
                self.P.set(j, i, p);
            }
        }
    }

    /// Applies forgetting `D ← D/μₖ` and Bierman's update for the innovation `e`.
    fn update_factored(&mut self, phi: &[T; N], mu: T, e: T) {
        if mu < T::one() {
            for j in 0..N {
                self.P.set(j, j, self.P.get(j, j) / mu);
            }
        }

        let h = MatrixData::new_ref::<1, N, T>(phi);
        bierman_update(
            &mut self.P,
            &mut self.theta,
            &h,
            0,
            self.r,
            e,
            &mut self.temp_f,
            &mut self.temp_k,
        );
    }

    /// Applies the textbook update with `P⁻·φ = temp_f/mu` and `P⁻ = P/p_divisor`, for `P·φ`
    /// in [`Self::temp_f`] and the innovation variance `alpha`, writing the upper triangle of
    /// `P` and mirroring it. Exponential forgetting passes `μₖ` for both; directional
    /// forgetting has already inflated `P` and passes `p_divisor = 1`.
    fn update_covariance(&mut self, mu: T, p_divisor: T, e: T, alpha: T) {
        for i in 0..N {
            let g = self.temp_f.get(i, 0) / mu;
            let k = g / alpha;
            self.temp_f.set(i, 0, g);
            self.temp_k.set(i, 0, k);
            self.theta.set(i, 0, self.theta.get(i, 0) + k * e);
        }

        for i in 0..N {
            for j in i..N {
                let p =
                    self.P.get(i, j) / p_divisor - self.temp_f.get(i, 0) * self.temp_k.get(j, 0);
                self.P.set(i, j, p);
                self.P.set(j, i, p);
            }
        }
    }
}

#[cfg(test)]
mod tests;
