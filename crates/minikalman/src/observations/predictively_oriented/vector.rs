//! # Predictively oriented (PrO) correction for vector linear observations.
//!
//! For a linear observation the PrO posterior is Kalman-shaped, `P⁻¹ = P̃⁻¹ + Hᵀ×W×H`, so only
//! the symmetric `m×m` weight `W` has to be found (`m = OBSERVATIONS`). With `G = H×P̃×Hᵀ`, the
//! measurement noise `R` and the prior innovation `v = z − H×x̃`:
//!
//! ```text
//! C = H×P×Hᵀ = G×(I + W×G)⁻¹,   S = C + R,   M = S + G,   a = M⁻¹×v
//! F(W) = S⁻¹ − a×aᵀ
//! ```
//!
//! and the solution is a fixed point `W = F(W)`. Up to a constant, `W` minimizes the profiled
//! objective
//!
//! ```text
//! J(W) = ½·(log det S + vᵀ×M⁻¹×v + tr(B⁻¹) + log det B),   B = I + Lᵀ×W×L,   G = L×Lᵀ
//! ```
//!
//! whose gradient with respect to `C` is `½·D` with `D = F(W) − W`. The step `W ← W + β·D`
//! therefore has the slope `−½·tr(D×C×D×C) ≤ 0`, i.e. it is a descent direction. Iterating
//! `W ← F(W)` without damping diverges for innovations beyond about two standard deviations.
//!
//! Every step starts at `β = 1` and halves `β` until `B` has a Cholesky factor (the PD guard:
//! the posterior covariance stays positive definite) and the Armijo condition
//! `J(W + β·D) ≤ J(W) − σ·β·½·tr(D×C×D×C)` holds. The guard changes only the path, never the
//! solution. If the full step passes right away, `β` is doubled instead while the objective
//! keeps decreasing: when `H×P̃×Hᵀ` and `R` differ by orders of magnitude, the full steps are
//! tiny and the plain iteration would need about `√(G/R)` steps. The iteration starts at
//! whichever of `W = 0` (the prior covariance) and `W = R⁻¹` (the Kalman filter's covariance)
//! has the lower objective.
//!
//! The iteration has converged once `tr(D×C×D×C) ≤ 10⁻¹⁰` (a relative error of `C` of about
//! `10⁻⁵`), or the machine epsilon of `T` if that is larger (`f32`, fixed point). It also stops early once the line search cannot make progress at the working
//! precision, which counts as converged only if the threshold holds. The rate is linear and
//! slows down for large innovations, where `β` becomes small: the PD guard needs `β` of about
//! `1/NIS` in the first steps.
//!
//! Parametrizing by `W` limits how far `C` can inflate: for a huge outlier, `W` approaches
//! `−G⁻¹`, and `C = (G⁻¹ + W)⁻¹` can grow to only about `G/ε` with the machine epsilon `ε`
//! of `T`. In `f32` that caps the inflation near `1.7·10⁷×G`, i.e. outliers beyond roughly
//! `10⁷` standard deviations are inflated less than the exact PrO solution (and reported as
//! not converged). In `f64` the cap is near `4.5·10¹⁵×G`.
//!
//! Given `W`, the posterior is
//!
//! ```text
//! x = x̃ + P̃×Hᵀ×M⁻¹×v
//! P = P̃ − P̃×Hᵀ×G⁻¹×(G − C)×G⁻¹×H×P̃
//! ```
//!
//! The covariance form mirrors the scalar `(g − c)/g²`: it involves only `G` and `C`, and
//! `G⁻¹` is applied through triangular solves with `L`, not formed explicitly.
//!
//! `MatrixDataType` has no logarithm, so the log-determinants use a private series
//! ([`ln`]) that needs only arithmetic.

use crate::kalman::CorrectionStatus;
use crate::matrix::{
    Matrix, MatrixDataMut, MatrixDataRef, MatrixDataType, MatrixMut, RowMajorSequentialData,
    RowMajorSequentialDataMut, SquareMatrix,
};

/// The Armijo constant `σ` of the sufficient-decrease condition.
const ARMIJO: f32 = 1e-4;

/// The maximum number of step doublings per line search, i.e. the largest `β` is `2¹⁴`.
///
/// Kept below `2¹⁵` so that `β` fits `I16F16`, whose largest value is about `32767`.
const MAX_EXPANSIONS: usize = 14;

/// A safety cap on the step halvings per line search.
///
/// The PD guard needs `β` of about `1/NIS` in the first steps, so huge outliers need many
/// halvings (about 53 for an outlier of `10⁸` standard deviations). Halving normally ends once
/// the trial equals `W` in `T`, which takes at most about 1100 halvings for `f64` (including
/// subnormals); the cap only bounds the loop for non-finite values.
const MAX_HALVINGS: usize = 1100;

/// Convergence threshold on the dimensionless `tr(D×C×D×C)`, raised to the machine epsilon of
/// `T` where that is larger (see [`tolerance`]).
const TOLERANCE: f64 = 1e-10;

/// Performs the PrO update of a vector linear observation from the prior innovation
/// `y = z − H×x`.
///
/// Stores the normalized innovation squared `yᵀ×(H×P×Hᵀ + R)⁻¹×y` of the prior in `nis` and
/// honors `gate`, like the regular update. Leaves `x` and `P` unchanged and returns
/// [`CorrectionStatus::NotPositiveDefinite`] unless both `G = H×P×Hᵀ` and `R` are positive
/// definite (which requires `OBSERVATIONS ≤ STATES`).
///
/// Otherwise applies the last accepted iterate and returns [`CorrectionStatus::Applied`] if the
/// iteration converged within `max_iterations` accepted steps, or
/// [`CorrectionStatus::NotConverged`] if not. Every iterate is a valid, positive definite
/// posterior with no higher objective than the start.
///
/// ## Buffers
/// - `factor`, `iterate` and `trial` are the caller's workspace and alias nothing.
/// - `S` and `temp_S_inv` serve as `m×m` scratch, and row 0 of `temp_PHt` as an `m`-vector,
///   during the iteration. `temp_S_inv` may alias `temp_HP` and `temp_KHP`, which are only
///   written after the iteration; `temp_PHt` never aliases `temp_S_inv`.
/// - `K` keeps `P̃×Hᵀ` until the end, since `temp_PHt` may alias `temp_KHP`, and holds the
///   mean gain `P̃×Hᵀ×M⁻¹` afterwards. `S` holds `H×P×Hᵀ + R` of the posterior.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn update_vector<const STATES: usize, const OBSERVATIONS: usize, T>(
    x: &mut impl MatrixMut<STATES, 1, T>,
    P: &mut impl MatrixMut<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    K: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    S: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_S_inv: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    temp_HP: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    temp_PHt: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    temp_KHP: &mut impl MatrixMut<STATES, STATES, T>,
    factor: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    iterate: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    trial: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    nis: &mut T,
    gate: Option<T>,
    max_iterations: usize,
) -> CorrectionStatus
where
    T: MatrixDataType,
{
    // G = H*P̃*Hᵀ; K is only written once the checks pass, as in the regular update.
    P.mult_transb(H, temp_PHt);
    H.mult(temp_PHt, factor);
    let checked = prepare::<OBSERVATIONS, T>(
        factor.as_mut_slice(),
        S.as_mut_slice(),
        &mut trial.as_mut_slice()[..OBSERVATIONS],
        R.as_slice(),
        y.as_slice(),
        nis,
        gate,
    );
    if let Err(status) = checked {
        return status;
    }

    // K = P̃*Hᵀ, kept until the end since temp_PHt is scratch during the iteration.
    K.copy_from(temp_PHt);

    let converged = {
        let mut solver = Solver::<OBSERVATIONS, T> {
            l: MatrixDataMut::from(factor.as_mut_slice()),
            w: MatrixDataMut::from(iterate.as_mut_slice()),
            trial: MatrixDataMut::from(trial.as_mut_slice()),
            work: MatrixDataMut::from(S.as_mut_slice()),
            c: MatrixDataMut::from(temp_S_inv.as_mut_slice()),
            a: MatrixDataMut::from(&mut temp_PHt.as_mut_slice()[..OBSERVATIONS]),
            r: MatrixDataRef::from(R.as_slice()),
            v: MatrixDataRef::from(y.as_slice()),
        };

        let Some(converged) = solver.solve(max_iterations) else {
            return CorrectionStatus::NotPositiveDefinite;
        };

        if !solver.finish() {
            return CorrectionStatus::NotPositiveDefinite;
        }

        converged
    };

    // P -= K*Φ*Kᵀ with Φ = G⁻¹(G−C)G⁻¹ in `trial`.
    trial.mult_transb(K, temp_HP);
    K.mult(temp_HP, temp_KHP);
    P.sub_inplace_a(temp_KHP);

    // K = P̃*Hᵀ*M⁻¹ with M⁻¹ in `iterate`; x += K*y
    K.mult(iterate, temp_PHt);
    K.copy_from(temp_PHt);
    K.multadd_rowvector(y, x);

    if converged {
        CorrectionStatus::Applied
    } else {
        CorrectionStatus::NotConverged
    }
}

/// Checks that `G + R`, `R` and `G` are positive definite, stores the NIS `vᵀ×(G + R)⁻¹×v` and
/// applies the gate, then replaces `G` by its lower Cholesky factor.
///
/// All arguments are row-major: `g` and `work` are `M×M`, `scratch` and `v` hold `M` values,
/// and `r` is `M×M`. `work` and `scratch` are overwritten. The NIS is only formed once the
/// Cholesky factor of `G + R` exists, so no division by a zero pivot happens (which panics for
/// fixed-point types). Returns the status to report if the correction must be skipped.
fn prepare<const M: usize, T>(
    g: &mut [T],
    work: &mut [T],
    scratch: &mut [T],
    r: &[T],
    v: &[T],
    nis: &mut T,
    gate: Option<T>,
) -> Result<(), CorrectionStatus>
where
    T: MatrixDataType,
{
    // NIS = vᵀ(G+R)⁻¹v = |L⁻¹v|² with G + R = L×Lᵀ.
    for ((w, &g), &r) in work.iter_mut().zip(g.iter()).zip(r) {
        *w = g + r;
    }

    if !MatrixDataMut::<M, M, T>::from(&mut *work).cholesky_decompose_lower() {
        return Err(CorrectionStatus::NotPositiveDefinite);
    }

    scratch.copy_from_slice(v);
    forward_substitute::<M, T>(work, scratch, 1);
    let value = scratch.iter().map(|&a| a * a).sum::<T>();

    work.copy_from_slice(r);
    let positive = MatrixDataMut::<M, M, T>::from(&mut *work).cholesky_decompose_lower()
        && MatrixDataMut::<M, M, T>::from(&mut *g).cholesky_decompose_lower();
    if !positive {
        return Err(CorrectionStatus::NotPositiveDefinite);
    }

    *nis = value;
    if gate.is_some_and(|gate| *nis > gate) {
        return Err(CorrectionStatus::Gated);
    }

    Ok(())
}

/// The damped measurement-space iteration for the PrO weight `W` on borrowed `m×m` row-major
/// views (see the module docs).
struct Solver<'a, const M: usize, T> {
    /// `G = H×P̃×Hᵀ` on entry; its lower Cholesky factor `L` after [`prepare`](Self::prepare).
    l: MatrixDataMut<'a, M, M, T>,
    /// The accepted iterate `W`. Holds `M⁻¹` after [`finish`](Self::finish).
    w: MatrixDataMut<'a, M, M, T>,
    /// The iterate to evaluate next. After an accepted step it holds `F(W)`, the trial at
    /// `β = 1`; after [`finish`](Self::finish) it holds `G⁻¹×(G − C)×G⁻¹`.
    trial: MatrixDataMut<'a, M, M, T>,
    /// Scratch for products and Cholesky factors. After a successful
    /// [`evaluate`](Self::evaluate) it holds the Cholesky factor of `S`; after
    /// [`finish`](Self::finish) it holds `S = C + R`.
    work: MatrixDataMut<'a, M, M, T>,
    /// Scratch. After a successful [`evaluate`](Self::evaluate) it holds `C` of the trial.
    c: MatrixDataMut<'a, M, M, T>,
    /// `a = M⁻¹×v` of the last successful [`evaluate`](Self::evaluate).
    a: MatrixDataMut<'a, M, 1, T>,
    /// The measurement noise covariance `R`.
    r: MatrixDataRef<'a, M, M, T>,
    /// The prior innovation `v = z − H×x̃`.
    v: MatrixDataRef<'a, M, 1, T>,
}

impl<'a, const M: usize, T> Solver<'a, M, T>
where
    T: MatrixDataType,
{
    /// Runs the damped iteration from the [`start`](Self::start) for at most `max_iterations`
    /// accepted steps, or until the line search cannot make progress.
    ///
    /// Returns whether it converged, i.e. whether `tr(D×C×D×C) ≤` [`tolerance`] at the result,
    /// or `None` if the start could not be evaluated (only possible through rounding, since
    /// `W = 0` is the prior). A line search that stalls at the working precision ends the
    /// iteration but does not count as convergence on its own: in `f32`, `W` near `−G⁻¹`
    /// cannot be resolved finely enough for huge outliers, and the iteration stalls far from
    /// the solution. The result is in [`w`](Self::w).
    fn solve(&mut self, max_iterations: usize) -> Option<bool> {
        let tolerance = tolerance::<T>();

        let mut objective = self.start()?;
        let mut slope = self.accept();

        for _ in 0..max_iterations {
            if slope.is_below(tolerance) {
                return Some(true);
            }

            let Some(next) = self.line_search(objective, slope) else {
                break;
            };

            objective = next;
            slope = self.accept();
        }

        Some(slope.is_below(tolerance))
    }

    /// Evaluates the start candidates `W = 0` (the prior covariance) and `W = R⁻¹` (the
    /// Kalman filter's covariance) and leaves the one with the lower objective evaluated in
    /// [`trial`](Self::trial). Returns its objective, or `None` if `W = 0` cannot be evaluated.
    ///
    /// From `W = 0`, each step grows `W` by only about `G⁻¹` while `W ≪ R⁻¹`, so a prior much
    /// wider than the noise (`G ≫ R`) would take about `√(G/R)` steps; from `W = R⁻¹`, `W`
    /// roughly halves per step instead.
    fn start(&mut self) -> Option<T> {
        self.trial.clear();
        let prior = self.evaluate()?;

        self.work.as_mut_slice().copy_from_slice(self.r.as_slice());
        if self.work.cholesky_decompose_lower() {
            self.work.invert_l_cholesky(&mut self.trial);
            if let Some(kalman) = self.evaluate() {
                if kalman < prior {
                    return Some(kalman);
                }
            }
        }

        self.trial.clear();
        self.evaluate()
    }

    /// Searches `β = 1, ½, ¼, …` along `D = F(W) − W`, starting from [`trial`](Self::trial)
    /// `= F(W)`, until the PD guard and the Armijo condition hold. If the full step passes
    /// right away, tries to [`expand`](Self::expand) it.
    ///
    /// `slope` is `tr(D×C×D×C)` at `W`. Returns the objective of the accepted trial, or `None`
    /// once halving no longer moves the trial (or it equals `W`) in `T`, or after
    /// [`MAX_HALVINGS`].
    fn line_search(&mut self, objective: T, slope: Slope<T>) -> Option<T> {
        let two = T::one() + T::one();
        let half_sigma = T::from_f32(ARMIJO).unwrap_or_else(T::zero) / two;
        let mut beta = T::one();

        for _ in 0..MAX_HALVINGS {
            if let Some(candidate) = self.evaluate() {
                if slope.armijo_holds(objective, candidate, beta, half_sigma) {
                    if beta == T::one() {
                        return self.expand(objective, slope, candidate);
                    }

                    return Some(candidate);
                }
            }

            beta = beta / two;
            if !self.halve() || self.trial.as_slice() == self.w.as_slice() {
                return None;
            }
        }

        None
    }

    /// Doubles `β = 1, 2, 4, …` of an accepted full step while the objective keeps decreasing
    /// and the Armijo condition still holds, and leaves the best trial evaluated.
    ///
    /// Without this, a prior much wider or much narrower than the noise (`G ≫ R` or
    /// `G ≪ R`) makes every full step tiny: `C` changes by only about `R` (or `G`) per step,
    /// so the iteration would need about `√(G/R)` steps. `objective` and `slope` belong to
    /// `W`; `accepted` is the objective of the full step. Returns the objective of the kept
    /// trial, or `None` if it cannot be evaluated again (only possible through rounding).
    fn expand(&mut self, objective: T, slope: Slope<T>, accepted: T) -> Option<T> {
        let two = T::one() + T::one();
        let half_sigma = T::from_f32(ARMIJO).unwrap_or_else(T::zero) / two;
        let mut best = accepted;
        let mut beta = T::one();

        for _ in 0..MAX_EXPANSIONS {
            self.double();
            beta = beta * two;
            match self.evaluate() {
                Some(candidate)
                    if candidate < best
                        && slope.armijo_holds(objective, candidate, beta, half_sigma) =>
                {
                    best = candidate;
                }
                _ => {
                    self.halve();
                    return self.evaluate();
                }
            }
        }

        Some(best)
    }

    /// Moves [`trial`](Self::trial) to twice its distance from [`w`](Self::w), i.e. doubles
    /// `β`.
    fn double(&mut self) {
        let w = self.w.as_slice();
        for (t, &w) in self.trial.as_mut_slice().iter_mut().zip(w) {
            *t = *t + (*t - w);
        }
    }

    /// Moves [`trial`](Self::trial) halfway towards [`w`](Self::w), i.e. halves `β`.
    ///
    /// Returns `false` if no entry moved, i.e. the trial is within rounding of `W` in `T`
    /// (round-half-to-even can keep `w + ulp` from ever reaching `w`).
    fn halve(&mut self) -> bool {
        let two = T::one() + T::one();
        let w = self.w.as_slice();
        let mut moved = false;
        for (t, &w) in self.trial.as_mut_slice().iter_mut().zip(w) {
            let next = w + (*t - w) / two;
            moved |= next != *t;
            *t = next;
        }

        moved
    }

    /// Evaluates the objective `J` at [`trial`](Self::trial) and leaves `C`, `a` and the
    /// Cholesky factor of `S` of the trial in [`c`](Self::c), [`a`](Self::a) and
    /// [`work`](Self::work).
    ///
    /// Returns `None` if `B = I + Lᵀ×W×L` (the PD guard), `M` or `S` has no Cholesky factor.
    fn evaluate(&mut self) -> Option<T> {
        // work = B = I + Lᵀ*W*L
        self.trial.mult(&self.l, &mut self.c);
        transpose_mult::<M, T>(
            self.l.as_slice(),
            self.c.as_slice(),
            self.work.as_mut_slice(),
        );
        for i in 0..M {
            self.work.set(i, i, self.work.get(i, i) + T::one());
        }

        if !self.work.cholesky_decompose_lower() {
            return None;
        }

        let log_det_b = log_determinant::<M, T>(self.work.as_slice());

        // c = C = L*B⁻¹*Lᵀ
        self.work.invert_l_cholesky(&mut self.c);
        let trace_b_inv = (0..M).map(|i| self.c.get(i, i)).sum::<T>();
        self.l.mult(&self.c, &mut self.work);
        self.work.mult_transb(&self.l, &mut self.c);
        symmetrize::<M, T>(self.c.as_mut_slice());

        // a = M⁻¹*v with M = C + R + L*Lᵀ
        self.work.copy_from(&self.c);
        self.work.add_inplace_a(&self.r);
        self.l.multadd_transb(&self.l, &mut self.work);
        if !self.work.cholesky_decompose_lower() {
            return None;
        }

        self.a.as_mut_slice().copy_from_slice(self.v.as_slice());
        cholesky_solve::<M, T>(self.work.as_slice(), self.a.as_mut_slice());
        let quadratic = dot(self.v.as_slice(), self.a.as_slice());

        // work = chol(S) with S = C + R
        self.work.copy_from(&self.c);
        self.work.add_inplace_a(&self.r);
        if !self.work.cholesky_decompose_lower() {
            return None;
        }

        let log_det_s = log_determinant::<M, T>(self.work.as_slice());
        let two = T::one() + T::one();
        Some((log_det_s + quadratic + trace_b_inv + log_det_b) / two)
    }

    /// Accepts the evaluated [`trial`](Self::trial) as `W`, replaces the trial by `F(W)` and
    /// returns the slope `tr(D×C×D×C)` with `D = F(W) − W`.
    ///
    /// Requires a successful [`evaluate`](Self::evaluate) of the trial right before.
    fn accept(&mut self) -> Slope<T> {
        self.w.copy_from(&self.trial);

        // trial = F(W) = S⁻¹ − a*aᵀ
        self.work.invert_l_cholesky(&mut self.trial);
        for i in 0..M {
            for j in 0..M {
                let value = self.trial.get(i, j) - self.a.get(i, 0) * self.a.get(j, 0);
                self.trial.set(i, j, value);
            }
        }

        // work = D*C; tr(D*C*D*C) = Σ (D*C)ᵢⱼ (D*C)ⱼᵢ
        for i in 0..M {
            for j in 0..M {
                let value = (0..M)
                    .map(|k| (self.trial.get(i, k) - self.w.get(i, k)) * self.c.get(k, j))
                    .sum::<T>();
                self.work.set(i, j, value);
            }
        }

        Slope::of::<M>(self.work.as_slice())
    }

    /// Forms the posterior pieces from the accepted `W`: `G⁻¹×(G − C)×G⁻¹` in
    /// [`trial`](Self::trial), `M⁻¹` in [`w`](Self::w) and `S = C + R` in
    /// [`work`](Self::work).
    ///
    /// Returns `false` if `W` cannot be evaluated again or `M` has no Cholesky factor, which
    /// only rounding can cause.
    fn finish(&mut self) -> bool {
        self.trial.copy_from(&self.w);
        if self.evaluate().is_none() {
            return false;
        }

        // trial = G − C
        self.l.mult_transb(&self.l, &mut self.trial);
        self.trial.sub_inplace_a(&self.c);

        // work = S = C + R; w = M⁻¹ with M = S + G
        self.c.add_inplace_a(&self.r);
        self.work.copy_from(&self.c);
        self.l.multadd_transb(&self.l, &mut self.c);
        if !self.c.cholesky_decompose_lower() {
            return false;
        }

        self.c.invert_l_cholesky(&mut self.w);

        // trial = L⁻ᵀ*L⁻¹*(G − C)*L⁻ᵀ*L⁻¹ = G⁻¹(G − C)G⁻¹
        let l = self.l.as_slice();
        let trial = self.trial.as_mut_slice();
        congruence::<M, T>(l, trial, forward_substitute::<M, T>);
        congruence::<M, T>(l, trial, back_substitute_transposed::<M, T>);
        true
    }
}

/// The slope magnitude `tr(D×C×D×C) = scale² × normalized`, kept factored because it
/// overflows `f32` for outliers beyond about `10⁹` standard deviations.
#[derive(Clone, Copy)]
struct Slope<T> {
    /// The largest magnitude of an entry of `D×C`; zero if `D = 0`.
    scale: T,
    /// `tr(D×C×D×C) / scale²`, at most `M²`.
    normalized: T,
}

impl<T> Slope<T>
where
    T: MatrixDataType,
{
    /// Computes the slope from the row-major `M×M` matrix `x = D×C`, using
    /// `tr(D×C×D×C) = Σ xᵢⱼ·xⱼᵢ`.
    fn of<const M: usize>(x: &[T]) -> Self {
        let scale = x.iter().fold(T::zero(), |max, &value| {
            let magnitude = if value < T::zero() { -value } else { value };
            if magnitude > max {
                magnitude
            } else {
                max
            }
        });

        if scale <= T::zero() {
            return Self {
                scale: T::zero(),
                normalized: T::zero(),
            };
        }

        let normalized = (0..M)
            .flat_map(|i| (0..M).map(move |j| (i, j)))
            .map(|(i, j)| (x[i * M + j] / scale) * (x[j * M + i] / scale))
            .sum::<T>();
        Self { scale, normalized }
    }

    /// Returns whether `tr(D×C×D×C) ≤ tolerance`, for a `tolerance` below one.
    ///
    /// Never forms `scale²`, which overflows (and panics for fixed-point types) for large
    /// slopes: above `scale = 1` it compares `normalized ≤ tolerance/scale²` by division.
    fn is_below(self, tolerance: T) -> bool {
        if self.normalized <= T::zero() {
            return true;
        }

        if self.scale <= T::one() {
            self.scale * self.scale * self.normalized <= tolerance
        } else {
            self.normalized <= tolerance / self.scale / self.scale
        }
    }

    /// Returns whether the trial objective `candidate` satisfies the Armijo condition
    /// `candidate ≤ objective − σ/2 × β × tr(D×C×D×C)` against the objective of `W`.
    ///
    /// Never forms `scale²`. Up to `scale = 1` (the case near convergence), the condition is
    /// evaluated as written, so a decrease below the rounding of `objective` still accepts a
    /// trial that does not increase the objective. Above, `objective − candidate` is divided
    /// by `scale` twice instead. The other factor, `β·σ/2·normalized`, stays below
    /// `2¹⁴·σ/2·M²` (see [`MAX_EXPANSIONS`]), so nothing overflows.
    fn armijo_holds(self, objective: T, candidate: T, beta: T, half_sigma: T) -> bool {
        let weight = beta * half_sigma * self.normalized;
        if self.scale <= T::one() {
            candidate <= objective - weight * self.scale * self.scale
        } else {
            (objective - candidate) / self.scale / self.scale >= weight
        }
    }
}

/// Returns the convergence threshold on `tr(D×C×D×C)`: [`TOLERANCE`], or the machine epsilon
/// `ε` of `T` if that is larger.
///
/// The stationarity residual cannot drop much below `ε` (it is a square of quantities with
/// relative rounding errors of at least `√ε` once `C` and `F(W)` are formed), so `f32`
/// (`ε ≈ 1.2·10⁻⁷`, a relative error of `C` of about `3·10⁻⁴`) and fixed-point types
/// (`ε = 2⁻¹⁶` for `I16F16`) use `ε`, and `f64` uses `10⁻¹⁰`.
fn tolerance<T>() -> T
where
    T: MatrixDataType,
{
    let one = T::one();
    let two = one + one;
    let mut epsilon = one;
    for _ in 0..200 {
        let half = epsilon / two;
        if one + half == one {
            break;
        }

        epsilon = half;
    }

    let tolerance = T::from_f64(TOLERANCE).unwrap_or_else(T::zero);
    if epsilon > tolerance {
        epsilon
    } else {
        tolerance
    }
}

/// Writes `out = aᵀ×b` for row-major `M×M` matrices.
fn transpose_mult<const M: usize, T>(a: &[T], b: &[T], out: &mut [T])
where
    T: MatrixDataType,
{
    for i in 0..M {
        for j in 0..M {
            out[i * M + j] = (0..M).map(|k| a[k * M + i] * b[k * M + j]).sum::<T>();
        }
    }
}

/// Replaces the row-major `M×M` matrix `a` by `(a + aᵀ)/2`.
fn symmetrize<const M: usize, T>(a: &mut [T])
where
    T: MatrixDataType,
{
    let two = T::one() + T::one();
    for i in 0..M {
        for j in (i + 1)..M {
            let mean = (a[i * M + j] + a[j * M + i]) / two;
            a[i * M + j] = mean;
            a[j * M + i] = mean;
        }
    }
}

/// Returns `aᵀ×b` of two vectors.
fn dot<T>(a: &[T], b: &[T]) -> T
where
    T: MatrixDataType,
{
    a.iter().zip(b).map(|(&a, &b)| a * b).sum::<T>()
}

/// Replaces the row-major `M×cols` matrix `b` by `L⁻¹×b`, with `L` the lower triangular `M×M`
/// Cholesky factor.
fn forward_substitute<const M: usize, T>(l: &[T], b: &mut [T], cols: usize)
where
    T: MatrixDataType,
{
    for col in 0..cols {
        for i in 0..M {
            let sum = (0..i).map(|k| l[i * M + k] * b[k * cols + col]).sum::<T>();
            b[i * cols + col] = (b[i * cols + col] - sum) / l[i * M + i];
        }
    }
}

/// Replaces the row-major `M×cols` matrix `b` by `L⁻ᵀ×b`, with `L` the lower triangular `M×M`
/// Cholesky factor.
fn back_substitute_transposed<const M: usize, T>(l: &[T], b: &mut [T], cols: usize)
where
    T: MatrixDataType,
{
    for col in 0..cols {
        for i in (0..M).rev() {
            let sum = ((i + 1)..M)
                .map(|k| l[k * M + i] * b[k * cols + col])
                .sum::<T>();
            b[i * cols + col] = (b[i * cols + col] - sum) / l[i * M + i];
        }
    }
}

/// Solves `(L×Lᵀ)×a = b` in place, with `L` the lower triangular `M×M` Cholesky factor and the
/// `M`-vector `a` holding `b` on entry.
fn cholesky_solve<const M: usize, T>(l: &[T], a: &mut [T])
where
    T: MatrixDataType,
{
    forward_substitute::<M, T>(l, a, 1);
    back_substitute_transposed::<M, T>(l, a, 1);
}

/// Replaces the symmetric row-major `M×M` matrix `a` by `X×a×Xᵀ`, where `solve` replaces its
/// argument `b` by `X×b`: `X×(X×a)ᵀ = X×a×Xᵀ` for symmetric `a`.
fn congruence<const M: usize, T>(l: &[T], a: &mut [T], solve: fn(&[T], &mut [T], usize))
where
    T: MatrixDataType,
{
    solve(l, a, M);
    for i in 0..M {
        for j in (i + 1)..M {
            a.swap(i * M + j, j * M + i);
        }
    }

    solve(l, a, M);
}

/// Returns `log det(L×Lᵀ) = 2·Σ ln Lᵢᵢ` of a lower triangular `M×M` Cholesky factor.
fn log_determinant<const M: usize, T>(l: &[T]) -> T
where
    T: MatrixDataType,
{
    let two = T::one() + T::one();
    two * (0..M).map(|i| ln(l[i * M + i])).sum::<T>()
}

/// Returns the natural logarithm of a positive, finite `x` using only arithmetic.
///
/// Reduces `x` to `[½, 2]` by powers of two and evaluates `ln x = 2·atanh((x−1)/(x+1))`, whose
/// series argument is then at most `⅓` in magnitude. The reduction is capped at 1100 steps
/// (beyond the exponent range of `f64`), so non-finite input terminates with a meaningless
/// result.
fn ln<T>(x: T) -> T
where
    T: MatrixDataType,
{
    let one = T::one();
    let two = one + one;
    let half = two.recip();

    let mut x = x;
    let mut exponent = T::zero();
    for _ in 0..1100 {
        if x > two {
            x = x / two;
            exponent += one;
        } else if x < half {
            x = x * two;
            exponent -= one;
        } else {
            break;
        }
    }

    let ln_2 = two * atanh_series(one / (two + one));
    two * atanh_series((x - one) / (x + one)) + exponent * ln_2
}

/// Returns `atanh t = t + t³/3 + t⁵/5 + …` for `|t| ≤ ⅓`, summed until a term no longer
/// changes the sum in `T` (at most 40 terms, ample for `f64`).
fn atanh_series<T>(t: T) -> T
where
    T: MatrixDataType,
{
    let one = T::one();
    let two = one + one;
    let t2 = t * t;

    let mut power = t;
    let mut denominator = one;
    let mut sum = t;
    for _ in 0..40 {
        power = power * t2;
        denominator += two;
        let next = sum + power / denominator;
        if next == sum {
            break;
        }

        sum = next;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;

    impl<'a, const M: usize, T> Solver<'a, M, T>
    where
        T: MatrixDataType,
    {
        /// Runs [`prepare`] on the solver's own views, with `trial` as scratch and no gate.
        fn prepare(&mut self, nis: &mut T) -> Result<(), CorrectionStatus> {
            super::prepare::<M, T>(
                self.l.as_mut_slice(),
                self.work.as_mut_slice(),
                &mut self.trial.as_mut_slice()[..M],
                self.r.as_slice(),
                self.v.as_slice(),
                nis,
                None,
            )
        }
    }

    /// A singular `G + R` is rejected before the NIS divides by its zero pivot, which would
    /// panic for fixed-point types.
    #[test]
    #[cfg(feature = "fixed")]
    fn singular_noise_is_rejected_in_fixed_point() {
        use fixed::types::I16F16;

        let one = I16F16::from_num(1);
        let mut g = [one; 4];
        let mut work = [I16F16::ZERO; 4];
        let mut scratch = [I16F16::ZERO; 2];
        let r = [one; 4];
        let v = [one, I16F16::ZERO];
        let mut nis = I16F16::ZERO;

        let status = prepare::<2, I16F16>(&mut g, &mut work, &mut scratch, &r, &v, &mut nis, None);

        assert_eq!(status, Err(CorrectionStatus::NotPositiveDefinite));
        assert_eq!(nis, I16F16::ZERO);
    }

    /// Slopes whose square overflows `I16F16` are still compared without panicking.
    #[test]
    #[cfg(feature = "fixed")]
    fn huge_slopes_do_not_overflow_in_fixed_point() {
        use fixed::types::I16F16;

        let slope = Slope {
            scale: I16F16::from_num(30_000),
            normalized: I16F16::from_num(2),
        };
        let tolerance = I16F16::from_num(0.001);
        let half_sigma = I16F16::from_num(0.00005);

        assert!(!slope.is_below(tolerance));
        let (objective, one) = (I16F16::from_num(30_000), I16F16::from_num(1));
        assert!(!slope.armijo_holds(objective, objective - one, one, half_sigma));
        let tiny_step = I16F16::from_num(0.0001);
        assert!(slope.armijo_holds(objective, I16F16::ZERO, tiny_step, half_sigma));
    }

    /// Runs the solver on `G`, `R` and `v` (all `f64`) and returns `C = H×P×Hᵀ` of the result
    /// together with whether it converged.
    fn solve(g: [f64; 4], r: [f64; 4], v: [f64; 2], max_iterations: usize) -> ([f64; 4], bool) {
        let mut buffers = [[0.0; 4]; 5];
        let mut a = [0.0; 2];
        buffers[0] = g;
        let [l, w, trial, work, c] = &mut buffers;
        let mut solver = Solver::<2, f64> {
            l: MatrixData::new_mut::<2, 2, f64>(l),
            w: MatrixData::new_mut::<2, 2, f64>(w),
            trial: MatrixData::new_mut::<2, 2, f64>(trial),
            work: MatrixData::new_mut::<2, 2, f64>(work),
            c: MatrixData::new_mut::<2, 2, f64>(c),
            a: MatrixData::new_mut::<2, 1, f64>(&mut a),
            r: MatrixDataRef::from(&r[..]),
            v: MatrixDataRef::from(&v[..]),
        };

        let mut nis = 0.0;
        assert!(solver.prepare(&mut nis).is_ok());
        let converged = solver.solve(max_iterations).expect("start evaluates");
        solver.trial.copy_from(&solver.w);
        solver.evaluate().expect("result evaluates");

        let mut result = [0.0; 4];
        result.copy_from_slice(solver.c.as_slice());
        (result, converged)
    }

    /// `ln` matches the standard library over many orders of magnitude.
    #[test]
    #[cfg(feature = "std")]
    fn ln_matches_std() {
        for &x in &[1e-300_f64, 1e-8, 0.3, 0.5, 1.0, 2.0, 3.7, 1e5, 1e300] {
            assert!(
                (ln(x) - x.ln()).abs() <= 1e-13 * x.ln().abs().max(1.0),
                "x = {x}"
            );
        }

        for &x in &[1e-30_f32, 0.25, 1.0, 7.0, 1e30] {
            assert!(
                (ln(x) - x.ln()).abs() <= 1e-5 * x.ln().abs().max(1.0),
                "x = {x}"
            );
        }
    }

    /// For diagonal `G` and `R` and an innovation along one axis, `a×aᵀ` stays diagonal and the
    /// vector iteration decouples into scalar problems, whose roots the scalar Newton solver
    /// finds exactly. (An innovation along both axes couples them through `a×aᵀ`.)
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))] // micromath's square root is approximate
    fn diagonal_problem_matches_scalar_newton() {
        let (g, r, v) = ([2.0, 0.5], [0.3, 1.5], [0.0, -4.0]);
        let (c, converged) = solve([g[0], 0.0, 0.0, g[1]], [r[0], 0.0, 0.0, r[1]], v, 500);

        // The convergence threshold corresponds to a relative error of about 1e-5.
        assert!(converged);
        for i in 0..2 {
            let expected = super::super::solve_stationarity(g[i], r[i], v[i], 50);
            assert!(
                (c[i * 3] - expected).abs() < 1e-4 * expected,
                "c[{i}] = {}",
                c[i * 3]
            );
        }

        assert!(c[1].abs() < 1e-9 && c[2].abs() < 1e-9);
    }

    /// The objective's directional derivative along `D` matches `−½·tr(D×C×D×C)`.
    #[test]
    #[cfg(any(feature = "std", feature = "libm"))] // micromath's square root is approximate
    fn slope_matches_finite_difference() {
        let mut buffers = [[0.0; 4]; 5];
        let mut a = [0.0; 2];
        buffers[0] = [2.0, 0.6, 0.6, 1.0];
        let r = [0.5, 0.1, 0.1, 0.8];
        let v = [1.5, -2.0];
        let [l, w, trial, work, c] = &mut buffers;
        let mut solver = Solver::<2, f64> {
            l: MatrixData::new_mut::<2, 2, f64>(l),
            w: MatrixData::new_mut::<2, 2, f64>(w),
            trial: MatrixData::new_mut::<2, 2, f64>(trial),
            work: MatrixData::new_mut::<2, 2, f64>(work),
            c: MatrixData::new_mut::<2, 2, f64>(c),
            a: MatrixData::new_mut::<2, 1, f64>(&mut a),
            r: MatrixDataRef::from(&r[..]),
            v: MatrixDataRef::from(&v[..]),
        };
        let mut nis = 0.0;
        assert!(solver.prepare(&mut nis).is_ok());

        let base = [0.3, 0.05, 0.05, 0.2];
        solver.trial.as_mut_slice().copy_from_slice(&base);
        let objective = solver.evaluate().expect("evaluates");
        let accepted = solver.accept();
        let slope = -accepted.scale * accepted.scale * accepted.normalized / 2.0;
        let mut direction = [0.0; 4];
        for (d, (&f, &w)) in direction
            .iter_mut()
            .zip(solver.trial.as_slice().iter().zip(&base))
        {
            *d = f - w;
        }

        let h = 1e-6;
        for (t, (&w, &d)) in solver
            .trial
            .as_mut_slice()
            .iter_mut()
            .zip(base.iter().zip(&direction))
        {
            *t = w + h * d;
        }

        let numeric = (solver.evaluate().expect("evaluates") - objective) / h;
        assert!(
            (numeric - slope).abs() < 1e-4 * slope.abs(),
            "{numeric} vs {slope}"
        );
    }

    /// Runs a correlated 2×2 problem with the innovation `v` through the whole solver in `T`
    /// and returns whether the posterior pieces are usable: `M⁻¹` and `S` have positive
    /// diagonals and `S` lies above `R`.
    fn smoke<T>(v: [f32; 2]) -> bool
    where
        T: MatrixDataType,
    {
        let value = |v: f32| T::from_f32(v).expect("representable");
        let mut buffers = [[T::zero(); 4]; 5];
        let mut a = [T::zero(); 2];
        buffers[0] = [2.0, 0.6, 0.6, 1.0].map(value);
        let r = [0.5, 0.1, 0.1, 0.8].map(value);
        let v = v.map(value);
        let [l, w, trial, work, c] = &mut buffers;
        let mut solver = Solver::<2, T> {
            l: MatrixData::new_mut::<2, 2, T>(l),
            w: MatrixData::new_mut::<2, 2, T>(w),
            trial: MatrixData::new_mut::<2, 2, T>(trial),
            work: MatrixData::new_mut::<2, 2, T>(work),
            c: MatrixData::new_mut::<2, 2, T>(c),
            a: MatrixData::new_mut::<2, 1, T>(&mut a),
            r: MatrixDataRef::from(&r[..]),
            v: MatrixDataRef::from(&v[..]),
        };

        let mut nis = T::zero();
        let usable = solver.prepare(&mut nis).is_ok()
            && solver.solve(BUDGET_FOR_SMOKE).is_some()
            && solver.finish();
        usable
            && nis >= T::zero()
            && solver.w.get(0, 0) > T::zero()
            && solver.w.get(1, 1) > T::zero()
            && solver.work.get(0, 0) > r[0]
            && solver.work.get(1, 1) > r[3]
    }

    /// Budget of the smoke tests.
    const BUDGET_FOR_SMOKE: usize = 20;

    /// The solver runs end to end in `f32` with every math backend, including `micromath`'s
    /// approximate square root.
    #[test]
    fn f32_smoke() {
        for v in [[0.0, 0.0], [1.5, -2.0], [40.0, 10.0]] {
            assert!(smoke::<f32>(v), "v = {v:?}");
        }
    }

    /// The solver runs end to end in the `I16F16` fixed-point type for moderate innovations.
    #[test]
    #[cfg(feature = "fixed")]
    fn fixed_point_smoke() {
        for v in [[0.0, 0.0], [1.5, -2.0], [5.0, 1.0]] {
            assert!(smoke::<fixed::types::I16F16>(v), "v = {v:?}");
        }
    }
}
