//! # Square-Root Covariance Filter
//!
//! A linear Kalman filter that propagates a square root `S` of the estimate covariance,
//! `P = S×Sᵀ`, instead of `P`. `P` then stays positive semi-definite by construction, and the
//! stored numbers span half the dynamic range: standard deviations instead of variances. This
//! matters most for fixed point, where `I16F16` resolves variances only down to about `1.5e-5`
//! but standard deviations down to that same value.
//!
//! Expect robustness, not lower error: in the fixed-point study of Bhaskara, Majji and Guzmán
//! (Sensors 24(19):6381, 2024), the square-root form kept the variances positive where the
//! conventional filter did not, but both were about equally inaccurate at short word lengths.
//!
//! ## Predict
//!
//! With the lower Cholesky factor `Lq` of `Q` (Tracy, "A Square-Root Kalman Filter Using Only
//! QR Decompositions", arXiv:2208.06452, eq. 20):
//!
//! ```text
//! x⁻ = A×x
//! S⁻ = qr_r([ (A×S)ᵀ ; Lqᵀ ])ᵀ     (2N × N prearray, qr_r is N × N upper triangular)
//! ```
//!
//! `S⁻×S⁻ᵀ = A×S×Sᵀ×Aᵀ + Lq×Lqᵀ = A×P×Aᵀ + Q`, computed without forming either product. See
//! [`MatrixMut::triangularize_upper`](crate::matrix::MatrixMut::triangularize_upper).
//!
//! ## Correct
//!
//! The measurement noise is decorrelated first: `R = Lr×Lrᵀ`, `H' = Lr⁻¹×H`, `z' = Lr⁻¹×z`, so
//! the noise of `z'` is the identity. Each row `h` of `H'` is then applied as a scalar
//! measurement with Potter's update:
//!
//! ```text
//! φ = Sᵀ×hᵀ,   d = φᵀ×φ + 1
//! K = S×φ / d
//! x ← x + K×(z'ᵢ − h×x)
//! S ← S − γ×K×φᵀ,   γ = √d / (√d + 1)
//! ```
//!
//! This is the textbook form with `a = 1/d` and `γ = 1/(1 + √a)`, rewritten so that no small
//! reciprocal is rounded in fixed point. `S` is no longer triangular afterwards, which is fine:
//! the next predict triangularizes it again.
//!
//! ## Buffers of the observation
//!
//! [`SquareRootKalman::correct`] consumes a [`RegularObservation`] and uses only those of its
//! buffers that may not alias each other:
//!
//! | Buffer | Before the gate | After the gate |
//! |---|---|---|
//! | `y` (innovation vector) | `z − H×x` of the prior | unchanged |
//! | `temp_HP` | `H×S` | `H' = Lr⁻¹×H` |
//! | `S` (innovation covariance) | `chol(H×S×Sᵀ×Hᵀ + R)` | `Lr = chol(R)` |
//! | `K` (gain), first `m` entries | `chol(…)⁻¹×y` for the NIS | `z' = Lr⁻¹×z` |
//!
//! The filter's own prearray holds `φ` and the gain vector of the scalar updates. So after a
//! correction, [`kalman_gain`](RegularObservation::kalman_gain) holds scratch values and not a
//! gain. The observation's [noise adaptation](RegularObservation::set_noise_adaptation) is
//! ignored: adapting `R` is not supported in factored form.
//!
//! ## Example
//!
//! ```
//! # #[cfg(feature = "alloc")] {
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build_square_root::<4>();
//! let mut gps = builder.observations().build::<1>();
//!
//! filter.state_transition_mut().apply(|a| {
//!     a.set(0, 0, 1.0);
//!     a.set(0, 1, 0.1);
//!     a.set(1, 1, 1.0);
//! });
//!
//! // Lq = diag(√Qᵢᵢ) for a diagonal Q.
//! filter.process_noise_factor_mut().apply(|lq| lq.make_scalar(0.1));
//! filter.covariance_factor_mut().apply(|s| s.make_scalar(2.0));
//!
//! gps.observation_matrix_mut().set(0, 0, 1.0);
//! gps.measurement_noise_covariance_mut().set(0, 0, 4.0);
//! gps.measurement_vector_mut().set(0, 0, 10.0);
//!
//! assert!(filter.predict());
//! assert!(filter.correct(&mut gps).is_applied());
//!
//! // P = S×Sᵀ for interoperation, e.g. with the smoothers.
//! let mut x = BufferBuilder::state_vector_x::<2>().new();
//! let mut p = BufferBuilder::estimate_covariance_P::<2>().new();
//! filter.state(&mut x, &mut p);
//! assert!(p.get(0, 0) > 0.0 && p.get(0, 0) < 4.0);
//! # }
//! ```

use crate::kalman::*;
use crate::matrix::triangular::{self, Transpose, Triangle};
use crate::matrix::{Matrix, MatrixDataMut, MatrixDataType, MatrixMut};
use crate::observations::regular::RegularObservation;
use core::marker::PhantomData;

/// A linear Kalman filter that propagates a square root `S` of the covariance, `P = S×Sᵀ`.
///
#[doc = diagram!("square-root-loop.mmd")]
/// See the [module documentation](self) for the math and the use of the observation's
/// buffers. Construct it with [`SquareRootKalman::new`] from caller-provided buffers, or with
/// [`KalmanFilterBuilder::build_square_root`](crate::regular::builder::KalmanFilterBuilder::build_square_root)
/// when the `alloc` feature is enabled.
///
/// A control input `B×u` can be added to the state through
/// [`state_vector_mut`](Self::state_vector_mut); control noise `B×Qᵤ×Bᵀ` would need another
/// prearray block and is not supported.
///
/// With fixed-point types, `R` still arrives as a covariance and is factored in every
/// correction, so `I16F16` cannot hold measurement variances below about `1.5e-5`. `Lq` is set
/// as a factor and has no such limit.
#[allow(non_snake_case)]
pub struct SquareRootKalman<const STATES: usize, const TWO_STATES: usize, T, A, X, S, LQ, TempPre> {
    /// State transition matrix (`STATES` × `STATES`).
    A: A,

    /// State vector (`STATES` × `1`).
    x: X,

    /// Square root of the estimate covariance, `P = S×Sᵀ` (`STATES` × `STATES`).
    ///
    /// Lower triangular after [`predict`](Self::predict); a general square matrix after a
    /// correction.
    S: S,

    /// A factor of the process noise, `Q = Lq×Lqᵀ` (`STATES` × `STATES`).
    Lq: LQ,

    /// The prearray `[(A×S)ᵀ; Lqᵀ]` of the predict (`TWO_STATES` × `STATES`).
    ///
    /// Also scratch for [`set_state`](Self::set_state),
    /// [`set_process_noise`](Self::set_process_noise) and the vectors `φ` and `K` of
    /// [`correct`](Self::correct).
    temp_pre: TempPre,

    /// Binds the element type.
    _phantom: PhantomData<T>,
}

impl<const STATES: usize, const TWO_STATES: usize, T, A, X, S, LQ, TempPre>
    SquareRootKalman<STATES, TWO_STATES, T, A, X, S, LQ, TempPre>
where
    T: MatrixDataType,
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    S: EstimateCovarianceMatrix<STATES, T>,
    LQ: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
    TempPre: MatrixMut<TWO_STATES, STATES, T>,
{
    /// Initializes a square-root filter from caller-provided buffers.
    ///
    /// The filter starts with whatever `x`, `S` and `Lq` contain. Use
    /// [`set_state`](Self::set_state) to start from a covariance `P`, and
    /// [`set_process_noise`](Self::set_process_noise) to set `Lq` from `Q`.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `S` - The covariance factor, `P = S×Sᵀ` (`STATES` × `STATES`).
    /// * `Lq` - The process noise factor, `Q = Lq×Lqᵀ` (`STATES` × `STATES`).
    /// * `temp_pre` - The prearray (`TWO_STATES` × `STATES`).
    ///
    /// ## Panics
    /// If `TWO_STATES` is not `2 × STATES` (unless the `no_assert` feature is enabled).
    #[allow(non_snake_case)]
    pub fn new(A: A, x: X, S: S, Lq: LQ, temp_pre: TempPre) -> Self {
        #[cfg(not(feature = "no_assert"))]
        assert_eq!(TWO_STATES, 2 * STATES, "TWO_STATES must be 2 × STATES");

        Self {
            A,
            x,
            S,
            Lq,
            temp_pre,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of states.
    #[inline(always)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Gets a reference to the state transition matrix `A`.
    #[inline(always)]
    pub fn state_transition(&self) -> &A {
        &self.A
    }

    /// Gets a mutable reference to the state transition matrix `A`.
    #[inline(always)]
    pub fn state_transition_mut(&mut self) -> &mut A {
        &mut self.A
    }

    /// Gets a reference to the state vector `x`.
    #[inline(always)]
    pub fn state_vector(&self) -> &X {
        &self.x
    }

    /// Gets a mutable reference to the state vector `x`, e.g. to add a control input `B×u`.
    #[inline(always)]
    pub fn state_vector_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Gets a reference to the covariance factor `S`, with `P = S×Sᵀ`.
    #[inline(always)]
    pub fn covariance_factor(&self) -> &S {
        &self.S
    }

    /// Gets a mutable reference to the covariance factor `S`, with `P = S×Sᵀ`.
    ///
    /// Any square `S` is valid, e.g. `diag(σᵢ)` for independent standard deviations `σᵢ`.
    #[inline(always)]
    pub fn covariance_factor_mut(&mut self) -> &mut S {
        &mut self.S
    }

    /// Gets a reference to the process noise factor `Lq`, with `Q = Lq×Lqᵀ`.
    #[inline(always)]
    pub fn process_noise_factor(&self) -> &LQ {
        &self.Lq
    }

    /// Gets a mutable reference to the process noise factor `Lq`, with `Q = Lq×Lqᵀ`.
    ///
    /// Any square `Lq` with `Lq×Lqᵀ = Q` is valid, including a singular one, e.g. noise on
    /// one state only. For a diagonal `Q`, `Lq = diag(√Qᵢᵢ)`. Setting the factor directly
    /// keeps the square root of `Q` out of the per-step cost.
    #[inline(always)]
    pub fn process_noise_factor_mut(&mut self) -> &mut LQ {
        &mut self.Lq
    }

    /// Sets the process noise factor `Lq` to the lower Cholesky factor of `Q`.
    ///
    /// Returns `false`, leaving `Lq` unchanged, if `Q` is not positive definite. This includes
    /// a positive semi-definite `Q` such as noise on one state only; set such a `Q` through
    /// [`process_noise_factor_mut`](Self::process_noise_factor_mut) instead.
    #[allow(non_snake_case)]
    pub fn set_process_noise<Q>(&mut self, Q: &Q) -> bool
    where
        Q: Matrix<STATES, STATES, T>,
    {
        if !self.cholesky_into_scratch(Q) {
            return false;
        }

        let factor = &self.temp_pre.as_slice()[..STATES * STATES];
        self.Lq.as_mut_slice().copy_from_slice(factor);
        true
    }

    /// Sets the filter from a state `x` and estimate covariance `P`, with `S = chol(P)`.
    ///
    #[doc = diagram!("square-root-loop.mmd", "init")]
    /// Returns `false`, leaving the filter unchanged, if `P` is not positive definite. A
    /// singular `P` can be set as a factor through
    /// [`covariance_factor_mut`](Self::covariance_factor_mut).
    #[allow(non_snake_case)]
    pub fn set_state<V, P>(&mut self, x: &V, P: &P) -> bool
    where
        V: StateVector<STATES, T>,
        P: Matrix<STATES, STATES, T>,
    {
        if !self.cholesky_into_scratch(P) {
            return false;
        }

        let factor = &self.temp_pre.as_slice()[..STATES * STATES];
        self.S.as_mut_slice().copy_from_slice(factor);
        self.x.as_mut_slice().copy_from_slice(x.as_slice());
        true
    }

    /// Writes the state `x` and the estimate covariance `P = S×Sᵀ` into the given buffers,
    /// e.g. to record them for a smoother.
    ///
    #[doc = diagram!("square-root-loop.mmd", "recover")]
    #[allow(non_snake_case)]
    pub fn state<V, P>(&self, x: &mut V, P: &mut P)
    where
        V: StateVectorMut<STATES, T>,
        P: MatrixMut<STATES, STATES, T>,
    {
        x.as_mut_slice().copy_from_slice(self.x.as_slice());
        let S = self.S.as_matrix();
        S.mult_symmetric_transb(S, P);
    }

    /// Copies `m` into the first `STATES × STATES` entries of the prearray and replaces them
    /// by their lower Cholesky factor. Returns `false` if `m` is not positive definite.
    fn cholesky_into_scratch(&mut self, m: &impl Matrix<STATES, STATES, T>) -> bool {
        let scratch = &mut self.temp_pre.as_mut_slice()[..STATES * STATES];
        scratch.copy_from_slice(m.as_slice());
        MatrixDataMut::<STATES, STATES, T>::from(scratch).cholesky_decompose_lower()
    }

    /// Performs the time update: `x⁻ = A×x` and `S⁻ = qr_r([(A×S)ᵀ; Lqᵀ])ᵀ`.
    ///
    #[doc = diagram!("square-root-loop.mmd", "predict")]
    /// Costs one `2N × N` Householder triangularization with `N` square roots, and no
    /// inversion. Afterwards `S` is lower triangular with a non-negative diagonal.
    ///
    /// Returns `false`, leaving the filter unchanged, if the triangularization meets a
    /// non-finite value, e.g. after an overflow.
    pub fn predict(&mut self) -> bool {
        self.fill_prearray();
        if !self.temp_pre.triangularize_upper() {
            return false;
        }

        // S⁻ is the transpose of the triangular top block of the prearray.
        let pre = self.temp_pre.as_slice();
        let s = self.S.as_mut_slice();
        for i in 0..STATES {
            for j in 0..STATES {
                s[i * STATES + j] = pre[j * STATES + i];
            }
        }

        // x⁻ = A×x, with the first row of the prearray as scratch.
        let a = self.A.as_slice();
        let x = self.x.as_mut_slice();
        let ax = &mut self.temp_pre.as_mut_slice()[..STATES];
        for (i, value) in ax.iter_mut().enumerate() {
            *value = (0..STATES).map(|k| a[i * STATES + k] * x[k]).sum::<T>();
        }

        x.copy_from_slice(ax);
        true
    }

    /// Writes the prearray `[(A×S)ᵀ; Lqᵀ]`: row `i` of the top block is column `i` of `A×S`,
    /// row `STATES + i` is column `i` of `Lq`.
    fn fill_prearray(&mut self) {
        let a = self.A.as_slice();
        let s = self.S.as_slice();
        let lq = self.Lq.as_slice();
        let pre = self.temp_pre.as_mut_slice();
        for i in 0..STATES {
            for j in 0..STATES {
                pre[i * STATES + j] = (0..STATES)
                    .map(|k| a[j * STATES + k] * s[k * STATES + i])
                    .sum::<T>();
                pre[(STATES + i) * STATES + j] = lq[j * STATES + i];
            }
        }
    }

    /// Applies a measurement: decorrelates `R`, then runs one Potter update per row of `H`.
    ///
    #[doc = diagram!("square-root-loop.mmd", "correct")]
    /// Writes the prior innovation `z − H×x` into the observation's
    /// [innovation vector](RegularObservation::innovation_vector) first, then computes the
    /// [NIS](RegularObservation::nis) `yᵀ×(H×P×Hᵀ + R)⁻¹×y` from `S` and applies the
    /// observation's [gate](RegularObservation::set_gate), before anything else is changed.
    /// The NIS and gate therefore match those of the regular filter. See the
    /// [module documentation](self) for which observation buffers hold what.
    ///
    /// Costs `OBSERVATIONS` square roots in the Potter updates and two
    /// `OBSERVATIONS` × `OBSERVATIONS` Cholesky decompositions.
    ///
    /// ## Returns
    /// * [`CorrectionStatus::NotPositiveDefinite`] if `H×P×Hᵀ + R` or `R` has no Cholesky
    ///   factor. `x` and `S` are unchanged; the NIS is unchanged if only `R` failed.
    /// * [`CorrectionStatus::Gated`] if the NIS exceeds the gate. `x` and `S` are unchanged.
    /// * [`CorrectionStatus::Applied`] otherwise.
    #[allow(non_snake_case)]
    pub fn correct<
        const OBSERVATIONS: usize,
        H,
        Z,
        R,
        IY,
        IS,
        K,
        TempSInv,
        TempHP,
        TempPHt,
        TempKHP,
    >(
        &mut self,
        observation: &mut RegularObservation<
            STATES,
            OBSERVATIONS,
            T,
            H,
            Z,
            R,
            IY,
            IS,
            K,
            TempSInv,
            TempHP,
            TempPHt,
            TempKHP,
        >,
    ) -> CorrectionStatus
    where
        H: ObservationMatrix<OBSERVATIONS, STATES, T>,
        Z: MeasurementVector<OBSERVATIONS, T>,
        R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
        IY: InnovationVector<OBSERVATIONS, T>,
        IS: InnovationCovarianceMatrix<OBSERVATIONS, T>,
        K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
        TempHP: TemporaryHPMatrix<OBSERVATIONS, STATES, T>,
    {
        let H = observation.H.as_slice();
        let R = observation.R.as_slice();
        let z = observation.z.as_slice();
        let y = observation.y.as_mut_slice();
        let factor = observation.S.as_mut_slice();
        let scratch = &mut observation.K.as_mut_slice()[..OBSERVATIONS];
        let h_prime = observation.temp_HP.as_mut_slice();

        innovation::<STATES, OBSERVATIONS, T>(H, z, self.x.as_slice(), y);

        let Some(nis) = self.innovation_nis::<OBSERVATIONS>(H, R, y, h_prime, factor, scratch)
        else {
            return CorrectionStatus::NotPositiveDefinite;
        };

        if !decorrelate::<STATES, OBSERVATIONS, T>(H, R, z, factor, h_prime, scratch) {
            return CorrectionStatus::NotPositiveDefinite;
        }

        observation.nis = nis;
        if observation.gate.is_some_and(|gate| nis > gate) {
            return CorrectionStatus::Gated;
        }

        for (row, &measurement) in h_prime.chunks_exact(STATES).zip(scratch.iter()) {
            self.potter_update(row, measurement);
        }

        CorrectionStatus::Applied
    }

    /// Returns the NIS `yᵀ×(H×S×Sᵀ×Hᵀ + R)⁻¹×y`, or `None` if `H×P×Hᵀ + R` has no Cholesky
    /// factor.
    ///
    /// `hs` (`m × N`) receives `H×S`, `factor` (`m × m`) the Cholesky factor `Lc` and
    /// `scratch` (`m`) the whitened innovation `Lc⁻¹×y`.
    #[allow(non_snake_case)]
    fn innovation_nis<const OBSERVATIONS: usize>(
        &self,
        H: &[T],
        R: &[T],
        y: &[T],
        hs: &mut [T],
        factor: &mut [T],
        scratch: &mut [T],
    ) -> Option<T> {
        let s = self.S.as_slice();
        for i in 0..OBSERVATIONS {
            for j in 0..STATES {
                hs[i * STATES + j] = (0..STATES)
                    .map(|k| H[i * STATES + k] * s[k * STATES + j])
                    .sum::<T>();
            }
        }

        for i in 0..OBSERVATIONS {
            for j in 0..OBSERVATIONS {
                let product = (0..STATES)
                    .map(|k| hs[i * STATES + k] * hs[j * STATES + k])
                    .sum::<T>();
                factor[i * OBSERVATIONS + j] = product + R[i * OBSERVATIONS + j];
            }
        }

        if !MatrixDataMut::<OBSERVATIONS, OBSERVATIONS, T>::from(&mut *factor)
            .cholesky_decompose_lower()
        {
            return None;
        }

        scratch.copy_from_slice(y);
        triangular::solve_triangular(
            factor,
            OBSERVATIONS,
            Triangle::Lower,
            Transpose::No,
            scratch,
            1,
        );
        Some(scratch.iter().map(|&w| w * w).sum::<T>())
    }

    /// Applies one scalar measurement with unit noise: the row `h` of `H'` and the whitened
    /// measurement `measurement`, with Potter's update of `x` and `S`.
    ///
    /// The first two rows of the prearray hold `φ = Sᵀ×hᵀ` and the gain `K = S×φ / d`.
    fn potter_update(&mut self, h: &[T], measurement: T) {
        let s = self.S.as_mut_slice();
        let x = self.x.as_mut_slice();
        let (phi, rest) = self.temp_pre.as_mut_slice().split_at_mut(STATES);
        let gain = &mut rest[..STATES];

        for (j, value) in phi.iter_mut().enumerate() {
            *value = (0..STATES).map(|k| s[k * STATES + j] * h[k]).sum::<T>();
        }

        let d = phi.iter().map(|&p| p * p).sum::<T>() + T::one();
        let root = d.square_root();
        let gamma = root / (root + T::one());

        for (i, value) in gain.iter_mut().enumerate() {
            *value = (0..STATES).map(|k| s[i * STATES + k] * phi[k]).sum::<T>() / d;
        }

        let residual = measurement - (0..STATES).map(|k| h[k] * x[k]).sum::<T>();
        for i in 0..STATES {
            x[i] += gain[i] * residual;
            for j in 0..STATES {
                s[i * STATES + j] -= gamma * gain[i] * phi[j];
            }
        }
    }
}

/// Writes the innovation `y = z − H×x` of `m` measurements and `N` states.
#[allow(non_snake_case)]
fn innovation<const STATES: usize, const OBSERVATIONS: usize, T>(
    H: &[T],
    z: &[T],
    x: &[T],
    y: &mut [T],
) where
    T: MatrixDataType,
{
    for (i, value) in y.iter_mut().enumerate().take(OBSERVATIONS) {
        *value = z[i] - (0..STATES).map(|k| H[i * STATES + k] * x[k]).sum::<T>();
    }
}

/// Decorrelates the measurement noise: `factor = Lr = chol(R)`, `h_prime = Lr⁻¹×H` and
/// `z_prime = Lr⁻¹×z`.
///
/// Returns `false` if `R` is not positive definite; `h_prime` and `z_prime` are then not
/// written.
#[allow(non_snake_case)]
fn decorrelate<const STATES: usize, const OBSERVATIONS: usize, T>(
    H: &[T],
    R: &[T],
    z: &[T],
    factor: &mut [T],
    h_prime: &mut [T],
    z_prime: &mut [T],
) -> bool
where
    T: MatrixDataType,
{
    factor.copy_from_slice(R);
    if !MatrixDataMut::<OBSERVATIONS, OBSERVATIONS, T>::from(&mut *factor)
        .cholesky_decompose_lower()
    {
        return false;
    }

    h_prime.copy_from_slice(H);
    z_prime.copy_from_slice(z);
    let lower = (Triangle::Lower, Transpose::No);
    triangular::solve_triangular(factor, OBSERVATIONS, lower.0, lower.1, h_prime, STATES)
        && triangular::solve_triangular(factor, OBSERVATIONS, lower.0, lower.1, z_prime, 1)
}
