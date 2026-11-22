//! # UD Filter
//!
//! A linear Kalman filter that keeps the estimate covariance in factored form,
//! `P = U·D·Uᵀ`, with `U` unit upper triangular and `D` diagonal (Bierman, *Factorization
//! Methods for Discrete Sequential Estimation*, 1977; Thornton, *Triangular Covariance
//! Factorizations for Kalman Filtering*, 1976). The steps follow Ramos, Brink, Ganesh and
//! Hurtado, "A summary on the UD Kalman Filter",
//! [arXiv:2203.06105](https://arxiv.org/abs/2203.06105).
//!
//! As long as `D ≥ 0`, `U·D·Uᵀ` is symmetric and positive semi-definite **by construction**,
//! whatever rounding does to the individual entries. Unlike a square-root filter, no step
//! takes a square root: the factorization, the prediction and the update only multiply, add
//! and divide. This makes the UD form the classic choice for fixed-point targets such as
//! `I16F16`.
//!
//! ## Storage
//!
//! `U` and `D` share one `STATES` × `STATES` buffer, the same memory as `P`: `D` on the
//! diagonal (`U`'s diagonal is always `1` and not stored), `U` above it, zeros below.
//! The process noise is kept factored the same way, `Q = U_q·D_q·U_qᵀ`, and factored once
//! with [`UdKalman::set_process_noise`].
//!
//! ## Steps
//!
//! - **Predict** ([`UdKalman::predict`]): Thornton's modified weighted Gram-Schmidt (MWGS)
//!   orthogonalization of `W = [A·U, U_q]` (`STATES` × `2·STATES`) with the weights
//!   `D̆ = diag(D, D_q)`, which yields the factors of `P⁻ = W·D̆·Wᵀ = A·P·Aᵀ + Q`.
//! - **Correct** ([`UdKalman::correct`]): Bierman's scalar update, one measurement row at a
//!   time. A correlated `R` is first decorrelated with its own UD factors,
//!   `R = U_R·D_R·U_Rᵀ`: the rows of `U_R⁻¹·H` and `U_R⁻¹·z` are independent, with variances
//!   `D_R`. `U_R` is unit triangular, so this needs no square root either.
//!
//! ## Example
//!
//! ```
//! # #[cfg(feature = "alloc")] {
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build_ud::<4>();
//! let mut gps = builder.observations().build::<1>();
//!
//! filter.state_transition_mut().apply(|a| {
//!     a.set(0, 0, 1.0);
//!     a.set(0, 1, 0.1);
//!     a.set(1, 1, 1.0);
//! });
//!
//! // Noise on the velocity only: a rank-deficient Q is fine.
//! let mut q = BufferBuilder::direct_process_noise_covariance_Q::<2>().new();
//! q.set(1, 1, 0.01);
//! assert!(filter.set_process_noise(&q));
//!
//! let x = BufferBuilder::state_vector_x::<2>().new();
//! let mut p = BufferBuilder::estimate_covariance_P::<2>().new();
//! p.make_identity();
//! assert!(filter.set_state(&x, &p));
//!
//! gps.observation_matrix_mut().set(0, 0, 1.0);
//! gps.measurement_noise_covariance_mut().set(0, 0, 4.0);
//! gps.measurement_vector_mut().set(0, 0, 1.0);
//!
//! filter.predict();
//! assert!(filter.correct(&mut gps).is_applied());
//!
//! // D stays non-negative, so U·D·Uᵀ stays positive semi-definite.
//! let ud = filter.ud_factors();
//! assert!(ud.get(0, 0) > 0.0 && ud.get(1, 1) > 0.0);
//! # }
//! ```
//!
//! ## Fixed-point range
//!
//! Every intermediate is a sum of products of entries of `A·U`, `U`, `D`, `H` and the
//! innovation. With `I16F16` (range ±32768, resolution `1.5·10⁻⁵`)
//! keep the diagonal of `P`, `Q` and `R` and the products `(A·U)ᵢⱼ²·Dⱼ` well below `10⁴`,
//! and the variances above about `10⁻³`: in the prediction, `wₖ·D̆·wⱼᵀ` sums `2·STATES` such
//! products, and in the update, `α = h·P·hᵀ + r` must fit as well. The `fixed` types panic on
//! overflow in debug builds and wrap in release builds, so scale the state if needed.

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};
use crate::observations::regular::RegularObservation;
use core::marker::PhantomData;

/// A linear Kalman filter with the estimate covariance in UD form, `P = U·D·Uᵀ`.
///
#[doc = diagram!("ud-loop.mmd")]
/// See the [module documentation](self) for the math and the storage layout. Construct it
/// with [`UdKalman::new`] from caller-provided buffers, or with
/// [`KalmanFilterBuilder::build_ud`](crate::regular::builder::KalmanFilterBuilder::build_ud)
/// when the `alloc` feature is enabled.
///
/// `TWO_STATES` sizes the prediction workspace and must be `2 × STATES`.
#[allow(non_snake_case)]
pub struct UdKalman<const STATES: usize, const TWO_STATES: usize, T, A, X, UD, UDQ, TempW, TempD> {
    /// State transition matrix (`STATES` × `STATES`).
    A: A,

    /// State vector (`STATES` × `1`).
    x: X,

    /// UD factors of the estimate covariance (`STATES` × `STATES`): `D` on the diagonal, the
    /// strictly upper part of `U` above it, zeros below.
    UD: UD,

    /// UD factors of the process noise `Q` (`STATES` × `STATES`), in the layout of [`Self::UD`].
    UDQ: UDQ,

    /// Prediction workspace `W = [A·U, U_q]` (`STATES` × `TWO_STATES`).
    temp_W: TempW,

    /// Prediction weights `D̆ = diag(D, D_q)` (`TWO_STATES` × `1`); also holds `A·x` briefly.
    temp_D: TempD,

    /// Binds the element type.
    _phantom: PhantomData<T>,
}

impl<const STATES: usize, const TWO_STATES: usize, T, A, X, UD, UDQ, TempW, TempD>
    UdKalman<STATES, TWO_STATES, T, A, X, UD, UDQ, TempW, TempD>
where
    T: MatrixDataType,
    A: StateTransitionMatrix<STATES, T>,
    X: StateVectorMut<STATES, T>,
    UD: EstimateCovarianceMatrix<STATES, T>,
    UDQ: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
    TempW: MatrixMut<STATES, TWO_STATES, T>,
    TempD: MatrixMut<TWO_STATES, 1, T>,
{
    /// Initializes a UD filter from caller-provided buffers.
    ///
    /// `UD` and `UDQ` must already hold UD factors (all zeros is `P = 0` and `Q = 0`); use
    /// [`set_state`](Self::set_state) and [`set_process_noise`](Self::set_process_noise) to
    /// factor a covariance into them.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `x` - The state vector (`STATES` × `1`).
    /// * `UD` - The UD factors of the estimate covariance (`STATES` × `STATES`).
    /// * `UDQ` - The UD factors of the process noise (`STATES` × `STATES`).
    /// * `temp_W` - The prediction workspace (`STATES` × `TWO_STATES`).
    /// * `temp_D` - The prediction weights (`TWO_STATES` × `1`).
    ///
    /// ## Panics
    /// Panics if `TWO_STATES` is not `2 × STATES`, unless the `no_assert` feature is enabled.
    ///
    /// ## Example
    /// Without allocation, the buffers come from the `impl_buffer_*` macros:
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::prelude::*;
    /// use minikalman::ud::UdKalman;
    ///
    /// const NUM_STATES: usize = 2;
    /// impl_buffer_A!(mut a, NUM_STATES, f32, 0.0);
    /// impl_buffer_x!(mut x, NUM_STATES, f32, 0.0);
    /// impl_buffer_P!(mut ud, NUM_STATES, f32, 0.0);
    /// impl_buffer_Q_direct!(mut udq, NUM_STATES, f32, 0.0);
    /// impl_buffer_ud_W!(mut w, NUM_STATES, f32, 0.0);
    /// impl_buffer_ud_D!(mut d, NUM_STATES, f32, 0.0);
    ///
    /// let mut filter = UdKalman::<NUM_STATES, { 2 * NUM_STATES }, f32, _, _, _, _, _, _>::new(
    ///     a, x, ud, udq, w, d,
    /// );
    /// filter.state_transition_mut().make_identity();
    /// filter.ud_factors_mut().make_identity();
    /// filter.process_noise_factors_mut().set(1, 1, 0.5);
    /// filter.predict();
    /// assert_eq!(filter.ud_factors().get(1, 1), 1.5);
    /// # }
    /// ```
    #[allow(non_snake_case)]
    pub fn new(A: A, x: X, UD: UD, UDQ: UDQ, temp_W: TempW, temp_D: TempD) -> Self {
        #[cfg(not(feature = "no_assert"))]
        assert_eq!(TWO_STATES, 2 * STATES, "TWO_STATES must be 2 × STATES");

        Self {
            A,
            x,
            UD,
            UDQ,
            temp_W,
            temp_D,
            _phantom: PhantomData,
        }
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

    /// Gets a mutable reference to the state vector `x`.
    #[inline(always)]
    pub fn state_vector_mut(&mut self) -> &mut X {
        &mut self.x
    }

    /// Gets a reference to the UD factors of the estimate covariance.
    ///
    /// `D` is on the diagonal, the strictly upper part of `U` above it; the lower part is zero.
    #[inline(always)]
    pub fn ud_factors(&self) -> &UD {
        &self.UD
    }

    /// Gets a mutable reference to the UD factors of the estimate covariance.
    ///
    /// Keep the layout of [`ud_factors`](Self::ud_factors) and the diagonal non-negative.
    #[inline(always)]
    pub fn ud_factors_mut(&mut self) -> &mut UD {
        &mut self.UD
    }

    /// Gets a reference to the UD factors of the process noise `Q = U_q·D_q·U_qᵀ`.
    #[inline(always)]
    pub fn process_noise_factors(&self) -> &UDQ {
        &self.UDQ
    }

    /// Gets a mutable reference to the UD factors of the process noise `Q = U_q·D_q·U_qᵀ`.
    ///
    /// Writing a diagonal `Q` here directly needs no factorization: its UD factors are the
    /// matrix itself.
    #[inline(always)]
    pub fn process_noise_factors_mut(&mut self) -> &mut UDQ {
        &mut self.UDQ
    }

    /// Sets the state `x` and factors the estimate covariance `P` into `U·D·Uᵀ`.
    ///
    #[doc = diagram!("ud-loop.mmd", "init")]
    /// Only the diagonal and the upper triangle of `P` are read. A zero pivot of a positive
    /// semi-definite `P` gives `D[j] = 0` and a unit column `j` of `U`.
    ///
    /// Returns `false` if a pivot came out negative, i.e. `P` is not positive semi-definite
    /// (or rounding pushed a zero pivot of a singular `P` slightly below zero). Unlike
    /// [`InformationFilter::set_state`](crate::information::InformationFilter::set_state),
    /// the filter is still updated in that case, with the negative pivots clamped to zero.
    #[allow(non_snake_case)]
    pub fn set_state<V, P>(&mut self, x: &V, P: &P) -> bool
    where
        V: StateVector<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        x.as_matrix().copy_to(self.x.as_matrix_mut());
        P.as_matrix().copy_to(self.UD.as_matrix_mut());

        factorize_ud(self.UD.as_matrix_mut())
    }

    /// Recovers the state `x` and the estimate covariance `P = U·D·Uᵀ` into the given buffers.
    ///
    #[doc = diagram!("ud-loop.mmd", "recover")]
    /// Costs `O(STATES³)` multiplications and no division.
    #[allow(non_snake_case)]
    pub fn state<V, P>(&self, x: &mut V, P: &mut P)
    where
        V: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        self.x.as_matrix().copy_to(x.as_matrix_mut());
        reconstruct_ud(self.UD.as_matrix(), P.as_matrix_mut());
    }

    /// Factors the process noise `Q` into `U_q·D_q·U_qᵀ`, once, for every following
    /// [`predict`](Self::predict).
    ///
    /// Only the diagonal and the upper triangle of `Q` are read. Rank-deficient `Q`, e.g. noise
    /// on some states only, or `Q = 0`, are fine. Returns `false` under the same condition as
    /// [`set_state`](Self::set_state), with negative pivots clamped to zero.
    #[allow(non_snake_case)]
    pub fn set_process_noise<Q>(&mut self, Q: &Q) -> bool
    where
        Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
    {
        Q.as_matrix().copy_to(self.UDQ.as_matrix_mut());

        factorize_ud(self.UDQ.as_matrix_mut())
    }

    /// Performs the time update / prediction step, `x⁻ = A·x` and `P⁻ = A·P·Aᵀ + Q`.
    ///
    #[doc = diagram!("ud-loop.mmd", "predict")]
    /// Uses Thornton's modified weighted Gram-Schmidt orthogonalization: the rows `wₖ` of
    /// `W = [A·U, U_q]` are orthogonalized from the last to the first under the weights
    /// `D̆ = diag(D, D_q)`:
    ///
    /// ```text
    /// D⁻[k] = wₖ·D̆·wₖᵀ
    /// U⁻[j][k] = (wⱼ·D̆·wₖᵀ) / D⁻[k],   wⱼ ← wⱼ − U⁻[j][k]·wₖ   for j < k
    /// ```
    ///
    /// `D⁻[k]` is a sum of non-negative terms; it is clamped at zero against rounding, and a
    /// zero `D⁻[k]` (e.g. for `P = 0` and `Q = 0`) gives a unit column `k` of `U⁻` instead of
    /// a division by zero. Costs `O(STATES³)` multiplications and `STATES` divisions.
    #[allow(non_snake_case)]
    pub fn predict(&mut self) {
        let A = self.A.as_matrix();
        let UD = self.UD.as_matrix_mut();
        let W = &mut self.temp_W;
        let d = &mut self.temp_D;

        predict_state(A, self.x.as_matrix_mut(), d);
        build_workspace(A, UD, self.UDQ.as_matrix(), W, d);
        weighted_gram_schmidt(W, d, UD);
    }

    /// Performs the measurement update step with Bierman's sequential scalar update.
    ///
    #[doc = diagram!("ud-loop.mmd", "correct")]
    /// The innovation `y = z − H·x` of the prior state is written to the observation's
    /// [innovation vector](RegularObservation::innovation_vector) first. `R` is factored as
    /// `R = U_R·D_R·U_Rᵀ`, and each row `i` of `H' = U_R⁻¹·H` is applied as a scalar
    /// measurement with variance `D_R[i]` (for a diagonal `R`, the rows of `H` with variances
    /// `R[i][i]`). With `f = Uᵀ·h'ᵀ` and `g = D·f`:
    ///
    /// ```text
    /// α₀ = r + g₀·f₀,   D[0] ← D[0]·r/α₀
    /// αⱼ = αⱼ₋₁ + gⱼ·fⱼ,   D[j] ← D[j]·αⱼ₋₁/αⱼ,   Uⱼ ← Uⱼ − fⱼ/αⱼ₋₁·k,   k ← k + gⱼ·Uⱼ(old)
    /// x ← x + k/α·ν
    /// ```
    ///
    /// Every `α ≥ r > 0`, so the update divides only by positive numbers and `D` stays
    /// non-negative. No square root is taken.
    ///
    /// ## NIS and gate
    /// The [NIS](RegularObservation::nis) is the sum `Σ νᵢ²/αᵢ` of the sequential
    /// residuals, which equals `yᵀ·S⁻¹·y`. If a [gate](RegularObservation::set_gate) is set,
    /// the full `S = H·U·D·Uᵀ·Hᵀ + R` is formed and UD-factored first, also without a square
    /// root, so that a gated measurement leaves the filter untouched.
    ///
    /// ## Status
    /// Returns [`CorrectionStatus::NotPositiveDefinite`], leaving `x` and the factors
    /// unchanged, if `R` (or, with a gate, `S`) is not positive definite, and
    /// [`CorrectionStatus::Gated`] if the NIS exceeds the gate.
    ///
    /// ## Buffers
    /// The observation's `S`, `K`, `temp_S_inv`, `temp_HP` and `temp_PHt` serve as scratch
    /// space and hold intermediate values afterwards; `K` is **not** the Kalman gain. They must
    /// not alias each other. [Noise adaptation](RegularObservation::set_noise_adaptation) is
    /// not supported and ignored.
    #[allow(non_snake_case)]
    pub fn correct<
        const OBSERVATIONS: usize,
        H,
        Z,
        R,
        IY,
        S,
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
            S,
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
        S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
        K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
        TempHP: TemporaryHPMatrix<OBSERVATIONS, STATES, T>,
        TempPHt: TemporaryPHTMatrix<STATES, OBSERVATIONS, T>,
    {
        let H = observation.H.as_matrix();
        let y = observation.y.as_matrix_mut();
        let UD = self.UD.as_matrix_mut();
        let x = self.x.as_matrix_mut();

        // y = z - H·x
        H.mult_rowvector(&*x, y);
        observation.z.as_matrix().sub_inplace_b(y);

        // R = U_R·D_R·U_Rᵀ, held in temp_S_inv; checked before anything is written.
        let UD_R = observation.temp_S_inv.as_matrix_mut();
        observation.R.as_matrix().copy_to(UD_R);
        factorize_ud(UD_R);
        if !has_positive_diagonal(UD_R) {
            return CorrectionStatus::NotPositiveDefinite;
        }

        let HP = observation.temp_HP.as_matrix_mut();
        let S = observation.S.as_matrix_mut();
        if let Some(gate) = observation.gate {
            let Some(nis) = full_nis(UD, H, observation.R.as_matrix(), y, HP, S) else {
                return CorrectionStatus::NotPositiveDefinite;
            };

            observation.nis = nis;
            if nis > gate {
                return CorrectionStatus::Gated;
            }
        }

        // H' = U_R⁻¹·H in temp_HP, y' = U_R⁻¹·y in the first column of S.
        H.copy_to(HP);
        for col in 0..STATES {
            solve_unit_upper(UD_R, HP, col);
        }

        for i in 0..OBSERVATIONS {
            S.set(i, 0, y.get(i, 0));
        }

        solve_unit_upper(UD_R, S, 0);

        let f = observation.K.as_matrix_mut();
        let k = observation.temp_PHt.as_matrix_mut();
        let mut nis = T::zero();
        for row in 0..OBSERVATIONS {
            let nu = S.get(row, 0);
            let alpha = bierman_update(UD, x, HP, row, UD_R.get(row, row), nu, f, k);
            nis += nu * (nu / alpha);
            update_residuals(HP, k, S, row);
        }

        if observation.gate.is_none() {
            observation.nis = nis;
        }

        CorrectionStatus::Applied
    }
}

/// Factors a symmetric positive semi-definite matrix in place into `M = U·D·Uᵀ`.
///
/// Reads the diagonal and the upper triangle of `m`, and leaves `D` on the diagonal, the
/// strictly upper part of the unit upper triangular `U` above it, and zeros below. Columns are
/// processed from the last to the first:
///
/// ```text
/// D[j] = M[j][j],   U[i][j] = M[i][j] / D[j],   M[k][i] −= U[k][j]·D[j]·U[i][j]   (k ≤ i < j)
/// ```
///
/// A pivot `D[j] ≤ 0` is set to zero and column `j` of `U` is left as the unit column, so
/// nothing is divided by zero. Returns `false` if a pivot was negative.
pub(crate) fn factorize_ud<const N: usize, T>(m: &mut impl MatrixMut<N, N, T>) -> bool
where
    T: MatrixDataType,
{
    let mut semi_definite = true;
    for j in (0..N).rev() {
        let d = m.get(j, j);
        if d > T::zero() {
            eliminate_column(m, j, d);
        } else {
            semi_definite &= d >= T::zero();
            m.set(j, j, T::zero());
            for i in 0..j {
                m.set(i, j, T::zero());
            }
        }

        for i in 0..j {
            m.set(j, i, T::zero());
        }
    }

    semi_definite
}

/// Computes column `j` of `U` for the positive pivot `d = D[j]` and removes its contribution
/// from the upper triangle of the leading `j` × `j` block.
fn eliminate_column<const N: usize, T>(m: &mut impl MatrixMut<N, N, T>, j: usize, d: T)
where
    T: MatrixDataType,
{
    for i in 0..j {
        m.set(i, j, m.get(i, j) / d);
    }

    for i in 0..j {
        let du = d * m.get(i, j);
        for k in 0..=i {
            m.set(k, i, m.get(k, i) - m.get(k, j) * du);
        }
    }
}

/// Writes `P = U·D·Uᵀ` from the UD factors `ud` into `p`.
pub(crate) fn reconstruct_ud<const N: usize, T>(
    ud: &impl Matrix<N, N, T>,
    p: &mut impl MatrixMut<N, N, T>,
) where
    T: MatrixDataType,
{
    for i in 0..N {
        for j in i..N {
            // U[i][k]·U[j][k] is non-zero only for k ≥ j ≥ i.
            let mut sum = ud.get(j, j) * unit_upper(ud, i, j);
            for k in (j + 1)..N {
                sum += ud.get(i, k) * ud.get(k, k) * ud.get(j, k);
            }

            p.set(i, j, sum);
            p.set(j, i, sum);
        }
    }
}

/// Returns entry `(row, col)` of the unit upper triangular `U` stored in the UD buffer `ud`.
fn unit_upper<const N: usize, T>(ud: &impl Matrix<N, N, T>, row: usize, col: usize) -> T
where
    T: MatrixDataType,
{
    if row == col {
        T::one()
    } else if row < col {
        ud.get(row, col)
    } else {
        T::zero()
    }
}

/// Returns whether every diagonal entry of `m` is strictly positive.
fn has_positive_diagonal<const N: usize, T>(m: &impl Matrix<N, N, T>) -> bool
where
    T: MatrixDataType,
{
    (0..N).all(|i| m.get(i, i) > T::zero())
}

/// Solves `U·e = b` in place for column `col` of `b`, with `U` the unit upper triangular
/// factor stored in `ud`, by back substitution from the last row.
fn solve_unit_upper<const M: usize, const C: usize, T>(
    ud: &impl Matrix<M, M, T>,
    b: &mut impl MatrixMut<M, C, T>,
    col: usize,
) where
    T: MatrixDataType,
{
    for i in (0..M).rev() {
        let mut value = b.get(i, col);
        for j in (i + 1)..M {
            value -= ud.get(i, j) * b.get(j, col);
        }

        b.set(i, col, value);
    }
}

/// Predicts the state, `x ← A·x`, using the first `STATES` entries of `scratch`.
fn predict_state<const STATES: usize, const TWO_STATES: usize, T>(
    a: &impl Matrix<STATES, STATES, T>,
    x: &mut impl MatrixMut<STATES, 1, T>,
    scratch: &mut impl MatrixMut<TWO_STATES, 1, T>,
) where
    T: MatrixDataType,
{
    for row in 0..STATES {
        let mut sum = T::zero();
        for col in 0..STATES {
            sum += a.get(row, col) * x.get(col, 0);
        }

        scratch.set(row, 0, sum);
    }

    for row in 0..STATES {
        x.set(row, 0, scratch.get(row, 0));
    }
}

/// Fills the MWGS workspace `W = [A·U, U_q]` and the weights `D̆ = diag(D, D_q)`.
#[allow(non_snake_case)]
fn build_workspace<const STATES: usize, const TWO_STATES: usize, T>(
    A: &impl Matrix<STATES, STATES, T>,
    UD: &impl Matrix<STATES, STATES, T>,
    UDQ: &impl Matrix<STATES, STATES, T>,
    W: &mut impl MatrixMut<STATES, TWO_STATES, T>,
    d: &mut impl MatrixMut<TWO_STATES, 1, T>,
) where
    T: MatrixDataType,
{
    for i in 0..STATES {
        d.set(i, 0, UD.get(i, i));
        d.set(STATES + i, 0, UDQ.get(i, i));
    }

    for row in 0..STATES {
        for col in 0..STATES {
            let mut sum = A.get(row, col);
            for k in 0..col {
                sum += A.get(row, k) * UD.get(k, col);
            }

            W.set(row, col, sum);
            W.set(row, STATES + col, unit_upper(UDQ, row, col));
        }
    }
}

/// Orthogonalizes the rows of `W` under the weights `d` (Thornton's MWGS) and writes the UD
/// factors of `W·diag(d)·Wᵀ` into `UD`.
#[allow(non_snake_case)]
fn weighted_gram_schmidt<const STATES: usize, const TWO_STATES: usize, T>(
    W: &mut impl MatrixMut<STATES, TWO_STATES, T>,
    d: &impl Matrix<TWO_STATES, 1, T>,
    UD: &mut impl MatrixMut<STATES, STATES, T>,
) where
    T: MatrixDataType,
{
    for k in (0..STATES).rev() {
        let mut dk = weighted_dot(W, d, k, k);
        if dk < T::zero() {
            dk = T::zero();
        }

        UD.set(k, k, dk);
        for j in 0..k {
            let u = if dk > T::zero() {
                weighted_dot(W, d, j, k) / dk
            } else {
                T::zero()
            };

            for l in 0..TWO_STATES {
                W.set(j, l, W.get(j, l) - u * W.get(k, l));
            }

            UD.set(j, k, u);
            UD.set(k, j, T::zero());
        }
    }
}

/// Returns `wᵢ·diag(d)·wⱼᵀ` for the rows `i` and `j` of `w`.
fn weighted_dot<const STATES: usize, const TWO_STATES: usize, T>(
    w: &impl Matrix<STATES, TWO_STATES, T>,
    d: &impl Matrix<TWO_STATES, 1, T>,
    i: usize,
    j: usize,
) -> T
where
    T: MatrixDataType,
{
    let mut sum = T::zero();
    for l in 0..TWO_STATES {
        sum += w.get(i, l) * (d.get(l, 0) * w.get(j, l));
    }

    sum
}

/// Computes the NIS `yᵀ·S⁻¹·y` from the full `S = H·U·D·Uᵀ·Hᵀ + R`, without a square root.
///
/// Forms `H·U` in `hu` and `S` in `s`, UD-factors `S` in place and solves `U_S·e = y` in the
/// first column of `hu`, giving `NIS = Σ eᵢ²/D_S[i]`. Returns `None` if `S` is not positive
/// definite.
#[allow(non_snake_case)]
fn full_nis<const STATES: usize, const OBSERVATIONS: usize, T>(
    UD: &impl Matrix<STATES, STATES, T>,
    H: &impl Matrix<OBSERVATIONS, STATES, T>,
    R: &impl Matrix<OBSERVATIONS, OBSERVATIONS, T>,
    y: &impl Matrix<OBSERVATIONS, 1, T>,
    hu: &mut impl MatrixMut<OBSERVATIONS, STATES, T>,
    s: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
) -> Option<T>
where
    T: MatrixDataType,
{
    for row in 0..OBSERVATIONS {
        for col in 0..STATES {
            let mut sum = H.get(row, col);
            for k in 0..col {
                sum += H.get(row, k) * UD.get(k, col);
            }

            hu.set(row, col, sum);
        }
    }

    for i in 0..OBSERVATIONS {
        for j in i..OBSERVATIONS {
            let mut sum = R.get(i, j);
            for c in 0..STATES {
                sum += hu.get(i, c) * (UD.get(c, c) * hu.get(j, c));
            }

            s.set(i, j, sum);
        }
    }

    factorize_ud(s);
    if !has_positive_diagonal(s) {
        return None;
    }

    for i in 0..OBSERVATIONS {
        hu.set(i, 0, y.get(i, 0));
    }

    solve_unit_upper(s, hu, 0);

    let mut nis = T::zero();
    for i in 0..OBSERVATIONS {
        let e = hu.get(i, 0);
        nis += e * (e / s.get(i, i));
    }

    Some(nis)
}

/// Applies Bierman's update for the scalar measurement `ν = h·Δx + v` with variance `r > 0`,
/// where `h` is row `row` of `h_rows`, to the factors `UD` and the state `x`.
///
/// Uses the first column of `f` for `f = Uᵀ·hᵀ` and the first column of `k` for the
/// unnormalized gain; on return, `k` holds the state correction `Δx = k/α·ν`. Returns the
/// innovation variance `α = h·U·D·Uᵀ·hᵀ + r` of the prior.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub(crate) fn bierman_update<const STATES: usize, const OBSERVATIONS: usize, T>(
    UD: &mut impl MatrixMut<STATES, STATES, T>,
    x: &mut impl MatrixMut<STATES, 1, T>,
    h_rows: &impl Matrix<OBSERVATIONS, STATES, T>,
    row: usize,
    r: T,
    nu: T,
    f: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
    k: &mut impl MatrixMut<STATES, OBSERVATIONS, T>,
) -> T
where
    T: MatrixDataType,
{
    // f = Uᵀ·hᵀ
    for j in 0..STATES {
        let mut sum = h_rows.get(row, j);
        for i in 0..j {
            sum += UD.get(i, j) * h_rows.get(row, i);
        }

        f.set(j, 0, sum);
    }

    let f0 = f.get(0, 0);
    let g0 = UD.get(0, 0) * f0;
    let mut alpha = r + g0 * f0;
    UD.set(0, 0, scale_pivot(UD.get(0, 0), r, g0 * f0, alpha));
    k.set(0, 0, g0);

    for j in 1..STATES {
        let fj = f.get(j, 0);
        let dj = UD.get(j, j);
        let gj = dj * fj;
        let alpha_prev = alpha;
        alpha += gj * fj;
        UD.set(j, j, scale_pivot(dj, alpha_prev, gj * fj, alpha));

        // U[i][j] − fⱼ·(k[i]/αⱼ₋₁): k/αⱼ₋₁ is the gain of the first j states and well scaled,
        // while fⱼ/αⱼ₋₁ or fⱼ·k[i] alone can fall below the resolution of a fixed-point type.
        for i in 0..j {
            let u = UD.get(i, j);
            UD.set(i, j, u - fj * (k.get(i, 0) / alpha_prev));
            k.set(i, 0, k.get(i, 0) + gj * u);
        }

        k.set(j, 0, gj);
    }

    // Δx = k/α·ν, kept in k for the residuals of the remaining rows.
    for i in 0..STATES {
        let dx = (k.get(i, 0) / alpha) * nu;
        x.set(i, 0, x.get(i, 0) + dx);
        k.set(i, 0, dx);
    }

    alpha
}

/// Returns the updated pivot `d·num/α`, where `α = num + part` with `num > 0` and `part ≥ 0`.
///
/// Divides the larger of `num` and `part` by `α` first, so that the ratio is at least `1/2`:
/// `d·(num/α)` if `num` dominates, else `num·(d/α)`. In a fixed-point type, the ratio of the
/// smaller part to `α` (e.g. `r/α` for a precise sensor) can fall below the resolution and
/// zero the pivot; `d/α ≤ 1/f²` for the `part = d·f²` of the caller stays in range unless `f`
/// is very small while `part` still dominates.
fn scale_pivot<T>(d: T, num: T, part: T, alpha: T) -> T
where
    T: MatrixDataType,
{
    if num >= part {
        d * (num / alpha)
    } else {
        num * (d / alpha)
    }
}

/// Moves the decorrelated residuals of the rows after `row` (first column of `residuals`)
/// to the corrected state: `νₗ ← νₗ − h'ₗ·Δx`.
fn update_residuals<const STATES: usize, const OBSERVATIONS: usize, T>(
    h_rows: &impl Matrix<OBSERVATIONS, STATES, T>,
    dx: &impl Matrix<STATES, OBSERVATIONS, T>,
    residuals: &mut impl MatrixMut<OBSERVATIONS, OBSERVATIONS, T>,
    row: usize,
) where
    T: MatrixDataType,
{
    for l in (row + 1)..OBSERVATIONS {
        let mut value = residuals.get(l, 0);
        for c in 0..STATES {
            value -= h_rows.get(l, c) * dx.get(c, 0);
        }

        residuals.set(l, 0, value);
    }
}

#[cfg(all(test, any(feature = "std", feature = "libm", feature = "micromath")))]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;

    /// Asserts that `U·D·Uᵀ` from factoring `m` reproduces `m`.
    fn assert_round_trip<const N: usize, const TOTAL: usize>(values: [f64; TOTAL]) {
        let original = MatrixData::new_array::<N, N, TOTAL, f64>(values);
        let mut ud = MatrixData::new_array::<N, N, TOTAL, f64>(values);
        let mut back = MatrixData::new_array::<N, N, TOTAL, f64>([0.0; TOTAL]);

        assert!(factorize_ud(&mut ud));
        reconstruct_ud(&ud, &mut back);

        for i in 0..N {
            assert!(ud.get(i, i) >= 0.0);
            for j in 0..N {
                let (a, e) = (back.get(i, j), original.get(i, j));
                assert!((a - e).abs() < 1e-12, "({i}, {j}): {a} != {e}");
                if j < i {
                    assert_eq!(ud.get(i, j), 0.0, "lower part must be zero");
                }
            }
        }
    }

    #[test]
    fn factorization_round_trips_spd() {
        assert_round_trip::<3, 9>([4.0, 1.0, 0.5, 1.0, 3.0, -0.2, 0.5, -0.2, 2.0]);
    }

    #[test]
    fn factorization_round_trips_rank_one() {
        // g·gᵀ with g = (1, 2, -1): pivots 0 below the last one.
        assert_round_trip::<3, 9>([1.0, 2.0, -1.0, 2.0, 4.0, -2.0, -1.0, -2.0, 1.0]);
    }

    #[test]
    fn factorization_round_trips_zero() {
        assert_round_trip::<2, 4>([0.0; 4]);
    }

    #[test]
    fn factorization_reports_negative_pivot() {
        let mut ud = MatrixData::new_array::<2, 2, 4, f64>([1.0, 0.0, 0.0, -1.0]);
        assert!(!factorize_ud(&mut ud));
        assert_eq!(ud.get(1, 1), 0.0, "negative pivot must be clamped");
        assert_eq!(ud.get(0, 0), 1.0);
    }

    #[test]
    fn unit_upper_solve_inverts_factor() {
        // U = [[1, 2], [0, 1]]: U·(−3, 2) = (1, 2).
        let ud = MatrixData::new_array::<2, 2, 4, f64>([5.0, 2.0, 0.0, 7.0]);
        let mut b = MatrixData::new_array::<2, 1, 2, f64>([1.0, 2.0]);
        solve_unit_upper(&ud, &mut b, 0);
        assert_eq!((b.get(0, 0), b.get(1, 0)), (-3.0, 2.0));
    }
}
