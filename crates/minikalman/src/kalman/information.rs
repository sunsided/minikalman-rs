//! # Information Filter
//!
//! A linear Kalman filter in information form. Instead of the state `x` and its estimate
//! covariance `P`, it keeps the information matrix `Y = P⁻¹` and the information vector
//! `ŷ = P⁻¹×x`.
//!
//! In this form, independent measurements fuse by addition:
//!
//! ```text
//! Y ← Y + Hᵀ×R⁻¹×H
//! ŷ ← ŷ + Hᵀ×R⁻¹×z
//! ```
//!
//! Each sensor's contribution depends only on its own `H`, `R` and `z`, so sensors can be fused
//! in any order, and a filter can start from no knowledge at all (`Y = 0`, `ŷ = 0`), which the
//! covariance form cannot represent.
//!
//! The price is paid in [`InformationFilter::predict`]: the time update needs `P` back, which
//! costs two `STATES` × `STATES` Cholesky inversions per prediction (see there). Prefer the
//! [`RegularKalman`](crate::regular::RegularKalman) filter when there are few sensors per
//! prediction step and the state is large.
//!
//! ## Example
//!
//! ```
//! # #[cfg(feature = "alloc")] {
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build_information();
//! let mut gps = builder.observations().build::<1>();
//! let mut odometer = builder.observations().build::<1>();
//!
//! filter.state_transition_mut().apply(|a| {
//!     a.set(0, 0, 1.0);
//!     a.set(0, 1, 0.1);
//!     a.set(1, 1, 1.0);
//! });
//! filter.process_noise_mut().make_scalar(0.01);
//!
//! gps.observation_matrix_mut().set(0, 0, 1.0);
//! gps.measurement_noise_covariance_mut().set(0, 0, 4.0);
//! gps.measurement_vector_mut().set(0, 0, 10.0);
//!
//! odometer.observation_matrix_mut().set(0, 1, 1.0);
//! odometer.measurement_noise_covariance_mut().set(0, 0, 0.25);
//! odometer.measurement_vector_mut().set(0, 0, 2.0);
//!
//! // Start from no knowledge (Y = 0, ŷ = 0) and add both sensors.
//! assert!(filter.fuse(&mut gps));
//! assert!(filter.fuse(&mut odometer));
//!
//! // Both states are observed now, so x and P can be recovered.
//! assert!(filter.predict());
//! # }
//! ```

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};
use crate::observations::regular::RegularObservation;
use core::marker::PhantomData;

/// A linear Kalman filter in information form.
///
#[doc = diagram!("information-loop.mmd")]
/// See the [module documentation](self) for the math. Construct it with
/// [`InformationFilter::new`] from caller-provided buffers, or with
/// [`KalmanFilterBuilder::build_information`](crate::regular::builder::KalmanFilterBuilder::build_information)
/// when the `alloc` feature is enabled.
///
/// Every step that inverts a matrix returns `false` and leaves the filter unchanged if that
/// matrix is not positive definite, instead of producing NaN. The matrices must also be
/// finite: as for the other filters, the Cholesky decomposition does not detect `NaN` or
/// infinite entries, which then propagate into `Y` and `ŷ`.
#[allow(non_snake_case)]
pub struct InformationFilter<const STATES: usize, T, A, Y, YH, Q, TempX, TempP, TempL> {
    /// State transition matrix (`STATES` × `STATES`).
    A: A,

    /// Information matrix `Y = P⁻¹` (`STATES` × `STATES`).
    ///
    /// Zero means no knowledge about the state at all.
    Y: Y,

    /// Information vector `ŷ = P⁻¹×x` (`STATES` × `1`).
    y_hat: YH,

    /// Direct process noise matrix (`STATES` × `STATES`).
    Q: Q,

    /// x-sized temporary vector, holding the recovered state during [`InformationFilter::predict`].
    temp_x: TempX,

    /// P-sized temporary matrix, holding the recovered estimate covariance `P` during
    /// [`InformationFilter::predict`] and `Hᵀ×R⁻¹×H` during [`InformationFilter::fuse`].
    temp_P: TempP,

    /// P-sized temporary matrix holding the Cholesky factor of the matrix being inverted.
    temp_L: TempL,

    /// Binds the element type.
    _phantom: PhantomData<T>,
}

impl<const STATES: usize, T, A, Y, YH, Q, TempX, TempP, TempL>
    InformationFilter<STATES, T, A, Y, YH, Q, TempX, TempP, TempL>
where
    T: MatrixDataType,
    A: StateTransitionMatrix<STATES, T>,
    Y: EstimateCovarianceMatrix<STATES, T>,
    YH: StateVectorMut<STATES, T>,
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
    TempX: PredictedStateEstimateVector<STATES, T>,
    TempP: TemporaryStateMatrix<STATES, T>,
    TempL: TemporaryStateMatrix<STATES, T>,
{
    /// Initializes an information filter from caller-provided buffers.
    ///
    /// The filter starts with whatever `Y` and `ŷ` contain; all zeros means no knowledge
    /// about the state. Use [`set_state`](Self::set_state) to start from a known `x` and `P`.
    ///
    /// ## Arguments
    /// * `A` - The state transition matrix (`STATES` × `STATES`).
    /// * `Y` - The information matrix (`STATES` × `STATES`).
    /// * `y_hat` - The information vector (`STATES` × `1`).
    /// * `Q` - The direct process noise matrix (`STATES` × `STATES`).
    /// * `temp_x` - A temporary vector (`STATES` × `1`).
    /// * `temp_P` - A temporary matrix (`STATES` × `STATES`).
    /// * `temp_L` - A second temporary matrix (`STATES` × `STATES`); must not alias `temp_P`.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub fn new(A: A, Y: Y, y_hat: YH, Q: Q, temp_x: TempX, temp_P: TempP, temp_L: TempL) -> Self {
        Self {
            A,
            Y,
            y_hat,
            Q,
            temp_x,
            temp_P,
            temp_L,
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

    /// Gets a reference to the direct process noise matrix `Q`.
    #[inline(always)]
    pub fn process_noise(&self) -> &Q {
        &self.Q
    }

    /// Gets a mutable reference to the direct process noise matrix `Q`.
    #[inline(always)]
    pub fn process_noise_mut(&mut self) -> &mut Q {
        &mut self.Q
    }

    /// Gets a reference to the information matrix `Y = P⁻¹`.
    #[inline(always)]
    pub fn information_matrix(&self) -> &Y {
        &self.Y
    }

    /// Gets a mutable reference to the information matrix `Y = P⁻¹`.
    #[inline(always)]
    pub fn information_matrix_mut(&mut self) -> &mut Y {
        &mut self.Y
    }

    /// Gets a reference to the information vector `ŷ = P⁻¹×x`.
    #[inline(always)]
    pub fn information_vector(&self) -> &YH {
        &self.y_hat
    }

    /// Gets a mutable reference to the information vector `ŷ = P⁻¹×x`.
    #[inline(always)]
    pub fn information_vector_mut(&mut self) -> &mut YH {
        &mut self.y_hat
    }

    /// Sets the filter from a state `x` and estimate covariance `P`: `Y = P⁻¹`, `ŷ = Y×x`.
    ///
    #[doc = diagram!("information-loop.mmd", "init")]
    /// Costs one `STATES` × `STATES` Cholesky inversion. Returns `false`, leaving the filter
    /// unchanged, if `P` is not positive definite.
    #[allow(non_snake_case)]
    pub fn set_state<X, P>(&mut self, x: &X, P: &P) -> bool
    where
        X: StateVector<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        if !invert_positive_definite(
            P.as_matrix(),
            self.temp_L.as_matrix_mut(),
            self.Y.as_matrix_mut(),
        ) {
            return false;
        }

        self.Y
            .as_matrix()
            .mult_rowvector(x.as_matrix(), self.y_hat.as_matrix_mut());

        true
    }

    /// Recovers the state `x = Y⁻¹×ŷ` and estimate covariance `P = Y⁻¹` into the given buffers.
    ///
    #[doc = diagram!("information-loop.mmd", "recover")]
    /// Costs one `STATES` × `STATES` Cholesky inversion. Returns `false`, leaving `x` and `P`
    /// unchanged, if `Y` is not positive definite, e.g. while some state is still unobserved.
    #[allow(non_snake_case)]
    pub fn state<X, P>(&mut self, x: &mut X, P: &mut P) -> bool
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
    {
        if !invert_positive_definite(
            self.Y.as_matrix(),
            self.temp_L.as_matrix_mut(),
            P.as_matrix_mut(),
        ) {
            return false;
        }

        P.as_matrix()
            .mult_rowvector(self.y_hat.as_matrix(), x.as_matrix_mut());

        true
    }

    /// Adds one sensor's measurement: `Y ← Y + Hᵀ×R⁻¹×H`, `ŷ ← ŷ + Hᵀ×R⁻¹×z`.
    ///
    #[doc = diagram!("information-loop.mmd", "fuse1", "fuse2")]
    /// Uses the observation matrix `H`, measurement noise `R` and measurement `z` of a regular
    /// observation; its innovation covariance, inverted-covariance, gain and `P×Hᵀ` buffers
    /// serve as scratch space. Independent sensors can be fused in any order.
    ///
    /// Costs one `OBSERVATIONS` × `OBSERVATIONS` Cholesky inversion and no inversion of `Y`, so
    /// this works from a zero information matrix. Returns `false`, leaving the filter unchanged,
    /// if `R` is not positive definite.
    #[allow(non_snake_case)]
    pub fn fuse<const OBSERVATIONS: usize, H, Z, R, IY, S, K, TempSInv, TempHP, TempPHt, TempKHP>(
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
    ) -> bool
    where
        H: ObservationMatrix<OBSERVATIONS, STATES, T>,
        Z: MeasurementVector<OBSERVATIONS, T>,
        R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
        S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
        K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
        TempPHt: TemporaryPHTMatrix<STATES, OBSERVATIONS, T>,
    {
        let H = observation.H.as_matrix();
        let R_inv = observation.temp_S_inv.as_matrix_mut();

        if !invert_positive_definite(
            observation.R.as_matrix(),
            observation.S.as_matrix_mut(),
            R_inv,
        ) {
            return false;
        }

        // HtRinv = Hᵀ×R⁻¹
        let Ht = observation.K.as_matrix_mut();
        let HtRinv = observation.temp_PHt.as_matrix_mut();
        transpose(H, Ht);
        Ht.mult(R_inv, HtRinv);

        // Y = Y + HtRinv×H
        let HtRinvH = self.temp_P.as_matrix_mut();
        HtRinv.mult(H, HtRinvH);
        self.Y.as_matrix_mut().add_inplace_a(HtRinvH);

        // ŷ = ŷ + HtRinv×z
        HtRinv.multadd_rowvector(observation.z.as_matrix(), self.y_hat.as_matrix_mut());

        true
    }

    /// Performs the time update / prediction step.
    ///
    #[doc = diagram!("information-loop.mmd", "predict")]
    /// The information form has no direct time update, so this recovers the state and
    /// covariance, predicts them, and converts back:
    ///
    /// ```text
    /// P = Y⁻¹,   x = P×ŷ
    /// P⁻ = A×P×Aᵀ + Q,   x⁻ = A×x
    /// Y = (P⁻)⁻¹,   ŷ = Y×x⁻
    /// ```
    ///
    /// ## Cost
    /// Two `STATES` × `STATES` Cholesky inversions (`O(STATES³)` each) on top of the work of a
    /// covariance-form prediction, which inverts nothing.
    ///
    /// Returns `false`, leaving the filter unchanged, if either `Y` or the predicted
    /// covariance `P⁻` is not positive definite. In particular, prediction fails until every
    /// state is observed, e.g. when starting from `Y = 0`.
    #[allow(non_snake_case)]
    pub fn predict(&mut self) -> bool {
        let A = self.A.as_matrix();
        let P = self.temp_P.as_matrix_mut();
        let L = self.temp_L.as_matrix_mut();
        let x = self.temp_x.as_matrix_mut();

        // P = Y⁻¹, x = P×ŷ
        if !invert_positive_definite(self.Y.as_matrix(), L, P) {
            return false;
        }

        P.mult_rowvector(self.y_hat.as_matrix(), x);

        // P⁻ = A×P×Aᵀ + Q, using L as scratch.
        crate::kalman::predict_covariance(A, P, self.Q.as_matrix(), L, None);

        // Y = (P⁻)⁻¹; the last fallible step, so nothing is written before it succeeds.
        if !invert_positive_definite(P, L, self.Y.as_matrix_mut()) {
            return false;
        }

        // ŷ = Y×(A×x), with ŷ holding x⁻ in between.
        let y_hat = self.y_hat.as_matrix_mut();
        A.mult_rowvector(x, y_hat);
        self.Y.as_matrix().mult_rowvector(y_hat, x);
        x.copy_to(y_hat);

        true
    }
}

/// Computes `inverse = m⁻¹` for a symmetric positive definite `m` via its Cholesky factor.
///
/// `scratch` receives the Cholesky factor; `m` itself is left untouched. Returns `false`,
/// leaving `inverse` unchanged, if `m` is not positive definite.
fn invert_positive_definite<const N: usize, T>(
    m: &impl Matrix<N, N, T>,
    scratch: &mut impl MatrixMut<N, N, T>,
    inverse: &mut impl MatrixMut<N, N, T>,
) -> bool
where
    T: MatrixDataType,
{
    m.copy_to(scratch);
    if !scratch.cholesky_decompose_lower() {
        return false;
    }

    scratch.invert_l_cholesky(inverse);

    true
}

/// Writes the transpose of `m` (`ROWS` × `COLS`) into `target` (`COLS` × `ROWS`).
fn transpose<const ROWS: usize, const COLS: usize, T>(
    m: &impl Matrix<ROWS, COLS, T>,
    target: &mut impl MatrixMut<COLS, ROWS, T>,
) where
    T: MatrixDataType,
{
    for row in 0..ROWS {
        for col in 0..COLS {
            target.set(col, row, m.get(row, col));
        }
    }
}
