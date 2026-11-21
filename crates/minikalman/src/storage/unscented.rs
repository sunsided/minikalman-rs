//! Storage for the unscented Kalman filter and its observations.

use super::layout::*;
use crate::buffers::types::{
    CrossCovarianceMatrixBuffer, SigmaObservedMatrixBuffer, SigmaPointMatrixBuffer,
    SigmaPropagatedMatrixBuffer, SigmaWeightsVectorBuffer, TempSigmaPMatrixBuffer,
};
use crate::matrix::{MatrixDataMut, MatrixDataType};
use crate::unscented::{UnscentedKalman, UnscentedObservation};

/// Borrowed `STATES` × `STATES` temporary of the unscented filter and its observations.
type TempSigmaPRef<'a, const STATES: usize, T> =
    TempSigmaPMatrixBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>;

/// An [`UnscentedKalman`] filter whose buffers borrow a [`FilterStorage`].
pub type BorrowedKalman<'a, const STATES: usize, const NUM_SIGMA: usize, T> = UnscentedKalman<
    STATES,
    NUM_SIGMA,
    T,
    StateVectorRef<'a, STATES, T>,
    EstimateCovarianceRef<'a, STATES, T>,
    ProcessNoiseRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
    SigmaPointMatrixBuffer<STATES, NUM_SIGMA, T, MatrixDataMut<'a, STATES, NUM_SIGMA, T>>,
    SigmaWeightsVectorBuffer<NUM_SIGMA, T, MatrixDataMut<'a, NUM_SIGMA, 1, T>>,
    SigmaPropagatedMatrixBuffer<STATES, NUM_SIGMA, T, MatrixDataMut<'a, STATES, NUM_SIGMA, T>>,
    TempSigmaPRef<'a, STATES, T>,
>;

/// An [`UnscentedObservation`] whose buffers borrow an [`ObservationStorage`].
pub type BorrowedObservation<
    'a,
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
> = UnscentedObservation<
    STATES,
    OBSERVATIONS,
    NUM_SIGMA,
    T,
    SigmaObservedMatrixBuffer<
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        MatrixDataMut<'a, OBSERVATIONS, NUM_SIGMA, T>,
    >,
    CrossCovarianceMatrixBuffer<
        STATES,
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, STATES, OBSERVATIONS, T>,
    >,
    MeasurementVectorRef<'a, OBSERVATIONS, T>,
    MeasurementNoiseRef<'a, OBSERVATIONS, T>,
    InnovationVectorRef<'a, OBSERVATIONS, T>,
    InnovationCovarianceRef<'a, OBSERVATIONS, T>,
    KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
    TemporarySInvRef<'a, OBSERVATIONS, T>,
    TempSigmaPRef<'a, STATES, T>,
>;

/// Storage for the buffers of an [`UnscentedKalman`] filter with `STATES` states and
/// `NUM_SIGMA` sigma points.
///
/// Holds `x`, `P`, `Q`, the predicted state, the sigma points, their weights, the propagated
/// sigma points and the `STATES` × `STATES` temporary as nested arrays, without allocation.
/// See [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory; the
/// sigma point matrices make this storage grow with `STATES²`, so a `static` is the better place
/// on small targets.
///
/// ## Sigma point count
///
/// Stable Rust cannot compute `NUM_SIGMA` from `STATES`, so it is a parameter and must be
/// `2 × STATES + 1`. [`new`](Self::new) checks this at compile time: a wrong count fails to
/// compile when the storage initializes a `const` or `static`, and when a call to `new` is
/// compiled to code (`cargo build`, not `cargo check`). [`UnscentedKalman::new`] still checks
/// it at runtime as well.
///
/// ```
/// use minikalman::unscented::FilterStorage;
///
/// static STORAGE: FilterStorage<2, 5, f32> = FilterStorage::new(0.0);
/// ```
///
/// ```compile_fail
/// use minikalman::unscented::FilterStorage;
///
/// static STORAGE: FilterStorage<2, 4, f32> = FilterStorage::new(0.0);
/// ```
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::unscented::{FilterStorage, ObservationStorage};
///
/// const STATES: usize = 2;
/// const NUM_SIGMA: usize = 2 * STATES + 1;
///
/// let mut filter_storage = FilterStorage::<STATES, NUM_SIGMA, f32>::new(0.0);
/// let mut measurement_storage = ObservationStorage::<STATES, 1, NUM_SIGMA, f32>::new(0.0);
///
/// // α = 1, β = 2, κ = 1
/// let mut filter = filter_storage.build(1.0, 2.0, 1.0);
/// let mut measurement = measurement_storage.build();
///
/// filter.estimate_covariance_mut().make_identity();
/// measurement.measurement_noise_covariance_mut().apply(|r| r.set_at(0, 0, 0.5));
///
/// filter.predict_nonlinear(|x| x[0] = x[0] + x[1]);
/// measurement.measurement_vector_mut().apply(|z| z[0] = 1.0);
/// filter.correct_sigma_point(&mut measurement, |sigma, observed| {
///     for j in 0..NUM_SIGMA {
///         observed[j] = sigma[j];
///     }
/// });
/// assert!(filter.state_vector().get_row(0) > 0.0);
/// ```
#[allow(non_snake_case)]
pub struct FilterStorage<const STATES: usize, const NUM_SIGMA: usize, T> {
    /// State vector `x`.
    x: [T; STATES],
    /// Estimate covariance matrix `P`.
    P: [[T; STATES]; STATES],
    /// Direct process noise covariance matrix `Q`.
    Q: [[T; STATES]; STATES],
    /// Scratch for the predicted state.
    predicted_x: [T; STATES],
    /// The sigma points, one per column.
    sigma_points: [[T; NUM_SIGMA]; STATES],
    /// The sigma point weights.
    sigma_weights: [T; NUM_SIGMA],
    /// The propagated sigma points, one per column.
    sigma_propagated: [[T; NUM_SIGMA]; STATES],
    /// Scratch for the covariance.
    temp_sigma_P: [[T; STATES]; STATES],
}

impl<const STATES: usize, const NUM_SIGMA: usize, T: Copy> FilterStorage<STATES, NUM_SIGMA, T> {
    /// Fails to compile when `NUM_SIGMA` is not `2 × STATES + 1`.
    const SIGMA_POINTS: () = assert!(
        NUM_SIGMA == 2 * STATES + 1,
        "NUM_SIGMA must be 2 × STATES + 1"
    );

    /// Creates the storage with every entry set to `init`.
    ///
    /// Fails to compile when `NUM_SIGMA` is not `2 × STATES + 1`; see
    /// [Sigma point count](Self#sigma-point-count).
    pub const fn new(init: T) -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::SIGMA_POINTS;
        Self {
            x: [init; STATES],
            P: [[init; STATES]; STATES],
            Q: [[init; STATES]; STATES],
            predicted_x: [init; STATES],
            sigma_points: [[init; NUM_SIGMA]; STATES],
            sigma_weights: [init; NUM_SIGMA],
            sigma_propagated: [[init; NUM_SIGMA]; STATES],
            temp_sigma_P: [[init; STATES]; STATES],
        }
    }
}

impl<const STATES: usize, const NUM_SIGMA: usize, T: MatrixDataType>
    FilterStorage<STATES, NUM_SIGMA, T>
{
    /// Wires an [`UnscentedKalman`] filter whose buffers borrow this storage.
    ///
    /// `alpha`, `beta` and `kappa` are the sigma point parameters; see
    /// [Choosing α, β and κ](UnscentedKalman#choosing-α-β-and-κ).
    pub fn build(
        &mut self,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> BorrowedKalman<'_, STATES, NUM_SIGMA, T> {
        UnscentedKalman::new(
            self.x.as_mut_slice().into(),
            flat(&mut self.P).into(),
            flat(&mut self.Q).into(),
            self.predicted_x.as_mut_slice().into(),
            flat(&mut self.sigma_points).into(),
            self.sigma_weights.as_mut_slice().into(),
            flat(&mut self.sigma_propagated).into(),
            flat(&mut self.temp_sigma_P).into(),
            alpha,
            beta,
            kappa,
        )
    }
}

/// Storage for the buffers of an [`UnscentedObservation`] of `OBSERVATIONS` measurements.
///
/// Holds `z`, `R`, `y`, `S`, `K`, the `S⁻¹` temporary, the observed sigma points, the cross
/// covariance and the `STATES` × `STATES` temporary as nested arrays. `NUM_SIGMA` is checked at
/// compile time as for [`FilterStorage`]. See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
#[allow(non_snake_case)]
pub struct ObservationStorage<
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
> {
    /// Measurement vector `z`.
    z: [T; OBSERVATIONS],
    /// Measurement noise covariance matrix `R`.
    R: [[T; OBSERVATIONS]; OBSERVATIONS],
    /// Innovation vector `y`.
    y: [T; OBSERVATIONS],
    /// Innovation covariance matrix `S`.
    S: [[T; OBSERVATIONS]; OBSERVATIONS],
    /// Kalman gain matrix `K`.
    K: [[T; OBSERVATIONS]; STATES],
    /// Scratch for `S⁻¹`.
    temp_S_inv: [[T; OBSERVATIONS]; OBSERVATIONS],
    /// The sigma points mapped into measurement space, one per column.
    sigma_observed: [[T; NUM_SIGMA]; OBSERVATIONS],
    /// The cross covariance of state and measurement.
    cross_covariance: [[T; OBSERVATIONS]; STATES],
    /// Scratch for the covariance update.
    temp_P: [[T; STATES]; STATES],
}

impl<const STATES: usize, const OBSERVATIONS: usize, const NUM_SIGMA: usize, T: Copy>
    ObservationStorage<STATES, OBSERVATIONS, NUM_SIGMA, T>
{
    /// Fails to compile when `NUM_SIGMA` is not `2 × STATES + 1`.
    const SIGMA_POINTS: () = assert!(
        NUM_SIGMA == 2 * STATES + 1,
        "NUM_SIGMA must be 2 × STATES + 1"
    );

    /// Creates the storage with every entry set to `init`.
    ///
    /// Fails to compile when `NUM_SIGMA` is not `2 × STATES + 1`; see
    /// [Sigma point count](FilterStorage#sigma-point-count).
    pub const fn new(init: T) -> Self {
        #[allow(clippy::let_unit_value)]
        let () = Self::SIGMA_POINTS;
        Self {
            z: [init; OBSERVATIONS],
            R: [[init; OBSERVATIONS]; OBSERVATIONS],
            y: [init; OBSERVATIONS],
            S: [[init; OBSERVATIONS]; OBSERVATIONS],
            K: [[init; OBSERVATIONS]; STATES],
            temp_S_inv: [[init; OBSERVATIONS]; OBSERVATIONS],
            sigma_observed: [[init; NUM_SIGMA]; OBSERVATIONS],
            cross_covariance: [[init; OBSERVATIONS]; STATES],
            temp_P: [[init; STATES]; STATES],
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, const NUM_SIGMA: usize, T: MatrixDataType>
    ObservationStorage<STATES, OBSERVATIONS, NUM_SIGMA, T>
{
    /// Wires an [`UnscentedObservation`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedObservation<'_, STATES, OBSERVATIONS, NUM_SIGMA, T> {
        UnscentedObservation::new(
            self.z.as_mut_slice().into(),
            flat(&mut self.R).into(),
            self.y.as_mut_slice().into(),
            flat(&mut self.S).into(),
            flat(&mut self.K).into(),
            flat(&mut self.temp_S_inv).into(),
            flat(&mut self.sigma_observed).into(),
            flat(&mut self.cross_covariance).into(),
            flat(&mut self.temp_P).into(),
        )
    }
}
