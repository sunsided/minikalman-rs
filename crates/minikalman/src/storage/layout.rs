//! The buffer layouts shared by several filter kinds, and the borrowed buffer types they wire.

use crate::buffers::types::{
    DirectProcessNoiseCovarianceMatrixMutBuffer, EstimateCovarianceMatrixBuffer,
    InnovationCovarianceMatrixBuffer, InnovationVectorBuffer, KalmanGainMatrixBuffer,
    MeasurementNoiseCovarianceMatrixBuffer, MeasurementVectorBuffer, ObservationMatrixMutBuffer,
    PredictedStateEstimateVectorBuffer, StateTransitionMatrixMutBuffer, StateVectorBuffer,
    TemporaryHPMatrixBuffer, TemporaryKHPMatrixBuffer, TemporaryPHTMatrixBuffer,
    TemporaryResidualCovarianceInvertedMatrixBuffer, TemporaryStateMatrixBuffer,
};
use crate::matrix::MatrixDataMut;

/// Borrowed state transition matrix `A` (`STATES` × `STATES`).
pub type StateTransitionRef<'a, const STATES: usize, T> =
    StateTransitionMatrixMutBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>;

/// Borrowed state vector `x` (`STATES` × `1`).
pub type StateVectorRef<'a, const STATES: usize, T> =
    StateVectorBuffer<STATES, T, MatrixDataMut<'a, STATES, 1, T>>;

/// Borrowed estimate covariance matrix `P` (`STATES` × `STATES`).
pub type EstimateCovarianceRef<'a, const STATES: usize, T> =
    EstimateCovarianceMatrixBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>;

/// Borrowed direct process noise covariance matrix `Q` (`STATES` × `STATES`).
pub type ProcessNoiseRef<'a, const STATES: usize, T> =
    DirectProcessNoiseCovarianceMatrixMutBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>;

/// Borrowed predicted state vector (`STATES` × `1`).
pub type PredictedStateRef<'a, const STATES: usize, T> =
    PredictedStateEstimateVectorBuffer<STATES, T, MatrixDataMut<'a, STATES, 1, T>>;

/// Borrowed temporary `STATES` × `STATES` matrix of the filter.
pub type TemporaryStateRef<'a, const STATES: usize, T> =
    TemporaryStateMatrixBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>;

/// Borrowed observation matrix `H` (`OBSERVATIONS` × `STATES`).
pub type ObservationMatrixRef<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    ObservationMatrixMutBuffer<OBSERVATIONS, STATES, T, MatrixDataMut<'a, OBSERVATIONS, STATES, T>>;

/// Borrowed measurement vector `z` (`OBSERVATIONS` × `1`).
pub type MeasurementVectorRef<'a, const OBSERVATIONS: usize, T> =
    MeasurementVectorBuffer<OBSERVATIONS, T, MatrixDataMut<'a, OBSERVATIONS, 1, T>>;

/// Borrowed measurement noise covariance matrix `R` (`OBSERVATIONS` × `OBSERVATIONS`).
pub type MeasurementNoiseRef<'a, const OBSERVATIONS: usize, T> =
    MeasurementNoiseCovarianceMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, OBSERVATIONS, OBSERVATIONS, T>,
    >;

/// Borrowed innovation vector `y` (`OBSERVATIONS` × `1`).
pub type InnovationVectorRef<'a, const OBSERVATIONS: usize, T> =
    InnovationVectorBuffer<OBSERVATIONS, T, MatrixDataMut<'a, OBSERVATIONS, 1, T>>;

/// Borrowed innovation covariance matrix `S` (`OBSERVATIONS` × `OBSERVATIONS`).
pub type InnovationCovarianceRef<'a, const OBSERVATIONS: usize, T> =
    InnovationCovarianceMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, OBSERVATIONS, OBSERVATIONS, T>,
    >;

/// Borrowed Kalman gain matrix `K` (`STATES` × `OBSERVATIONS`).
pub type KalmanGainRef<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    KalmanGainMatrixBuffer<STATES, OBSERVATIONS, T, MatrixDataMut<'a, STATES, OBSERVATIONS, T>>;

/// Borrowed temporary for `S⁻¹` (`OBSERVATIONS` × `OBSERVATIONS`).
pub type TemporarySInvRef<'a, const OBSERVATIONS: usize, T> =
    TemporaryResidualCovarianceInvertedMatrixBuffer<
        OBSERVATIONS,
        T,
        MatrixDataMut<'a, OBSERVATIONS, OBSERVATIONS, T>,
    >;

/// Borrowed temporary for `H×P` (`OBSERVATIONS` × `STATES`).
pub type TemporaryHPRef<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    TemporaryHPMatrixBuffer<OBSERVATIONS, STATES, T, MatrixDataMut<'a, OBSERVATIONS, STATES, T>>;

/// Borrowed temporary for `P×Hᵀ` (`STATES` × `OBSERVATIONS`).
pub type TemporaryPHtRef<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    TemporaryPHTMatrixBuffer<STATES, OBSERVATIONS, T, MatrixDataMut<'a, STATES, OBSERVATIONS, T>>;

/// Borrowed temporary for `K×H×P` (`STATES` × `STATES`).
pub type TemporaryKHPRef<'a, const STATES: usize, T> =
    TemporaryKHPMatrixBuffer<STATES, T, MatrixDataMut<'a, STATES, STATES, T>>;

/// Flattens a row-major nested array into the flat slice the buffer types are built from.
///
/// `[[T; COLS]; ROWS]` is contiguous without padding, so this is the same memory as the flat
/// `[T; ROWS * COLS]` buffer of the `impl_buffer_*` macros.
#[inline(always)]
pub(crate) fn flat<T, const ROWS: usize, const COLS: usize>(
    matrix: &mut [[T; COLS]; ROWS],
) -> &mut [T] {
    matrix.as_flattened_mut()
}

/// The six filter buffers of the regular and the extended Kalman filter.
///
/// The information filter reuses them with `Y` in place of `P` and `ŷ` in place of `x`.
#[allow(non_snake_case)]
pub(crate) struct StateBuffers<const STATES: usize, T> {
    /// State transition matrix `A`, or its Jacobian for the EKF.
    A: [[T; STATES]; STATES],
    /// State vector `x`.
    x: [T; STATES],
    /// Estimate covariance matrix `P`.
    P: [[T; STATES]; STATES],
    /// Direct process noise covariance matrix `Q`.
    Q: [[T; STATES]; STATES],
    /// Scratch for the predicted state.
    predicted_x: [T; STATES],
    /// Scratch for the covariance prediction.
    temp_P: [[T; STATES]; STATES],
}

/// The six filter buffers, wired as borrowed buffer types in constructor order.
pub(crate) type StateWiring<'a, const STATES: usize, T> = (
    StateTransitionRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    EstimateCovarianceRef<'a, STATES, T>,
    ProcessNoiseRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
    TemporaryStateRef<'a, STATES, T>,
);

impl<const STATES: usize, T: Copy> StateBuffers<STATES, T> {
    /// Creates the buffers with every entry set to `init`.
    pub(crate) const fn new(init: T) -> Self {
        Self {
            A: [[init; STATES]; STATES],
            x: [init; STATES],
            P: [[init; STATES]; STATES],
            Q: [[init; STATES]; STATES],
            predicted_x: [init; STATES],
            temp_P: [[init; STATES]; STATES],
        }
    }
}

impl<const STATES: usize, T> StateBuffers<STATES, T> {
    /// Borrows every buffer as its buffer type, in the order of `RegularKalmanBuilder::new`.
    pub(crate) fn wire(&mut self) -> StateWiring<'_, STATES, T> {
        (
            flat(&mut self.A).into(),
            self.x.as_mut_slice().into(),
            flat(&mut self.P).into(),
            flat(&mut self.Q).into(),
            self.predicted_x.as_mut_slice().into(),
            flat(&mut self.temp_P).into(),
        )
    }
}

/// The ten buffers of a regular or extended observation.
#[allow(non_snake_case)]
pub(crate) struct ObservationBuffers<const STATES: usize, const OBSERVATIONS: usize, T> {
    /// Observation matrix `H`, or its Jacobian for the EKF.
    H: [[T; STATES]; OBSERVATIONS],
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
    /// Scratch for `H×P`.
    temp_HP: [[T; STATES]; OBSERVATIONS],
    /// Scratch for `P×Hᵀ`.
    temp_PHt: [[T; OBSERVATIONS]; STATES],
    /// Scratch for `K×H×P`.
    temp_KHP: [[T; STATES]; STATES],
}

/// The ten observation buffers, wired as borrowed buffer types in constructor order.
pub(crate) type ObservationWiring<'a, const STATES: usize, const OBSERVATIONS: usize, T> = (
    ObservationMatrixRef<'a, STATES, OBSERVATIONS, T>,
    MeasurementVectorRef<'a, OBSERVATIONS, T>,
    MeasurementNoiseRef<'a, OBSERVATIONS, T>,
    InnovationVectorRef<'a, OBSERVATIONS, T>,
    InnovationCovarianceRef<'a, OBSERVATIONS, T>,
    KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
    TemporarySInvRef<'a, OBSERVATIONS, T>,
    TemporaryHPRef<'a, STATES, OBSERVATIONS, T>,
    TemporaryPHtRef<'a, STATES, OBSERVATIONS, T>,
    TemporaryKHPRef<'a, STATES, T>,
);

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy>
    ObservationBuffers<STATES, OBSERVATIONS, T>
{
    /// Creates the buffers with every entry set to `init`.
    pub(crate) const fn new(init: T) -> Self {
        Self {
            H: [[init; STATES]; OBSERVATIONS],
            z: [init; OBSERVATIONS],
            R: [[init; OBSERVATIONS]; OBSERVATIONS],
            y: [init; OBSERVATIONS],
            S: [[init; OBSERVATIONS]; OBSERVATIONS],
            K: [[init; OBSERVATIONS]; STATES],
            temp_S_inv: [[init; OBSERVATIONS]; OBSERVATIONS],
            temp_HP: [[init; STATES]; OBSERVATIONS],
            temp_PHt: [[init; OBSERVATIONS]; STATES],
            temp_KHP: [[init; STATES]; STATES],
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T>
    ObservationBuffers<STATES, OBSERVATIONS, T>
{
    /// Borrows every buffer as its buffer type, in the order of `RegularObservationBuilder::new`.
    pub(crate) fn wire(&mut self) -> ObservationWiring<'_, STATES, OBSERVATIONS, T> {
        (
            flat(&mut self.H).into(),
            self.z.as_mut_slice().into(),
            flat(&mut self.R).into(),
            self.y.as_mut_slice().into(),
            flat(&mut self.S).into(),
            flat(&mut self.K).into(),
            flat(&mut self.temp_S_inv).into(),
            flat(&mut self.temp_HP).into(),
            flat(&mut self.temp_PHt).into(),
            flat(&mut self.temp_KHP).into(),
        )
    }
}

/// The six persistent buffers of a lean regular or extended observation, which borrows its
/// temporaries per call from a [`CorrectionScratch`](crate::regular::CorrectionScratch).
#[allow(non_snake_case)]
pub(crate) struct LeanObservationBuffers<const STATES: usize, const OBSERVATIONS: usize, T> {
    /// Observation matrix `H`, or its Jacobian for the EKF.
    H: [[T; STATES]; OBSERVATIONS],
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
}

/// The six lean observation buffers, wired as borrowed buffer types in constructor order.
pub(crate) type LeanObservationWiring<'a, const STATES: usize, const OBSERVATIONS: usize, T> = (
    ObservationMatrixRef<'a, STATES, OBSERVATIONS, T>,
    MeasurementVectorRef<'a, OBSERVATIONS, T>,
    MeasurementNoiseRef<'a, OBSERVATIONS, T>,
    InnovationVectorRef<'a, OBSERVATIONS, T>,
    InnovationCovarianceRef<'a, OBSERVATIONS, T>,
    KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
);

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy>
    LeanObservationBuffers<STATES, OBSERVATIONS, T>
{
    /// Creates the buffers with every entry set to `init`.
    pub(crate) const fn new(init: T) -> Self {
        Self {
            H: [[init; STATES]; OBSERVATIONS],
            z: [init; OBSERVATIONS],
            R: [[init; OBSERVATIONS]; OBSERVATIONS],
            y: [init; OBSERVATIONS],
            S: [[init; OBSERVATIONS]; OBSERVATIONS],
            K: [[init; OBSERVATIONS]; STATES],
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T>
    LeanObservationBuffers<STATES, OBSERVATIONS, T>
{
    /// Borrows every buffer as its buffer type, in the order of
    /// `RegularObservationBuilder::new_lean`.
    pub(crate) fn wire(&mut self) -> LeanObservationWiring<'_, STATES, OBSERVATIONS, T> {
        (
            flat(&mut self.H).into(),
            self.z.as_mut_slice().into(),
            flat(&mut self.R).into(),
            self.y.as_mut_slice().into(),
            flat(&mut self.S).into(),
            flat(&mut self.K).into(),
        )
    }
}
