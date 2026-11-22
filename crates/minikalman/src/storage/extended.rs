//! Storage for the extended Kalman filter and its observations.

use super::layout::*;
use crate::extended::{
    ExtendedKalman, ExtendedKalmanBuilder, ExtendedObservation, ExtendedObservationBuilder,
};
use crate::matrix::MatrixDataType;

/// An [`ExtendedKalman`] filter whose buffers borrow a [`FilterStorage`].
pub type BorrowedKalman<'a, const STATES: usize, T> = ExtendedKalman<
    STATES,
    T,
    StateTransitionRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    EstimateCovarianceRef<'a, STATES, T>,
    ProcessNoiseRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
    TemporaryStateRef<'a, STATES, T>,
>;

/// An [`ExtendedObservation`] whose buffers borrow an [`ObservationStorage`].
pub type BorrowedObservation<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    ExtendedObservation<
        STATES,
        OBSERVATIONS,
        T,
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
    >;

/// Storage for the buffers of an [`ExtendedKalman`] filter with `STATES` states.
///
/// Holds the Jacobian `A`, `x`, `P`, `Q` and the two filter temporaries as nested arrays, the
/// same layout as [`regular::FilterStorage`](crate::regular::FilterStorage). See there for
/// placement and memory.
///
/// ## Example
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
/// use minikalman::extended::{FilterStorage, ObservationStorage};
/// use minikalman::prelude::*;
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut measurement_storage = ObservationStorage::<2, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut measurement = measurement_storage.build();
///
/// filter.state_transition_jacobian_mut().make_identity();
/// filter.estimate_covariance_mut().make_identity();
/// measurement.observation_jacobian_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// measurement.measurement_noise_covariance_mut().apply(|r| r.set_at(0, 0, 0.5));
///
/// filter.predict_nonlinear(|x, next| {
///     next[0] = x[0] + x[1];
///     next[1] = x[1];
/// });
/// measurement.measurement_vector_mut().apply(|z| z[0] = 1.0);
/// filter.correct_nonlinear(&mut measurement, |x, z| z[0] = x[0]);
/// assert!(filter.state_vector().get_row(0) > 0.0);
/// # }
/// ```
pub struct FilterStorage<const STATES: usize, T>(StateBuffers<STATES, T>);

impl<const STATES: usize, T: Copy> FilterStorage<STATES, T> {
    /// Creates the storage with every entry set to `init`, e.g. `0.0` or `I16F16::ZERO`.
    pub const fn new(init: T) -> Self {
        Self(StateBuffers::new(init))
    }
}

impl<const STATES: usize, T: MatrixDataType> FilterStorage<STATES, T> {
    /// Wires an [`ExtendedKalman`] filter whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedKalman<'_, STATES, T> {
        let (a, x, p, q, predicted_x, temp_p) = self.0.wire();
        ExtendedKalmanBuilder::new::<STATES, T>(a, x, p, q, predicted_x, temp_p)
    }
}

/// Storage for the buffers of an [`ExtendedObservation`] of `OBSERVATIONS` measurements.
///
/// Holds the Jacobian `H`, `z`, `R`, `y`, `S`, `K` and the four observation temporaries as
/// nested arrays, the same layout as
/// [`regular::ObservationStorage`](crate::regular::ObservationStorage). See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
pub struct ObservationStorage<const STATES: usize, const OBSERVATIONS: usize, T>(
    ObservationBuffers<STATES, OBSERVATIONS, T>,
);

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy>
    ObservationStorage<STATES, OBSERVATIONS, T>
{
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self(ObservationBuffers::new(init))
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: MatrixDataType>
    ObservationStorage<STATES, OBSERVATIONS, T>
{
    /// Wires an [`ExtendedObservation`] whose buffers borrow this storage.
    #[allow(non_snake_case)]
    pub fn build(&mut self) -> BorrowedObservation<'_, STATES, OBSERVATIONS, T> {
        let (h, z, r, y, s, k, temp_S_inv, temp_HP, temp_PHt, temp_KHP) = self.0.wire();
        ExtendedObservationBuilder::new::<STATES, OBSERVATIONS, T>(
            h, z, r, y, s, k, temp_S_inv, temp_HP, temp_PHt, temp_KHP,
        )
    }
}

/// A lean [`ExtendedObservation`] whose buffers borrow a [`LeanObservationStorage`] and whose
/// temporaries come from a [`CorrectionScratch`](crate::extended::CorrectionScratch) per call.
pub type BorrowedLeanObservation<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    ExtendedObservation<
        STATES,
        OBSERVATIONS,
        T,
        ObservationMatrixRef<'a, STATES, OBSERVATIONS, T>,
        MeasurementVectorRef<'a, OBSERVATIONS, T>,
        MeasurementNoiseRef<'a, OBSERVATIONS, T>,
        InnovationVectorRef<'a, OBSERVATIONS, T>,
        InnovationCovarianceRef<'a, OBSERVATIONS, T>,
        KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
        (),
        (),
        (),
        (),
    >;

/// Storage for the persistent buffers of a lean [`ExtendedObservation`] of `OBSERVATIONS`
/// measurements, without the four temporaries.
///
/// Holds the Jacobian `H`, `z`, `R`, `y`, `S` and `K` as nested arrays. The observation it builds corrects
/// with [`ExtendedKalman::correct_nonlinear_in`] and borrows its temporaries from a
/// [`CorrectionScratch`](crate::extended::CorrectionScratch), which all lean observations of a
/// filter can share. See [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
///
/// For 4 states and 3 measurements in `f32`, this holds 48 values (192 bytes) where
/// [`ObservationStorage`] holds 97 (388 bytes); the 49 values of the temporaries are paid once,
/// in the shared scratch.
///
/// ## Example
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
/// use minikalman::prelude::*;
/// use minikalman::extended::{CorrectionScratch, FilterStorage, LeanObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut measurement_storage = LeanObservationStorage::<2, 1, f32>::new(0.0);
/// let mut scratch = CorrectionScratch::<2, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut measurement = measurement_storage.build();
///
/// filter.state_transition_jacobian_mut().make_identity();
/// filter.estimate_covariance_mut().make_identity();
/// measurement.observation_jacobian_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// measurement.measurement_noise_covariance_mut().apply(|r| r.set_at(0, 0, 0.5));
///
/// filter.predict_nonlinear(|x, next| {
///     next[0] = x[0] + x[1];
///     next[1] = x[1];
/// });
/// measurement.measurement_vector_mut().apply(|z| z[0] = 1.0);
/// filter.correct_nonlinear_in(&mut measurement, &mut scratch, |x, z| z[0] = x[0]);
/// assert!(filter.state_vector().get_row(0) > 0.0);
/// # }
/// ```
pub struct LeanObservationStorage<const STATES: usize, const OBSERVATIONS: usize, T>(
    LeanObservationBuffers<STATES, OBSERVATIONS, T>,
);

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy>
    LeanObservationStorage<STATES, OBSERVATIONS, T>
{
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self(LeanObservationBuffers::new(init))
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: MatrixDataType>
    LeanObservationStorage<STATES, OBSERVATIONS, T>
{
    /// Wires a lean [`ExtendedObservation`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedLeanObservation<'_, STATES, OBSERVATIONS, T> {
        let (h, z, r, y, s, k) = self.0.wire();
        ExtendedObservationBuilder::new_lean::<STATES, OBSERVATIONS, T>(h, z, r, y, s, k)
    }
}
