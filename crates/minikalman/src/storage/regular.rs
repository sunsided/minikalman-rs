//! Storage for the regular Kalman filter, its observations and its controls.

use super::layout::*;
use crate::buffers::types::{
    ControlMatrixMutBuffer, ControlProcessNoiseCovarianceMatrixMutBuffer, ControlVectorBuffer,
    TemporaryBQMatrixBuffer,
};
use crate::matrix::{MatrixDataMut, MatrixDataType};
use crate::regular::{
    Control, ControlBuilder, RegularKalman, RegularKalmanBuilder, RegularObservation,
    RegularObservationBuilder,
};

/// A [`RegularKalman`] filter whose buffers borrow a [`FilterStorage`].
pub type BorrowedKalman<'a, const STATES: usize, T> = RegularKalman<
    STATES,
    T,
    StateTransitionRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    EstimateCovarianceRef<'a, STATES, T>,
    ProcessNoiseRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
    TemporaryStateRef<'a, STATES, T>,
>;

/// A [`RegularObservation`] whose buffers borrow an [`ObservationStorage`].
pub type BorrowedObservation<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    RegularObservation<
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

/// A [`Control`] whose buffers borrow a [`ControlStorage`].
pub type BorrowedControl<'a, const STATES: usize, const CONTROLS: usize, T> = Control<
    STATES,
    CONTROLS,
    T,
    ControlMatrixMutBuffer<STATES, CONTROLS, T, MatrixDataMut<'a, STATES, CONTROLS, T>>,
    ControlVectorBuffer<CONTROLS, T, MatrixDataMut<'a, CONTROLS, 1, T>>,
    ControlProcessNoiseCovarianceMatrixMutBuffer<
        CONTROLS,
        T,
        MatrixDataMut<'a, CONTROLS, CONTROLS, T>,
    >,
    TemporaryBQMatrixBuffer<STATES, CONTROLS, T, MatrixDataMut<'a, STATES, CONTROLS, T>>,
>;

/// Storage for the buffers of a [`RegularKalman`] filter with `STATES` states.
///
/// Holds `A`, `x`, `P`, `Q` and the two filter temporaries as nested arrays. Its size is exactly
/// that of the buffers the `impl_buffer_*` macros would declare for the same filter, and it
/// allocates nothing. [`build`](Self::build) wires a filter whose buffers borrow this storage.
///
/// ## Memory and placement
///
/// * [`new`](Self::new) is a `const fn`, so the storage can live in a `static`. A `static`
///   with a zero `init` lands in `.bss` and costs no flash; a non-zero `init` puts the whole
///   storage in `.data`.
/// * On small targets, place large storage in a `static` rather than on the stack, so it is
///   never built and moved by value. To get a `&'static mut` to a `static` soundly, use a
///   take-once cell such as `static_cell::ConstStaticCell` (const-initialized, so nothing is
///   moved at runtime), not a `static mut` reference.
/// * [`build`](Self::build) borrows the storage and moves no arrays: the filter holds only
///   slices into it. It does not reset the contents, so a second `build` sees the values the
///   previous filter left behind.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{FilterStorage, ObservationStorage};
///
/// let mut filter_storage = FilterStorage::<3, f32>::new(0.0);
/// let mut measurement_storage = ObservationStorage::<3, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut measurement = measurement_storage.build();
///
/// filter.state_transition_mut().make_identity();
/// filter.estimate_covariance_mut().make_identity();
/// measurement.observation_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// measurement.measurement_noise_covariance_mut().apply(|r| r.set_at(0, 0, 0.5));
///
/// filter.predict();
/// measurement.measurement_vector_mut().apply(|z| z[0] = 1.0);
/// filter.correct(&mut measurement);
/// assert!(filter.state_vector().get_row(0) > 0.0);
/// ```
pub struct FilterStorage<const STATES: usize, T>(StateBuffers<STATES, T>);

impl<const STATES: usize, T: Copy> FilterStorage<STATES, T> {
    /// Creates the storage with every entry set to `init`, e.g. `0.0` or `I16F16::ZERO`.
    pub const fn new(init: T) -> Self {
        Self(StateBuffers::new(init))
    }
}

impl<const STATES: usize, T: MatrixDataType> FilterStorage<STATES, T> {
    /// Wires a [`RegularKalman`] filter whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedKalman<'_, STATES, T> {
        let (a, x, p, q, predicted_x, temp_p) = self.0.wire();
        RegularKalmanBuilder::new::<STATES, T>(a, x, p, q, predicted_x, temp_p)
    }
}

/// Storage for the buffers of a [`RegularObservation`] of `OBSERVATIONS` measurements.
///
/// Holds `H`, `z`, `R`, `y`, `S`, `K` and the four observation temporaries as nested arrays.
/// See [`FilterStorage`] for placement and memory; the same applies here. An
/// [`InformationFilter`](crate::information::InformationFilter) fuses the same observation type.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::ObservationStorage;
///
/// let mut storage = ObservationStorage::<3, 1, f32>::new(0.0);
/// let mut measurement = storage.build();
/// measurement.measurement_vector_mut().apply(|z| z[0] = 2.0);
/// assert_eq!(measurement.measurement_vector().get_row(0), 2.0);
/// ```
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
    /// Wires a [`RegularObservation`] whose buffers borrow this storage.
    #[allow(non_snake_case)]
    pub fn build(&mut self) -> BorrowedObservation<'_, STATES, OBSERVATIONS, T> {
        let (h, z, r, y, s, k, temp_S_inv, temp_HP, temp_PHt, temp_KHP) = self.0.wire();
        RegularObservationBuilder::new::<STATES, OBSERVATIONS, T>(
            h, z, r, y, s, k, temp_S_inv, temp_HP, temp_PHt, temp_KHP,
        )
    }
}

/// A lean [`RegularObservation`] whose buffers borrow a [`LeanObservationStorage`] and whose
/// temporaries come from a [`CorrectionScratch`](crate::regular::CorrectionScratch) per call.
pub type BorrowedLeanObservation<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    RegularObservation<
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

/// Storage for the persistent buffers of a lean [`RegularObservation`] of `OBSERVATIONS`
/// measurements, without the four temporaries.
///
/// Holds `H`, `z`, `R`, `y`, `S` and `K` as nested arrays. The observation it builds corrects
/// with [`RegularKalman::correct_in`] and borrows its temporaries from a
/// [`CorrectionScratch`](crate::regular::CorrectionScratch), which all lean observations of a
/// filter can share. See [`FilterStorage`] for placement and memory.
///
/// For 4 states and 3 measurements in `f32`, this holds 48 values (192 bytes) where
/// [`ObservationStorage`] holds 97 (388 bytes); the 49 values of the temporaries are paid once,
/// in the shared scratch.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{CorrectionScratch, FilterStorage, LeanObservationStorage};
///
/// let mut filter_storage = FilterStorage::<3, f32>::new(0.0);
/// let mut measurement_storage = LeanObservationStorage::<3, 1, f32>::new(0.0);
/// let mut scratch = CorrectionScratch::<3, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut measurement = measurement_storage.build();
///
/// filter.state_transition_mut().make_identity();
/// filter.estimate_covariance_mut().make_identity();
/// measurement.observation_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// measurement.measurement_noise_covariance_mut().apply(|r| r.set_at(0, 0, 0.5));
///
/// filter.predict();
/// measurement.measurement_vector_mut().apply(|z| z[0] = 1.0);
/// filter.correct_in(&mut measurement, &mut scratch);
/// assert!(filter.state_vector().get_row(0) > 0.0);
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
    /// Wires a lean [`RegularObservation`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedLeanObservation<'_, STATES, OBSERVATIONS, T> {
        let (h, z, r, y, s, k) = self.0.wire();
        RegularObservationBuilder::new_lean::<STATES, OBSERVATIONS, T>(h, z, r, y, s, k)
    }
}

/// Storage for the buffers of a [`Control`] with `CONTROLS` inputs.
///
/// Holds `B`, `u`, the control process noise `Q` and the `B×Q` temporary as nested arrays.
/// See [`FilterStorage`] for placement and memory; the same applies here.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::ControlStorage;
///
/// let mut storage = ControlStorage::<3, 1, f32>::new(0.0);
/// let mut control = storage.build();
/// control.control_vector_mut().apply(|u| u[0] = 1.0);
/// assert_eq!(control.control_vector().get_row(0), 1.0);
/// ```
#[allow(non_snake_case)]
pub struct ControlStorage<const STATES: usize, const CONTROLS: usize, T> {
    /// Control matrix `B`.
    B: [[T; CONTROLS]; STATES],
    /// Control vector `u`.
    u: [T; CONTROLS],
    /// Control process noise covariance matrix `Q`.
    Q: [[T; CONTROLS]; CONTROLS],
    /// Scratch for `B×Q`.
    temp_BQ: [[T; CONTROLS]; STATES],
}

impl<const STATES: usize, const CONTROLS: usize, T: Copy> ControlStorage<STATES, CONTROLS, T> {
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self {
            B: [[init; CONTROLS]; STATES],
            u: [init; CONTROLS],
            Q: [[init; CONTROLS]; CONTROLS],
            temp_BQ: [[init; CONTROLS]; STATES],
        }
    }
}

impl<const STATES: usize, const CONTROLS: usize, T: MatrixDataType>
    ControlStorage<STATES, CONTROLS, T>
{
    /// Wires a [`Control`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedControl<'_, STATES, CONTROLS, T> {
        ControlBuilder::new::<STATES, CONTROLS, T>(
            flat(&mut self.B).into(),
            self.u.as_mut_slice().into(),
            flat(&mut self.Q).into(),
            flat(&mut self.temp_BQ).into(),
        )
    }
}
