//! Storage for the fixed-gain (Luenberger) observer and its observations.

use core::marker::PhantomData;

use super::layout::*;
use crate::kalman::{Current, GainForm};
use crate::matrix::MatrixDataType;
use crate::regular::{FixedGainGate, FixedGainObservation, LuenbergerObserver, PendingCorrection};

/// A [`Current`]-form [`LuenbergerObserver`] whose buffers borrow an [`ObserverStorage`].
pub type BorrowedLuenbergerObserver<'a, const STATES: usize, T> = LuenbergerObserver<
    STATES,
    T,
    StateTransitionRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
>;

/// A [`Predictor`](crate::kalman::Predictor)-form [`LuenbergerObserver`] whose buffers borrow
/// a [`PredictorObserverStorage`].
pub type BorrowedPredictorObserver<'a, const STATES: usize, T> = LuenbergerObserver<
    STATES,
    T,
    StateTransitionRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
    PendingCorrection<StateVectorRef<'a, STATES, T>>,
>;

/// A [`FixedGainObservation`] of the [`GainForm`] `F` whose buffers borrow a
/// [`FixedGainObservationStorage`].
pub type BorrowedFixedGainObservation<
    'a,
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
    F = Current,
> = FixedGainObservation<
    STATES,
    OBSERVATIONS,
    T,
    ObservationMatrixRef<'a, STATES, OBSERVATIONS, T>,
    MeasurementVectorRef<'a, OBSERVATIONS, T>,
    InnovationVectorRef<'a, OBSERVATIONS, T>,
    KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
    F,
>;

/// A gated [`FixedGainObservation`] of the [`GainForm`] `F` whose buffers borrow a
/// [`GatedFixedGainObservationStorage`].
pub type BorrowedGatedFixedGainObservation<
    'a,
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
    F = Current,
> = FixedGainObservation<
    STATES,
    OBSERVATIONS,
    T,
    ObservationMatrixRef<'a, STATES, OBSERVATIONS, T>,
    MeasurementVectorRef<'a, OBSERVATIONS, T>,
    InnovationVectorRef<'a, OBSERVATIONS, T>,
    KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
    F,
    FixedGainGate<OBSERVATIONS, T, InnovationCovarianceRef<'a, OBSERVATIONS, T>>,
>;

/// Storage for the buffers of a [`Current`]-form [`LuenbergerObserver`] with `STATES` states.
///
/// Holds `A`, `x` and the predicted-state temporary as nested arrays, `STATES² + 2·STATES`
/// scalars and nothing else: no covariance. See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory; the same
/// applies here. [`PredictorObserverStorage`] is the [`Predictor`](crate::kalman::Predictor)
/// form.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{FixedGainObservationStorage, ObserverStorage};
///
/// let mut observer_storage = ObserverStorage::<2, f32>::new(0.0);
/// let mut position_storage = FixedGainObservationStorage::<2, 1, f32, Current>::new(0.0);
///
/// let mut observer = observer_storage.build();
/// let mut position = position_storage.build();
/// observer.state_transition_mut().make_identity();
/// position.observation_matrix_mut().set_at(0, 0, 1.0);
/// position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.5, 0.0]));
///
/// observer.predict();
/// position.measurement_vector_mut().set_at(0, 0, 2.0);
/// assert!(observer.correct(&mut position).is_applied());
/// assert_eq!(observer.state_vector().get_at(0, 0), 1.0);
/// ```
#[allow(non_snake_case)]
pub struct ObserverStorage<const STATES: usize, T> {
    /// State transition matrix `A`.
    A: [[T; STATES]; STATES],
    /// State vector `x`.
    x: [T; STATES],
    /// Scratch for the predicted state.
    predicted_x: [T; STATES],
}

impl<const STATES: usize, T: Copy> ObserverStorage<STATES, T> {
    /// Creates the storage with every entry set to `init`, e.g. `0.0` or `I16F16::ZERO`.
    pub const fn new(init: T) -> Self {
        Self {
            A: [[init; STATES]; STATES],
            x: [init; STATES],
            predicted_x: [init; STATES],
        }
    }
}

impl<const STATES: usize, T: MatrixDataType> ObserverStorage<STATES, T> {
    /// Wires a [`Current`]-form [`LuenbergerObserver`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedLuenbergerObserver<'_, STATES, T> {
        LuenbergerObserver::new(
            flat(&mut self.A).into(),
            self.x.as_mut_slice().into(),
            self.predicted_x.as_mut_slice().into(),
        )
    }
}

/// Storage for the buffers of a [`Predictor`](crate::kalman::Predictor)-form
/// [`LuenbergerObserver`] with `STATES` states.
///
/// Holds the buffers of an [`ObserverStorage`] plus the pending correction,
/// `STATES² + 3·STATES` scalars.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{FixedGainObservationStorage, PredictorObserverStorage};
///
/// let mut observer_storage = PredictorObserverStorage::<2, f32>::new(0.0);
/// let mut position_storage = FixedGainObservationStorage::<2, 1, f32, Predictor>::new(0.0);
///
/// let mut observer = observer_storage.build();
/// let mut position = position_storage.build();
/// observer.state_transition_mut().make_identity();
/// position.observation_matrix_mut().set_at(0, 0, 1.0);
/// position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.5, 0.0]));
///
/// position.measurement_vector_mut().set_at(0, 0, 2.0);
/// assert!(observer.correct(&mut position).is_applied());
/// assert_eq!(observer.state_vector().get_at(0, 0), 0.0); // still the prior
/// observer.predict();
/// assert_eq!(observer.state_vector().get_at(0, 0), 1.0);
/// ```
#[allow(non_snake_case)]
pub struct PredictorObserverStorage<const STATES: usize, T> {
    /// The buffers shared with the current form.
    observer: ObserverStorage<STATES, T>,
    /// The pending correction `Σ L·y`.
    pending: [T; STATES],
}

impl<const STATES: usize, T: Copy> PredictorObserverStorage<STATES, T> {
    /// Creates the storage with every entry set to `init`; [`build`](Self::build) clears the
    /// pending correction.
    pub const fn new(init: T) -> Self {
        Self {
            observer: ObserverStorage::new(init),
            pending: [init; STATES],
        }
    }
}

impl<const STATES: usize, T: MatrixDataType> PredictorObserverStorage<STATES, T> {
    /// Wires a [`Predictor`](crate::kalman::Predictor)-form [`LuenbergerObserver`] whose
    /// buffers borrow this storage.
    ///
    /// The pending correction is cleared, so a correction left by an earlier observer on this
    /// storage is discarded instead of reaching the new observer's first prediction. `A` and
    /// `x̂` are kept.
    pub fn build(&mut self) -> BorrowedPredictorObserver<'_, STATES, T> {
        self.pending.fill(T::zero());
        let observer = &mut self.observer;
        LuenbergerObserver::new_predictor(
            flat(&mut observer.A).into(),
            observer.x.as_mut_slice().into(),
            observer.predicted_x.as_mut_slice().into(),
            self.pending.as_mut_slice().into(),
        )
    }
}

/// Storage for the buffers of a [`FixedGainObservation`] of `OBSERVATIONS` measurements with a
/// gain in the [`GainForm`] `F`.
///
/// Holds `H`, `z`, `y` and the gain as nested arrays, `2·STATES·OBSERVATIONS + 2·OBSERVATIONS`
/// scalars. The form is part of the type, so a gain kept across [`build`](Self::build)s is
/// always read in the form it was written in. See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
#[allow(non_snake_case)]
pub struct FixedGainObservationStorage<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
    F = Current,
> {
    /// Observation matrix `H`.
    H: [[T; STATES]; OBSERVATIONS],
    /// Measurement vector `z`.
    z: [T; OBSERVATIONS],
    /// Innovation vector `y`.
    y: [T; OBSERVATIONS],
    /// The fixed gain `K` or `L`.
    K: [[T; OBSERVATIONS]; STATES],
    /// Marks the gain form.
    _form: PhantomData<F>,
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy, F>
    FixedGainObservationStorage<STATES, OBSERVATIONS, T, F>
{
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self {
            H: [[init; STATES]; OBSERVATIONS],
            z: [init; OBSERVATIONS],
            y: [init; OBSERVATIONS],
            K: [[init; OBSERVATIONS]; STATES],
            _form: PhantomData,
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: MatrixDataType, F: GainForm>
    FixedGainObservationStorage<STATES, OBSERVATIONS, T, F>
{
    /// Wires a [`FixedGainObservation`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedFixedGainObservation<'_, STATES, OBSERVATIONS, T, F> {
        FixedGainObservation::new(
            flat(&mut self.H).into(),
            self.z.as_mut_slice().into(),
            self.y.as_mut_slice().into(),
            flat(&mut self.K).into(),
        )
    }
}

/// Storage for the buffers of a [`FixedGainObservation`] with an innovation gate.
///
/// Holds the buffers of a [`FixedGainObservationStorage`] plus `S∞⁻¹`,
/// `2·STATES·OBSERVATIONS + 2·OBSERVATIONS + OBSERVATIONS²` scalars. The storage keeps the
/// gain and `S∞⁻¹` across [`build`](Self::build)s, but neither the gate threshold nor the NIS:
/// set the threshold again with [`set_gate`](FixedGainObservation::set_gate) after each
/// `build`.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::GatedFixedGainObservationStorage;
///
/// let mut storage = GatedFixedGainObservationStorage::<2, 1, f32, Current>::new(0.0);
/// let mut position = storage.build();
/// position.observation_matrix_mut().set_at(0, 0, 1.0);
/// position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.5, 0.0]));
/// position.set_innovation_covariance_inverse(&MatrixData::new_array::<1, 1, 1, f32>([1.0]));
/// position.set_gate(Some(6.63));
/// ```
#[allow(non_snake_case)]
pub struct GatedFixedGainObservationStorage<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
    F = Current,
> {
    /// The buffers of the observation without the gate.
    observation: FixedGainObservationStorage<STATES, OBSERVATIONS, T, F>,
    /// The inverse steady-state innovation covariance `S∞⁻¹`.
    S_inv: [[T; OBSERVATIONS]; OBSERVATIONS],
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy, F>
    GatedFixedGainObservationStorage<STATES, OBSERVATIONS, T, F>
{
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self {
            observation: FixedGainObservationStorage::new(init),
            S_inv: [[init; OBSERVATIONS]; OBSERVATIONS],
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: MatrixDataType, F: GainForm>
    GatedFixedGainObservationStorage<STATES, OBSERVATIONS, T, F>
{
    /// Wires a gated [`FixedGainObservation`] whose buffers borrow this storage.
    ///
    /// The gate starts disabled; `S∞⁻¹` is whatever the storage holds.
    pub fn build(&mut self) -> BorrowedGatedFixedGainObservation<'_, STATES, OBSERVATIONS, T, F> {
        self.observation
            .build()
            .with_gate(flat(&mut self.S_inv).into())
    }
}
