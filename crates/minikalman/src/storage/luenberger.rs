//! Storage for the fixed-gain (Luenberger) observer and its observations.

use super::layout::*;
use crate::kalman::GainForm;
use crate::matrix::MatrixDataType;
use crate::regular::{FixedGainObservation, LuenbergerObserver};

/// A [`LuenbergerObserver`] whose buffers borrow an [`ObserverStorage`].
pub type BorrowedLuenbergerObserver<'a, const STATES: usize, T> = LuenbergerObserver<
    STATES,
    T,
    StateTransitionRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
>;

/// A [`FixedGainObservation`] whose buffers borrow a [`FixedGainObservationStorage`].
pub type BorrowedFixedGainObservation<'a, const STATES: usize, const OBSERVATIONS: usize, T> =
    FixedGainObservation<
        STATES,
        OBSERVATIONS,
        T,
        ObservationMatrixRef<'a, STATES, OBSERVATIONS, T>,
        MeasurementVectorRef<'a, OBSERVATIONS, T>,
        InnovationVectorRef<'a, OBSERVATIONS, T>,
        KalmanGainRef<'a, STATES, OBSERVATIONS, T>,
    >;

/// Storage for the buffers of a [`LuenbergerObserver`] with `STATES` states.
///
/// Holds `A`, `x` and the predicted-state temporary as nested arrays, `STATES² + 2·STATES`
/// scalars and nothing else: no covariance. See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory; the same
/// applies here.
///
/// ## Example
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{FixedGainObservationStorage, ObserverStorage};
///
/// let mut observer_storage = ObserverStorage::<2, f32>::new(0.0);
/// let mut position_storage = FixedGainObservationStorage::<2, 1, f32>::new(0.0);
///
/// let mut observer = observer_storage.build();
/// let mut position = position_storage.build(GainForm::Current);
/// observer.state_transition_mut().make_identity();
/// position.observation_matrix_mut().set_at(0, 0, 1.0);
/// position.set_gain_current(&MatrixData::new_array::<2, 1, 2, f32>([0.5, 0.0]));
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
    /// Wires a [`LuenbergerObserver`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedLuenbergerObserver<'_, STATES, T> {
        LuenbergerObserver::new(
            flat(&mut self.A).into(),
            self.x.as_mut_slice().into(),
            self.predicted_x.as_mut_slice().into(),
        )
    }
}

/// Storage for the buffers of a [`FixedGainObservation`] of `OBSERVATIONS` measurements.
///
/// Holds `H`, `z`, `y` and the gain as nested arrays, `2·STATES·OBSERVATIONS + 2·OBSERVATIONS`
/// scalars. The storage keeps the gain across [`build`](Self::build)s but not its form, so
/// `build` takes the [`GainForm`] of the gain the storage holds; set a new gain with
/// [`set_gain_current`](FixedGainObservation::set_gain_current) or
/// [`set_gain_predictor`](FixedGainObservation::set_gain_predictor), which record the form.
/// See [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
#[allow(non_snake_case)]
pub struct FixedGainObservationStorage<const STATES: usize, const OBSERVATIONS: usize, T> {
    /// Observation matrix `H`.
    H: [[T; STATES]; OBSERVATIONS],
    /// Measurement vector `z`.
    z: [T; OBSERVATIONS],
    /// Innovation vector `y`.
    y: [T; OBSERVATIONS],
    /// The fixed gain `K` or `L`.
    K: [[T; OBSERVATIONS]; STATES],
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: Copy>
    FixedGainObservationStorage<STATES, OBSERVATIONS, T>
{
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self {
            H: [[init; STATES]; OBSERVATIONS],
            z: [init; OBSERVATIONS],
            y: [init; OBSERVATIONS],
            K: [[init; OBSERVATIONS]; STATES],
        }
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T: MatrixDataType>
    FixedGainObservationStorage<STATES, OBSERVATIONS, T>
{
    /// Wires a [`FixedGainObservation`] whose buffers borrow this storage.
    ///
    /// ## Arguments
    /// * `form` - The convention of the gain the storage holds, e.g. the form it was set with
    ///   before a previous observation was dropped. For a zeroed gain that is set afterwards,
    ///   either form works.
    pub fn build(
        &mut self,
        form: GainForm,
    ) -> BorrowedFixedGainObservation<'_, STATES, OBSERVATIONS, T> {
        FixedGainObservation::new(
            flat(&mut self.H).into(),
            self.z.as_mut_slice().into(),
            self.y.as_mut_slice().into(),
            flat(&mut self.K).into(),
            form,
        )
    }
}
