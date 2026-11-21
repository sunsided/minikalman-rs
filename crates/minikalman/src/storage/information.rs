//! Storage for the information filter.

use super::layout::*;
use crate::information::InformationFilter;
use crate::matrix::MatrixDataType;

/// An [`InformationFilter`] whose buffers borrow a [`FilterStorage`].
pub type BorrowedInformationFilter<'a, const STATES: usize, T> = InformationFilter<
    STATES,
    T,
    StateTransitionRef<'a, STATES, T>,
    EstimateCovarianceRef<'a, STATES, T>,
    StateVectorRef<'a, STATES, T>,
    ProcessNoiseRef<'a, STATES, T>,
    PredictedStateRef<'a, STATES, T>,
    TemporaryStateRef<'a, STATES, T>,
    TemporaryStateRef<'a, STATES, T>,
>;

/// Storage for the buffers of an [`InformationFilter`] with `STATES` states.
///
/// Holds `A`, the information matrix `Y`, the information vector `ŷ`, `Q` and the three
/// temporaries as nested arrays. With a zero `init`, the filter starts with zero information.
/// Observations are [`regular::ObservationStorage`](crate::regular::ObservationStorage)s. See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
///
/// ## Example
///
/// ```
/// use minikalman::information::FilterStorage;
/// use minikalman::prelude::*;
/// use minikalman::regular::ObservationStorage;
///
/// let mut filter_storage = FilterStorage::<1, f32>::new(0.0);
/// let mut measurement_storage = ObservationStorage::<1, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut measurement = measurement_storage.build();
/// measurement.observation_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// measurement.measurement_noise_covariance_mut().apply(|r| r.set_at(0, 0, 0.5));
/// measurement.measurement_vector_mut().apply(|z| z[0] = 2.0);
///
/// assert!(filter.fuse(&mut measurement));
/// ```
pub struct FilterStorage<const STATES: usize, T> {
    /// `A`, `Y` (in the `P` slot), `ŷ` (in the `x` slot), `Q`, `temp_x` and `temp_P`.
    state: StateBuffers<STATES, T>,
    /// The second temporary matrix, holding Cholesky factors; never aliases `temp_P`.
    temp_l: [[T; STATES]; STATES],
}

impl<const STATES: usize, T: Copy> FilterStorage<STATES, T> {
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self {
            state: StateBuffers::new(init),
            temp_l: [[init; STATES]; STATES],
        }
    }
}

impl<const STATES: usize, T: MatrixDataType> FilterStorage<STATES, T> {
    /// Wires an [`InformationFilter`] whose buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedInformationFilter<'_, STATES, T> {
        let (a, y_hat, y, q, temp_x, temp_p) = self.state.wire();
        InformationFilter::new(
            a,
            y,
            y_hat,
            q,
            temp_x,
            temp_p,
            flat(&mut self.temp_l).into(),
        )
    }
}
