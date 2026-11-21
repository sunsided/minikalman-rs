//! Storage for the Rauch-Tung-Striebel smoother.

use super::layout::flat;
use crate::matrix::{MatrixDataMut, MatrixDataType};
use crate::smoother::RtsSmoother;

/// An [`RtsSmoother`] whose scratch buffers borrow an [`RtsSmootherStorage`].
pub type BorrowedRtsSmoother<'a, const STATES: usize, T> = RtsSmoother<
    STATES,
    T,
    MatrixDataMut<'a, STATES, STATES, T>,
    MatrixDataMut<'a, STATES, STATES, T>,
    MatrixDataMut<'a, STATES, STATES, T>,
    MatrixDataMut<'a, STATES, 1, T>,
>;

/// Storage for the scratch buffers of an [`RtsSmoother`] with `STATES` states.
///
/// Holds the gain, the inverse and a temporary (`STATES` × `STATES` each) and a `STATES` × `1`
/// temporary as nested arrays. The recorded history stays with the caller. See
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement and memory.
///
/// ## Example
///
/// ```
/// use minikalman::smoother::RtsSmootherStorage;
///
/// let mut storage = RtsSmootherStorage::<2, f32>::new(0.0);
/// let smoother = storage.build();
/// # let _ = smoother;
/// ```
pub struct RtsSmootherStorage<const STATES: usize, T> {
    /// The smoother gain.
    gain: [[T; STATES]; STATES],
    /// The inverse of the predicted covariance.
    inverse: [[T; STATES]; STATES],
    /// Scratch matrix.
    temp: [[T; STATES]; STATES],
    /// Scratch vector.
    temp_x: [T; STATES],
}

impl<const STATES: usize, T: Copy> RtsSmootherStorage<STATES, T> {
    /// Creates the storage with every entry set to `init`.
    pub const fn new(init: T) -> Self {
        Self {
            gain: [[init; STATES]; STATES],
            inverse: [[init; STATES]; STATES],
            temp: [[init; STATES]; STATES],
            temp_x: [init; STATES],
        }
    }
}

impl<const STATES: usize, T: MatrixDataType> RtsSmootherStorage<STATES, T> {
    /// Wires an [`RtsSmoother`] whose scratch buffers borrow this storage.
    pub fn build(&mut self) -> BorrowedRtsSmoother<'_, STATES, T> {
        RtsSmoother::new(
            MatrixDataMut::new(flat(&mut self.gain)),
            MatrixDataMut::new(flat(&mut self.inverse)),
            MatrixDataMut::new(flat(&mut self.temp)),
            MatrixDataMut::new(self.temp_x.as_mut_slice()),
        )
    }
}
