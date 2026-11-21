//! Per-call scratch for the corrections of regular and extended observations.

use crate::matrix::MatrixDataMut;

/// The temporaries of one regular or extended correction, borrowed per call so that several
/// observations share one set.
///
/// A correction needs four temporaries: an `OBSERVATIONS × OBSERVATIONS` one (the Cholesky
/// factor of `S` for the gain, `S⁻¹` for the robust and predictively oriented corrections and
/// `settle_gain`, scratch for noise adaptation), `H×P`
/// (`OBSERVATIONS × STATES`), `P×Hᵀ` (`STATES × OBSERVATIONS`) and `K×H×P`
/// (`STATES × STATES`). They hold nothing between two corrections, yet the observations built
/// with [`RegularObservationBuilder::new`](crate::regular::RegularObservationBuilder::new) own
/// them, so a filter with several observations stores them once per observation. A *lean*
/// observation (built with
/// [`RegularObservationBuilder::new_lean`](crate::regular::RegularObservationBuilder::new_lean),
/// [`ExtendedObservationBuilder::new_lean`](crate::extended::ExtendedObservationBuilder::new_lean)
/// or a `LeanObservationStorage`) holds only `H`, `z`, `R`, `y`, `S` and `K`, and borrows the
/// temporaries from this scratch for the duration of one call, e.g.
/// [`RegularKalman::correct_in`](crate::regular::RegularKalman::correct_in).
///
/// One scratch sized for the largest observation, `MAX_OBSERVATIONS`, serves every observation
/// of the filter with `OBSERVATIONS ≤ MAX_OBSERVATIONS`; a larger observation fails to compile
/// (when the call is instantiated, so `cargo check` does not report it, `cargo build` does).
/// A smaller observation uses the leading part of each buffer.
///
/// ## Soundness
///
/// Each correction takes the scratch as `&mut`, so the borrow checker allows one correction to
/// use it at a time: the sharing is disjoint in time by construction, and the four temporaries
/// are four distinct buffers. No correction reads a value a previous correction left in the
/// scratch, so the results are bit-identical to those of an observation that owns its
/// temporaries.
///
/// ## Memory and placement
///
/// The scratch holds `MAX² + 2·MAX·STATES + STATES²` values as nested arrays and allocates
/// nothing. For the radar example (4 states, 3 measurements, `f32`) that is 49 values, which
/// each further lean observation saves. [`new`](Self::new) is a `const fn`, so the scratch can
/// live in a `static` (in `.bss` for a zero `init`); see
/// [`regular::FilterStorage`](crate::regular::FilterStorage) for placement.
///
/// ## Example
///
/// Two observations of different sizes share one scratch, without `std` or `alloc`:
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{CorrectionScratch, FilterStorage, LeanObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut position_storage = LeanObservationStorage::<2, 1, f32>::new(0.0);
/// let mut full_storage = LeanObservationStorage::<2, 2, f32>::new(0.0);
/// let mut scratch = CorrectionScratch::<2, 2, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut position = position_storage.build();
/// let mut full = full_storage.build();
///
/// filter.state_transition_mut().make_identity();
/// filter.estimate_covariance_mut().make_identity();
/// position.observation_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// position.measurement_noise_covariance_mut().make_identity();
/// full.observation_matrix_mut().make_identity();
/// full.measurement_noise_covariance_mut().make_identity();
///
/// filter.predict();
/// position.measurement_vector_mut().apply(|z| z[0] = 1.0);
/// filter.correct_in(&mut position, &mut scratch);
/// full.measurement_vector_mut().apply(|z| z.as_mut_slice().copy_from_slice(&[1.0, 0.5]));
/// filter.correct_in(&mut full, &mut scratch);
///
/// assert!(filter.state_vector().get_row(0) > 0.5);
/// ```
///
/// The scratch cannot be lent to two corrections at once:
///
/// ```compile_fail
/// use minikalman::prelude::*;
/// use minikalman::regular::{CorrectionScratch, FilterStorage, LeanObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut observation_storage = LeanObservationStorage::<2, 1, f32>::new(0.0);
/// let mut scratch = CorrectionScratch::<2, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut observation = observation_storage.build();
///
/// let first = &mut scratch;
/// let second = &mut scratch; // error[E0499]: cannot borrow `scratch` as mutable more than once
/// filter.correct_in(&mut observation, first);
/// filter.correct_in(&mut observation, second);
/// ```
///
/// One borrow after the other compiles:
///
/// ```
/// use minikalman::prelude::*;
/// use minikalman::regular::{CorrectionScratch, FilterStorage, LeanObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut observation_storage = LeanObservationStorage::<2, 1, f32>::new(0.0);
/// let mut scratch = CorrectionScratch::<2, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut observation = observation_storage.build();
///
/// let first = &mut scratch;
/// filter.correct_in(&mut observation, first);
/// let second = &mut scratch;
/// filter.correct_in(&mut observation, second);
/// ```
///
/// A scratch smaller than the observation does not compile either:
///
/// ```compile_fail
/// use minikalman::prelude::*;
/// use minikalman::regular::{CorrectionScratch, FilterStorage, LeanObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut observation_storage = LeanObservationStorage::<2, 2, f32>::new(0.0);
/// let mut scratch = CorrectionScratch::<2, 1, f32>::new(0.0); // serves 1, not 2
///
/// let mut filter = filter_storage.build();
/// let mut observation = observation_storage.build();
/// filter.correct_in(&mut observation, &mut scratch);
/// ```
#[allow(non_snake_case)]
pub struct CorrectionScratch<const STATES: usize, const MAX_OBSERVATIONS: usize, T> {
    /// Scratch for `S⁻¹`; an observation of `m` measurements uses the first `m²` values.
    S_inv: [[T; MAX_OBSERVATIONS]; MAX_OBSERVATIONS],
    /// Scratch for `H×P`; an observation of `m` measurements uses the first `m·STATES` values.
    HP: [[T; STATES]; MAX_OBSERVATIONS],
    /// Scratch for `P×Hᵀ`; an observation of `m` measurements uses the first `STATES·m` values.
    PHt: [[T; MAX_OBSERVATIONS]; STATES],
    /// Scratch for `K×H×P`, independent of the number of measurements.
    KHP: [[T; STATES]; STATES],
}

/// The four temporaries of one correction of `OBSERVATIONS` measurements, as views into a
/// [`CorrectionScratch`] that live as long as its borrow.
#[allow(non_snake_case)]
pub(crate) struct ScratchViews<'a, const STATES: usize, const OBSERVATIONS: usize, T> {
    /// `S⁻¹` (`OBSERVATIONS × OBSERVATIONS`).
    pub(crate) S_inv: MatrixDataMut<'a, OBSERVATIONS, OBSERVATIONS, T>,
    /// `H×P` (`OBSERVATIONS × STATES`).
    pub(crate) HP: MatrixDataMut<'a, OBSERVATIONS, STATES, T>,
    /// `P×Hᵀ` (`STATES × OBSERVATIONS`).
    pub(crate) PHt: MatrixDataMut<'a, STATES, OBSERVATIONS, T>,
    /// `K×H×P` (`STATES × STATES`).
    pub(crate) KHP: MatrixDataMut<'a, STATES, STATES, T>,
}

impl<const STATES: usize, const MAX_OBSERVATIONS: usize, T: Copy>
    CorrectionScratch<STATES, MAX_OBSERVATIONS, T>
{
    /// Creates the scratch with every entry set to `init`, e.g. `0.0` or `I16F16::ZERO`.
    ///
    /// The value does not matter for the results, since every correction writes each
    /// temporary before it reads it.
    pub const fn new(init: T) -> Self {
        Self {
            S_inv: [[init; MAX_OBSERVATIONS]; MAX_OBSERVATIONS],
            HP: [[init; STATES]; MAX_OBSERVATIONS],
            PHt: [[init; MAX_OBSERVATIONS]; STATES],
            KHP: [[init; STATES]; STATES],
        }
    }
}

impl<const STATES: usize, const MAX_OBSERVATIONS: usize, T>
    CorrectionScratch<STATES, MAX_OBSERVATIONS, T>
{
    /// Returns the maximum number of measurements of an observation this scratch serves.
    #[inline(always)]
    pub const fn max_observations(&self) -> usize {
        MAX_OBSERVATIONS
    }

    /// Splits the scratch into the four temporaries of an observation of `OBSERVATIONS`
    /// measurements, each sliced to exactly its size.
    ///
    /// Fails to compile if `OBSERVATIONS > MAX_OBSERVATIONS`.
    pub(crate) fn views<const OBSERVATIONS: usize>(
        &mut self,
    ) -> ScratchViews<'_, STATES, OBSERVATIONS, T> {
        const {
            assert!(
                OBSERVATIONS <= MAX_OBSERVATIONS,
                "the observation has more measurements than the CorrectionScratch serves"
            );
        }

        ScratchViews {
            S_inv: MatrixDataMut::new(
                &mut self.S_inv.as_flattened_mut()[..OBSERVATIONS * OBSERVATIONS],
            ),
            HP: MatrixDataMut::new(&mut self.HP.as_flattened_mut()[..OBSERVATIONS * STATES]),
            PHt: MatrixDataMut::new(&mut self.PHt.as_flattened_mut()[..STATES * OBSERVATIONS]),
            KHP: self.KHP.as_flattened_mut().into(),
        }
    }
}
