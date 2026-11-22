//! The time-step handle that ties control inputs to the prediction.

/// A time update in progress: the prediction has run, and control inputs may follow.
///
/// [`RegularKalman::predict`](crate::regular::RegularKalman::predict) and
/// [`predict_tuned`](crate::regular::RegularKalman::predict_tuned) return this handle. It
/// borrows the filter mutably, so the borrow checker only lets controls in between the
/// prediction and the first correction:
///
/// ```text
/// x ← A·x + B₁·u₁ + B₂·u₂
/// P ← A·P·Aᵀ + Q + B₁·Qᵤ₁·B₁ᵀ + B₂·Qᵤ₂·B₂ᵀ
/// ```
///
/// Each [`Control`](crate::regular::Control) carries its own `B`, `u` and `Qᵤ`, so any number
/// of *distinct* controls can be chained. Applying the same control twice in one step adds
/// its input twice; this is a misuse the handle does not prevent. Dropping the handle without
/// a control leaves the plain prediction, so `filter.predict();` keeps working.
///
/// ## Example
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
/// use minikalman::prelude::*;
/// use minikalman::regular::{ControlStorage, FilterStorage, ObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut throttle_storage = ControlStorage::<2, 1, f32>::new(0.0);
/// let mut steering_storage = ControlStorage::<2, 1, f32>::new(0.0);
/// let mut gps_storage = ObservationStorage::<2, 1, f32>::new(0.0);
///
/// let mut filter = filter_storage.build();
/// let mut throttle = throttle_storage.build();
/// let mut steering = steering_storage.build();
/// let mut gps = gps_storage.build();
///
/// filter.state_transition_mut().make_identity();
/// filter.estimate_covariance_mut().make_identity();
/// throttle.control_matrix_mut().apply(|b| b.set_at(0, 0, 1.0));
/// throttle.control_vector_mut().apply(|u| u[0] = 2.0);
/// steering.control_matrix_mut().apply(|b| b.set_at(1, 0, 1.0));
/// steering.control_vector_mut().apply(|u| u[0] = 3.0);
/// gps.observation_matrix_mut().apply(|h| h.set_at(0, 0, 1.0));
/// gps.measurement_noise_covariance_mut().make_identity();
///
/// filter
///     .predict() // x ← A·x, P ← A·P·Aᵀ + Q
///     .control(&mut throttle) // x ← x + B₁·u₁, P ← P + B₁·Qᵤ₁·B₁ᵀ
///     .control(&mut steering); // x ← x + B₂·u₂, P ← P + B₂·Qᵤ₂·B₂ᵀ
/// assert_eq!(filter.state_vector().as_slice(), &[2.0, 3.0]);
///
/// gps.measurement_vector_mut().apply(|z| z[0] = 2.0);
/// filter.correct(&mut gps);
/// # }
/// ```
///
/// A control after a correction does not compile, since the handle still borrows the filter:
///
/// ```compile_fail,E0499
/// use minikalman::prelude::*;
/// use minikalman::regular::{ControlStorage, FilterStorage, ObservationStorage};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut control_storage = ControlStorage::<2, 1, f32>::new(0.0);
/// let mut gps_storage = ObservationStorage::<2, 1, f32>::new(0.0);
/// let mut filter = filter_storage.build();
/// let mut control = control_storage.build();
/// let mut gps = gps_storage.build();
///
/// let mut step = filter.predict();
/// filter.correct(&mut gps);
/// step.control(&mut control);
/// ```
///
/// Nor does a handle without a prediction; only `predict` creates one:
///
/// ```compile_fail,E0624
/// use minikalman::regular::{ControlStorage, FilterStorage, TimeUpdate};
///
/// let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
/// let mut control_storage = ControlStorage::<2, 1, f32>::new(0.0);
/// let mut filter = filter_storage.build();
/// let mut control = control_storage.build();
///
/// TimeUpdate::new(&mut filter).control(&mut control);
/// ```
#[doc(alias = "time_step")]
pub struct TimeUpdate<'a, F> {
    /// The filter whose prediction just ran; controls are applied to its state and covariance.
    pub(crate) filter: &'a mut F,
}

impl<'a, F> TimeUpdate<'a, F> {
    /// Wraps a filter whose prediction has just run.
    ///
    /// Only the filters' prediction methods call this, so a handle always follows a prediction.
    #[inline(always)]
    pub(crate) fn new(filter: &'a mut F) -> Self {
        Self { filter }
    }
}
