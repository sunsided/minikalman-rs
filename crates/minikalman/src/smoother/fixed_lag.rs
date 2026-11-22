//! # Fixed-lag Rauch-Tung-Striebel smoother
//!
//! The [`RtsSmoother`] needs the whole run before it can smooth. The [`FixedLagSmoother`]
//! works online instead: it keeps the last `N` filter steps in statically sized ring buffers
//! and, after every new step `t`, runs the RTS backward pass over that window. The result is the
//! estimate `x_t−N+1|t` of the step `N−1` steps back, refined with the `N−1` measurements that
//! came after it. That is the same value the fixed-interval smoother computes for that step over
//! the history `0..=t`. The [`FixedLagSmoother`] docs show this as a block diagram.
//!
//! Each push costs `N−1` RTS steps, each with a Cholesky inversion of a `STATES × STATES`
//! matrix.
//!
//! ## Example
//!
//! ```
//! # #[cfg(all(feature = "alloc", any(feature = "std", feature = "libm", feature = "micromath")))] {
//! use minikalman::matrix::MatrixDataArray;
//! use minikalman::models::motion;
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//! use minikalman::smoother::{FixedLagSmoother, RtsSmoother};
//!
//! type Vector = MatrixDataArray<2, 1, 2, f32>;
//! type Square = MatrixDataArray<2, 2, 4, f32>;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build();
//! let mut measurement = builder.observations().build::<1>();
//!
//! let model = motion::constant_velocity(1.0, 0.1);
//! filter.state_transition_mut().apply(|a| model.fill_transition(a));
//! filter.direct_process_noise_mut().apply(|q| model.fill_process_noise(q));
//! filter.estimate_covariance_mut().apply(|p| p.make_scalar(10.0));
//! measurement.observation_matrix_mut().apply(|h| h.set(0, 0, 1.0));
//! measurement.measurement_noise_covariance_mut().apply(|r| r.set(0, 0, 1.0));
//!
//! let rts = RtsSmoother::new(
//!     Square::new([0.0; 4]),
//!     Square::new([0.0; 4]),
//!     Square::new([0.0; 4]),
//!     Vector::new([0.0; 2]),
//! );
//! let mut smoother =
//!     FixedLagSmoother::<2, 3, f32, _, _, _, _, _, _>::new(rts, Vector::new([0.0; 2]), Square::new([0.0; 4]));
//!
//! let (mut x, mut p) = (Vector::new([0.0; 2]), Square::new([0.0; 4]));
//! let (mut x_predicted, mut p_predicted) = (x.clone(), p.clone());
//!
//! for (t, z) in [0.1, 1.2, 1.9, 3.1].into_iter().enumerate() {
//!     filter.predict();
//!     filter.state_vector().inspect(|v| v.copy_to(&mut x_predicted));
//!     filter.estimate_covariance().inspect(|m| m.copy_to(&mut p_predicted));
//!
//!     measurement.measurement_vector_mut().apply(|v| v.set(0, 0, z));
//!     filter.correct(&mut measurement);
//!     filter.state_vector().inspect(|v| v.copy_to(&mut x));
//!     filter.estimate_covariance().inspect(|m| m.copy_to(&mut p));
//!
//!     let smoothed = filter.state_transition().inspect(|a| {
//!         smoother.push(a, &x, &p, &x_predicted, &p_predicted)
//!     });
//!
//!     // The window of 3 steps is full from step 2 on; the estimate is for step t − 2.
//!     assert_eq!(smoothed.is_some(), t >= 2);
//!     if smoothed.is_some() {
//!         let _position = smoother.state().get(0, 0);
//!     }
//! }
//! # }
//! ```

use super::RtsSmoother;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};

/// Smooths each filter step with the `N−1` steps that follow it, online, over a ring buffer of
/// the last `N` steps.
///
#[doc = diagram!("fixed-lag-smoother.mmd", "ring")]
/// All buffers are sized at compile time through `N`, so the smoother does not allocate. The
/// state transition matrix `A` is constant, as for the [`RtsSmoother`].
pub struct FixedLagSmoother<const STATES: usize, const N: usize, T, X, P, Gain, Inv, Temp, TempX> {
    /// Runs the RTS steps over the window.
    rts: RtsSmoother<STATES, T, Gain, Inv, Temp, TempX>,
    /// Filtered states `x_k|k` of the window, indexed through [`Self::slot`].
    x: [X; N],
    /// Filtered covariances `P_k|k` of the window, indexed through [`Self::slot`].
    p: [P; N],
    /// Predicted states `x_k|k−1` of the window, indexed through [`Self::slot`].
    x_predicted: [X; N],
    /// Predicted covariances `P_k|k−1` of the window, indexed through [`Self::slot`].
    p_predicted: [P; N],
    /// The smoothed state of the oldest step after a full push; scratch during the pass.
    x_smoothed: X,
    /// The smoothed covariance of the oldest step after a full push; scratch during the pass.
    p_smoothed: P,
    /// Scratch for the state of the step being smoothed; swapped with [`Self::x_smoothed`].
    x_work: X,
    /// Scratch for the covariance of the step being smoothed; swapped with [`Self::p_smoothed`].
    p_work: P,
    /// Ring position the next push writes to; the oldest step once the window is full.
    /// Always `< N`.
    head: usize,
    /// Number of steps pushed so far, saturating at `N`.
    len: usize,
}

impl<const STATES: usize, const N: usize, T, X, P, Gain, Inv, Temp, TempX>
    FixedLagSmoother<STATES, N, T, X, P, Gain, Inv, Temp, TempX>
where
    X: MatrixMut<STATES, 1, T> + Clone,
    P: MatrixMut<STATES, STATES, T> + Clone,
    Gain: MatrixMut<STATES, STATES, T>,
    Inv: MatrixMut<STATES, STATES, T>,
    Temp: MatrixMut<STATES, STATES, T>,
    TempX: MatrixMut<STATES, 1, T>,
    T: MatrixDataType,
{
    /// Creates an empty smoother with a window of `N` steps.
    ///
    /// `x` (`STATES × 1`) and `p` (`STATES × STATES`) are cloned into the `2N + 2` vector and
    /// `2N + 2` matrix buffers of the window; their contents do not matter.
    ///
    /// ## Panics
    /// Panics if `N` is zero.
    pub fn new(rts: RtsSmoother<STATES, T, Gain, Inv, Temp, TempX>, x: X, p: P) -> Self {
        assert!(N > 0, "the window must hold at least one step");

        Self {
            rts,
            x: core::array::from_fn(|_| x.clone()),
            p: core::array::from_fn(|_| p.clone()),
            x_predicted: core::array::from_fn(|_| x.clone()),
            p_predicted: core::array::from_fn(|_| p.clone()),
            x_smoothed: x.clone(),
            p_smoothed: p.clone(),
            x_work: x,
            p_work: p,
            head: 0,
            len: 0,
        }
    }

    /// Adds filter step `t` and smooths the step `N−1` steps back.
    ///
    #[doc = diagram!("fixed-lag-smoother.mmd", "push", "gain", "update")]
    /// * `x`, `p` — the filtered `x_t|t`, `P_t|t`, recorded after `correct()`.
    /// * `x_predicted`, `p_predicted` — the predicted `x_t|t−1`, `P_t|t−1`, recorded after
    ///   `predict()` (and any `control()`) but before `correct()`.
    ///
    /// The oldest step drops out of the window once it holds `N` steps.
    ///
    /// ## Returns
    /// * `None` while fewer than `N` steps have been pushed. Nothing is smoothed yet.
    /// * `Some(true)` once the window is full: [`state`](Self::state) and
    ///   [`covariance`](Self::covariance) now hold `x_t−N+1|t` and `P_t−N+1|t`.
    /// * `Some(false)` as above, but at least one predicted covariance in the window was not
    ///   positive definite and its step kept the filtered values, as in [`RtsSmoother::smooth`].
    #[allow(non_snake_case)]
    pub fn push<A, XF, PF, XP, PP>(
        &mut self,
        A: &A,
        x: &XF,
        p: &PF,
        x_predicted: &XP,
        p_predicted: &PP,
    ) -> Option<bool>
    where
        A: Matrix<STATES, STATES, T>,
        XF: Matrix<STATES, 1, T>,
        PF: Matrix<STATES, STATES, T>,
        XP: Matrix<STATES, 1, T>,
        PP: Matrix<STATES, STATES, T>,
    {
        x.copy_to(&mut self.x[self.head]);
        p.copy_to(&mut self.p[self.head]);
        x_predicted.copy_to(&mut self.x_predicted[self.head]);
        p_predicted.copy_to(&mut self.p_predicted[self.head]);
        self.head = (self.head + 1) % N;
        self.len = (self.len + 1).min(N);

        if self.len < N {
            return None;
        }

        Some(self.smooth_window(A))
    }

    /// The smoothed state `x_t−N+1|t` of the last full [`push`](Self::push).
    ///
    /// Unspecified until a push has returned `Some`.
    pub fn state(&self) -> &X {
        &self.x_smoothed
    }

    /// The smoothed covariance `P_t−N+1|t` of the last full [`push`](Self::push).
    ///
    /// Unspecified until a push has returned `Some`.
    pub fn covariance(&self) -> &P {
        &self.p_smoothed
    }

    /// Ring index of the `i`-th oldest step of a full window, `i < N`.
    fn slot(&self, i: usize) -> usize {
        (self.head + i) % N
    }

    /// Runs the RTS backward pass from the newest to the oldest step of the full window, leaving
    /// the window untouched and the result in `x_smoothed`, `p_smoothed`. Returns `false` if a
    /// step was skipped.
    #[allow(non_snake_case)]
    fn smooth_window(&mut self, A: &impl Matrix<STATES, STATES, T>) -> bool {
        let newest = self.slot(N - 1);
        self.x[newest].copy_to(&mut self.x_smoothed);
        self.p[newest].copy_to(&mut self.p_smoothed);

        let mut all_smoothed = true;
        for i in (0..N - 1).rev() {
            let (k, next) = (self.slot(i), self.slot(i + 1));
            self.x[k].copy_to(&mut self.x_work);
            self.p[k].copy_to(&mut self.p_work);
            all_smoothed &= self.rts.step(
                A,
                &mut self.x_work,
                &mut self.p_work,
                &self.x_smoothed,
                &self.p_smoothed,
                &self.x_predicted[next],
                &self.p_predicted[next],
            );
            core::mem::swap(&mut self.x_work, &mut self.x_smoothed);
            core::mem::swap(&mut self.p_work, &mut self.p_smoothed);
        }

        all_smoothed
    }
}
