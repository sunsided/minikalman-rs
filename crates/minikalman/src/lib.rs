//! # Kalman Filters for Embedded Targets
//!
//! This is the Rust port of the [kalman-clib](https://github.com/sunsided/kalman-clib/) library,
//! a microcontroller targeted Kalman filter implementation. It can use [`micromath`](https://docs.rs/micromath)
//! for square root and reciprocal calculations on `no_std`; [`libm`](https://docs.rs/libm) is supported as well.
//!
//! This implementation uses statically allocated buffers for all matrix operations. The storage
//! types, such as [`regular::FilterStorage`] and [`regular::ObservationStorage`], hold all buffers
//! of a filter part as const-generic arrays with a `const fn new`, and `build()` wires the filter
//! to them. For custom layouts, the helper macros create single buffers (see e.g.
//! [`impl_buffer_A`]).
//!
//! If allocation is available (via `std` or `alloc` crate features), the [`KalmanFilterBuilder`](regular::builder::KalmanFilterBuilder) can be
//! used to quickly create a [`RegularKalman`](regular::RegularKalman) filter instance with all necessary buffers, alongside
//! [`Control`](regular::Control) and [`Observation`](regular::RegularObservation) instances.
//! Similar types exist for Extended Kalman Filters.
//!
//! ## Choosing a Filter
//!
//! Pick the filter from the model, then add what the measurements need:
// `diagram!` is defined below and cannot be used in crate-level attributes.
#![doc = concat!(
    "\n```mermaid\n",
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/diagrams/choose-filter.mmd")),
    "```\n\n"
)]
//! Gating and noise adaptation are set on the observation
//! ([`RegularObservation::set_gate`](regular::RegularObservation::set_gate),
//! [`RegularObservation::set_noise_adaptation`](regular::RegularObservation::set_noise_adaptation)
//! and their extended and unscented counterparts). The predictively oriented correction is
//! [`RegularKalman::correct_predictively_oriented`](regular::RegularKalman::correct_predictively_oriented)
//! or [`ExtendedKalman::correct_nonlinear_predictively_oriented`](extended::ExtendedKalman::correct_nonlinear_predictively_oriented).
//! The outlier-robust Student-t correction is
//! [`RegularKalman::correct_robust`](regular::RegularKalman::correct_robust),
//! [`ExtendedKalman::correct_nonlinear_robust`](extended::ExtendedKalman::correct_nonlinear_robust)
//! or [`UnscentedKalman::correct_sigma_point_robust`](unscented::UnscentedKalman::correct_sigma_point_robust).
//! The [`smoother`] module refines a recorded run ([`RtsSmoother`](smoother::RtsSmoother))
//! or a sliding window ([`FixedLagSmoother`](smoother::FixedLagSmoother)).
//!
//! ## Wiring Up a Filter
//!
//! Filters, controls and observations own their matrices as buffers. Without allocation, the
//! storage types or the `impl_buffer_*` macros hold them in static or stack arrays; with the
//! `alloc` feature, the builders allocate them:
#![doc = concat!(
    "\n```mermaid\n",
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/diagrams/regular-setup.mmd")),
    "```\n\n"
)]
//! ## Crate Features
//!
//! * `std` - Disabled by default. Disables the `no_std` configuration attribute (enabling `std` support).
//! * `alloc` - Enables allocation support for builder types.
//! * `libm` - Enables [libm](https://crates.io/crates/libm) support.
//! * `micromath` - Enables [micromath](https://crates.io/crates/micromath) support. If `std` or
//!   `libm` is also enabled, those take precedence; use `default-features = false` to pick micromath.
//! * `fixed` - Enables fixed-point support via the [fixed](https://crates.io/crates/fixed) crate.
//! * `unsafe` - Enables some unsafe pointer operations. Disabled by default; when turned off,
//!   compiles the crate as `#![forbid(unsafe)]`.
//! * `nalgebra` - Disabled by default. Enables copies between minikalman's matrices and
//!   [nalgebra](https://crates.io/crates/nalgebra) matrices; see [nalgebra interop](#nalgebra-interop).
//!   Together with `unsafe`, a `MatrixDataArray` can also back an owned nalgebra matrix.
//!
//! ## nalgebra Interop
//!
//! minikalman does not depend on nalgebra, and no filter uses it. nalgebra's dependency tree
//! adds to binary size, which matters on small microcontrollers
//! ([#13](https://github.com/sunsided/minikalman-rs/issues/13)), so the `nalgebra` feature is
//! opt-in; without it, nalgebra is not compiled at all.
//!
//! With the feature, every matrix and buffer gets `to_nalgebra`, which copies it into a
//! stack-allocated nalgebra `SMatrix`. Mutable matrices and buffers also get
//! `copy_from_nalgebra`, which fills them from any nalgebra matrix of the same shape. Neither
//! needs the `unsafe` feature or allocates, and both keep every value at its `(row, column)`
//! index although nalgebra is column-major. The matching nalgebra version is re-exported as
//! `minikalman::nalgebra`.
//!
//! There are no public zero-copy views. nalgebra's products, `gemv` and `axpy` read and write
//! out of bounds on operands whose row stride is not 1, which a view of a row-major matrix has
//! ([dimforge/nalgebra#1616](https://github.com/dimforge/nalgebra/issues/1616), fix in review in
//! [dimforge/nalgebra#1627](https://github.com/dimforge/nalgebra/pull/1627)). The views can be
//! made public again once a fixed nalgebra release fits the MSRV
//! ([#138](https://github.com/sunsided/minikalman-rs/issues/138)).
//!
//! ```
//! # #[cfg(feature = "nalgebra")] {
//! use minikalman::buffers::types::KalmanGainMatrixBuffer;
//! use minikalman::nalgebra::SMatrix;
//! use minikalman::prelude::*;
//!
//! // A 3×2 Kalman gain, stored row by row.
//! let mut k = KalmanGainMatrixBuffer::<3, 2, f32, _>::from([0.0; 6]);
//!
//! k.copy_from_nalgebra(&SMatrix::<f32, 3, 2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
//! assert_eq!(k.as_slice()[4], 5.0);
//!
//! let kt = k.to_nalgebra().transpose();
//! assert_eq!(kt[(0, 2)], 5.0);
//! # }
//! ```
//!
//! ## Example
//!
//! On `std` or `alloc` crates, the [`KalmanFilterBuilder`](regular::builder::KalmanFilterBuilder) is enabled. An overly simplified example
//! for setting up and operating the Kalman Filter could look like this:
//!
//! ```no_run
//! # #[cfg(feature = "alloc")] {
//! use minikalman::regular::builder::KalmanFilterBuilder;
//! use minikalman::prelude::MatrixMut;
//!
//! const NUM_STATES: usize = 3;
//! const NUM_CONTROLS: usize = 2;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
//! let mut filter = builder.build();
//! let mut control = builder.controls().build::<NUM_CONTROLS>();
//! let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
//!
//! // Set up the system dynamics, control matrices, observation matrices, ...
//!
//! // Filter!
//! loop {
//!     // Update your control vector(s).
//!     control.control_vector_mut().apply(|u| {
//!         u[0] = 0.0;
//!         u[1] = 1.0;
//!     });
//!
//!     // Update your measurement vectors.
//!     measurement.measurement_vector_mut().apply(|z| {
//!         z[0] = 42.0;
//!     });
//!
//!     // Predict, then apply each control input to the prediction.
//!     filter.predict().control(&mut control);
//!
//!     // Apply any measurements.
//!     filter.correct(&mut measurement);
//!
//!     // Access the state
//!     let state = filter.state_vector();
//!     let covariance = filter.estimate_covariance();
//! }
//! # }
//! ```
//!
//! ## Extended Kalman Filters
//!
//! The general setup remains the same, however the `predict` and `correct` methods are
//! replaced with their nonlinear counterparts:
//!
//! ```no_run
//! # #[cfg(feature = "alloc")] {
//! use minikalman::extended::builder::KalmanFilterBuilder;
//! use minikalman::prelude::MatrixMut;
//!
//! const NUM_STATES: usize = 3;
//! const NUM_CONTROLS: usize = 2;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
//! let mut filter = builder.build();
//! let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
//!
//! // Set up the system dynamics, control matrices, observation matrices, ...
//!
//! // Filter!
//! loop {
//!     // Obtain the control values.
//!     let control_value = 1.0;
//!
//!     // Update prediction using nonlinear transfer function.
//!     filter.predict_nonlinear(|current, next| {
//!         next[0] = current[0] * current[0];
//!         next[1] = current[1].sin() * control_value;
//!     });
//!
//!     // Update your measurement vectors.
//!     measurement.measurement_vector_mut().apply(|z| {
//!         z[0] = 42.0;
//!     });
//!
//!     // Apply any measurements using a nonlinear measurement function.
//!     filter.correct_nonlinear(&mut measurement, |state, observation| {
//!         observation[0] = state[0].cos() + state[1].sin();
//!     });
//!
//!     // Access the state
//!     let state = filter.state_vector();
//!     let covariance = filter.estimate_covariance();
//! }
//! # }
//! ```
//!
//! ## `no_std` Example
//!
//! Systems without the liberty of heap allocation can use the storage types:
//!
//! ```
//! use minikalman::prelude::*;
//! use minikalman::regular::{FilterStorage, ObservationStorage};
//!
//! const NUM_STATES: usize = 3;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! let mut filter_storage = FilterStorage::<NUM_STATES, f32>::new(0.0);
//! let mut measurement_storage = ObservationStorage::<NUM_STATES, NUM_OBSERVATIONS, f32>::new(0.0);
//!
//! let mut filter = filter_storage.build();
//! let mut measurement = measurement_storage.build();
//! # filter.predict();
//! # filter.correct(&mut measurement);
//! ```
//!
//! The storage has the size of the buffers it holds and allocates nothing; `build()` borrows
//! it and moves no arrays. Large storage belongs in a `static` (see
//! [`regular::FilterStorage`] for placement), not on a small stack.
//!
//! Alternatively, the helper macros wire up single buffers. This comes at the cost of
//! potentially confusing IDEs due to recursive macro expansion, so buyer beware. In the example
//! below, types are set up as arrays bound to `static mut` variables. Each one is borrowed
//! exactly once through `&raw mut`, since edition 2024 denies plain references to a
//! `static mut`; outside a doc example, prefer the storage above or a take-once cell.
//!
//! ```
//! # #![allow(non_snake_case)]
//! # #![allow(non_upper_case_globals)]
//! use minikalman::buffers::types::*;
//! use minikalman::prelude::*;
//! use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};
//!
//! const NUM_STATES: usize = 3;
//! const NUM_OBSERVATIONS: usize = 1;
//!
//! // System buffers.
//! impl_buffer_x!(static mut gravity_x, NUM_STATES, f32, 0.0);
//! impl_buffer_A!(static mut gravity_A, NUM_STATES, f32, 0.0);
//! impl_buffer_P!(static mut gravity_P, NUM_STATES, f32, 0.0);
//! impl_buffer_Q_direct!(static mut gravity_Q, NUM_STATES, f32, 0.0);
//!
//! // Observation buffers.
//! impl_buffer_z!(static mut gravity_z, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_H!(static mut gravity_H, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
//! impl_buffer_R!(static mut gravity_R, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_y!(static mut gravity_y, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_S!(static mut gravity_S, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_K!(static mut gravity_K, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
//!
//! // Filter temporaries.
//! impl_buffer_temp_x!(static mut gravity_temp_x, NUM_STATES, f32, 0.0);
//! impl_buffer_temp_P!(static mut gravity_temp_P, NUM_STATES, f32, 0.0);
//!
//! // Observation temporaries.
//! impl_buffer_temp_S_inv!(static mut gravity_temp_S_inv, NUM_OBSERVATIONS, f32, 0.0);
//!
//! // Observation temporaries.
//! impl_buffer_temp_HP!(static mut gravity_temp_HP, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
//! impl_buffer_temp_PHt!(static mut gravity_temp_PHt, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
//! impl_buffer_temp_KHP!(static mut gravity_temp_KHP, NUM_STATES, f32, 0.0);
//!
//! // SAFETY: each `static mut` below is borrowed once and never re-entered.
//! let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(
//!     StateTransitionMatrixMutBuffer::from(unsafe { (&mut *&raw mut gravity_A).as_mut_slice() }),
//!     StateVectorBuffer::from(unsafe { (&mut *&raw mut gravity_x).as_mut_slice() }),
//!     EstimateCovarianceMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_P).as_mut_slice() }),
//!     DirectProcessNoiseCovarianceMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_Q).as_mut_slice() }),
//!     PredictedStateEstimateVectorBuffer::from(unsafe { (&mut *&raw mut gravity_temp_x).as_mut_slice() }),
//!     TemporaryStateMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_temp_P).as_mut_slice() }),
//! );
//!
//! let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
//!     ObservationMatrixMutBuffer::from(unsafe { (&mut *&raw mut gravity_H).as_mut_slice() }),
//!     MeasurementVectorBuffer::from(unsafe { (&mut *&raw mut gravity_z).as_mut_slice() }),
//!     MeasurementNoiseCovarianceMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_R).as_mut_slice() }),
//!     InnovationVectorBuffer::from(unsafe { (&mut *&raw mut gravity_y).as_mut_slice() }),
//!     InnovationCovarianceMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_S).as_mut_slice() }),
//!     KalmanGainMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_K).as_mut_slice() }),
//!     TemporaryResidualCovarianceInvertedMatrixBuffer::from(unsafe {
//!         (&mut *&raw mut gravity_temp_S_inv).as_mut_slice()
//!     }),
//!     TemporaryHPMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_temp_HP).as_mut_slice() }),
//!     TemporaryPHTMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_temp_PHt).as_mut_slice() }),
//!     TemporaryKHPMatrixBuffer::from(unsafe { (&mut *&raw mut gravity_temp_KHP).as_mut_slice() }),
//! );
//! ```
//!
//! After that, the `filter` and `measurement` variables can be used similar to the example above.

// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
// Enable `no_std` if the `no_std` crate feature is enabled.
#![cfg_attr(not(feature = "std"), no_std)]
// Forbid unsafe code unless the `unsafe` crate feature is explicitly enabled.
#![cfg_attr(not(feature = "unsafe"), forbid(unsafe_code))]

#[cfg(any(feature = "std", feature = "alloc"))]
extern crate alloc;
extern crate core;

/// Expands to a mermaid block for a doc comment, from the diagram `docs/diagrams/<file>`, with
/// the given nodes highlighted. See `docs/diagrams/README.md` for the visual language.
///
/// Use it as `#[doc = diagram!("regular-loop.mmd", "predict")]`.
macro_rules! diagram {
    ($file:literal $(, $focus:literal)*) => {
        concat!(
            "\n```mermaid\n",
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/diagrams/", $file)),
            $("    class ", $focus, " focus\n",)*
            "```\n\n"
        )
    };
}

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
mod kalman_builder;

pub mod angles;
pub mod buffers;
mod controls;
mod kalman;
pub mod matrix;
pub mod models;
mod observations;
pub mod smoother;
mod state_macro;
mod static_macros;
mod storage;

#[cfg(test)]
mod test_dummies;

#[cfg(all(test, feature = "alloc"))]
mod test_filter;

#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[cfg(feature = "alloc")]
pub use crate::buffers::builder::BufferBuilder;

/// Re-export `num_traits`.
pub use num_traits;

/// Re-export `nalgebra`, so the copies use the same nalgebra version as your code.
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
#[cfg(feature = "nalgebra")]
pub use nalgebra;

/// Compiles and runs the code examples in the README as doctests.
// Gated on `alloc` here because GitHub shows rustdoc's hidden `#` lines, so the README
// examples cannot carry their own `cfg` guards.
#[cfg(all(doctest, feature = "alloc"))]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDoctests;

/// Runs the README's statically allocated example without the `alloc` gate, to show it needs
/// neither `std` nor `alloc`. `tests/readme.rs` checks that the README contains it verbatim.
#[cfg(doctest)]
#[doc = include_str!("../docs/readme-no-alloc.md")]
pub struct ReadmeNoAllocDoctest;

/// Regular Kalman Filters
///
/// A linear Kalman filter for `x̂ₖ = A·x̂ₖ₋₁ + B·u` and `z = H·x̂ + v`. Each step predicts with
/// [`RegularKalman::predict`](regular::RegularKalman::predict), optionally applies one or more
/// [`Control`](regular::Control)s through the returned [`TimeUpdate`](regular::TimeUpdate)
/// (`filter.predict().control(&mut throttle).control(&mut steering)`), and corrects with one
/// or more [`RegularObservation`](regular::RegularObservation)s:
#[doc = diagram!("regular-loop.mmd")]
///
/// For a time-invariant model, the [`LuenbergerObserver`](regular::LuenbergerObserver) runs a
/// fixed gain without any covariance, e.g. the steady-state gain that
/// [`RegularKalman::settle_gain`](regular::RegularKalman::settle_gain) computes from `Q` and `R`.
pub mod regular {
    pub use crate::controls::*;
    pub use crate::kalman::TimeUpdate;
    pub use crate::kalman::luenberger::*;
    pub use crate::kalman::regular::*;
    pub use crate::observations::fixed_gain::*;
    pub use crate::observations::predictively_oriented::PredictivelyOrientedWorkspace;
    pub use crate::observations::regular::*;
    pub use crate::observations::robust::{RobustCorrection, RobustWeight};
    pub use crate::observations::scratch::CorrectionScratch;
    pub use crate::observations::{CovarianceUpdate, NoiseAdaptation};
    pub use crate::storage::luenberger::*;
    pub use crate::storage::regular::*;

    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub use crate::kalman_builder::regular as builder;
}

/// Information Filters: linear Kalman filters in information form, for additive sensor fusion.
///
/// The [`InformationFilter`](information::InformationFilter) keeps the information matrix
/// `Y = P⁻¹` and the information vector `ŷ = Y·x̂` instead of `x̂` and `P`. Each step predicts
/// with [`predict`](information::InformationFilter::predict), then adds every sensor with
/// [`fuse`](information::InformationFilter::fuse); [`state`](information::InformationFilter::state)
/// converts back to `x̂` and `P` when needed:
#[doc = diagram!("information-loop.mmd")]
pub mod information {
    pub use crate::kalman::information::*;
    pub use crate::storage::information::*;
}

/// UD Filters: linear Kalman filters with the covariance factored as `P = U·D·Uᵀ`, without
/// square roots, for fixed-point targets and ill-conditioned problems.
///
/// The [`UdKalman`](ud::UdKalman) keeps `P` positive semi-definite by construction. Each step
/// predicts with [`predict`](ud::UdKalman::predict) (Thornton's MWGS) and corrects with
/// [`correct`](ud::UdKalman::correct) (Bierman's sequential update) from regular observations;
/// [`state`](ud::UdKalman::state) recovers `P` when needed:
#[doc = diagram!("ud-loop.mmd")]
pub mod ud {
    pub use crate::kalman::ud::*;
}

/// Recursive Least Squares (RLS): online estimation of model parameters, with forgetting.
///
/// The [`RecursiveLeastSquares`](rls::RecursiveLeastSquares) estimator identifies a parameter
/// vector `θ` of a model whose output is linear in it, `yₖ = φₖᵀ·θ + vₖ`, e.g. the pole and
/// gain of a plant or a battery's internal resistance, from one sample at a time in `O(N²)`.
/// It is the Kalman filter with `A = I`, `Q = 0` and `H = φₖᵀ`; forgetting with the factor
/// `μ` is [`RegularKalman::predict_tuned(√μ)`](regular::RegularKalman::predict_tuned) while
/// the windup guard is off or inactive.
/// Its estimate can feed a Kalman filter's `A` and `B` each step (dual estimation, see the
/// `rls-adaptive` example). Each sample forgets, then corrects:
#[doc = diagram!("rls-update.mmd")]
pub mod rls {
    pub use crate::kalman::rls::*;
    pub use crate::storage::rls::*;
}

/// Square-Root Covariance Filters: linear Kalman filters that propagate `S` with `P = S·Sᵀ`.
///
/// The [`SquareRootKalman`](square_root::SquareRootKalman) keeps `P` positive semi-definite by
/// construction and stores standard deviations instead of variances, which helps with `f32`
/// and fixed point. It predicts with a QR triangularization
/// ([`predict`](square_root::SquareRootKalman::predict)) and corrects with Potter's sequential
/// update ([`correct`](square_root::SquareRootKalman::correct)), using the same
/// [`RegularObservation`](regular::RegularObservation)s as the regular filter:
#[doc = diagram!("square-root-loop.mmd")]
pub mod square_root {
    pub use crate::kalman::square_root::*;
}

/// Extended Kalman Filters (EKF)
///
/// A Kalman filter for `x̂ₖ = f(x̂ₖ₋₁)` and `z = h(x̂) + v` with nonlinear `f` and `h`. Before each
/// step, the caller writes the Jacobians `A = ∂f/∂x` and `H = ∂h/∂x` at the current state. Each
/// step predicts with
/// [`ExtendedKalman::predict_nonlinear`](extended::ExtendedKalman::predict_nonlinear) and
/// corrects with one or more [`ExtendedObservation`](extended::ExtendedObservation)s via
/// [`ExtendedKalman::correct_nonlinear`](extended::ExtendedKalman::correct_nonlinear):
#[doc = diagram!("extended-loop.mmd")]
///
/// ## Filtering on manifolds: error-state and invariant EKF
///
/// Attitudes and poses live on a Lie group such as SO(3) or SE(2), not in a vector space, so
/// `x̂ ← x̂ + K·y` on Euler angles or raw quaternion entries breaks the group structure. The
/// error-state (multiplicative) EKF and the invariant EKF (Barrau and Bonnabel, "The
/// Invariant Extended Kalman Filter as a Stable Observer", IEEE TAC 62(4), 2017,
/// [arXiv:1410.1465](https://arxiv.org/abs/1410.1465)) avoid this: the caller keeps the
/// *nominal state* `X̂` on the group, and the filter estimates only a small error `ξ` in the
/// Lie algebra, with `X = X̂·Exp(ξ)` (or `Exp(ξ)·X̂`). [`ExtendedKalman`](extended::ExtendedKalman)
/// runs this without a Lie-group library, with `ξ` as its state `x`:
#[doc = diagram!("extended-error-state.mmd", "nominal", "inject")]
/// 1. **Predict.** Move `X̂` on the group in caller code, write the Jacobian of the error
///    dynamics with [`state_transition_jacobian_mut`](extended::ExtendedKalman::state_transition_jacobian_mut),
///    then call [`predict_nonlinear`](extended::ExtendedKalman::predict_nonlinear) with a
///    closure that writes `ξ = 0`. Only `P ← A·P·Aᵀ + Q` matters.
/// 2. **Correct.** Write the error Jacobian with `observation_jacobian_matrix_mut` and call
///    [`correct_nonlinear_with`](extended::ExtendedKalman::correct_nonlinear_with). The
///    observation closure ignores `x` and writes the prediction from `X̂`; the `fix` closure
///    can turn `z − h(X̂)` into an invariant innovation, e.g. rotate it into the body frame.
///    Both closures may borrow `X̂` from a field next to the filter. The gate and the NIS see
///    the innovation after `fix`. A rotated innovation `R̂ᵀ·y` has the noise covariance
///    `R̂ᵀ·R·R̂`, so write that into `measurement_noise_covariance_mut` before each correction
///    unless `R` is isotropic (`σ²·I`), which is the same in every frame.
/// 3. **Inject.** Read `ξ` from [`state_vector`](extended::ExtendedKalman::state_vector),
///    set `X̂ ← X̂·Exp(ξ)`, reset `ξ` to zero via
///    [`state_vector_mut`](extended::ExtendedKalman::state_vector_mut), and optionally move
///    `P` to the new error with the reset Jacobian, `P ← J·P·Jᵀ`, via
///    [`estimate_covariance_mut`](extended::ExtendedKalman::estimate_covariance_mut). For
///    `X̂·Exp(ξ)` that Jacobian is the right Jacobian `J_r(ξ)`; it is close to `I` for small `ξ`.
///
/// **When the invariant form pays off.** For dynamics with
/// `f(a·b) = a·f(b) + f(a)·b − a·f(I)·b` ("group affine", which includes odometry and gyro
/// integration `X̂ ← X̂·Exp(u)`), the invariant error evolves independently of the estimate,
/// so `A` depends only on the input, e.g. `A = Ad_Exp(u)⁻¹`. If the observation is invariant
/// too, e.g. a world-frame position fix `Y = X·d` paired with the body-frame error, `H` is
/// constant. Neither Jacobian is then evaluated at a wrong estimate, and Barrau and Bonnabel
/// prove local stability around any trajectory under the conditions of the linear Kalman
/// filter (Theorems 1 and 4 and eq. (7) in arXiv:1410.1465v4). The guarantee is local, and the
/// correction is still first order. Without that structure the error-state EKF still keeps
/// the state on the group, but `A` and `H` depend on `X̂` as in any EKF.
///
/// The `se2-iekf` example (`examples/se2_iekf.rs`) runs this on SE(2) with odometry and
/// position fixes, next to a plain EKF on `(x, y, θ)` started with the same 170° heading
/// error. Both converge there; with the default seed the invariant filter keeps its heading
/// within 5° after about 21 s, the plain EKF after about 46 s.
pub mod extended {
    pub use crate::kalman::extended::*;
    pub use crate::observations::extended::*;
    pub use crate::observations::robust::{RobustCorrection, RobustWeight};
    pub use crate::observations::scratch::CorrectionScratch;
    pub use crate::observations::{CovarianceUpdate, NoiseAdaptation};
    pub use crate::storage::extended::*;

    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub use crate::kalman_builder::extended as builder;
}

/// Unscented Kalman Filters (UKF)
///
/// A nonlinear Kalman filter without Jacobians: it propagates a set of sigma points through
/// the state transition `f` and the observation `h` and fits a mean and covariance to them.
/// Each step predicts with
/// [`UnscentedKalman::predict_nonlinear`](unscented::UnscentedKalman::predict_nonlinear) and
/// corrects with one or more [`UnscentedObservation`](unscented::UnscentedObservation)s
/// through
/// [`correct_sigma_point`](unscented::KalmanFilterSigmaPointCorrect::correct_sigma_point):
#[doc = diagram!("unscented-loop.mmd")]
pub mod unscented {
    pub use crate::kalman::unscented::*;
    pub use crate::kalman::{
        CrossCovarianceMatrix, KalmanFilterSigmaPointCorrect, KalmanFilterSigmaPointPredict,
        KalmanFilterUnscentedObservationCorrectFilter,
        KalmanFilterUnscentedObservationCorrectRecalibrated,
        KalmanFilterUnscentedObservationCorrectRobust, KalmanFilterUnscentedParams,
        KalmanFilterUnscentedParamsMut, SigmaObservedMatrix, SigmaPointMatrix,
        SigmaPropagatedMatrix, SigmaWeightsVector, SigmaWeightsVectorMut, TempSigmaPMatrix,
        UnscentedKalmanFilter,
    };
    pub use crate::observations::NoiseAdaptation;
    pub use crate::observations::robust::{RobustCorrection, RobustWeight};
    pub use crate::observations::unscented::*;
    pub use crate::storage::unscented::*;

    /// Builder types for unscented Kalman filters
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub mod builder {
        pub use crate::kalman_builder::unscented::*;
    }
}

/// Exports all macros and common types.
pub mod prelude {
    #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
    #[cfg(feature = "alloc")]
    pub use crate::buffers::builder::*;

    pub use crate::kalman::*;
    pub use crate::matrix::*;
    pub use crate::observations::predictively_oriented::PredictivelyOrientedWorkspace;
    pub use crate::observations::robust::{RobustCorrection, RobustWeight};
    pub use crate::observations::scratch::CorrectionScratch;
    pub use crate::observations::{CovarianceUpdate, NoiseAdaptation};

    pub use crate::{
        impl_buffer_A, impl_buffer_B, impl_buffer_H, impl_buffer_K, impl_buffer_P,
        impl_buffer_Q_control, impl_buffer_Q_direct, impl_buffer_R, impl_buffer_S,
        impl_buffer_cross_covariance, impl_buffer_sigma_observed, impl_buffer_sigma_points,
        impl_buffer_sigma_propagated, impl_buffer_sigma_weights, impl_buffer_temp_BQ,
        impl_buffer_temp_HP, impl_buffer_temp_KHP, impl_buffer_temp_P, impl_buffer_temp_PHt,
        impl_buffer_temp_S_inv, impl_buffer_temp_prearray, impl_buffer_temp_sigma_P,
        impl_buffer_temp_x, impl_buffer_u, impl_buffer_ud_D, impl_buffer_ud_W, impl_buffer_x,
        impl_buffer_y, impl_buffer_z,
    };

    pub use crate::{
        size_buffer_A, size_buffer_B, size_buffer_H, size_buffer_K, size_buffer_P,
        size_buffer_Q_control, size_buffer_Q_direct, size_buffer_R, size_buffer_S, size_buffer_u,
        size_buffer_x, size_buffer_y, size_buffer_z,
    };
    pub use crate::{
        size_buffer_sigma_observed, size_buffer_sigma_points, size_buffer_sigma_weights,
    };
    pub use crate::{
        size_buffer_temp_BQ, size_buffer_temp_HP, size_buffer_temp_KHP, size_buffer_temp_P,
        size_buffer_temp_PHt, size_buffer_temp_S_inv, size_buffer_temp_x,
    };

    #[cfg_attr(docsrs, doc(cfg(feature = "fixed")))]
    #[cfg(feature = "fixed")]
    pub use fixed::types::{I16F16, I32F32};
}

/// Sizes a buffer fitting the state transition matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_A!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_A {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the state covariance matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_P!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_P {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the state vector (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_x!(NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_x {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the direct process noise matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_Q_direct!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_Q_direct {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the control vector (`num_controls` × `1`).
///
/// ## Arguments
/// * `num_controls` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_u!(NUM_CONTROLS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_u {
    ( $num_controls:expr ) => {{
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_CONTROLS_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the control transition matrix (`num_states` × `num_controls`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
/// * `num_controls` - The number of controls.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_B!(NUM_STATES, NUM_CONTROLS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_B {
    ( $num_states:expr, $num_controls:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_STATES_ * NUM_CONTROLS_) as usize
    }};
}

/// Sizes a buffer fitting the control process noise matrix (`num_controls` × `num_controls`).
///
/// ## Arguments
/// * `num_controls` - The number of control inputs to the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_Q_control!(NUM_CONTROLS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_Q_control {
    ( $num_controls:expr ) => {{
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_CONTROLS_ * NUM_CONTROLS_) as usize
    }};
}

/// Sizes a buffer fitting the measurement vector z (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_z!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_z {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the measurement transformation matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_H!(NUM_OBSERVATIONS, NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_H {
    ( $num_measurements:expr, $num_states:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_OBSERVATIONS_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the measurement uncertainty matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_R!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_R {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * NUM_OBSERVATIONS_) as usize
    }};
}

/// Sizes a buffer fitting the innovation vector (`num_measurements` × `1`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_y!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_y {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the innovation covariance matrix (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_S!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_S {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * NUM_OBSERVATIONS_) as usize
    }};
}

/// Sizes a buffer fitting the Kalman gain matrix (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_K!(NUM_STATES, NUM_OBSERVATIONS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_K {
    ( $num_states:expr, $num_measurements:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_STATES_ * NUM_OBSERVATIONS_) as usize
    }};
}

/// Sizes a buffer fitting the temporary x predictions (`num_states` × `1`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_temp_x!(NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_x {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * 1) as usize
    }};
}

/// Sizes a buffer fitting the temporary P matrix (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_temp_P!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_P {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the temporary B×Q matrix (`num_states` × `num_controls`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_controls` - The number of controls.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_CONTROLS: usize = 1;
/// assert_eq!(size_buffer_temp_BQ!(NUM_STATES, NUM_CONTROLS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_BQ {
    ( $num_states:expr, $num_controls:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_CONTROLS_: usize = ($num_controls) as usize;
        (NUM_STATES_ * NUM_CONTROLS_) as usize
    }};
}

/// Sizes a buffer fitting the temporary S-inverted (`num_measurements` × `num_measurements`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_temp_S_inv!(NUM_OBSERVATIONS), 1);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_S_inv {
    ( $num_measurements:expr ) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_OBSERVATIONS_ * NUM_OBSERVATIONS_) as usize
    }};
}

/// Sizes a buffer fitting the temporary H×P matrix (`num_measurements` × `num_states`).
///
/// ## Arguments
/// * `num_measurements` - The number of measurements.
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_temp_HP!(NUM_OBSERVATIONS, NUM_STATES), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_HP {
    ( $num_measurements:expr, $num_states:expr) => {{
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_OBSERVATIONS_) as usize
    }};
}

/// Sizes a buffer fitting the temporary P×H^-1 buffer (`num_states` × `num_measurements`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_measurements` - The number of measurements.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 1;
/// assert_eq!(size_buffer_temp_PHt!(NUM_STATES, NUM_OBSERVATIONS), 3);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_PHt {
    ( $num_states:expr, $num_measurements:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_OBSERVATIONS_: usize = ($num_measurements) as usize;
        (NUM_STATES_ * NUM_OBSERVATIONS_) as usize
    }};
}

/// Sizes a buffer fitting the temporary K×(H×P) buffer (`num_states` × `num_states`).
///
/// ## Arguments
/// * `num_states` - The number of states.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_temp_KHP!(NUM_STATES), 9);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_temp_KHP {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * NUM_STATES_) as usize
    }};
}

/// Sizes a buffer fitting the sigma point matrix (`num_states` × `2*num_states+1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_sigma_points!(NUM_STATES), 21);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_sigma_points {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_STATES_ * (2 * NUM_STATES_ + 1)) as usize
    }};
}

/// Sizes a buffer fitting the sigma weights vector (`2*num_states+1`).
///
/// ## Arguments
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_sigma_weights!(NUM_STATES), 7);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_sigma_weights {
    ( $num_states:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        (2 * NUM_STATES_ + 1) as usize
    }};
}

/// Sizes a buffer fitting the observed sigma points (`num_observations` × `2*num_states+1`).
///
/// ## Arguments
/// * `num_observations` - The number of observations.
/// * `num_states` - The number of states describing the system.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_OBSERVATIONS: usize = 2;
/// const NUM_STATES: usize = 3;
/// assert_eq!(size_buffer_sigma_observed!(NUM_OBSERVATIONS, NUM_STATES), 14);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_sigma_observed {
    ( $num_observations:expr, $num_states:expr ) => {{
        const NUM_OBS_: usize = ($num_observations) as usize;
        const NUM_STATES_: usize = ($num_states) as usize;
        (NUM_OBS_ * (2 * NUM_STATES_ + 1)) as usize
    }};
}

/// Sizes a buffer fitting the cross-covariance matrix (`num_states` × `num_observations`).
///
/// ## Arguments
/// * `num_states` - The number of states.
/// * `num_observations` - The number of observations.
///
/// ## Example
/// ```
/// # use minikalman::*;
/// const NUM_STATES: usize = 3;
/// const NUM_OBSERVATIONS: usize = 2;
/// assert_eq!(size_buffer_cross_covariance!(NUM_STATES, NUM_OBSERVATIONS), 6);
/// ```
#[macro_export]
#[allow(non_snake_case)]
macro_rules! size_buffer_cross_covariance {
    ( $num_states:expr, $num_observations:expr ) => {{
        const NUM_STATES_: usize = ($num_states) as usize;
        const NUM_OBS_: usize = ($num_observations) as usize;
        (NUM_STATES_ * NUM_OBS_) as usize
    }};
}
