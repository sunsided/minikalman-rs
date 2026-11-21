# Changelog

All notable changes to this project will be documented in this file.
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added a per-call correction scratch, so several observations share their temporaries soundly.
  `CorrectionScratch<STATES, MAX_OBSERVATIONS, T>` (in `regular`, `extended` and the prelude, a
  `const fn new`) holds `S⁻¹`, `H×P`, `P×Hᵀ` and `K×H×P` for any observation with up to
  `MAX_OBSERVATIONS` measurements; a larger one fails to compile. Lean observations hold only `H`,
  `z`, `R`, `y`, `S` and `K` (`RegularObservationBuilder::new_lean`,
  `ExtendedObservationBuilder::new_lean`, or `regular::LeanObservationStorage` /
  `extended::LeanObservationStorage`) and correct with `RegularKalman::correct_in` /
  `correct_with_in` and `ExtendedKalman::correct_nonlinear_in` / `correct_nonlinear_with_in` (also
  on the observations, and through the traits `KalmanFilterObservationCorrectIn` and
  `KalmanFilterNonlinearObservationCorrectIn`). Each takes the scratch as `&mut`, so the borrow
  checker allows one correction at a time; the results are bit-identical to the owning
  observations. For the radar EKF (4 states, 3 measurements, `f32`) each further observation
  stores 48 instead of 97 values. Owning observations and every existing method are unchanged.
  The inherited "MAY be aliased" buffer docs now say that each temporary is a buffer of its own
  and that sharing goes through the scratch ([#123]).
- With the opt-in `nalgebra` feature, every matrix and buffer gets `to_nalgebra()` (a stack copy
  into `SMatrix`, trait `ToNalgebra`), and mutable ones also get `copy_from_nalgebra()` (trait
  `CopyFromNalgebra`). They do not need the `unsafe` feature, and nalgebra stays out of the
  dependency tree unless the feature is enabled. nalgebra is re-exported as `minikalman::nalgebra`.
  Zero-copy views are not public yet, because nalgebra's products, `gemv` and `axpy` access memory
  out of bounds on operands with a row stride other than 1
  ([dimforge/nalgebra#1616](https://github.com/dimforge/nalgebra/issues/1616)); see [#138] for
  re-exposing them ([#135]).
- Added the UD filter `ud::UdKalman` (built with `KalmanFilterBuilder::build_ud`), which keeps the
  estimate covariance factored as `P = U·D·Uᵀ` and predicts with Thornton's MWGS and corrects with
  Bierman's sequential update. `P` stays positive semi-definite by construction, and no step takes a
  square root, which suits fixed-point targets: in an `I16F16` test where the plain filter's `P` turns
  indefinite, the UD filter's `P` stays positive semi-definite over 10⁴ steps. Correlated `R`, rank-deficient `Q`, the gate
  and the NIS are supported. New buffer macros `impl_buffer_ud_W` and `impl_buffer_ud_D` ([#111]).
- Added an outlier-robust correction that models the measurement noise as Student-t
  (variational Bayes, Piché, Särkkä and Hartikainen, IEEE MLSP 2012): `correct_robust` /
  `correct_robust_with` on the regular filter, `correct_nonlinear_robust` /
  `correct_nonlinear_robust_with` on the extended filter (any number of measurements) and
  `correct_sigma_point_robust` / `correct_sigma_point_robust_with` on the unscented filter, with
  `RobustCorrection` and `RobustWeight::StudentT { dof }`. It is a Kalman update with `R/w`, so `P`
  never grows past the prior; the weight `w` is iterated on `OBSERVATIONS × OBSERVATIONS` matrices
  with one caller-provided scratch matrix, reported by `robust_weight()`, and the status is
  `NotConverged` if it does not settle within the budget. The gate and NIS use the prior
  innovation, `R` is not adapted, and the regular and extended corrections update `P` as
  `set_covariance_update` says (the Joseph form by default) ([#108]).
- Added the `state!` macro (`macro_rules!`, no proc-macro) for named state and observation vectors:
  `state! { pub struct Radar(f32) { x, y, vx, vy } }` declares a `Copy` struct with the entry count
  `Radar::N`, index constants such as `Radar::vx`, `read`/`write` for vectors and
  `read_column`/`write_column` for the unscented filter's sigma point columns ([#121]).
- Added storage types that hold all buffers of a filter part as const-generic arrays, next to the
  `impl_buffer_*` macros and the `alloc` builders: `regular::{FilterStorage, ObservationStorage,
  ControlStorage}`, `extended::{FilterStorage, ObservationStorage}`,
  `unscented::{FilterStorage, ObservationStorage}`, `information::FilterStorage` and
  `smoother::RtsSmootherStorage`. `new` is a `const fn`, so storage can live in a `static`;
  `build()` wires the existing filter types to borrow it without allocating or moving arrays.
  Each storage has the size of the buffers it replaces. The unscented storage checks
  `NUM_SIGMA == 2 × STATES + 1` at compile time and is the first way to build a UKF without
  allocation or macros ([#120]).
- Added a `compass` example: a heading filter whose heading crosses north, corrected with
  `correct_with` and `angles::wrap_180`, next to an unwrapped filter that swings the long way
  around. See [#40].
- Added a `battery-soc` example: an Extended Kalman Filter that estimates a battery cell's state of
  charge from its terminal voltage with a first-order RC equivalent-circuit model and a nonlinear
  open-circuit voltage curve. It runs in CI and fails if the estimate does not converge ([#68]).
- Added an `se2-iekf` example and an "error-state and invariant EKF" section in the `extended` docs:
  `ExtendedKalman` estimates a body-frame error on SE(2) while the caller keeps the pose on the group,
  next to a plain EKF started with the same 170° heading error. It runs in CI and fails if the
  invariant filter does not converge ([#112]).
- Added `models::gauss_markov::first_order` to fill the state transition and process noise entries of a
  first-order Gauss-Markov state, e.g. for sensor bias drift or colored noise, from its time constant and
  steady-state variance. Fixed-point types use `models::gauss_markov::from_coefficient` with a precomputed
  `exp(−∆t/τ)` ([#63]).
- Added the normalized innovation squared `NIS = yᵀ×S⁻¹×y` of the last correction as `nis()` on
  the regular, extended and unscented observations, to check filter consistency ([#59]).
- Added `set_gate` to the regular, extended and unscented observations. A correction whose NIS
  exceeds the gate, e.g. a chi-square quantile, is skipped and leaves `x` and `P` unchanged.
  The gate is off by default ([#59]).
- Added `CorrectionStatus` (`Applied`, `NotPositiveDefinite`, `Gated`; `#[non_exhaustive]`), returned by `correct`,
  `correct_with`, `correct_nonlinear`, `correct_nonlinear_with`, `correct_sigma_point` and
  `correct_sigma_point_with`, so callers can tell whether a measurement was skipped ([#65]).
- Documented how to choose the Unscented Kalman Filter parameters `α`, `β` and `κ`, including the
  cubature Kalman filter settings (`α = 1, β = 0, κ = 0`) and the default `α = 1, β = 2, κ = 1`
  ([#64]).
- Added `information::InformationFilter`, a linear filter in information form (`Y = P⁻¹`, `ŷ = P⁻¹×x`)
  where independent sensors fuse by addition and in any order, starting from zero information if
  needed ([#67]). `fuse` adds a `RegularObservation`; `predict` converts to `P` and back, costing two
  extra `STATES` × `STATES` Cholesky inversions. Steps on matrices that are not positive definite return
  `false` and leave the filter unchanged. `KalmanFilterBuilder::build_information` creates one with heap
  allocated buffers.
- Added `smoother::RtsSmoother`, a Rauch-Tung-Striebel backward pass that refines a recorded filter
  history of `x`, `P` and their predictions in place, using the measurements that came later ([#62]).
  It does not allocate and skips steps whose predicted covariance is not positive definite.
- Added `smoother::FixedLagSmoother`, an online RTS smoother over a sliding window of the last `N`
  steps in const-generic ring buffers. Each `push` of a filter step returns the step `N−1` steps
  back, smoothed with the steps after it; it does not allocate ([#88]).
- Added `correct_with_gain` / `correct_with_gain_with` (regular) and `correct_nonlinear_with_gain` /
  `correct_nonlinear_with_gain_with` (extended) to apply an externally supplied Kalman gain, e.g. a
  precomputed steady-state gain, set via the new `kalman_gain_mut` accessor. They skip the inversion
  of `S` and update `P` with the Joseph form, so it stays symmetric and positive semi-definite for
  any gain. The gate is not evaluated and the NIS is not updated ([#66]).
- Added opt-in online adaptation of the measurement noise covariance `R` to the regular and extended
  observations with `set_noise_adaptation(Some(NoiseAdaptation { forgetting, floor }))`. After each
  applied correction, `R ← α×R + (1−α)×(ε×εᵀ + H×P⁺×Hᵀ)` from the residual `ε = z − H×x⁺`, with a
  floor on the diagonal. It needs no history window and no extra buffers. Gated and skipped
  corrections leave `R` unchanged. `Q` and the unscented observation are not adapted ([#61]).
- Added `set_noise_adaptation` to the unscented observation. After each applied sigma point
  correction, including the recalibrated one, `R` is adapted as for the regular observation, with the
  predicted measurement covariance in place of `H×P×Hᵀ`, written as
  `R ← R + (1−α)×(ε×εᵀ − R×S⁻¹×R)` with `ε = R×S⁻¹×y` ([#89]).
- Added opt-in online adaptation of the process noise covariance `Q` to the regular, extended and
  unscented filters with `set_process_noise_adaptation(Some(NoiseAdaptation { forgetting, floor }))`.
  After each applied `correct`, `correct_with`, `correct_nonlinear`, `correct_nonlinear_with` or
  sigma point correction, `Q ← α×Q + (1−α)×Δx×Δxᵀ` from the state correction `Δx = K×y`
  (Mohamed and Schwarz), with a floor on the diagonal. `Q` stays symmetric positive semi-definite by
  construction. The estimate is exact for a random walk and biased otherwise. It needs no extra
  buffers, and filters with an immutable `Q` are unaffected ([#89]).
- Added `correct_predictively_oriented` to `RegularKalman` and to scalar `RegularObservation`s
  (one measurement), with the trait `KalmanFilterObservationCorrectPredictivelyOriented`. It
  applies the predictively oriented (PrO) update, which inflates the covariance when the innovation
  is larger than predicted. This makes the filter robust to heavy-tailed noise and outliers, at a
  small accuracy cost under Gaussian noise. It costs a few scalar Newton steps on top of a regular
  correction and needs no extra buffers ([#69]).
- Added `correct_predictively_oriented_vector` to `RegularKalman` and `RegularObservation`, with
  the trait `KalmanFilterObservationCorrectPredictivelyOrientedVector`: the PrO update for
  observations with any number of measurements. It solves for the `m×m` measurement-space weight
  with a damped fixed-point iteration (positive-definiteness guard, Armijo backtracking, and step
  doubling for badly scaled priors), not by applying the scalar update row by row, which is not
  the joint update. Single measurements still use the scalar Newton solver. The iteration needs
  three caller-provided `m×m` scratch matrices, bundled in the new
  `PredictivelyOrientedWorkspace`; the observation buffers are unchanged. A correction that runs
  out of its iteration budget, or whose line search stalls at the working precision, is still
  applied and returns the new `CorrectionStatus::NotConverged`, for which `is_applied()` is
  `true` ([#87]).
- Added covariance recalibration after Jiang, Shi and Moura, "Mitigating Overconfidence in Nonlinear
  Kalman Filters via Covariance Recalibration" (Automatica, 2026, arXiv:2407.05717):
  `ExtendedKalman::correct_nonlinear_recalibrated` re-linearizes the observation at the updated
  state through a Jacobian callback, and `UnscentedKalman::correct_sigma_point_recalibrated`
  observes the sigma points again after shifting them by `K×y`. The covariance then comes from the
  second pass, which keeps the filters from becoming overconfident. If the trace of `P` would grow,
  the measurement is backed out and the new `CorrectionStatus::BackedOut` is returned. Both have
  `_with` variants, and the observation side is exposed as the traits
  `KalmanFilterNonlinearObservationCorrectRecalibrated` and
  `KalmanFilterUnscentedObservationCorrectRecalibrated` ([#60]).
- Added `innovation_vector()` to the regular and extended observations. It holds the innovation of
  the prior state after every correction, including gated or rejected ones, as the input for
  learned-gain filters ([#92]).
- The measurement noise adaptation now also runs in `correct_with_gain*`,
  `correct_nonlinear_with_gain*` and `correct_nonlinear_recalibrated*`. Their gain is not the
  optimal one for the covariance they produce, so the residual form would converge to a biased `R`;
  they use the innovation form `R ← R + (1−α)×(y×yᵀ − S)` instead and discard a candidate that is
  not positive definite. `correct_predictively_oriented` still leaves `R` unchanged, because its
  inflated covariance makes both forms inconsistent ([#86]).
- Added the predictively oriented (PrO) update for scalar extended observations:
  `ExtendedKalman::correct_nonlinear_predictively_oriented` uses the Jacobian as `H` and the
  nonlinear innovation `z − h(x)`, and `correct_nonlinear_predictively_oriented_iterated`
  relinearizes `h` at the PrO mean for a fixed number of rounds through a Jacobian callback, as
  in the iterated EKF. `P` is written once, so it stays positive definite. Both have `_with`
  variants, and the observation side is exposed as the trait
  `KalmanFilterNonlinearObservationCorrectPredictivelyOriented` ([#93]).
- Added `triangularize_upper` on `MatrixMut`, a Householder QR that leaves the upper triangular `R`
  with `Mᵀ×M = Rᵀ×R` in place without forming `Mᵀ×M`. It scales each reflector by its column's
  largest entry, so `I16F16` handles entries whose squares overflow, and skips zero columns instead of
  dividing by zero. Added the triangular solves `solve_lower_triangular_inplace`,
  `solve_lower_triangular_transposed_inplace`, `solve_upper_triangular_inplace` and
  `solve_upper_triangular_transposed_inplace` on `SquareMatrix`, which return `false` on a zero
  diagonal; the predictively oriented correction now uses them ([#109]).
- **Behavior change:** `correct`, `correct_with`, `correct_nonlinear` and `correct_nonlinear_with`
  now update `P` with the Joseph form by default, which keeps `P` exactly symmetric and needs no
  extra buffer; results change at the rounding level. `set_covariance_update(CovarianceUpdate::Standard)`
  on `RegularObservation` and `ExtendedObservation` restores the cheaper `P ← P − K×H×P`. A new `numerical_robustness` test bench measures how `P` degrades in `f32`, `f64`,
  `I16F16` and `I32F32`; in `f32` and `f64`, the Joseph form leaves `P` indefinite less often on
  ill-conditioned measurements but does not prevent it, and does not help once `I16F16` variances
  approach its resolution ([#107]).
- Added `square_root::SquareRootKalman`, a linear filter that propagates a square root `S` of the
  covariance, `P = S×Sᵀ`, so `P` stays positive semi-definite by construction. It predicts with a
  Householder triangularization of `[(A×S)ᵀ; Lqᵀ]` and corrects `RegularObservation`s with Potter's
  sequential update after decorrelating `R`. The gate and NIS match the regular filter. In an `f32`
  test with `R` twelve orders of magnitude below `P`, it keeps `P` positive definite where the
  regular filter's variance turns negative. Build it with `KalmanFilterBuilder::build_square_root`
  or from static buffers with `impl_buffer_temp_prearray!` for the `2N × N` prearray ([#110]).
- Added a conformance suite (`cargo test -p minikalman --test conformance`, `task conformance`).
  It checks the regular, extended and unscented filters, the information filter and the RTS
  smoother against reference vectors from an independent NumPy/SciPy implementation, against
  analytic identities (EKF = KF, information filter = KF, RTS = batch least squares, the
  `solve_discrete_are` steady state), against chi-square bounds on NEES and NIS over Monte Carlo
  runs, against invariants (`P` symmetric positive semidefinite, rejected corrections leave the
  estimate untouched, bit-identical reruns), and against regression baselines. Tolerances are
  stated per number type (`f64`, `f32`, `I32F32`, `I16F16`), and CI runs it on `std`, `libm` and
  `micromath` ([#131]).
- Added README examples, run as doctests, for the information filter, the UD filter, the
  Unscented Kalman Filter with cubature weights, robust corrections (NIS gate, predictively
  oriented correction, noise adaptation, `CovarianceUpdate::Standard`), angle wrapping with
  `correct_with` and `angles::wrap_pi`, and the RTS smoother, with a short table of contents
  under "Examples". A new first example sets up a filter from local `impl_buffer_*!` buffers
  without `std` or `alloc`, and also runs as a doctest without `alloc`. The README examples no
  longer use rustdoc's hidden `#` lines, which GitHub shows; the README doctests now require
  `alloc` instead, and a test rejects hidden lines in the README's Rust blocks.
- The `gravity` example now drops an object from 25 m, measures its height at 30 Hz with 3 cm
  noise, and estimates `g` from a poor guess. It prints a summary table with the 1σ uncertainty
  from `P`, fails if the true `g` leaves the 3σ band, and prints the README's chart with
  `-- --chart`. The measurements are seeded, so the output is reproducible.

### Fixed

- `nalgebra` together with `unsafe` now builds without `std` ([#135]).
- The stm32 cross-build example declared its fixed-point measurement vector with
  `impl_buffer_Q_control!` and took references to function-local `static mut` buffers in a safe
  function, which is unsound when re-entered (e.g. from an interrupt) and denied by the
  `static_mut_refs` lint in edition 2024. Both examples now keep their buffers in one
  const-initialized `static` storage handed out once through `static_cell::ConstStaticCell`, and
  name their states with `state!` ([#122]).
- The Cholesky decomposition now fails on a `NaN` or infinite pivot, e.g. from overflowing entries,
  instead of reporting success and producing a non-finite factor. Corrections then skip the update
  as for any other covariance that is not positive definite.
- A correction no longer turns the state into NaN when the innovation covariance `S = H×P×Hᵀ + R`
  is not positive definite, e.g. with `P = 0` and `R = 0`. The regular, extended and unscented
  filters now skip such an update and leave `x` and `P` unchanged.
- The Unscented Kalman Filter now draws its sigma points along the axes (`±√Pᵢᵢ`) when `P` is only
  positive semi-definite, e.g. when a variance is zero. Previously it read leftovers of a failed
  Cholesky decomposition.
- The Unscented Kalman Filter and its observations now panic on construction if `NUM_SIGMA` is not
  `2 × STATES + 1`, instead of producing a wrong covariance.
- Enabling `micromath` together with the default features (`libm`) compiles again. If `std` or `libm`
  is enabled, it takes precedence over `micromath`.
- With the `nalgebra` and `unsafe` features, a `MatrixDataArray` smaller than `ROWS × COLS` now panics
  when nalgebra accesses it, instead of reading past the array.

### Changed

- **Breaking:** the `nalgebra` feature now uses nalgebra 0.33 (was 0.32), the newest release that
  builds on the MSRV 1.80. The commented-out `nalgebra_std`/`nalgebra_alloc`/`nalgebra_libm` features
  are gone ([#135]).
- **Breaking** for custom implementations of the update and observation traits
  (`KalmanFilterUpdate`, `KalmanFilterNonlinearUpdate`, `KalmanFilterSigmaPointCorrect`,
  `KalmanFilterObservationCorrectFilter`, `KalmanFilterNonlinearObservationCorrectFilter` and
  `KalmanFilterUnscentedObservationCorrectFilter`): their `correct*` methods now return
  `CorrectionStatus` instead of `()`. Calling code is unaffected unless it passed the result on as
  `()`, e.g. as the last expression of a closure that must return `()` ([#65]).
- Raised the MSRV to `1.71.0` and the minimum `fixed` version to `1.24.0`. The `fixed` feature uses
  `sqrt`, which `fixed` only provides since `1.24.0`, and that release requires Rust `1.71`. CI now
  checks the library on the MSRV ([#91]).
- Raised the MSRV to `1.80.0`, so const-generic storage can flatten nested `[[T; C]; R]` arrays
  with the safe `as_flattened_mut` instead of `unsafe` code. The CI MSRV job now checks on `1.80`
  ([#119]).
- `correct_with_gain*` now forms only the lower triangle of `K×S×Kᵀ` and mirrors it, so `P` comes out
  exactly symmetric; results change at the level of rounding, and an asymmetric `P` passed in is
  symmetrized from its lower triangle ([#107]).

### Removed

- Removed `UnscentedObservation::correct_nonlinear`. It averaged the sigma points with equal weights,
  ignoring the UKF weights, and read observed sigma points that callers could not set. Use
  `correct_sigma_point` / `correct_sigma_point_with` on the filter instead.

[#138]: https://github.com/sunsided/minikalman-rs/issues/138
[#123]: https://github.com/sunsided/minikalman-rs/issues/123
[#135]: https://github.com/sunsided/minikalman-rs/issues/135
[#112]: https://github.com/sunsided/minikalman-rs/issues/112
[#111]: https://github.com/sunsided/minikalman-rs/issues/111
[#108]: https://github.com/sunsided/minikalman-rs/issues/108
[#107]: https://github.com/sunsided/minikalman-rs/issues/107
[#131]: https://github.com/sunsided/minikalman-rs/issues/131
[#91]: https://github.com/sunsided/minikalman-rs/issues/91
[#92]: https://github.com/sunsided/minikalman-rs/issues/92
[#86]: https://github.com/sunsided/minikalman-rs/issues/86
[#93]: https://github.com/sunsided/minikalman-rs/issues/93
[#87]: https://github.com/sunsided/minikalman-rs/issues/87
[#69]: https://github.com/sunsided/minikalman-rs/issues/69
[#68]: https://github.com/sunsided/minikalman-rs/issues/68
[#63]: https://github.com/sunsided/minikalman-rs/issues/63
[#59]: https://github.com/sunsided/minikalman-rs/issues/59
[#61]: https://github.com/sunsided/minikalman-rs/issues/61
[#60]: https://github.com/sunsided/minikalman-rs/issues/60
[#65]: https://github.com/sunsided/minikalman-rs/issues/65
[#64]: https://github.com/sunsided/minikalman-rs/issues/64
[#67]: https://github.com/sunsided/minikalman-rs/issues/67
[#62]: https://github.com/sunsided/minikalman-rs/issues/62
[#88]: https://github.com/sunsided/minikalman-rs/issues/88
[#66]: https://github.com/sunsided/minikalman-rs/issues/66
[#89]: https://github.com/sunsided/minikalman-rs/issues/89
[#109]: https://github.com/sunsided/minikalman-rs/issues/109
[#119]: https://github.com/sunsided/minikalman-rs/issues/119
[#110]: https://github.com/sunsided/minikalman-rs/issues/110
[#120]: https://github.com/sunsided/minikalman-rs/issues/120
[#121]: https://github.com/sunsided/minikalman-rs/issues/121
[#122]: https://github.com/sunsided/minikalman-rs/issues/122

## [0.8.1] - 2026-10-30

[0.8.1]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.8.1

### Fixed

- The README examples compile again: they used `system_covariance`, `state_transition_mut` and
  `observation_matrix_mut`, which are now `estimate_covariance`, `state_transition_jacobian_mut` and
  `observation_jacobian_matrix_mut`. The README is now doctested so they stay in sync.
- The hero image and the `radar_2d` example link in the README work on crates.io.

## [0.8.0] - 2026-10-30

[0.8.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.8.0

### Added

- Added `models::motion::constant_velocity` and `models::motion::constant_acceleration` to fill the state transition
  and process noise matrices for a given time step.
- Added `correct_with` (regular) and `correct_nonlinear_with` (extended) to normalize the innovation
  before it is applied, e.g. to wrap angles across ±π or 0°/360° ([#40]).
- Added `angles::wrap_pi` and `angles::wrap_180`, which also work for fixed-point types.
- Added `correct_sigma_point_with` and `set_state_residual` to the Unscented Kalman Filter to normalize
  measurement and state differences, e.g. for bearings or headings across ±π ([#40]).
- Added a WebAssembly example (`xbuild-tests/wasm`) that runs without `wasm-bindgen`, and an interactive
  [live demo](https://sunsided.github.io/minikalman-rs/) comparing the Extended and Unscented filters on radar
  tracking, with draggable radars, drawn paths and a toggle for bearing wrap-around.

### Fixed

- The Unscented Kalman Filter now redraws its sigma points from the current state and covariance when
  `correct_sigma_point` runs more than once between predictions, e.g. once per sensor. Previously each
  further correction reused the prior's sigma points and shrank the covariance again by the same amount,
  until it turned negative and the filter diverged. Correcting before the first prediction now uses
  sigma points drawn from the initial state instead of zeroed buffers.
- The Unscented Kalman Filter now averages sigma points relative to the centre sigma point. Results are
  unchanged for ordinary states, but sigma points on either side of a wrap point now average correctly once
  a residual function is set.
- `cargo test` and `cargo clippy --all-targets` work again with `--no-default-features --features libm`: tests,
  examples, benches and doctests that use the builders now require the `alloc` feature.

### Changed

- **Breaking** for custom implementations of `KalmanFilterUpdate`, `KalmanFilterNonlinearUpdate`,
  `KalmanFilterObservationCorrectFilter` and `KalmanFilterNonlinearObservationCorrectFilter`:
  implement `correct_with` / `correct_nonlinear_with` instead of `correct` / `correct_nonlinear`,
  which are now provided methods. Code that only calls these methods is unaffected.
- **Breaking** for custom implementations of `KalmanFilterSigmaPointCorrect`,
  `KalmanFilterUnscentedObservationCorrectFilter` and `KalmanFilterUnscentedParamsMut`: implement
  `correct_sigma_point_with`, `correct_with_observed_residuals` and `set_state_residual`.

### Internal

- Added actionlint and zizmor checks for the GitHub workflows; actions are pinned to commit SHAs and kept up to
  date by Dependabot.
- Added CI jobs that build for `wasm32-unknown-unknown` and deploy the demo to GitHub Pages, and a CI test run
  with `libm` but without `alloc`.
- Added a `cargo-release` configuration and a workflow that creates GitHub releases from version tags.

[#40]: https://github.com/sunsided/minikalman-rs/issues/40

## [0.7.0] - 2026-04-26

[0.7.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.7.0

### Added

- Added `copy_to` and `copy_from` to matrix trait.
- Added Unscented Kalman Filter (UKF) support, including builder types and covariance computation.
- Added integration tests for UKF covariance computation and type coverage.
- Added support for GitNexus code intelligence.
- Added dependency graph submission in CI/CD.

### Changed

- The `copy` matrix operation is now deprecated in favor of `copy_to`.

### Fixed

- Fixed pre-existing clippy warnings for doc indentation and lifetime syntax.
- Fixed clippy warnings in proptest files.
- Fixed typo in README.md regarding libm usage.

### Internal

- Addressed CI/CD review: fixed ineffective skips, dropped no-op, aligned CI runs, and renamed test.
- Used UFCS in data_type tests to exercise trait impls.
- Cleaned up comments in data.rs.
- Updated tea.yaml.

## [0.6.0] - 2024-06-22

[0.6.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.6.0

**Note:** This release contains breaking changes.

### Fixed

- [#32](https://github.com/sunsided/minikalman-rs/pull/32):
  Process noise is now separated into direct process noise, acting directly on the
  state transition, and control process noise, acting only on control inputs through the
  control matrix.

### Internal

- [#31](https://github.com/sunsided/minikalman-rs/pull/31):
  The Regular and Extended Kalman Filter types were split, renamed and moved into separate modules.
  `RegularKalman` and `ExtendedKalman` now only provide their respective functionalities.

## [0.5.0] - 2024-06-20

[0.5.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.5.0

**Note:** This release contains major breaking changes.

### Added

- [#10](https://github.com/sunsided/minikalman-rs/pull/10):
  Builder types were added for Kalman filters, control inputs and observations. The `KalmanFilterBuilder` type
  serves as a simple entry point on `alloc` crate features.
- [#21](https://github.com/sunsided/minikalman-rs/pull/21):
  Added the functions `make_identity`, `make_scalar`, `make_comatrix` and `set_diagonal_to_scalar` for square matrices.
- [#25](https://github.com/sunsided/minikalman-rs/pull/25):
  Added support for Extended Kalman Filters.
- [#28](https://github.com/sunsided/minikalman-rs/pull/28):
  `micromath` is now an optional dependency again.

### Changed

- [#9](https://github.com/sunsided/minikalman-rs/pull/9):
  Data ownership was reworked: Filters, control inputs and measurements/observations are now backed by a generic
  buffer type that can operate on stack- or heap allocated arrays, immutable and mutable references. Some features
  are gated behind the `alloc` crate feature.
- Types were remodeled into new modules in order to arrange them in a slightly more logical way.
- [#18](https://github.com/sunsided/minikalman-rs/pull/18):
  "Inputs" naming was changed to "Control" to align more closely with common usages of Kalman filters.
- [#20](https://github.com/sunsided/minikalman-rs/pull/20):
  "Measurements" naming was changed to "Observation" in places where it aligns with common usages of Kalman filters.
  In addition, type name ambiguity between process and measurement noise covariance was reduced.

### Removed

- The `create_buffer_X` macros were removed from the crate due to their relatively complicated use.

### Internal

- [#8](https://github.com/sunsided/minikalman-rs/pull/8):
  The repository was restructured into a Cargo workspace to allow for easier handling of cross-compilation examples.
- [#27](https://github.com/sunsided/minikalman-rs/pull/27):
  Added an EKF example with radar measurements of a moving object.

## [0.4.0] - 2024-06-07

[0.4.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.4.0

### Added

- [#7](https://github.com/sunsided/minikalman-rs/pull/7):
  Added the `libm` crate feature for [libm](https://github.com/rust-lang/libm) support.
- [#7](https://github.com/sunsided/minikalman-rs/pull/7):
  Added the `float` crate feature to enable `f32` and `f64` built-in support.

### Removed

- Removed the `no_std` crate feature in favor of the `std` feature (disabled by default).

### Internal

- [#7](https://github.com/sunsided/minikalman-rs/pull/7):
  Add an example project for STM32F303 cross-compilation.
- Added CI/CD spell-checks and pre-commit hooks.

## [0.3.0] - 2024-06-04

[0.3.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.3.0

### Added

- [#6](https://github.com/sunsided/minikalman-rs/pull/6):
  Added support for fixed-point values via the [fixed](https://crates.io/crates/fixed) crate.

### Changed

- [#5](https://github.com/sunsided/minikalman-rs/pull/5):
  The macros, matrix, filter and measurement structs are now generic on the data type.
  If no type is provided, it defaults to `f32`.

### Removed

- The dependency on `micromath` was removed due to the generic type implementations.

## [0.2.3] - 2024-06-03

[0.2.3]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.2.3

### Internal

- Improved documentation of `Kalman` and `Measurement` structs.

## [0.2.2] - 2024-06-03

[0.2.2]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.2.2

### Internal

- Added usage examples to macro documentation.

### Removed

- Removed duplicate `create_buffer_P_temp` and `create_buffer_BQ_temp` macros in favor of `create_buffer_temp_P`
  and `create_buffer_temp_BQ`.

## [0.2.1] - 2024-06-03

[0.2.1]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.2.1

### Internal

- Conditionally enable `docsrs` feature gate when building documentation on [docs.rs](https://docs.rs/crate/minikalman).

## [0.2.0] - 2024-06-03

[0.2.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.2.0

### Internal

- Remove requirement to specify either `std` or `no_std` crate feature. If `no_std` is not specified,
  `std` is now implied. This resolves builds on [docs.rs](https://docs.rs/crate/minikalman) and quirks when
  using the crate.
- Some documentation hyperlinks were now corrected.

## [0.1.0] - 2024-06-03

[0.1.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.1.0

### Internal

- Set MSRV to `1.70.0` and Rust Edition to `2021`.
- Added CI/CD cross-platform builds, code coverage and ensure examples and benchmarks build correctly.
- Remove dependency on [stdint](https://github.com/sunsided/stdint-rs) crate unless explicitly enabled with the
  `stdint` crate feature. This should unblock builds on Windows.

## [0.0.2] - 2023-07-11

[0.0.2]: https://github.com/sunsided/minikalman-rs/releases/tag/0.0.2

### Added

- Initial release as a port from the [kalman-clib](https://github.com/sunsided/kalman-clib/) library.
