# Changelog

All notable changes to this project will be documented in this file.
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added directional forgetting to `rls::RecursiveLeastSquares`. Select it with
  `storage.build(p0).with_forgetting::<rls::Directional>()`. The mode is a type parameter that
  defaults to `rls::Exponential`, so a build only contains the code of the mode it uses. It forgets
  only along the current regressor, so the parameters that are still excited keep tracking while
  another regressor is idle. Variances in unexcited directions stay bounded without the windup
  guard, and an estimate no longer jumps when an idle regressor is excited again. Both covariance
  forms support it. In the factored form, an Agee–Turner rank-one update keeps `D ≥ 0`. The
  default stays exponential forgetting ([#209]).

### Fixed

- Fixed an `I16F16` overflow in `rls::RecursiveLeastSquares` (factored form, both forgetting
  modes). After a precise sample, `φᵀ·P·φ` squared the large entries of `Uᵀ·φ` before scaling them
  by the small `D`, which panicked in debug builds and wrapped in release builds. In `f32`/`f64`,
  the last bit of `φᵀ·P·φ` can now differ.

### Internal

- Added property tests for recursive least squares (`tests/proptest_rls.rs`) and for the UD
  rank-one update. Among them: both covariance forms agree under directional forgetting, the
  information along an idle direction is kept exactly, rejected and empty updates change nothing,
  and short `I16F16` runs of the factored form neither overflow nor make `D` negative.
- Added Kani proofs that `rank_one_update` and one directional RLS update (both forms) in `I16F16`
  cannot overflow, divide by zero or index out of bounds within stated input ranges, and a `Kani`
  workflow that runs them.

## [0.9.0] - 2026-11-22

[0.9.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.9.0

### Documentation

- Documented the `λ` of `predict_tuned` / `predict_tuned_nonlinear` as a fading-memory (forgetting)
  factor: what it does, why it works (exponential data weighting), its exact relation to recursive
  least squares (`μ = λ²`), when to prefer it over `Q`, and its windup caveat.

### Added

- Added two navigation workloads to the Renode benchmarks in `benches/mcu`: a 6-state attitude
  error-state EKF (gyro prediction at a varying `Δt`, 3-measurement accelerometer update,
  scalar heading) and a 12-state GNSS/INS error-state EKF (IMU prediction, 5-measurement GNSS
  update, sequential barometer and heading updates), in `f32`, with and without the `unsafe`
  feature. The quaternion and Jacobian glue, the filter operations and the error reset are
  measured separately; the run also reports each workload's stack use and the `size_of` of its
  storage, and the README converts the counts into a rough time budget at typical clocks. The
  model lives in `benches/mcu/src/nav.rs` for the GNSS/INS example to share ([#185]).
- Added `RegularKalman::predict_companion` and `predict_companion_tuned`: the time update for a
  companion (shift) transition, such as Harvey's state-space form of an ARMA model, given only
  the transition's first column. It updates `x` and `P` in place in `O(n²)` instead of `O(n³)`,
  with one row of `P` as stack scratch, and matches the dense `predict` / `predict_tuned` bit
  for bit unless an intermediate overflows (an exact zero in `x` may differ in sign). The dense path is
  unchanged ([#195]).
- Added the `gnss-ins` example: a 12-state GNSS/INS error-state EKF `(δp, δv, δθ, δb_g)` in
  `f32` on 5 904 bytes of storage, without allocation. IMU prediction at a varying `Δt`, a
  gated 5-measurement GNSS update (horizontal position, 3-D velocity) and barometer and
  heading as sequential scalar updates, each followed by the reset into the caller's nominal
  state. A simulated 120 s flight with turns, climbs, accelerations, GNSS outliers and a 10 s
  dropout checks the errors, NEES, NIS and gate decisions against bounds from a 300-seed
  sweep. The module docs cover the time budget on Cortex-M (from the Renode workload of
  [#185]), the `sinf`/`cosf` cost on the M4F, and when `f64` or the UD / square-root filters
  are warranted. It shares its model with `benches/mcu/src/nav.rs` ([#186]).
- Added the `attitude-eskf` example: a quaternion error-state (multiplicative) EKF for the
  attitude of an IMU in `f32` on inline storage, with the error `(δθ, δb)` of attitude and gyro
  bias, gyro prediction at a varying `Δt` from timestamps, an accelerometer tilt update gated by
  its NIS (with a hold-off after a rejection) and an external heading. It compares the gate with
  a ±10 % `|a|` window on a simulated rotation sequence with known truth and checks the attitude
  error, NEES and NIS. The "Filtering on manifolds" recipe and the README link to it ([#184]).
- Added `log_det_innovation_covariance()` and `log_likelihood()` to the regular and extended
  observations: `ln det S` and the Gaussian log-likelihood
  `ℓ = −½·(m·ln 2π + ln det S + yᵀ·S⁻¹·y)` of the last correction, for maximum-likelihood
  fitting (e.g. ARMA models via the prediction-error decomposition). Both are computed on
  request from the `S` the observation keeps, factored as the gain does, so corrections cost
  exactly what they did. Float-only (`f32`/`f64` with a float backend), through the new
  `Logarithm` trait. The crate docs gained a "Likelihood of a Model" section ([#194]).
- Added the `augmented-state` example: joint (augmented-state) estimation of a motor plant's
  `(a, b)` with an EKF and a UKF on `[x, a, b]`, on the same run as the dual `rls-adaptive`
  example, with a comparison table, a sweep of the parameter process noise and
  allocation-free storage. The crate docs gained an "Estimating Model Parameters" section that
  compares joint, dual and RLS-only estimation and their limits ([#187]).
- Added the `filter!` macro, a declaration language over the storage types. One block
  declares `state`s, `observation`s (full by default, `observation lean` with a `scratch` for
  `correct_in`), `control`s, `kalman` filters, `luenberger` observers (current form, or
  `luenberger predictor`) and `system`s of them. Each item becomes a module with the library's
  storage type, a `const fn new()`, a `build()` that only wires (like the storage types'
  `build()`), and a `model()` that writes the declared `A`, `Q`, `P`, `x₀`, `H`, `R`, `B`, `Qᵤ`
  and gains, and doubles as a reset. Matrix entries are addressed by name (`A[pos, vel] = DT;`,
  `H = [pos];`, `A = |a| constant_velocity(DT, 1.0).fill_transition(a);`); items refer to each other by Rust path,
  so one observation and one control serve several filters across blocks, modules and crates.
  A `system` and a `luenberger` storage are checked at compile time to be exactly the sum of
  their parts; nothing is allocated (except with `storage = boxed;`, [#159]), no `static` is
  created and no `unsafe` is generated.
  `macro_rules!` only, no new dependency, no feature, builds on the MSRV. The new `filter-dsl`
  example runs two filters sharing an observation and a control ([#157]).
- Added placement modes to `filter!` items ([#159]).
  - `kalman`, `observation` and `control` blocks take leading placement statements.
  - `storage = owned;` generates `Owned` plus `new_owned()`: buffers held by value, no
    lifetime, usable as a struct field.
  - `storage = boxed;` (needs `alloc`) generates `Boxed` plus `new_boxed()`: each matrix
    allocated directly on the heap.
  - `M = extern;` leaves a matrix out of `Storage`; `build` then borrows it as a
    `&mut [Scalar; N]` whose length the type checks, for DMA regions or shared arenas.
  - `model()` works in every mode and writes declared values into extern buffers.
  - Every item, scratch, observer and system exports `BYTES` and `EXTERN_BYTES` for its
    footprint.
  - Unsupported combinations are compile errors:
    - `extern` with `owned` or `boxed`;
    - `lean` with `owned` or `boxed`;
    - placement statements after the model statements or in a `luenberger` block;
    - owned or extern items as `system` members.
- Added NIS gating to the fixed-gain (Luenberger) observer. `FixedGainObservation` takes an
  optional gate as a defaulted last type parameter: `NoGate` (the default, no extra memory) or a
  `FixedGainGate` holding a stored steady-state `S∞⁻¹` (`OBSERVATIONS²` scalars), added with
  `with_gate`, `KalmanFilterObservationBuilder::build_fixed_gain_gated` or
  `GatedFixedGainObservationStorage`. `set_gate(Some(threshold))` uses the threshold convention of
  `RegularObservation::set_gate`; a correction whose `NIS = yᵀ·S∞⁻¹·y` exceeds it returns
  `CorrectionStatus::Gated` and leaves `x̂` unchanged (in the predictor form it leaves no pending
  correction, so the next `predict` is a pure model step), for `m² + m` multiply-adds. Fill
  `S∞⁻¹` with `set_innovation_covariance` (a Cholesky inverse of the `S∞` that `settle_gain`
  leaves in the design observation, reachable through the new
  `RegularObservation::innovation_covariance_mut`) or `set_innovation_covariance_inverse`. In
  steady state the NIS equals the Kalman filter's to `3e-11`; the docs explain the start-up
  over-rejection and the lock-out of a fixed `S∞`, and how to avoid them. The trait
  `KalmanFilterObservationCorrectFixedGain` gains a defaulted `is_gated` method ([#144]).
- Added the time-step handle `regular::TimeUpdate`: `RegularKalman::predict` and `predict_tuned`
  now return it, and controls go through it, one `control` per distinct `Control`:
  `filter.predict().control(&mut throttle).control(&mut steering)`. The handle borrows the filter,
  so a control can no longer follow a correction, and no handle exists without a prediction.
  Results are bit-identical to `predict()` followed by `control()` per input. Plain
  `filter.predict();` still compiles, the `KalmanFilterPredict` and `KalmanFilterApplyControl`
  traits are unchanged, and the standalone `filter.control(&mut c)` stays, documented as "after
  `predict`, once per input". Applying the same control twice in one step is still not prevented.
  Only the regular filter has control inputs; the extended, unscented, information, UD and
  square-root filters have none and return no handle. Calling code breaks only where it used the
  `()` of the inherent `predict` / `predict_tuned`, e.g. as the tail expression of a function or
  closure that returns `()`; add a `;` ([#153]).
- Added the opt-in `derive` feature for zero-copy named vectors (needs Rust 1.88; the crate MSRV
  stays 1.85 without it). It re-exports the `UniformArray` derive from `uniform-array-derive` and
  `ensure_uniform_type` from `ensure-uniform-type`, and adds `NamedVector` with the
  `named_vector!` macro and the `VectorView` / `VectorViewMut` extension traits (in the prelude):
  `filter.state_vector().view::<Radar>()` borrows a state, measurement or control vector as a
  `#[repr(C)]` struct, and `view_mut` writes through its fields. A struct of another length fails
  to build. The cast comes from the derive in the user's crate, gated on that crate's `unsafe`
  feature; minikalman adds no `unsafe` code. `state!` is unchanged and stays the default ([#124]).
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
- Added the fixed-gain (Luenberger) observer `regular::LuenbergerObserver` (doc alias
  `steady_state_kalman`) with `regular::FixedGainObservation`: the steady-state Kalman filter
  without `P`, `Q` or `R`, at `n² + 2·n·m` multiply-adds per step with no square root, division or
  data-dependent branch. The gain convention is a type parameter (`GainForm`, sealed, with the
  zero-sized `Current` and `Predictor`): `FixedGainObservation<…, Current>` holds the crate's
  predict-then-correct gain `K`, `FixedGainObservation<…, Predictor>` the textbook gain `L` of
  `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)` that `place` and python-control `dlqe` return, and an
  observer of one form does not compile with an observation of the other. The time and the
  measurement update stay separate calls in both forms: `predict()` returns the same
  `regular::TimeUpdate` handle as `RegularKalman::predict`, through which any number of distinct
  controls are applied (`observer.predict().control(&mut throttle).control(&mut steering)`), only
  before the corrections of the step, and `correct` updates. In the predictor form (`new_predictor`,
  `KalmanFilterBuilder::build_predictor_observer`, `PredictorObserverStorage`), `correct` defers
  `L·y` into a pending correction (one extra `n`-vector) without touching `x̂`, and the next
  `predict` applies `x̂ ← A·x̂ + Σ L·y`, bit-for-bit the fused recursion.
  `RegularKalman::settle_gain` iterates the filter's own covariance recursion until the
  observation's gain settles and reproduces SciPy's DARE (`K∞` to `2e-12` in `f64`); it
  overwrites the filter's `P` and the observation's `S` and `K` and leaves `x`, the gate and the
  noise adaptation alone. Also new: `Control::apply_control_to_state` (trait
  `KalmanFilterControlApplyToState`), `RegularObservation::innovation_covariance`, the storage
  types `regular::{ObserverStorage, PredictorObserverStorage, FixedGainObservationStorage}`, the
  builders `KalmanFilterBuilder::build_observer`, `build_predictor_observer` and
  `build_fixed_gain::<M, F>`, and the `luenberger` example.
  NIS gating for the fixed-gain observation is tracked in [#144] ([#143]).
- Added recursive least squares, `rls::RecursiveLeastSquares` (storage: `rls::EstimatorStorage`),
  an online estimator of model parameters `θ` in `y = φᵀ·θ + v`: the Kalman filter with `A = I`,
  `Q = 0` and `H = φᵀ`, `O(N²)` per sample and no matrix inversion. The forgetting factor `μ`
  matches `RegularKalman::predict_tuned(λ)` with `μ = λ²` while the windup guard is inactive. By
  default `P` is kept as `U·D·Uᵀ` and updated with Bierman's update from the UD filter, so it stays
  positive semi-definite in `f32` and `I16F16`; a plain covariance form is available. A covariance
  windup guard is on by default: it bounds `tr(P)` by slowing, and at the bound stopping,
  forgetting (variable-rate forgetting with `μ ≤ μₖ ≤ 1`). Gating and the
  NIS behave as on observations. The `rls-adaptive` example feeds the estimate into a Kalman
  filter's `A` and `B` each step; after a mid-run change of the plant, its speed RMSE is 0.09
  against 0.70 for the fixed-model filter ([#145]).
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

- The crate is now dual-licensed as `MIT OR Apache-2.0` (previously MIT only), and the package
  ships both license texts.

- `Matrix::mult` with a one- or two-column result starts its dot products from `−0.0` instead of
  `+0.0` for `f32` and `f64`, so the first addition folds away (`−0.0 + x = x` exactly, while
  `+0.0 + x` must be computed in case `x` is `−0.0`). The start value comes from the new
  `MatrixDataType::additive_identity` (default `zero()`, so fixed point is unchanged). The
  2-state, 1-measurement filter step runs 3.6 % fewer instructions on a Cortex-M4F at
  `opt-level = 3`, 5.4 % fewer on a Cortex-M0+, and 3.7 % fewer on x86-64; no filter
  operation of `benches/mcu` runs more instructions for any type, core or profile, and the
  `thumbv7em-none-eabihf` code size does not grow. The other dot-product kernels keep `+0.0`:
  each regressed some Cortex-M row. A result differs from before only when every term of an
  entry is `−0.0`: it is now `−0.0` instead of `+0.0` (equal under `==`), as the mat-vec
  kernels already returned. This is accepted rather than corrected, since a correction would
  cost the saved addition again; the tests pin every other result bit for bit. Empty sums stay
  `+0.0` ([#181]).
- The regular and extended corrections view their own buffers at the exact static length once,
  at the start of the update, so the inlined kernels behind need no length checks of their own.
  Bit-identical results, no new buffers, no API change. Fewer instructions per correction for
  most sizes and types (Cortex-M4F `opt-level = "s"`, 2×1 `f32`: 506 → 492 for the Joseph
  correction) and a smaller step at `"s"` (`thumbv7em`, 4×2 `f32`: 2948 → 2876 B Joseph), but
  +0.1–0.4 % at 6×3 and for the `I16F16` 4×2 Joseph correction ([#180]).
- The single-measurement gain of the floating-point types no longer carries the Cholesky
  fallback of the `L×D×Lᵀ` factorization: for one measurement both reject exactly the same `S`,
  so the fallback could never run. This takes the fallback and its square root out of the code:
  the `thumbv7em` 2×1 `f32` step at `opt-level = "s"` shrinks by about 330 B. Steps with two or
  three measurements keep the fallback, which decides for subnormal or near-singular `S`; it
  keeps the 4×2 `f32` steps at `"s"` above their size before the optimization series:
  Joseph +332 B (2876 vs 2544), standard +300 B (2624 vs 2324), extended +348 B
  (3514 vs 3166) ([#179]).
- With the `unsafe` feature, `libm` and no `std`, the `f32` square root on Arm hard-float targets
  (`target_abi = "eabihf"`, e.g. the Cortex-M4F) uses the FPU's `vsqrt.f32` instead of
  `libm::sqrtf` in software: 7 instead of 79 instructions per root on the M4F, up to 26 % fewer
  in the square-root filter's correction, with bit-identical results. Opt-in: without `unsafe`
  nothing changes, and `micromath` keeps its approximation ([#174]).
- Made the regular and extended corrections and the dense kernels cheaper, with bit-identical
  results ([#163]):
  - `correct` reuses the `H×P` that the gain needed instead of forming it again, and the Joseph
    form keeps a copy of `S = H×P×Hᵀ + R` (in the `K×H×P` temporary, when `m ≤ N`) instead of
    forming it again after the Cholesky factorization.
  - The kernels (`mult`, `mult_transb`, `multadd_transb`, `multscale_transb`, the row-vector
    products, the element-wise sums, Cholesky and its inverse) and the Joseph helpers take their
    slices at the static length once per call, which removes the per-element bounds checks.
    `buffer_len()` and `is_valid()` are unchanged; a buffer shorter than the matrix now panics
    before the first write instead of part-way.
  - `mult` and the `mult_transb` family accumulate a whole output row at once (i-k-j order),
    which LLVM vectorizes; every entry still sums its terms in the same order.
  - A 4×2 `f32` Joseph step on `thumbv7em-none-eabihf` (`opt-level = "s"`) shrinks from 2558 to
    2202 bytes of `.text`.
- Symmetric covariance products compute only their lower triangle and mirror it ([#167]):
  `A×P×Aᵀ` in the prediction of the regular, extended and information filters (also with the
  fading-memory `λ`), `B×Qᵤ×Bᵀ` of controls, the `K×S×Kᵀ` and `K×H×P + (K×H×P)ᵀ` terms of the
  Joseph update, `Hᵀ×R⁻¹×H` of the information filter, the RTS smoother's gain sandwich, the
  square-root filter's `S×Sᵀ`, the scalar PrO update and the unscented `Pxz×Kᵀ` / `K×S×Kᵀ`.
  Each product's lower triangle is the same bit for bit; the upper one is now an exact copy,
  so `P` comes out exactly symmetric from `predict` (for a symmetric `Q`) and from every Joseph
  `correct`, also in fixed point. Over a run, results change at the level of rounding, since
  later steps read the mirrored entries. A 6×3 `f64` prediction is about 16% faster; a 4×2
  `f32` Joseph step on `thumbv7em-none-eabihf` grows from 2198 to 2218 bytes of `.text`
  (`opt-level = "s"`). New kernels `mult_symmetric_transb`, `multadd_symmetric_transb` and
  `multscale_symmetric_transb`.
- The regular and extended corrections with the Kalman gain (`correct`, `correct_with`,
  `correct_nonlinear` and `correct_nonlinear_with`, including the first pass of the
  recalibrated ones) solve the Kalman gain from the Cholesky factor of `S` instead
  of forming `S⁻¹` ([#165]). Each row of `K = P×Hᵀ×S⁻¹` takes one forward and one back
  substitution, and the NIS is `‖L⁻¹×y‖²` from one forward substitution. The substitutions
  multiply with one reciprocal per pivot, so a correction needs fewer divisions than the
  explicit inverse (`2m` instead of `3m`). Results change at the
  level of rounding and are more accurate where `S` is ill-conditioned: in the
  `numerical_robustness` sweep, the corrected `f32` state stays within `10⁻⁴` of the `f64` one
  instead of `0.4`, and `I16F16` no longer overflows. Gating and `NotPositiveDefinite`
  (nothing written) are unchanged. `S⁻¹` is never formed: the gain factors a copy of `S` in the
  `temp_S_inv` temporary (an `m×m` copy per correction), so the observation's `S` now holds
  `H×P⁻×Hᵀ + R` after every regular, extended or robust correction, also in the standard form
  (where it used to hold its Cholesky factor) and when gated or rejected. The Joseph form
  therefore no longer copies `S` into the `K×H×P` temporary or forms it again. The robust
  corrections take their prior NIS from the Cholesky factor the same way, so it agrees with the
  plain correction's (bit for bit where `G + R` rounds like `H×P×Hᵀ + R`, as in the tests).
- For floating-point types and up to three measurements, the regular and extended corrections
  factor `S` square-root free as `L×D×Lᵀ` in local storage instead of with a Cholesky
  factorization ([#166]). For one measurement this is `K = P×Hᵀ/s` and `NIS = y²/s`. The choice
  is made at compile time on the element type (the new `MatrixDataType::SQUARE_ROOT_FREE_GAIN`,
  `true` for `f32` and `f64`) and on `OBSERVATIONS`. Fixed-point types and more measurements
  keep the Cholesky solve: `L×D×Lᵀ` divides by the squared pivots, which overflows fixed point
  near its resolution. A pivot that is not positive or not finite is rejected as
  `NotPositiveDefinite`, as before; near-singular `S` can flip between accepted and rejected at
  the level of rounding. `D⁻¹` is applied with one reciprocal per pivot, except for a single
  measurement, which divides by `s` to keep `K` correctly rounded. Where the square-root free
  factorization rejects a pivot (also a subnormal one, whose reciprocal overflows), the
  Cholesky factorization of a copy decides, so a rejection that Cholesky would accept is no
  longer reported. The factor's local storage is fixed at fifteen values for any
  `OBSERVATIONS`. A correction whose NIS is `NaN`, which every gate lets through, now returns
  `NotPositiveDefinite` without applying. The robust corrections take their prior NIS from the
  same factorization, falling back to the Cholesky factor they already have. A 4×2
  `f32` Joseph step on `thumbv7em-none-eabihf` (`opt-level = "s"`) shrinks from 2218 to
  1958 bytes of `.text` and links no `sqrt`.
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
- **Breaking:** raised the MSRV to `1.85.0` and moved the workspace to Rust edition 2024 with
  Cargo resolver 3, which picks dependency versions that build on the MSRV. Code that uses the
  `impl_buffer_*!(static mut …)` forms must borrow the buffers through `&raw const` / `&raw mut`
  on edition 2024, where `static_mut_refs` is denied; the docs now show this and point to the
  storage types or a take-once cell instead ([#150]).
- `correct_with_gain*` now forms only the lower triangle of `K×S×Kᵀ` and mirrors it, so `P` comes out
  exactly symmetric; results change at the level of rounding, and an asymmetric `P` passed in is
  symmetrized from its lower triangle ([#107]).

### Removed

- Removed `UnscentedObservation::correct_nonlinear`. It averaged the sigma points with equal weights,
  ignoring the UKF weights, and read observed sigma points that callers could not set. Use
  `correct_sigma_point` / `correct_sigma_point_with` on the filter instead.

### Internal

- Added a standalone benchmark harness in `benches/compare/`, outside the workspace. It checks
  every filter variant against an independent `f64` reference before timing it, times
  predict, correct and full steps of minikalman's variants (and of other Rust Kalman crates at
  pinned versions) on fixed problems, counts allocations per step, and measures the
  `thumbv7em-none-eabihf` code size of one step. `run.sh` reproduces everything; the README
  describes the method. A CI job only compiles it ([#161]).
- Added `benches/mcu/`, a `no_std` firmware that counts the instructions of each filter
  operation (predict, correct, step, observer, UD, square root, EKF; `f32`, `I16F16`,
  `I32F32`; 2×1, 4×2, 6×3) on an emulated Cortex-M4F and Cortex-M0+ in Renode. The counts are
  deterministic; `BASE=<rev> run.sh` compares against another revision. They are instruction
  counts, not cycles ([#169]).
- Moved the innovation-form measurement noise adaptation, the process noise adaptation, the
  settle-gain comparison, the recalibration copies and traces, the robust correction's helpers
  and the triangular solves from `get`/`set` onto slices of the static length, which removes
  their bounds checks ([#168]). Results are bit-identical, pinned by tests against the old
  loops. The residual-form adaptation and the `Δx` copies of the process noise adaptation stay
  on `get`/`set`: their slice versions changed the instruction counts of the filter step on
  Cortex-M4F although they only run with the adaptation on. The filter operations of
  `benches/mcu` and the `thumbv7em-none-eabihf` code size of `benches/compare` are unchanged.
- Gated the tests and doc examples that use `f32`/`f64` on a float backend (`std`, `libm` or
  `micromath`), so `cargo test --no-default-features` and `--no-default-features --features
  fixed` compile and pass again. CI runs both configurations ([#177]).
  The same now holds with `alloc` but no float backend (`--features alloc` and
  `--features alloc,fixed`), and CI runs those as well.

[#209]: https://github.com/sunsided/minikalman-rs/issues/209
[#185]: https://github.com/sunsided/minikalman-rs/issues/185
[#186]: https://github.com/sunsided/minikalman-rs/issues/186
[#177]: https://github.com/sunsided/minikalman-rs/issues/177
[#181]: https://github.com/sunsided/minikalman-rs/issues/181
[#184]: https://github.com/sunsided/minikalman-rs/issues/184
[#187]: https://github.com/sunsided/minikalman-rs/issues/187
[#194]: https://github.com/sunsided/minikalman-rs/issues/194
[#195]: https://github.com/sunsided/minikalman-rs/issues/195
[#180]: https://github.com/sunsided/minikalman-rs/issues/180
[#179]: https://github.com/sunsided/minikalman-rs/issues/179
[#174]: https://github.com/sunsided/minikalman-rs/issues/174
[#168]: https://github.com/sunsided/minikalman-rs/issues/168
[#169]: https://github.com/sunsided/minikalman-rs/issues/169
[#161]: https://github.com/sunsided/minikalman-rs/issues/161
[#163]: https://github.com/sunsided/minikalman-rs/issues/163
[#167]: https://github.com/sunsided/minikalman-rs/issues/167
[#165]: https://github.com/sunsided/minikalman-rs/issues/165
[#166]: https://github.com/sunsided/minikalman-rs/issues/166
[#159]: https://github.com/sunsided/minikalman-rs/issues/159
[#157]: https://github.com/sunsided/minikalman-rs/issues/157
[#153]: https://github.com/sunsided/minikalman-rs/issues/153
[#124]: https://github.com/sunsided/minikalman-rs/issues/124
[#144]: https://github.com/sunsided/minikalman-rs/issues/144
[#143]: https://github.com/sunsided/minikalman-rs/issues/143
[#145]: https://github.com/sunsided/minikalman-rs/issues/145
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
[#150]: https://github.com/sunsided/minikalman-rs/issues/150
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
