# Conformance suite

A set of checks that the filters in `minikalman` compute what the math says they compute.
It is meant for two audiences:

- **Maintainers**: any change that breaks the math, or quietly makes a filter less accurate,
  fails a test with a message that names the scenario, the step and the error.
- **Users**: you can run it yourself, read what each check proves, and regenerate the
  reference data with an implementation that does not use this crate.

```sh
# All categories, f32/f64 plus the fixed-point types (about 1 s after the build)
cargo test -p minikalman --test conformance --features std,fixed

# The same suite on the no_std math backends
cargo test -p minikalman --test conformance --no-default-features --features libm,alloc
cargo test -p minikalman --test conformance --no-default-features --features micromath,alloc

# Or all three in one go
task conformance

# See the measured error of every comparison next to its tolerance
cargo test -p minikalman --test conformance --features std,fixed -- --nocapture --test-threads 1

# Repeat the Monte Carlo checks with another seed
MINIKALMAN_CONFORMANCE_SEED=7 cargo test -p minikalman --test conformance --features std consistency
```

The suite needs the `alloc` feature (it uses the builders); without it the test target is
empty. CI runs it for `std,fixed`, `libm,alloc` and `micromath,alloc`.

## What it proves and what it does not

It proves that, on the scenarios below, the regular, extended and unscented filters, the
information filter and the RTS smoother

1. reproduce an independent NumPy/SciPy implementation step for step,
2. satisfy the identities that must hold between them,
3. produce covariances that match the actual errors (NEES/NIS),
4. keep `P` symmetric and positive semidefinite, honour `CorrectionStatus`, and are
   deterministic,
5. are no less accurate than when the baselines were taken,

in `f64`, `f32`, and (for the regular filter and smoother) `I32F32` and `I16F16`.

It does not prove correctness for models it does not exercise: other dimensions,
ill-conditioned problems, the features listed under [follow-ups](#follow-ups), or numbers
outside the ranges the scenarios use. Passing tests on a few scenarios are evidence, not a
proof for every input; the property tests (`proptest_*.rs`) and fuzz targets (`fuzz/`)
cover the input space more broadly, and the feature tests (`nis_gating.rs`,
`information_filter.rs`, `rts_smoother.rs`, ...) pin individual behaviours.

## Layout

| File                   | Content                                                                       |
|------------------------|-------------------------------------------------------------------------------|
| `generate.py`          | The independent implementation; writes `reference.rs`                         |
| `reference.rs`         | Generated model matrices, measurements, expected outputs and chi-square bounds |
| `scenarios.rs`         | Sets each scenario up in the crate and records every step into a `History`    |
| `support.rs`           | Number types, tolerances, `assert_close`, covariance checks                   |
| `reference_vectors.rs` | Category 1                                                                    |
| `identities.rs`        | Category 2                                                                    |
| `monte_carlo.rs`, `consistency.rs` | Category 3                                                        |
| `invariants.rs`        | Category 4                                                                    |
| `baselines.rs`         | Category 5                                                                    |

## Scenarios

All inputs, including the measurements, come from `generate.py` and are loaded element by
element, so the crate and the reference see exactly the same numbers.

| Scenario   | Filter                    | Model                                                                                             |
|------------|---------------------------|---------------------------------------------------------------------------------------------------|
| `cv`       | regular, information, RTS | `[position, velocity]`, `dt = 0.1`, full-rank white-acceleration `Q` (`q = 0.5`), position measured, `R = 0.25`, 50 steps |
| `gravity`  | regular                   | `[position, velocity, acceleration]`, `dt = 1`, free fall at 9.81 m/s², position measured, 15 steps  |
| `radar`    | extended                  | constant-velocity target `[px, py, vx, vy]`, radar at `(−10, −2)` measures range and bearing, 40 steps |
| `pendulum` | unscented (`α = 1, β = 2, κ = 1`) | Euler-stepped pendulum `[angle, rate]`, bob's horizontal position measured, 60 steps        |

## Categories and pass criteria

A comparison passes when `|actual − expected| ≤ abs + rel × |expected|` for every entry.

### 1. Reference vectors (`reference_vectors.rs`)

Every filtered state and covariance (for `cv` also every prediction) must match the
NumPy/SciPy implementation in `generate.py`. That implementation is written from the
textbook equations and shares no code with the crate. For the UKF it follows the same
convention as the crate and as filterpy: the correction reuses the sigma points propagated
by the prediction instead of redrawing them from the predicted `P` (see
[findings](#findings)). If filterpy is importable, `generate.py` also runs filterpy's
`UnscentedKalmanFilter` on the pendulum and asserts it agrees to `1e-12`. That check
passed with filterpy 1.4.5.

| Type     | Tolerance (`abs`, `rel`) | Largest error measured | Feature sets                 |
|----------|--------------------------|------------------------|------------------------------|
| `f64`    | `1e-12`, `1e-11`         | `9.8e-15`              | `std`, `libm`                |
| `f32`    | `5e-5`, `1e-5`           | `8.5e-5` on `gravity` (positions up to 1100), else `1.5e-6` | `std`, `libm` |
| `I32F32` | `1e-7`, `1e-7`           | `1.6e-8`               | `fixed` (regular, RTS)       |
| `I16F16` | `1e-3`, `1e-3`           | `2.1e-4`               | `fixed` (regular, `cv` only) |
| `f32`, `f64` | `0.3`, `0.1`         | `0.2` (RTS)            | `micromath`                  |

Under micromath the gravity and information-filter comparisons are ignored, because the
approximate `sqrt` and `recip` move them further than any useful tolerance (see
[findings](#findings)).

### 2. Analytic identities (`identities.rs`, `f64`)

| Identity                                                    | Tolerance                     |
|-------------------------------------------------------------|-------------------------------|
| EKF with `f(x) = A×x`, `h(x) = H×x` equals the KF (`cv`, `gravity`) | `1e-12`, `1e-11` (measured: bit-identical) |
| UKF on a linear model with `Q = 0` equals the KF             | `1e-12`, `1e-11` (measured `2.9e-15`) |
| Information filter equals the KF (`cv`, `gravity`)           | `1e-12`, `1e-11` (measured `8e-13`, on `gravity` positions up to 1100) |
| RTS smoother equals the batch least-squares smoother, solved by `generate.py` as one 102 × 102 linear system | `1e-10` abs (measured `4.5e-13`; the batch solve itself costs about `1e-12`) |
| After 500 steps the KF's `P⁻`, `P` and `K` equal the steady state from SciPy's `solve_discrete_are` | `1e-12`, `1e-11` |
| Scalar random walk: one step equals `K = (p₀+q)/(p₀+q+r)` in closed form (KF, EKF, information filter); steady-state `P⁻ = (q + √(q² + 4qr))/2` | `1e-12`, `1e-11` |

`generate.py` checks its own side too: it fails if the Riccati residual of the SciPy
solution exceeds `1e-12`, or if its RTS smoother and batch smoother differ by more than
`1e-10`.

### 3. Statistical consistency (`consistency.rs`, `f64`)

For each of `cv` (KF), `radar` (EKF) and `pendulum` (UKF), 100 Monte Carlo runs of 50 steps
draw the initial truth from `N(x₀, P₀)`, process noise from the filter's own `Q` and
measurement noise from its `R`.

- **NEES** `eᵀP⁻¹e` of `n` states, averaged over the 100 runs at steps 20, 30, 40 and 50,
  must lie in the two-sided `χ²(100·n)/100` interval at `α/4` each (Bonferroni):
  `[1.349, 2.816]` for `n = 2`, `[3.046, 5.119]` for `n = 4`. NEES is correlated over time,
  so it is not pooled over steps. Steps before 20 are not tested: with the checks at
  step 10, the UKF failed 2 of 40 seeds there, during the transient from its large initial
  uncertainty.
- **NIS** of `m` measurements, averaged over all runs and steps, must lie in the two-sided
  `χ²(5000·m)/5000` interval: `[0.935, 1.067]` for `m = 1`, `[1.908, 2.094]` for `m = 2`.
  Innovations are white, so pooling is valid.

With `α = 10⁻³` per test, a correct linear filter fails one scenario with probability at most
`2·10⁻³` for a random seed. The seed is fixed, so CI is deterministic. For the EKF and UKF
the bounds are an approximation, since their errors are not exactly Gaussian. Across 300
seeds (`MINIKALMAN_CONFORMANCE_SEED=1..300`) the KF and the EKF never failed. The UKF failed
13 times (4.3 %), mostly on the NEES at step 20, so it is slightly overconfident on this
model. The bounds come from `scipy.stats.chi2` and live in `reference.rs` (`chi2` module).

### 4. Invariants (`invariants.rs`)

- Every filtered (and smoothed) `P` of every scenario and number type is symmetric within
  `ε × max(1, max Pᵢᵢ)`, and `P + ε·I` has a Cholesky factor, with `ε` a few units of the
  type's resolution: `1e-12` (`f64`), `1e-5` (`f32`), `1e-7` (`I32F32`), `1e-3` (`I16F16`).
  `ε` does not depend on the math backend, so micromath gets no extra slack here, and it
  passes.
- A correction that returns `Gated` (NIS above the gate) or `NotPositiveDefinite` (negative
  `R`) leaves `x` and `P` **bit-for-bit** unchanged, for the regular, extended and unscented
  filters. An information-filter `fuse` with a non-positive-definite `R` returns `false` and
  leaves `Y` and `ŷ` unchanged.
- Running every scenario twice gives bit-identical outputs (`f64`, `f32`, `I16F16`).

### 5. Regression baselines (`baselines.rs`, `f64`)

The position RMSE, the average NEES and the average NIS of the three Monte Carlo
scenarios, with the default seed, must stay within ±0.5 % of the pinned values. These
numbers come from this crate, not from an independent source. They do not show that the
filters are correct; the categories above do. They catch a change that passes every
identity but makes a filter measurably worse. The band applies in both directions, so an
improvement also fails until someone updates the baseline on purpose. Under micromath the
baselines are ignored.

The Monte Carlo runs draw from `rand`'s `StdRng` and `rand_distr`, which `Cargo.lock`
pins. Upgrading either can change the random streams. The consistency checks then still
hold (with probability `1 − α`), but the baselines need to be refreshed in the same
change.

## Findings

Building the suite surfaced these behaviours, documented here rather than changed:

- **UKF and process noise.** The correction passes the sigma points from the prediction
  through `h`, and those points were spread by `A×P×Aᵀ` before `Q` was added. So the
  innovation covariance lacks `H×Q×Hᵀ`, and on a linear model with `Q ≠ 0` the UKF is not
  the KF: on `cv` the states differ by up to `6.3e-3` and `P` by up to `4.4e-3` (up to 10 %
  relative, in the early steps). filterpy 1.4.5 behaves the same way: `update` passes
  `sigmas_f` through `hx`. Redrawing the sigma points from the predicted `x` and `P` before
  the correction would make the identity exact. The identity is therefore tested with
  `Q = 0`.
- **micromath accuracy.** With micromath's approximate `sqrt` and `recip` (also used for
  `f64`, through `f32`), the filters drift from the exact result. They are off by up to
  0.03 on `cv`, 0.1 on `radar` and `pendulum`, and 0.2 for the RTS smoother and for the
  UKF with `Q = 0` against the KF. The gravity `P` is off by up to 0.8 (130 %), and the
  information filter's `x` by up to 0.63. On `pendulum` the UKF's NEES at step 20 drops to
  1.07, below the chi-square interval (underconfident). Under micromath, 9 checks are
  ignored, each with its reason in the `ignore` attribute: four reference comparisons
  (gravity and the information filter), the information filter = KF identity, the UKF
  consistency check and the three baselines. Everything else runs, including the
  invariants and the KF and EKF consistency checks.

## Regenerating the references

```sh
python3 crates/minikalman/tests/conformance/generate.py   # needs numpy and scipy
git diff crates/minikalman/tests/conformance/reference.rs
```

The noise comes from `numpy.random.RandomState`, whose stream NumPy keeps stable, and is
shaped by Cholesky factors, which are unique. On the same NumPy and LAPACK the
regenerated file is byte-identical. Elsewhere the last digits may differ, so the portable
check is that the suite still passes against the regenerated file. The header records the
generator's versions (here Python 3.11, NumPy 1.24.3, SciPy 1.11.1).
`reference.rs` is excluded from rustfmt (`#[rustfmt::skip]` in `main.rs`), so the
generator's layout is what gets committed.

## Adding a filter or a feature

A new filter variant plugs in by adding a runner to `scenarios.rs` that returns a
`History`. Every category is then about one line away:

- **Reference vectors**: compare its `History` to an existing reference
  (`cv::X_POST`, ...), or add a scenario to `generate.py` with its own NumPy implementation
  and regenerate.
- **Identities**: compare it to `run_regular` on a linear scenario.
- **Invariants**: add it to `covariances_stay_valid` and `all_runs`, and test its rejected
  corrections like the existing ones.
- **Consistency and baselines**: add a function to `monte_carlo.rs` and call
  `assert_consistent` and `assert_baseline`.

The features in progress plug in like this:

- **Joseph-form update (#107)**: must equal the standard update on `cv`/`gravity`
  (identity), must reproduce `cv::P_POST` in `f32` and the fixed-point types, ideally with a
  tighter tolerance. Its robustness bench adds ill-conditioned cases to the invariants.
- **Student-t correction (#108)**: with a very large `ν` it must equal the KF (identity).
  Its outlier behaviour needs its own NumPy reference scenario with heavy-tailed noise.
- **Householder QR and square-root filter (#109, #110)**: `S×Sᵀ` must reproduce
  `cv::P_POST`/`gravity::P_POST`. The triangular factor makes the PSD invariant hold by
  construction, so the useful test is the reference comparison in `f32` and fixed point.
- **UD filter (#111)**: `U×D×Uᵀ` must reproduce the `cv` reference, with a fixed-point
  tolerance at least as tight as the regular filter's.
- **Invariant EKF example (#112)**: a Monte Carlo NEES check on its Lie-group model in
  `monte_carlo.rs`, with an empirical band like the EKF's.

## Follow-ups

Not covered yet:

- Fixed-lag smoother, predictively oriented corrections, noise adaptation, covariance
  recalibration, control inputs, angle wrapping (all have feature tests, but no reference
  vectors or Monte Carlo checks yet).
- EKF and UKF in fixed point; Monte Carlo checks in `f32`.
- Checking the generator in CI: regenerate and compare numerically, since a byte diff can
  fail on another LAPACK.
- The `nalgebra` interop types.
