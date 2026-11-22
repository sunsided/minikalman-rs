# minikalman lab: the filters in the browser

Live page: <https://sunsided.github.io/minikalman-rs/>

Builds minikalman for `wasm32-unknown-unknown` and runs it in the browser. The
module exports plain `extern "C"` functions that take and return numbers, so it
loads with `WebAssembly.instantiate` alone; no `wasm-bindgen`, bundler or
generated JS glue is needed. The page is plain HTML, CSS and ES modules; its only
third-party asset is the Recursive font from a pinned jsdelivr URL, with a system
font fallback.

## Experiments

Every trace is computed by minikalman in the module; JS only draws.

| Section | What it compares | Rust | JS |
|---|---|---|---|
| Smoothing a noisy position | A constant-velocity filter on a signal you draw | `lib.rs` | `exp-basics.js` |
| EKF vs UKF | Extended and unscented filters on range/bearing radar measurements; draw, drag and steer the target, a second radar, bearing wrapping | `radar.rs` | `exp-radar.js` |
| Outliers | Plain, NIS-gated and Student-t robust corrections on the same readings; outlier rate and size, a real jump that locks the gate out, re-acquiring after N rejections | `outliers.rs` | `exp-outliers.js` |
| Numerics | Standard vs Joseph covariance update in `f32`, `I16F16` and `I32F32` with ten sequential scalar sensors: smallest variance, asymmetry, definiteness | `numerics.rs` | `exp-numerics.js` |
| Smoothing | Filter vs fixed-lag vs RTS smoother through a gap in the readings | `smoothing.rs` | `exp-smoothing.js` |
| Parameter estimation | Dual RLS + Kalman filter vs a joint augmented-state EKF on a motor plant with a load step | `parameters.rs` | `exp-parameters.js` |
| Attitude | A quaternion error-state EKF with an NIS-gated accelerometer vs a ±10 % window on the accelerometer magnitude, with horizon dials | `attitude.rs` | `exp-attitude.js` |

`lab.js` holds the shared glue: loading the module, reading a run, the canvas
plots, knobs, theme. `main.js` starts each experiment; one failing does not stop
the others.

## The module's interface

### Batch experiments

Each lab experiment is one export, `<name>_run(knobs…) -> u32`, that simulates a
whole run with a seeded noise source, writes it to a static trace buffer and
returns the number of steps. Nothing allocates; knobs are clamped on the Rust
side. The doc comment of each `*_run` function lists its knobs, its trace
channels and its summary values.

| Export | Description |
|---|---|
| `trace_ptr() -> *const f32` | The trace of the last run, channel-major: channel `c` of `n` steps is at `[c·n, (c+1)·n)` |
| `summary_ptr() -> *const f32` | 64 scalar results of the last run (RMSE, counts, …) |
| `outliers_run(seed, sigma, rate, size, gate, dof, jump, reacquire)` | See `outliers.rs` |
| `numerics_run(seed, sensors, log_r, log_q, steps)` | See `numerics.rs` |
| `smoothing_run(seed, sigma, q, gap)` | See `smoothing.rs` |
| `parameters_run(seed, sensor_sigma, log_q, forgetting)` | See `parameters.rs` |
| `attitude_run(seed, dynamics, gate, hold_off, window, heading)` | See `attitude.rs` |

Read both buffers as a `Float32Array` over `memory.buffer` after each run. Each
experiment on the page gets its own instance of the module, so a trap in one
cannot leave another in a bad state; the glue replaces a trapped instance.
Fixed-point overflow wraps in the release build.

### 1D smoothing

| Export                       | Description                                                         |
|------------------------------|---------------------------------------------------------------------|
| `reset(dt, q, r)`            | Restart with time step, process noise density, measurement variance |
| `step(z) -> f32`             | Predict, correct with position `z`, return the filtered position    |
| `velocity() -> f32`          | Estimated velocity                                                  |
| `position_variance() -> f32` | Variance of the position estimate                                   |

### Radar tracking

State is `[x, y, vx, vy]`; `variant` is `0` for the EKF and `1` for the UKF.

| Export                                       | Description                                                |
|----------------------------------------------|------------------------------------------------------------|
| `radar_reset(x, y, q, sigma_range, sigma_bearing)` | Restart both filters at `(x, y)` with unknown velocity |
| `radar_predict(dt)`                          | Advance both filters by `dt` seconds                       |
| `radar_correct(sensor_x, sensor_y, range, bearing)` | Correct both filters with a measurement from a radar at `(sensor_x, sensor_y)` (bearing in radians) |
| `radar_set_wrap(enabled)`                    | Wrap the bearing residual into (−π, π] (default) or not; survives `radar_reset` |
| `radar_state(variant, i) -> f32`             | Entry `i` of the state                                     |
| `radar_covariance(variant, i, j) -> f32`     | Entry `(i, j)` of the estimate covariance                  |

Bearings from `atan2` jump by 2π behind a radar. Both filters wrap the bearing
residual with `angles::wrap_pi` (`correct_nonlinear_with` for the EKF,
`correct_sigma_point_with` for the UKF), so they track the target straight
through the jump.

## Build and run

From the repository root:

```shell
rustup target add wasm32-unknown-unknown
CARGO_PROFILE_RELEASE_DEBUG=false CARGO_PROFILE_RELEASE_STRIP=true \
  cargo build --release -p minikalman-wasm --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/minikalman_wasm.wasm xbuild-tests/wasm/www/
python3 -m http.server --directory xbuild-tests/wasm/www
```

Then open <http://localhost:8000>. The workspace release profile keeps debug
info for the embedded targets; the two environment variables drop it here,
which brings the module to about 195 kB (70 kB gzipped). `task example:wasm`
does the same.

The `GitHub Pages` workflow builds the same module and deploys `www/` on every
push to `main` that touches the library or this example.

To check the module without a browser:

```shell
node xbuild-tests/wasm/smoke.mjs target/wasm32-unknown-unknown/release/minikalman_wasm.wasm
```

It runs the 1D and radar filters, then every lab experiment at its page defaults
and at the extremes of its knobs, and checks the claim each experiment's note
makes (for example, that the RTS smoother bridges the gap and that `I16F16`
loses definiteness while `f32` does not).
