# minikalman in the browser

Live demo: <https://sunsided.github.io/minikalman-rs/>

Builds minikalman for `wasm32-unknown-unknown` and runs it in the browser. The
module exports plain `extern "C"` functions that take and return numbers, so it
loads with `WebAssembly.instantiate` alone; no `wasm-bindgen` or generated JS
glue is needed.

The page shows two demos:

- **Smoothing a noisy position**: a linear constant-velocity filter in 1D.
- **Tracking a moving target**: radars measure range and bearing to a target
  circling past them. An Extended and an Unscented Kalman filter run side by
  side on the same measurements; each plot shows the crescent-shaped
  measurement uncertainty against the filter's 2σ ellipse. Toggles turn on a
  second radar and turn off wrapping of the bearing residual, to show what goes
  wrong when the target passes behind a radar.

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
which brings the module to about 47 kB (21 kB without the radar demo).

The `GitHub Pages` workflow builds the same module and deploys `www/` on every
push to `main` that touches the library or this example.

To check the module without a browser:

```shell
node xbuild-tests/wasm/smoke.mjs target/wasm32-unknown-unknown/release/minikalman_wasm.wasm
```
