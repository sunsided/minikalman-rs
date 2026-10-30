# minikalman in the browser

Builds minikalman for `wasm32-unknown-unknown` and runs a constant-velocity
filter in the browser. The module exports plain `extern "C"` functions that
take and return `f32`, so it loads with `WebAssembly.instantiate` alone; no
`wasm-bindgen` or generated JS glue is needed.

| Export                       | Description                                                         |
|------------------------------|---------------------------------------------------------------------|
| `reset(dt, q, r)`            | Restart with time step, process noise density, measurement variance |
| `step(z) -> f32`             | Predict, correct with position `z`, return the filtered position    |
| `velocity() -> f32`          | Estimated velocity                                                  |
| `position_variance() -> f32` | Variance of the position estimate                                   |

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
which brings the module to about 21 kB.

To check the module without a browser:

```shell
node xbuild-tests/wasm/smoke.mjs target/wasm32-unknown-unknown/release/minikalman_wasm.wasm
```
