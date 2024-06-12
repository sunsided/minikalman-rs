# Kalman Filters for Embedded Targets (in Rust)

[![Crates.io](https://img.shields.io/crates/v/minikalman)](https://crates.io/crates/minikalman-rs)
[![Crates.io](https://img.shields.io/crates/l/minikalman)](https://crates.io/crates/minikalman-rs)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/sunsided/minikalman-rs/rust.yml)
[![docs.rs](https://img.shields.io/docsrs/minikalman)](https://docs.rs/minikalman/)
[![codecov](https://codecov.io/gh/sunsided/minikalman-rs/graph/badge.svg?token=YJYARXA8EL)](https://codecov.io/gh/sunsided/minikalman-rs)

This is the Rust port of my [kalman-clib](https://github.com/sunsided/kalman-clib/) library,
a microcontroller targeted Kalman filter implementation, as well as the
[libfixkalman](https://github.com/sunsided/libfixkalman) C library for Q16.16 fixed-point Kalman filters.
It uses [`micromath`](https://docs.rs/micromath) for square root calculations on `no_std`. Depending on the
configuration, this crate may
require `f32` / FPU support.

This implementation uses statically allocated buffers for all matrix operations. Due to lack
of `const` generics for array allocations in Rust, this crate also provides helper macros
to create the required arrays.

<div align="center">
    <img src="docs/hero.webp" width="780" alt="Kalman Filter Library Hero Picture" />
</div>

## `no_std` vs `std`, `alloc`

This crate builds as `no_std` by default. To build with `std` support, run:

```
cargo build --features=std
```

Independently of `std` you can turn on `alloc` features. This enables simplified builders with heap-allocated buffers:

```
cargo build --features=alloc
```

## Examples

### Targets with allocations (`std` or `alloc`)

```rust
const NUM_STATES: usize = 3;
const NUM_INPUTS: usize = 2;
const NUM_MEASUREMENTS: usize = 1;

fn example() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.measurements().build::<NUM_MEASUREMENTS>();
    let mut input = builder.inputs().build::<NUM_INPUTS>();

    // Set up the system dynamics, input matrices, observation matrices, ...

    // Filter!
    loop {
        // Update your input vector(s).
        input.input_vector_apply(|u| {
            u[0] = 0.0;
            u[1] = 1.0;
        });

        // Update your measurement vectors.
        measurement.measurement_vector_apply(|z| {
            z[0] = 42.0;
        });

        // Update prediction and apply the inputs.
        filter.predict();

        // Apply any inputs.
        filter.input(&mut input);

        // Apply any measurements.
        filter.correct(&mut measurement);

        // Access the state
        let state = filter.state_vector_ref();
        let covariance = filter.system_covariance_ref();
    }
}
```

### Embedded Targets

An example for STM32F303 microcontrollers can be found in the
[`xbuild-tests/stm32`] directory. It showcases both fixed-point and floating-point support.

### `Q16.16` fixed-point

Run the `fixed` example with the `fixed` crate feature. This enables `I16F16` type support, similar to
the [libfixkalman](https://github.com/sunsided/libfixkalman) C library.

```shell
cargo run --example fixed --features=fixed
```

To disable floating-point support, run

```shell
cargo run --example fixed --no-default-features --features=fixed
```

### `f32`/`f64` floating-point

The provided example code will print output only on `float` builds. Selecting this feature
simply enables the following implementation for `f32` and `f64`:

```rust
impl MatrixDataType for f32 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        self.recip()
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        self.sqrt()
    }
}
```

To run the example [`gravity`] simulation, run

```shell
cargo run --example gravity --features=float,libm
```

or

```shell
cargo run --example gravity --features=float,std
```

This will estimate the (earth's) gravitational constant (g ≈ 9.807 m/s²) through observation
of the position of a free-falling object. When executed, it should print something along the lines of:

```
At t = 0, predicted state: s = 3 m, v = 6 m/s, a = 6 m/s²
At t = 0, measurement: s = 0 m, noise ε = 0.13442 m
At t = 0, corrected state: s = 0.908901 m, v = 3.6765568 m/s, a = 5.225519 m/s²
At t = 1, predicted state: s = 7.1982174 m, v = 8.902076 m/s, a = 5.225519 m/s²
At t = 1, measurement: s = 4.905 m, noise ε = 0.45847 m
At t = 1, corrected state: s = 5.6328573 m, v = 7.47505 m/s, a = 4.5993752 m/s²
At t = 2, predicted state: s = 15.407595 m, v = 12.074425 m/s, a = 4.5993752 m/s²
At t = 2, measurement: s = 19.62 m, noise ε = -0.56471 m
At t = 2, corrected state: s = 18.50683 m, v = 14.712257 m/s, a = 5.652767 m/s²
At t = 3, predicted state: s = 36.04547 m, v = 20.365025 m/s, a = 5.652767 m/s²
At t = 3, measurement: s = 44.145 m, noise ε = 0.21554 m
At t = 3, corrected state: s = 42.8691 m, v = 25.476515 m/s, a = 7.3506646 m/s²
At t = 4, predicted state: s = 72.02094 m, v = 32.82718 m/s, a = 7.3506646 m/s²
At t = 4, measurement: s = 78.48 m, noise ε = 0.079691 m
At t = 4, corrected state: s = 77.09399 m, v = 36.10087 m/s, a = 8.258889 m/s²
At t = 5, predicted state: s = 117.3243 m, v = 44.359756 m/s, a = 8.258889 m/s²
At t = 5, measurement: s = 122.63 m, noise ε = -0.32692 m
At t = 5, corrected state: s = 120.94025 m, v = 46.38022 m/s, a = 8.736543 m/s²
At t = 6, predicted state: s = 171.68874 m, v = 55.11676 m/s, a = 8.736543 m/s²
At t = 6, measurement: s = 176.58 m, noise ε = -0.1084 m
At t = 6, corrected state: s = 174.93135 m, v = 56.704926 m/s, a = 9.062785 m/s²
At t = 7, predicted state: s = 236.16766 m, v = 65.76771 m/s, a = 9.062785 m/s²
At t = 7, measurement: s = 240.35 m, noise ε = 0.085656 m
At t = 7, corrected state: s = 238.87048 m, v = 66.942894 m/s, a = 9.276019 m/s²
At t = 8, predicted state: s = 310.4514 m, v = 76.21891 m/s, a = 9.276019 m/s²
At t = 8, measurement: s = 313.92 m, noise ε = 0.8946 m
At t = 8, corrected state: s = 313.03793 m, v = 77.22877 m/s, a = 9.44006 m/s²
At t = 9, predicted state: s = 394.98672 m, v = 86.66882 m/s, a = 9.44006 m/s²
At t = 9, measurement: s = 397.31 m, noise ε = 0.69236 m
At t = 9, corrected state: s = 396.6648 m, v = 87.26297 m/s, a = 9.527418 m/s²
At t = 10, predicted state: s = 488.69147 m, v = 96.79039 m/s, a = 9.527418 m/s²
At t = 10, measurement: s = 490.5 m, noise ε = -0.33747 m
At t = 10, corrected state: s = 489.46213 m, v = 97.03994 m/s, a = 9.560934 m/s²
At t = 11, predicted state: s = 591.28253 m, v = 106.600876 m/s, a = 9.560934 m/s²
At t = 11, measurement: s = 593.51 m, noise ε = 0.75873 m
At t = 11, corrected state: s = 592.75964 m, v = 107.04147 m/s, a = 9.615404 m/s²
At t = 12, predicted state: s = 704.6088 m, v = 116.656876 m/s, a = 9.615404 m/s²
At t = 12, measurement: s = 706.32 m, noise ε = 0.18135 m
At t = 12, corrected state: s = 705.4952 m, v = 116.90193 m/s, a = 9.643473 m/s²
At t = 13, predicted state: s = 827.2188 m, v = 126.5454 m/s, a = 9.643473 m/s²
At t = 13, measurement: s = 828.94 m, noise ε = -0.015764 m
At t = 13, corrected state: s = 827.97705 m, v = 126.74077 m/s, a = 9.66432 m/s²
At t = 14, predicted state: s = 959.55 m, v = 136.40509 m/s, a = 9.66432 m/s²
At t = 14, measurement: s = 961.38 m, noise ε = 0.17869 m
At t = 14, corrected state: s = 960.39984 m, v = 136.6101 m/s, a = 9.684802 m/s²
```

[`gravity`]: https://github.com/sunsided/minikalman-rs/tree/main/crates/minikalman/examples/gravity.rs

[`xbuild-tests/stm32`]: https://github.com/sunsided/minikalman-rs/tree/main/xbuild-tests/stm32
