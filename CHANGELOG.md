# Changelog

All notable changes to this project will be documented in this file.
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added `copy_to` and `copy_from` to matrix trait.

### Changed

- The `copy` matrix operation is now deprecated in favor of `copy_to`.

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
