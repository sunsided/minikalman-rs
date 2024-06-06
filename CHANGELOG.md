# Changelog

All notable changes to this project will be documented in this file.
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Removed the `no_std` crate feature in favor of the `std` feature (disabled by default).

### Internal

- Add an example project for STM32F303 cross-compilation.

## [0.3.0] - 2024-06-04

[0.3.0]: https://github.com/sunsided/minikalman-rs/releases/tag/v0.3.0

### Added

- Added support for fixed-point values via the [fixed](https://crates.io/crates/fixed) crate. 

### Changed

- The macros, matrix, filter and measurement structs are now generic on the data type.
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
