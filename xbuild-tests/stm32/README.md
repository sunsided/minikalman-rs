# STM32F303 embedded

This directory contains an example project targeting STM32F303 microcontrollers.
For a general project overview, see the [README.md in the repository root](../../README.md) for more details.

## Filter memory

Both examples (`kalman_float_example.rs`, `kalman_fixed_example.rs`) keep all filter buffers in
one `static`:

- The storage types (`regular::FilterStorage`, `regular::ObservationStorage`) have a `const fn new`,
  so the `static` is initialized at compile time. With a zero initial value it lands in `.bss`
  (276 bytes per example, including the cell's flag) and is never built on the stack or moved.
- `static_cell::ConstStaticCell::try_take` hands out the only `&'static mut` to it. A second or
  re-entrant call, e.g. from an interrupt handler, gets `None` instead of a second mutable
  reference to the same memory. This replaces the earlier `static mut` references, which were
  unsound when re-entered and are denied by the `static_mut_refs` lint in edition 2024.
- `build()` borrows the storage; the filter holds only slices into it.

### Cortex-M0/M0+ (`thumbv6m-none-eabi`)

These cores have no atomic compare-and-swap. `static_cell` uses `portable-atomic`, which then
needs its `critical-section` feature and a critical-section implementation, for example on a
single-core part:

```toml
[dependencies]
static_cell = "2.1.1"
portable-atomic = { version = "1", default-features = false, features = ["critical-section"] }
cortex-m = { version = "0.7.7", features = ["critical-section-single-core"] }
```

Without the `critical-section` feature, `static_cell` fails to compile for `thumbv6m-none-eabi`
("`compare_exchange` requires atomic CAS but not available on this target by default").

## Build configuration

These projects use special configuration that would normally be part of `Cargo.toml`
but had to be moved to the workspace's configuration instead:

```toml
[profile.dev]
panic = "abort"

[profile.release]
panic = "abort"
codegen-units = 1  # better optimizations
debug = true       # symbols are nice, and they don't increase the size on Flash
lto = true         # better optimizations
```
