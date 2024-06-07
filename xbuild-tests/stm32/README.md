# STM32F303 embedded

This directory contains an example project targeting STM32F303 microcontrollers.
For a general project overview, see the [README.md in the repository root](../../README.md) for more details.

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
