# Kalman filter benchmarks

A standalone harness that times minikalman's filter variants on a few small, fixed problems,
next to other Rust Kalman filter crates driven through their documented APIs. Its main use is
to measure changes to minikalman's kernels before and after, on the same inputs.

This directory is its own Cargo workspace. The minikalman workspace excludes it, so a normal
`cargo build` or `cargo test` of the repository never resolves the other crates.
It holds no results. Numbers depend on the machine, the toolchain and the load, so produce
your own (see [Reproducing](#reproducing)).

## What is measured

- **Problems** (`src/problem.rs`), defined once as row-major `f64`:
  - Constant velocity in 1, 2 and 3 dimensions: 2×1, 4×2 and 6×3 (states × measurements),
    `dt = 0.1`. `F = [[I, dt·I], [0, I]]`, `Q` is the continuous white-noise form (positive
    definite), `H` selects the positions, and `R` is full with small correlations.
  - An extended filter: a 4-state constant-velocity model with a range-and-bearing measurement.
    The trajectory stays in the first quadrant, so no wrapper needs to wrap the bearing.
  - Deterministic measurements (xorshift Gaussian): 64 warm-up steps plus 256 timed ones.
- **Filters** (`src/*_impl.rs`), each in `f32` and `f64`:
  - minikalman: storage types in Joseph form (the default) and standard form
    (`storage_joseph`, `storage_standard`), the boxed `alloc` builder (`boxed_joseph`), a lean
    observation with a shared `CorrectionScratch` (`lean_scratch_joseph`), the UD and
    square-root filters (`ud`, `square_root`), the fixed-gain Luenberger observer with its gain
    from `settle_gain` (`observer_fixed_gain`), and the extended filter (`extended_joseph`).
  - kfilter, smelly and kalman_filters, at the exact versions pinned in `Cargo.toml` (`=x.y.z`)
    and `Cargo.lock`. Each wrapper only forwards to that crate's public API; the module docs of
    each wrapper say which types and constructors it uses.
- **Correctness gate first** (`src/bin/check.rs`). Every filter runs 320 predict+correct steps
  next to an independent `f64` reference (`src/reference.rs`: dense loops, Gauss–Jordan
  inverse, Joseph form). The run exits non-zero unless the maximum relative error
  `|a − b| / (1 + |b|)` of `x` and `P` stays at or below `1e-9` (`f64`) or `2e-3` (`f32`) at
  every step. The fixed-gain observer is checked against a fixed-gain recursion with the same
  gain, and its gain against the reference filter's converged gain (`≤ 1e-5`). A filter that
  is fast but wrong never reaches the timing step.
- **Timing** (`benches/compare.rs`, criterion). Every filter goes through the same
  monomorphized loop, without `dyn`. It starts at the reference state after the warm-up and is
  restored to it every 256 iterations, inside the timed loop for every filter. Measurements
  cycle, and the filter and each measurement pass through `black_box`. The time includes
  writing `z` through each crate's API. `predict`, `correct` and a full `step` (predict +
  correct) are timed separately. Read `step` as the primary number: the per-phase times are not
  additive, because the out-of-order core overlaps adjacent calls differently per loop shape.
- **Allocations and memory** (`check`): a counting `#[global_allocator]` over 100 steps after
  the agreement run, the live heap after construction, and `size_of` of the filter types.
- **Code size** (`codesize/`, its own workspace): one `no_std` binary per filter for
  `thumbv7em-none-eabihf` (Cortex-M4F), `f32`, 4×2. `step` is the linker entry and takes the
  filter memory as arguments, so nothing is constant-folded. `--gc-sections`, fat LTO,
  `panic = "abort"`, `opt-level = "s"` (profile `release`) or `3` (profile `o3`). Subtract the
  `baseline` binary (an empty `step`) to get the size of the step alone.

## Reproducing

Requirements: a stable Rust toolchain (edition 2024), `python3` for the tables, and for the
code size step `rustup target add thumbv7em-none-eabihf` plus `llvm-size`.

```sh
benches/compare/run.sh                       # gate, metrics, all timings, tables, code size
BENCH_FILTER='minikalman/' benches/compare/run.sh   # minikalman's variants only
CPU=4 SKIP_CODESIZE=1 benches/compare/run.sh        # pin to core 4, skip the thumb build
```

`run.sh` stops at the first failing step; a filter that leaves the tolerance fails the gate
before anything is timed. Outputs go to `benches/compare/results/` (git-ignored):
`check.json` (errors, allocations, live heap), `tables.md` (criterion medians of predict,
correct and step per problem) and `codesize.txt`. The single steps also run on their own:

```sh
cd benches/compare
cargo run --release --bin check > check.json
cargo bench --bench compare -- 'kf_4x2/f32/minikalman'
python3 -I tables.py target/criterion
(cd codesize && cargo build --release --bins && cargo build --profile o3 --bins)
```

To compare a change, run the timings on the base commit and on the change on the same machine,
pinned to the same core, a few times each; `tables.py` takes several criterion directories and
reports the median of the per-run medians with half the spread between runs.

## Caveats

- Timings are wall-clock on one core of a general-purpose machine. Frequency scaling, turbo,
  hybrid cores and background load move the numbers by 10–15 % between runs; pin to one core,
  repeat the runs, and treat smaller differences as noise.
- The default target is the x86-64 baseline (SSE2, no AVX or FMA). `RUSTFLAGS="-C
  target-cpu=native"` is optional and changes what the compiler can vectorize, for every crate.
- On microcontrollers, code size and operation counts matter more than these timings; this
  harness takes no cycle counts on hardware.
- Each wrapper uses the documented primary types of its crate. A faster path that a crate
  offers but the wrapper does not use would not show here.
