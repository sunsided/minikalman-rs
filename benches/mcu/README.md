# Instruction counts on Cortex-M (Renode)

A `no_std` firmware that runs minikalman's filter operations on an emulated Cortex-M4F and
Cortex-M0+ in [Renode](https://renode.io), and counts the instructions each operation
executes. The counts are deterministic: the same binary gives the same numbers on every run
and every machine, so a change to the kernels can be compared before and after without the
noise of wall-clock timings.

This directory is its own Cargo workspace. The minikalman workspace excludes it, so a normal
`cargo build` or `cargo test` of the repository never builds it. It holds no results;
produce your own (see [Reproducing](#reproducing)).

> **Instruction counts are not cycles.** Renode executes instructions; it does not model the
> pipeline, flash wait states, the FPU's latencies (`vdiv.f32` and `vsqrt.f32` take 14 cycles
> on a Cortex-M4F, a `vmul.f32` one) or the M0+'s 1- or 32-cycle multiplier. The counts are a
> proxy for the work an operation does, good for comparing two versions of the same code on
> the same core, and a lower bound for its cycles. Renode's DWT `CYCCNT` is no better: it is
> derived from virtual time, i.e. the instruction count scaled by `frequency / MIPS` (checked:
> with `frequency = 168 MHz` at the default 100 MIPS it reads 1.68× the instruction count).
> For cycles, measure on hardware (DWT `CYCCNT` on an STM32F303, for example).

## What is measured

- **Cores and platforms.** The instruction stream depends on the core and the target, not on
  the chip around it, so any Renode platform with that core and enough memory will do:
  - `thumbv7em-none-eabihf` (hard float, as `xbuild-tests/stm32` on the STM32F303) on
    Renode's `stm32f412.repl`, a Cortex-M4F. The STM32F303 itself has no Renode platform.
  - `thumbv6m-none-eabi` (no FPU, soft float) on `stm32l071.repl`, a Cortex-M0+.
    `renode/stm32l071-large.repl` enlarges its flash to 384 KiB and its RAM to 64 KiB: the
    `opt-level = 3` image with `I32F32` needs about 300 KiB, and the `I32F32` 6×3 stack
    leaves only about 1 KiB of the chip's 20 KiB. One binary per element type keeps the
    images small.
- **Problems**, the same as in `benches/compare`: constant velocity in 1, 2 and 3 dimensions
  (2×1, 4×2 and 6×3, states × measurements, `dt = 0.1`, full `R`), and the 4-state
  range–bearing radar for the extended filter (`f32` only).
- **Element types**: `f32`, `I16F16` and `I32F32` (`fixed` crate). minikalman is built with
  `libm` and `fixed`, without `alloc`; all buffers are on the stack.
- **Operations**, each per call:
  - `predict`, `correct_joseph` (the default) and `correct_standard` of the regular filter,
    and `step_joseph` (predict + correct in one measured section);
  - `observer_step`: predict + correct of the fixed-gain (Luenberger) observer;
  - `ud_predict`/`ud_correct` (Bierman/Thornton) and `sqrt_predict`/`sqrt_correct`
    (square-root covariance filter);
  - `ekf_radar_4x2/predict` and `correct_joseph`, the latter including the Jacobian and
    `h(x)` (`sqrtf`, `atan2f`);
  - `prim/*`: one `add`, `mul`, `div` and `sqrt` of each type, plus `copy` (only the operand
    loads and the store), to read the filter counts against.
- **Method** (`src/main.rs`, `renode/bench.py`):
  - Each operation runs between calls to the `#[inline(never)]` marker functions `mk_begin`
    and `mk_end(name, ok)`. Renode hooks both and logs the core's `ExecutedInstructions`
    in between. The marker overhead (`calib/empty`) is subtracted.
  - Setup, the model, 8 warm-up steps and the measurements are outside the markers. The
    filter references go through `black_box`, so nothing is specialized on the constant
    model.
  - 16 iterations per operation, in a realistic sequence: predict, then correct with a
    fresh measurement. The table shows the mean, and the range when iterations differ.
    Soft float, fixed-point division and the libm `sqrtf` have data-dependent paths, so the
    range is not always zero.
  - `ok` is the operation's status (`is_applied()`). A rejected correction would take a
    shortcut, so the table marks it ("not applied").
  - With `COUNT_CALLS=1`, calls to the soft-float and 64/128-bit helpers (`__aeabi_*`,
    `__*ti3`) and to `sqrtf` and `atan2f` inside each section are counted too. Helpers that
    the compiler inlines (the `fixed` crate's `I32F32` division, for example) are not
    visible there. Use it on the M4F only: with hooks on the soft-float helpers, the M0+
    run stalls in Renode 1.17.
  - The firmware paints the free stack and reports the unused part at the end. The run
    fails if the stack came within 256 B of the statics.
- Profiles: `release` (`opt-level = "s"`) and `o3` (`opt-level = 3`), both with fat LTO,
  one codegen unit and `panic = "abort"`.

## Reproducing

Requirements: Renode on `PATH` (tested with 1.17.0), a stable Rust toolchain with
`rustup target add thumbv7em-none-eabihf thumbv6m-none-eabi`, `llvm-nm` (or `rust-nm` from
`cargo-binutils`) and `python3`.

```sh
benches/mcu/run.sh                                   # both cores, both profiles, all types
BASE=origin/main benches/mcu/run.sh                  # the same against minikalman at a revision
TARGETS=thumbv7em-none-eabihf PROFILES=release TYPES=f32 benches/mcu/run.sh   # one binary
task bench:mcu
```

`run.sh` builds one binary per target, profile and type, runs each headless in Renode and
writes the raw logs to `benches/mcu/results/head/` (and `base/` with `BASE`), then prints the
tables (also written to `results/tables.md`). With `BASE=<rev>` it extracts the whole tree at
`<rev>` with `git archive`, copies this firmware and its `Cargo.lock` into it, and builds it
the same way, so both sides use the same firmware, dependencies and toolchain. A full run of
both sides takes a few minutes, most of it building; each emulator run takes seconds.

The run stops when a binary panics (the panic handler calls the hooked `mk_fail`) or does
not reach its end, and when the stack came too close to the statics.

## Checks

- **Deterministic**: two runs of the same binary give byte-identical logs.
- **Exact**: the counter is not quantized to translation blocks. In a first prototype, a
  loop of 100 iterations of 4 instructions measured 407 (400 plus the loop setup).
- **Against the disassembly**: for `f32` 2×1 `predict` on the M4F (`opt-level = "s"`),
  the body of `RegularKalman::predict` executes 337 instructions by hand count of its
  `llvm-objdump` listing (two 2×2×2 product loops, the 4-element `P += Q` loop, prologue
  and epilogue), and an execution trace of the section
  (`cpu CreateExecutionTracing "tr" @trace.txt PC`) has exactly 337 PCs in that range and
  407 in total, one less than the counter's 408 because the trace starts after the first
  instruction of `mk_begin`.
- **CI**: not set up. Renode publishes a Docker image (`antmicro/renode`) and a GitHub
  action; a job would need the two Rust targets and could run `run.sh` with
  `PROFILES=release` in a few minutes. The counts do not depend on the runner, so such a
  job could compare against a checked-in baseline.

## Caveats

- See the box at the top: these are instruction counts, not cycles. On the M4F, one
  `vdiv.f32` counts like one `vadd.f32` but costs 14 times as much; on the M0+, a load from
  flash with wait states counts like a register move.
- `f32` square roots go through `libm::sqrtf`, which has no Arm intrinsic and runs in
  software (about 80 instructions) even on the M4F, which has `vsqrt.f32`.
- The `opt-level` and the LTO settings change the counts a lot; compare like with like.
