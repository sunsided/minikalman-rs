# Instruction counts on Cortex-M (Renode)

A `no_std` firmware that runs minikalman's filter operations on an emulated Cortex-M4F and
Cortex-M0+ in [Renode](https://renode.io), and counts the instructions each operation
executes. The counts are deterministic: the same binary gives the same numbers on every run
and every machine, so a change to the kernels can be compared before and after without the
noise of wall-clock timings.

This directory is its own Cargo workspace. The minikalman workspace excludes it, so a normal
`cargo build` or `cargo test` of the repository never builds it. It holds no results apart
from the snapshot in the [time budget](#time-budget); produce your own (see
[Reproducing](#reproducing)).

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
  range–bearing radar for the extended filter (`f32` only). Two navigation workloads
  (`f32` only, see [Navigation workloads](#navigation-workloads)): a 6-state attitude
  error-state EKF and a 12-state GNSS/INS error-state EKF.
- **Element types**: `f32`, `I16F16` and `I32F32` (`fixed` crate). minikalman is built with
  `libm`, `fixed` and `unsafe`, without `alloc`; all buffers are on the stack. `unsafe` makes the
  `f32` square root use `vsqrt.f32` on the M4F (#174). The navigation workloads are two more
  binaries, `nav` (with `unsafe`, rows `f32/…`) and `nav-libm` (without it, rows
  `f32-libm/…`, M4F only: the M0+ has no FPU and uses `libm::sqrtf` either way).
- **Operations**, each per call:
  - `predict`, `correct_joseph` (the default) and `correct_standard` of the regular filter,
    and `step_joseph` (predict + correct in one measured section);
  - `observer_step`: predict + correct of the fixed-gain (Luenberger) observer;
  - `ar_<n>/predict_dense` and `ar_<n>/predict_companion` (n = 2, 4, 6, only with the
    firmware feature `companion`, i.e. `HEAD_FEATURES=companion`): `predict` with a dense
    companion `A` and `predict_companion` with its first column, on the same AR(n) model.
    The extra code moves other rows by up to a few percent (fat LTO lays out the image
    differently), so compare the other rows from a run without the feature;
  - `ud_predict`/`ud_correct` (Bierman/Thornton) and `sqrt_predict`/`sqrt_correct`
    (square-root covariance filter);
  - `ekf_radar_4x2/predict` and `correct_joseph`, the latter including the Jacobian and
    `h(x)` (`square_root`, `atan2f`);
  - `prim/*`: one `add`, `mul`, `div` and `sqrt` of each type, plus `copy` (only the operand
    loads and the store), to read the filter counts against.
- **Method** (`src/main.rs`, `renode/bench.py`):
  - Each operation runs between calls to the `#[inline(never)]` marker functions `mk_begin`
    and `mk_end(name_ptr, name_len, ok)`. Renode hooks both and logs the core's `ExecutedInstructions`
    in between. The marker overhead (`calib/empty`) is subtracted.
  - Setup, the model, 8 warm-up steps and the measurements are outside the markers. The
    filter references go through `black_box`, so nothing is specialized on the constant
    model.
  - 16 iterations per filter operation (`prim/*` and `calib/empty`: 4), in a realistic sequence: predict, then correct with a
    fresh measurement. The table shows the mean, and the range when iterations differ.
    Soft float, fixed-point division and the libm `sqrtf` have data-dependent paths, so the
    range is not always zero.
  - `ok` is the operation's status (`is_applied()`). A rejected correction would take a
    shortcut, so the table marks it ("not applied").
  - With `COUNT_CALLS=1`, calls to the soft-float and 64/128-bit helpers (`__aeabi_*`,
    `__*ti3`) and to `sqrtf`, `atan2f`, `sinf` and `cosf` inside each section are counted too. Helpers that
    the compiler inlines (the `fixed` crate's `I32F32` division, for example) are not
    visible there. Use it on the M4F only: with hooks on the soft-float helpers, the M0+
    run stalls in Renode 1.17.
  - The firmware paints the free stack and reports the unused part at the end. The run
    fails if the stack came within 256 B of the statics. The navigation workloads repaint
    it before their measured steps (reporting the free stack so far first; `run.sh` takes
    the smallest) and log their stack use and storage sizes with the hooked `mk_note`, as
    `NOTE <name> <bytes>` lines that `report.py` prints as a second table.
- Profiles: `release` (`opt-level = "s"`) and `o3` (`opt-level = 3`), both with fat LTO,
  one codegen unit and `panic = "abort"`.

## Navigation workloads

Two error-state filters of the shapes flight controllers run, in `src/nav_bench.rs`, with the
model (quaternion helpers and the 3×3 blocks of `examples/attitude_eskf.rs`, placed at an
offset) in `src/nav.rs`, which holds no benchmark code so the GNSS/INS example (#186) can
include it:

- **`att_6x3`**: attitude `δθ` and gyro bias `δb_g` (6 states). Gyro prediction with a
  jittered `Δt` around 2.5 ms (400 Hz), the 3-measurement accelerometer tilt update and a
  scalar heading update (`att_6x1`), both NIS-gated.
- **`ins_12x5`**: position, velocity, attitude error and gyro bias (12 states). IMU
  prediction (nominal strapdown integration plus `F` with the `−R̂·[f]×·Δt` block), a
  5-measurement GNSS update (horizontal position, 3-D velocity) and sequential scalar
  barometer and heading updates (`ins_12x1`), all gated.

The rows split one step into its parts:

- `predict_glue`: moving the nominal state (`q̂ ← q̂ ⊗ Exp(u)`, and for the INS `p̂`, `v̂`)
  and writing `F` and `Q` for this `Δt`;
- `predict`: `P ← F·P·Fᵀ + Q` (`predict_nonlinear`);
- `correct_*`: the update including its Jacobian, `h(x̂)` and, for the heading, `atan2f`;
- `reset`: injecting `δx` into the nominal state and `P ← G·P·Gᵀ`, after every correction;
- `step` (attitude): `predict_glue` + `predict` + `correct_accel` + `reset`, one 400 Hz
  cycle; `imu_step` (INS): `predict_glue` + `predict`.

The simulated truth rotates at about 1 rad/s and, for the INS, moves at constant velocity;
the filters start at the truth, and after the measured steps the firmware checks (outside
the markers) that the attitude error is below 1.1° and the INS position and velocity errors
below 1 m and 0.3 m/s, so a wrong model fails the run. The same model and data over 60 s on
the host (both biases converge to the true values, position NEES about 2.2 for 3 degrees of
freedom, from the diagonal of `P`) were checked when the workload was written.

`Exp(u)` uses a fourth-order series below 0.05 rad, exact in `f32` there, and `libm::sinf`/`cosf`
above. The series matters: `libm`'s `sinf` and `cosf` compute in `f64`, which the M4F's
single-precision FPU does not have. With them on every gyro sample, `att_6x3/predict_glue`
took 5 586 instructions (30 `__aeabi_dmul` and 16 `__aeabi_dadd` per call), with the series 925.

The memory table lists `size_of` of each storage type, the nominal state outside the filter,
`stack_per_op` (the deepest stack the measured steps use below the function that owns the
filter, i.e. what a step costs when the storage is a `static`) and `stack_total` (from the
top of RAM, including `main`, the storage on the stack and the deepest step).

No fixed-point rows: the gyro noise per step, `σ_g²·Δt = (3·10⁻⁴)²·0.0025 = 2.25·10⁻¹⁰`, is
below one `I32F32` LSB (`2⁻³² = 2.33·10⁻¹⁰`), and the bias random walk (`2.5·10⁻¹¹`) rounds to
zero, so `Q` vanishes and the attitude block of `P` keeps only a few significant bits; the
quaternion glue would also need fixed-point `sin`/`cos`. A fixed-point filter of this shape
needs scaled units (for example milliradians), which is a different model.

### Time budget

Counts at the revision of #185 (minikalman 0.8.1), mean per call; `M4F` is with the `unsafe`
feature (without it, `f32-libm/…`, the GNSS update takes 41 647 instead of 41 230 at `"s"`;
the other rows change by less than 100):

| operation | M4F `"s"` | M4F `3` | M0+ `"s"` | M0+ `3` |
|---|---:|---:|---:|---:|
| `att_6x3/predict_glue` | 925 | 232 | 14 372 | 13 579 |
| `att_6x3/predict` | 4 250 | 1 204 | 53 261 | 51 259 |
| `att_6x3/correct_accel` | 8 224 | 2 321 | 118 137 | 111 170 |
| `att_6x3/reset` | 1 572 | 354 | 29 962 | 23 968 |
| `att_6x1/correct_yaw` | 2 559 | 1 050 | 39 270 | 37 886 |
| `att_6x3/step` | 14 979 | 4 112 | 215 707 | 199 938 |
| `ins_12x5/predict_glue` | 2 420 | 757 | 28 681 | 27 037 |
| `ins_12x5/predict` | 27 212 | 9 759 | 343 256 | 335 166 |
| `ins_12x5/imu_step` | 29 632 | 10 516 | 371 879 | 362 149 |
| `ins_12x5/correct_gnss` | 41 230 | 12 100 | 628 863 | 599 811 |
| `ins_12x5/reset` | 2 732 | 857 | 52 023 | 40 544 |
| `ins_12x1/correct_baro` | 7 018 | 2 874 | 100 344 | 97 234 |
| `ins_12x1/correct_heading` | 7 252 | 3 039 | 114 530 | 111 122 |

Memory in bytes (`f32`, the same on both cores): attitude `FilterStorage<6>` 624,
`ObservationStorage<6, 3>` 564, `ObservationStorage<6, 1>` 260, nominal state 28, together
1 476; GNSS/INS `FilterStorage<12>` 2 400, `ObservationStorage<12, 5>` 1 876, two
`ObservationStorage<12, 1>` 788 each, nominal state 52, together 5 904. Stack per step
(`stack_per_op`): 464 to 924 B for the attitude filter, 676 to 1 484 B for the GNSS/INS
filter, depending on core and profile.

**The conversion is rough.** Instructions are not cycles (see the box at the top). Below, a
count becomes time at 1 to 2 cycles per instruction: on the M4F, `vdiv.f32` and `vsqrt.f32`
take 14 cycles and flash wait states (5 at 168 MHz on an STM32F4, mostly hidden by its ART
cache) add more; on the M0+, loads, stores and taken branches take 2 cycles, and the soft-float
helpers are already in the count. Measure on hardware before you commit to a rate.

| loop | M4F 168 MHz | M0+ 48 MHz | M0+ 133 MHz |
|---|---|---|---|
| attitude `step`, 400 Hz (2.5 ms) | `"s"` 89–178 µs (4–7 %), `3` 24–49 µs (1–2 %): **fits** | `"s"` 4.5–9.0 ms (180–360 %): **does not fit** | `"s"` 1.6–3.2 ms (65–130 %), `3` 1.5–3.0 ms (60–120 %): **does not fit reliably**; 200 Hz does |
| GNSS/INS: 400 Hz `imu_step` + 10 Hz GNSS, baro, heading (3 resets) | load `"s"` 7–15 %, `3` 3–5 %; worst tick (IMU + all updates) `"s"` 0.56–1.1 ms, `3` 0.19–0.37 ms: **fits** within one 2.5 ms period | load 330–660 %: **does not fit**; one GNSS update alone takes 13–26 ms | load 120–240 %: **does not fit**; about 100 Hz IMU (35–70 % load) does |

So on a Cortex-M4F the 12-state GNSS/INS filter at 400 Hz uses a small part of the CPU even at
`opt-level = "s"`, and `opt-level = 3` cuts the counts by about 3×. On a Cortex-M0+ (soft
float) the 6-state attitude filter needs about 200 k instructions per cycle, so 400 Hz is out
of reach at 48 MHz and marginal at 133 MHz; the GNSS/INS filter fits only at a lower IMU rate.

## Reproducing

Requirements: Renode on `PATH` (tested with 1.17.0), a stable Rust toolchain with
`rustup target add thumbv7em-none-eabihf thumbv6m-none-eabi`, and `python3`. `COUNT_CALLS=1`
also needs `llvm-nm` (or `rust-nm` from `cargo-binutils`).

```sh
benches/mcu/run.sh                                   # both cores, both profiles, all types
BASE=origin/main benches/mcu/run.sh                  # the same against minikalman at a revision
BASE=origin/main HEAD_FEATURES=companion benches/mcu/run.sh   # plus the head-only companion rows
TARGETS=thumbv7em-none-eabihf PROFILES=release TYPES=f32 benches/mcu/run.sh   # one binary
TARGETS=thumbv7em-none-eabihf TYPES=sqrt-check benches/mcu/run.sh   # f32 sqrt == libm::sqrtf?
TYPES="nav nav-libm" benches/mcu/run.sh             # only the navigation workloads
task bench:mcu
```

The `sqrt-check` type first compares minikalman's `f32` square root with `libm::sqrtf`, bit
for bit, on 65 558 inputs over the whole `u32` range on the target, and logs `sqrt_check/ok`
only if all match (a mismatch fails the run).

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
- `f32` square roots use `vsqrt.f32` on the M4F only because the firmware enables minikalman's
  `unsafe` feature. Without it they go through `libm::sqrtf`, which has no Arm intrinsic and
  runs in software (about 80 instructions). The M0+ has no FPU and always uses `libm::sqrtf`.
- The `opt-level` and the LTO settings change the counts a lot; compare like with like.
