//! A 12-state GNSS/INS error-state EKF that fits on the flight controller: IMU prediction at
//! 400 Hz, gated GNSS position and velocity at 10 Hz, barometer and heading as scalar updates,
//! all in `f32` on 5 904 bytes of storage known at compile time, without allocation.
//!
//! Position estimation on small UAVs often runs on a companion computer, in C++ with `double`,
//! because a full GNSS/INS filter is assumed too big for the flight controller. It is not. On a
//! Cortex-M4F at 168 MHz the loop below takes roughly 7.5 to 15 % of the CPU at
//! `opt-level = "s"` (see [Cost](#cost)). This example is a template for that filter. It builds
//! on the `attitude-eskf` example, which explains the error state on SO(3), the quaternion
//! helpers and the gating in more detail.
//!
//! ## The state
//!
//! The caller keeps the *nominal* state: position `p̂`, velocity `v̂`, attitude `q̂` and gyro
//! bias `b̂` (52 bytes, [`Nominal`]). The Kalman filter estimates only the small error
//!
//! ```text
//! δx = (δp, δv, δθ, δb_g)        12 states, 3 each
//! p = p̂ + δp    v = v̂ + δv    q = q̂ ⊗ Exp(δθ)    b_g = b̂ + δb_g
//! ```
//!
//! in a local world frame with x and y horizontal and z up, gravity `(0, 0, −9.81) m/s²`. `δθ` is
//! the body-frame (local) attitude error of Solà, "Quaternion kinematics for the error-state
//! Kalman filter" (arXiv:1711.02508). The quaternion and the 3×3 model blocks come from
//! `benches/mcu/src/nav.rs`, which the Renode benchmark measures (see [Cost](#cost)). The
//! example includes that file with `#[path]`, so both always run the same model.
//!
//! ## One step
//!
//! 1. **Predict (every IMU sample).** The gyro measures `ω_m = ω + b_g + n_g`, the
//!    accelerometer the specific force `f = Rᵀ·(a − g)`. With `Δt` from the timestamps (it
//!    varies) and `u = (ω_m − b̂)·Δt`, the nominal state is integrated (strapdown):
//!
//!    ```text
//!    a = R̂·f + g        p̂ ← p̂ + v̂·Δt + ½·a·Δt²        v̂ ← v̂ + a·Δt        q̂ ← q̂ ⊗ Exp(u)
//!    ```
//!
//!    To first order the error evolves with
//!
//!    ```text
//!        | I  Δt·I       0          0    |  δp
//!    F = | 0   I    −R̂·[f]×·Δt      0    |  δv
//!        | 0   0     Exp(u)ᵀ      −Δt·I  |  δθ
//!        | 0   0        0           I    |  δb_g
//!    ```
//!
//!    and `Q = diag(0, σ_a²·Δt·I, σ_g²·Δt·I, σ_b²·Δt·I)` for the accelerometer and gyro noise
//!    densities and the bias random walk. The `−R̂·[f]×` block is what makes the tilt
//!    observable from GNSS velocity: a tilt error turns gravity into a horizontal acceleration
//!    error, so the velocity drifts. `F` and `Q` depend on `Δt`, `R̂` and `f`, so both are
//!    written on every step. `predict_nonlinear` keeps `δx = 0`; only `P ← F·P·Fᵀ + Q` matters.
//! 2. **GNSS (10 Hz).** Five measurements: horizontal position and 3-D velocity, so `H` is a
//!    constant selection, written once. GNSS altitude is left out (it is the worst axis of a
//!    fix); the barometer measures the height.
//! 3. **Barometer (25 Hz) and heading (10 Hz), one scalar update each.** Several `correct`
//!    calls per step are fine, each with its own observation, rate, noise and gate. A
//!    one-measurement observation needs 788 bytes, and its gain is a division, not a matrix
//!    inversion. The heading `ψ = atan2(R₁₀, R₀₀)` has the attitude-dependent `H` of
//!    `attitude-eskf`, and its innovation is wrapped to `(−π, π]`.
//! 4. **Reset after every correction.** `p̂ ← p̂ + δp`, `v̂ ← v̂ + δv`, `q̂ ← q̂ ⊗ Exp(δθ)`,
//!    `b̂ ← b̂ + δb_g`, `δx ← 0`, and `P ← G·P·Gᵀ` with `G` the identity except
//!    `I − ½[δθ]×` on the attitude block. The next sequential update then linearizes at the
//!    corrected state. After a rejected correction `δx` is zero and the reset changes nothing.
//!
//! Every correction uses the Joseph form, the default of `correct_nonlinear`. It keeps `P`
//! symmetric and positive semi-definite to first order in rounding, which `f32` needs.
//!
//! ## Gating, outliers and the dropout
//!
//! Each update is gated by its NIS `yᵀ·S⁻¹·y` against a chi-square quantile (`set_gate`):
//! 99 % for the GNSS (`15.09` for five degrees of freedom) and the barometer, 99.9 % for the
//! heading, as in `attitude-eskf`. 2 % of the fixes are multipath outliers 30 m off; their NIS
//! is in the hundreds, and all of them are rejected. As expected for a 99 % gate, about 1 % of
//! the clean fixes are rejected too, which costs nothing at 10 Hz.
//!
//! The GNSS drops out for 10 s (a bridge or a tunnel). The IMU carries on, `P` grows, and the
//! position error grows to about a metre (up to 4 m over the seed sweep), inside the larger `P`. So the gate accepts the first
//! fix after the dropout. The gate trusts `P`: if the error outgrows `P`, because `Q` is too
//! small or the dropout is too long for the model, every fix is rejected, and the filter never
//! recovers (the lockout of `attitude-eskf`). A real system re-initializes position and
//! velocity from GNSS after a streak of rejected fixes. This example does not need that. Over the
//! seed sweep below, the first fix after the dropout had a mean NIS of 5.0 (outliers excluded),
//! as consistent as any other fix. It was rejected with 4 of 300 seeds: twice an outlier, twice
//! a clean fix just above the gate. Seed 71 rejected two clean fixes in a row and accepted the
//! third. The example checks that at most five are rejected before one is accepted.
//!
//! **Start-up.** The filter starts at rest from one sample of each sensor: roll and pitch from
//! the accelerometer, the heading from the heading reading, position and velocity from the
//! first fix, the altitude from the barometer, and a zero gyro bias. The prior covers the error
//! of each, so the gates accept the first updates. The attitude prior is the one of
//! `attitude-eskf` (world vertical rotated into the body frame).
//!
//! ## Simulation
//!
//! 120 s of a flight with known truth: 10 s at rest, accelerating to 12 m/s at 2 m/s², a 15 s
//! turn at 0.35 rad/s (4.2 m/s² centripetal acceleration), a climb of about 30 m, 15 s of S-turns, a
//! straight deceleration with the GNSS dropout from 70 s to 80 s, a descent to a stop and a
//! hover. While moving, roll and pitch swing by about ±14° and ±11°, and the yaw follows the
//! track. The IMU runs at about 400 Hz with ±0.25 ms timestamp jitter and 1 % dropped
//! samples. The gyro has a bias of about 1.3°/s with a random walk. All of it comes from one
//! seeded generator ([`SEED`], or the first command-line argument). The rate and the
//! acceleration are held over each IMU interval and the accelerometer reports the force in the
//! attitude at its start. So the truth is integrated exactly as the filter integrates its
//! nominal state, and the only approximations are the first-order error model and `f32`.
//!
//! With the default seed the example prints, after the first 10 s:
//!
//! ```text
//! position error  RMS  0.203 m     max  0.745 m
//! velocity error  RMS  0.048 m/s   max  0.270 m/s
//! attitude error  RMS  0.250°       max  0.539°
//! mean NEES (expected 3): δp 3.83, δv 3.06, δθ 3.30, δb_g 4.08
//!
//! Update     count  gated  mean NIS of accepted (expected just below)
//!   GNSS      1088     35  4.87 (5)
//!   baro      2968     35  0.92 (1)
//!   heading   1187      2  0.97 (1)
//! GNSS outliers rejected: 21 of 21; clean fixes rejected: 1.3 %
//! Fixes rejected after the dropout before the first accepted one: 0
//! ```
//!
//! The largest position and velocity errors typically occur at the end of the dropout. The NEES of each
//! 3-state block is computed against its 3×3 block of `P`; the NIS of each update is that of
//! the accepted corrections. The example checks the RMS and largest errors, the four mean NEES,
//! the mean NIS of each update, that every outlier is rejected, the rejection rate of the
//! clean fixes and the recovery after the dropout. If one fails, it exits with a non-zero status.
//!
//! **Seed sweep.** The bounds come from seeds 1 to 300, run with
//! `for s in $(seq 300); do cargo run -q --release --example gnss-ins --features std -- $s; done`.
//! All 300 passed. Each bound is documented with the range the sweep gave (for example, the
//! mean NEES of the four blocks had medians of 2.9 to 3.1 and ranged from 1.8 to 5.7). The
//! filter is consistent: there is no hold-off, extra noise or tuning beyond the true sensor
//! parameters.
//!
//! ## Cost
//!
//! Storage, `f32`, no allocation, as printed at the start and the same on every target:
//!
//! | buffer | bytes |
//! |---|---:|
//! | `FilterStorage<12>` | 2 400 |
//! | `ObservationStorage<12, 5>` (GNSS) | 1 876 |
//! | `ObservationStorage<12, 1>` (barometer, heading) | 2 × 788 |
//! | nominal state | 52 |
//! | together | 5 904 |
//!
//! The storage is checked at compile time against the `BYTES` of the same shape declared with
//! `filter!` (module `declared`). `filter!` generates regular filters only, so the extended
//! filter here is wired by hand from the storage types. On a microcontroller the storage is
//! a `static` (`Storage::new` is `const`). The stack per step is 676 to 1 484 bytes on top
//! (the Renode run below).
//!
//! The instruction counts of this model are measured in Renode by the `ins_12x5` workload of
//! `benches/mcu` (README sections "Navigation workloads" and
//! ["Time budget"](https://github.com/sunsided/minikalman-rs/blob/main/benches/mcu/README.md#time-budget)).
//! Mean per call, at the revision of #185:
//!
//! | operation | M4F `"s"` | M4F `3` | M0+ `"s"` |
//! |---|---:|---:|---:|
//! | `imu_step` (strapdown, `F`, `Q`, `P ← F·P·Fᵀ + Q`) | 29 632 | 10 516 | 371 879 |
//! | `correct_gnss` | 41 230 | 12 100 | 628 863 |
//! | `correct_baro` | 7 018 | 2 874 | 100 344 |
//! | `correct_heading` | 7 252 | 3 039 | 114 530 |
//! | `reset` | 2 732 | 857 | 52 023 |
//!
//! At this example's rates (400 Hz IMU, 10 Hz GNSS and heading, 25 Hz barometer, a reset after
//! each correction), that is about 12.6 M instructions per second at `"s"` and 4.5 M at
//! `opt-level = 3`. At 1 to 2 cycles per instruction this is 7.5 to 15 % (`"s"`) or 3 to 5 %
//! (`3`) of a Cortex-M4F at 168 MHz. Instructions are not cycles (division, square root and
//! flash wait states cost more), so measure on hardware before you commit to a rate. A
//! Cortex-M0+ (soft float) needs about 160 M instructions per second, so it does not keep up
//! at 400 Hz; about 100 Hz IMU fits at 133 MHz.
//!
//! **Trigonometry on the M4F.** `libm`'s `sinf` and `cosf` compute in `f64`, which the M4F's
//! single-precision FPU does not have, so they run in soft float, at thousands of instructions
//! per pair. With them on every gyro sample, the attitude prediction of #185 cost 5 586
//! instructions instead of 925. The cheap alternative is used here: `Exp(u)` uses a
//! fourth-order series below 0.05 rad, which is exact in `f32` there (`nav::Quat::exp`). At
//! 400 Hz that covers rates up to 20 rad/s (10 rad/s across a dropped sample), so the hot path never calls `sinf`/`cosf`. Only
//! the start-up and the simulation use them. The heading's `atan2f` is cheaper: the whole
//! heading update costs about 230 instructions more than the barometer update.
//!
//! ## When `f32` is not enough
//!
//! `f32` has a 24-bit mantissa (`ε ≈ 1.2·10⁻⁷`). Here the per-step `Q` is about `10⁻⁴` of the
//! diagonal of `P` it is added to, three orders of magnitude above `ε`, and the Joseph form
//! keeps `P` symmetric. Use `f64` (fast on a Cortex-M7 with the double-precision FPU) or the
//! factored filters, the UD filter (`minikalman::ud`, Bierman/Thornton) or the square-root
//! filter (`minikalman::square_root`), when that margin shrinks. That happens with a
//! tactical-grade IMU at kHz rates (tiny `Q` per step), with cm-level RTK fixes against a `P`
//! that also holds metre-level states, and with long missions where the bias variances become
//! tiny. The factored filters keep `P` positive definite by construction and carry about twice
//! the precision of `P` in the same word size. The warning signs are a NEES that drifts away
//! from its expected value, negative diagonal entries of `P`, or gates that start to reject
//! everything.
//!
//! ## Not modelled
//!
//! - **Delayed and out-of-sequence GNSS.** A real fix is 50 to 200 ms old when it arrives.
//!   The usual remedies keep a short history of nominal states and either re-run the IMU
//!   steps from the fix time, or correct the delayed state and carry the correction forward.
//!   This is out of scope here; the simulated fixes arrive at the time they are valid.
//! - The accelerometer bias (a 15-state filter), the GNSS antenna lever arm, Earth rotation and
//!   curvature (the world frame is flat and fixed), time-correlated GNSS and barometer errors,
//!   magnetometer disturbances (the heading is a clean external yaw), coning and sculling, and
//!   the yaw singularity at ±90° pitch.
#![forbid(unsafe_code)]

use std::f32::consts::PI;
use std::process::ExitCode;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use minikalman::angles::wrap_pi;
use minikalman::extended::{
    BorrowedKalman, BorrowedObservation, FilterStorage, ObservationStorage,
};
use minikalman::prelude::*;

/// The quaternion helpers and 3×3 model blocks, shared with the Renode benchmark.
///
/// The file lives in `benches/mcu`, whose `ins_12x5` workload measures exactly this model on
/// a Cortex-M4F and a Cortex-M0+, so the example and the instruction counts cannot drift apart.
#[path = "../../../benches/mcu/src/nav.rs"]
#[allow(dead_code)]
mod nav;

use nav::{
    GRAVITY, INS_ATTITUDE, INS_BIAS, INS_POSITION, INS_VELOCITY, Mat3, Quat, Vec3,
    attitude_transition, cross, dot, inject_attitude, mat_mul, norm, read3, reset_covariance,
    reset_jacobian, skew, transpose_mul, velocity_attitude_transition, write_block, write_diagonal,
    yaw, yaw_jacobian,
};

/// Number of error states: `δp`, `δv`, `δθ` and `δb_g`, three each.
const STATES: usize = 12;

/// Number of GNSS measurements: horizontal position (2) and 3-D velocity (3).
const GNSS_OBSERVATIONS: usize = 5;

/// Simulated duration in microseconds (120 s).
const DURATION_US: u64 = 120_000_000;

/// Nominal IMU sample period in microseconds (400 Hz).
const IMU_PERIOD_US: u64 = 2_500;

/// Largest timestamp jitter of an IMU sample in microseconds, in either direction.
const IMU_JITTER_US: u64 = 250;

/// Probability that an IMU sample is dropped, which doubles the next `Δt`.
const DROP_PROBABILITY: f64 = 0.01;

/// A GNSS fix arrives on every this many IMU samples (10 Hz).
const GNSS_EVERY: usize = 40;

/// A barometer reading arrives on every this many IMU samples (25 Hz).
const BARO_EVERY: usize = 16;

/// A heading arrives on every this many IMU samples (10 Hz), half a period after the GNSS fix.
const HEADING_EVERY: usize = 40;

/// Print the progress table every this many microseconds.
const PRINT_EVERY_US: u64 = 10_000_000;

/// Gyro white noise density `σ_g` in rad/s/√Hz.
const GYRO_NOISE_DENSITY: f32 = 3e-4;

/// Gyro bias random walk `σ_b` in rad/s/√s.
const GYRO_BIAS_WALK: f32 = 1e-4;

/// Accelerometer white noise density `σ_a` in m/s²/√Hz (the velocity random walk).
const ACCEL_NOISE_DENSITY: f32 = 2e-3;

/// The true gyro bias at start-up in rad/s.
const INITIAL_BIAS: Vec3 = [0.01, -0.02, 0.005];

/// The true attitude at start-up as (roll, pitch, yaw) in radians.
const INITIAL_EULER: Vec3 = [2.0 * PI / 180.0, -3.0 * PI / 180.0, 40.0 * PI / 180.0];

/// Standard deviation of each horizontal GNSS position axis in m.
const GNSS_POSITION_STD: f32 = 1.0;

/// Standard deviation of each GNSS velocity axis in m/s.
const GNSS_VELOCITY_STD: f32 = 0.1;

/// Probability that a GNSS fix is an outlier (multipath), outside start-up and the dropout.
const OUTLIER_PROBABILITY: f64 = 0.02;

/// Horizontal position offset of an outlier fix in m, in a random direction.
const OUTLIER_OFFSET: f32 = 30.0;

/// The GNSS dropout (a bridge or a tunnel), start and end in microseconds.
const DROPOUT_US: (u64, u64) = (70_000_000, 80_000_000);

/// Standard deviation of the barometric altitude in m.
const BARO_STD: f32 = 0.3;

/// Standard deviation of the heading in radians (2°).
const HEADING_STD: f32 = 2.0 * PI / 180.0;

/// Prior standard deviation of roll and pitch after levelling, in radians (2°).
const PRIOR_TILT_STD: f32 = 2.0 * PI / 180.0;

/// Prior standard deviation of the heading after the first reading, in radians (5°).
const PRIOR_YAW_STD: f32 = 5.0 * PI / 180.0;

/// Prior standard deviation of each velocity axis after the first fix, in m/s.
const PRIOR_VELOCITY_STD: f32 = 0.3;

/// Prior standard deviation of each gyro bias axis in rad/s.
const PRIOR_BIAS_STD: f32 = 0.05;

/// GNSS gate: the 99 % chi-square quantile for five degrees of freedom.
const GNSS_GATE: f32 = 15.09;

/// Barometer gate: the 99 % chi-square quantile for one degree of freedom.
const BARO_GATE: f32 = 6.63;

/// Heading gate: the 99.9 % chi-square quantile for one degree of freedom, as in `attitude-eskf`.
const HEADING_GATE: f32 = 10.83;

/// Statistics (errors, NEES) start after this many microseconds.
const SETTLE_US: u64 = 10_000_000;

/// Largest RMS errors after [`SETTLE_US`] of position (m), velocity (m/s) and attitude (°).
///
/// This and the bounds below leave margin over seeds 1 to 300, which all passed (see the
/// module docs): RMS at most 0.56 m, 0.15 m/s and 0.37°.
const MAX_RMS: [f32; 3] = [1.0, 0.3, 0.75];

/// Largest errors after [`SETTLE_US`] of position (m), velocity (m/s) and attitude (°).
///
/// 300 seeds: at most 3.8 m, 0.85 m/s (typically at the end of the GNSS dropout) and 1.39°.
const MAX_ERROR: [f32; 3] = [6.0, 1.5, 2.5];

/// Accepted range of the mean NEES of each 3-state block (expected 3).
///
/// One run's time average is autocorrelated, so it scatters: over 300 seeds the four means
/// had medians of 2.9 to 3.1 and ranged from 1.8 to 5.7.
const NEES_RANGE: (f32, f32) = (1.0, 8.0);

/// Accepted range of the mean NIS of the accepted GNSS updates (300 seeds: 4.65 to 5.12).
///
/// Just below the 5 degrees of freedom, since the gate cuts the upper 1 % tail.
const GNSS_NIS_RANGE: (f32, f32) = (4.0, 6.0);

/// Accepted range of the mean NIS of the accepted barometer and heading updates.
///
/// 300 seeds: 0.87 to 0.98 (barometer) and 0.87 to 1.09 (heading), for one degree of freedom.
const SCALAR_NIS_RANGE: (f32, f32) = (0.7, 1.3);

/// Most fixes the gate may reject after the dropout before it accepts one.
///
/// Over 300 seeds, at most 2 (seed 71: two clean fixes just above the gate). A lockout would
/// reject every fix.
const MAX_REJECTED_AFTER_DROPOUT: u32 = 5;

/// Largest share of clean GNSS fixes the gate may reject (300 seeds: 0.3 to 2.0 %).
const MAX_CLEAN_REJECTION: f32 = 0.04;

/// Fixed seed so CI runs are reproducible; the first command-line argument overrides it.
const SEED: u64 = 186;

/// Runs the self-checks, the simulation, and checks the filter against the bounds.
fn main() -> ExitCode {
    let seed = std::env::args()
        .nth(1)
        .map_or(SEED, |s| s.parse().expect("the seed is an integer"));

    check_helpers();
    report_sizes();

    let mut sim = Simulation::new(seed);
    let mut storage = Storage::new();
    let mut ins = storage.build(sim.start_up());
    let mut stats = Stats::default();
    let mut next_print = 0;

    println!("\nSeed {seed}");
    println!("     t  phase        |δp| m  |δv| m/s  |δθ| °   σ_p m  GNSS used/gated");
    while let Some(sample) = sim.step() {
        run_step(&mut ins, &sample, &mut stats);

        if sample.time_us >= next_print {
            print_row(&sample, &ins, &stats);
            next_print += PRINT_EVERY_US;
        }
    }

    stats.report();
    check(&stats)
}

/// Feeds one IMU sample and the aiding measurements that came with it to the filter.
fn run_step(ins: &mut Ins<'_>, sample: &Sample, stats: &mut Stats) {
    ins.predict(sample.gyro, sample.accel, sample.dt);

    if let Some(fix) = &sample.gnss {
        let status = ins.correct_gnss(&fix.z);
        stats.gnss.record(status, ins.gnss.nis());
        stats.record_outlier(fix.outlier, status);

        if sample.time_us >= DROPOUT_US.1 && !stats.recovered {
            stats.recovered = status.is_applied();
            stats.rejected_after_dropout += u32::from(!stats.recovered);
        }
    }

    if let Some(altitude) = sample.baro {
        let status = ins.correct_baro(altitude);
        stats.baro.record(status, ins.baro.nis());
    }

    if let Some(heading) = sample.heading {
        let status = ins.correct_heading(heading);
        stats.heading.record(status, ins.heading.nis());
    }

    if sample.time_us >= SETTLE_US {
        stats.record_errors(&sample.truth, ins);
    }
}

/// The same storage declared with `filter!`, only for its `BYTES`.
///
/// `filter!` generates regular filters only (`predict_nonlinear` and the extended
/// observations are wired by hand), but its storage has the same shape, so the compile-time
/// check in [`report_sizes`] pins the hand-wired [`Storage`] to the declaration.
#[allow(dead_code)]
mod declared {
    minikalman::filter! {
        /// The error state `δx = (δp, δv, δθ, δb_g)`.
        pub state ErrorState(f32) { px, py, pz, vx, vy, vz, ax, ay, az, bx, by, bz }

        /// GNSS: horizontal position and 3-D velocity.
        pub observation gnss: GnssFix for ErrorState { px, py, vx, vy, vz } {
            H = [px, py, vx, vy, vz];
        }

        /// Barometric altitude.
        pub observation baro: BaroFix for ErrorState { pz } {
            H = [pz];
        }

        /// Heading; its `H` depends on the attitude and is written on every update.
        pub observation heading: HeadingFix for ErrorState { yaw } {}

        /// The error-state filter.
        pub kalman ins for ErrorState {}

        /// Everything in one value.
        pub system shape { ins, gnss, baro, heading }
    }
}

/// Prints the size of every buffer the filter needs: the storage and the nominal state.
fn report_sizes() {
    use core::mem::size_of;
    const _: () = assert!(declared::shape::BYTES == size_of::<Storage>());

    let filter = size_of::<FilterStorage<STATES, f32>>();
    let gnss = size_of::<ObservationStorage<STATES, GNSS_OBSERVATIONS, f32>>();
    let scalar = size_of::<ObservationStorage<STATES, 1, f32>>();
    let nominal = size_of::<Nominal>();

    println!("Storage (f32, no allocation):");
    println!("  FilterStorage<12>          {filter:5} bytes");
    println!("  ObservationStorage<12, 5>  {gnss:5} bytes (GNSS)");
    println!("  ObservationStorage<12, 1>  {scalar:5} bytes (barometer)");
    println!("  ObservationStorage<12, 1>  {scalar:5} bytes (heading)");
    println!(
        "  Storage total              {:5} bytes (= filter! shape::BYTES)",
        size_of::<Storage>()
    );
    println!("  Nominal state              {nominal:5} bytes (p̂, v̂, q̂, b̂)");
    println!(
        "  Together                   {:5} bytes",
        size_of::<Storage>() + nominal
    );
}

/// The inline buffers of the filter and its three observations.
struct Storage {
    /// `F`, `δx`, `P`, `Q` and the filter temporaries.
    filter: FilterStorage<STATES, f32>,
    /// `H`, `z`, `R`, `y`, `S`, `K` and temporaries of the GNSS update.
    gnss: ObservationStorage<STATES, GNSS_OBSERVATIONS, f32>,
    /// The same for the barometer update.
    baro: ObservationStorage<STATES, 1, f32>,
    /// The same for the heading update.
    heading: ObservationStorage<STATES, 1, f32>,
}

impl Storage {
    /// Creates zeroed storage; `const`, so it can be a `static` on a microcontroller.
    const fn new() -> Self {
        Self {
            filter: FilterStorage::new(0.0),
            gnss: ObservationStorage::new(0.0),
            baro: ObservationStorage::new(0.0),
            heading: ObservationStorage::new(0.0),
        }
    }

    /// Wires the filter at the start-up estimate with its prior, noise, constant `H` and gates.
    fn build(&mut self, nominal: Nominal) -> Ins<'_> {
        let mut filter = self.filter.build();
        let mut gnss = self.gnss.build();
        let mut baro = self.baro.build();
        let mut heading = self.heading.build();

        let attitude_prior = attitude_prior(&nominal.attitude);
        filter.estimate_covariance_mut().apply(|p| {
            (0..2).for_each(|i| p.set_at(i, i, GNSS_POSITION_STD * GNSS_POSITION_STD));
            p.set_at(INS_POSITION + 2, INS_POSITION + 2, BARO_STD * BARO_STD);
            write_diagonal(p, INS_VELOCITY, PRIOR_VELOCITY_STD * PRIOR_VELOCITY_STD);
            write_block(p, INS_ATTITUDE, INS_ATTITUDE, &attitude_prior);
            write_diagonal(p, INS_BIAS, PRIOR_BIAS_STD * PRIOR_BIAS_STD);
        });

        gnss.observation_jacobian_matrix_mut().apply(|h| {
            h.set_at(0, INS_POSITION, 1.0);
            h.set_at(1, INS_POSITION + 1, 1.0);
            (0..3).for_each(|i| h.set_at(2 + i, INS_VELOCITY + i, 1.0));
        });
        gnss.measurement_noise_covariance_mut().apply(|r| {
            (0..2).for_each(|i| r.set_at(i, i, GNSS_POSITION_STD * GNSS_POSITION_STD));
            (2..5).for_each(|i| r.set_at(i, i, GNSS_VELOCITY_STD * GNSS_VELOCITY_STD));
        });
        gnss.set_gate(Some(GNSS_GATE));

        baro.observation_jacobian_matrix_mut()
            .apply(|h| h.set_at(0, INS_POSITION + 2, 1.0));
        baro.measurement_noise_covariance_mut()
            .apply(|r| r.set_at(0, 0, BARO_STD * BARO_STD));
        baro.set_gate(Some(BARO_GATE));

        heading
            .measurement_noise_covariance_mut()
            .apply(|r| r.set_at(0, 0, HEADING_STD * HEADING_STD));
        heading.set_gate(Some(HEADING_GATE));

        Ins {
            filter,
            gnss,
            baro,
            heading,
            nominal,
        }
    }
}

/// Returns the prior covariance of `δθ` for the levelled tilt and the initial heading.
///
/// The uncertainty is about the *world* vertical, so it is `R̂ᵀ·diag(σ_t², σ_t², σ_ψ²)·R̂` in the
/// body frame of `δθ` (see `attitude-eskf`).
fn attitude_prior(attitude: &Quat) -> Mat3 {
    let r = attitude.rotation();
    let variance = [PRIOR_TILT_STD, PRIOR_TILT_STD, PRIOR_YAW_STD].map(|s| s * s);

    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..3).map(|k| r[k][i] * variance[k] * r[k][j]).sum())
    })
}

/// The nominal (full) state, kept by the caller outside the filter.
#[derive(Clone, Copy, Debug)]
struct Nominal {
    /// Position `p̂` in m, world frame (x, y horizontal, z up).
    position: Vec3,
    /// Velocity `v̂` in m/s, world frame.
    velocity: Vec3,
    /// Attitude `q̂`, body to world.
    attitude: Quat,
    /// Gyro bias `b̂` in rad/s.
    bias: Vec3,
}

/// The GNSS/INS error-state filter: `δx = (δp, δv, δθ, δb_g)` in the filter, the nominal outside.
struct Ins<'a> {
    /// The error-state filter. `δx` is zero between steps.
    filter: BorrowedKalman<'a, STATES, f32>,
    /// The GNSS position and velocity update (`H` constant, set once).
    gnss: BorrowedObservation<'a, STATES, GNSS_OBSERVATIONS, f32>,
    /// The barometric altitude update (`H` constant, set once).
    baro: BorrowedObservation<'a, STATES, 1, f32>,
    /// The heading update (`H` written on every update).
    heading: BorrowedObservation<'a, STATES, 1, f32>,
    /// The nominal state.
    nominal: Nominal,
}

impl Ins<'_> {
    /// Integrates one IMU sample over `dt` seconds into the nominal state and propagates `P`.
    ///
    /// `gyro` is the body rate in rad/s, `accel` the specific force in m/s², both held over `dt`.
    fn predict(&mut self, gyro: Vec3, accel: Vec3, dt: f32) {
        let r = self.nominal.attitude.rotation();
        let u: Vec3 = core::array::from_fn(|i| (gyro[i] - self.nominal.bias[i]) * dt);
        self.integrate_nominal(&r, u, accel, dt);

        let rotation = attitude_transition(u);
        let velocity_attitude = velocity_attitude_transition(&r, accel, dt);
        self.filter.state_transition_jacobian_mut().apply(|f| {
            f.make_identity();
            (0..3).for_each(|i| {
                f.set_at(INS_POSITION + i, INS_VELOCITY + i, dt);
                f.set_at(INS_ATTITUDE + i, INS_BIAS + i, -dt);
            });
            write_block(f, INS_VELOCITY, INS_ATTITUDE, &velocity_attitude);
            write_block(f, INS_ATTITUDE, INS_ATTITUDE, &rotation);
        });

        self.filter.direct_process_noise_mut().apply(|q| {
            write_diagonal(
                q,
                INS_VELOCITY,
                ACCEL_NOISE_DENSITY * ACCEL_NOISE_DENSITY * dt,
            );
            write_diagonal(
                q,
                INS_ATTITUDE,
                GYRO_NOISE_DENSITY * GYRO_NOISE_DENSITY * dt,
            );
            write_diagonal(q, INS_BIAS, GYRO_BIAS_WALK * GYRO_BIAS_WALK * dt);
        });

        // The error stays zero; only P ← F·P·Fᵀ + Q matters.
        self.filter
            .predict_nonlinear(|_, next| (0..STATES).for_each(|i| next.set_row(i, 0.0)));
    }

    /// Strapdown integration: `p̂`, `v̂` with `a = R̂·f − g`, and `q̂ ← q̂ ⊗ Exp(u)`.
    fn integrate_nominal(&mut self, r: &Mat3, u: Vec3, accel: Vec3, dt: f32) {
        let n = &mut self.nominal;
        let force = mat_mul(r, accel);
        let a = [force[0], force[1], force[2] - GRAVITY];

        n.position =
            core::array::from_fn(|i| n.position[i] + n.velocity[i] * dt + 0.5 * a[i] * dt * dt);
        n.velocity = core::array::from_fn(|i| n.velocity[i] + a[i] * dt);
        n.attitude = n.attitude.mul(&Quat::exp(u)).normalized();
    }

    /// Corrects with a GNSS fix (horizontal position in m, velocity in m/s), then resets.
    fn correct_gnss(&mut self, z: &[f32; GNSS_OBSERVATIONS]) -> CorrectionStatus {
        let (p, v) = (self.nominal.position, self.nominal.velocity);
        let predicted = [p[0], p[1], v[0], v[1], v[2]];
        self.gnss
            .measurement_vector_mut()
            .apply(|m| (0..GNSS_OBSERVATIONS).for_each(|i| m.set_row(i, z[i])));

        let status = self.filter.correct_nonlinear(&mut self.gnss, |_, h| {
            (0..GNSS_OBSERVATIONS).for_each(|i| h.set_row(i, predicted[i]));
        });

        self.reset();
        status
    }

    /// Corrects with a barometric altitude in m, then resets.
    fn correct_baro(&mut self, altitude: f32) -> CorrectionStatus {
        let predicted = self.nominal.position[2];
        self.baro
            .measurement_vector_mut()
            .apply(|m| m.set_row(0, altitude));

        let status = self
            .filter
            .correct_nonlinear(&mut self.baro, |_, h| h.set_row(0, predicted));

        self.reset();
        status
    }

    /// Corrects with a heading in radians, then resets. The innovation is wrapped to `(−π, π]`.
    fn correct_heading(&mut self, psi: f32) -> CorrectionStatus {
        let rotation = self.nominal.attitude.rotation();
        let jacobian = yaw_jacobian(&rotation);
        self.heading.observation_jacobian_matrix_mut().apply(|h| {
            (0..3).for_each(|i| h.set_at(0, INS_ATTITUDE + i, jacobian[i]));
        });
        self.heading
            .measurement_vector_mut()
            .apply(|z| z.set_row(0, psi));

        let status = self.filter.correct_nonlinear_with(
            &mut self.heading,
            |_, h| h.set_row(0, yaw(&rotation)),
            |y| y[0] = wrap_pi(y[0]),
        );

        self.reset();
        status
    }

    /// Moves `δx` into the nominal state, zeroes it and applies `P ← G·P·Gᵀ`.
    ///
    /// After a rejected correction the error is zero and this changes nothing.
    fn reset(&mut self) {
        let x = self.filter.state_vector();
        let (dp, dv) = (read3(x, INS_POSITION), read3(x, INS_VELOCITY));
        let (dtheta, dbias) = (read3(x, INS_ATTITUDE), read3(x, INS_BIAS));

        let n = &mut self.nominal;
        n.position = core::array::from_fn(|i| n.position[i] + dp[i]);
        n.velocity = core::array::from_fn(|i| n.velocity[i] + dv[i]);
        n.attitude = inject_attitude(&n.attitude, dtheta);
        n.bias = core::array::from_fn(|i| n.bias[i] + dbias[i]);

        let g = reset_jacobian(dtheta);
        self.filter
            .estimate_covariance_mut()
            .apply(|p| reset_covariance(p, INS_ATTITUDE, &g));
        self.filter
            .state_vector_mut()
            .apply(|v| (0..STATES).for_each(|i| v.set_row(i, 0.0)));
    }

    /// Returns the errors `truth − estimate` of position, velocity, attitude and bias.
    fn errors(&self, truth: &Truth) -> [Vec3; 4] {
        let n = &self.nominal;
        let diff = |a: Vec3, b: Vec3| -> Vec3 { core::array::from_fn(|i| a[i] - b[i]) };

        [
            diff(truth.position, n.position),
            diff(truth.velocity, n.velocity),
            attitude_error(&n.attitude, &truth.attitude),
            diff(truth.bias, n.bias),
        ]
    }

    /// Returns the NEES of each error block against its 3×3 block of `P`.
    fn nees(&self, truth: &Truth) -> [f32; 4] {
        let p = self.filter.estimate_covariance();
        let errors = self.errors(truth);
        let offsets = [INS_POSITION, INS_VELOCITY, INS_ATTITUDE, INS_BIAS];

        core::array::from_fn(|k| {
            let o = offsets[k];
            let block: Mat3 =
                core::array::from_fn(|r| core::array::from_fn(|c| p.get_at(o + r, o + c)));
            mahalanobis(&block, errors[k])
        })
    }

    /// Returns the standard deviation of the 3-D position from the trace of its block of `P`.
    fn position_std(&self) -> f32 {
        let p = self.filter.estimate_covariance();
        (0..3).map(|i| p.get_at(i, i)).sum::<f32>().sqrt()
    }
}

/// Returns the attitude error `δθ` with `truth = estimate ⊗ Exp(δθ)`, to first order.
fn attitude_error(estimate: &Quat, truth: &Quat) -> Vec3 {
    let q = estimate.conjugate().mul(truth);
    let sign = if q.w < 0.0 { -2.0 } else { 2.0 };
    q.v.map(|x| sign * x)
}

/// Returns `eᵀ·P⁻¹·e` for a symmetric positive definite 3×3 `P`, via its adjugate.
fn mahalanobis(p: &Mat3, e: Vec3) -> f32 {
    let adj: Mat3 = core::array::from_fn(|r| {
        core::array::from_fn(|c| {
            // Cofactor (c, r) of P, i.e. the adjugate's (r, c) entry.
            let (r1, r2) = ((c + 1) % 3, (c + 2) % 3);
            let (c1, c2) = ((r + 1) % 3, (r + 2) % 3);
            p[r1][c1] * p[r2][c2] - p[r1][c2] * p[r2][c1]
        })
    });
    let det = dot(p[0], [adj[0][0], adj[1][0], adj[2][0]]);

    dot(e, mat_mul(&adj, e)) / det
}

/// The segments of the simulated flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Standing on the ground: the filter learns the gyro bias.
    Rest,
    /// Accelerating forward at 2 m/s² to 12 m/s, then cruising.
    Accelerate,
    /// A 15 s turn at 0.35 rad/s: 4.2 m/s² centripetal acceleration.
    Turn,
    /// A climb of about 30 m at 3 m/s.
    Climb,
    /// S-turns, ±0.5 rad/s.
    Weave,
    /// Straight and slowing down; the GNSS drops out for 10 s.
    Dropout,
    /// Descending and slowing to a stop.
    Descend,
    /// Hovering at the end.
    Hover,
}

impl Phase {
    /// Returns the phase at `t` seconds and the time since it began.
    fn at(t: f32) -> (Phase, f32) {
        match t {
            t if t < 10.0 => (Phase::Rest, t),
            t if t < 20.0 => (Phase::Accelerate, t - 10.0),
            t if t < 35.0 => (Phase::Turn, t - 20.0),
            t if t < 50.0 => (Phase::Climb, t - 35.0),
            t if t < 65.0 => (Phase::Weave, t - 50.0),
            t if t < 85.0 => (Phase::Dropout, t - 65.0),
            t if t < 100.0 => (Phase::Descend, t - 85.0),
            t => (Phase::Hover, t - 100.0),
        }
    }

    /// Returns a short name for the table.
    fn name(self) -> &'static str {
        match self {
            Phase::Rest => "rest",
            Phase::Accelerate => "accelerate",
            Phase::Turn => "turn",
            Phase::Climb => "climb",
            Phase::Weave => "weave",
            Phase::Dropout => "dropout",
            Phase::Descend => "descend",
            Phase::Hover => "hover",
        }
    }

    /// Returns the commanded (speed change in m/s², track rate in rad/s, vertical acceleration
    /// in m/s²) `s` seconds into the phase.
    fn command(self, s: f32) -> Vec3 {
        let pulse =
            |from: f32, to: f32, value: f32| if (from..to).contains(&s) { value } else { 0.0 };

        match self {
            Phase::Rest | Phase::Hover => [0.0; 3],
            Phase::Accelerate => [pulse(0.0, 6.0, 2.0), 0.0, 0.0],
            Phase::Turn => [0.0, 0.35, 0.0],
            Phase::Climb => [0.0, 0.0, pulse(0.0, 2.0, 1.5) - pulse(11.0, 13.0, 1.5)],
            Phase::Weave => [0.0, 0.5 * (2.0 * PI * 0.1 * s).sin(), 0.0],
            Phase::Dropout => [pulse(5.0, 20.0, -0.5), 0.0, 0.0],
            Phase::Descend => [-0.3, 0.0, pulse(0.0, 2.0, -1.0) + pulse(10.0, 12.0, 1.0)],
        }
    }

    /// Returns the world-frame angular rate in rad/s `t` seconds into the flight for the track
    /// rate `track_rate`: roll and pitch wobble while moving, yaw follows the track.
    fn world_rate(self, t: f32, track_rate: f32) -> Vec3 {
        if matches!(self, Phase::Rest | Phase::Hover) {
            return [0.0; 3];
        }

        // A cosine rate integrates to a sine angle: about ±14° of roll and ±11° of pitch.
        let wave = |amplitude: f32, hz: f32| amplitude * (2.0 * PI * hz * t).cos();
        [wave(0.3, 0.2), wave(0.2, 0.17), track_rate]
    }
}

/// The true state of the simulated vehicle.
#[derive(Clone, Copy, Debug)]
struct Truth {
    /// Position in m, world frame.
    position: Vec3,
    /// Velocity in m/s, world frame.
    velocity: Vec3,
    /// Attitude, body to world.
    attitude: Quat,
    /// Gyro bias in rad/s.
    bias: Vec3,
}

/// One GNSS fix.
struct GnssFix {
    /// Horizontal position (m) and 3-D velocity (m/s).
    z: [f32; GNSS_OBSERVATIONS],
    /// Whether the position was offset by [`OUTLIER_OFFSET`].
    outlier: bool,
}

/// One IMU sample with the aiding measurements that arrived with it.
struct Sample {
    /// Timestamp in microseconds.
    time_us: u64,
    /// Time since the previous IMU sample in seconds.
    dt: f32,
    /// Gyro reading `ω + b + n` in rad/s, held over the last `dt`.
    gyro: Vec3,
    /// Accelerometer reading `Rᵀ·(a + g) + n` in m/s², held over the last `dt`.
    accel: Vec3,
    /// A GNSS fix on every [`GNSS_EVERY`]-th sample, except during the dropout.
    gnss: Option<GnssFix>,
    /// A barometric altitude in m on every [`BARO_EVERY`]-th sample.
    baro: Option<f32>,
    /// A heading in radians on every [`HEADING_EVERY`]-th sample.
    heading: Option<f32>,
    /// The truth at this time.
    truth: Truth,
}

/// The simulated vehicle and its sensors, all noise from one seeded generator.
struct Simulation {
    /// The noise source.
    rng: StdRng,
    /// The current truth.
    truth: Truth,
    /// The commanded horizontal speed in m/s.
    speed: f32,
    /// The commanded horizontal direction of travel in radians.
    track: f32,
    /// The timestamp of the last IMU sample in microseconds.
    time_us: u64,
    /// The number of IMU samples so far.
    count: usize,
}

impl Simulation {
    /// Starts at rest at the origin in the initial attitude with the initial bias.
    fn new(seed: u64) -> Self {
        let attitude = from_euler(INITIAL_EULER);
        Self {
            rng: StdRng::seed_from_u64(seed),
            truth: Truth {
                position: [0.0; 3],
                velocity: [0.0; 3],
                attitude,
                bias: INITIAL_BIAS,
            },
            speed: 0.0,
            track: INITIAL_EULER[2],
            time_us: 0,
            count: 0,
        }
    }

    /// Returns the nominal state the filter starts from, from the first readings at rest.
    ///
    /// Roll and pitch come from one accelerometer sample, the heading from one heading
    /// reading, position and velocity from one GNSS fix and the altitude from the barometer.
    /// The gyro bias starts at zero. A start close to the truth keeps `δx` small, which every
    /// first-order step (`F`, `H`, the reset `G`, the gates) relies on.
    fn start_up(&mut self) -> Nominal {
        let f = self.accel_sample(
            transpose_mul(&self.truth.attitude.rotation(), [0.0, 0.0, GRAVITY]),
            2.5e-3,
        ); // one 400 Hz sample
        let roll = f[1].atan2(f[2]);
        let pitch = (-f[0]).atan2(f[1].hypot(f[2]));
        let fix = self.gnss_sample(false);

        Nominal {
            position: [fix[0], fix[1], self.baro_sample()],
            velocity: [fix[2], fix[3], fix[4]],
            attitude: from_euler([roll, pitch, self.heading_sample()]),
            bias: [0.0; 3],
        }
    }

    /// Advances to the next (jittered, possibly delayed) IMU timestamp; `None` at the end.
    ///
    /// The body rate and the acceleration are held over the interval, and the accelerometer
    /// reports the specific force in the attitude at its start, so the truth is integrated
    /// exactly as the filter integrates its nominal state. The filter's only approximations
    /// are then its first-order error model and its `f32` arithmetic. (A real IMU integrates
    /// over the interval; coning and sculling corrections are not modelled.)
    fn step(&mut self) -> Option<Sample> {
        let start = self.time_us;
        let skipped = u64::from(self.rng.gen_bool(DROP_PROBABILITY));
        let jitter = self.rng.gen_range(0..=2 * IMU_JITTER_US);
        let next = start + (1 + skipped) * IMU_PERIOD_US + jitter - IMU_JITTER_US;
        if next > DURATION_US {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let (t, dt) = (start as f32 * 1e-6, (next - start) as f32 * 1e-6);
        let (phase, s) = Phase::at(t);
        let [speed_rate, track_rate, climb] = phase.command(s);
        let acceleration = self.acceleration(speed_rate, track_rate, climb);

        let r = self.truth.attitude.rotation();
        let rate = transpose_mul(&r, phase.world_rate(t, track_rate));
        let force = [acceleration[0], acceleration[1], acceleration[2] + GRAVITY];
        let gyro = self.gyro_sample(rate, dt);
        let accel = self.accel_sample(transpose_mul(&r, force), dt);

        self.advance_truth(rate, acceleration, dt);
        self.speed += speed_rate * dt;
        self.track += track_rate * dt;
        self.time_us = next;
        self.count += 1;

        Some(Sample {
            time_us: next,
            dt,
            gyro,
            accel,
            gnss: self.gnss_fix(next),
            baro: (self.count % BARO_EVERY == 0).then(|| self.baro_sample()),
            heading: (self.count % HEADING_EVERY == HEADING_EVERY / 2)
                .then(|| self.heading_sample()),
            truth: self.truth,
        })
    }

    /// Returns the world-frame acceleration in m/s² for the commanded speed change, track rate
    /// and vertical acceleration.
    fn acceleration(&self, speed_rate: f32, track_rate: f32, climb: f32) -> Vec3 {
        let (sin, cos) = self.track.sin_cos();
        let centripetal = self.speed * track_rate;

        [
            speed_rate * cos - centripetal * sin,
            speed_rate * sin + centripetal * cos,
            climb,
        ]
    }

    /// Moves the truth over `dt` seconds with the body rate `rate` and the acceleration `a`.
    fn advance_truth(&mut self, rate: Vec3, a: Vec3, dt: f32) {
        let walk = Normal::new(0.0, GYRO_BIAS_WALK * dt.sqrt()).expect("a valid deviation");
        let t = &mut self.truth;

        t.position =
            core::array::from_fn(|i| t.position[i] + t.velocity[i] * dt + 0.5 * a[i] * dt * dt);
        t.velocity = core::array::from_fn(|i| t.velocity[i] + a[i] * dt);
        t.attitude = t
            .attitude
            .mul(&Quat::exp(rate.map(|w| w * dt)))
            .normalized();
        t.bias = t.bias.map(|b| b + walk.sample(&mut self.rng));
    }

    /// Returns the gyro reading for the body rate `rate` held over `dt` seconds.
    fn gyro_sample(&mut self, rate: Vec3, dt: f32) -> Vec3 {
        let noise = Normal::new(0.0, GYRO_NOISE_DENSITY / dt.sqrt()).expect("a valid deviation");
        core::array::from_fn(|i| rate[i] + self.truth.bias[i] + noise.sample(&mut self.rng))
    }

    /// Returns the accelerometer reading for the body-frame specific force `force` held over `dt`.
    fn accel_sample(&mut self, force: Vec3, dt: f32) -> Vec3 {
        let noise = Normal::new(0.0, ACCEL_NOISE_DENSITY / dt.sqrt()).expect("a valid deviation");
        force.map(|f| f + noise.sample(&mut self.rng))
    }

    /// Returns the GNSS fix due at `time_us`, if any: none during the dropout, some outliers.
    fn gnss_fix(&mut self, time_us: u64) -> Option<GnssFix> {
        let due = self.count % GNSS_EVERY == 0;
        if !due || (DROPOUT_US.0..DROPOUT_US.1).contains(&time_us) {
            return None;
        }

        let outlier = time_us > SETTLE_US && self.rng.gen_bool(OUTLIER_PROBABILITY);
        Some(GnssFix {
            z: self.gnss_sample(outlier),
            outlier,
        })
    }

    /// Returns a GNSS reading, offset by [`OUTLIER_OFFSET`] in a random direction if `outlier`.
    fn gnss_sample(&mut self, outlier: bool) -> [f32; GNSS_OBSERVATIONS] {
        let position = Normal::new(0.0, GNSS_POSITION_STD).expect("a valid deviation");
        let velocity = Normal::new(0.0, GNSS_VELOCITY_STD).expect("a valid deviation");
        let (p, v) = (self.truth.position, self.truth.velocity);
        let direction = self.rng.gen_range(-PI..PI);
        let offset = if outlier { OUTLIER_OFFSET } else { 0.0 };

        [
            p[0] + offset * direction.cos() + position.sample(&mut self.rng),
            p[1] + offset * direction.sin() + position.sample(&mut self.rng),
            v[0] + velocity.sample(&mut self.rng),
            v[1] + velocity.sample(&mut self.rng),
            v[2] + velocity.sample(&mut self.rng),
        ]
    }

    /// Returns a barometric altitude in m.
    fn baro_sample(&mut self) -> f32 {
        let noise = Normal::new(0.0, BARO_STD).expect("a valid deviation");
        self.truth.position[2] + noise.sample(&mut self.rng)
    }

    /// Returns a heading reading in radians, wrapped to `(−π, π]`.
    fn heading_sample(&mut self) -> f32 {
        let noise = Normal::new(0.0, HEADING_STD).expect("a valid deviation");
        wrap_pi(yaw(&self.truth.attitude.rotation()) + noise.sample(&mut self.rng))
    }
}

/// Returns the rotation from (roll, pitch, yaw) in radians, `R = R_z(ψ)·R_y(θ)·R_x(φ)`.
fn from_euler([roll, pitch, yaw]: Vec3) -> Quat {
    let qx = Quat::exp([roll, 0.0, 0.0]);
    let qy = Quat::exp([0.0, pitch, 0.0]);
    let qz = Quat::exp([0.0, 0.0, yaw]);
    qz.mul(&qy).mul(&qx)
}

/// Gate decisions and NIS of one aiding sensor.
#[derive(Clone, Copy, Debug, Default)]
struct Aid {
    /// Number of updates.
    updates: u32,
    /// Number the NIS gate rejected.
    gated: u32,
    /// Number applied to the estimate.
    accepted: u32,
    /// Sum of the NIS of the accepted updates.
    nis_sum: f32,
}

impl Aid {
    /// Records one update. Only an applied update counts as accepted: a correction rejected
    /// for a `S` that is not positive definite leaves the NIS of the previous update.
    fn record(&mut self, status: CorrectionStatus, nis: f32) {
        self.updates += 1;
        if status.is_applied() {
            self.accepted += 1;
            self.nis_sum += nis;
        } else if status == CorrectionStatus::Gated {
            self.gated += 1;
        }
    }

    /// Returns the mean NIS of the accepted updates.
    fn mean_nis(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let mean = self.nis_sum / self.accepted.max(1) as f32;
        mean
    }
}

/// The largest and the mean squared value of one error.
#[derive(Clone, Copy, Debug, Default)]
struct Error {
    /// Sum of the squared errors.
    sq_sum: f32,
    /// Largest error.
    max: f32,
}

/// Running statistics of the filter.
#[derive(Debug, Default)]
struct Stats {
    /// The GNSS updates.
    gnss: Aid,
    /// The barometer updates.
    baro: Aid,
    /// The heading updates.
    heading: Aid,
    /// Number of outlier fixes.
    outliers: u32,
    /// Number of outlier fixes the gate rejected.
    outliers_gated: u32,
    /// Position, velocity and attitude errors after settling (m, m/s, rad).
    errors: [Error; 3],
    /// Sums of the NEES of `δp`, `δv`, `δθ` and `δb_g` after settling.
    nees_sum: [f32; 4],
    /// Number of samples after settling.
    count: u32,
    /// Whether a fix after the dropout was accepted.
    recovered: bool,
    /// Number of fixes after the dropout rejected before the first accepted one.
    rejected_after_dropout: u32,
}

impl Stats {
    /// Records whether an outlier fix was rejected.
    fn record_outlier(&mut self, outlier: bool, status: CorrectionStatus) {
        if outlier {
            self.outliers += 1;
            self.outliers_gated += u32::from(status == CorrectionStatus::Gated);
        }
    }

    /// Returns the share of clean GNSS fixes the gate rejected.
    fn clean_rejection(&self) -> f32 {
        let clean = self.gnss.updates - self.outliers;
        #[allow(clippy::cast_precision_loss)]
        let rate = (self.gnss.gated - self.outliers_gated) as f32 / clean.max(1) as f32;
        rate
    }

    /// Records the errors and the NEES of the filter against the truth.
    fn record_errors(&mut self, truth: &Truth, ins: &Ins<'_>) {
        let errors = ins.errors(truth);
        for (e, v) in self.errors.iter_mut().zip(errors) {
            let n = norm(v);
            e.sq_sum += n * n;
            e.max = e.max.max(n);
        }

        for (sum, nees) in self.nees_sum.iter_mut().zip(ins.nees(truth)) {
            *sum += nees;
        }

        self.count += 1;
    }

    /// Returns the RMS of each error after settling.
    fn rms(&self) -> [f32; 3] {
        #[allow(clippy::cast_precision_loss)]
        let n = self.count.max(1) as f32;
        self.errors.map(|e| (e.sq_sum / n).sqrt())
    }

    /// Returns the mean NEES of each block after settling.
    fn mean_nees(&self) -> [f32; 4] {
        #[allow(clippy::cast_precision_loss)]
        let n = self.count.max(1) as f32;
        self.nees_sum.map(|s| s / n)
    }

    /// Prints the error, consistency and gate statistics.
    fn report(&self) {
        let [p, v, a] = self.rms();
        let [pm, vm, am] = self.errors.map(|e| e.max);
        let [np, nv, na, nb] = self.mean_nees();

        println!("\nAfter {} s:", SETTLE_US / 1_000_000);
        println!("  position error  RMS {p:6.3} m     max {pm:6.3} m");
        println!("  velocity error  RMS {v:6.3} m/s   max {vm:6.3} m/s");
        println!(
            "  attitude error  RMS {:6.3}°       max {:6.3}°",
            a.to_degrees(),
            am.to_degrees()
        );
        println!("  mean NEES (expected 3): δp {np:.2}, δv {nv:.2}, δθ {na:.2}, δb_g {nb:.2}");
        println!("\nUpdate     count  gated  mean NIS of accepted (expected just below)");
        for (name, aid, dof) in [
            ("GNSS", self.gnss, 5),
            ("baro", self.baro, 1),
            ("heading", self.heading, 1),
        ] {
            println!(
                "  {name:8} {:5}  {:5}  {:.2} ({dof})",
                aid.updates,
                aid.gated,
                aid.mean_nis()
            );
        }
        println!(
            "GNSS outliers rejected: {} of {}; clean fixes rejected: {:.1} %",
            self.outliers_gated,
            self.outliers,
            100.0 * self.clean_rejection(),
        );
        println!(
            "Fixes rejected after the dropout before the first accepted one: {}",
            self.rejected_after_dropout,
        );
    }
}

/// Prints one row of the progress table.
fn print_row(sample: &Sample, ins: &Ins<'_>, stats: &Stats) {
    #[allow(clippy::cast_precision_loss)]
    let t = sample.time_us as f32 * 1e-6;
    let [p, v, a, _] = ins.errors(&sample.truth).map(norm);

    println!(
        "{t:6.1}  {:11} {p:7.3}  {v:8.3}  {:6.3}  {:6.3}  {:5}/{}",
        Phase::at(t).0.name(),
        a.to_degrees(),
        ins.position_std(),
        stats.gnss.accepted,
        stats.gnss.gated,
    );
}

/// Returns success if the filter met every bound; prints the first failure otherwise.
fn check(stats: &Stats) -> ExitCode {
    let inside = |v: f32, (lo, hi): (f32, f32)| (lo..=hi).contains(&v);
    let below =
        |values: [f32; 3], bounds: [f32; 3]| values.iter().zip(bounds).all(|(v, b)| *v <= b);
    let in_units = |[p, v, a]: [f32; 3]| [p, v, a.to_degrees()];
    let scalar_nis = [stats.baro.mean_nis(), stats.heading.mean_nis()];

    let checks = [
        (below(in_units(stats.rms()), MAX_RMS), "RMS error"),
        (
            below(in_units(stats.errors.map(|e| e.max)), MAX_ERROR),
            "largest error",
        ),
        (
            stats.mean_nees().iter().all(|&n| inside(n, NEES_RANGE)),
            "mean NEES",
        ),
        (
            inside(stats.gnss.mean_nis(), GNSS_NIS_RANGE),
            "mean GNSS NIS",
        ),
        (
            scalar_nis.iter().all(|&n| inside(n, SCALAR_NIS_RANGE)),
            "mean barometer or heading NIS",
        ),
        (stats.outliers_gated == stats.outliers, "outlier rejection"),
        (
            stats.recovered && stats.rejected_after_dropout <= MAX_REJECTED_AFTER_DROPOUT,
            "recovery after the dropout",
        ),
        (
            stats.clean_rejection() <= MAX_CLEAN_REJECTION,
            "clean GNSS rejection rate",
        ),
    ];

    if let Some((_, what)) = checks.iter().find(|(ok, _)| !ok) {
        eprintln!("GNSS/INS out of bounds: {what}");
        return ExitCode::FAILURE;
    }

    println!("GNSS/INS within bounds");
    ExitCode::SUCCESS
}

/// Asserts that the INS-specific model blocks match their definitions by finite differences.
///
/// The attitude blocks (`Exp(u)ᵀ`, the yaw row, the reset) are checked by `attitude-eskf`;
/// here the velocity–attitude block `−R̂·[f]×·Δt` of `F`, the yaw row at a tilted attitude, and
/// the series of `Exp` against `sinf`/`cosf` where it hands over.
fn check_helpers() {
    let nominal = from_euler([0.4, -0.3, 2.0]);
    let delta = [1e-3, -2e-3, 1.5e-3];
    let perturbed = nominal.mul(&Quat::exp(delta));
    let (r, rp) = (nominal.rotation(), perturbed.rotation());

    // δv after one step: (R̂·Exp(δθ) − R̂)·f·Δt ≈ −R̂·[f]×·δθ·Δt.
    let (f, dt) = ([1.0, -2.0, 9.5], 0.01);
    let moved: Vec3 = core::array::from_fn(|i| (mat_mul(&rp, f)[i] - mat_mul(&r, f)[i]) * dt);
    let linear = mat_mul(&velocity_attitude_transition(&r, f, dt), delta);
    assert_close(moved, linear, 1e-6, "F_vθ");
    assert_close(mat_mul(&skew(f), delta), cross(f, delta), 1e-7, "[f]×");

    let yaw_change = wrap_pi(yaw(&rp) - yaw(&r));
    assert!(
        (yaw_change - dot(yaw_jacobian(&r), delta)).abs() < 1e-5,
        "yaw H self-check failed"
    );

    // Just below and above the series threshold, Exp agrees with the closed form.
    for angle in [0.049_f32, 0.051] {
        let q = Quat::exp([angle, 0.0, 0.0]);
        let exact = [(0.5 * angle).cos(), (0.5 * angle).sin()];
        assert!(
            (q.w - exact[0]).abs() < 1e-7 && (q.v[0] - exact[1]).abs() < 1e-7,
            "Exp series self-check failed"
        );
    }
}

/// Panics with `what` if any entry of `a` and `b` differs by more than `tolerance`.
fn assert_close(a: Vec3, b: Vec3, tolerance: f32, what: &str) {
    let close = a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance);
    assert!(close, "{what} self-check failed: {a:?} vs {b:?}");
}
