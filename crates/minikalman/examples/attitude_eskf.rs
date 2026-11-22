//! Error-state (multiplicative) EKF for the attitude of an IMU on SO(3): a quaternion
//! propagated with the gyro, tilt corrected with the accelerometer, heading with an external
//! yaw source, all in `f32` and without allocation.
//!
//! This is the attitude filter of a small flight controller or IMU board. Many such boards run
//! a complementary filter instead, because a covariance filter looks too heavy. Here it is
//! six states, three [`Storage`] buffers of known size, and a handful of 3×3 blocks.
//!
//! ## Why an error state on SO(3)
//!
//! An attitude is a rotation, not a vector. A quaternion has four entries for three degrees
//! of freedom and must stay unit length, so `x̂ ← x̂ + K·y` on its entries leaves the sphere,
//! and its 4×4 covariance is singular. Euler angles have three entries but wrap and lock
//! near ±90° pitch. The error-state filter avoids both. The caller keeps the *nominal*
//! attitude `q̂` (a unit quaternion) and gyro bias `b̂`. The Kalman filter estimates only a
//! small error `δx = (δθ, δb)`, six numbers in a vector space, with
//!
//! ```text
//! q = q̂ ⊗ Exp(δθ)        b = b̂ + δb
//! ```
//!
//! `δθ` is a rotation vector in the body frame, the local angular error of Solà,
//! "Quaternion kinematics for the error-state Kalman filter" (arXiv:1711.02508). Because `δθ` is small, its
//! first-order models are accurate, and `P` is a regular 6×6 covariance. This is the same
//! nominal + error split as the `se2-iekf` example and the crate docs' "Filtering on
//! manifolds" recipe.
//!
//! ## One step
//!
//! 1. **Predict (every gyro sample).** The gyro measures `ω_m = ω + b + n`. With the time
//!    since the last sample `Δt` (from timestamps, so it varies) and `u = (ω_m − b̂)·Δt`, the
//!    nominal attitude moves on the group, `q̂ ← q̂ ⊗ Exp(u)`. To first order, the error
//!    evolves as
//!
//!    ```text
//!    δθ ← Exp(u)ᵀ·δθ − Δt·δb − Δt·n       F = | Exp(u)ᵀ  −Δt·I |
//!    δb ← δb + random walk                    |    0        I  |
//!    ```
//!
//!    with `Q = diag(σ_g²·Δt·I, σ_b²·Δt·I)` for the gyro noise density `σ_g` and the bias
//!    random walk `σ_b`. `F` and `Q` depend on `Δt`, so both are written on every step.
//!    `predict_nonlinear` keeps `δx = 0`; only `P ← F·P·Fᵀ + Q` matters.
//! 2. **Correct tilt (accelerometer).** At rest, the accelerometer measures gravity in the
//!    body frame, `f = Rᵀ·g` with `g = (0, 0, 9.81) m/s²`. With `R = R̂·Exp(δθ)`,
//!    `Rᵀ ≈ (I − [δθ]×)·R̂ᵀ`, so `f ≈ f̂ + [f̂]×·δθ` with the prediction `f̂ = R̂ᵀ·g`:
//!
//!    ```text
//!    H = | [f̂]×  0 |      (3×6)
//!    ```
//!
//!    `[f̂]×·f̂ = 0`: a rotation about the gravity direction does not change `f`, so the
//!    accelerometer says nothing about the heading, which needs its own source.
//! 3. **Correct heading (optional).** An external yaw `ψ = atan2(R₁₀, R₀₀)` (a magnetometer
//!    heading, a dual-antenna GNSS, a vision system). Its derivative with respect to a
//!    world-frame rotation `φ` is `h = (−R₀₀·R₂₀/c, −R₁₀·R₂₀/c, 1)` with `c = R₀₀² + R₁₀²`, and
//!    `φ = R̂·δθ`, so `H = [h·R̂  0]` (1×6). The innovation is wrapped to `(−π, π]`.
//! 4. **Reset.** After each correction, `q̂ ← q̂ ⊗ Exp(δθ)`, `b̂ ← b̂ + δb`, `δx ← 0`, and
//!    `P ← G·P·Gᵀ` with `G = diag(I − ½[δθ]×, I)` (Solà's reset Jacobian), which moves `P` to
//!    the new, zero error. `G ≈ I` for small `δθ`.
//!
//! ## Why a gate beats a ±10 % |a| window
//!
//! The tilt update is only valid while the body does not accelerate. A common rule accepts the
//! accelerometer only when `|f|` is within ±10 % of `g`. That rule is blind to horizontal
//! acceleration: a car cornering at 3 m/s² (or a quadcopter holding a bank) sees
//! `|f| = √(9.81² + 3²) = 10.26 m/s²`, 4.6 % above `g`, so the window accepts it, while the
//! direction of `f` is 17° off the vertical. The filter is then pulled towards that wrong tilt.
//!
//! The NIS gate looks at the direction instead. The innovation `y = f − f̂` is compared with
//! what the filter expects, `S = H·P·Hᵀ + R`, through `NIS = yᵀ·S⁻¹·y`. With `P` small after a
//! few seconds, a 3 m/s² sideways error gives an NIS in the thousands, far above the 99 %
//! chi-square quantile for three degrees of freedom (`11.34`), and `set_gate` rejects it.
//! The example runs a second filter with the window instead of the gate on the same samples
//! and prints both: the window accepts every dynamic sample, its tilt is pulled off by
//! tens of degrees, and its heading updates then fail their own gate, so it does not recover.
//!
//! Two details make the gate work in practice:
//!
//! - **Hold-off.** Accelerations below about three noise deviations (0.15 m/s² here) pass any
//!   gate. They occur right next to rejected samples, where a manoeuvre starts or a vibration
//!   crosses zero, and bias the tilt. After a gated sample the filter therefore skips the next
//!   [`HOLD_OFF_SAMPLES`] accelerometer samples (0.2 s). Over seeds 1 to 300, the mean NEES
//!   of `δθ` had a median of 4.2 and a 95th percentile of 9.9 without it, and one seed locked
//!   out (72); with it, 3.05 and 5.3. The price is that each false rejection at rest also
//!   skips 0.2 s of good samples.
//! - **Lockout.** The gate trusts `P`. If the attitude is wrong by more than `P` admits, every
//!   sample is rejected, the error grows faster than `P`, and the filter never recovers. Start
//!   close to the truth (roll and pitch from an accelerometer sample at rest, heading from the
//!   first heading reading, as done here), keep `σ_g` honest, and re-initialize after a long
//!   rejection streak. Gate a slow measurement, like the 5 Hz heading here, more loosely
//!   ([`YAW_GATE`]), since each rejection costs more.
//!
//! ## Simulation
//!
//! 60 s of a known rotation sequence (rest, tumbling by up to about 35°, a 15 s turn with
//! 3 m/s² centripetal acceleration, 8 s of horizontal shaking at 2.5 m/s², slow motion) with a
//! constant-plus-random-walk gyro bias of about 2°/s, gyro samples at about 200 Hz with ±1 ms
//! timestamp jitter and 1 % dropped samples, the accelerometer at every fourth gyro sample and
//! a heading with 2° noise at 5 Hz, all from a fixed seed. The filters start from one
//! accelerometer and one heading sample and know nothing about the bias. The true rate is held
//! over each sample interval, so the truth is exact and the filter's only approximation is its
//! first-order error model. The example checks the gated filter's attitude error, the mean NEES
//! of `δθ` and of `δb`, the mean NIS of the accepted tilt updates and the rejection rates
//! against bounds and exits with a non-zero status if one fails; the window filter is printed
//! only.
//!
//! Not modelled: accelerometer bias and scale errors, magnetometer disturbances (the heading
//! is a clean external yaw), gyro scale errors and the yaw singularity at ±90° pitch (see
//! [`yaw_jacobian`]).
//!
//! The quaternion and 3×3 helpers stay in the example on purpose: the crate has no Lie-group
//! types and needs no new dependency. The model is written as free functions over 3×3 blocks
//! ([`attitude_transition`], [`gravity_jacobian`], [`yaw_jacobian`], [`reset_jacobian`]) so
//! a larger error state (position, velocity, attitude, bias) can place the same blocks at
//! another offset.

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

/// Number of error states: the attitude error `δθ` (3) and the gyro bias error `δb` (3).
const STATES: usize = 6;

/// Number of accelerometer measurements: the specific force in the body frame.
const ACCEL_OBSERVATIONS: usize = 3;

/// Number of heading measurements: one yaw angle.
const YAW_OBSERVATIONS: usize = 1;

/// A 3×3 matrix as rows.
type Mat3 = [[f32; 3]; 3];

/// A 3-vector.
type Vec3 = [f32; 3];

/// Simulated duration in microseconds (60 s).
const DURATION_US: u64 = 60_000_000;

/// Nominal gyro sample period in microseconds (200 Hz).
const GYRO_PERIOD_US: u64 = 5_000;

/// Largest timestamp jitter of a gyro sample in microseconds, in either direction.
const GYRO_JITTER_US: u64 = 1_000;

/// Probability that a gyro sample is dropped, which doubles the next `Δt`.
const DROP_PROBABILITY: f64 = 0.01;

/// The accelerometer is read on every this many gyro samples (about 50 Hz).
const ACCEL_EVERY: usize = 4;

/// The external yaw arrives on every this many gyro samples (about 5 Hz).
const YAW_EVERY: usize = 40;

/// Print both filters every this many microseconds.
const PRINT_EVERY_US: u64 = 5_000_000;

/// Magnitude of gravity in m/s².
const GRAVITY: f32 = 9.81;

/// Gyro white noise density `σ_g` in rad/s/√Hz.
const GYRO_NOISE_DENSITY: f32 = 3e-4;

/// Gyro bias random walk `σ_b` in rad/s/√s.
const GYRO_BIAS_WALK: f32 = 1e-4;

/// The true gyro bias at start-up in rad/s.
const INITIAL_BIAS: Vec3 = [0.03, -0.02, 0.01];

/// Standard deviation of each accelerometer axis in m/s².
const ACCEL_NOISE_STD: f32 = 0.05;

/// Standard deviation of the external yaw in radians (2°).
const YAW_NOISE_STD: f32 = 2.0 * PI / 180.0;

/// The true attitude at start-up as (roll, pitch, yaw) in radians.
const INITIAL_EULER: Vec3 = [3.0 * PI / 180.0, -2.0 * PI / 180.0, 40.0 * PI / 180.0];

/// Prior standard deviation of roll and pitch after levelling, in radians (2°).
const PRIOR_TILT_STD: f32 = 2.0 * PI / 180.0;

/// Prior standard deviation of the heading after the first yaw reading, in radians (5°).
const PRIOR_YAW_STD: f32 = 5.0 * PI / 180.0;

/// Prior standard deviation of each gyro bias axis in rad/s.
const PRIOR_BIAS_STD: f32 = 0.05;

/// Tilt gate: the 99 % chi-square quantile for three degrees of freedom.
const ACCEL_GATE: f32 = 11.34;

/// Heading gate: the 99.9 % chi-square quantile for one degree of freedom.
///
/// Wider than the tilt gate on purpose. A rejected tilt sample costs little, the next one
/// comes 20 ms later. The heading comes at 5 Hz, and while the yaw bias is still uncertain a
/// few rejected headings let the error grow faster than `P`, so the gate keeps rejecting (the
/// same lockout as for the tilt). With the 99 % quantile `6.63`, one of 200 seeds locked the
/// heading out for 4 s.
const YAW_GATE: f32 = 10.83;

/// Accelerometer samples skipped after a gated one (0.2 s at 50 Hz).
///
/// Accelerations below about three noise deviations pass the gate. They are common right where
/// a manoeuvre starts or a vibration crosses zero, next to samples the gate rejected, so the
/// filter waits for the motion to settle.
const HOLD_OFF_SAMPLES: u32 = 10;

/// Relative `|f|` window of the comparison filter: accept when `| |f| − g | ≤ 0.1·g`.
const ACCEL_WINDOW: f32 = 0.1;

/// A world-frame acceleration above this many m/s² marks a sample as dynamic in the statistics.
const DYNAMIC_THRESHOLD: f32 = 0.3;

/// Statistics (errors, NEES) start after this many microseconds, once the heading converged.
const SETTLE_US: u64 = 5_000_000;

/// Largest attitude error of the gated filter after [`SETTLE_US`], in degrees.
///
/// This and the bounds below leave margin over seeds 1 to 300, which gave at most 2.5° here.
const MAX_ATTITUDE_ERROR_DEG: f32 = 3.0;

/// Largest final attitude error of the gated filter, in degrees (300 seeds: at most 0.98°).
const FINAL_ATTITUDE_ERROR_DEG: f32 = 1.5;

/// Accepted range of the mean NEES of `δθ` and of `δb` (expected 3 for a consistent filter).
///
/// One run's time average is autocorrelated, so it scatters widely: over 300 seeds the mean
/// NEES of `δθ` had a median of 3.05 and ranged from 1.5 to 7.7, that of `δb` 1.3 to 7.6.
const NEES_RANGE: (f32, f32) = (1.0, 10.0);

/// Accepted range of the mean NIS of the accepted tilt updates (300 seeds: 2.75 to 3.04).
const NIS_RANGE: (f32, f32) = (2.5, 3.5);

/// Smallest share of dynamic accelerometer samples the gate and the hold-off must skip.
const MIN_DYNAMIC_REJECTION: f32 = 0.9;

/// Largest share of static accelerometer samples the gate may reject (300 seeds: at most 1.6 %).
///
/// The hold-off skips about ten times as many good samples after these, which is not counted.
const MAX_STATIC_REJECTION: f32 = 0.05;

/// Fixed seed so CI runs are reproducible.
const SEED: u64 = 184;

/// Runs the self-checks, the simulation, and checks the gated filter against the bounds.
fn main() -> ExitCode {
    check_helpers();
    report_sizes();

    let mut sim = Simulation::new(SEED);
    let guess = initial_attitude(sim.accel_sample(), sim.yaw_sample());

    let mut gated_storage = Storage::new();
    let mut window_storage = Storage::new();
    let mut gated = gated_storage.build(guess, Some(ACCEL_GATE));
    let mut window = window_storage.build(guess, None);
    let mut stats = Stats::default();
    let mut next_print = 0;

    println!("\n     t   phase    gated err  window err  gated |δb|  tilt used/skipped");
    while let Some(sample) = sim.step() {
        run_step(&mut gated, &mut window, &sample, &mut stats);

        if sample.time_us >= next_print {
            print_row(&sample, &gated, &window, &stats);
            next_print += PRINT_EVERY_US;
        }
    }

    stats.report();
    check(&stats, &sim.truth, &gated)
}

/// Feeds one simulated sample to both filters and records the statistics.
fn run_step(gated: &mut Eskf<'_>, window: &mut Eskf<'_>, sample: &Sample, stats: &mut Stats) {
    gated.predict(sample.gyro, sample.dt);
    window.predict(sample.gyro, sample.dt);

    if let Some(f) = sample.accel {
        let status = gated.correct_accel(f);
        stats.record_accel(sample, status, gated.accel.nis(), within_window(f));

        if within_window(f) {
            window.correct_accel(f);
        }
    }

    if let Some(yaw) = sample.yaw {
        let statuses = [gated.correct_yaw(yaw), window.correct_yaw(yaw)];
        stats.record_yaw(statuses);
    }

    if sample.time_us >= SETTLE_US {
        stats.record_errors(&sample.truth, gated, window);
    }
}

/// Returns whether `| |f| − g | ≤ 0.1·g`, the magnitude rule of the comparison filter.
fn within_window(f: Vec3) -> bool {
    (norm(f) - GRAVITY).abs() <= ACCEL_WINDOW * GRAVITY
}

/// Returns the attitude with roll and pitch from a resting accelerometer sample and the given yaw.
///
/// Starting the filter close to the truth keeps `δθ` small, which every first-order step
/// (`F`, `H`, the reset `G`, the gate) relies on.
fn initial_attitude(f: Vec3, yaw: f32) -> Quat {
    let roll = f[1].atan2(f[2]);
    let pitch = (-f[0]).atan2(f[1].hypot(f[2]));
    Quat::from_euler([roll, pitch, yaw])
}

/// Prints the size of the filter storage, which is all the filter needs besides the nominal state.
fn report_sizes() {
    use core::mem::size_of;
    let filter = size_of::<FilterStorage<STATES, f32>>();
    let accel = size_of::<ObservationStorage<STATES, ACCEL_OBSERVATIONS, f32>>();
    let yaw = size_of::<ObservationStorage<STATES, YAW_OBSERVATIONS, f32>>();

    println!("Storage (f32, no allocation):");
    println!("  FilterStorage<6>           {filter:5} bytes");
    println!("  ObservationStorage<6, 3>   {accel:5} bytes (accelerometer)");
    println!("  ObservationStorage<6, 1>   {yaw:5} bytes (yaw)");
    println!(
        "  Storage total              {:5} bytes, plus {} bytes of nominal quaternion and bias",
        size_of::<Storage>(),
        size_of::<Quat>() + size_of::<Vec3>(),
    );
}

/// The inline buffers of the attitude filter and its two observations.
struct Storage {
    /// `F`, `δx`, `P`, `Q` and the filter temporaries.
    filter: FilterStorage<STATES, f32>,
    /// `H`, `z`, `R`, `y`, `S`, `K` and temporaries of the accelerometer update.
    accel: ObservationStorage<STATES, ACCEL_OBSERVATIONS, f32>,
    /// The same for the yaw update.
    yaw: ObservationStorage<STATES, YAW_OBSERVATIONS, f32>,
}

impl Storage {
    /// Creates zeroed storage; `const`, so it can be a `static` on a microcontroller.
    const fn new() -> Self {
        Self {
            filter: FilterStorage::new(0.0),
            accel: ObservationStorage::new(0.0),
            yaw: ObservationStorage::new(0.0),
        }
    }

    /// Wires the filter at `attitude` with the prior, the noise and the given tilt gate.
    fn build(&mut self, attitude: Quat, accel_gate: Option<f32>) -> Eskf<'_> {
        let mut filter = self.filter.build();
        let mut accel = self.accel.build();
        let mut yaw = self.yaw.build();

        let attitude_prior = attitude_prior(&attitude);
        filter.estimate_covariance_mut().apply(|p| {
            write_block(p, 0, 0, &attitude_prior);
            (3..STATES).for_each(|i| p.set_at(i, i, PRIOR_BIAS_STD * PRIOR_BIAS_STD));
        });

        accel.measurement_noise_covariance_mut().apply(|r| {
            (0..ACCEL_OBSERVATIONS).for_each(|i| r.set_at(i, i, ACCEL_NOISE_STD * ACCEL_NOISE_STD));
        });
        accel.set_gate(accel_gate);

        yaw.measurement_noise_covariance_mut()
            .apply(|r| r.set_at(0, 0, YAW_NOISE_STD * YAW_NOISE_STD));
        yaw.set_gate(Some(YAW_GATE));

        Eskf {
            filter,
            accel,
            yaw,
            attitude,
            bias: [0.0; 3],
            hold_off: 0,
        }
    }
}

/// Returns the prior covariance of `δθ` for the levelled tilt and the initial heading.
///
/// The uncertainty is about the *world* vertical, so it is `R̂ᵀ·diag(σ_t², σ_t², σ_ψ²)·R̂` in the
/// body frame of `δθ`. A diagonal body-frame prior puts the heading variance on the body z axis,
/// which is tilted, so the accelerometer would see part of it and move the heading.
fn attitude_prior(attitude: &Quat) -> Mat3 {
    let r = attitude.rotation().0;
    let variance = [PRIOR_TILT_STD, PRIOR_TILT_STD, PRIOR_YAW_STD].map(|s| s * s);

    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..3).map(|k| r[k][i] * variance[k] * r[k][j]).sum())
    })
}

/// The error-state filter: `δx = (δθ, δb)` in the Kalman filter, `q̂` and `b̂` outside.
struct Eskf<'a> {
    /// The error-state filter. `δx` is zero between steps.
    filter: BorrowedKalman<'a, STATES, f32>,
    /// The accelerometer tilt update.
    accel: BorrowedObservation<'a, STATES, ACCEL_OBSERVATIONS, f32>,
    /// The external yaw update.
    yaw: BorrowedObservation<'a, STATES, YAW_OBSERVATIONS, f32>,
    /// The nominal attitude `q̂`, body to world.
    attitude: Quat,
    /// The nominal gyro bias `b̂` in rad/s.
    bias: Vec3,
    /// Accelerometer samples still to skip after the last gated one.
    hold_off: u32,
}

impl Eskf<'_> {
    /// Integrates one gyro sample over `dt` seconds and propagates `P`.
    fn predict(&mut self, gyro: Vec3, dt: f32) {
        let u: Vec3 = core::array::from_fn(|i| (gyro[i] - self.bias[i]) * dt);
        self.attitude = self.attitude.mul(&Quat::exp(u)).normalized();

        let rotation = attitude_transition(u);
        self.filter.state_transition_jacobian_mut().apply(|f| {
            f.make_identity();
            write_block(f, 0, 0, &rotation);
            (0..3).for_each(|i| f.set_at(i, i + 3, -dt));
        });

        let qa = GYRO_NOISE_DENSITY * GYRO_NOISE_DENSITY * dt;
        let qb = GYRO_BIAS_WALK * GYRO_BIAS_WALK * dt;
        self.filter.direct_process_noise_mut().apply(|q| {
            (0..3).for_each(|i| {
                q.set_at(i, i, qa);
                q.set_at(i + 3, i + 3, qb);
            });
        });

        // The error stays zero; only P = F·P·Fᵀ + Q matters.
        self.filter.predict_nonlinear(|_, next| {
            (0..STATES).for_each(|i| next.set_row(i, 0.0));
        });
    }

    /// Corrects the tilt with a body-frame specific force in m/s², then resets the error.
    ///
    /// Returns `None` for a sample skipped by the hold-off after a gated one.
    fn correct_accel(&mut self, f: Vec3) -> Option<CorrectionStatus> {
        if self.hold_off > 0 {
            self.hold_off -= 1;
            return None;
        }

        let predicted = self.attitude.rotation().transpose_mul([0.0, 0.0, GRAVITY]);
        let jacobian = gravity_jacobian(predicted);

        self.accel.observation_jacobian_matrix_mut().apply(|h| {
            h.clear();
            write_block(h, 0, 0, &jacobian);
        });
        self.accel
            .measurement_vector_mut()
            .apply(|z| (0..3).for_each(|i| z.set_row(i, f[i])));

        let status = self.filter.correct_nonlinear(&mut self.accel, |_, h| {
            (0..3).for_each(|i| h.set_row(i, predicted[i]));
        });

        if status == CorrectionStatus::Gated {
            self.hold_off = HOLD_OFF_SAMPLES;
        }

        self.reset();
        Some(status)
    }

    /// Corrects the heading with a yaw in radians, then resets the error.
    fn correct_yaw(&mut self, yaw: f32) -> CorrectionStatus {
        let rotation = self.attitude.rotation();
        let jacobian = yaw_jacobian(&rotation);

        self.yaw.observation_jacobian_matrix_mut().apply(|h| {
            h.clear();
            (0..3).for_each(|i| h.set_at(0, i, jacobian[i]));
        });
        self.yaw
            .measurement_vector_mut()
            .apply(|z| z.set_row(0, yaw));

        let status = self.filter.correct_nonlinear_with(
            &mut self.yaw,
            |_, h| h.set_row(0, rotation.yaw()),
            |y| y[0] = wrap_pi(y[0]),
        );

        self.reset();
        status
    }

    /// Moves the error into the nominal state, resets it to zero and moves `P` along.
    ///
    /// After a rejected correction the error is zero and this changes nothing.
    fn reset(&mut self) {
        let x = self.filter.state_vector();
        let dtheta = [x.get_row(0), x.get_row(1), x.get_row(2)];
        let dbias = [x.get_row(3), x.get_row(4), x.get_row(5)];

        self.attitude = self.attitude.mul(&Quat::exp(dtheta)).normalized();
        self.bias = core::array::from_fn(|i| self.bias[i] + dbias[i]);

        let g = reset_jacobian(dtheta);
        self.filter.estimate_covariance_mut().apply(|p| {
            let reset = reset_covariance(&g, &read_mat6(p));
            (0..STATES).for_each(|r| (0..STATES).for_each(|c| p.set_at(r, c, reset[r][c])));
        });

        self.filter
            .state_vector_mut()
            .apply(|v| (0..STATES).for_each(|i| v.set_row(i, 0.0)));
    }

    /// Returns the attitude error `δθ` with `truth = q̂ ⊗ Exp(δθ)`, in radians.
    fn attitude_error(&self, truth: &Quat) -> Vec3 {
        self.attitude.conjugate().mul(truth).log()
    }

    /// Returns the NEES of the attitude error and of the bias error against the 3×3 blocks of `P`.
    fn nees(&self, truth: &Truth) -> (f32, f32) {
        let p = read_mat6(self.filter.estimate_covariance());
        let block = |o: usize| -> Mat3 {
            core::array::from_fn(|r| core::array::from_fn(|c| p[o + r][o + c]))
        };
        let bias_error: Vec3 = core::array::from_fn(|i| truth.bias[i] - self.bias[i]);

        (
            mahalanobis(&block(0), self.attitude_error(&truth.attitude)),
            mahalanobis(&block(3), bias_error),
        )
    }
}

/// Returns the attitude block `Exp(u)ᵀ` of `F` for the gyro increment `u = (ω_m − b̂)·Δt`.
///
/// The bias block of `F` is `−Δt·I` next to it and `I` below; see the module docs.
fn attitude_transition(u: Vec3) -> Mat3 {
    transpose(&Quat::exp(u).rotation().0)
}

/// Returns the tilt Jacobian `[f̂]×` of the specific force `f̂ = R̂ᵀ·g` predicted in the body frame.
fn gravity_jacobian(predicted: Vec3) -> Mat3 {
    skew(predicted)
}

/// Returns the row `∂ψ/∂δθ` of the yaw `ψ = atan2(R₁₀, R₀₀)` for the attitude `R̂`.
///
/// Singular at ±90° pitch, where the yaw is undefined.
fn yaw_jacobian(rotation: &Rotation) -> Vec3 {
    let r = &rotation.0;
    let c = r[0][0] * r[0][0] + r[1][0] * r[1][0];
    let world = [-r[0][0] * r[2][0] / c, -r[1][0] * r[2][0] / c, 1.0];

    // h_world · R̂, since the world-frame error is φ = R̂·δθ.
    core::array::from_fn(|j| (0..3).map(|i| world[i] * r[i][j]).sum())
}

/// Returns the attitude block `I − ½[δθ]×` of the reset Jacobian `G`; the bias block is `I`.
fn reset_jacobian(dtheta: Vec3) -> Mat3 {
    let s = skew(dtheta);
    core::array::from_fn(|r| core::array::from_fn(|c| f32::from(u8::from(r == c)) - 0.5 * s[r][c]))
}

/// Returns `G·P·Gᵀ` for `G = diag(g, I₃)`, including the attitude–bias cross blocks.
fn reset_covariance(g: &Mat3, p: &[[f32; STATES]; STATES]) -> [[f32; STATES]; STATES] {
    let gfull = |r: usize, c: usize| -> f32 {
        match (r < 3, c < 3) {
            (true, true) => g[r][c],
            (false, false) => f32::from(u8::from(r == c)),
            _ => 0.0,
        }
    };

    core::array::from_fn(|r| {
        core::array::from_fn(|c| {
            (0..STATES)
                .flat_map(|i| (0..STATES).map(move |k| (i, k)))
                .map(|(i, k)| gfull(r, i) * p[i][k] * gfull(c, k))
                .sum()
        })
    })
}

/// Copies a 6×6 covariance out of the filter.
fn read_mat6(p: &impl Matrix<STATES, STATES, f32>) -> [[f32; STATES]; STATES] {
    core::array::from_fn(|r| core::array::from_fn(|c| p.get_at(r, c)))
}

/// Writes a 3×3 block into `m` with its top-left corner at `(row, col)`.
fn write_block<const R: usize, const C: usize>(
    m: &mut impl MatrixMut<R, C, f32>,
    row: usize,
    col: usize,
    block: &Mat3,
) {
    for (r, values) in block.iter().enumerate() {
        for (c, value) in values.iter().enumerate() {
            m.set_at(row + r, col + c, *value);
        }
    }
}

/// A unit quaternion `w + x·i + y·j + z·k` (Hamilton convention), body to world.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Quat {
    /// The scalar part.
    w: f32,
    /// The vector part.
    v: Vec3,
}

impl Quat {
    /// Returns the Hamilton product `self ⊗ other`.
    fn mul(&self, other: &Quat) -> Quat {
        let (a, b) = (self.v, other.v);
        let c = cross(a, b);

        Quat {
            w: self.w * other.w - dot(a, b),
            v: core::array::from_fn(|i| self.w * b[i] + other.w * a[i] + c[i]),
        }
    }

    /// Returns the inverse of a unit quaternion.
    fn conjugate(&self) -> Quat {
        Quat {
            w: self.w,
            v: self.v.map(|x| -x),
        }
    }

    /// Returns the quaternion scaled to unit length, which `f32` products slowly lose.
    fn normalized(&self) -> Quat {
        let n = (self.w * self.w + dot(self.v, self.v)).sqrt();

        Quat {
            w: self.w / n,
            v: self.v.map(|x| x / n),
        }
    }

    /// Returns `Exp(θ)`: the rotation by `|θ|` radians about `θ/|θ|` .
    ///
    /// Small angles use the Taylor series of `sin(|θ|/2)/|θ|`.
    fn exp(theta: Vec3) -> Quat {
        let angle = norm(theta);
        let half = 0.5 * angle;
        let scale = if angle < 1e-3 {
            0.5 - angle * angle / 48.0
        } else {
            half.sin() / angle
        };

        Quat {
            w: half.cos(),
            v: theta.map(|x| x * scale),
        }
    }

    /// Returns `Log(self)`, the rotation vector with angle in `[0, π]` .
    fn log(&self) -> Vec3 {
        // q and −q are the same rotation; pick the one with w ≥ 0 for the short way round.
        let (w, v) = if self.w < 0.0 {
            (-self.w, self.v.map(|x| -x))
        } else {
            (self.w, self.v)
        };
        let n = norm(v);
        let scale = if n < 1e-6 {
            2.0 / w
        } else {
            2.0 * n.atan2(w) / n
        };

        v.map(|x| x * scale)
    }

    /// Returns the rotation from (roll, pitch, yaw) in radians, `R = R_z(ψ)·R_y(θ)·R_x(φ)`.
    fn from_euler([roll, pitch, yaw]: Vec3) -> Quat {
        let qx = Quat::exp([roll, 0.0, 0.0]);
        let qy = Quat::exp([0.0, pitch, 0.0]);
        let qz = Quat::exp([0.0, 0.0, yaw]);
        qz.mul(&qy).mul(&qx)
    }

    /// Returns the rotation matrix `R` (body to world).
    fn rotation(&self) -> Rotation {
        let Quat { w, v: [x, y, z] } = *self;

        Rotation([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ])
    }
}

/// A rotation matrix `R` (body to world), rows.
#[derive(Clone, Copy, Debug)]
struct Rotation(Mat3);

impl Rotation {
    /// Returns `Rᵀ·v`: a world-frame vector in the body frame.
    fn transpose_mul(&self, v: Vec3) -> Vec3 {
        core::array::from_fn(|c| (0..3).map(|r| self.0[r][c] * v[r]).sum())
    }

    /// Returns `R·v`: a body-frame vector in the world frame.
    fn mul(&self, v: Vec3) -> Vec3 {
        self.0.map(|row| dot(row, v))
    }

    /// Returns the yaw `ψ = atan2(R₁₀, R₀₀)` in `(−π, π]` radians.
    fn yaw(&self) -> f32 {
        self.0[1][0].atan2(self.0[0][0])
    }
}

/// Returns the skew-symmetric matrix `[v]×`, so that `[v]×·w = v × w`.
fn skew([x, y, z]: Vec3) -> Mat3 {
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}

/// Returns `mᵀ`.
fn transpose(m: &Mat3) -> Mat3 {
    core::array::from_fn(|r| core::array::from_fn(|c| m[c][r]))
}

/// Returns `m·v`.
fn mat_mul(m: &Mat3, v: Vec3) -> Vec3 {
    m.map(|row| dot(row, v))
}

/// Returns `a × b`.
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Returns `a · b`.
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Returns `|v|`.
fn norm(v: Vec3) -> f32 {
    dot(v, v).sqrt()
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

/// The segments of the rotation sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// At rest, level-ish: the filter learns the tilt and the bias.
    Rest,
    /// Rolling and pitching by up to about 35° while yawing, without linear acceleration.
    Tumble,
    /// A turn at 0.3 rad/s with 3 m/s² centripetal acceleration: `|f|` inside the ±10 % window.
    Turn,
    /// Horizontal shaking at 2 Hz and 2.5 m/s² with small rotations.
    Shake,
    /// Slow rotations on all axes, at rest.
    Slow,
}

impl Phase {
    /// Returns the phase at `t` seconds and the time since it began.
    fn at(t: f32) -> (Phase, f32) {
        match t {
            t if t < 5.0 => (Phase::Rest, t),
            t if t < 15.0 => (Phase::Tumble, t - 5.0),
            t if t < 30.0 => (Phase::Turn, t - 15.0),
            t if t < 38.0 => (Phase::Shake, t - 30.0),
            t => (Phase::Slow, t - 38.0),
        }
    }

    /// Returns a short name for the table.
    fn name(self) -> &'static str {
        match self {
            Phase::Rest => "rest",
            Phase::Tumble => "tumble",
            Phase::Turn => "turn",
            Phase::Shake => "shake",
            Phase::Slow => "slow",
        }
    }
}

/// Returns the true body angular velocity in rad/s at `t` seconds.
fn body_rate(t: f32) -> Vec3 {
    let wave = |amplitude: f32, hz: f32, s: f32| amplitude * (2.0 * PI * hz * s).sin();

    match Phase::at(t) {
        (Phase::Rest, _) => [0.0; 3],
        (Phase::Tumble, s) => [wave(0.5, 0.2, s), wave(0.4, 0.1, s), 0.3],
        (Phase::Turn, _) => [0.0, 0.0, 0.3],
        (Phase::Shake, s) => [wave(0.2, 1.5, s), wave(0.15, 1.1, s), 0.0],
        (Phase::Slow, s) => [wave(0.1, 0.05, s), wave(0.1, 0.07, s), wave(0.1, 0.03, s)],
    }
}

/// Returns the true world-frame linear acceleration in m/s² at `t` seconds.
fn world_acceleration(t: f32) -> Vec3 {
    match Phase::at(t) {
        (Phase::Turn, s) => {
            // Centripetal: perpendicular to the velocity, which turns at 0.3 rad/s.
            let heading = 0.3 * s;
            [-3.0 * heading.sin(), 3.0 * heading.cos(), 0.0]
        }
        (Phase::Shake, s) => [2.5 * (2.0 * PI * 2.0 * s).sin(), 0.0, 0.0],
        _ => [0.0; 3],
    }
}

/// The true state of the simulated IMU.
#[derive(Clone, Copy, Debug)]
struct Truth {
    /// The true attitude, body to world.
    attitude: Quat,
    /// The true gyro bias in rad/s.
    bias: Vec3,
}

/// One gyro sample with the optional accelerometer and yaw readings taken at the same time.
struct Sample {
    /// Timestamp in microseconds.
    time_us: u64,
    /// Time since the previous gyro sample in seconds.
    dt: f32,
    /// The gyro reading `ω + b + n` in rad/s, valid over the last `dt`.
    gyro: Vec3,
    /// The accelerometer reading in m/s², on every [`ACCEL_EVERY`]-th sample.
    accel: Option<Vec3>,
    /// The external yaw in radians, on every [`YAW_EVERY`]-th sample.
    yaw: Option<f32>,
    /// The true world-frame acceleration at this time, for the statistics.
    world_accel: Vec3,
    /// The truth at this time.
    truth: Truth,
}

/// The simulated IMU: truth, timestamps and sensor noise from one seeded generator.
struct Simulation {
    /// The noise source.
    rng: StdRng,
    /// The current truth.
    truth: Truth,
    /// The timestamp of the last gyro sample in microseconds.
    time_us: u64,
    /// The number of gyro samples so far.
    count: usize,
}

impl Simulation {
    /// Starts at rest in the initial attitude with the initial bias.
    fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            truth: Truth {
                attitude: Quat::from_euler(INITIAL_EULER),
                bias: INITIAL_BIAS,
            },
            time_us: 0,
            count: 0,
        }
    }

    /// Advances to the next (jittered, possibly delayed) gyro timestamp; `None` at the end.
    ///
    /// The true rate is held over the interval, so the truth is integrated exactly and the
    /// gyro reports that rate plus bias and white noise of density [`GYRO_NOISE_DENSITY`].
    fn step(&mut self) -> Option<Sample> {
        let start = self.time_us;
        let skipped = u64::from(self.rng.gen_bool(DROP_PROBABILITY));
        let jitter = self.rng.gen_range(0..=2 * GYRO_JITTER_US);
        let next = start + (1 + skipped) * GYRO_PERIOD_US + jitter - GYRO_JITTER_US;
        if next > DURATION_US {
            return None;
        }

        #[allow(clippy::cast_precision_loss)]
        let (t, dt) = (start as f32 * 1e-6, (next - start) as f32 * 1e-6);
        let omega = body_rate(t);
        let gyro = self.gyro_sample(omega, dt);
        self.advance_truth(omega, dt);
        self.time_us = next;
        self.count += 1;

        Some(Sample {
            time_us: next,
            dt,
            gyro,
            accel: (self.count % ACCEL_EVERY == 0).then(|| self.accel_sample()),
            yaw: (self.count % YAW_EVERY == 0).then(|| self.yaw_sample()),
            #[allow(clippy::cast_precision_loss)]
            world_accel: world_acceleration(next as f32 * 1e-6),
            truth: self.truth,
        })
    }

    /// Returns the gyro reading for the true rate `omega` held over `dt` seconds.
    fn gyro_sample(&mut self, omega: Vec3, dt: f32) -> Vec3 {
        let noise = Normal::new(0.0, GYRO_NOISE_DENSITY / dt.sqrt()).expect("a valid deviation");
        core::array::from_fn(|i| omega[i] + self.truth.bias[i] + noise.sample(&mut self.rng))
    }

    /// Rotates the truth by `omega·dt` and lets the bias walk.
    fn advance_truth(&mut self, omega: Vec3, dt: f32) {
        let walk = Normal::new(0.0, GYRO_BIAS_WALK * dt.sqrt()).expect("a valid deviation");
        let step = Quat::exp(omega.map(|w| w * dt));

        self.truth.attitude = self.truth.attitude.mul(&step).normalized();
        self.truth.bias = self.truth.bias.map(|b| b + walk.sample(&mut self.rng));
    }

    /// Returns the accelerometer reading `Rᵀ·(a + g) + n` at the current time.
    fn accel_sample(&mut self) -> Vec3 {
        let noise = Normal::new(0.0, ACCEL_NOISE_STD).expect("a valid deviation");
        #[allow(clippy::cast_precision_loss)]
        let a = world_acceleration(self.time_us as f32 * 1e-6);
        let force = [a[0], a[1], a[2] + GRAVITY];
        let body = self.truth.attitude.rotation().transpose_mul(force);

        body.map(|f| f + noise.sample(&mut self.rng))
    }

    /// Returns the external yaw reading at the current time, wrapped to `(−π, π]`.
    fn yaw_sample(&mut self) -> f32 {
        let noise = Normal::new(0.0, YAW_NOISE_STD).expect("a valid deviation");
        wrap_pi(self.truth.attitude.rotation().yaw() + noise.sample(&mut self.rng))
    }
}

/// Accelerometer decisions of one class of samples (static or dynamic).
#[derive(Clone, Copy, Debug, Default)]
struct Decisions {
    /// Number of accelerometer samples.
    samples: u32,
    /// Number the NIS gate rejected.
    gate_rejected: u32,
    /// Number skipped by the hold-off after a gated sample.
    held: u32,
    /// Number the ±10 % window would reject.
    window_rejected: u32,
}

impl Decisions {
    /// Returns the share of samples the gate rejected.
    fn gate_rate(&self) -> f32 {
        self.share(self.gate_rejected)
    }

    /// Returns the share of samples that did not reach the filter: gated or held off.
    fn skip_rate(&self) -> f32 {
        self.share(self.gate_rejected + self.held)
    }

    /// Returns `count` as a share of all samples.
    fn share(&self, count: u32) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let rate = count as f32 / self.samples.max(1) as f32;
        rate
    }
}

/// Running statistics of the gated filter and the comparison filter.
#[derive(Debug, Default)]
struct Stats {
    /// Accelerometer decisions on samples without linear acceleration.
    static_samples: Decisions,
    /// Accelerometer decisions on samples with linear acceleration.
    dynamic_samples: Decisions,
    /// Sum of the NIS of the accepted tilt updates.
    nis_sum: f32,
    /// Number of accepted tilt updates.
    nis_count: u32,
    /// Sum of the attitude NEES after settling.
    nees_attitude_sum: f32,
    /// Sum of the bias NEES after settling.
    nees_bias_sum: f32,
    /// Number of samples after settling.
    error_count: u32,
    /// Sum of the squared attitude errors of the gated filter after settling, in rad².
    gated_sq_sum: f32,
    /// Largest attitude error of the gated filter after settling, in radians.
    gated_max: f32,
    /// Largest attitude error of the window filter after settling, in radians.
    window_max: f32,
    /// Number of heading updates.
    yaw_samples: u32,
    /// Number of heading updates the yaw gate rejected, for the gated and the window filter.
    yaw_rejected: [u32; 2],
}

impl Stats {
    /// Records one accelerometer decision of the gated filter and what the window would do.
    fn record_accel(
        &mut self,
        sample: &Sample,
        status: Option<CorrectionStatus>,
        nis: f32,
        inside: bool,
    ) {
        let class = if norm(sample.world_accel) > DYNAMIC_THRESHOLD {
            &mut self.dynamic_samples
        } else {
            &mut self.static_samples
        };
        class.samples += 1;
        class.gate_rejected += u32::from(status == Some(CorrectionStatus::Gated));
        class.held += u32::from(status.is_none());
        class.window_rejected += u32::from(!inside);

        if status.is_some_and(CorrectionStatus::is_applied) {
            self.nis_sum += nis;
            self.nis_count += 1;
        }
    }

    /// Records the yaw gate decisions of the gated and the window filter.
    fn record_yaw(&mut self, statuses: [CorrectionStatus; 2]) {
        self.yaw_samples += 1;
        for (count, status) in self.yaw_rejected.iter_mut().zip(statuses) {
            *count += u32::from(status == CorrectionStatus::Gated);
        }
    }

    /// Records the attitude errors of both filters and the NEES of the gated one.
    fn record_errors(&mut self, truth: &Truth, gated: &Eskf<'_>, window: &Eskf<'_>) {
        let gated_error = norm(gated.attitude_error(&truth.attitude));
        let (nees_attitude, nees_bias) = gated.nees(truth);

        self.gated_sq_sum += gated_error * gated_error;
        self.gated_max = self.gated_max.max(gated_error);
        self.window_max = self
            .window_max
            .max(norm(window.attitude_error(&truth.attitude)));
        self.nees_attitude_sum += nees_attitude;
        self.nees_bias_sum += nees_bias;
        self.error_count += 1;
    }

    /// Returns the mean NIS of the accepted tilt updates.
    fn mean_nis(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let mean = self.nis_sum / self.nis_count.max(1) as f32;
        mean
    }

    /// Returns the mean attitude NEES and bias NEES after settling.
    fn mean_nees(&self) -> (f32, f32) {
        #[allow(clippy::cast_precision_loss)]
        let n = self.error_count.max(1) as f32;
        (self.nees_attitude_sum / n, self.nees_bias_sum / n)
    }

    /// Returns the RMS attitude error of the gated filter after settling, in radians.
    fn rms(&self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let rms = (self.gated_sq_sum / self.error_count.max(1) as f32).sqrt();
        rms
    }

    /// Prints the rejection table and the consistency statistics.
    fn report(&self) {
        println!(
            "\nAccelerometer samples  count  gated (rate)      held off  skipped  ±10 % window rejects"
        );
        for (name, d) in [
            ("static", self.static_samples),
            ("dynamic", self.dynamic_samples),
        ] {
            println!(
                "  {name:8}           {:5}  {:5} ({:5.1} %)  {:8}  {:5.1} %  {:5}",
                d.samples,
                d.gate_rejected,
                100.0 * d.gate_rate(),
                d.held,
                100.0 * d.skip_rate(),
                d.window_rejected,
            );
        }

        let (nees_attitude, nees_bias) = self.mean_nees();
        println!(
            "\nAfter {:.0} s: attitude error RMS {:.3}°, max {:.3}° (gated) vs. max {:.2}° (window)",
            SETTLE_US as f32 * 1e-6,
            self.rms().to_degrees(),
            self.gated_max.to_degrees(),
            self.window_max.to_degrees(),
        );
        println!(
            "Mean NEES δθ {nees_attitude:.2}, δb {nees_bias:.2} (expected 3); mean NIS of accepted tilt updates {:.2} (expected just below 3)",
            self.mean_nis(),
        );
        println!(
            "Heading updates rejected by the yaw gate: {} of {} (gated), {} (window)",
            self.yaw_rejected[0], self.yaw_samples, self.yaw_rejected[1],
        );
    }
}

/// Prints one row of the progress table.
fn print_row(sample: &Sample, gated: &Eskf<'_>, window: &Eskf<'_>, stats: &Stats) {
    #[allow(clippy::cast_precision_loss)]
    let t = sample.time_us as f32 * 1e-6;
    let truth = &sample.truth;
    let bias_error: Vec3 = core::array::from_fn(|i| truth.bias[i] - gated.bias[i]);
    let [st, dy] = [stats.static_samples, stats.dynamic_samples];
    let skipped = st.gate_rejected + st.held + dy.gate_rejected + dy.held;
    let used = st.samples + dy.samples - skipped;

    println!(
        "{t:6.1}   {:7}  {:8.3}°  {:9.3}°  {:7.4} rad/s  {:5}/{}",
        Phase::at(t).0.name(),
        norm(gated.attitude_error(&truth.attitude)).to_degrees(),
        norm(window.attitude_error(&truth.attitude)).to_degrees(),
        norm(bias_error),
        used,
        skipped,
    );
}

/// Returns success if the gated filter met every bound; prints the first failure otherwise.
fn check(stats: &Stats, truth: &Truth, gated: &Eskf<'_>) -> ExitCode {
    let final_error = norm(gated.attitude_error(&truth.attitude)).to_degrees();
    let (nees_attitude, nees_bias) = stats.mean_nees();
    let inside = |v: f32, (lo, hi): (f32, f32)| (lo..=hi).contains(&v);

    let checks = [
        (
            final_error <= FINAL_ATTITUDE_ERROR_DEG,
            "final attitude error",
        ),
        (
            stats.gated_max.to_degrees() <= MAX_ATTITUDE_ERROR_DEG,
            "largest attitude error",
        ),
        (inside(nees_attitude, NEES_RANGE), "mean NEES of δθ"),
        (inside(nees_bias, NEES_RANGE), "mean NEES of δb"),
        (
            inside(stats.mean_nis(), NIS_RANGE),
            "mean NIS of the tilt updates",
        ),
        (
            stats.dynamic_samples.skip_rate() >= MIN_DYNAMIC_REJECTION,
            "dynamic rejection rate",
        ),
        (
            stats.static_samples.gate_rate() <= MAX_STATIC_REJECTION,
            "static rejection rate",
        ),
    ];

    if let Some((_, what)) = checks.iter().find(|(ok, _)| !ok) {
        eprintln!("Attitude ESKF out of bounds: {what}");
        return ExitCode::FAILURE;
    }

    println!("Attitude ESKF converged: final attitude error {final_error:.3}°");
    ExitCode::SUCCESS
}

/// Asserts that the quaternion helpers and the Jacobians match their definitions.
///
/// `Log(Exp(θ)) = θ` and `R(Exp(θ))·v = Rodrigues`, then finite differences of the
/// accelerometer prediction, the yaw, the error propagation and the reset against
/// [`gravity_jacobian`], [`yaw_jacobian`], [`attitude_transition`] and [`reset_jacobian`].
fn check_helpers() {
    let thetas = [[0.3, -0.2, 1.1], [1e-4, 2e-4, -1e-4], [-1.0, 2.0, -0.5]];
    for theta in thetas {
        assert_close(Quat::exp(theta).log(), theta, 1e-5, "Log(Exp(θ))");
        assert_close(
            Quat::exp(theta).rotation().mul([1.0, 0.0, 0.0]),
            rodrigues(theta, [1.0, 0.0, 0.0]),
            1e-5,
            "R(Exp(θ))",
        );
    }

    let nominal = Quat::from_euler([0.4, -0.3, 2.0]);
    let delta = [1e-3, -2e-3, 1.5e-3];
    let perturbed = nominal.mul(&Quat::exp(delta));

    let gravity = [0.0, 0.0, GRAVITY];
    let predicted = nominal.rotation().transpose_mul(gravity);
    let moved = perturbed.rotation().transpose_mul(gravity);
    let linear = mat_mul(&gravity_jacobian(predicted), delta);
    assert_close(
        core::array::from_fn(|i| moved[i] - predicted[i]),
        linear,
        1e-4,
        "tilt H",
    );

    let yaw_change = wrap_pi(perturbed.rotation().yaw() - nominal.rotation().yaw());
    let yaw_linear = dot(yaw_jacobian(&nominal.rotation()), delta);
    assert!(
        (yaw_change - yaw_linear).abs() < 1e-5,
        "yaw H self-check failed"
    );

    // Error propagation: (q̂ ⊗ Exp(δθ)) ⊗ Exp(u) = (q̂ ⊗ Exp(u)) ⊗ Exp(F_θθ·δθ).
    let u = [0.02, -0.01, 0.03];
    let after = nominal
        .mul(&Quat::exp(u))
        .conjugate()
        .mul(&perturbed.mul(&Quat::exp(u)))
        .log();
    assert_close(after, mat_mul(&attitude_transition(u), delta), 1e-6, "F_θθ");

    // Reset: an error δ around q̂ ⊗ Exp(δθ) is G·δ around the reset nominal, to first order.
    let dtheta = [0.05, -0.04, 0.03];
    let reset_nominal = nominal.mul(&Quat::exp(dtheta));
    let shifted: Vec3 = core::array::from_fn(|i| dtheta[i] + delta[i]);
    let new_error = reset_nominal
        .conjugate()
        .mul(&nominal.mul(&Quat::exp(shifted)))
        .log();
    assert_close(
        new_error,
        mat_mul(&reset_jacobian(dtheta), delta),
        2e-5,
        "reset G",
    );
}

/// Returns `v` rotated by `θ` with the Rodrigues formula, for [`check_helpers`].
fn rodrigues(theta: Vec3, v: Vec3) -> Vec3 {
    let angle = norm(theta);
    let k = theta.map(|x| x / angle);
    let (s, c) = angle.sin_cos();
    let kxv = cross(k, v);
    let kdv = dot(k, v);

    core::array::from_fn(|i| v[i] * c + kxv[i] * s + k[i] * kdv * (1.0 - c))
}

/// Panics with `what` if any entry of `a` and `b` differs by more than `tolerance`.
fn assert_close(a: Vec3, b: Vec3, tolerance: f32, what: &str) {
    let close = a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance);
    assert!(close, "{what} self-check failed: {a:?} vs {b:?}");
}
