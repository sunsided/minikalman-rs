//! Attitude: a quaternion error-state EKF with an NIS-gated accelerometer, next to the same
//! filter with a `±10 % |a|` window instead of the gate.
//!
//! A condensed port of the `attitude-eskf` example. The filter keeps a nominal quaternion
//! `q̂` and gyro bias `b̂`; the six-state `f32` EKF estimates only the small error
//! `(δθ, δb)`. The gyro predicts at a jittered 200 Hz, the accelerometer corrects the tilt
//! at 50 Hz (`H = [[f̂]× 0]`), an external heading corrects the yaw at 5 Hz, and every
//! correction is injected into `q̂` and reset.
//!
//! The accelerometer measures gravity only while the body does not accelerate. In the turn
//! (15 s to 30 s) and the shake (30 s to 38 s), the window lets most dynamic samples
//! through, since `|g + a|` stays close to `g`; the gate looks at the direction of the
//! innovation instead and rejects them.

use core::f32::consts::PI;

use minikalman::angles::wrap_pi;
use minikalman::extended::{
    BorrowedKalman, BorrowedObservation, FilterStorage, ObservationStorage,
};
use minikalman::prelude::*;

use crate::lab::{Noise, Trace, knob, record};

/// Number of error states: `δθ` (3) and `δb` (3).
const STATES: usize = 6;

/// A 3×3 matrix, row-major.
type Mat3 = [[f32; 3]; 3];

/// A 3-vector.
type Vec3 = [f32; 3];

/// Duration of a run in microseconds.
const DURATION_US: u64 = 60_000_000;

/// Nominal gyro period in microseconds (200 Hz).
const GYRO_PERIOD_US: u64 = 5_000;

/// Largest timestamp jitter in microseconds.
const GYRO_JITTER_US: u64 = 1_000;

/// Probability that a gyro sample is dropped.
const DROP_PROBABILITY: f64 = 0.01;

/// An accelerometer sample comes with every this many gyro samples (50 Hz).
const ACCEL_EVERY: usize = 4;

/// A heading comes with every this many gyro samples (5 Hz).
const YAW_EVERY: usize = 40;

/// Standard gravity in m/s².
const GRAVITY: f32 = 9.81;

/// Gyro white noise density in rad/s/√Hz.
const GYRO_NOISE_DENSITY: f32 = 3e-4;

/// Gyro bias random walk in rad/s/√s.
const GYRO_BIAS_WALK: f32 = 1e-4;

/// The true gyro bias at start-up in rad/s.
const INITIAL_BIAS: Vec3 = [0.03, -0.02, 0.01];

/// Accelerometer noise per axis in m/s².
const ACCEL_NOISE_STD: f32 = 0.05;

/// Heading noise in radians (2°).
const YAW_NOISE_STD: f32 = 2.0 * PI / 180.0;

/// The true attitude at start-up as (roll, pitch, yaw).
const INITIAL_EULER: Vec3 = [3.0 * PI / 180.0, -2.0 * PI / 180.0, 40.0 * PI / 180.0];

/// Prior standard deviation of roll and pitch (2°).
const PRIOR_TILT_STD: f32 = 2.0 * PI / 180.0;

/// Prior standard deviation of the heading (5°).
const PRIOR_YAW_STD: f32 = 5.0 * PI / 180.0;

/// Prior standard deviation of each gyro bias axis in rad/s.
const PRIOR_BIAS_STD: f32 = 0.05;

/// Heading gate: the 99.9 % chi-square quantile for one degree of freedom.
const YAW_GATE: f32 = 10.83;

/// Statistics start after this many microseconds, once the bias has settled.
const SETTLE_US: u64 = 5_000_000;

/// A world-frame acceleration above this many m/s² marks a sample as dynamic.
const DYNAMIC_THRESHOLD: f32 = 0.3;

/// The recorded samples: one per accelerometer sample.
const SAMPLES: usize = (DURATION_US / GYRO_PERIOD_US) as usize / ACCEL_EVERY;

/// The trace channels, in order.
mod channel {
    /// Time in seconds.
    pub const TIME: usize = 0;
    /// True roll, pitch, yaw in degrees.
    pub const TRUTH: usize = 1;
    /// Gated filter roll, pitch, yaw in degrees.
    pub const GATED: usize = 4;
    /// Window filter roll, pitch, yaw in degrees.
    pub const WINDOW: usize = 7;
    /// Attitude error of the gated, then the window filter, in degrees.
    pub const ERROR: usize = 10;
    /// Accelerometer decision of the gated filter: 1 applied, 2 gated, 3 held off.
    pub const GATE: usize = 12;
    /// `1` if the window filter used the accelerometer sample.
    pub const WINDOW_USED: usize = 13;
    /// The true world-frame acceleration magnitude in m/s².
    pub const ACCELERATION: usize = 14;
    /// The NIS of the gated filter's accelerometer innovation (NaN if held off).
    pub const NIS: usize = 15;
    /// Number of channels.
    pub const COUNT: usize = 16;
}

/// The settings of a run.
#[derive(Clone, Copy)]
struct Settings {
    /// Scale of the dynamic acceleration in the turn and the shake.
    dynamics: f32,
    /// NIS gate of the tilt update (chi-square, three degrees of freedom).
    gate: f32,
    /// Accelerometer samples skipped after a gated one.
    hold_off: u32,
    /// Relative `|f|` window of the comparison filter.
    window: f32,
    /// Whether the external heading is used.
    heading: bool,
}

/// Simulates one run and writes it to the trace.
///
/// Channels: see [`channel`], one sample per accelerometer reading (50 Hz). Summary:
/// `[RMS attitude error after 5 s (gated, window), largest attitude error after 5 s (gated,
/// window), share of dynamic samples the gated filter skipped, share of static samples it
/// gated, share of dynamic samples the window accepted, mean NIS of the accepted tilt
/// updates, final gyro bias error (gated) in °/s]`.
///
/// * `seed` - Noise seed.
/// * `dynamics` - Scale of the turn's and the shake's acceleration, `0..=3`.
/// * `gate` - Tilt gate, `1..=1000`; the 99 % quantile is `11.34`.
/// * `hold_off` - Accelerometer samples skipped after a gated one, `0..=50`.
/// * `window` - Relative `|f|` window of the comparison filter, `0.01..=0.5`.
/// * `heading` - Use the external heading (non-zero) or not.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_lines)]
pub extern "C" fn attitude_run(
    seed: u32,
    dynamics: f32,
    gate: f32,
    hold_off: u32,
    window: f32,
    heading: u32,
) -> u32 {
    let settings = Settings {
        dynamics: knob(dynamics, 0.0, 3.0),
        gate: knob(gate, 1.0, 1000.0),
        hold_off: hold_off.min(50),
        window: knob(window, 0.01, 0.5),
        heading: heading != 0,
    };

    let mut sim = Simulation::new(seed, settings.dynamics);
    let guess = initial_attitude(sim.accel_sample(), sim.yaw_sample());
    let mut gated_storage = Storage::new();
    let mut window_storage = Storage::new();

    record(channel::COUNT, SAMPLES, |trace, summary| {
        let mut gated = gated_storage.build(guess, Some(settings.gate));
        let mut window = window_storage.build(guess, None);
        let mut stats = Stats::default();
        let mut k = 0;
        while let Some(sample) = sim.step() {
            gated.predict(sample.gyro, sample.dt);
            window.predict(sample.gyro, sample.dt);
            if let Some(yaw) = sample.yaw.filter(|_| settings.heading) {
                gated.correct_yaw(yaw);
                window.correct_yaw(yaw);
            }
            let Some(f) = sample.accel else { continue };
            if k >= SAMPLES {
                break;
            }

            let status = gated.correct_accel(f, settings.hold_off);
            let within = (norm(f) - GRAVITY).abs() <= settings.window * GRAVITY;
            if within {
                window.correct_accel(f, 0);
            }
            stats.record(&sample, status, within, gated.accel.nis());
            write_sample(trace, k, &sample, &gated, &window, status, within);
            if sample.time_us >= SETTLE_US {
                stats.record_errors(
                    trace.get(channel::ERROR, k),
                    trace.get(channel::ERROR + 1, k),
                );
            }
            k += 1;
        }
        // Fill any samples the jittered clock did not reach.
        for j in k..SAMPLES {
            for c in 0..channel::COUNT {
                trace.set(c, j, f32::NAN);
            }
        }

        let bias_error: Vec3 = core::array::from_fn(|i| sim.truth.bias[i] - gated.bias[i]);
        stats.summarize(summary);
        summary[9] = norm(bias_error).to_degrees();
    })
}

/// Writes the recorded channels of sample `k`.
#[allow(clippy::cast_precision_loss)]
fn write_sample(
    trace: &mut Trace<'_>,
    k: usize,
    sample: &Sample,
    gated: &Eskf<'_>,
    window: &Eskf<'_>,
    status: Option<CorrectionStatus>,
    within: bool,
) {
    trace.set(channel::TIME, k, sample.time_us as f32 * 1e-6);
    for (base, attitude) in [
        (channel::TRUTH, &sample.truth.attitude),
        (channel::GATED, &gated.attitude),
        (channel::WINDOW, &window.attitude),
    ] {
        let euler = attitude.euler();
        for (i, angle) in euler.iter().enumerate() {
            trace.set(base + i, k, angle.to_degrees());
        }
    }
    let error = |eskf: &Eskf<'_>| {
        norm(eskf.attitude.conjugate().mul(&sample.truth.attitude).log()).to_degrees()
    };
    trace.set(channel::ERROR, k, error(gated));
    trace.set(channel::ERROR + 1, k, error(window));
    let decision = match status {
        None => 3.0,
        Some(CorrectionStatus::Gated) => 2.0,
        Some(_) => 1.0,
    };
    trace.set(channel::GATE, k, decision);
    trace.set(channel::WINDOW_USED, k, f32::from(u8::from(within)));
    trace.set(channel::ACCELERATION, k, norm(sample.world_accel));
    trace.set(
        channel::NIS,
        k,
        if status.is_some() {
            gated.accel.nis()
        } else {
            f32::NAN
        },
    );
}

/// The accelerometer statistics of a run.
#[derive(Default)]
struct Stats {
    /// Dynamic samples, and those the gated filter skipped (gated or held off).
    dynamic: (u32, u32),
    /// Static samples the gate saw, and those it gated.
    static_gated: (u32, u32),
    /// Dynamic samples the window accepted.
    window_dynamic: u32,
    /// Sum and count of the NIS of accepted tilt updates after settling.
    nis: (f64, u32),
    /// Sum of squares, maximum and count of the attitude errors (gated, window).
    errors: ([f64; 2], [f32; 2], u32),
}

impl Stats {
    /// Records the decisions on one accelerometer sample.
    fn record(
        &mut self,
        sample: &Sample,
        status: Option<CorrectionStatus>,
        within: bool,
        nis: f32,
    ) {
        let dynamic = norm(sample.world_accel) > DYNAMIC_THRESHOLD;
        let gated = status == Some(CorrectionStatus::Gated);
        if dynamic {
            self.dynamic.0 += 1;
            self.dynamic.1 += u32::from(status.is_none() || gated);
            self.window_dynamic += u32::from(within);
        } else if status.is_some() {
            self.static_gated.0 += 1;
            self.static_gated.1 += u32::from(gated);
        }
        if status == Some(CorrectionStatus::Applied) && sample.time_us >= SETTLE_US {
            self.nis.0 += f64::from(nis);
            self.nis.1 += 1;
        }
    }

    /// Records the attitude errors in degrees of one sample after settling.
    fn record_errors(&mut self, gated: f32, window: f32) {
        for (i, e) in [gated, window].into_iter().enumerate() {
            self.errors.0[i] += f64::from(e) * f64::from(e);
            self.errors.1[i] = self.errors.1[i].max(e);
        }
        self.errors.2 += 1;
    }

    /// Writes the first nine summary values.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn summarize(&self, summary: &mut [f32]) {
        let n = f64::from(self.errors.2.max(1));
        summary[0] = (self.errors.0[0] / n).sqrt() as f32;
        summary[1] = (self.errors.0[1] / n).sqrt() as f32;
        summary[2] = self.errors.1[0];
        summary[3] = self.errors.1[1];
        let share = |part: u32, whole: u32| {
            if whole == 0 {
                f32::NAN
            } else {
                part as f32 / whole as f32
            }
        };
        summary[4] = share(self.dynamic.1, self.dynamic.0);
        summary[5] = share(self.static_gated.1, self.static_gated.0);
        summary[6] = share(self.window_dynamic, self.dynamic.0);
        summary[7] = if self.nis.1 == 0 {
            f32::NAN
        } else {
            (self.nis.0 / f64::from(self.nis.1)) as f32
        };
    }
}

/// Estimates roll and pitch from an accelerometer reading at rest, and takes the yaw.
fn initial_attitude(f: Vec3, yaw: f32) -> Quat {
    let roll = f[1].atan2(f[2]);
    let pitch = (-f[0]).atan2(f[1].hypot(f[2]));
    Quat::from_euler([roll, pitch, yaw])
}

/// The statically allocated buffers of one filter.
struct Storage {
    /// The six-state filter.
    filter: FilterStorage<STATES, f32>,
    /// The accelerometer observation.
    accel: ObservationStorage<STATES, 3, f32>,
    /// The heading observation.
    yaw: ObservationStorage<STATES, 1, f32>,
}

impl Storage {
    /// Creates zeroed buffers.
    const fn new() -> Self {
        Self {
            filter: FilterStorage::new(0.0),
            accel: ObservationStorage::new(0.0),
            yaw: ObservationStorage::new(0.0),
        }
    }

    /// Builds a filter at `attitude` with zero bias and the prior; `accel_gate` gates the tilt.
    fn build(&mut self, attitude: Quat, accel_gate: Option<f32>) -> Eskf<'_> {
        let mut filter = self.filter.build();
        let mut accel = self.accel.build();
        let mut yaw = self.yaw.build();

        // The prior is given in the world frame (tilt, heading); δθ is in the body frame.
        let r = attitude.rotation().0;
        let variance = [PRIOR_TILT_STD, PRIOR_TILT_STD, PRIOR_YAW_STD].map(|s| s * s);
        let prior: Mat3 = core::array::from_fn(|i| {
            core::array::from_fn(|j| (0..3).map(|k| r[k][i] * variance[k] * r[k][j]).sum())
        });
        filter.estimate_covariance_mut().apply(|p| {
            write_block(p, 0, 0, &prior);
            (3..STATES).for_each(|i| p.set_at(i, i, PRIOR_BIAS_STD * PRIOR_BIAS_STD));
        });
        accel
            .measurement_noise_covariance_mut()
            .apply(|r| (0..3).for_each(|i| r.set_at(i, i, ACCEL_NOISE_STD * ACCEL_NOISE_STD)));
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

/// An error-state filter: the EKF of `(δθ, δb)` and the nominal attitude and bias.
struct Eskf<'a> {
    /// The error-state filter.
    filter: BorrowedKalman<'a, STATES, f32>,
    /// The accelerometer observation.
    accel: BorrowedObservation<'a, STATES, 3, f32>,
    /// The heading observation.
    yaw: BorrowedObservation<'a, STATES, 1, f32>,
    /// The nominal attitude, body to world.
    attitude: Quat,
    /// The nominal gyro bias in rad/s.
    bias: Vec3,
    /// Accelerometer samples still to skip after a gated one.
    hold_off: u32,
}

impl Eskf<'_> {
    /// Propagates the nominal attitude with the gyro and `P` with the error dynamics.
    fn predict(&mut self, gyro: Vec3, dt: f32) {
        let u: Vec3 = core::array::from_fn(|i| (gyro[i] - self.bias[i]) * dt);
        self.attitude = self.attitude.mul(&Quat::exp(u)).normalized();
        let rotation = transpose(&Quat::exp(u).rotation().0);
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
        self.filter
            .predict_nonlinear(|_, next| (0..STATES).for_each(|i| next.set_row(i, 0.0)));
    }

    /// Corrects the tilt with an accelerometer reading, unless held off after a gated one;
    /// `hold_off` is the number of samples to skip after this one if it is gated.
    fn correct_accel(&mut self, f: Vec3, hold_off: u32) -> Option<CorrectionStatus> {
        if self.hold_off > 0 {
            self.hold_off -= 1;
            return None;
        }
        let predicted = self.attitude.rotation().transpose_mul([0.0, 0.0, GRAVITY]);
        let jacobian = skew(predicted);
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
            self.hold_off = hold_off;
        }
        self.reset();
        Some(status)
    }

    /// Corrects the heading with an external yaw, wrapping the innovation.
    fn correct_yaw(&mut self, yaw: f32) -> CorrectionStatus {
        let rotation = self.attitude.rotation();
        let r = &rotation.0;
        let c = r[0][0] * r[0][0] + r[1][0] * r[1][0];
        let world = [-r[0][0] * r[2][0] / c, -r[1][0] * r[2][0] / c, 1.0];
        // h_world · R̂, since the world-frame error is φ = R̂·δθ.
        let jacobian: Vec3 = core::array::from_fn(|j| (0..3).map(|i| world[i] * r[i][j]).sum());
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

    /// Injects the error into the nominal state, resets it, and moves `P` with the reset
    /// Jacobian `G = diag(I − ½[δθ]×, I)`.
    fn reset(&mut self) {
        let x = self.filter.state_vector();
        let dtheta = [x.get_row(0), x.get_row(1), x.get_row(2)];
        let dbias = [x.get_row(3), x.get_row(4), x.get_row(5)];
        self.attitude = self.attitude.mul(&Quat::exp(dtheta)).normalized();
        self.bias = core::array::from_fn(|i| self.bias[i] + dbias[i]);
        let s = skew(dtheta);
        let g = |r: usize, c: usize| -> f32 {
            match (r < 3, c < 3) {
                (true, true) => f32::from(u8::from(r == c)) - 0.5 * s[r][c],
                (false, false) => f32::from(u8::from(r == c)),
                _ => 0.0,
            }
        };
        self.filter.estimate_covariance_mut().apply(|p| {
            let old: [[f32; STATES]; STATES] =
                core::array::from_fn(|r| core::array::from_fn(|c| p.get_at(r, c)));
            for r in 0..STATES {
                for c in 0..STATES {
                    let sum = old
                        .iter()
                        .enumerate()
                        .flat_map(|(i, row)| row.iter().enumerate().map(move |(k, v)| (i, k, v)))
                        .map(|(i, k, v)| g(r, i) * v * g(c, k))
                        .sum();
                    p.set_at(r, c, sum);
                }
            }
        });
        self.filter
            .state_vector_mut()
            .apply(|v| (0..STATES).for_each(|i| v.set_row(i, 0.0)));
    }
}

/// Writes a 3×3 block into `m` at `(row, col)`.
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

/// A unit quaternion `w + v`, body to world.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Quat {
    /// The scalar part.
    w: f32,
    /// The vector part.
    v: Vec3,
}

impl Quat {
    /// The Hamilton product `self ⊗ other`.
    fn mul(&self, other: &Quat) -> Quat {
        let (a, b) = (self.v, other.v);
        let c = cross(a, b);
        Quat {
            w: self.w * other.w - dot(a, b),
            v: core::array::from_fn(|i| self.w * b[i] + other.w * a[i] + c[i]),
        }
    }

    /// The conjugate, i.e. the inverse rotation.
    fn conjugate(&self) -> Quat {
        Quat {
            w: self.w,
            v: self.v.map(|x| -x),
        }
    }

    /// Scales to unit length.
    fn normalized(&self) -> Quat {
        let n = (self.w * self.w + dot(self.v, self.v)).sqrt();
        Quat {
            w: self.w / n,
            v: self.v.map(|x| x / n),
        }
    }

    /// The rotation by the rotation vector `theta`.
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

    /// The rotation vector of this rotation, the short way round.
    fn log(&self) -> Vec3 {
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

    /// The rotation `Rz(yaw)·Ry(pitch)·Rx(roll)`.
    fn from_euler([roll, pitch, yaw]: Vec3) -> Quat {
        let qx = Quat::exp([roll, 0.0, 0.0]);
        let qy = Quat::exp([0.0, pitch, 0.0]);
        let qz = Quat::exp([0.0, 0.0, yaw]);
        qz.mul(&qy).mul(&qx)
    }

    /// Returns (roll, pitch, yaw) in radians.
    fn euler(&self) -> Vec3 {
        let r = self.rotation().0;
        [
            r[2][1].atan2(r[2][2]),
            (-r[2][0]).clamp(-1.0, 1.0).asin(),
            r[1][0].atan2(r[0][0]),
        ]
    }

    /// The rotation matrix.
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

/// A rotation matrix, body to world.
#[derive(Clone, Copy, Debug)]
struct Rotation(Mat3);

impl Rotation {
    /// Returns `Rᵀ·v`, i.e. `v` in the body frame.
    fn transpose_mul(&self, v: Vec3) -> Vec3 {
        core::array::from_fn(|c| (0..3).map(|r| self.0[r][c] * v[r]).sum())
    }

    /// Returns the heading `atan2(R₁₀, R₀₀)`.
    fn yaw(&self) -> f32 {
        self.0[1][0].atan2(self.0[0][0])
    }
}

/// The skew-symmetric matrix `[v]×`.
fn skew([x, y, z]: Vec3) -> Mat3 {
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}

/// The transpose of `m`.
fn transpose(m: &Mat3) -> Mat3 {
    core::array::from_fn(|r| core::array::from_fn(|c| m[c][r]))
}

/// The cross product `a × b`.
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The dot product `a · b`.
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The Euclidean norm of `v`.
fn norm(v: Vec3) -> f32 {
    dot(v, v).sqrt()
}

/// The true body rate in rad/s at `t` seconds: rest, tumble, turn, shake, slow motion.
fn body_rate(t: f32) -> Vec3 {
    // A cosine rate integrates to a sine angle, so each axis swings about where it started.
    let wave = |amplitude: f32, hz: f32, s: f32| amplitude * (2.0 * PI * hz * s).cos();
    match t {
        t if t < 5.0 => [0.0; 3],
        t if t < 15.0 => [wave(0.5, 0.2, t - 5.0), wave(0.3, 0.1, t - 5.0), 0.3],
        t if t < 30.0 => [0.0, 0.0, 0.3],
        t if t < 38.0 => [wave(0.2, 1.5, t - 30.0), wave(0.15, 1.1, t - 30.0), 0.0],
        t => [
            wave(0.1, 0.05, t - 38.0),
            wave(0.1, 0.07, t - 38.0),
            wave(0.1, 0.03, t - 38.0),
        ],
    }
}

/// The true world-frame linear acceleration in m/s² at `t` seconds, scaled by `scale`.
fn world_acceleration(t: f32, scale: f32) -> Vec3 {
    let a = match t {
        t if (15.0..30.0).contains(&t) => {
            // Centripetal: perpendicular to the velocity, which turns at 0.3 rad/s.
            let heading = 0.3 * (t - 15.0);
            [-3.0 * heading.sin(), 3.0 * heading.cos(), 0.0]
        }
        t if (30.0..38.0).contains(&t) => [2.5 * (2.0 * PI * 2.0 * (t - 30.0)).sin(), 0.0, 0.0],
        _ => [0.0; 3],
    };
    a.map(|v| v * scale)
}

/// The true state of the simulated IMU.
#[derive(Clone, Copy, Debug)]
struct Truth {
    /// The true attitude, body to world.
    attitude: Quat,
    /// The true gyro bias in rad/s.
    bias: Vec3,
}

/// One gyro sample with the optional accelerometer and heading readings.
struct Sample {
    /// Timestamp in microseconds.
    time_us: u64,
    /// Time since the previous gyro sample in seconds.
    dt: f32,
    /// The gyro reading `ω + b + n` in rad/s.
    gyro: Vec3,
    /// The accelerometer reading in m/s², on every [`ACCEL_EVERY`]-th sample.
    accel: Option<Vec3>,
    /// The heading in radians, on every [`YAW_EVERY`]-th sample.
    yaw: Option<f32>,
    /// The true world-frame acceleration at this time.
    world_accel: Vec3,
    /// The truth at this time.
    truth: Truth,
}

/// The simulated IMU.
struct Simulation {
    /// The noise source.
    noise: Noise,
    /// The current truth.
    truth: Truth,
    /// The timestamp of the last gyro sample in microseconds.
    time_us: u64,
    /// The number of gyro samples so far.
    count: usize,
    /// The scale of the dynamic acceleration.
    dynamics: f32,
}

impl Simulation {
    /// Starts at rest in the initial attitude with the initial bias.
    fn new(seed: u32, dynamics: f32) -> Self {
        Self {
            noise: Noise::new(seed),
            truth: Truth {
                attitude: Quat::from_euler(INITIAL_EULER),
                bias: INITIAL_BIAS,
            },
            time_us: 0,
            count: 0,
            dynamics,
        }
    }

    /// Returns a normal sample with standard deviation `sigma`.
    #[allow(clippy::cast_possible_truncation)]
    fn gaussian(&mut self, sigma: f32) -> f32 {
        sigma * self.noise.gaussian() as f32
    }

    /// Advances to the next (jittered, possibly delayed) gyro timestamp; `None` at the end.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn step(&mut self) -> Option<Sample> {
        let start = self.time_us;
        let skipped = u64::from(self.noise.chance(DROP_PROBABILITY));
        let jitter = (self.noise.uniform() * (2 * GYRO_JITTER_US + 1) as f64) as u64;
        let next = start + (1 + skipped) * GYRO_PERIOD_US + jitter - GYRO_JITTER_US;
        if next > DURATION_US {
            return None;
        }

        let (t, dt) = (start as f32 * 1e-6, (next - start) as f32 * 1e-6);
        let omega = body_rate(t);
        let gyro_sigma = GYRO_NOISE_DENSITY / dt.sqrt();
        let gyro =
            core::array::from_fn(|i| omega[i] + self.truth.bias[i] + self.gaussian(gyro_sigma));
        let step = Quat::exp(omega.map(|w| w * dt));
        self.truth.attitude = self.truth.attitude.mul(&step).normalized();
        let walk = GYRO_BIAS_WALK * dt.sqrt();
        self.truth.bias = core::array::from_fn(|i| self.truth.bias[i] + self.gaussian(walk));
        self.time_us = next;
        self.count += 1;

        Some(Sample {
            time_us: next,
            dt,
            gyro,
            accel: (self.count % ACCEL_EVERY == 0).then(|| self.accel_sample()),
            yaw: (self.count % YAW_EVERY == 0).then(|| self.yaw_sample()),
            world_accel: world_acceleration(next as f32 * 1e-6, self.dynamics),
            truth: self.truth,
        })
    }

    /// Returns the accelerometer reading `Rᵀ·(a + g) + n` at the current time.
    #[allow(clippy::cast_precision_loss)]
    fn accel_sample(&mut self) -> Vec3 {
        let a = world_acceleration(self.time_us as f32 * 1e-6, self.dynamics);
        let body = self
            .truth
            .attitude
            .rotation()
            .transpose_mul([a[0], a[1], a[2] + GRAVITY]);
        body.map(|f| f + self.gaussian(ACCEL_NOISE_STD))
    }

    /// Returns the heading at the current time, wrapped to `(−π, π]`.
    fn yaw_sample(&mut self) -> f32 {
        let yaw = self.truth.attitude.rotation().yaw();
        wrap_pi(yaw + self.gaussian(YAW_NOISE_STD))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gate_keeps_the_attitude_and_the_window_does_not() {
        attitude_run(184, 1.0, 11.34, 10, 0.1, 1);
        let s: [f32; 10] = crate::lab::summary();
        assert!(s[2] < 3.0, "gated max error {}°", s[2]);
        assert!(
            s[3] > 3.0 * s[2],
            "window max error {}° vs gated {}°",
            s[3],
            s[2]
        );
        assert!(s[4] > 0.9, "dynamic samples skipped: {}", s[4]);
        assert!(s[5] < 0.05, "static samples gated: {}", s[5]);
    }

    #[test]
    fn without_dynamics_both_agree() {
        attitude_run(1, 0.0, 11.34, 10, 0.1, 1);
        let s: [f32; 4] = crate::lab::summary();
        assert!(s[2] < 3.0 && s[3] < 3.0, "{s:?}");
    }

    #[test]
    fn quaternion_helpers_round_trip() {
        let q = Quat::from_euler([0.1, -0.2, 0.3]);
        let e = q.euler();
        assert!(
            (e[0] - 0.1).abs() < 1e-5 && (e[1] + 0.2).abs() < 1e-5 && (e[2] - 0.3).abs() < 1e-5
        );
        let v = [0.2, -0.1, 0.05];
        let back = Quat::exp(v).log();
        assert!(v.iter().zip(back).all(|(a, b)| (a - b).abs() < 1e-5));
    }
}
