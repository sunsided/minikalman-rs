//! The navigation workloads: a 6-state attitude ESKF and a 12-state GNSS/INS ESKF (`f32`).
//!
//! Each filter keeps the nominal state (quaternion, bias, and for the INS position and
//! velocity) outside minikalman and estimates only the error `δx`, as in
//! `crates/minikalman/examples/attitude_eskf.rs`. One step splits into separately measured
//! parts: the glue that moves the nominal state and writes `F` and `Q` (`predict_glue`),
//! the covariance prediction (`predict`), each correction including its Jacobian and `h(x̂)`
//! (`correct_*`), and the error injection with `P ← G·P·Gᵀ` (`reset`). `step` and
//! `imu_step` measure a whole cycle in one section.
//!
//! The data come from a simulated truth (constant body rates of about 1 rad/s at 400 Hz with
//! jitter, so `|u|` is about 3 mrad; for the INS a constant velocity) with
//! noise of the configured standard deviations, and the gates are set as a real filter
//! would set them. The model is in [`crate::nav`].

use core::hint::black_box;
use core::mem::size_of;

use minikalman::extended::{
    BorrowedKalman, BorrowedObservation, FilterStorage, ObservationStorage,
};
use minikalman::prelude::*;

use crate::nav::*;
use crate::{__sheap, Noise, STACK_PAINT, measure, report_stack};

/// The label of every section: the element type, plus `-libm` when minikalman is built
/// without `unsafe` and its `f32` square root is `libm::sqrtf` on the M4F too. Keyed on the
/// `unsafe` feature itself, so the label follows the build even when `nav-libm` is combined
/// with features that enable `unsafe`.
macro_rules! label {
    ($s:literal) => {
        if cfg!(feature = "unsafe") {
            concat!("f32/", $s)
        } else {
            concat!("f32-libm/", $s)
        }
    };
}

/// Attitude states: `δθ` (3) and `δb_g` (3).
const ATT_N: usize = 6;

/// Offset of the gyro bias error in the attitude state.
const ATT_BIAS: usize = 3;

/// GNSS/INS states: `δp`, `δv`, `δθ`, `δb_g` (3 each).
const INS_N: usize = 12;

/// GNSS measurements: horizontal position (2) and velocity (3).
const GNSS_M: usize = 5;

/// Nominal IMU period in seconds (400 Hz); every sample adds up to ±0.5 ms of jitter.
const IMU_DT: f32 = 0.0025;

/// The true body rate in rad/s (|ω| ≈ 1.1 rad/s).
const RATE: Vec3 = [0.6, -0.4, 0.8];

/// The true gyro bias in rad/s.
const TRUE_BIAS: Vec3 = [0.01, -0.02, 0.005];

/// The true (constant) velocity of the INS in m/s.
const TRUE_VELOCITY: Vec3 = [1.0, 0.5, 0.0];

/// Gyro white noise density in rad/s/√Hz.
const GYRO_NOISE: f32 = 3e-4;

/// Gyro bias random walk in rad/s/√s.
const GYRO_BIAS_WALK: f32 = 1e-4;

/// Accelerometer white noise density in m/s²/√Hz (velocity random walk of the INS).
const ACCEL_NOISE: f32 = 2e-3;

/// Accelerometer standard deviation per axis in m/s² (tilt update).
const ACCEL_STD: f32 = 0.05;

/// Heading standard deviation in radians.
const YAW_STD: f32 = 0.035;

/// GNSS horizontal position standard deviation in m.
const GNSS_POS_STD: f32 = 0.5;

/// GNSS velocity standard deviation in m/s.
const GNSS_VEL_STD: f32 = 0.1;

/// Barometric altitude standard deviation in m.
const BARO_STD: f32 = 0.3;

/// NIS gates: 99 % (99.9 % for the heading) chi-square quantiles for 3, 1, 5 and 1 dof.
const GATE_ACCEL: f32 = 11.34;
/// See [`GATE_ACCEL`].
const GATE_YAW: f32 = 10.83;
/// See [`GATE_ACCEL`].
const GATE_GNSS: f32 = 15.09;
/// See [`GATE_ACCEL`].
const GATE_BARO: f32 = 6.63;

/// Largest attitude error in radians (about 1°) the self-check after the measured steps
/// accepts; the filters start at the truth, so a wrong model or a diverging filter trips it.
const MAX_ATTITUDE_ERROR: f32 = 0.02;

/// Unmeasured steps before the measured ones.
const WARMUP: usize = crate::WARMUP;

/// Measured iterations per operation.
const ITERATIONS: usize = crate::ITERATIONS;

/// One IMU sample with the aiding measurements at the same instant.
#[derive(Clone, Copy)]
struct Sample {
    /// Time since the previous sample in s.
    dt: f32,
    /// Gyro reading in rad/s (rate + bias + noise).
    gyro: Vec3,
    /// Accelerometer reading (specific force, body frame) in m/s².
    accel: Vec3,
    /// External heading in rad.
    yaw: f32,
    /// GNSS horizontal position (m) and velocity (m/s).
    gnss: [f32; GNSS_M],
    /// Barometric altitude in m.
    baro: f32,
}

/// The simulated truth and its sensors.
struct Simulation {
    /// True attitude.
    attitude: Quat,
    /// True position in m.
    position: Vec3,
    /// Deterministic noise source.
    noise: Noise,
}

impl Simulation {
    /// Starts at the origin, level, heading 0.
    fn new(seed: u32) -> Self {
        Self {
            attitude: Quat::IDENTITY,
            position: [0.0; 3],
            noise: Noise(seed),
        }
    }

    /// A zero-mean sample with standard deviation `std` (uniform, `√12·std` wide).
    fn noise(&mut self, std: f32) -> f32 {
        3.464_101_6 * std * self.noise.next()
    }

    /// Advances the truth by one jittered IMU period and returns the sensor readings.
    fn next(&mut self) -> Sample {
        let dt = IMU_DT + 0.001 * self.noise.next();
        let rate = RATE;
        self.attitude = self
            .attitude
            .mul(&Quat::exp(rate.map(|w| w * dt)))
            .normalized();
        self.position = core::array::from_fn(|i| self.position[i] + TRUE_VELOCITY[i] * dt);

        let r = self.attitude.rotation();
        let gyro_std = GYRO_NOISE / dt.square_root();
        let gravity = transpose_mul(&r, [0.0, 0.0, GRAVITY]);

        Sample {
            dt,
            gyro: core::array::from_fn(|i| rate[i] + TRUE_BIAS[i] + self.noise(gyro_std)),
            accel: core::array::from_fn(|i| gravity[i] + self.noise(ACCEL_STD)),
            yaw: yaw(&r) + self.noise(YAW_STD),
            gnss: [
                self.position[0] + self.noise(GNSS_POS_STD),
                self.position[1] + self.noise(GNSS_POS_STD),
                TRUE_VELOCITY[0] + self.noise(GNSS_VEL_STD),
                TRUE_VELOCITY[1] + self.noise(GNSS_VEL_STD),
                TRUE_VELOCITY[2] + self.noise(GNSS_VEL_STD),
            ],
            baro: self.position[2] + self.noise(BARO_STD),
        }
    }
}

/// The attitude ESKF: `δx = (δθ, δb_g)` in the filter, `q̂` and `b̂` outside.
struct Attitude<'a> {
    /// The error-state filter; `δx` is zero between steps.
    filter: BorrowedKalman<'a, ATT_N, f32>,
    /// The accelerometer tilt update.
    accel: BorrowedObservation<'a, ATT_N, 3, f32>,
    /// The heading update.
    yaw: BorrowedObservation<'a, ATT_N, 1, f32>,
    /// Nominal attitude `q̂`.
    attitude: Quat,
    /// Nominal gyro bias `b̂` in rad/s.
    bias: Vec3,
}

impl Attitude<'_> {
    /// Moves `q̂` with one gyro sample and writes `F` and `Q` for its `dt`.
    fn predict_glue(&mut self, gyro: Vec3, dt: f32) -> bool {
        let u: Vec3 = core::array::from_fn(|i| (gyro[i] - self.bias[i]) * dt);
        self.attitude = self.attitude.mul(&Quat::exp(u)).normalized();

        let rotation = attitude_transition(u);
        self.filter.state_transition_jacobian_mut().apply(|f| {
            f.make_identity();
            write_block(f, ATT_ATTITUDE, ATT_ATTITUDE, &rotation);
            (0..3).for_each(|i| f.set_at(ATT_ATTITUDE + i, ATT_BIAS + i, -dt));
        });

        let (qa, qb) = (
            GYRO_NOISE * GYRO_NOISE * dt,
            GYRO_BIAS_WALK * GYRO_BIAS_WALK * dt,
        );
        self.filter.direct_process_noise_mut().apply(|q| {
            write_diagonal(q, ATT_ATTITUDE, qa);
            write_diagonal(q, ATT_BIAS, qb);
        });
        true
    }

    /// `P ← F·P·Fᵀ + Q`; the error stays zero.
    fn predict(&mut self) -> bool {
        self.filter
            .predict_nonlinear(|_, next| (0..ATT_N).for_each(|i| next.set_row(i, 0.0)));
        true
    }

    /// Tilt update from a body-frame specific force, including the Jacobian and `h(x̂)`.
    fn correct_accel(&mut self, f: Vec3) -> bool {
        let predicted = transpose_mul(&self.attitude.rotation(), [0.0, 0.0, GRAVITY]);
        let jacobian = gravity_jacobian(predicted);
        self.accel
            .observation_jacobian_matrix_mut()
            .apply(|h| write_block(h, 0, ATT_ATTITUDE, &jacobian));
        self.accel
            .measurement_vector_mut()
            .apply(|z| (0..3).for_each(|i| z.set_row(i, f[i])));

        self.filter
            .correct_nonlinear(&mut self.accel, |_, h| {
                (0..3).for_each(|i| h.set_row(i, predicted[i]));
            })
            .is_applied()
    }

    /// Heading update, including the Jacobian, `atan2f` and the innovation wrap.
    fn correct_yaw(&mut self, psi: f32) -> bool {
        let rotation = self.attitude.rotation();
        let jacobian = yaw_jacobian(&rotation);
        self.yaw.observation_jacobian_matrix_mut().apply(|h| {
            (0..3).for_each(|i| h.set_at(0, ATT_ATTITUDE + i, jacobian[i]));
        });
        self.yaw
            .measurement_vector_mut()
            .apply(|z| z.set_row(0, psi));

        self.filter
            .correct_nonlinear_with(
                &mut self.yaw,
                |_, h| h.set_row(0, yaw(&rotation)),
                |y| y[0] = minikalman::angles::wrap_pi(y[0]),
            )
            .is_applied()
    }

    /// Moves `δx` into `q̂` and `b̂`, zeroes it and applies `P ← G·P·Gᵀ`.
    fn reset(&mut self) -> bool {
        let x = self.filter.state_vector();
        let (dtheta, dbias) = (read3(x, ATT_ATTITUDE), read3(x, ATT_BIAS));
        self.attitude = inject_attitude(&self.attitude, dtheta);
        self.bias = core::array::from_fn(|i| self.bias[i] + dbias[i]);

        let g = reset_jacobian(dtheta);
        self.filter
            .estimate_covariance_mut()
            .apply(|p| reset_covariance(p, ATT_ATTITUDE, &g));
        self.filter
            .state_vector_mut()
            .apply(|v| (0..ATT_N).for_each(|i| v.set_row(i, 0.0)));
        true
    }

    /// One 400 Hz cycle: gyro prediction, accelerometer update and reset.
    fn step(&mut self, s: &Sample) -> bool {
        self.predict_glue(s.gyro, s.dt);
        self.predict();
        let applied = self.correct_accel(s.accel);
        self.reset();
        applied
    }
}

/// The GNSS/INS ESKF: `δx = (δp, δv, δθ, δb_g)` in the filter, the nominal state outside.
struct Ins<'a> {
    /// The error-state filter; `δx` is zero between steps.
    filter: BorrowedKalman<'a, INS_N, f32>,
    /// The GNSS position and velocity update.
    gnss: BorrowedObservation<'a, INS_N, GNSS_M, f32>,
    /// The barometric altitude update.
    baro: BorrowedObservation<'a, INS_N, 1, f32>,
    /// The heading update.
    heading: BorrowedObservation<'a, INS_N, 1, f32>,
    /// Nominal position in m.
    position: Vec3,
    /// Nominal velocity in m/s.
    velocity: Vec3,
    /// Nominal attitude `q̂`.
    attitude: Quat,
    /// Nominal gyro bias `b̂` in rad/s.
    bias: Vec3,
}

impl Ins<'_> {
    /// Integrates one IMU sample into the nominal state and writes `F` and `Q` for its `dt`.
    fn predict_glue(&mut self, gyro: Vec3, accel: Vec3, dt: f32) -> bool {
        let r = self.attitude.rotation();
        let specific = mat_mul(&r, accel);
        let a: Vec3 = [specific[0], specific[1], specific[2] - GRAVITY];
        self.position = core::array::from_fn(|i| {
            self.position[i] + self.velocity[i] * dt + 0.5 * a[i] * dt * dt
        });
        self.velocity = core::array::from_fn(|i| self.velocity[i] + a[i] * dt);
        let u: Vec3 = core::array::from_fn(|i| (gyro[i] - self.bias[i]) * dt);
        self.attitude = self.attitude.mul(&Quat::exp(u)).normalized();

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

        let qv = ACCEL_NOISE * ACCEL_NOISE * dt;
        let (qa, qb) = (
            GYRO_NOISE * GYRO_NOISE * dt,
            GYRO_BIAS_WALK * GYRO_BIAS_WALK * dt,
        );
        self.filter.direct_process_noise_mut().apply(|q| {
            write_diagonal(q, INS_VELOCITY, qv);
            write_diagonal(q, INS_ATTITUDE, qa);
            write_diagonal(q, INS_BIAS, qb);
        });
        true
    }

    /// `P ← F·P·Fᵀ + Q`; the error stays zero.
    fn predict(&mut self) -> bool {
        self.filter
            .predict_nonlinear(|_, next| (0..INS_N).for_each(|i| next.set_row(i, 0.0)));
        true
    }

    /// GNSS update with horizontal position and velocity (`H` is constant, set once).
    fn correct_gnss(&mut self, z: &[f32; GNSS_M]) -> bool {
        let predicted = [
            self.position[0],
            self.position[1],
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
        ];
        self.gnss
            .measurement_vector_mut()
            .apply(|m| (0..GNSS_M).for_each(|i| m.set_row(i, z[i])));

        self.filter
            .correct_nonlinear(&mut self.gnss, |_, h| {
                (0..GNSS_M).for_each(|i| h.set_row(i, predicted[i]));
            })
            .is_applied()
    }

    /// Barometric altitude update (`H` is constant, set once).
    fn correct_baro(&mut self, altitude: f32) -> bool {
        let predicted = self.position[2];
        self.baro
            .measurement_vector_mut()
            .apply(|m| m.set_row(0, altitude));

        self.filter
            .correct_nonlinear(&mut self.baro, |_, h| h.set_row(0, predicted))
            .is_applied()
    }

    /// Heading update, including the Jacobian, `atan2f` and the innovation wrap.
    fn correct_heading(&mut self, psi: f32) -> bool {
        let rotation = self.attitude.rotation();
        let jacobian = yaw_jacobian(&rotation);
        self.heading.observation_jacobian_matrix_mut().apply(|h| {
            (0..3).for_each(|i| h.set_at(0, INS_ATTITUDE + i, jacobian[i]));
        });
        self.heading
            .measurement_vector_mut()
            .apply(|z| z.set_row(0, psi));

        self.filter
            .correct_nonlinear_with(
                &mut self.heading,
                |_, h| h.set_row(0, yaw(&rotation)),
                |y| y[0] = minikalman::angles::wrap_pi(y[0]),
            )
            .is_applied()
    }

    /// Moves `δx` into the nominal state, zeroes it and applies `P ← G·P·Gᵀ`.
    fn reset(&mut self) -> bool {
        let x = self.filter.state_vector();
        let (dp, dv) = (read3(x, INS_POSITION), read3(x, INS_VELOCITY));
        let (dtheta, dbias) = (read3(x, INS_ATTITUDE), read3(x, INS_BIAS));
        self.position = core::array::from_fn(|i| self.position[i] + dp[i]);
        self.velocity = core::array::from_fn(|i| self.velocity[i] + dv[i]);
        self.attitude = inject_attitude(&self.attitude, dtheta);
        self.bias = core::array::from_fn(|i| self.bias[i] + dbias[i]);

        let g = reset_jacobian(dtheta);
        self.filter
            .estimate_covariance_mut()
            .apply(|p| reset_covariance(p, INS_ATTITUDE, &g));
        self.filter
            .state_vector_mut()
            .apply(|v| (0..INS_N).for_each(|i| v.set_row(i, 0.0)));
        true
    }

    /// One IMU cycle: nominal integration and covariance prediction.
    fn imu_step(&mut self, s: &Sample) -> bool {
        self.predict_glue(s.gyro, s.accel, s.dt);
        self.predict()
    }
}

/// Builds the attitude filter at the truth's start with its prior, noise and gates.
fn build_attitude<'a>(
    fs: &'a mut FilterStorage<ATT_N, f32>,
    accel: &'a mut ObservationStorage<ATT_N, 3, f32>,
    yaw: &'a mut ObservationStorage<ATT_N, 1, f32>,
) -> Attitude<'a> {
    let (mut filter, mut accel, mut yaw) = (fs.build(), accel.build(), yaw.build());
    filter.estimate_covariance_mut().apply(|p| {
        write_diagonal(p, ATT_ATTITUDE, 1.2e-3);
        write_diagonal(p, ATT_BIAS, 2.5e-3);
    });
    accel
        .measurement_noise_covariance_mut()
        .apply(|r| (0..3).for_each(|i| r.set_at(i, i, ACCEL_STD * ACCEL_STD)));
    accel.set_gate(Some(GATE_ACCEL));
    yaw.measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, YAW_STD * YAW_STD));
    yaw.set_gate(Some(GATE_YAW));

    Attitude {
        filter,
        accel,
        yaw,
        attitude: Quat::IDENTITY,
        bias: [0.0; 3],
    }
}

/// Builds the GNSS/INS filter at the truth's start with its prior, noise, constant `H` and gates.
fn build_ins<'a>(
    fs: &'a mut FilterStorage<INS_N, f32>,
    gnss: &'a mut ObservationStorage<INS_N, GNSS_M, f32>,
    baro: &'a mut ObservationStorage<INS_N, 1, f32>,
    heading: &'a mut ObservationStorage<INS_N, 1, f32>,
) -> Ins<'a> {
    let (mut filter, mut gnss, mut baro, mut heading) =
        (fs.build(), gnss.build(), baro.build(), heading.build());
    filter.estimate_covariance_mut().apply(|p| {
        write_diagonal(p, INS_POSITION, 1.0);
        write_diagonal(p, INS_VELOCITY, 0.01);
        write_diagonal(p, INS_ATTITUDE, 1.2e-3);
        write_diagonal(p, INS_BIAS, 2.5e-3);
    });
    gnss.observation_jacobian_matrix_mut().apply(|h| {
        h.set_at(0, INS_POSITION, 1.0);
        h.set_at(1, INS_POSITION + 1, 1.0);
        (0..3).for_each(|i| h.set_at(2 + i, INS_VELOCITY + i, 1.0));
    });
    gnss.measurement_noise_covariance_mut().apply(|r| {
        (0..2).for_each(|i| r.set_at(i, i, GNSS_POS_STD * GNSS_POS_STD));
        (2..GNSS_M).for_each(|i| r.set_at(i, i, GNSS_VEL_STD * GNSS_VEL_STD));
    });
    gnss.set_gate(Some(GATE_GNSS));
    baro.observation_jacobian_matrix_mut()
        .apply(|h| h.set_at(0, INS_POSITION + 2, 1.0));
    baro.measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, BARO_STD * BARO_STD));
    baro.set_gate(Some(GATE_BARO));
    heading
        .measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, YAW_STD * YAW_STD));
    heading.set_gate(Some(GATE_YAW));

    Ins {
        filter,
        gnss,
        baro,
        heading,
        position: [0.0; 3],
        velocity: TRUE_VELOCITY,
        attitude: Quat::IDENTITY,
        bias: [0.0; 3],
    }
}

/// Measures the attitude ESKF per operation, then its stack use and storage sizes.
#[inline(never)]
pub fn attitude() {
    let mut fs = FilterStorage::<ATT_N, f32>::new(0.0);
    let mut accel = ObservationStorage::<ATT_N, 3, f32>::new(0.0);
    let mut yaw = ObservationStorage::<ATT_N, 1, f32>::new(0.0);
    let mut f = build_attitude(&mut fs, &mut accel, &mut yaw);
    let mut sim = Simulation::new(0x2545_F491);

    for _ in 0..WARMUP {
        let s = sim.next();
        assert!(f.step(&s));
        assert!(f.correct_yaw(s.yaw) && f.reset());
    }

    let f = black_box(&mut f);
    repaint_stack();
    let top = stack_pointer();
    attitude_loop(f, &mut sim);
    let deepest = deepest_stack();
    let (used, total) = (top - deepest, stack_start() - deepest);

    assert!(attitude_error(&f.attitude, &sim.attitude) < MAX_ATTITUDE_ERROR);

    note(label!("att_6x3/stack_per_op"), used);
    note(label!("att_6x3/stack_total"), total);
    note(
        label!("att_6x3/FilterStorage<6>"),
        size_of::<FilterStorage<ATT_N, f32>>(),
    );
    note(
        label!("att_6x3/ObservationStorage<6,3>"),
        size_of::<ObservationStorage<ATT_N, 3, f32>>(),
    );
    note(
        label!("att_6x3/ObservationStorage<6,1>"),
        size_of::<ObservationStorage<ATT_N, 1, f32>>(),
    );
    note(
        label!("att_6x3/nominal_state"),
        size_of::<Quat>() + size_of::<Vec3>(),
    );
}

/// Measures the GNSS/INS ESKF per operation, then its stack use and storage sizes.
#[inline(never)]
pub fn gnss_ins() {
    let mut fs = FilterStorage::<INS_N, f32>::new(0.0);
    let mut gnss = ObservationStorage::<INS_N, GNSS_M, f32>::new(0.0);
    let mut baro = ObservationStorage::<INS_N, 1, f32>::new(0.0);
    let mut heading = ObservationStorage::<INS_N, 1, f32>::new(0.0);
    let mut f = build_ins(&mut fs, &mut gnss, &mut baro, &mut heading);
    let mut sim = Simulation::new(0x9E37_79B9);

    for _ in 0..WARMUP {
        let s = sim.next();
        assert!(f.imu_step(&s));
        assert!(f.correct_gnss(&s.gnss) && f.reset());
        assert!(f.correct_baro(s.baro) && f.reset());
        assert!(f.correct_heading(s.yaw) && f.reset());
    }

    let f = black_box(&mut f);
    repaint_stack();
    let top = stack_pointer();
    gnss_ins_loop(f, &mut sim);
    let deepest = deepest_stack();
    let (used, total) = (top - deepest, stack_start() - deepest);

    assert!(attitude_error(&f.attitude, &sim.attitude) < MAX_ATTITUDE_ERROR);
    assert!((0..3).all(|i| (f.position[i] - sim.position[i]).abs() < 1.0));
    assert!((0..3).all(|i| (f.velocity[i] - TRUE_VELOCITY[i]).abs() < 0.3));

    note(label!("ins_12x5/stack_per_op"), used);
    note(label!("ins_12x5/stack_total"), total);
    note(
        label!("ins_12x5/FilterStorage<12>"),
        size_of::<FilterStorage<INS_N, f32>>(),
    );
    note(
        label!("ins_12x5/ObservationStorage<12,5>"),
        size_of::<ObservationStorage<INS_N, GNSS_M, f32>>(),
    );
    note(
        label!("ins_12x5/ObservationStorage<12,1>"),
        size_of::<ObservationStorage<INS_N, 1, f32>>(),
    );
    note(
        label!("ins_12x5/nominal_state"),
        size_of::<Quat>() + 3 * size_of::<Vec3>(),
    );
}

/// The measured iterations of the attitude filter, in a frame of their own, so that the
/// stack they and the operations use is all below the caller's stack pointer.
#[inline(never)]
fn attitude_loop(f: &mut Attitude<'_>, sim: &mut Simulation) {
    for _ in 0..ITERATIONS {
        let s = sim.next();
        measure(label!("att_6x3/predict_glue"), || {
            f.predict_glue(s.gyro, s.dt)
        });
        measure(label!("att_6x3/predict"), || f.predict());
        measure(label!("att_6x3/correct_accel"), || f.correct_accel(s.accel));
        measure(label!("att_6x3/reset"), || f.reset());
        measure(label!("att_6x1/correct_yaw"), || f.correct_yaw(s.yaw));
        measure(label!("att_6x3/reset"), || f.reset());
        let s = sim.next();
        measure(label!("att_6x3/step"), || f.step(&s));
    }
}

/// The measured iterations of the GNSS/INS filter, see [`attitude_loop`].
#[inline(never)]
fn gnss_ins_loop(f: &mut Ins<'_>, sim: &mut Simulation) {
    for _ in 0..ITERATIONS {
        let s = sim.next();
        measure(label!("ins_12x5/predict_glue"), || {
            f.predict_glue(s.gyro, s.accel, s.dt)
        });
        measure(label!("ins_12x5/predict"), || f.predict());
        let s = sim.next();
        measure(label!("ins_12x5/imu_step"), || f.imu_step(&s));
        measure(label!("ins_12x5/correct_gnss"), || f.correct_gnss(&s.gnss));
        measure(label!("ins_12x5/reset"), || f.reset());
        measure(label!("ins_12x1/correct_baro"), || f.correct_baro(s.baro));
        measure(label!("ins_12x5/reset"), || f.reset());
        measure(label!("ins_12x1/correct_heading"), || {
            f.correct_heading(s.yaw)
        });
        measure(label!("ins_12x5/reset"), || f.reset());
    }
}

/// Returns the angle in radians between the estimate and the truth.
fn attitude_error(estimate: &Quat, truth: &Quat) -> f32 {
    2.0 * norm(estimate.conjugate().mul(truth).v)
}

/// Logs a named value (bytes); hooked by `renode/bench.py`, which writes `NOTE <name> <value>`.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn mk_note(name: *const u8, len: usize, value: usize) {
    black_box((name, len, value));
}

/// Logs `value` as `name`.
fn note(name: &'static str, value: usize) {
    mk_note(name.as_ptr(), name.len(), value);
}

/// Returns the current stack pointer.
#[inline(always)]
fn stack_pointer() -> usize {
    let sp: usize;
    // SAFETY: reads a register, touches no memory.
    unsafe { core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack)) };
    sp
}

/// Repaints the free stack up to just below this function's frame, so [`deepest_stack`]
/// sees only what runs after the return.
///
/// Repainting hides the deeper use before it from the final free-stack check, so it first
/// reports the free stack so far (`STACK_FREE`; `run.sh` takes the smallest).
#[inline(never)]
fn repaint_stack() {
    report_stack();
    let top = stack_pointer() - 16;
    let mut p = core::ptr::addr_of_mut!(__sheap);
    while (p as usize) < top {
        // SAFETY: the words between the statics and the live stack are unused here.
        unsafe {
            p.write_volatile(STACK_PAINT);
            p = p.add(1);
        }
    }
}

/// Returns the lowest stack address written since the last paint.
fn deepest_stack() -> usize {
    let mut p = core::ptr::addr_of_mut!(__sheap);
    // SAFETY: reads within RAM between the statics and the stack.
    while unsafe { p.read_volatile() } == STACK_PAINT {
        p = unsafe { p.add(1) };
    }
    p as usize
}

unsafe extern "C" {
    /// Initial stack pointer, the top of RAM (cortex-m-rt).
    static _stack_start: u32;
}

/// Returns the address the stack grows down from.
fn stack_start() -> usize {
    core::ptr::addr_of!(_stack_start) as usize
}
