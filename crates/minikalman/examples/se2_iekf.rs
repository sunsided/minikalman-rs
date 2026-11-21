//! Invariant EKF example: a planar robot pose on the Lie group SE(2), estimated from wheel
//! odometry and position fixes with [`ExtendedKalman`](minikalman::extended::ExtendedKalman).
//!
//! The filter state `x` is not the pose. It is a small error `ξ = (ρ, θ)` in the Lie algebra
//! (the tangent space), and the pose itself, the *nominal state* `X̂ = (R̂, t̂)`, is kept by the
//! caller. The true pose is `X = X̂·Exp(ξ)`, i.e. the error is expressed in the robot's body
//! frame (Solà et al., "A micro Lie theory for state estimation in robotics",
//! arXiv:1812.01537, Appendix C for the SE(2) formulas used here).
//!
//! ## One step
//!
//! 1. **Predict.** The odometry increment `u = (v·Δt, 0, ω·Δt)` moves the nominal state on the
//!    group, `X̂ ← X̂·Exp(u)`. The error then evolves as `ξ ← Ad_{Exp(u)}⁻¹·ξ` (Solà (161)),
//!    which does not depend on the estimate at all. That matrix is written as the state
//!    transition Jacobian, and `predict_nonlinear` keeps `ξ = 0` while it propagates `P`.
//! 2. **Correct.** A position fix `z = t + v` is a *left-invariant observation* `Y = X·d` with
//!    `d = (0, 0, 1)` (Barrau and Bonnabel, "The Invariant Extended Kalman Filter as a Stable
//!    Observer", IEEE TAC 2017; equation numbers from arXiv:1410.1465v4, eq. (15)). Its invariant innovation
//!    `R̂ᵀ·(z − t̂)` (eq. (17)) is `ρ` plus noise to first order, so `H = [I₂ 0]` is constant.
//!    `correct_nonlinear_with` computes `z − t̂`, and its `fix` closure rotates it into the body
//!    frame.
//! 3. **Inject.** `X̂ ← X̂·Exp(ξ)`, then `ξ ← 0`. The covariance is carried over to the new
//!    error with the right Jacobian, `P ← J_r(ξ)·P·J_r(ξ)ᵀ` (Solà (163)).
//!
//! ## Comparison
//!
//! A conventional EKF on `(x, y, θ)` runs on the same odometry and fixes with the same prior,
//! noise and exact motion model. Both start with the heading 170° off. The invariant filter
//! must end within [`POSITION_TOLERANCE`] and [`HEADING_TOLERANCE`] of the truth or the
//! example exits with a non-zero status; the conventional filter's result is only printed.
//!
//! The SE(2) helpers below stay in the example on purpose: the crate has no Lie-group types.

#![forbid(unsafe_code)]

use std::f32::consts::PI;
use std::process::ExitCode;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use minikalman::angles::wrap_pi;
use minikalman::extended::builder::{
    KalmanFilterBuilder, KalmanFilterObservationType, KalmanFilterType,
};
use minikalman::prelude::*;

/// Number of filter states: the error `(ρx, ρy, θ)` or the pose `(x, y, θ)`.
const NUM_STATES: usize = 3;

/// Number of observations: a 2-D position fix.
const NUM_OBSERVATIONS: usize = 2;

/// The extended filter over [`NUM_STATES`] states with owned buffers.
type Filter = KalmanFilterType<NUM_STATES, f32>;

/// The position fix with owned buffers.
type Fix = KalmanFilterObservationType<NUM_STATES, NUM_OBSERVATIONS, f32>;

/// A 3×3 matrix as rows.
type Mat3 = [[f32; 3]; 3];

/// Odometry time step in seconds.
const DELTA_T: f32 = 0.1;

/// Number of simulated odometry steps (60 seconds).
const NUM_STEPS: usize = 600;

/// A position fix arrives every this many odometry steps (1 Hz).
const FIX_EVERY: usize = 10;

/// Print both estimates every this many steps.
const PRINT_EVERY: usize = 50;

/// Forward speed of the robot in m/s.
const SPEED: f32 = 1.0;

/// Initial heading error of both filters in radians (170°).
const INITIAL_HEADING_ERROR: f32 = 170.0 * PI / 180.0;

/// Prior standard deviation of the position in meters.
const PRIOR_POSITION_STD: f32 = 1.0;

/// Prior standard deviation of the heading in radians, wide enough to cover the 170° error.
const PRIOR_HEADING_STD: f32 = PI;

/// Standard deviation of the odometry speed noise in m/s.
const SPEED_NOISE_STD: f32 = 0.05;

/// Standard deviation of the odometry turn rate noise in rad/s.
const TURN_RATE_NOISE_STD: f32 = 0.02;

/// Modelled lateral slip in m/s; the true robot never slips, this keeps `Q` positive definite.
const LATERAL_NOISE_STD: f32 = 0.01;

/// Standard deviation of each position fix axis in meters.
const FIX_NOISE_STD: f32 = 0.5;

/// Largest final position error of the invariant filter that counts as converged, in meters.
const POSITION_TOLERANCE: f32 = 1.0;

/// Largest final heading error of the invariant filter that counts as converged, in radians (5°).
const HEADING_TOLERANCE: f32 = 5.0 * PI / 180.0;

/// Fixed seed so CI runs are reproducible.
const SEED: u64 = 7;

/// Runs the self-checks, the simulation, and checks that the invariant filter converged.
fn main() -> ExitCode {
    check_lie_helpers();

    let truth_start = Pose::default();
    let guess = Pose {
        theta: wrap_pi(truth_start.theta + INITIAL_HEADING_ERROR),
        ..truth_start
    };

    let mut truth = truth_start;
    let mut sensors = Sensors::new();
    let mut invariant = InvariantFilter::new(guess);
    let mut naive = NaiveFilter::new(guess);
    let mut settled = [Settling::default(); 2];

    for step in 1..=NUM_STEPS {
        let true_twist = twist(step);
        truth = truth.compose(&Pose::exp(increment(true_twist)));

        let measured = increment(sensors.odometry(true_twist));
        invariant.predict(measured);
        naive.predict(measured);

        if step % FIX_EVERY == 0 {
            let fix = sensors.position(&truth);
            invariant.correct(fix);
            naive.correct(fix);
        }

        settled[0].update(step, &truth, &invariant.pose());
        settled[1].update(step, &truth, &naive.pose());

        if step == 1 || step % PRINT_EVERY == 0 {
            print_errors(step, &truth, &invariant.pose(), &naive.pose());
        }
    }

    println!(
        "Heading error last above {:.0}°: invariant at t={:.1} s, naive at t={:.1} s",
        HEADING_TOLERANCE.to_degrees(),
        settled[0].seconds(),
        settled[1].seconds(),
    );

    check_convergence(&truth, &invariant.pose(), &naive.pose())
}

/// Tracks when a filter's heading error was last above [`HEADING_TOLERANCE`].
#[derive(Clone, Copy, Default)]
struct Settling {
    /// The last step with a heading error above the tolerance, zero if none.
    last_unsettled: usize,
}

impl Settling {
    /// Records the step if the estimate's heading error is above the tolerance.
    fn update(&mut self, step: usize, truth: &Pose, estimate: &Pose) {
        if errors(truth, estimate).1 > HEADING_TOLERANCE {
            self.last_unsettled = step;
        }
    }

    /// Returns the time of the last unsettled step in seconds.
    fn seconds(self) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let t = self.last_unsettled as f32 * DELTA_T;
        t
    }
}

/// A planar pose `(R(θ), t)` in SE(2), with the heading in `(−π, π]` radians.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Pose {
    /// World-frame x position in meters.
    x: f32,
    /// World-frame y position in meters.
    y: f32,
    /// Heading in radians, counter-clockwise from the world x axis.
    theta: f32,
}

impl Pose {
    /// Returns the group product `self·other`: `(R_a·R_b, t_a + R_a·t_b)` (Solà (155)).
    fn compose(&self, other: &Pose) -> Pose {
        let (x, y) = rotate(self.theta, other.x, other.y);

        Pose {
            x: self.x + x,
            y: self.y + y,
            theta: wrap_pi(self.theta + other.theta),
        }
    }

    /// Returns the inverse `(Rᵀ, −Rᵀ·t)` (Solà (154)).
    fn inverse(&self) -> Pose {
        let (x, y) = rotate(-self.theta, self.x, self.y);

        Pose {
            x: -x,
            y: -y,
            theta: -self.theta,
        }
    }

    /// Returns `Exp(τ)` for the tangent `τ = (ρx, ρy, θ)`: `(R(θ), V(θ)·ρ)` (Solà (156)).
    fn exp(tau: [f32; 3]) -> Pose {
        let [a, b] = v_coefficients(tau[2]);

        Pose {
            x: a * tau[0] - b * tau[1],
            y: b * tau[0] + a * tau[1],
            theta: wrap_pi(tau[2]),
        }
    }

    /// Returns `Log(self) = (V(θ)⁻¹·t, θ)` (Solà (157)).
    fn log(&self) -> [f32; 3] {
        let [a, b] = v_coefficients(self.theta);
        let det = a * a + b * b;

        [
            (a * self.x + b * self.y) / det,
            (-b * self.x + a * self.y) / det,
            self.theta,
        ]
    }

    /// Returns `Ad_self⁻¹ = [[Rᵀ, Rᵀ·[1]×·t], [0, 1]]` (Solà (161)).
    ///
    /// For `X̂ ← X̂·self` it maps the body-frame error before the step onto the error after it.
    fn inverse_adjoint(&self) -> Mat3 {
        let (c, s) = (self.theta.cos(), self.theta.sin());
        let (tx, ty) = rotate(-self.theta, -self.y, self.x);

        [[c, s, tx], [-s, c, ty], [0.0, 0.0, 1.0]]
    }
}

/// Returns `(sin θ/θ, (1 − cos θ)/θ)`, so that `V(θ) = [[a, −b], [b, a]]` (Solà (158)).
///
/// Below 1e-3 rad it uses the Taylor series, since both quotients lose all precision in `f32`.
fn v_coefficients(theta: f32) -> [f32; 2] {
    if theta.abs() < 1e-3 {
        [1.0 - theta * theta / 6.0, theta / 2.0]
    } else {
        [theta.sin() / theta, (1.0 - theta.cos()) / theta]
    }
}

/// Returns the right Jacobian `J_r(τ)` of SE(2), so that `Exp(τ + δ) ≈ Exp(τ)·Exp(J_r·δ)`.
///
/// Solà (163), with the `θ → 0` limit `[[1, 0, −ρy/2], [0, 1, ρx/2], [0, 0, 1]]` below 1e-3 rad.
fn right_jacobian(tau: [f32; 3]) -> Mat3 {
    let [r1, r2, theta] = tau;

    if theta.abs() < 1e-3 {
        return [[1.0, 0.0, -r2 / 2.0], [0.0, 1.0, r1 / 2.0], [0.0, 0.0, 1.0]];
    }

    let (s, c) = theta.sin_cos();
    let t2 = theta * theta;

    [
        [
            s / theta,
            (1.0 - c) / theta,
            (theta * r1 - r2 + r2 * c - r1 * s) / t2,
        ],
        [
            (c - 1.0) / theta,
            s / theta,
            (r1 + theta * r2 - r1 * c - r2 * s) / t2,
        ],
        [0.0, 0.0, 1.0],
    ]
}

/// Rotates the vector `(x, y)` by `theta` radians.
fn rotate(theta: f32, x: f32, y: f32) -> (f32, f32) {
    let (s, c) = theta.sin_cos();
    (c * x - s * y, s * x + c * y)
}

/// Returns `J·P·Jᵀ`.
fn sandwich(j: &Mat3, p: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];

    for (r, row) in out.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = (0..3)
                .flat_map(|i| (0..3).map(move |k| (i, k)))
                .map(|(i, k)| j[r][i] * p[i][k] * j[c][k])
                .sum();
        }
    }

    out
}

/// The invariant (error-state) EKF: `x` holds the body-frame error `ξ`, the pose is outside.
struct InvariantFilter {
    /// The error-state filter. `x` is zero between steps.
    filter: Filter,
    /// The position fix, with the constant invariant Jacobian `H = [I₂ 0]`.
    fix: Fix,
    /// The nominal pose `X̂`.
    nominal: Pose,
}

impl InvariantFilter {
    /// Creates the filter at the initial guess with the shared prior and noise settings.
    fn new(guess: Pose) -> Self {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let mut filter = builder.build();
        let mut fix = builder.observations().build::<NUM_OBSERVATIONS>();

        filter.estimate_covariance_mut().apply(|p| {
            p.set_at(0, 0, PRIOR_POSITION_STD * PRIOR_POSITION_STD);
            p.set_at(1, 1, PRIOR_POSITION_STD * PRIOR_POSITION_STD);
            p.set_at(2, 2, PRIOR_HEADING_STD * PRIOR_HEADING_STD);
        });

        // Body-frame odometry noise, the same on every step.
        let [qv, qlat, qw] = body_process_noise();
        filter.direct_process_noise_mut().apply(|q| {
            q.set_at(0, 0, qv);
            q.set_at(1, 1, qlat);
            q.set_at(2, 2, qw);
        });

        set_position_fix(&mut fix);

        Self {
            filter,
            fix,
            nominal: guess,
        }
    }

    /// Moves the nominal pose by the odometry increment and propagates `P`.
    fn predict(&mut self, increment: [f32; 3]) {
        let step = Pose::exp(increment);
        self.nominal = self.nominal.compose(&step);

        let a = step.inverse_adjoint();
        self.filter
            .state_transition_jacobian_mut()
            .apply(|mat| write_mat3(mat, &a));

        // The error stays zero; only P = A·P·Aᵀ + Q matters.
        self.filter.predict_nonlinear(|_, next| {
            (0..NUM_STATES).for_each(|i| next.set_row(i, 0.0));
        });
    }

    /// Corrects with a world-frame position fix in meters, then injects the error.
    fn correct(&mut self, z: [f32; 2]) {
        self.fix.measurement_vector_mut().apply(|vec| {
            vec.set_row(0, z[0]);
            vec.set_row(1, z[1]);
        });

        // Disjoint field borrows: the closures read `self.nominal` while `self.filter` and
        // `self.fix` are borrowed mutably.
        self.filter.correct_nonlinear_with(
            &mut self.fix,
            |_, h| {
                h.set_row(0, self.nominal.x);
                h.set_row(1, self.nominal.y);
            },
            |y| {
                let (bx, by) = rotate(-self.nominal.theta, y[0], y[1]);
                y[0] = bx;
                y[1] = by;
            },
        );

        self.inject();
    }

    /// Moves the error into the nominal pose, `X̂ ← X̂·Exp(ξ)`, and resets `ξ` to zero.
    fn inject(&mut self) {
        let x = self.filter.state_vector();
        let xi = [x.get_row(0), x.get_row(1), x.get_row(2)];
        self.nominal = self.nominal.compose(&Pose::exp(xi));

        let j = right_jacobian(xi);
        self.filter.estimate_covariance_mut().apply(|p| {
            let reset = sandwich(&j, &read_mat3(p));
            write_mat3(p, &reset);
        });

        self.filter.state_vector_mut().apply(|vec| {
            (0..NUM_STATES).for_each(|i| vec.set_row(i, 0.0));
        });
    }

    /// Returns the current pose estimate.
    fn pose(&self) -> Pose {
        self.nominal
    }
}

/// The conventional EKF on the pose vector `(x, y, θ)`.
struct NaiveFilter {
    /// The pose filter.
    filter: Filter,
    /// The position fix, with `H = [I₂ 0]`.
    fix: Fix,
}

impl NaiveFilter {
    /// Creates the filter at the initial guess with the shared prior and noise settings.
    fn new(guess: Pose) -> Self {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let mut filter = builder.build();
        let mut fix = builder.observations().build::<NUM_OBSERVATIONS>();

        filter.state_vector_mut().apply(|vec| {
            vec.set_row(0, guess.x);
            vec.set_row(1, guess.y);
            vec.set_row(2, guess.theta);
        });

        // Isotropic position prior, so it is the same in the world and body frames.
        filter.estimate_covariance_mut().apply(|p| {
            p.set_at(0, 0, PRIOR_POSITION_STD * PRIOR_POSITION_STD);
            p.set_at(1, 1, PRIOR_POSITION_STD * PRIOR_POSITION_STD);
            p.set_at(2, 2, PRIOR_HEADING_STD * PRIOR_HEADING_STD);
        });

        set_position_fix(&mut fix);

        Self { filter, fix }
    }

    /// Moves the pose by the odometry increment with the exact SE(2) motion and propagates `P`.
    fn predict(&mut self, increment: [f32; 3]) {
        let pose = self.pose();
        let step = Pose::exp(increment);
        let (dx, dy) = rotate(pose.theta, step.x, step.y);

        self.filter.state_transition_jacobian_mut().apply(|a| {
            a.make_identity();
            a.set_at(0, 2, -dy);
            a.set_at(1, 2, dx);
        });

        // The body-frame odometry noise, rotated into the world frame at the current heading.
        let [qv, qlat, qw] = body_process_noise();
        let (c, s) = (pose.theta.cos(), pose.theta.sin());
        self.filter.direct_process_noise_mut().apply(|q| {
            q.set_at(0, 0, c * c * qv + s * s * qlat);
            q.set_at(0, 1, c * s * (qv - qlat));
            q.set_at(1, 0, c * s * (qv - qlat));
            q.set_at(1, 1, s * s * qv + c * c * qlat);
            q.set_at(2, 2, qw);
        });

        self.filter.predict_nonlinear(|state, next| {
            let pose = Pose {
                x: state.get_row(0),
                y: state.get_row(1),
                theta: state.get_row(2),
            };
            let moved = pose.compose(&step);
            next.set_row(0, moved.x);
            next.set_row(1, moved.y);
            next.set_row(2, moved.theta);
        });
    }

    /// Corrects with a world-frame position fix in meters.
    fn correct(&mut self, z: [f32; 2]) {
        self.fix.measurement_vector_mut().apply(|vec| {
            vec.set_row(0, z[0]);
            vec.set_row(1, z[1]);
        });

        self.filter.correct_nonlinear(&mut self.fix, |state, h| {
            h.set_row(0, state.get_row(0));
            h.set_row(1, state.get_row(1));
        });

        self.filter
            .state_vector_mut()
            .apply(|vec| vec.set_row(2, wrap_pi(vec.get_row(2))));
    }

    /// Returns the current pose estimate.
    fn pose(&self) -> Pose {
        let x = self.filter.state_vector();
        Pose {
            x: x.get_row(0),
            y: x.get_row(1),
            theta: x.get_row(2),
        }
    }
}

/// Sets `H = [I₂ 0]` and the isotropic fix noise, which is the same in any frame.
fn set_position_fix(fix: &mut Fix) {
    fix.observation_jacobian_matrix_mut().apply(|h| {
        h.set_at(0, 0, 1.0);
        h.set_at(1, 1, 1.0);
    });

    fix.measurement_noise_covariance_mut()
        .make_scalar(FIX_NOISE_STD * FIX_NOISE_STD);
}

/// Returns the per-step odometry noise variances `(forward, lateral, heading)` in the body frame.
fn body_process_noise() -> [f32; 3] {
    [
        (SPEED_NOISE_STD * DELTA_T).powi(2),
        (LATERAL_NOISE_STD * DELTA_T).powi(2),
        (TURN_RATE_NOISE_STD * DELTA_T).powi(2),
    ]
}

/// Copies a 3×3 matrix out of a filter buffer.
fn read_mat3(mat: &impl Matrix<3, 3, f32>) -> Mat3 {
    core::array::from_fn(|r| core::array::from_fn(|c| mat.get_at(r, c)))
}

/// Copies a 3×3 matrix into a filter buffer.
fn write_mat3(mat: &mut impl MatrixMut<3, 3, f32>, values: &Mat3) {
    for (r, row) in values.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            mat.set_at(r, c, *value);
        }
    }
}

/// Returns the true forward speed (m/s) and turn rate (rad/s) at the given step.
///
/// The robot drives a slalom: a constant speed with a slowly alternating turn.
fn twist(step: usize) -> [f32; 2] {
    #[allow(clippy::cast_precision_loss)]
    let t = step as f32 * DELTA_T;
    [SPEED, 0.4 * (0.2 * t).sin()]
}

/// Returns the body-frame tangent increment `(v·Δt, 0, ω·Δt)` for the given twist.
fn increment(twist: [f32; 2]) -> [f32; 3] {
    [twist[0] * DELTA_T, 0.0, twist[1] * DELTA_T]
}

/// Noisy odometry and position fix sensors.
struct Sensors {
    /// Seeded random source for reproducible noise.
    rng: StdRng,
    /// Speed noise in m/s.
    speed_noise: Normal<f32>,
    /// Turn rate noise in rad/s.
    turn_noise: Normal<f32>,
    /// Position fix noise per axis in meters.
    fix_noise: Normal<f32>,
}

impl Sensors {
    /// Creates the sensors with the configured noise levels and [`SEED`].
    fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(SEED),
            speed_noise: Normal::new(0.0, SPEED_NOISE_STD).expect("valid std"),
            turn_noise: Normal::new(0.0, TURN_RATE_NOISE_STD).expect("valid std"),
            fix_noise: Normal::new(0.0, FIX_NOISE_STD).expect("valid std"),
        }
    }

    /// Returns a noisy reading of the true twist.
    fn odometry(&mut self, twist: [f32; 2]) -> [f32; 2] {
        [
            twist[0] + self.speed_noise.sample(&mut self.rng),
            twist[1] + self.turn_noise.sample(&mut self.rng),
        ]
    }

    /// Returns a noisy world-frame position fix of the true pose.
    fn position(&mut self, truth: &Pose) -> [f32; 2] {
        [
            truth.x + self.fix_noise.sample(&mut self.rng),
            truth.y + self.fix_noise.sample(&mut self.rng),
        ]
    }
}

/// Returns the position error in meters and the absolute heading error in radians.
fn errors(truth: &Pose, estimate: &Pose) -> (f32, f32) {
    let position = (estimate.x - truth.x).hypot(estimate.y - truth.y);
    let heading = wrap_pi(estimate.theta - truth.theta).abs();
    (position, heading)
}

/// Prints the position and heading errors of both filters.
fn print_errors(step: usize, truth: &Pose, invariant: &Pose, naive: &Pose) {
    let (ip, ih) = errors(truth, invariant);
    let (np, nh) = errors(truth, naive);

    #[allow(clippy::cast_precision_loss)]
    let t = step as f32 * DELTA_T;
    println!(
        "t={t:5.1} s  invariant: {ip:6.2} m {:6.1}°   naive: {np:6.2} m {:6.1}°",
        ih.to_degrees(),
        nh.to_degrees(),
    );
}

/// Returns success if the invariant filter ended within the tolerances; reports the naive one.
fn check_convergence(truth: &Pose, invariant: &Pose, naive: &Pose) -> ExitCode {
    let (np, nh) = errors(truth, naive);
    println!(
        "Naive EKF final error: {np:.2} m, {:.1}° (not checked)",
        nh.to_degrees()
    );

    let (ip, ih) = errors(truth, invariant);
    if ip <= POSITION_TOLERANCE && ih <= HEADING_TOLERANCE {
        println!(
            "Invariant EKF converged: final error {ip:.2} m, {:.1}°",
            ih.to_degrees()
        );
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "Invariant EKF did not converge: final error {ip:.2} m, {:.1}° exceeds {POSITION_TOLERANCE} m or {:.1}°",
            ih.to_degrees(),
            HEADING_TOLERANCE.to_degrees(),
        );
        ExitCode::FAILURE
    }
}

/// Asserts that the SE(2) helpers match their defining identities.
///
/// `Log(Exp(τ)) = τ`, `Exp(u)⁻¹·Exp(ξ)·Exp(u) = Exp(Ad_{Exp(u)}⁻¹·ξ)` (exact), and
/// `Log(Exp(τ)⁻¹·Exp(τ + δ)) ≈ J_r(τ)·δ` to first order in `δ`.
fn check_lie_helpers() {
    let taus = [[0.3, -0.2, 1.1], [0.1, 0.4, 1e-5], [-1.0, 2.0, -2.5]];

    for tau in taus {
        assert_close(&Pose::exp(tau).log(), &tau, 1e-4, "Log(Exp(τ))");

        let step = Pose::exp(tau);
        let xi = [0.2, -0.1, 0.3];
        let conjugated = step.inverse().compose(&Pose::exp(xi)).compose(&step).log();
        assert_close(&conjugated, &mul(&step.inverse_adjoint(), xi), 1e-4, "Ad⁻¹");

        let delta = [1e-2, -2e-2, 1.5e-2];
        let perturbed = [tau[0] + delta[0], tau[1] + delta[1], tau[2] + delta[2]];
        let local = Pose::exp(tau)
            .inverse()
            .compose(&Pose::exp(perturbed))
            .log();
        assert_close(&local, &mul(&right_jacobian(tau), delta), 1e-3, "J_r");
    }
}

/// Returns `m·v`.
fn mul(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|r| (0..3).map(|c| m[r][c] * v[c]).sum())
}

/// Panics with `what` if any entry of `a` and `b` differs by more than `tolerance`.
fn assert_close(a: &[f32; 3], b: &[f32; 3], tolerance: f32, what: &str) {
    let close = a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance);
    assert!(close, "{what} self-check failed: {a:?} vs {b:?}");
}
