//! Joint (augmented-state) estimation: one filter estimates the speed and the model together.
//!
//! This is the companion of the [`rls-adaptive`] example and runs on the same plant, load step,
//! noise and seed, so the numbers compare directly. A motor's speed follows the first-order
//! model `xₖ₊₁ = a·xₖ + b·uₖ + w`, measured with a noisy sensor `z = x + v`. At step 400 the
//! load changes and the true `(a, b)` jump from `(0.9, 0.5)` to `(0.7, 1.0)`.
//!
//! ## Joint vs. dual estimation
//!
//! - **Dual** estimation (the [`rls-adaptive`] example) runs two estimators side by side: a
//!   recursive least squares estimator identifies `(a, b)`, and a linear Kalman filter takes
//!   `A = â`, `B = b̂` from it each step. Each one treats the other's output as exact, apart
//!   from the parameter variance that is added to `Q`.
//! - **Joint** estimation (this example) appends the unknown parameters to the state and runs
//!   one nonlinear filter on the augmented state `s = [x, a, b]`. Its covariance also holds the
//!   cross-covariance between the speed and the parameters, so a speed residual corrects
//!   `â` and `b̂` in proportion to how much they explain it. No new API is needed: it is an
//!   ordinary extended (or unscented) Kalman filter.
//!
//! ## The augmented model
//!
//! The parameters are modelled as a random walk, so the transition is
//!
//! ```text
//! f(s, u) = [a·x + b·u, a, b]
//! ```
//!
//! It is nonlinear because of the products `a·x` and `b·u`. The EKF linearizes it with the
//! Jacobian at the current estimate, and the observation stays linear:
//!
//! ```text
//!     ⎡ a  x  u ⎤
//! F = ⎢ 0  1  0 ⎥        H = [1  0  0]
//!     ⎣ 0  0  1 ⎦
//! ```
//!
//! The UKF variant propagates seven sigma points through `f` instead and needs no Jacobian.
//!
//! ## Process noise on the parameters
//!
//! `Q = diag(σ_w², q, q)`. The speed entry is the plant noise; `q` is how far the filter
//! assumes `a` and `b` may wander per step. With `q = 0` the parameter variance only shrinks,
//! and the filter stops learning: it cannot follow the load step. A larger `q` keeps the
//! parameter variance up, so the filter re-identifies faster, but it also lets measurement noise
//! into `â` and `b̂`. It plays the role of RLS's forgetting factor; `q = 10⁻⁴` gives a memory
//! of the same order as the `μ = 0.98` of [`rls-adaptive`]. The example sweeps `q` to show the
//! trade-off.
//!
//! ## Output and bounds
//!
//! The example prints the speed RMSE and the RMS parameter errors of the joint EKF, the joint
//! UKF and the dual RLS + KF, all from the same recorded run, then the `q` sweep and the size of
//! each estimator's (allocation-free) storage. It fails unless both joint filters keep
//! `|â − a| < 0.05` and `|b̂ − b| < 0.1` (RMS) before the step and recover to those bounds once
//! settled after it. All estimators start at the true pre-step `(a, b)` with variance 1, so
//! the bound before the step shows that the estimates do not drift; the bound after it is the
//! recovery.
//!
//! On this run, the joint filters re-identify the plant about twice as fast as the dual
//! estimator (30 to 45 vs. about 90 steps) and settle with smaller parameter errors, because the
//! cross-covariance lets every speed residual move `â` and `b̂` at once. The dual estimator has
//! the lower speed RMSE while the model holds: its linear filter does not carry the parameter
//! uncertainty in its own covariance. The UKF matches the EKF here, since `f` is only bilinear.
//!
//! The noise is seeded, so the output is the same on every run.
//!
//! Run with `cargo run --example augmented-state --features=std`.
//!
//! [`rls-adaptive`]: https://github.com/sunsided/minikalman-rs/tree/main/crates/minikalman/examples/rls_adaptive.rs

#![forbid(unsafe_code)]

use std::ops::Range;

use minikalman::extended::{
    FilterStorage as EkfStorage, ObservationStorage as EkfObservationStorage,
};
use minikalman::prelude::*;
use minikalman::regular::{
    ControlStorage, FilterStorage as KfStorage, ObservationStorage as KfObservationStorage,
};
use minikalman::rls::EstimatorStorage;
use minikalman::unscented::{
    FilterStorage as UkfStorage, ObservationStorage as UkfObservationStorage,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// The true `(a, b)` before the load change; also every estimator's initial guess.
const NOMINAL: [f64; 2] = [0.9, 0.5];

/// The true `(a, b)` after the load change.
const CHANGED: [f64; 2] = [0.7, 1.0];

/// Number of steps.
const STEPS: usize = 800;

/// The step at which the load changes.
const CHANGE_AT: usize = 400;

/// Standard deviation of the process noise `w`.
const PROCESS_SIGMA: f64 = 0.05;

/// Standard deviation of the sensor noise `v`.
const SENSOR_SIGMA: f64 = 0.2;

/// The forgetting factor of the dual estimator's RLS.
const FORGETTING: f64 = 0.98;

/// Initial variance of each parameter, in every estimator.
const PARAMETER_VARIANCE: f64 = 1.0;

/// How many steps the input holds its value before it may switch.
const INPUT_HOLD: usize = 20;

/// The parameter process noise `q` of the joint filters in the comparison table.
const PARAMETER_NOISE: f64 = 1e-4;

/// The parameter process noises of the sweep.
const PARAMETER_NOISE_SWEEP: [f64; 4] = [0.0, 1e-5, 1e-4, 1e-3];

/// The bound on the RMS error of `â` before the step and once settled after it.
const A_BOUND: f64 = 0.05;

/// The bound on the RMS error of `b̂` before the step and once settled after it.
const B_BOUND: f64 = 0.1;

/// Number of augmented states: `[x, a, b]`.
const STATES: usize = 3;

/// Number of sigma points of the UKF, `2·STATES + 1`.
const SIGMA_POINTS: usize = 2 * STATES + 1;

/// The steps before the load change, after the initial transient.
const BEFORE: Range<usize> = 50..CHANGE_AT;

/// The first steps after the load change, while the estimators re-identify the plant.
const RIGHT_AFTER: Range<usize> = CHANGE_AT..CHANGE_AT + 50;

/// The steps after the load change, once the estimators have settled.
const SETTLED: Range<usize> = CHANGE_AT + 50..STEPS;

/// One recorded step of the plant.
struct Sample {
    /// True speed after this step.
    x: f64,
    /// Measured speed after this step.
    z: f64,
    /// Input applied in this step.
    u: f64,
    /// True `(a, b)` in this step.
    theta: [f64; 2],
}

/// One step of an estimator's output.
struct Estimate {
    /// Speed estimate `x̂`.
    x: f64,
    /// Parameter estimate `(â, b̂)`.
    theta: [f64; 2],
}

fn main() {
    let plant = simulate_plant();
    let joint_ekf = run_joint_ekf(&plant, PARAMETER_NOISE);
    let joint_ukf = run_joint_ukf(&plant, PARAMETER_NOISE);
    let dual = run_dual(&plant);

    print_comparison(
        &plant,
        &[
            ("joint EKF", &joint_ekf),
            ("joint UKF", &joint_ukf),
            ("dual RLS + KF", &dual),
        ],
    );
    print_sweep(&plant);
    print_storage_sizes();

    for (name, run) in [("joint EKF", &joint_ekf), ("joint UKF", &joint_ukf)] {
        for range in [BEFORE, SETTLED] {
            let [a, b] = parameter_rms(&plant, run, range.clone());
            assert!(
                a < A_BOUND && b < B_BOUND,
                "{name}: parameter error over {range:?} is (â: {a:.3}, b̂: {b:.3})"
            );
        }
    }
}

/// Runs the plant with the noise sequence of the `rls-adaptive` example and records each step.
fn simulate_plant() -> Vec<Sample> {
    let mut rng = StdRng::seed_from_u64(42);
    let process = Normal::new(0.0, PROCESS_SIGMA).expect("valid sigma");
    let sensor = Normal::new(0.0, SENSOR_SIGMA).expect("valid sigma");

    let (mut x, mut u) = (0.0, 1.0);
    let mut plant = Vec::with_capacity(STEPS);
    for k in 0..STEPS {
        let theta = if k < CHANGE_AT { NOMINAL } else { CHANGED };
        x = theta[0] * x + theta[1] * u + process.sample(&mut rng);
        let z = x + sensor.sample(&mut rng);
        plant.push(Sample { x, z, u, theta });

        if (k + 1) % INPUT_HOLD == 0 && rng.gen_bool(0.5) {
            u = -u;
        }
    }

    plant
}

/// Runs the joint EKF on the augmented state `[x, a, b]` with parameter process noise `q`.
fn run_joint_ekf(plant: &[Sample], q: f64) -> Vec<Estimate> {
    let mut filter_storage = EkfStorage::<STATES, f64>::new(0.0);
    let mut measurement_storage = EkfObservationStorage::<STATES, 1, f64>::new(0.0);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();

    filter.state_vector_mut().apply(|s| {
        s[1] = NOMINAL[0];
        s[2] = NOMINAL[1];
    });
    filter.estimate_covariance_mut().apply(|p| {
        p.set_at(0, 0, 1.0);
        p.set_at(1, 1, PARAMETER_VARIANCE);
        p.set_at(2, 2, PARAMETER_VARIANCE);
    });
    filter.direct_process_noise_mut().apply(|n| set_noise(n, q));
    measurement
        .observation_jacobian_matrix_mut()
        .apply(|h| h.set_at(0, 0, 1.0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, SENSOR_SIGMA * SENSOR_SIGMA));

    let mut run = Vec::with_capacity(plant.len());
    for sample in plant {
        let u = sample.u;
        let (x, a) = (filter.state_vector()[0], filter.state_vector()[1]);

        // F = ∂f/∂s at the current estimate.
        filter.state_transition_jacobian_mut().apply(|f| {
            f.make_identity();
            f.set_at(0, 0, a);
            f.set_at(0, 1, x);
            f.set_at(0, 2, u);
        });
        filter.predict_nonlinear(|s, next| {
            next[0] = s[1] * s[0] + s[2] * u;
            next[1] = s[1];
            next[2] = s[2];
        });

        measurement
            .measurement_vector_mut()
            .apply(|z| z[0] = sample.z);
        filter.correct_nonlinear(&mut measurement, |s, z| z[0] = s[0]);
        run.push(augmented_estimate(filter.state_vector()));
    }

    run
}

/// Runs the joint UKF on the augmented state `[x, a, b]` with parameter process noise `q`.
fn run_joint_ukf(plant: &[Sample], q: f64) -> Vec<Estimate> {
    let mut filter_storage = UkfStorage::<STATES, SIGMA_POINTS, f64>::new(0.0);
    let mut measurement_storage = UkfObservationStorage::<STATES, 1, SIGMA_POINTS, f64>::new(0.0);
    // α = 1, β = 2, κ = 1, as in the radar-2d-ukf example.
    let mut filter = filter_storage.build(1.0, 2.0, 1.0);
    let mut measurement = measurement_storage.build();

    filter.state_vector_mut().apply(|s| {
        s[1] = NOMINAL[0];
        s[2] = NOMINAL[1];
    });
    filter.estimate_covariance_mut().apply(|p| {
        p.set_at(0, 0, 1.0);
        p.set_at(1, 1, PARAMETER_VARIANCE);
        p.set_at(2, 2, PARAMETER_VARIANCE);
    });
    filter.direct_process_noise_mut().apply(|n| set_noise(n, q));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, SENSOR_SIGMA * SENSOR_SIGMA));

    let mut run = Vec::with_capacity(plant.len());
    for sample in plant {
        let u = sample.u;
        filter.predict_nonlinear(|s| s[0] = s[1] * s[0] + s[2] * u);

        measurement
            .measurement_vector_mut()
            .apply(|z| z[0] = sample.z);
        // H = [1, 0, 0]: each sigma point's speed is its predicted measurement. Row 0 of the
        // row-major sigma point matrix holds the speeds.
        filter.correct_sigma_point(&mut measurement, |sigma, observed| {
            for j in 0..SIGMA_POINTS {
                observed[j] = sigma[j];
            }
        });
        run.push(augmented_estimate(filter.state_vector()));
    }

    run
}

/// Writes `Q = diag(σ_w², q, q)` into the augmented filter's process noise.
fn set_noise<M: MatrixMut<STATES, STATES, f64>>(noise: &mut M, q: f64) {
    noise.set_at(0, 0, PROCESS_SIGMA * PROCESS_SIGMA);
    noise.set_at(1, 1, q);
    noise.set_at(2, 2, q);
}

/// Reads speed and parameters from an augmented state vector `[x, a, b]`.
fn augmented_estimate(state: &impl Matrix<STATES, 1, f64>) -> Estimate {
    Estimate {
        x: state.get_row(0),
        theta: [state.get_row(1), state.get_row(2)],
    }
}

/// Runs the dual estimator of the `rls-adaptive` example: RLS identifies `(a, b)` and feeds a
/// linear Kalman filter of the speed.
fn run_dual(plant: &[Sample]) -> Vec<Estimate> {
    let mut rls_storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut rls = rls_storage.build(PARAMETER_VARIANCE);
    rls.set_forgetting_factor(FORGETTING);
    rls.set_measurement_variance(SENSOR_SIGMA * SENSOR_SIGMA);
    rls.parameters_mut().apply(|theta| {
        theta.set(0, 0, NOMINAL[0]);
        theta.set(1, 0, NOMINAL[1]);
    });

    let mut filter = SpeedFilter::new();
    let mut run = Vec::with_capacity(plant.len());
    for sample in plant {
        // Identify: zₖ ≈ a·x̂ₖ₋₁ + b·uₖ₋₁, with the filter's previous estimate.
        let phi = [filter.speed(), sample.u];
        rls.update(&phi, sample.z);
        let theta = [rls.parameters().get_row(0), rls.parameters().get_row(1)];

        // The parameter uncertainty adds φᵀ·P_θ·φ to the model's process noise.
        let mut p_theta = [0.0; 4];
        rls.covariance(&mut MatrixData::new_mut::<2, 2, f64>(&mut p_theta));
        let model_variance = phi[0] * phi[0] * p_theta[0]
            + 2.0 * phi[0] * phi[1] * p_theta[1]
            + phi[1] * phi[1] * p_theta[3];

        filter.step(theta, sample.u, sample.z, model_variance);
        run.push(Estimate {
            x: filter.speed(),
            theta,
        });
    }

    run
}

/// A one-state Kalman filter of the speed, with the model `(a, b)` given per step; the
/// adaptive filter of the `rls-adaptive` example.
struct SpeedFilter {
    /// The filter buffers: `A`, `x`, `P`, `Q` and temporaries.
    filter: KfStorage<1, f64>,
    /// The control buffers: `B` and `u`.
    control: ControlStorage<1, 1, f64>,
    /// The measurement buffers: `H`, `z` and `R`.
    measurement: KfObservationStorage<1, 1, f64>,
}

impl SpeedFilter {
    /// Creates a filter at `x̂ = 0` with `P = 1`, the process and the sensor noise.
    fn new() -> Self {
        let mut speed = Self {
            filter: KfStorage::new(0.0),
            control: ControlStorage::new(0.0),
            measurement: KfObservationStorage::new(0.0),
        };

        let mut filter = speed.filter.build();
        filter.estimate_covariance_mut().set(0, 0, 1.0);
        let mut measurement = speed.measurement.build();
        measurement.observation_matrix_mut().set(0, 0, 1.0);
        measurement
            .measurement_noise_covariance_mut()
            .set(0, 0, SENSOR_SIGMA * SENSOR_SIGMA);
        speed
    }

    /// Returns the speed estimate `x̂`.
    fn speed(&mut self) -> f64 {
        self.filter.build().state_vector().get_row(0)
    }

    /// Predicts with `x̂ ← a·x̂ + b·u` and the process noise plus `model_variance`, and
    /// corrects with the measurement `z`.
    fn step(&mut self, [a, b]: [f64; 2], u: f64, z: f64, model_variance: f64) {
        let mut filter = self.filter.build();
        let mut control = self.control.build();
        let mut measurement = self.measurement.build();

        filter.state_transition_mut().set(0, 0, a);
        filter
            .direct_process_noise_mut()
            .set(0, 0, PROCESS_SIGMA * PROCESS_SIGMA + model_variance);
        control.control_matrix_mut().set(0, 0, b);
        control.control_vector_mut().set(0, 0, u);
        measurement.measurement_vector_mut().set(0, 0, z);

        filter.predict().control(&mut control);
        filter.correct(&mut measurement);
    }
}

/// Prints the speed RMSE per phase, the RMS parameter errors before and after the step, and
/// how many steps after the step `(â, b̂)` first come within the bounds.
fn print_comparison(plant: &[Sample], runs: &[(&str, &Vec<Estimate>)]) {
    println!(
        "                 speed RMSE (σ_v = {SENSOR_SIGMA})   â − a (RMS)     b̂ − b (RMS)     re-id."
    );
    println!("  estimator      before  +50    settled  before  settled  before  settled  (steps)");
    for (name, run) in runs {
        let speed = [BEFORE, RIGHT_AFTER, SETTLED].map(|r| speed_rmse(plant, run, r));
        let [a_before, b_before] = parameter_rms(plant, run, BEFORE);
        let [a_settled, b_settled] = parameter_rms(plant, run, SETTLED);
        println!(
            "  {name:<13}  {:.3}   {:.3}  {:.3}    {a_before:.3}   {a_settled:.3}    {b_before:.3}   {b_settled:.3}    {}",
            speed[0],
            speed[1],
            speed[2],
            steps_to_reidentify(plant, run)
        );
    }
}

/// Prints how the parameter process noise `q` trades re-identification speed against noise.
fn print_sweep(plant: &[Sample]) {
    println!();
    println!("  joint EKF      â − a (RMS)     b̂ − b (RMS)     speed RMSE  re-id.");
    println!("  q              before  settled  before  settled  settled     (steps)");
    for q in PARAMETER_NOISE_SWEEP {
        let run = run_joint_ekf(plant, q);
        let [a_before, b_before] = parameter_rms(plant, &run, BEFORE);
        let [a_settled, b_settled] = parameter_rms(plant, &run, SETTLED);
        let speed = speed_rmse(plant, &run, SETTLED);
        println!(
            "  {q:<13.0e}  {a_before:.3}   {a_settled:.3}    {b_before:.3}   {b_settled:.3}    {speed:.3}       {}",
            steps_to_reidentify(plant, &run)
        );
    }
}

/// Prints the size of each estimator's buffers; none of them allocates.
fn print_storage_sizes() {
    let ekf =
        size_of::<EkfStorage<STATES, f64>>() + size_of::<EkfObservationStorage<STATES, 1, f64>>();
    let ukf = size_of::<UkfStorage<STATES, SIGMA_POINTS, f64>>()
        + size_of::<UkfObservationStorage<STATES, 1, SIGMA_POINTS, f64>>();
    let dual = size_of::<EstimatorStorage<2, f64>>() + size_of::<SpeedFilter>();

    println!();
    println!("  storage (f64, no allocation): joint EKF {ekf} B, joint UKF {ukf} B, dual {dual} B");
}

/// Returns how many steps after the load change `â` and `b̂` first come within [`A_BOUND`] and
/// [`B_BOUND`] of the new `(a, b)`, or `"never"`.
fn steps_to_reidentify(plant: &[Sample], run: &[Estimate]) -> String {
    let within = |k: &usize| {
        let [a, b] = plant[*k].theta;
        let [a_hat, b_hat] = run[*k].theta;
        (a_hat - a).abs() < A_BOUND && (b_hat - b).abs() < B_BOUND
    };

    (CHANGE_AT..STEPS)
        .find(within)
        .map_or_else(|| "never".to_owned(), |k| (k + 1 - CHANGE_AT).to_string())
}

/// Returns the root mean square of `x̂ − x` over the steps `range`.
fn speed_rmse(plant: &[Sample], run: &[Estimate], range: Range<usize>) -> f64 {
    rms(range.map(|k| run[k].x - plant[k].x))
}

/// Returns the root mean square of `â − a` and `b̂ − b` over the steps `range`.
fn parameter_rms(plant: &[Sample], run: &[Estimate], range: Range<usize>) -> [f64; 2] {
    [0, 1].map(|i| rms(range.clone().map(|k| run[k].theta[i] - plant[k].theta[i])))
}

/// Returns the root mean square of `errors`.
#[allow(clippy::cast_precision_loss)]
fn rms(errors: impl ExactSizeIterator<Item = f64>) -> f64 {
    let n = errors.len() as f64;
    (errors.map(|e| e * e).sum::<f64>() / n).sqrt()
}
