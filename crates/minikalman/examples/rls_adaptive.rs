//! Dual estimation: recursive least squares identifies a plant online and feeds a Kalman filter.
//!
//! A motor's speed follows the first-order model `xₖ₊₁ = a·xₖ + b·uₖ + w`, measured with a noisy
//! sensor `z = x + v`. Halfway through the run, the load changes and the true `(a, b)` jump from
//! `(0.9, 0.5)` to `(0.7, 1.0)`. Two Kalman filters estimate the speed:
//!
//! - the **fixed** filter uses the nominal model `(0.9, 0.5)` throughout;
//! - the **adaptive** filter takes `A = â` and `B = b̂` from a [`RecursiveLeastSquares`]
//!   estimator each step. The estimator regresses each new measurement on the previous speed
//!   estimate and input, `zₖ = [x̂ₖ₋₁, uₖ₋₁]·[a, b]ᵀ + noise`, with the forgetting factor
//!   `μ = 0.98` (a memory of about 50 steps). The parameter uncertainty `φᵀ·P_θ·φ` is added
//!   to the filter's process noise, so that it trusts a freshly identified model less.
//!
//! Before the change, the fixed filter has the exact model, its best case; the adaptive filter
//! pays a little for estimating the model from noisy data. After the change, the fixed
//! filter's model is wrong and its error grows about eightfold, while the adaptive filter
//! re-identifies the plant within a few dozen steps and recovers. The example prints the
//! parameter estimates and the speed RMSE of both filters per phase, and fails if the adaptive
//! filter is not clearly better after the change.
//!
//! The noise is seeded, so the output is the same on every run. With `-- --chart`, it also
//! prints the mermaid chart shown in the README.
//!
//! Run with `cargo run --example rls-adaptive --features=std`.
//!
//! [`RecursiveLeastSquares`]: minikalman::rls::RecursiveLeastSquares

#![forbid(unsafe_code)]

use minikalman::prelude::*;
use minikalman::regular::{ControlStorage, FilterStorage, ObservationStorage};
use minikalman::rls::EstimatorStorage;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// The true `(a, b)` before the load change; also the fixed filter's model.
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

/// The forgetting factor of the estimator.
const FORGETTING: f64 = 0.98;

/// Initial variance of each parameter in the estimator.
const PARAMETER_VARIANCE: f64 = 1.0;

/// How many steps the input holds its value before it may switch.
const INPUT_HOLD: usize = 20;

/// Every how many steps a row goes into the parameter table.
const TABLE_EVERY: usize = 100;

/// The window (steps) of the rolling RMSE in the chart.
const CHART_WINDOW: usize = 25;

/// One step of the simulation.
struct Step {
    /// True speed.
    x: f64,
    /// Speed estimate of the fixed-model filter.
    fixed: f64,
    /// Speed estimate of the adaptive filter.
    adaptive: f64,
    /// The estimated `(a, b)` after this step.
    theta: [f64; 2],
}

fn main() {
    let history = simulate();
    print_parameters(&history);
    let (fixed, adaptive) = print_rmse(&history);

    if std::env::args().any(|arg| arg == "--chart") {
        print_chart(&history);
    }

    assert!(
        adaptive < 0.6 * fixed,
        "after the change, the adaptive filter should beat the fixed model clearly"
    );
}

/// Runs the plant, the estimator and both filters, and returns every step.
fn simulate() -> Vec<Step> {
    let mut rng = StdRng::seed_from_u64(42);
    let process = Normal::new(0.0, PROCESS_SIGMA).expect("valid sigma");
    let sensor = Normal::new(0.0, SENSOR_SIGMA).expect("valid sigma");

    let mut rls_storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut rls = rls_storage.build(PARAMETER_VARIANCE);
    rls.set_forgetting_factor(FORGETTING);
    rls.set_measurement_variance(SENSOR_SIGMA * SENSOR_SIGMA);
    rls.parameters_mut().apply(|theta| {
        theta.set(0, 0, NOMINAL[0]);
        theta.set(1, 0, NOMINAL[1]);
    });

    let mut fixed = SpeedFilter::new();
    let mut adaptive = SpeedFilter::new();
    let (mut x, mut u) = (0.0, 1.0);
    let mut history = Vec::with_capacity(STEPS);

    for k in 0..STEPS {
        let [a, b] = if k < CHANGE_AT { NOMINAL } else { CHANGED };
        x = a * x + b * u + process.sample(&mut rng);
        let z = x + sensor.sample(&mut rng);

        // Identify: zₖ ≈ a·x̂ₖ₋₁ + b·uₖ₋₁, with the adaptive filter's previous estimate.
        rls.update(&[adaptive.speed(), u], z);
        let theta = [rls.parameters().get_row(0), rls.parameters().get_row(1)];

        // The parameter uncertainty adds φᵀ·P_θ·φ to the model's process noise.
        let mut p_theta = [0.0; 4];
        rls.covariance(&mut MatrixData::new_mut::<2, 2, f64>(&mut p_theta));
        let phi = [adaptive.speed(), u];
        let model_variance = phi[0] * phi[0] * p_theta[0]
            + 2.0 * phi[0] * phi[1] * p_theta[1]
            + phi[1] * phi[1] * p_theta[3];

        fixed.step(NOMINAL, u, z, 0.0);
        adaptive.step(theta, u, z, model_variance);
        history.push(Step {
            x,
            fixed: fixed.speed(),
            adaptive: adaptive.speed(),
            theta,
        });

        if (k + 1) % INPUT_HOLD == 0 && rng.gen_bool(0.5) {
            u = -u;
        }
    }

    history
}

/// A one-state Kalman filter of the speed, with the model `(a, b)` given per step.
struct SpeedFilter {
    /// The filter buffers: `A`, `x`, `P`, `Q` and temporaries.
    filter: FilterStorage<1, f64>,
    /// The control buffers: `B` and `u`.
    control: ControlStorage<1, 1, f64>,
    /// The measurement buffers: `H`, `z` and `R`.
    measurement: ObservationStorage<1, 1, f64>,
}

impl SpeedFilter {
    /// Creates a filter at `x̂ = 0` with `P = 1`, the process and the sensor noise.
    fn new() -> Self {
        let mut speed = Self {
            filter: FilterStorage::new(0.0),
            control: ControlStorage::new(0.0),
            measurement: ObservationStorage::new(0.0),
        };

        let mut filter = speed.filter.build();
        filter.estimate_covariance_mut().set(0, 0, 1.0);
        filter
            .direct_process_noise_mut()
            .set(0, 0, PROCESS_SIGMA * PROCESS_SIGMA);

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

/// Prints the true and the estimated parameters every [`TABLE_EVERY`] steps.
fn print_parameters(history: &[Step]) {
    println!("  step   a      â       b      b̂");
    for (k, step) in history.iter().enumerate() {
        if (k + 1) % TABLE_EVERY == 0 || k + 1 == CHANGE_AT + 25 {
            let [a, b] = if k < CHANGE_AT { NOMINAL } else { CHANGED };
            let [a_hat, b_hat] = step.theta;
            println!("  {:4}   {a:.2}   {a_hat:.3}   {b:.2}   {b_hat:.3}", k + 1);
        }
    }
}

/// Prints the speed RMSE of both filters before the change, right after it, and once the
/// estimator has settled. Returns the settled RMSE of the fixed and the adaptive filter.
fn print_rmse(history: &[Step]) -> (f64, f64) {
    let phases = [
        ("before the change", 50..CHANGE_AT),
        ("first 50 steps after", CHANGE_AT..CHANGE_AT + 50),
        ("settled after", CHANGE_AT + 50..STEPS),
    ];

    println!();
    println!("  speed RMSE            fixed   adaptive   (sensor σ = {SENSOR_SIGMA})");
    let mut settled = (0.0, 0.0);
    for (name, range) in phases {
        let fixed = rmse(&history[range.clone()], |s| s.fixed);
        let adaptive = rmse(&history[range], |s| s.adaptive);
        println!("  {name:<20}  {fixed:.3}   {adaptive:.3}");
        settled = (fixed, adaptive);
    }

    settled
}

/// Returns the root mean square of `estimate − x` over `steps`.
#[allow(clippy::cast_precision_loss)]
fn rmse(steps: &[Step], estimate: impl Fn(&Step) -> f64) -> f64 {
    let sum: f64 = steps.iter().map(|s| (estimate(s) - s.x).powi(2)).sum();
    (sum / steps.len() as f64).sqrt()
}

/// Prints a mermaid chart of the rolling speed RMSE of both filters.
fn print_chart(history: &[Step]) {
    let windows: Vec<&[Step]> = history.chunks(CHART_WINDOW).collect();
    let labels: Vec<String> = (1..=windows.len())
        .map(|i| format!("\"{}\"", i * CHART_WINDOW))
        .collect();
    let line = |estimate: fn(&Step) -> f64| {
        let values: Vec<String> = windows
            .iter()
            .map(|w| format!("{:.2}", rmse(w, estimate)))
            .collect();
        values.join(", ")
    };

    println!();
    println!("```mermaid");
    println!(
        "%%{{init: {{\"themeVariables\": {{\"xyChart\": {{\"plotColorPalette\": \"#dc2626, #2563eb\"}}}}}}}}%%"
    );
    println!("xychart-beta");
    println!("    title \"Speed RMSE per {CHART_WINDOW} steps: fixed model vs. RLS-adaptive\"");
    println!("    x-axis \"step\" [{}]", labels.join(", "));
    println!("    y-axis \"RMSE\" 0 --> 1");
    println!("    line [{}]", line(|s| s.fixed));
    println!("    line [{}]", line(|s| s.adaptive));
    println!("```");
}
