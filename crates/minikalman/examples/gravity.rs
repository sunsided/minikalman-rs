//! Estimating the gravitational acceleration from a dropped object.
//!
//! An object is dropped from 25 m and a range sensor above it measures its height at 30 Hz with
//! a noise of σ = 3 cm, for 2 s. The filter tracks height, velocity and `g` as a state, starting
//! from a poor guess of `g = 5 ± 5 m/s²`. It prints a summary table with the estimate and its
//! 1σ uncertainty from `P`, and fails if the true `g` ever leaves the 3σ band.
//!
//! The measurements are seeded, so the output is the same on every run. With `-- --chart`, it
//! also prints the mermaid chart shown in the README.
//!
//! Run with `cargo run --example gravity --features=std`.

#![forbid(unsafe_code)]

use minikalman::prelude::*;
use minikalman::regular::builder::{KalmanFilterBuilder, KalmanFilterType};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// States: height (m), upwards velocity (m/s), and `g` (m/s², pointing down).
const NUM_STATES: usize = 3;

/// Observations: the height (m).
const NUM_OBSERVATIONS: usize = 1;

/// The true gravitational acceleration (m/s²) the simulation drops the object with.
const TRUE_G: f32 = 9.807;

/// Height (m) the object is dropped from, at rest.
const DROP_HEIGHT: f32 = 25.0;

/// Sensor rate (Hz).
const RATE: usize = 30;

/// Time step (s) between two measurements.
const DT: f32 = 1.0 / RATE as f32;

/// Number of measurements: 2 s of data.
const NUM_STEPS: usize = 2 * RATE;

/// Standard deviation (m) of the height sensor.
const SENSOR_SIGMA: f32 = 0.03;

/// Initial guess for `g` (m/s²) and its standard deviation.
const G_GUESS: (f32, f32) = (5.0, 5.0);

/// Every how many steps a row goes into the summary table.
const TABLE_EVERY: usize = 6;

/// The first step in the chart; before it, the 3σ band is wider than the chart.
const CHART_START: usize = 9;

/// Every how many steps a point goes into the chart.
const CHART_EVERY: usize = 3;

/// The estimate of `g` after one correction.
struct Estimate {
    /// Time (s) since the drop.
    t: f32,
    /// The measured height (m).
    z: f32,
    /// The estimated `g` (m/s²).
    g: f32,
    /// The standard deviation (m/s²) of the estimated `g`, from `P`.
    sigma: f32,
}

impl Estimate {
    /// The distance of the true `g` from the estimate, in standard deviations.
    fn error_in_sigmas(&self) -> f32 {
        (self.g - TRUE_G).abs() / self.sigma
    }
}

fn main() {
    let history = run_filter();
    print_table(&history);

    let last = history.last().expect("at least one step");
    println!(
        "\nResult: g = {:.3} ± {:.3} m/s² (1σ), true 9.807 m/s², from {} measurements over {:.1} s",
        last.g, last.sigma, NUM_STEPS, last.t
    );

    if std::env::args().any(|arg| arg == "--chart") {
        print_chart(&history);
    }

    let worst = history
        .iter()
        .map(Estimate::error_in_sigmas)
        .fold(0.0, f32::max);
    assert!(worst < 3.0, "true g left the 3σ band ({worst:.2}σ)");
    assert!(last.sigma < 0.1, "g did not converge (σ = {})", last.sigma);
}

/// Simulates the drop, runs the filter over it and returns the estimate after every step.
fn run_filter() -> Vec<Estimate> {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    set_up_filter(&mut filter);
    measurement.observation_matrix_mut().set(0, 0, 1.0);
    measurement
        .measurement_noise_covariance_mut()
        .set(0, 0, SENSOR_SIGMA * SENSOR_SIGMA);

    let mut rng = StdRng::seed_from_u64(9807);
    let noise = Normal::new(0.0, SENSOR_SIGMA).expect("valid standard deviation");

    let mut history = Vec::with_capacity(NUM_STEPS);
    for step in 1..=NUM_STEPS {
        let t = step as f32 * DT;
        let z = DROP_HEIGHT - 0.5 * TRUE_G * t * t + noise.sample(&mut rng);

        filter.predict();
        measurement.measurement_vector_mut().set(0, 0, z);
        filter.correct(&mut measurement);

        history.push(Estimate {
            t,
            z,
            g: filter.state_vector().get(2, 0),
            sigma: filter.estimate_covariance().get(2, 2).sqrt(),
        });
    }

    history
}

/// Sets the free-fall model, the initial guess and its uncertainty.
///
/// ```math
/// h₁ = h₀ + Δt×v₀ − ½×Δt²×g
/// v₁ = v₀ − Δt×g
/// g₁ = g
/// ```
fn set_up_filter(filter: &mut KalmanFilterType<NUM_STATES, f32>) {
    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set(0, 1, DT);
        a.set(0, 2, -0.5 * DT * DT);
        a.set(1, 2, -DT);
    });

    // The height is roughly known, the object starts at rest, g is a poor guess.
    filter.state_vector_mut().apply(|x| {
        x.set(0, 0, DROP_HEIGHT + 0.5);
        x.set(1, 0, 0.0);
        x.set(2, 0, G_GUESS.0);
    });
    filter.estimate_covariance_mut().apply(|p| {
        p.set(0, 0, 1.0);
        p.set(1, 1, 0.1);
        p.set(2, 2, G_GUESS.1 * G_GUESS.1);
    });
}

/// Prints every [`TABLE_EVERY`]th estimate and the last one.
fn print_table(history: &[Estimate]) {
    println!("  t (s)  height (m)  g (m/s²)  ± 1σ    |ĝ − g|");
    for (i, e) in history.iter().enumerate() {
        if (i + 1) % TABLE_EVERY == 0 || i + 1 == history.len() {
            println!(
                "  {:5.2}  {:10.3}  {:8.3}  ± {:5.3}  {:4.2}σ",
                e.t,
                e.z,
                e.g,
                e.sigma,
                e.error_in_sigmas()
            );
        }
    }
}

/// Prints a mermaid `xychart-beta` of the estimate and its 3σ band, for the README.
fn print_chart(history: &[Estimate]) {
    let points: Vec<&Estimate> = history
        .iter()
        .skip(CHART_START - 1)
        .step_by(CHART_EVERY)
        .collect();
    let series = |f: &dyn Fn(&Estimate) -> f32| {
        points
            .iter()
            .map(|e| format!("{:.2}", f(e)))
            .collect::<Vec<_>>()
            .join(", ")
    };

    // Line colours in order: true g (green), +3σ (amber), estimate (blue), −3σ (amber).
    println!("\n```mermaid");
    println!(
        "%%{{init: {{\"themeVariables\": {{\"xyChart\": {{\"plotColorPalette\": \
         \"#16a34a, #f59e0b, #2563eb, #f59e0b\"}}}}}}}}%%"
    );
    println!("xychart-beta");
    println!("    title \"ĝ with ±3σ band vs. true g\"");
    println!(
        "    x-axis \"t (s)\" [{}]",
        points
            .iter()
            .map(|e| format!("\"{:.1}\"", e.t))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("    y-axis \"g (m/s²)\" 4 --> 14");
    println!("    line [{}]", series(&|_| TRUE_G));
    println!("    line [{}]", series(&|e| e.g + 3.0 * e.sigma));
    println!("    line [{}]", series(&|e| e.g));
    println!("    line [{}]", series(&|e| e.g - 3.0 * e.sigma));
    println!("```");
}
