//! Designing a fixed-gain (Luenberger) observer with `Q` and `R`, then running it without a
//! covariance.
//!
//! A constant-velocity tracker (`dt = 0.1 s`, acceleration variance `1 (m/s²)²`) measures the
//! position with a variance of `0.25 m²`. The design step iterates a regular Kalman filter to
//! its steady-state gain `K∞` with `settle_gain`, as it could on a host or at boot. The run
//! step uses only the observer's storage (`A`, `x`, one temporary) and the fixed-gain
//! observation (`H`, `z`, `y`, `K`), in `f32` and in `I16F16`, next to the full Kalman filter
//! on the same noisy readings.

#![forbid(unsafe_code)]

use minikalman::models::motion;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;
use minikalman::regular::{
    FilterStorage, FixedGainObservationStorage, ObservationStorage, ObserverStorage,
    PredictorObserverStorage,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Time step in seconds.
const DT: f32 = 0.1;

/// Acceleration variance of the motion model, in (m/s²)².
const ACCELERATION_VARIANCE: f32 = 1.0;

/// Position measurement variance, in m².
const R: f32 = 0.25;

/// Number of simulated steps.
const STEPS: usize = 400;

/// Steps after which the start-up transient is taken to be over.
const SETTLED: usize = 100;

fn main() {
    let gain = design_gain();
    println!(
        "K∞ = [{:.6}, {:.6}] (SciPy: [0.181201, 0.180975])",
        gain[0], gain[1]
    );

    let (truth, readings) = simulate();
    let kalman = run_kalman(&readings);
    let observer = run_observer(gain, &readings);
    let fixed = run_fixed_point(gain, &readings);
    let ahead = run_predictor(gain, &readings);

    let gap = (SETTLED..STEPS)
        .map(|k| (kalman[k] - observer[k]).abs())
        .fold(0.0, f32::max);
    println!(
        "position RMSE after {SETTLED} steps: Kalman {:.4} m, observer {:.4} m, I16F16 observer {:.4} m",
        rmse(&kalman, &truth),
        rmse(&observer, &truth),
        rmse(&fixed, &truth)
    );
    println!("largest Kalman vs observer gap after {SETTLED} steps: {gap:e} m");

    // The predictor form with L = A·K predicts the next position from the current estimate.
    let ahead_rmse = ((SETTLED..STEPS - 1)
        .map(|k| (ahead[k] - truth[k + 1]).powi(2))
        .sum::<f32>()
        / (STEPS - 1 - SETTLED) as f32)
        .sqrt();
    println!("predictor form: one-step-ahead position RMSE {ahead_rmse:.4} m");
    println!(
        "memory: Kalman {} + {} bytes, observer {} + {} bytes (f32)",
        size_of::<FilterStorage<2, f32>>(),
        size_of::<ObservationStorage<2, 1, f32>>(),
        size_of::<ObserverStorage<2, f32>>(),
        size_of::<FixedGainObservationStorage<2, 1, f32>>()
    );
    assert!(gap < 1e-3, "the observer is the steady-state Kalman filter");
}

/// Iterates a regular Kalman filter's covariance recursion until its gain settles, and returns
/// the current-form steady-state gain `K∞`.
fn design_gain() -> [f32; 2] {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut filter = builder.build();
    let mut position = builder.observations().build::<1>();
    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);

    filter
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    filter
        .direct_process_noise_mut()
        .apply(|q| model.fill_process_noise(q));
    filter.estimate_covariance_mut().make_identity();
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    position.measurement_noise_covariance_mut().set_at(0, 0, R);

    let iterations = filter
        .settle_gain(&mut position, 1e-6, 10_000)
        .expect("the constant-velocity model is detectable");
    println!("settle_gain: {iterations} iterations");

    let k = position.kalman_gain();
    [k.get_at(0, 0), k.get_at(1, 0)]
}

/// Simulates the target and returns its true positions and the noisy readings.
fn simulate() -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(7);
    let acceleration = Normal::new(0.0, ACCELERATION_VARIANCE.sqrt()).expect("valid");
    let noise = Normal::new(0.0, R.sqrt()).expect("valid");

    let (mut position, mut velocity) = (0.0_f32, 1.0_f32);
    let mut truth = Vec::with_capacity(STEPS);
    let mut readings = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let a = acceleration.sample(&mut rng);
        position += velocity * DT + 0.5 * a * DT * DT;
        velocity += a * DT;
        truth.push(position);
        readings.push(position + noise.sample(&mut rng));
    }

    (truth, readings)
}

/// Runs the full Kalman filter and returns its position estimates.
fn run_kalman(readings: &[f32]) -> Vec<f32> {
    let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
    let mut position_storage = ObservationStorage::<2, 1, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut position = position_storage.build();
    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);

    filter
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    filter
        .direct_process_noise_mut()
        .apply(|q| model.fill_process_noise(q));
    filter.estimate_covariance_mut().make_scalar(10.0);
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    position.measurement_noise_covariance_mut().set_at(0, 0, R);

    readings
        .iter()
        .map(|&z| {
            filter.predict();
            position.measurement_vector_mut().set_at(0, 0, z);
            filter.correct(&mut position);
            filter.state_vector().get_at(0, 0)
        })
        .collect()
}

/// Runs the fixed-gain observer in `f32` and returns its position estimates.
fn run_observer(gain: [f32; 2], readings: &[f32]) -> Vec<f32> {
    let mut observer_storage = ObserverStorage::<2, f32>::new(0.0);
    let mut position_storage = FixedGainObservationStorage::<2, 1, f32>::new(0.0);
    let mut observer = observer_storage.build();
    let mut position = position_storage.build();

    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>(gain));

    readings
        .iter()
        .map(|&z| {
            observer.predict();
            position.measurement_vector_mut().set_at(0, 0, z);
            let status = observer.correct(&mut position);
            assert!(status.is_applied());
            observer.state_vector().get_at(0, 0)
        })
        .collect()
}

/// Runs the fixed-gain observer in the predictor form with `L = A·K` and returns its
/// one-step-ahead predictions `x̂ₖ₊₁|ₖ` of the position.
///
/// `correct` defers `L·y` without touching the state; the next `predict` applies it.
fn run_predictor(gain: [f32; 2], readings: &[f32]) -> Vec<f32> {
    let mut observer_storage = PredictorObserverStorage::<2, f32>::new(0.0);
    let mut position_storage = FixedGainObservationStorage::<2, 1, f32, Predictor>::new(0.0);
    let mut observer = observer_storage.build();
    let mut position = position_storage.build();

    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    // L = A·K for A = [[1, dt], [0, 1]].
    let predictor_gain = [gain[0] + DT * gain[1], gain[1]];
    position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>(predictor_gain));

    readings
        .iter()
        .map(|&z| {
            position.measurement_vector_mut().set_at(0, 0, z);
            let status = observer.correct(&mut position);
            assert!(status.is_applied());
            observer.predict();
            observer.state_vector().get_at(0, 0)
        })
        .collect()
}

/// Runs the fixed-gain observer in `I16F16`, with the gain designed in `f32`, and returns its
/// position estimates.
fn run_fixed_point(gain: [f32; 2], readings: &[f32]) -> Vec<f32> {
    let mut observer_storage = ObserverStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut position_storage = FixedGainObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut observer = observer_storage.build();
    let mut position = position_storage.build();

    let model = motion::constant_velocity(I16F16::from_num(DT), I16F16::ONE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, I16F16::ONE);
    position.set_gain(&MatrixData::new_array::<2, 1, 2, I16F16>(
        gain.map(I16F16::from_num),
    ));

    // Seed the state from the first reading to keep the start-up innovation small.
    observer
        .state_vector_mut()
        .set_at(0, 0, I16F16::from_num(readings[0]));

    readings
        .iter()
        .map(|&z| {
            observer.predict();
            position
                .measurement_vector_mut()
                .set_at(0, 0, I16F16::from_num(z));
            let status = observer.correct(&mut position);
            assert!(status.is_applied());
            observer.state_vector().get_at(0, 0).to_num::<f32>()
        })
        .collect()
}

/// Root-mean-square error of the estimates after the start-up transient.
fn rmse(estimates: &[f32], truth: &[f32]) -> f32 {
    let sum: f32 = (SETTLED..STEPS)
        .map(|k| (estimates[k] - truth[k]).powi(2))
        .sum();
    (sum / (STEPS - SETTLED) as f32).sqrt()
}
