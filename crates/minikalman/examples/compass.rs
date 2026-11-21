//! Tracking a compass heading across north.
//!
//! A vehicle turns at a constant rate of 2°/s, starting at 340°. Its heading crosses north
//! (359° → 1°) after ten seconds. A compass reports the heading in `[0°, 360°)` once a
//! second. The filter estimates the heading and the turn rate.
//!
//! Two filters run side by side:
//!
//! - **wrapped** corrects with [`KalmanFilter::correct_with`] and normalizes the innovation
//!   with [`angles::wrap_180`], so a reading of 1° against a prediction of 359° is a +2°
//!   innovation.
//! - **naive** uses the plain [`KalmanFilter::correct`]. It sees an innovation of about
//!   -358° at north and swings the long way around.

#![forbid(unsafe_code)]

use minikalman::angles;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;

/// Number of states: heading (°) and turn rate (°/s).
const NUM_STATES: usize = 2;

/// Number of observations: the compass heading (°).
const NUM_OBSERVATIONS: usize = 1;

/// Time between compass readings, in seconds.
const DELTA_T: f32 = 1.0;

/// True heading at `t = 0`, in degrees.
const START_HEADING: f32 = 340.0;

/// True turn rate, in degrees per second.
const TURN_RATE: f32 = 2.0;

/// Number of compass readings to simulate.
const NUM_STEPS: u16 = 30;

/// Readings after which the wrapped filter must have converged.
const SETTLE_STEPS: u16 = 5;

/// Largest heading error (°) the wrapped filter may show once it has settled.
const MAX_SETTLED_ERROR: f32 = 2.0;

fn main() {
    let wrapped = track(true);
    let naive = track(false);

    let mut max_wrapped_error: f32 = 0.0;
    let mut max_naive_error: f32 = 0.0;

    println!("   t   true  reading  wrapped (error)    naive (error)");
    for (step, (wrapped, naive)) in (1..=NUM_STEPS).zip(wrapped.iter().zip(&naive)) {
        let wrapped_error = heading_error(wrapped.estimate, wrapped.truth);
        let naive_error = heading_error(naive.estimate, naive.truth);

        println!(
            "{:4.0}  {:5.1}  {:7.1}  {:7.1} ({wrapped_error:5.1})  {:7.1} ({naive_error:6.1})",
            wrapped.t, wrapped.truth, wrapped.reading, wrapped.estimate, naive.estimate
        );

        if step > SETTLE_STEPS {
            max_wrapped_error = max_wrapped_error.max(wrapped_error.abs());
        }

        max_naive_error = max_naive_error.max(naive_error.abs());
    }

    println!(
        "Largest error: wrapped {max_wrapped_error:.1}° (after settling), naive {max_naive_error:.1}°"
    );

    assert!(
        max_wrapped_error < MAX_SETTLED_ERROR,
        "wrapped filter lost track: {max_wrapped_error}°"
    );
    assert!(
        max_naive_error > 90.0,
        "naive filter should swing the long way at north: {max_naive_error}°"
    );
}

/// One compass reading and the filter's heading after correcting with it.
struct Sample {
    /// Time of the reading, in seconds.
    t: f32,
    /// True heading in `[0°, 360°)`.
    truth: f32,
    /// Noisy compass reading in `[0°, 360°)`.
    reading: f32,
    /// Filtered heading after the correction, in degrees.
    estimate: f32,
}

/// Runs a constant turn rate filter over [`NUM_STEPS`] compass readings.
///
/// With `normalize`, the innovation is wrapped to `(-180°, 180°]` and the heading is kept in
/// `[0°, 360°)`. Without it, the filter uses the plain [`KalmanFilter::correct`].
fn track(normalize: bool) -> Vec<Sample> {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // Start at 330° and believe the vehicle is not turning.
    filter.state_vector_mut().apply(|x| {
        x[0] = 330.0;
        x[1] = 0.0;
    });

    // heading = heading + rate×∆t; rate = rate
    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 1, DELTA_T);
    });

    // Unsure about both the heading (σ = 20°) and the turn rate (σ = 5°/s).
    filter.estimate_covariance_mut().apply(|p| {
        p.set_at(0, 0, 400.0);
        p.set_at(1, 1, 25.0);
    });

    // The compass observes the heading directly with σ = 1°.
    measurement
        .observation_matrix_mut()
        .apply(|h| h.set_at(0, 0, 1.0));
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(1.0);

    let mut samples = Vec::new();
    for step in 1..=NUM_STEPS {
        let t = f32::from(step) * DELTA_T;
        let truth = to_compass(START_HEADING + TURN_RATE * t);
        let reading = to_compass(truth + sensor_noise(t));
        measurement.measurement_vector_mut().set_at(0, 0, reading);

        filter.predict();
        if normalize {
            filter.correct_with(&mut measurement, |y| y[0] = angles::wrap_180(y[0]));
            filter.state_vector_mut().apply(|x| x[0] = to_compass(x[0]));
        } else {
            filter.correct(&mut measurement);
        }

        let estimate = filter.state_vector().get_at(0, 0);
        samples.push(Sample {
            t,
            truth,
            reading,
            estimate,
        });
    }

    samples
}

/// Maps any angle in degrees to the compass range `[0°, 360°)`.
fn to_compass(degrees: f32) -> f32 {
    degrees.rem_euclid(360.0)
}

/// Signed shortest difference `estimate - truth` in degrees, in `(-180°, 180°]`.
fn heading_error(estimate: f32, truth: f32) -> f32 {
    angles::wrap_180(estimate - truth)
}

/// Deterministic compass noise of up to ±1° at time `t` (seconds), so CI runs are repeatable.
fn sensor_noise(t: f32) -> f32 {
    (t * 1.7).sin()
}
