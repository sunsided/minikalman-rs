//! Tracking a noisy 1D position in the browser.
//!
//! This crate builds minikalman for `wasm32-unknown-unknown` and exposes a
//! constant-velocity filter through plain `extern "C"` functions. All arguments
//! and return values are `f32`, so the module can be loaded with
//! `WebAssembly.instantiate` alone; no `wasm-bindgen` or JS glue is required.
//! See `www/index.html` for the browser side and `smoke.mjs` for a Node check.

use std::cell::RefCell;

use minikalman::prelude::*;
use minikalman::regular::builder::{
    KalmanFilterBuilder, KalmanFilterObservationType, KalmanFilterType,
};

const NUM_STATES: usize = 2; // position, velocity
const NUM_OBSERVATIONS: usize = 1; // position

struct Tracker {
    filter: KalmanFilterType<NUM_STATES, f32>,
    measurement: KalmanFilterObservationType<NUM_STATES, NUM_OBSERVATIONS, f32>,
}

impl Tracker {
    fn new(dt: f32, q: f32, r: f32) -> Self {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

        // Start at rest at the origin, with high uncertainty.
        filter.state_vector_mut().set_zero();
        filter
            .estimate_covariance_mut()
            .apply(|mat| mat.make_scalar(1000.0));

        // p = p + v×∆t
        // v = v
        filter.state_transition_mut().apply(|mat| {
            mat.set_at(0, 0, 1.0);
            mat.set_at(0, 1, dt);
            mat.set_at(1, 0, 0.0);
            mat.set_at(1, 1, 1.0);
        });

        // Discrete white noise acceleration with spectral density q.
        filter.direct_process_noise_mut().apply(|mat| {
            let dt2 = dt * dt;
            mat.set_at(0, 0, q * dt2 * dt2 / 4.0);
            mat.set_at(0, 1, q * dt2 * dt / 2.0);
            mat.set_at(1, 0, q * dt2 * dt / 2.0);
            mat.set_at(1, 1, q * dt2);
        });

        // We observe the position only.
        measurement.observation_matrix_mut().apply(|mat| {
            mat.set_at(0, 0, 1.0);
            mat.set_at(0, 1, 0.0);
        });
        measurement
            .measurement_noise_covariance_mut()
            .apply(|mat| mat.set_at(0, 0, r));

        Self {
            filter,
            measurement,
        }
    }

    fn step(&mut self, z: f32) -> f32 {
        self.filter.predict();
        self.measurement
            .measurement_vector_mut()
            .apply(|vec| vec.set_at(0, 0, z));
        self.filter.correct(&mut self.measurement);
        self.filter.state_vector().get_at(0, 0)
    }
}

thread_local! {
    static TRACKER: RefCell<Tracker> = RefCell::new(Tracker::new(1.0, 1.0, 1.0));
}

/// Resets the filter with time step `dt`, process noise density `q`
/// and measurement noise variance `r`.
#[unsafe(no_mangle)]
pub extern "C" fn reset(dt: f32, q: f32, r: f32) {
    TRACKER.with(|t| *t.borrow_mut() = Tracker::new(dt, q, r));
}

/// Feeds one position measurement and returns the filtered position.
#[unsafe(no_mangle)]
pub extern "C" fn step(z: f32) -> f32 {
    TRACKER.with(|t| t.borrow_mut().step(z))
}

/// Returns the estimated velocity.
#[unsafe(no_mangle)]
pub extern "C" fn velocity() -> f32 {
    TRACKER.with(|t| t.borrow().filter.state_vector().get_at(1, 0))
}

/// Returns the variance of the position estimate.
#[unsafe(no_mangle)]
pub extern "C" fn position_variance() -> f32 {
    TRACKER.with(|t| t.borrow().filter.estimate_covariance().get_at(0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_constant_velocity() {
        reset(0.1, 0.01, 0.25);
        for i in 0..200 {
            step(2.0 * 0.1 * i as f32);
        }
        assert!((velocity() - 2.0).abs() < 0.05, "v = {}", velocity());
        assert!(position_variance() < 0.25);
    }
}
