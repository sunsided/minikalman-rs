//! Tracking a 2D target with a range/bearing radar at the origin.
//!
//! Both an Extended and an Unscented Kalman filter run on the same inputs so the
//! page can compare them. The state is `[x, y, vx, vy]` under a constant-velocity
//! model; each measurement is `[range, bearing]`. Noise in range and bearing
//! makes a crescent-shaped region in x/y, which both filters approximate with a
//! Gaussian ellipse.
//!
//! Bearings are not wrapped, so the page keeps the target in the upper half-plane,
//! away from the ±π discontinuity of `atan2`.

use std::cell::RefCell;

use minikalman::extended::builder::{
    KalmanFilterBuilder as EkfBuilder, KalmanFilterObservationType as EkfObservation,
    KalmanFilterType as Ekf,
};
use minikalman::prelude::*;
use minikalman::unscented::builder::{
    KalmanFilterBuilder as UkfBuilder, KalmanFilterObservationType as UkfObservation,
    KalmanFilterType as Ukf,
};

const NUM_STATES: usize = 4; // x, y, vx, vy
const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
const NUM_OBSERVATIONS: usize = 2; // range, bearing

/// Selects the Extended Kalman filter in [`radar_state`] and [`radar_covariance`].
pub const EKF: u32 = 0;

/// Selects the Unscented Kalman filter in [`radar_state`] and [`radar_covariance`].
pub const UKF: u32 = 1;

struct Radar {
    ekf: Ekf<NUM_STATES, f32>,
    ekf_measurement: EkfObservation<NUM_STATES, NUM_OBSERVATIONS, f32>,
    ukf: Ukf<NUM_STATES, NUM_SIGMA, f32>,
    ukf_measurement: UkfObservation<NUM_STATES, NUM_OBSERVATIONS, NUM_SIGMA, f32>,
    q: f32,
}

impl Radar {
    fn new(x: f32, y: f32, q: f32, sigma_range: f32, sigma_bearing: f32) -> Self {
        let ekf_builder = EkfBuilder::<NUM_STATES, f32>::default();
        let ukf_builder = UkfBuilder::<NUM_STATES, f32>::default();
        let mut radar = Self {
            ekf: ekf_builder.build(),
            ekf_measurement: ekf_builder.observations().build::<NUM_OBSERVATIONS>(),
            ukf: ukf_builder.build::<NUM_SIGMA>(),
            ukf_measurement: ukf_builder
                .observations()
                .build::<NUM_OBSERVATIONS, NUM_SIGMA>(),
            q,
        };

        // Known initial position, unknown velocity.
        initial_state(radar.ekf.state_vector_mut(), x, y);
        initial_state(radar.ukf.state_vector_mut(), x, y);
        initial_covariance(radar.ekf.estimate_covariance_mut());
        initial_covariance(radar.ukf.estimate_covariance_mut());

        let noise = (sigma_range * sigma_range, sigma_bearing * sigma_bearing);
        diagonal(
            radar.ekf_measurement.measurement_noise_covariance_mut(),
            noise,
        );
        diagonal(
            radar.ukf_measurement.measurement_noise_covariance_mut(),
            noise,
        );

        radar
    }

    fn predict(&mut self, dt: f32) {
        process_noise(self.ekf.direct_process_noise_mut(), self.q, dt);
        process_noise(self.ukf.direct_process_noise_mut(), self.q, dt);

        self.ekf.state_transition_jacobian_mut().apply(|mat| {
            mat.make_identity();
            mat.set_at(0, 2, dt);
            mat.set_at(1, 3, dt);
        });
        self.ekf.predict_nonlinear(|state, next| {
            next[0] = state[0] + state[2] * dt;
            next[1] = state[1] + state[3] * dt;
            next[2] = state[2];
            next[3] = state[3];
        });

        self.ukf.predict_nonlinear(|next| {
            let x = next.as_matrix().get_at(0, 0);
            let y = next.as_matrix().get_at(1, 0);
            let vx = next.as_matrix().get_at(2, 0);
            let vy = next.as_matrix().get_at(3, 0);
            next.as_matrix_mut().set_at(0, 0, x + vx * dt);
            next.as_matrix_mut().set_at(1, 0, y + vy * dt);
        });
    }

    fn correct(&mut self, range: f32, bearing: f32) {
        measure(
            self.ekf_measurement.measurement_vector_mut(),
            range,
            bearing,
        );
        measure(
            self.ukf_measurement.measurement_vector_mut(),
            range,
            bearing,
        );

        // Linearize h(x) = [√(x² + y²), atan2(y, x)] at the prior.
        let x = self.ekf.state_vector().get_at(0, 0);
        let y = self.ekf.state_vector().get_at(1, 0);
        let norm_sq = x * x + y * y;
        let norm = norm_sq.sqrt();
        self.ekf_measurement
            .observation_jacobian_matrix_mut()
            .apply(|mat| {
                mat.set_zero();
                mat.set_at(0, 0, x / norm);
                mat.set_at(0, 1, y / norm);
                mat.set_at(1, 0, -y / norm_sq);
                mat.set_at(1, 1, x / norm_sq);
            });
        self.ekf
            .correct_nonlinear(&mut self.ekf_measurement, |state, observation| {
                let (x, y) = (state[0], state[1]);
                observation[0] = (x * x + y * y).sqrt();
                observation[1] = y.atan2(x);
            });

        self.ukf
            .correct_sigma_point(&mut self.ukf_measurement, |sigma_pred, observed| {
                for j in 0..NUM_SIGMA {
                    let x = sigma_pred[j];
                    let y = sigma_pred[NUM_SIGMA + j];
                    observed[j] = (x * x + y * y).sqrt();
                    observed[NUM_SIGMA + j] = y.atan2(x);
                }
            });
    }

    fn state(&self, variant: u32, i: usize) -> f32 {
        match variant {
            EKF => self.ekf.state_vector().get_at(i, 0),
            _ => self.ukf.state_vector().get_at(i, 0),
        }
    }

    fn covariance(&self, variant: u32, i: usize, j: usize) -> f32 {
        match variant {
            EKF => self.ekf.estimate_covariance().get_at(i, j),
            _ => self.ukf.estimate_covariance().get_at(i, j),
        }
    }
}

fn initial_state(vec: &mut impl MatrixMut<NUM_STATES, 1, f32>, x: f32, y: f32) {
    vec.set_at(0, 0, x);
    vec.set_at(1, 0, y);
    vec.set_at(2, 0, 0.0);
    vec.set_at(3, 0, 0.0);
}

fn initial_covariance(mat: &mut impl MatrixMut<NUM_STATES, NUM_STATES, f32>) {
    mat.make_scalar(1.0);
    mat.set_at(2, 2, 25.0);
    mat.set_at(3, 3, 25.0);
}

/// Discrete white noise acceleration with density `q`, applied to each axis.
fn process_noise(mat: &mut impl MatrixMut<NUM_STATES, NUM_STATES, f32>, q: f32, dt: f32) {
    let dt2 = dt * dt;
    mat.set_zero();
    for (p, v) in [(0, 2), (1, 3)] {
        mat.set_at(p, p, q * dt2 * dt2 / 4.0);
        mat.set_at(p, v, q * dt2 * dt / 2.0);
        mat.set_at(v, p, q * dt2 * dt / 2.0);
        mat.set_at(v, v, q * dt2);
    }
}

fn diagonal(mat: &mut impl MatrixMut<NUM_OBSERVATIONS, NUM_OBSERVATIONS, f32>, (a, b): (f32, f32)) {
    mat.set_zero();
    mat.set_at(0, 0, a);
    mat.set_at(1, 1, b);
}

fn measure(vec: &mut impl MatrixMut<NUM_OBSERVATIONS, 1, f32>, range: f32, bearing: f32) {
    vec.set_at(0, 0, range);
    vec.set_at(1, 0, bearing);
}

thread_local! {
    static RADAR: RefCell<Radar> = RefCell::new(Radar::new(0.0, 10.0, 1.0, 1.0, 0.05));
}

/// Restarts both filters at position `(x, y)` with process noise density `q`
/// and measurement standard deviations for range and bearing (radians).
#[unsafe(no_mangle)]
pub extern "C" fn radar_reset(x: f32, y: f32, q: f32, sigma_range: f32, sigma_bearing: f32) {
    RADAR.with(|r| *r.borrow_mut() = Radar::new(x, y, q, sigma_range, sigma_bearing));
}

/// Advances both filters by `dt` seconds.
#[unsafe(no_mangle)]
pub extern "C" fn radar_predict(dt: f32) {
    RADAR.with(|r| r.borrow_mut().predict(dt));
}

/// Corrects both filters with a range and bearing (radians) measurement.
#[unsafe(no_mangle)]
pub extern "C" fn radar_correct(range: f32, bearing: f32) {
    RADAR.with(|r| r.borrow_mut().correct(range, bearing));
}

/// Returns entry `i` of `[x, y, vx, vy]` for the [`EKF`] or [`UKF`] variant.
#[unsafe(no_mangle)]
pub extern "C" fn radar_state(variant: u32, i: u32) -> f32 {
    RADAR.with(|r| r.borrow().state(variant, i as usize))
}

/// Returns entry `(i, j)` of the estimate covariance for the [`EKF`] or [`UKF`] variant.
#[unsafe(no_mangle)]
pub extern "C" fn radar_covariance(variant: u32, i: u32, j: u32) -> f32 {
    RADAR.with(|r| r.borrow().covariance(variant, i as usize, j as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_variants_track_a_passing_target() {
        let (vx, vy) = (4.0, 0.5);
        let dt = 0.1;
        radar_reset(-40.0, 15.0, 0.5, 0.5, 0.02);
        for step in 1..=200 {
            radar_predict(dt);
            if step % 5 == 0 {
                let t = step as f32 * dt;
                let (x, y) = (-40.0 + vx * t, 15.0 + vy * t);
                radar_correct((x * x + y * y).sqrt(), y.atan2(x));
            }
        }

        for variant in [EKF, UKF] {
            assert!((radar_state(variant, 0) - 40.0).abs() < 1.0, "x");
            assert!((radar_state(variant, 1) - 25.0).abs() < 1.0, "y");
            assert!((radar_state(variant, 2) - vx).abs() < 0.3, "vx");
            assert!((radar_state(variant, 3) - vy).abs() < 0.3, "vy");
            assert!(radar_covariance(variant, 0, 0) < 1.0);
        }
    }
}
