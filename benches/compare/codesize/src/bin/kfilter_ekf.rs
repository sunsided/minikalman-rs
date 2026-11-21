#![no_std]
#![no_main]
use kalman_codesize as _;
use kfilter::kalman::{Kalman, KalmanFilter, KalmanPredict, KalmanUpdate};
use kfilter::measurement::{LinearisableMeasurement, Measurement};
use kfilter::system::LinearNoInputSystem;
use na34::{SMatrix, SVector};

/// The radar measurement with a Jacobian refreshed before each update.
pub struct Radar {
    z: SVector<f32, 2>,
    h: SMatrix<f32, 2, 4>,
    ht: SMatrix<f32, 4, 2>,
    r: SMatrix<f32, 2, 2>,
}
impl Measurement<f32, 4, 2> for Radar {
    fn covariance(&self) -> &SMatrix<f32, 2, 2> {
        &self.r
    }
    fn measurement(&self) -> &SVector<f32, 2> {
        &self.z
    }
    fn set_measurement(&mut self, z: SVector<f32, 2>) {
        self.z = z;
    }
    fn predict(&self, x: &SVector<f32, 4>) -> SVector<f32, 2> {
        SVector::<f32, 2>::new(
            libm::sqrtf(x[0] * x[0] + x[1] * x[1]),
            libm::atan2f(x[1], x[0]),
        )
    }
}
impl LinearisableMeasurement<f32, 4, 2> for Radar {
    fn observation(&self) -> &SMatrix<f32, 2, 4> {
        &self.h
    }
    fn observation_transpose(&self) -> &SMatrix<f32, 4, 2> {
        &self.ht
    }
}
/// One radar EKF predict + update of kfilter with a custom measurement.
#[unsafe(no_mangle)]
pub fn step(
    k: &mut Kalman<f32, 4, 0, LinearNoInputSystem<f32, 4>>,
    m: &mut Radar,
    z: &[f32; 2],
) -> bool {
    let _ = k.predict();
    m.z = SVector::<f32, 2>::new(z[0], z[1]);
    let x = *k.state();
    let r2 = x[0] * x[0] + x[1] * x[1];
    let r = libm::sqrtf(r2);
    m.h = SMatrix::<f32, 2, 4>::new(
        x[0] / r,
        x[1] / r,
        0.0,
        0.0,
        -x[1] / r2,
        x[0] / r2,
        0.0,
        0.0,
    );
    m.ht = m.h.transpose();
    k.update(m).is_ok()
}
