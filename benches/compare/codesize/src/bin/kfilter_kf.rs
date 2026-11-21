#![no_std]
#![no_main]
use kalman_codesize as _;
use kfilter::kalman::{Kalman1MLinearNoInput, KalmanPredict};
use na34::SVector;
/// One predict + update of kfilter's `Kalman1M`.
#[unsafe(no_mangle)]
pub fn step(k: &mut Kalman1MLinearNoInput<f32, 4, 2>, z: &[f32; 2]) -> bool {
    let _ = k.predict();
    k.update(SVector::<f32, 2>::new(z[0], z[1])).is_ok()
}
