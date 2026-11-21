#![no_std]
#![no_main]
use kalman_codesize as _;
/// Empty step: the size of the scaffolding alone.
#[unsafe(no_mangle)]
pub fn step(z: &[f32; 2], out: &mut [f32; 2]) {
    out[0] = z[0];
    out[1] = z[1];
}
