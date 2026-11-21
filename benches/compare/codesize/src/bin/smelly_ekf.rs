#![no_std]
#![no_main]
use kalman_codesize as _;
use na35::{SMatrix, SVector};
use smelly::SExtendedKalmanFilter;
/// One radar EKF predict + update of smelly's fixed-size extended filter.
#[unsafe(no_mangle)]
pub fn step(k: &mut SExtendedKalmanFilter<f32, 4, 2>, z: &[f32; 2]) -> bool {
    let _ = k.predict();
    k.update(
        &SVector::<f32, 2>::new(z[0], z[1]),
        |x| {
            SVector::<f32, 2>::new(
                libm::sqrtf(x[0] * x[0] + x[1] * x[1]),
                libm::atan2f(x[1], x[0]),
            )
        },
        |x| {
            let r2 = x[0] * x[0] + x[1] * x[1];
            let r = libm::sqrtf(r2);
            SMatrix::<f32, 2, 4>::new(
                x[0] / r,
                x[1] / r,
                0.0,
                0.0,
                -x[1] / r2,
                x[0] / r2,
                0.0,
                0.0,
            )
        },
    )
    .is_ok()
}

/// The crate links `alloc`, so the binary provides a global allocator. This stub never hands
/// out memory.
struct NoAlloc;
unsafe impl core::alloc::GlobalAlloc for NoAlloc {
    unsafe fn alloc(&self, _: core::alloc::Layout) -> *mut u8 {
        core::ptr::null_mut()
    }
    unsafe fn dealloc(&self, _: *mut u8, _: core::alloc::Layout) {}
}
#[global_allocator]
static A: NoAlloc = NoAlloc;
