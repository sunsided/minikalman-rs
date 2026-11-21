#![no_std]
#![no_main]
use kalman_codesize as _;
use na35::SVector;
use smelly::SKalmanFilter;
/// One predict + update of smelly's fixed-size filter.
#[unsafe(no_mangle)]
pub fn step(k: &mut SKalmanFilter<f32, 4, 2>, z: &[f32; 2]) -> bool {
    let _ = k.predict();
    k.update(&SVector::<f32, 2>::new(z[0], z[1])).is_ok()
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
