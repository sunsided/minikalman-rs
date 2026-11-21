#![no_std]
#![no_main]
use kalman_codesize as _;
use minikalman::prelude::*;
use minikalman::regular::{FilterStorage, ObservationStorage};
/// One predict + correct on caller-owned storage (Joseph form, the default).
#[unsafe(no_mangle)]
pub fn step(
    fs: &mut FilterStorage<4, f32>,
    os: &mut ObservationStorage<4, 2, f32>,
    z: &[f32; 2],
) -> bool {
    let mut f = fs.build();
    let mut o = os.build();
    o.measurement_vector_mut().set(0, 0, z[0]);
    o.measurement_vector_mut().set(1, 0, z[1]);
    let _ = f.predict();
    f.correct(&mut o).is_applied()
}
