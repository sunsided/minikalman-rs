#![no_std]
#![no_main]
use kalman_codesize as _;
use minikalman::extended::{FilterStorage, ObservationStorage};
use minikalman::prelude::*;
/// One radar EKF predict + correct on caller-owned storage.
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
    let _ = f.predict_nonlinear(|x, next| {
        for i in 0..2 {
            next[i] = x[i] + 0.1 * x[i + 2];
            next[i + 2] = x[i + 2];
        }
    });
    let (px, py) = (f.state_vector().get(0, 0), f.state_vector().get(1, 0));
    let r2 = px * px + py * py;
    let r = libm::sqrtf(r2);
    o.observation_jacobian_matrix_mut().apply(|h| {
        h.set(0, 0, px / r);
        h.set(0, 1, py / r);
        h.set(1, 0, -py / r2);
        h.set(1, 1, px / r2);
    });
    f.correct_nonlinear(&mut o, |x, obs| {
        let (px, py) = (x.get(0, 0), x.get(1, 0));
        obs.set(0, 0, libm::sqrtf(px * px + py * py));
        obs.set(1, 0, libm::atan2f(py, px));
    })
    .is_applied()
}
