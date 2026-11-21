```rust
use minikalman::prelude::*;
use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};

const NUM_STATES: usize = 2; // position, velocity
const NUM_OBSERVATIONS: usize = 1; // position

// Filter buffers: A, x̂, P, Q and two temporaries, as local arrays.
impl_buffer_A!(mut a, NUM_STATES, f32, 0.0);
impl_buffer_x!(mut x, NUM_STATES, f32, 0.0);
impl_buffer_P!(mut p, NUM_STATES, f32, 0.0);
impl_buffer_Q_direct!(mut q, NUM_STATES, f32, 0.0);
impl_buffer_temp_x!(mut temp_x, NUM_STATES, f32, 0.0);
impl_buffer_temp_P!(mut temp_p, NUM_STATES, f32, 0.0);

// Observation buffers: H, z, R, the innovation y, its covariance S, the gain K and temporaries.
impl_buffer_H!(mut h, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
impl_buffer_z!(mut z, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_R!(mut r, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_y!(mut y, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_S!(mut s, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_K!(mut k, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_temp_S_inv!(mut temp_s_inv, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_temp_HP!(mut temp_hp, NUM_OBSERVATIONS, NUM_STATES, f32, 0.0);
impl_buffer_temp_PHt!(mut temp_pht, NUM_STATES, NUM_OBSERVATIONS, f32, 0.0);
impl_buffer_temp_KHP!(mut temp_khp, NUM_STATES, f32, 0.0);

let mut filter = RegularKalmanBuilder::new::<NUM_STATES, f32>(a, x, p, q, temp_x, temp_p);
let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, f32>(
    h, z, r, y, s, k, temp_s_inv, temp_hp, temp_pht, temp_khp,
);

// Constant velocity with a time step of 0.1 s, and noise on the velocity only.
filter.state_transition_mut().apply(|a| {
    a.set(0, 0, 1.0);
    a.set(0, 1, 0.1);
    a.set(1, 1, 1.0);
});
filter.direct_process_noise_mut().set(1, 1, 0.01);
filter.estimate_covariance_mut().make_scalar(10.0);
measurement.observation_matrix_mut().set(0, 0, 1.0);
measurement.measurement_noise_covariance_mut().set(0, 0, 0.01);

for position in [0.1, 0.2, 0.32, 0.4, 0.49, 0.61, 0.7, 0.79] {
    filter.predict();
    measurement.measurement_vector_mut().set(0, 0, position);
    filter.correct(&mut measurement);
}

// The object moves at about 1 m/s.
let velocity = filter.state_vector().get(1, 0);
assert!(velocity > 0.5 && velocity < 1.5);
```
