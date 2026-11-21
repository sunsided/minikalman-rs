//! The square-root covariance filter in `I16F16`, built from static buffers without `alloc`.

#![forbid(unsafe_code)]
#![cfg(feature = "fixed")]
#![allow(non_snake_case)]

use minikalman::prelude::*;
use minikalman::regular::RegularObservationBuilder;
use minikalman::square_root::SquareRootKalman;

const NUM_STATES: usize = 2;
const NUM_OBSERVATIONS: usize = 1;

/// Tracks a target moving at one unit per second with position measurements, with process
/// noise on the velocity only, i.e. a singular `Lq` that `set_process_noise` would reject.
#[test]
fn tracks_a_constant_velocity_target() {
    let zero = I16F16::ZERO;
    impl_buffer_A!(mut A, NUM_STATES, I16F16, zero);
    impl_buffer_x!(mut x, NUM_STATES, I16F16, zero);
    impl_buffer_P!(mut S, NUM_STATES, I16F16, zero);
    impl_buffer_Q_direct!(mut Lq, NUM_STATES, I16F16, zero);
    impl_buffer_temp_prearray!(mut pre, NUM_STATES, I16F16, zero);

    impl_buffer_z!(mut z, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_H!(mut H, NUM_OBSERVATIONS, NUM_STATES, I16F16, zero);
    impl_buffer_R!(mut R, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_y!(mut y, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_S!(mut innovation_S, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_K!(mut K, NUM_STATES, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_temp_S_inv!(mut temp_S_inv, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_temp_HP!(mut temp_HP, NUM_OBSERVATIONS, NUM_STATES, I16F16, zero);
    impl_buffer_temp_PHt!(mut temp_PHt, NUM_STATES, NUM_OBSERVATIONS, I16F16, zero);
    impl_buffer_temp_KHP!(mut temp_KHP, NUM_STATES, I16F16, zero);

    let half = I16F16::from_num(0.5);
    A.as_mut_slice()
        .copy_from_slice(&[I16F16::ONE, half, zero, I16F16::ONE]);
    S.as_mut_slice()
        .copy_from_slice(&[I16F16::from_num(2), zero, zero, I16F16::ONE]);
    Lq.as_mut_slice()
        .copy_from_slice(&[zero, zero, zero, I16F16::from_num(0.05)]);
    H.as_mut_slice().copy_from_slice(&[I16F16::ONE, zero]);
    R.as_mut_slice().copy_from_slice(&[I16F16::from_num(0.25)]);

    let mut filter =
        SquareRootKalman::<NUM_STATES, 4, I16F16, _, _, _, _, _>::new(A, x, S, Lq, pre);
    let mut measurement = RegularObservationBuilder::new::<NUM_STATES, NUM_OBSERVATIONS, I16F16>(
        H,
        z,
        R,
        y,
        innovation_S,
        K,
        temp_S_inv,
        temp_HP,
        temp_PHt,
        temp_KHP,
    );

    for k in 1..=40 {
        assert!(filter.predict());
        let position = I16F16::from_num(k) * half;
        measurement.measurement_vector_mut().set(0, 0, position);
        assert!(filter.correct(&mut measurement).is_applied());
    }

    let velocity = filter.state_vector().get(1, 0).to_num::<f64>();
    assert!((velocity - 1.0).abs() < 0.05, "velocity {velocity}");

    let s = filter.covariance_factor();
    let variance = |i: usize| (0..2).map(|k| s.get(i, k) * s.get(i, k)).sum::<I16F16>();
    assert!(variance(0) > zero && variance(1) > zero);
}
