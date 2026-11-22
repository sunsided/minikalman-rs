//! The const-generic storage types against the `impl_buffer_*` macros and the `alloc` builders.
//!
//! Each storage must have the size of the buffers it replaces, and a filter built from it must
//! behave bit for bit like one built from separate buffers.

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use core::mem::{size_of, size_of_val};

use minikalman::prelude::*;
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use minikalman::regular::{RegularKalmanBuilder, RegularObservationBuilder};
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use minikalman::unscented::{UnscentedKalman, UnscentedObservation};

/// Observed distances of a falling object, as in the gravity example.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const REAL_DISTANCE: [f32; 15] = [
    0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
    706.32, 828.94, 961.38,
];

/// Observation noise of the gravity example.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const OBSERVATION_ERROR: [f32; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

/// Time step of the radar examples, in seconds.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const DELTA_T: f32 = 0.1;

/// Radar position in the global frame, in meters.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const RX: f32 = 2.0;

/// Radar position in the global frame, in meters.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const RY: f32 = 1.0;

/// Number of radar sigma points.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const NUM_SIGMA: usize = 9;

/// The bit patterns of `x` and `P` after every step of a run.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
type Trace = Vec<Vec<u32>>;

/// Appends the bit patterns of a filter's `x` and `P` to a trace.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn record(trace: &mut Trace, x: &[f32], p: &[f32]) {
    trace.push(x.iter().chain(p).map(|v| v.to_bits()).collect());
}

/// A noise-free radar measurement of an object moving diagonally at 1 m/s: range, bearing and
/// range rate at `time`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn radar_measurement(time: f32) -> [f32; 3] {
    let range = ((time - RX).powi(2) + (time - RY).powi(2)).sqrt();
    let bearing = (time - RY).atan2(time - RX);
    let rate = ((time - RX) + (time - RY)) / range;
    [range, bearing, rate]
}

/// Maps a radar state `[x, y, vx, vy]` to range, bearing and range rate.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn radar_observe(x: f32, y: f32, vx: f32, vy: f32) -> [f32; 3] {
    let (dx, dy) = (x - RX, y - RY);
    let range = (dx * dx + dy * dy).sqrt();
    [range, dy.atan2(dx), (dx * vx + dy * vy) / range]
}

/// Runs the gravity example on a regular filter and its observation and returns the trace.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
macro_rules! run_gravity {
    ($filter:ident, $measurement:ident) => {{
        $filter.state_vector_mut().apply(|x| x[2] = 6.0);
        $filter.state_transition_mut().apply(|a| {
            a.make_identity();
            a.set_at(0, 1, 1.0);
            a.set_at(0, 2, 0.5);
            a.set_at(1, 2, 1.0);
        });
        $filter.estimate_covariance_mut().apply(|p| {
            p.set_at(0, 0, 0.1);
            p.set_at(1, 1, 1.0);
            p.set_at(2, 2, 1.0);
        });
        $measurement
            .observation_matrix_mut()
            .apply(|h| h.set_at(0, 0, 1.0));
        $measurement
            .measurement_noise_covariance_mut()
            .apply(|r| r.set_at(0, 0, 0.5));

        let mut trace = Trace::new();
        for (real, error) in REAL_DISTANCE.iter().zip(OBSERVATION_ERROR) {
            $filter.predict();
            $measurement
                .measurement_vector_mut()
                .apply(|z| z[0] = real + error);
            $filter.correct(&mut $measurement);
            record(
                &mut trace,
                $filter.state_vector().as_slice(),
                $filter.estimate_covariance().as_slice(),
            );
        }

        trace
    }};
}

/// Runs the radar example on an extended filter and its observation and returns the trace.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
macro_rules! run_radar_ekf {
    ($filter:ident, $measurement:ident) => {{
        $filter.state_vector_mut().apply(|x| {
            x[2] = 0.5;
            x[3] = 2.0;
        });
        $filter.estimate_covariance_mut().make_identity();
        $filter.direct_process_noise_mut().make_scalar(0.1);
        $measurement.measurement_noise_covariance_mut().apply(|r| {
            r.set_at(0, 0, 0.5);
            r.set_at(1, 1, 0.1);
            r.set_at(2, 2, 0.2);
        });

        let mut trace = Trace::new();
        for step in 1..=100 {
            $filter.state_transition_jacobian_mut().apply(|a| {
                a.make_identity();
                a.set_at(0, 2, DELTA_T);
                a.set_at(1, 3, DELTA_T);
            });
            $filter.predict_nonlinear(|x, next| {
                next[0] = x[0] + x[2] * DELTA_T;
                next[1] = x[1] + x[3] * DELTA_T;
                next[2] = x[2];
                next[3] = x[3];
            });

            if step % 10 == 0 {
                let z = radar_measurement(step as f32 * DELTA_T);
                $measurement.measurement_vector_mut().apply(|v| {
                    v[0] = z[0];
                    v[1] = z[1];
                    v[2] = z[2];
                });

                let x = $filter.state_vector().as_slice();
                let (dx, dy) = (x[0] - RX, x[1] - RY);
                let n2 = dx * dx + dy * dy;
                let n = n2.sqrt();
                $measurement.observation_jacobian_matrix_mut().apply(|h| {
                    h.set_at(0, 0, dx / n);
                    h.set_at(0, 1, dy / n);
                    h.set_at(1, 0, -dy / n2);
                    h.set_at(1, 1, dx / n2);
                    h.set_at(2, 2, dx / n);
                    h.set_at(2, 3, dy / n);
                });
                $filter.correct_nonlinear(&mut $measurement, |x, z| {
                    let o = radar_observe(x[0], x[1], x[2], x[3]);
                    z[0] = o[0];
                    z[1] = o[1];
                    z[2] = o[2];
                });
            }

            record(
                &mut trace,
                $filter.state_vector().as_slice(),
                $filter.estimate_covariance().as_slice(),
            );
        }

        trace
    }};
}

/// Runs the radar example on an unscented filter and its observation and returns the trace.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
macro_rules! run_radar_ukf {
    ($filter:ident, $measurement:ident) => {{
        $filter.state_vector_mut().apply(|x| {
            x[2] = 0.5;
            x[3] = 2.0;
        });
        $filter.estimate_covariance_mut().make_identity();
        $filter.direct_process_noise_mut().make_scalar(0.1);
        $measurement.measurement_noise_covariance_mut().apply(|r| {
            r.set_at(0, 0, 0.5);
            r.set_at(1, 1, 0.1);
            r.set_at(2, 2, 0.2);
        });

        let mut trace = Trace::new();
        for step in 1..=100 {
            $filter.predict_nonlinear(|x| {
                x[0] += x[2] * DELTA_T;
                x[1] += x[3] * DELTA_T;
            });

            if step % 10 == 0 {
                let z = radar_measurement(step as f32 * DELTA_T);
                $measurement.measurement_vector_mut().apply(|v| {
                    v[0] = z[0];
                    v[1] = z[1];
                    v[2] = z[2];
                });
                $filter.correct_sigma_point(&mut $measurement, |sigma, observed| {
                    for j in 0..NUM_SIGMA {
                        let o = radar_observe(
                            sigma[j],
                            sigma[NUM_SIGMA + j],
                            sigma[2 * NUM_SIGMA + j],
                            sigma[3 * NUM_SIGMA + j],
                        );
                        observed[j] = o[0];
                        observed[NUM_SIGMA + j] = o[1];
                        observed[2 * NUM_SIGMA + j] = o[2];
                    }
                });
            }

            record(
                &mut trace,
                $filter.state_vector().as_slice(),
                $filter.estimate_covariance().as_slice(),
            );
        }

        trace
    }};
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn regular_storage_matches_the_macros_on_gravity() {
    let mut filter_storage = minikalman::regular::FilterStorage::<3, f32>::new(0.0);
    let mut measurement_storage = minikalman::regular::ObservationStorage::<3, 1, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();
    let from_storage = run_gravity!(filter, measurement);

    impl_buffer_x!(mut x, 3, f32, 0.0);
    impl_buffer_A!(mut A, 3, f32, 0.0);
    impl_buffer_P!(mut P, 3, f32, 0.0);
    impl_buffer_Q_direct!(mut Q, 3, f32, 0.0);
    impl_buffer_temp_x!(mut temp_x, 3, f32, 0.0);
    impl_buffer_temp_P!(mut temp_P, 3, f32, 0.0);
    impl_buffer_z!(mut z, 1, f32, 0.0);
    impl_buffer_H!(mut H, 1, 3, f32, 0.0);
    impl_buffer_R!(mut R, 1, f32, 0.0);
    impl_buffer_y!(mut y, 1, f32, 0.0);
    impl_buffer_S!(mut S, 1, f32, 0.0);
    impl_buffer_K!(mut K, 3, 1, f32, 0.0);
    impl_buffer_temp_S_inv!(mut temp_S_inv, 1, f32, 0.0);
    impl_buffer_temp_HP!(mut temp_HP, 1, 3, f32, 0.0);
    impl_buffer_temp_PHt!(mut temp_PHt, 3, 1, f32, 0.0);
    impl_buffer_temp_KHP!(mut temp_KHP, 3, f32, 0.0);
    let mut filter = RegularKalmanBuilder::new::<3, f32>(A, x, P, Q, temp_x, temp_P);
    let mut measurement = RegularObservationBuilder::new::<3, 1, f32>(
        H, z, R, y, S, K, temp_S_inv, temp_HP, temp_PHt, temp_KHP,
    );
    let from_macros = run_gravity!(filter, measurement);

    assert_eq!(from_storage, from_macros);
    let g = f32::from_bits(from_storage[14][2]);
    assert!(g > 9.0 && g < 10.0, "g = {g}");
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
#[cfg(feature = "alloc")]
fn regular_storage_with_control_matches_the_builder() {
    use minikalman::regular::builder::KalmanFilterBuilder;

    macro_rules! run_with_control {
        ($filter:ident, $control:ident) => {{
            $filter.state_transition_mut().make_identity();
            $filter.estimate_covariance_mut().make_identity();
            $control.control_matrix_mut().apply(|b| b.set_at(1, 0, 1.0));
            $control
                .process_noise_covariance_mut()
                .apply(|q| q.set_at(0, 0, 0.1));
            $control.control_vector_mut().apply(|u| u[0] = 2.0);

            let mut trace = Trace::new();
            for _ in 0..5 {
                $filter.predict();
                $filter.control(&mut $control);
                record(
                    &mut trace,
                    $filter.state_vector().as_slice(),
                    $filter.estimate_covariance().as_slice(),
                );
            }

            trace
        }};
    }

    let mut filter_storage = minikalman::regular::FilterStorage::<3, f32>::new(0.0);
    let mut control_storage = minikalman::regular::ControlStorage::<3, 1, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut control = control_storage.build();
    let from_storage = run_with_control!(filter, control);

    let builder = KalmanFilterBuilder::<3, f32>::default();
    let mut filter = builder.build();
    let mut control = builder.controls().build::<1>();
    let from_builder = run_with_control!(filter, control);

    assert_eq!(from_storage, from_builder);
    assert_eq!(f32::from_bits(from_storage[4][1]), 10.0);
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
#[cfg(feature = "alloc")]
fn extended_storage_matches_the_builder_on_radar() {
    use minikalman::extended::builder::KalmanFilterBuilder;

    let mut filter_storage = minikalman::extended::FilterStorage::<4, f32>::new(0.0);
    let mut measurement_storage = minikalman::extended::ObservationStorage::<4, 3, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();
    let from_storage = run_radar_ekf!(filter, measurement);

    let builder = KalmanFilterBuilder::<4, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<3>();
    let from_builder = run_radar_ekf!(filter, measurement);

    assert_eq!(from_storage, from_builder);
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn unscented_storage_matches_the_macros_on_radar() {
    let mut filter_storage = minikalman::unscented::FilterStorage::<4, NUM_SIGMA, f32>::new(0.0);
    let mut measurement_storage =
        minikalman::unscented::ObservationStorage::<4, 3, NUM_SIGMA, f32>::new(0.0);
    let mut filter = filter_storage.build(1.0, 2.0, 1.0);
    let mut measurement = measurement_storage.build();
    let from_storage = run_radar_ukf!(filter, measurement);

    impl_buffer_x!(mut x, 4, f32, 0.0);
    impl_buffer_P!(mut P, 4, f32, 0.0);
    impl_buffer_Q_direct!(mut Q, 4, f32, 0.0);
    impl_buffer_temp_x!(mut predicted_x, 4, f32, 0.0);
    impl_buffer_sigma_points!(mut sigma_points, 4, NUM_SIGMA, f32, 0.0);
    impl_buffer_sigma_weights!(mut sigma_weights, NUM_SIGMA, f32, 0.0);
    impl_buffer_sigma_propagated!(mut sigma_propagated, 4, NUM_SIGMA, f32, 0.0);
    impl_buffer_temp_sigma_P!(mut temp_sigma_P, 4, f32, 0.0);
    impl_buffer_z!(mut z, 3, f32, 0.0);
    impl_buffer_R!(mut R, 3, f32, 0.0);
    impl_buffer_y!(mut y, 3, f32, 0.0);
    impl_buffer_S!(mut S, 3, f32, 0.0);
    impl_buffer_K!(mut K, 4, 3, f32, 0.0);
    impl_buffer_temp_S_inv!(mut temp_S_inv, 3, f32, 0.0);
    impl_buffer_sigma_observed!(mut sigma_observed, 3, NUM_SIGMA, f32, 0.0);
    impl_buffer_cross_covariance!(mut cross_covariance, 4, 3, f32, 0.0);
    impl_buffer_temp_sigma_P!(mut temp_P, 4, f32, 0.0);
    let mut filter = UnscentedKalman::<4, NUM_SIGMA, f32, _, _, _, _, _, _, _, _>::new(
        x,
        P,
        Q,
        predicted_x,
        sigma_points,
        sigma_weights,
        sigma_propagated,
        temp_sigma_P,
        1.0,
        2.0,
        1.0,
    );
    let mut measurement =
        UnscentedObservation::<4, 3, NUM_SIGMA, f32, _, _, _, _, _, _, _, _, _>::new(
            z,
            R,
            y,
            S,
            K,
            temp_S_inv,
            sigma_observed,
            cross_covariance,
            temp_P,
        );
    let from_macros = run_radar_ukf!(filter, measurement);

    assert_eq!(from_storage, from_macros);
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
#[cfg(feature = "alloc")]
fn information_storage_matches_the_builder() {
    use minikalman::regular::builder::KalmanFilterBuilder;

    macro_rules! run_information {
        ($filter:ident, $measurement:ident) => {{
            $filter.state_transition_mut().apply(|a| {
                a.make_identity();
                a.set_at(0, 1, 0.5);
            });
            $filter.process_noise_mut().make_scalar(0.1);
            $measurement.observation_matrix_mut().make_identity();
            $measurement
                .measurement_noise_covariance_mut()
                .make_scalar(0.5);

            let mut trace = Trace::new();
            for z in [0.1, 0.6, 1.2, 1.9, 2.4] {
                $measurement.measurement_vector_mut().apply(|v| {
                    v[0] = z;
                    v[1] = 1.0;
                });
                assert!($filter.fuse(&mut $measurement));
                if z > 0.5 {
                    assert!($filter.predict());
                }

                record(
                    &mut trace,
                    $filter.information_vector().as_slice(),
                    $filter.information_matrix().as_slice(),
                );
            }

            trace
        }};
    }

    let mut filter_storage = minikalman::information::FilterStorage::<2, f32>::new(0.0);
    let mut measurement_storage = minikalman::regular::ObservationStorage::<2, 2, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();
    let from_storage = run_information!(filter, measurement);

    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut filter = builder.build_information();
    let mut measurement = builder.observations().build::<2>();
    let from_builder = run_information!(filter, measurement);

    assert_eq!(from_storage, from_builder);
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn smoother_storage_matches_separate_buffers() {
    use minikalman::smoother::{RtsSmoother, RtsSmootherStorage};

    /// Smooths a fixed two-step history and returns the smoothed `x` and `P` bits.
    fn smooth<Gain, Inv, Temp, TempX>(
        smoother: &mut RtsSmoother<2, f32, Gain, Inv, Temp, TempX>,
    ) -> Vec<u32>
    where
        Gain: MatrixMut<2, 2, f32>,
        Inv: MatrixMut<2, 2, f32>,
        Temp: MatrixMut<2, 2, f32>,
        TempX: MatrixMut<2, 1, f32>,
    {
        type Vector = MatrixDataArray<2, 1, 2, f32>;
        type Square = MatrixDataArray<2, 2, 4, f32>;

        let a = Square::new([1.0, 1.0, 0.0, 1.0]);
        let mut x = [Vector::new([0.0, 1.0]), Vector::new([1.1, 0.9])];
        let mut p = [
            Square::new([1.0, 0.0, 0.0, 1.0]),
            Square::new([0.5, 0.1, 0.1, 0.5]),
        ];
        let x_predicted = [Vector::new([0.0, 1.0]), Vector::new([1.0, 1.0])];
        let p_predicted = [
            Square::new([1.0, 0.0, 0.0, 1.0]),
            Square::new([2.1, 1.0, 1.0, 1.1]),
        ];
        assert!(smoother.smooth(&a, &mut x, &mut p, &x_predicted, &p_predicted));

        let x = x.iter().flat_map(|v| v.as_slice().to_vec());
        let p = p.iter().flat_map(|m| m.as_slice().to_vec());
        x.chain(p).map(f32::to_bits).collect()
    }

    let mut storage = RtsSmootherStorage::<2, f32>::new(0.0);
    let from_storage = smooth(&mut storage.build());

    let mut separate = RtsSmoother::new(
        MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]),
        MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]),
        MatrixData::new_array::<2, 2, 4, f32>([0.0; 4]),
        MatrixData::new_array::<2, 1, 2, f32>([0.0; 2]),
    );
    let from_separate = smooth(&mut separate);

    assert_eq!(from_storage, from_separate);
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn storage_sizes_equal_the_macro_buffers() {
    impl_buffer_x!(x, 4, f32, 0.0);
    impl_buffer_A!(A, 4, f32, 0.0);
    impl_buffer_P!(P, 4, f32, 0.0);
    impl_buffer_Q_direct!(Q, 4, f32, 0.0);
    impl_buffer_temp_x!(temp_x, 4, f32, 0.0);
    impl_buffer_temp_P!(temp_P, 4, f32, 0.0);
    let filter = size_of_val(&x)
        + size_of_val(&A)
        + size_of_val(&P)
        + size_of_val(&Q)
        + size_of_val(&temp_x)
        + size_of_val(&temp_P);
    assert_eq!(filter, (4 + 16 * 4 + 4) * size_of::<f32>());
    assert_eq!(
        size_of::<minikalman::regular::FilterStorage<4, f32>>(),
        filter
    );
    assert_eq!(
        size_of::<minikalman::extended::FilterStorage<4, f32>>(),
        filter
    );
    impl_buffer_temp_P!(temp_L, 4, f32, 0.0);
    assert_eq!(
        size_of::<minikalman::information::FilterStorage<4, f32>>(),
        filter + size_of_val(&temp_L)
    );

    impl_buffer_z!(z, 3, f32, 0.0);
    impl_buffer_H!(H, 3, 4, f32, 0.0);
    impl_buffer_R!(R, 3, f32, 0.0);
    impl_buffer_y!(y, 3, f32, 0.0);
    impl_buffer_S!(S, 3, f32, 0.0);
    impl_buffer_K!(K, 4, 3, f32, 0.0);
    impl_buffer_temp_S_inv!(temp_S_inv, 3, f32, 0.0);
    impl_buffer_temp_HP!(temp_HP, 3, 4, f32, 0.0);
    impl_buffer_temp_PHt!(temp_PHt, 4, 3, f32, 0.0);
    impl_buffer_temp_KHP!(temp_KHP, 4, f32, 0.0);
    let observation = size_of_val(&z)
        + size_of_val(&H)
        + size_of_val(&R)
        + size_of_val(&y)
        + size_of_val(&S)
        + size_of_val(&K)
        + size_of_val(&temp_S_inv)
        + size_of_val(&temp_HP)
        + size_of_val(&temp_PHt)
        + size_of_val(&temp_KHP);
    assert_eq!(observation, 97 * size_of::<f32>());
    assert_eq!(
        size_of::<minikalman::regular::ObservationStorage<4, 3, f32>>(),
        observation
    );
    assert_eq!(
        size_of::<minikalman::extended::ObservationStorage<4, 3, f32>>(),
        observation
    );

    impl_buffer_u!(u, 2, f32, 0.0);
    impl_buffer_B!(B, 4, 2, f32, 0.0);
    impl_buffer_Q_control!(Q_control, 2, f32, 0.0);
    impl_buffer_temp_BQ!(temp_BQ, 4, 2, f32, 0.0);
    assert_eq!(
        size_of::<minikalman::regular::ControlStorage<4, 2, f32>>(),
        size_of_val(&u) + size_of_val(&B) + size_of_val(&Q_control) + size_of_val(&temp_BQ)
    );

    impl_buffer_sigma_points!(sigma_points, 4, NUM_SIGMA, f32, 0.0);
    impl_buffer_sigma_weights!(sigma_weights, NUM_SIGMA, f32, 0.0);
    impl_buffer_sigma_propagated!(sigma_propagated, 4, NUM_SIGMA, f32, 0.0);
    impl_buffer_temp_sigma_P!(temp_sigma_P, 4, f32, 0.0);
    assert_eq!(
        size_of::<minikalman::unscented::FilterStorage<4, NUM_SIGMA, f32>>(),
        size_of_val(&x)
            + size_of_val(&P)
            + size_of_val(&Q)
            + size_of_val(&temp_x)
            + size_of_val(&sigma_points)
            + size_of_val(&sigma_weights)
            + size_of_val(&sigma_propagated)
            + size_of_val(&temp_sigma_P)
    );

    impl_buffer_sigma_observed!(sigma_observed, 3, NUM_SIGMA, f32, 0.0);
    impl_buffer_cross_covariance!(cross_covariance, 4, 3, f32, 0.0);
    assert_eq!(
        size_of::<minikalman::unscented::ObservationStorage<4, 3, NUM_SIGMA, f32>>(),
        size_of_val(&z)
            + size_of_val(&R)
            + size_of_val(&y)
            + size_of_val(&S)
            + size_of_val(&K)
            + size_of_val(&temp_S_inv)
            + size_of_val(&sigma_observed)
            + size_of_val(&cross_covariance)
            + size_of_val(&temp_sigma_P)
    );

    assert_eq!(
        size_of::<minikalman::smoother::RtsSmootherStorage<4, f32>>(),
        3 * size_of_val(&temp_P) + size_of_val(&temp_x)
    );
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn storage_can_live_in_a_static() {
    /// A filter storage placed in `.bss` by a const initializer.
    static FILTER: minikalman::regular::FilterStorage<3, f32> =
        minikalman::regular::FilterStorage::new(0.0);
    /// An unscented storage placed in `.bss` by a const initializer.
    static UNSCENTED: minikalman::unscented::FilterStorage<3, 7, f32> =
        minikalman::unscented::FilterStorage::new(0.0);

    assert_eq!(size_of_val(&FILTER), (3 + 9 * 4 + 3) * size_of::<f32>());
    assert!(size_of_val(&UNSCENTED) > 0);
}

#[test]
#[cfg(feature = "fixed")]
fn fixed_point_storage_runs() {
    use fixed::types::I16F16;

    let mut filter_storage = minikalman::regular::FilterStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut measurement_storage =
        minikalman::regular::ObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();

    filter.state_transition_mut().make_identity();
    filter.estimate_covariance_mut().make_identity();
    measurement
        .observation_matrix_mut()
        .apply(|h| h.set_at(0, 0, I16F16::ONE));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, I16F16::from_num(0.5)));

    for _ in 0..5 {
        filter.predict();
        measurement
            .measurement_vector_mut()
            .apply(|z| z[0] = I16F16::from_num(2));
        filter.correct(&mut measurement);
    }

    let x = filter.state_vector().get_row(0);
    assert!(
        x > I16F16::from_num(1.5) && x < I16F16::from_num(2.5),
        "x = {x}"
    );
}
