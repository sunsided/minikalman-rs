//! The per-call scratch (`CorrectionScratch` and the lean observations) against the owning
//! observations.
//!
//! A lean observation that borrows its temporaries per call must behave bit for bit like one
//! that owns them, whatever the scratch held before the call, and must need only its
//! persistent buffers.

#![forbid(unsafe_code)]
#![allow(non_snake_case)]

use core::mem::size_of;

use minikalman::prelude::*;

/// Time step of the radar example, in seconds.
const DELTA_T: f32 = 0.1;

/// Radar position in the global frame, in meters.
const RX: f32 = 2.0;

/// Radar position in the global frame, in meters.
const RY: f32 = 1.0;

/// The bit patterns of `x`, `P`, `K`, `y`, the NIS and the status after every correction.
type Trace = Vec<Vec<u32>>;

/// Appends the bit patterns of one correction's results to a trace.
fn record(trace: &mut Trace, parts: &[&[f32]], nis: f32, status: CorrectionStatus) {
    let mut row: Vec<u32> = parts
        .iter()
        .flat_map(|p| p.iter())
        .map(|v| v.to_bits())
        .collect();
    row.push(nis.to_bits());
    row.push(status.is_applied().into());
    trace.push(row);
}

/// A noise-free radar measurement of an object moving diagonally at 1 m/s: range, bearing and
/// range rate at `time`, plus a large outlier on every fifth measurement for the gate.
fn radar_measurement(step: usize) -> [f32; 3] {
    let time = step as f32 * DELTA_T;
    let range = ((time - RX).powi(2) + (time - RY).powi(2)).sqrt();
    let bearing = (time - RY).atan2(time - RX);
    let rate = ((time - RX) + (time - RY)) / range;
    let outlier = if step % 50 == 0 { 40.0 } else { 0.0 };
    [range + outlier, bearing, rate]
}

/// Maps a radar state `[x, y, vx, vy]` to range, bearing and range rate.
fn radar_observe(x: f32, y: f32, vx: f32, vy: f32) -> [f32; 3] {
    let (dx, dy) = (x - RX, y - RY);
    let range = (dx * dx + dy * dy).sqrt();
    [range, dy.atan2(dx), (dx * vx + dy * vy) / range]
}

/// Runs the radar EKF of `examples/radar_2d.rs` with the given observation settings and
/// correction, and returns the trace of every correction.
macro_rules! run_radar {
    ($filter:ident, $measurement:ident, $update:expr, $adapt:expr,
     |$f:ident, $m:ident| $correct:expr) => {{
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
        $measurement.set_covariance_update($update);
        $measurement.set_gate(Some(30.0));
        $measurement.set_noise_adaptation($adapt);

        let mut trace = Trace::new();
        for step in 1..=200 {
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
            if step % 5 != 0 {
                continue;
            }

            let z = radar_measurement(step);
            $measurement
                .measurement_vector_mut()
                .apply(|v| v.as_mut_slice().copy_from_slice(&z));
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

            let status = {
                let $f = &mut $filter;
                let $m = &mut $measurement;
                $correct
            };
            record(
                &mut trace,
                &[
                    $filter.state_vector().as_slice(),
                    $filter.estimate_covariance().as_slice(),
                    $measurement.kalman_gain().as_slice(),
                    $measurement.innovation_vector().as_slice(),
                    $measurement.measurement_noise_covariance().as_slice(),
                ],
                $measurement.nis(),
                status,
            );
        }

        trace
    }};
}

/// The radar observation function, writing range, bearing and range rate into `z`.
fn observe(x: &[f32], z: &mut [f32]) {
    z.copy_from_slice(&radar_observe(x[0], x[1], x[2], x[3]));
}

/// Wraps the bearing of a radar innovation into `[-π, π]`.
fn wrap(y: &mut [f32]) {
    y[1] = minikalman::angles::wrap_pi(y[1]);
}

/// Runs the radar EKF with an owning observation and with a lean one on a fresh NaN-filled
/// scratch per call, and asserts both traces are bit-identical.
fn assert_radar_matches(update: CovarianceUpdate, adapt: Option<NoiseAdaptation<f32>>) {
    let mut filter_storage = minikalman::extended::FilterStorage::<4, f32>::new(0.0);
    let mut owning_storage = minikalman::extended::ObservationStorage::<4, 3, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut owning = owning_storage.build();
    let expected = run_radar!(filter, owning, update, adapt, |f, m| f
        .correct_nonlinear_with(m, |x, z| observe(x.as_slice(), z.as_mut_slice()), wrap));

    let mut filter_storage = minikalman::extended::FilterStorage::<4, f32>::new(0.0);
    let mut lean_storage = minikalman::extended::LeanObservationStorage::<4, 3, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut lean = lean_storage.build();
    let actual = run_radar!(filter, lean, update, adapt, |f, m| {
        // A larger scratch full of NaN: any read before a write would poison the result.
        let mut scratch = minikalman::extended::CorrectionScratch::<4, 4, f32>::new(f32::NAN);
        f.correct_nonlinear_with_in(
            m,
            &mut scratch,
            |x, z| observe(x.as_slice(), z.as_mut_slice()),
            wrap,
        )
    });

    assert_eq!(actual, expected);
    let gated = expected.iter().filter(|row| row.last() == Some(&0)).count();
    assert!(gated > 0, "the outliers must exercise the gate");
}

#[test]
fn lean_radar_ekf_matches_the_owning_observation_joseph() {
    assert_radar_matches(CovarianceUpdate::Joseph, None);
}

#[test]
fn lean_radar_ekf_matches_the_owning_observation_standard() {
    assert_radar_matches(CovarianceUpdate::Standard, None);
}

#[test]
fn lean_radar_ekf_matches_the_owning_observation_with_noise_adaptation() {
    assert_radar_matches(
        CovarianceUpdate::Joseph,
        Some(NoiseAdaptation {
            forgetting: 0.95,
            floor: 0.01,
        }),
    );
}

#[test]
fn owning_observation_also_corrects_in_a_scratch() {
    let mut filter_storage = minikalman::extended::FilterStorage::<4, f32>::new(0.0);
    let mut owning_storage = minikalman::extended::ObservationStorage::<4, 3, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut owning = owning_storage.build();
    let expected = run_radar!(filter, owning, CovarianceUpdate::Joseph, None, |f, m| f
        .correct_nonlinear(m, |x, z| observe(x.as_slice(), z.as_mut_slice())));

    let mut filter_storage = minikalman::extended::FilterStorage::<4, f32>::new(0.0);
    let mut owning_storage = minikalman::extended::ObservationStorage::<4, 3, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut owning = owning_storage.build();
    let mut scratch = minikalman::extended::CorrectionScratch::<4, 3, f32>::new(0.0);
    let actual = run_radar!(filter, owning, CovarianceUpdate::Joseph, None, |f, m| f
        .correct_nonlinear_in(m, &mut scratch, |x, z| observe(
            x.as_slice(),
            z.as_mut_slice()
        )));

    assert_eq!(actual, expected);
}

/// Runs a 1-D constant-velocity tracker corrected by a position-and-velocity sensor and a
/// position-only sensor in turn, and returns the trace of every correction.
macro_rules! run_two_sensors {
    ($filter:ident, $both:ident, $position:ident, |$f:ident, $m:ident| $correct:expr) => {{
        $filter.state_transition_mut().apply(|a| {
            a.make_identity();
            a.set_at(0, 1, 1.0);
        });
        $filter.estimate_covariance_mut().make_scalar(10.0);
        $filter.direct_process_noise_mut().make_scalar(0.01);
        $both.observation_matrix_mut().make_identity();
        $both.measurement_noise_covariance_mut().apply(|r| {
            r.set_at(0, 0, 1.0);
            r.set_at(1, 1, 0.5);
        });
        $position
            .observation_matrix_mut()
            .apply(|h| h.set_at(0, 0, 1.0));
        $position
            .measurement_noise_covariance_mut()
            .apply(|r| r.set_at(0, 0, 0.25));

        let mut trace = Trace::new();
        for step in 0..40 {
            let t = step as f32;
            $filter.predict();
            $both
                .measurement_vector_mut()
                .apply(|z| z.as_mut_slice().copy_from_slice(&[2.0 * t + 0.3, 2.1]));
            let status = {
                let $f = &mut $filter;
                let $m = &mut $both;
                $correct
            };
            record(
                &mut trace,
                &[
                    $filter.state_vector().as_slice(),
                    $filter.estimate_covariance().as_slice(),
                    $both.kalman_gain().as_slice(),
                ],
                $both.nis(),
                status,
            );

            $position
                .measurement_vector_mut()
                .apply(|z| z[0] = 2.0 * t - 0.2);
            let status = {
                let $f = &mut $filter;
                let $m = &mut $position;
                $correct
            };
            record(
                &mut trace,
                &[
                    $filter.state_vector().as_slice(),
                    $filter.estimate_covariance().as_slice(),
                    $position.kalman_gain().as_slice(),
                ],
                $position.nis(),
                status,
            );
        }

        trace
    }};
}

#[test]
fn one_scratch_serves_regular_observations_of_different_sizes() {
    use minikalman::regular::{
        CorrectionScratch, FilterStorage, LeanObservationStorage, ObservationStorage,
    };

    let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
    let mut both_storage = ObservationStorage::<2, 2, f32>::new(0.0);
    let mut position_storage = ObservationStorage::<2, 1, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut both = both_storage.build();
    let mut position = position_storage.build();
    let expected = run_two_sensors!(filter, both, position, |f, m| f.correct(m));

    let mut filter_storage = FilterStorage::<2, f32>::new(0.0);
    let mut both_storage = LeanObservationStorage::<2, 2, f32>::new(0.0);
    let mut position_storage = LeanObservationStorage::<2, 1, f32>::new(0.0);
    let mut scratch = CorrectionScratch::<2, 2, f32>::new(f32::NAN);
    let mut filter = filter_storage.build();
    let mut both = both_storage.build();
    let mut position = position_storage.build();
    let actual = run_two_sensors!(filter, both, position, |f, m| f.correct_in(m, &mut scratch));

    assert_eq!(actual, expected);
}

#[test]
fn regular_observation_corrects_with_a_fix_in_a_scratch() {
    use minikalman::regular::{CorrectionScratch, FilterStorage, LeanObservationStorage};

    let mut filter_storage = FilterStorage::<1, f32>::new(0.0);
    let mut lean_storage = LeanObservationStorage::<1, 1, f32>::new(0.0);
    let mut scratch = CorrectionScratch::<1, 1, f32>::new(f32::NAN);
    let mut filter = filter_storage.build();
    let mut compass = lean_storage.build();
    assert_eq!(scratch.max_observations(), 1);

    filter.state_transition_mut().make_identity();
    filter.estimate_covariance_mut().make_identity();
    filter.state_vector_mut().set_at(0, 0, 359.0);
    compass.observation_matrix_mut().make_identity();
    compass.measurement_noise_covariance_mut().make_identity();
    compass.measurement_vector_mut().set_at(0, 0, 1.0);

    let status = filter.correct_with_in(&mut compass, &mut scratch, |y| {
        y[0] = minikalman::angles::wrap_180(y[0]);
    });

    assert!(status.is_applied());
    let heading = filter.state_vector().get_at(0, 0);
    assert!(heading > 359.5 && heading <= 361.0, "heading = {heading}");
    assert_eq!(compass.innovation_vector().get_row(0), 2.0);
}

#[test]
fn builders_wire_lean_observations_from_macro_buffers() {
    use minikalman::extended::ExtendedObservationBuilder;
    use minikalman::regular::{CorrectionScratch, RegularObservationBuilder};

    impl_buffer_x!(mut x, 2, f32, 0.0);
    impl_buffer_P!(mut P, 2, f32, 0.0);
    impl_buffer_z!(mut z, 1, f32, 0.0);
    impl_buffer_H!(mut H, 1, 2, f32, 0.0);
    impl_buffer_R!(mut R, 1, f32, 0.0);
    impl_buffer_y!(mut y, 1, f32, 0.0);
    impl_buffer_S!(mut S, 1, f32, 0.0);
    impl_buffer_K!(mut K, 2, 1, f32, 0.0);
    let mut regular = RegularObservationBuilder::new_lean::<2, 1, f32>(H, z, R, y, S, K);

    impl_buffer_z!(mut ez, 1, f32, 0.0);
    impl_buffer_H!(mut eH, 1, 2, f32, 0.0);
    impl_buffer_R!(mut eR, 1, f32, 0.0);
    impl_buffer_y!(mut ey, 1, f32, 0.0);
    impl_buffer_S!(mut eS, 1, f32, 0.0);
    impl_buffer_K!(mut eK, 2, 1, f32, 0.0);
    let mut extended = ExtendedObservationBuilder::new_lean::<2, 1, f32>(eH, ez, eR, ey, eS, eK);

    let mut scratch = CorrectionScratch::<2, 1, f32>::new(0.0);
    P.make_identity();
    regular.observation_matrix_mut().set_at(0, 0, 1.0);
    regular.measurement_noise_covariance_mut().set_at(0, 0, 1.0);
    regular.measurement_vector_mut().set_at(0, 0, 2.0);
    assert!(
        regular
            .correct_in(&mut x, &mut P, &mut scratch)
            .is_applied()
    );
    // micromath's approximate square root makes the inversion inexact.
    #[cfg(any(feature = "std", feature = "libm"))]
    assert!((x.get_row(0) - 1.0).abs() < 1e-6, "x = {}", x.get_row(0));
    assert!(x.get_row(0) > 0.5, "x = {}", x.get_row(0));

    extended.observation_jacobian_matrix_mut().set_at(0, 0, 1.0);
    extended
        .measurement_noise_covariance_mut()
        .set_at(0, 0, 1.0);
    extended.measurement_vector_mut().set_at(0, 0, 2.0);
    let status = extended.correct_nonlinear_in(&mut x, &mut P, &mut scratch, |x, z| {
        z.set_row(0, x.get_row(0));
    });
    assert!(status.is_applied());
    assert!(x.get_row(0) > 1.0);

    // The same corrections through the traits, as generic code would call them.
    let before = x.get_row(0);
    let status =
        KalmanFilterObservationCorrectIn::correct_in(&mut regular, &mut x, &mut P, &mut scratch);
    assert!(status.is_applied());
    let status = KalmanFilterNonlinearObservationCorrectIn::correct_nonlinear_in(
        &mut extended,
        &mut x,
        &mut P,
        &mut scratch,
        |x, z| z.set_row(0, x.get_row(0)),
    );
    assert!(status.is_applied());
    assert!(x.get_row(0) > before);
}

#[cfg(feature = "fixed")]
#[test]
fn lean_fixed_point_observation_matches_the_owning_one() {
    use minikalman::regular::{
        CorrectionScratch, FilterStorage, LeanObservationStorage, ObservationStorage,
    };

    /// Runs a 1-D tracker in `I16F16` and returns the bits of `x` and `P` after each step.
    fn run(
        correct: &mut dyn FnMut(&mut minikalman::regular::BorrowedKalman<'_, 2, I16F16>, I16F16),
    ) -> Vec<i32> {
        let mut filter_storage = FilterStorage::<2, I16F16>::new(I16F16::ZERO);
        let mut filter = filter_storage.build();
        filter.state_transition_mut().apply(|a| {
            a.make_identity();
            a.set_at(0, 1, I16F16::ONE);
        });
        filter.estimate_covariance_mut().make_identity();

        let mut bits = Vec::new();
        for step in 0..20 {
            filter.predict();
            correct(&mut filter, I16F16::from_num(step) * I16F16::from_num(1.5));
            let x = filter.state_vector().as_slice();
            let p = filter.estimate_covariance().as_slice();
            bits.extend(x.iter().chain(p).map(|v| v.to_bits()));
        }

        bits
    }

    let mut owning_storage = ObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut owning = owning_storage.build();
    owning.observation_matrix_mut().set_at(0, 0, I16F16::ONE);
    owning
        .measurement_noise_covariance_mut()
        .set_at(0, 0, I16F16::ONE);
    let expected = run(&mut |filter, z| {
        owning.measurement_vector_mut().set_at(0, 0, z);
        filter.correct(&mut owning);
    });

    let mut lean_storage = LeanObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut lean = lean_storage.build();
    let mut scratch = CorrectionScratch::<2, 1, I16F16>::new(I16F16::MAX);
    lean.observation_matrix_mut().set_at(0, 0, I16F16::ONE);
    lean.measurement_noise_covariance_mut()
        .set_at(0, 0, I16F16::ONE);
    let actual = run(&mut |filter, z| {
        lean.measurement_vector_mut().set_at(0, 0, z);
        filter.correct_in(&mut lean, &mut scratch);
    });

    assert_eq!(actual, expected);
}

#[test]
fn lean_radar_observation_saves_the_scratch() {
    use minikalman::extended::{CorrectionScratch, LeanObservationStorage, ObservationStorage};

    let owning = size_of::<ObservationStorage<4, 3, f32>>();
    let lean = size_of::<LeanObservationStorage<4, 3, f32>>();
    let scratch = size_of::<CorrectionScratch<4, 3, f32>>();

    assert_eq!(owning, 97 * size_of::<f32>());
    assert_eq!(lean, 48 * size_of::<f32>());
    assert_eq!(scratch, 49 * size_of::<f32>());

    // Two radar observations: 2 × 97 owning, against 2 × 48 lean plus one scratch of 49.
    assert_eq!(2 * owning - (2 * lean + scratch), 49 * size_of::<f32>());
}
