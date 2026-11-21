//! Zero-copy views through `#[derive(UniformArray)]` against the copying `state!` helpers.
//!
//! This test crate plays the user's crate: the derive's casts are gated on *its* `unsafe`
//! feature, which cargo sets here from minikalman's `unsafe` feature.

#![deny(missing_docs)]

use minikalman::prelude::*;
use minikalman::{UniformArray, extended, regular};

minikalman::state! {
    /// A falling object, as in the gravity example; the copying route.
    pub struct Fall(f32) {
        /// Height, in meters.
        s,
        /// Velocity, in meters per second.
        v,
        /// Acceleration, in meters per second squared.
        g,
    }
}

/// The zero-copy counterpart of [`Fall`], with the same field names.
#[derive(UniformArray)]
#[repr(C)]
pub struct FallView {
    /// Height, in meters.
    s: f32,
    /// Velocity, in meters per second.
    v: f32,
    /// Acceleration, in meters per second squared.
    g: f32,
}

minikalman::state! {
    /// A planar track: position and velocity; the copying route.
    pub struct Radar(f32) {
        /// East position, in meters.
        x,
        /// North position, in meters.
        y,
        /// East velocity, in meters per second.
        vx,
        /// North velocity, in meters per second.
        vy,
    }
}

/// The zero-copy counterpart of [`Radar`], with the same field names.
#[derive(UniformArray, Debug, PartialEq)]
#[repr(C)]
pub struct RadarView {
    /// East position, in meters.
    x: f32,
    /// North position, in meters.
    y: f32,
    /// East velocity, in meters per second.
    vx: f32,
    /// North velocity, in meters per second.
    vy: f32,
}

minikalman::state! {
    /// A radar measurement; the copying route.
    pub struct RadarZ(f32) {
        /// Distance to the radar, in meters.
        range,
        /// Bearing from the radar, in radians.
        bearing,
        /// Range rate, in meters per second.
        range_rate,
    }
}

/// The zero-copy counterpart of [`RadarZ`], with the same field names.
#[derive(UniformArray)]
#[repr(C)]
pub struct RadarZView {
    /// Distance to the radar, in meters.
    range: f32,
    /// Bearing from the radar, in radians.
    bearing: f32,
    /// Range rate, in meters per second.
    range_rate: f32,
}

minikalman::named_vector!(FallView);
minikalman::named_vector!(RadarView);
minikalman::named_vector!(RadarZView);

/// Observed distances of a falling object, as in the gravity example.
const REAL_DISTANCE: [f32; 15] = [
    0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
    706.32, 828.94, 961.38,
];

/// Observation noise of the gravity example.
const OBSERVATION_ERROR: [f32; 15] = [
    0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946, 0.69236,
    -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
];

/// Time step of the radar example, in seconds.
const DELTA_T: f32 = 0.1;

/// Maps a radar state to its measurement, with the radar at `(2, 1)`.
fn observe(x: f32, y: f32, vx: f32, vy: f32) -> RadarZ {
    let (dx, dy) = (x - 2.0, y - 1.0);
    let range = (dx * dx + dy * dy).sqrt();
    RadarZ {
        range,
        bearing: dy.atan2(dx),
        range_rate: (dx * vx + dy * vy) / range,
    }
}

/// Writes a measurement through a zero-copy view.
fn write_view(z: RadarZ, view: &mut RadarZView) {
    view.range = z.range;
    view.bearing = z.bearing;
    view.range_rate = z.range_rate;
}

/// Appends the bit patterns of `x` and `P` to a trace.
fn record(trace: &mut Vec<u32>, x: &[f32], p: &[f32]) {
    trace.extend(x.iter().chain(p).map(|v| v.to_bits()));
}

/// Runs the gravity example and returns the trace of `x` and `P`; `view` selects the
/// zero-copy views over the `state!` copies for the state.
fn run_gravity(view: bool) -> Vec<u32> {
    let mut filter_storage = regular::FilterStorage::<{ Fall::N }, f32>::new(0.0);
    let mut measurement_storage = regular::ObservationStorage::<{ Fall::N }, 1, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();

    if view {
        filter.state_vector_mut().view_mut::<FallView>().g = 6.0;
    } else {
        Fall {
            s: 0.0,
            v: 0.0,
            g: 6.0,
        }
        .write(filter.state_vector_mut().as_mut_slice());
    }

    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(Fall::s, Fall::v, 1.0);
        a.set_at(Fall::s, Fall::g, 0.5);
        a.set_at(Fall::v, Fall::g, 1.0);
    });
    filter.estimate_covariance_mut().apply(|p| {
        p.set_at(Fall::s, Fall::s, 0.1);
        p.set_at(Fall::v, Fall::v, 1.0);
        p.set_at(Fall::g, Fall::g, 1.0);
    });
    measurement
        .observation_matrix_mut()
        .apply(|h| h.set_at(0, Fall::s, 1.0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| r.set_at(0, 0, 0.5));

    let mut trace = Vec::new();
    for (real, error) in REAL_DISTANCE.iter().zip(OBSERVATION_ERROR) {
        filter.predict();
        measurement.measurement_vector_mut()[0] = real + error;
        filter.correct(&mut measurement);

        let s = if view {
            filter.state_vector().view::<FallView>().s
        } else {
            Fall::read(filter.state_vector().as_slice()).s
        };
        trace.push(s.to_bits());
        record(
            &mut trace,
            filter.state_vector().as_slice(),
            filter.estimate_covariance().as_slice(),
        );
    }

    trace
}

/// Runs the radar EKF and returns the trace of `x` and `P`; `view` selects the zero-copy views
/// over the `state!` copies inside the model closures and for the measurement.
fn run_radar(view: bool) -> Vec<u32> {
    let mut filter_storage = extended::FilterStorage::<{ Radar::N }, f32>::new(0.0);
    let mut measurement_storage =
        extended::ObservationStorage::<{ Radar::N }, { RadarZ::N }, f32>::new(0.0);
    let mut filter = filter_storage.build();
    let mut measurement = measurement_storage.build();

    *filter.state_vector_mut().view_mut::<RadarView>() = RadarView {
        x: 0.0,
        y: 0.0,
        vx: 0.5,
        vy: 2.0,
    };
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);
    measurement.measurement_noise_covariance_mut().apply(|r| {
        r.set_at(RadarZ::range, RadarZ::range, 0.5);
        r.set_at(RadarZ::bearing, RadarZ::bearing, 0.1);
        r.set_at(RadarZ::range_rate, RadarZ::range_rate, 0.2);
    });

    let mut trace = Vec::new();
    for step in 1..=100 {
        filter.state_transition_jacobian_mut().apply(|a| {
            a.make_identity();
            a.set_at(Radar::x, Radar::vx, DELTA_T);
            a.set_at(Radar::y, Radar::vy, DELTA_T);
        });
        filter.predict_nonlinear(|x, next| {
            if view {
                let s = x.view::<RadarView>();
                let n = next.view_mut::<RadarView>();
                n.x = s.x + s.vx * DELTA_T;
                n.y = s.y + s.vy * DELTA_T;
                n.vx = s.vx;
                n.vy = s.vy;
            } else {
                let s = Radar::read(x.as_slice());
                Radar {
                    x: s.x + s.vx * DELTA_T,
                    y: s.y + s.vy * DELTA_T,
                    ..s
                }
                .write(next.as_mut_slice());
            }
        });

        if step % 10 == 0 {
            correct_radar(&mut filter, &mut measurement, step, view);
        }

        record(
            &mut trace,
            filter.state_vector().as_slice(),
            filter.estimate_covariance().as_slice(),
        );
    }

    trace
}

/// Corrects the radar EKF with a noise-free measurement of an object moving diagonally at 1 m/s.
fn correct_radar(
    filter: &mut extended::BorrowedKalman<'_, { Radar::N }, f32>,
    measurement: &mut extended::BorrowedObservation<'_, { Radar::N }, { RadarZ::N }, f32>,
    step: i32,
    view: bool,
) {
    let t = step as f32 * DELTA_T;
    let truth = observe(t, t, 1.0, 1.0);
    if view {
        write_view(truth, measurement.measurement_vector_mut().view_mut());
    } else {
        truth.write(measurement.measurement_vector_mut().as_mut_slice());
    }

    let s = filter.state_vector().view::<RadarView>();
    let (dx, dy) = (s.x - 2.0, s.y - 1.0);
    let n2 = dx * dx + dy * dy;
    let n = n2.sqrt();
    measurement.observation_jacobian_matrix_mut().apply(|h| {
        h.set_at(RadarZ::range, Radar::x, dx / n);
        h.set_at(RadarZ::range, Radar::y, dy / n);
        h.set_at(RadarZ::bearing, Radar::x, -dy / n2);
        h.set_at(RadarZ::bearing, Radar::y, dx / n2);
        h.set_at(RadarZ::range_rate, Radar::vx, dx / n);
        h.set_at(RadarZ::range_rate, Radar::vy, dy / n);
    });

    filter.correct_nonlinear(measurement, |x, z| {
        if view {
            let s = x.view::<RadarView>();
            write_view(observe(s.x, s.y, s.vx, s.vy), z.view_mut());
        } else {
            let s = Radar::read(x.as_slice());
            observe(s.x, s.y, s.vx, s.vy).write(z.as_mut_slice());
        }
    });
}

#[test]
fn layout_matches_the_vector() {
    assert_eq!(size_of::<RadarView>(), Radar::N * size_of::<f32>());
    assert_eq!(<RadarView as NamedVector<f32>>::N, Radar::N);
    assert_eq!(<RadarZView as NamedVector<f32>>::N, RadarZ::N);
    assert_eq!(<FallView as NamedVector<f32>>::N, Fall::N);
}

#[test]
fn views_alias_the_buffer() {
    let mut storage = extended::FilterStorage::<{ Radar::N }, f32>::new(0.0);
    let mut filter = storage.build();

    filter.state_vector_mut().view_mut::<RadarView>().vx = 3.0;
    filter.state_vector_mut()[Radar::vy] = 4.0;

    let view = filter.state_vector().view::<RadarView>();
    assert_eq!(filter.state_vector().as_slice(), &[0.0, 0.0, 3.0, 4.0]);
    assert!(core::ptr::eq(
        core::ptr::from_ref(view).cast::<f32>(),
        filter.state_vector().as_slice().as_ptr()
    ));
    assert_eq!(
        *view,
        RadarView {
            x: 0.0,
            y: 0.0,
            vx: 3.0,
            vy: 4.0
        }
    );
}

#[test]
fn gravity_views_match_copies() {
    assert_eq!(run_gravity(true), run_gravity(false));
}

#[test]
fn radar_views_match_copies() {
    assert_eq!(run_radar(true), run_radar(false));
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn short_slices_panic() {
    let _ = <RadarView as NamedVector<f32>>::from_vector(&[0.0; 3]);
}

#[test]
#[cfg(feature = "fixed")]
fn fixed_point_scalars_work() {
    use fixed::types::I16F16;

    /// A fixed-point falling object.
    #[derive(UniformArray)]
    #[repr(C)]
    struct FixedFall {
        /// Height, in meters.
        s: I16F16,
        /// Velocity, in meters per second.
        v: I16F16,
    }

    minikalman::named_vector!(FixedFall);

    let mut storage = regular::FilterStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut filter = storage.build();
    filter.state_vector_mut().view_mut::<FixedFall>().v = I16F16::from_num(2);

    assert_eq!(filter.state_vector()[1], I16F16::from_num(2));
    assert_eq!(filter.state_vector().view::<FixedFall>().s, I16F16::ZERO);
}
