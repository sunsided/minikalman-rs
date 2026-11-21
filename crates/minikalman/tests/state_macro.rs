//! The `state!` macro: index constants, vector and column helpers, and use inside filter closures.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use minikalman::prelude::*;
use minikalman::unscented::{FilterStorage, ObservationStorage};

minikalman::state! {
    /// A planar track: position and velocity.
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

minikalman::state! {
    /// A radar measurement.
    pub struct RadarZ(f32) {
        /// Distance to the radar, in meters.
        range,
        /// Bearing from the radar, in radians.
        bearing,
        /// Range rate, in meters per second.
        range_rate,
    }
}

/// Time step, in seconds.
const DELTA_T: f32 = 0.1;

/// Number of sigma points for [`Radar`].
const NUM_SIGMA: usize = 2 * Radar::N + 1;

/// Maps a radar state to its measurement, with the radar at `(2, 1)`.
fn observe(s: Radar) -> RadarZ {
    let (dx, dy) = (s.x - 2.0, s.y - 1.0);
    let range = (dx * dx + dy * dy).sqrt();
    RadarZ {
        range,
        bearing: dy.atan2(dx),
        range_rate: (dx * s.vx + dy * s.vy) / range,
    }
}

/// Runs a radar UKF and returns the bits of `x` and `P` at the end; `named` selects the
/// `state!` helpers or hand-written indices in the closures.
fn run_ukf(named: bool) -> Vec<u32> {
    let mut filter_storage = FilterStorage::<{ Radar::N }, NUM_SIGMA, f32>::new(0.0);
    let mut measurement_storage =
        ObservationStorage::<{ Radar::N }, { RadarZ::N }, NUM_SIGMA, f32>::new(0.0);
    let mut filter = filter_storage.build(1.0, 2.0, 1.0);
    let mut measurement = measurement_storage.build();

    Radar {
        x: 0.0,
        y: 0.0,
        vx: 0.5,
        vy: 2.0,
    }
    .write(filter.state_vector_mut().as_mut_slice());
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);
    measurement.measurement_noise_covariance_mut().apply(|r| {
        r.set_at(RadarZ::range, RadarZ::range, 0.5);
        r.set_at(RadarZ::bearing, RadarZ::bearing, 0.1);
        r.set_at(RadarZ::range_rate, RadarZ::range_rate, 0.2);
    });

    for step in 1..=50 {
        filter.predict_nonlinear(|x| {
            if named {
                let s = Radar::read(x.as_slice());
                Radar {
                    x: s.x + s.vx * DELTA_T,
                    y: s.y + s.vy * DELTA_T,
                    ..s
                }
                .write(x.as_mut_slice());
            } else {
                x[0] += x[2] * DELTA_T;
                x[1] += x[3] * DELTA_T;
            }
        });

        if step % 10 == 0 {
            let t = step as f32 * DELTA_T;
            observe(Radar {
                x: t,
                y: t,
                vx: 1.0,
                vy: 1.0,
            })
            .write(measurement.measurement_vector_mut().as_mut_slice());
            filter.correct_sigma_point(&mut measurement, |sigma, observed| {
                for j in 0..NUM_SIGMA {
                    if named {
                        observe(Radar::read_column(sigma, NUM_SIGMA, j))
                            .write_column(observed, NUM_SIGMA, j);
                    } else {
                        let z = observe(Radar {
                            x: sigma[j],
                            y: sigma[NUM_SIGMA + j],
                            vx: sigma[2 * NUM_SIGMA + j],
                            vy: sigma[3 * NUM_SIGMA + j],
                        });
                        observed[j] = z.range;
                        observed[NUM_SIGMA + j] = z.bearing;
                        observed[2 * NUM_SIGMA + j] = z.range_rate;
                    }
                }
            });
        }
    }

    let x = filter.state_vector().as_slice().iter();
    let p = filter.estimate_covariance().as_slice().iter();
    x.chain(p).map(|v| v.to_bits()).collect()
}

#[test]
fn constants_follow_declaration_order() {
    assert_eq!(Radar::N, 4);
    assert_eq!([Radar::x, Radar::y, Radar::vx, Radar::vy], [0, 1, 2, 3]);
    assert_eq!(RadarZ::N, 3);
    assert_eq!(RadarZ::range_rate, 2);
}

#[test]
fn vectors_round_trip() {
    let mut vector = [0.0_f32; Radar::N + 1];
    let state = Radar {
        x: 1.0,
        y: 2.0,
        vx: 3.0,
        vy: 4.0,
    };
    state.write(&mut vector);

    assert_eq!(vector, [1.0, 2.0, 3.0, 4.0, 0.0]);
    assert_eq!(Radar::read(&vector), state);
}

#[test]
fn columns_round_trip() {
    let mut matrix = [0.0_f32; Radar::N * 3];
    let state = Radar {
        x: 1.0,
        y: 2.0,
        vx: 3.0,
        vy: 4.0,
    };
    state.write_column(&mut matrix[..], 3, 1);

    assert_eq!(
        matrix,
        [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0]
    );
    assert_eq!(Radar::read_column(&matrix[..], 3, 1), state);
}

#[test]
#[should_panic]
fn short_vectors_panic() {
    let _ = Radar::read(&[0.0; 3]);
}

#[test]
fn named_ukf_matches_indexed_ukf() {
    assert_eq!(run_ukf(true), run_ukf(false));
}

#[test]
#[cfg(feature = "fixed")]
fn fixed_point_scalars_work() {
    use fixed::types::I16F16;

    minikalman::state! {
        /// A fixed-point falling object.
        struct Fall(I16F16) {
            /// Height, in meters.
            s,
            /// Velocity, in meters per second.
            v,
        }
    }

    let mut vector = [I16F16::ZERO; Fall::N];
    Fall {
        s: I16F16::ONE,
        v: I16F16::from_num(2),
    }
    .write(&mut vector);
    assert_eq!(Fall::read(&vector).v, I16F16::from_num(2));
}

#[cfg(not(test))]
minikalman::state! {
    /// Compiled out: a `#[cfg]` on the invocation gates the struct and every generated item.
    pub struct Disabled(f32) {
        /// Never compiled.
        a,
    }
}

#[cfg(test)]
minikalman::state! {
    /// Compiled in through a `#[cfg]` on the invocation.
    pub struct Enabled(f32) {
        /// The only entry.
        a,
    }
}

#[test]
fn cfg_on_the_invocation_gates_the_whole_expansion() {
    assert_eq!(Enabled::N, 1);
    assert_eq!(Enabled::read(&[2.0]).a, 2.0);
}
