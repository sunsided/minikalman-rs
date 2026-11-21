//! Tests of the `filter!` DSL: the generated filters against the same filters wired by hand,
//! sharing across filters, blocks and modules, and the `build()`/`model()` split.

#![allow(non_snake_case)]

use minikalman::prelude::*;

/// Time step, in seconds.
const DT: f32 = 0.1;

/// The number of steps every scenario runs.
const STEPS: usize = 50;

/// The declarations every filter over `Track` shares, in their own module and block.
mod shared {
    use super::DT;

    minikalman::filter! {
        /// Position and velocity on a line.
        pub state Track(f32) { pos, vel }

        /// Commanded acceleration.
        pub control throttle: Throttle for Track { accel } {
            B[pos, accel] = 0.5 * DT * DT;
            B[vel, accel] = DT;
            Q = diag(0.01);
        }

        /// A position fix, without temporaries.
        pub observation lean gps: Gps for Track { pos } {
            H = [pos];
            R = diag(0.5);
        }

        /// Temporaries for every lean correction of a `Track` filter.
        pub scratch scratch for [gps];
    }
}

/// The filters, in a second module and block; they reach the shared items only by path.
mod filters {
    use super::DT;
    pub use super::shared::{Track, gps, scratch, throttle};
    use minikalman::models::motion::constant_velocity;

    minikalman::filter! {
        /// Trusts the model less; reacts quickly.
        pub kalman fast for Track {
            A = |a| constant_velocity(DT, 1.0).fill_transition(a);
            Q = |q| constant_velocity(DT, 1.0).fill_process_noise(q);
            P = identity;
            x = Track { pos: 0.0, vel: 0.0 };
        }

        /// Trusts the model more; smooths.
        pub kalman slow for crate::shared::Track {
            A = |a| constant_velocity(DT, 0.01).fill_transition(a);
            Q = |q| constant_velocity(DT, 0.01).fill_process_noise(q);
            P = identity;
        }

        /// Fixed gain, no covariance; `H` comes from the `gps` declaration.
        pub luenberger steady for Track {
            A = identity;
            A[pos, vel] = DT;
        } gains {
            gps {
                K[pos, pos] = 0.5;
                K[vel, pos] = 0.8;
            }
        }

        /// Everything in one value.
        pub system tracking { fast, slow, steady, gps, throttle, scratch }
    }
}

use filters::{steady, tracking};
use shared::{Gps, Throttle, Track};

/// The generated storage is exactly the library storages, nothing more.
const _: () = {
    use core::mem::size_of;
    use minikalman::regular::*;

    let expected = 2 * size_of::<FilterStorage<2, f32>>()
        + size_of::<ObserverStorage<2, f32>>()
        + size_of::<FixedGainObservationStorage<2, 1, f32>>()
        + size_of::<LeanObservationStorage<2, 1, f32>>()
        + size_of::<ControlStorage<2, 1, f32>>()
        + size_of::<CorrectionScratch<2, 1, f32>>();
    assert!(size_of::<tracking::Storage>() == expected);
    // 2·(4+2+4+4+2+4) + (4+2+2) + (2+1+1+2) + (2+1+1+1+1+2) + (2+1+1+2) + (1+2+2+4) = 77 f32
    assert!(size_of::<tracking::Storage>() == 77 * 4);
    assert!(size_of::<steady::Storage>() == (8 + 6) * 4);
};

/// A `static` of the zeroed storage compiles: `new()` is `const`.
static STATIC_STORAGE: tracking::Storage = tracking::new();

/// A deterministic, noisy position fix for step `k` under a constant acceleration of 1 m/s².
fn fix(k: usize) -> f32 {
    let t = k as f32 * DT;
    0.5 * t * t + 0.3 * (k as f32 * 1.7).sin()
}

/// The scenario with the DSL: one storage, one build, one model, then the loop.
fn run_dsl() -> [Track; 3] {
    let mut storage = tracking::new();
    let mut sys = tracking::build(&mut storage);
    tracking::model(&mut sys);

    for k in 0..STEPS {
        Throttle { accel: 1.0 }.write(sys.throttle.control_vector_mut().as_mut_slice());
        let z = Gps { pos: fix(k) };

        // Two Kalman filters: same control object, same observation object, same scratch.
        for filter in [&mut sys.fast, &mut sys.slow] {
            filter.predict().control(&mut sys.throttle);
            z.write(sys.gps.measurement_vector_mut().as_mut_slice());
            assert!(filter.correct_in(&mut sys.gps, sys.scratch).is_applied());
        }

        sys.steady.observer.predict().control(&mut sys.throttle);
        z.write(sys.steady.gps.measurement_vector_mut().as_mut_slice());
        assert!(
            sys.steady
                .observer
                .correct(&mut sys.steady.gps)
                .is_applied()
        );
    }

    [
        Track::read(sys.fast.state_vector().as_slice()),
        Track::read(sys.slow.state_vector().as_slice()),
        Track::read(sys.steady.observer.state_vector().as_slice()),
    ]
}

/// The same scenario on the library API without the DSL.
fn run_by_hand() -> [Track; 3] {
    use minikalman::models::motion::constant_velocity;
    use minikalman::regular::*;

    let mut fast_storage = FilterStorage::<2, f32>::new(0.0);
    let mut slow_storage = FilterStorage::<2, f32>::new(0.0);
    let mut observer_storage = ObserverStorage::<2, f32>::new(0.0);
    let mut fixed_gps_storage = FixedGainObservationStorage::<2, 1, f32>::new(0.0);
    let mut gps_storage = LeanObservationStorage::<2, 1, f32>::new(0.0);
    let mut throttle_storage = ControlStorage::<2, 1, f32>::new(0.0);
    let mut scratch = CorrectionScratch::<2, 1, f32>::new(0.0);

    let mut fast = fast_storage.build();
    let mut slow = slow_storage.build();
    let mut observer = observer_storage.build();
    let mut fixed_gps = fixed_gps_storage.build();
    let mut gps = gps_storage.build();
    let mut throttle = throttle_storage.build();

    for (filter, q) in [(&mut fast, 1.0), (&mut slow, 0.01)] {
        let model = constant_velocity(DT, q);
        filter
            .state_transition_mut()
            .apply(|a| model.fill_transition(a));
        filter
            .direct_process_noise_mut()
            .apply(|m| model.fill_process_noise(m));
        filter.estimate_covariance_mut().make_identity();
    }

    throttle.control_matrix_mut().apply(|b| {
        b.set_at(Track::pos, Throttle::accel, 0.5 * DT * DT);
        b.set_at(Track::vel, Throttle::accel, DT);
    });
    throttle.process_noise_covariance_mut().make_scalar(0.01);
    gps.observation_matrix_mut()
        .set_at(Gps::pos, Track::pos, 1.0);
    gps.measurement_noise_covariance_mut().make_scalar(0.5);
    observer.state_transition_mut().make_identity();
    observer
        .state_transition_mut()
        .set_at(Track::pos, Track::vel, DT);
    fixed_gps
        .observation_matrix_mut()
        .set_at(Gps::pos, Track::pos, 1.0);
    fixed_gps.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.5, 0.8]));

    for k in 0..STEPS {
        Throttle { accel: 1.0 }.write(throttle.control_vector_mut().as_mut_slice());
        let z = Gps { pos: fix(k) };
        for filter in [&mut fast, &mut slow] {
            filter.predict().control(&mut throttle);
            z.write(gps.measurement_vector_mut().as_mut_slice());
            filter.correct_in(&mut gps, &mut scratch);
        }

        observer.predict().control(&mut throttle);
        z.write(fixed_gps.measurement_vector_mut().as_mut_slice());
        assert!(observer.correct(&mut fixed_gps).is_applied());
    }

    [
        Track::read(fast.state_vector().as_slice()),
        Track::read(slow.state_vector().as_slice()),
        Track::read(observer.state_vector().as_slice()),
    ]
}

#[test]
fn dsl_matches_hand_wired_filters_bit_for_bit() {
    let dsl = run_dsl();
    assert_eq!(dsl, run_by_hand());

    // All three track the true trajectory (pos ≈ 0.5·t², vel ≈ t at t = 4.9 s).
    let t = (STEPS - 1) as f32 * DT;
    for track in dsl {
        assert!((track.pos - 0.5 * t * t).abs() < 1.5, "{track:?}");
        assert!((track.vel - t).abs() < 1.5, "{track:?}");
    }

    // The filters differ, so each really ran its own model.
    assert_ne!(dsl[0], dsl[1]);
    assert_ne!(dsl[1], dsl[2]);
}

#[test]
fn build_wires_only_and_model_writes_and_resets() {
    let mut storage = filters::fast::new();

    let mut filter = filters::fast::build(&mut storage);
    assert_eq!(filter.estimate_covariance().get_at(0, 0), 0.0);
    assert_eq!(filter.state_transition().get_at(0, 0), 0.0);

    filters::fast::model(&mut filter);
    assert_eq!(filter.estimate_covariance().get_at(0, 0), 1.0);
    assert_eq!(filter.state_transition().get_at(0, 1), DT);
    filter.estimate_covariance_mut().set_at(0, 0, 42.0);
    filter.state_vector_mut().set_at(Track::vel, 0, 3.0);

    // A second build keeps what the storage holds.
    let mut filter = filters::fast::build(&mut storage);
    assert_eq!(filter.estimate_covariance().get_at(0, 0), 42.0);

    // model() resets the declared values, here P and x.
    filters::fast::model(&mut filter);
    assert_eq!(filter.estimate_covariance().get_at(0, 0), 1.0);
    assert_eq!(filter.state_vector().get_at(Track::vel, 0), 0.0);
}

/// `H = [...]` is a whole-matrix assignment: `model()` clears entries outside the selection.
#[test]
fn model_resets_a_selected_observation_matrix() {
    let mut storage = shared::gps::new();
    let mut gps = shared::gps::build(&mut storage);
    shared::gps::model(&mut gps);
    assert_eq!(gps.observation_matrix().get_at(0, Track::pos), 1.0);

    gps.observation_matrix_mut().set_at(0, Track::vel, 7.0);
    shared::gps::model(&mut gps);
    assert_eq!(gps.observation_matrix().get_at(0, Track::pos), 1.0);
    assert_eq!(gps.observation_matrix().get_at(0, Track::vel), 0.0);
}

#[test]
fn a_system_model_writes_every_member() {
    let mut storage = tracking::new();
    let mut sys = tracking::build(&mut storage);
    assert_eq!(sys.gps.observation_matrix().get_at(0, 0), 0.0);

    tracking::model(&mut sys);
    assert_eq!(
        sys.gps.observation_matrix().get_at(Gps::pos, Track::pos),
        1.0
    );
    assert_eq!(sys.gps.measurement_noise_covariance().get_at(0, 0), 0.5);
    assert_eq!(
        sys.throttle
            .control_matrix()
            .get_at(Track::vel, Throttle::accel),
        DT
    );
    assert_eq!(sys.throttle.process_noise_covariance().get_at(0, 0), 0.01);
    assert_eq!(sys.slow.estimate_covariance().get_at(1, 1), 1.0);
    assert_eq!(sys.steady.observer.state_transition().get_at(0, 1), DT);
    assert_eq!(sys.steady.gps.observation_matrix().get_at(0, 0), 1.0);
    assert_eq!(sys.steady.gps.gain().get_at(Track::vel, Gps::pos), 0.8);
}

#[test]
fn the_static_storage_is_zeroed_and_sized() {
    assert_eq!(core::mem::size_of_val(&STATIC_STORAGE), 77 * 4);
    assert_eq!(shared::scratch::MAX_OBSERVATIONS, 1);
    assert_eq!(shared::gps::OBSERVATIONS, 1);
    assert_eq!(shared::throttle::CONTROLS, 1);
    assert_eq!(filters::fast::STATES, 2);
    assert_eq!(steady::STATES, 2);
}

/// Full observations, a settled gain handed to an observer, and every statement form.
mod full {
    use super::DT;
    use minikalman::models::motion::constant_velocity;

    minikalman::filter! {
        pub state Track(f32) { pos, vel }

        /// Full by default: works with `correct` and `settle_gain`.
        pub observation gps: Gps for Track { pos } {
            H[pos, pos] = 1.0;
            R[pos, pos] = 0.25;
            z = Gps { pos: 0.0 };
        }

        /// A filler closure for `H`, through `fill_h` too.
        pub observation odometry: Odometry for Track { vel } {
            H = |h| h.set_at(Odometry::vel, Track::vel, 1.0);
            R = diag(0.1);
        }

        /// Square `H` forms, which `fill_h` writes without `make_identity`.
        pub observation both: Both for Track { pos, vel } { H = identity; R = diag(0.2); }
        pub observation scaled: Scaled for Track { pos, vel } { H = diag(2.0); }

        pub control push: Push for Track { accel } {
            B = |b| b.set_at(Track::vel, Push::accel, DT);
            Q = identity;
            u = Push { accel: 2.0 };
        }

        pub kalman design for Track {
            A = |a| constant_velocity(DT, 1.0).fill_transition(a);
            Q = |q| constant_velocity(DT, 1.0).fill_process_noise(q);
            P = diag(10.0);
            x = Track { pos: 1.0, vel: -1.0 };
        }

        /// The gains come from `settle_gain`, so the gain blocks stay empty.
        pub luenberger steady for Track {
            A = |a| constant_velocity(DT, 1.0).fill_transition(a);
            x = Track { pos: 1.0, vel: -1.0 };
        } gains {
            gps {}
            odometry {}
            both {}
            scaled { K[vel, vel] = 0.3; }
        }

        pub system design_system { design, gps, odometry, push }
    }
}

#[test]
fn a_full_observation_settles_a_gain_for_the_observer() {
    use full::{Gps, Track, design_system, steady};

    let mut storage = design_system::new();
    let mut sys = design_system::build(&mut storage);
    design_system::model(&mut sys);
    assert_eq!(sys.design.estimate_covariance().get_at(0, 0), 10.0);
    assert_eq!(Track::read(sys.design.state_vector().as_slice()).vel, -1.0);
    assert_eq!(sys.push.control_vector().get_at(0, 0), 2.0);
    assert_eq!(sys.push.process_noise_covariance().get_at(0, 0), 1.0);
    assert_eq!(sys.push.control_matrix().get_at(Track::vel, 0), DT);
    assert_eq!(sys.odometry.observation_matrix().get_at(0, Track::vel), 1.0);

    let iterations = sys.design.settle_gain(&mut sys.gps, 1e-6, 10_000).unwrap();
    assert!(iterations > 1);

    let mut observer_storage = steady::new();
    let mut observer = steady::build(&mut observer_storage);
    observer.gps.set_gain(sys.gps.kalman_gain());

    // An empty gain block leaves the gain alone; `H` still comes from the declaration.
    steady::model(&mut observer);
    let gain = observer.gps.gain().get_at(0, 0);
    assert_eq!(gain, sys.gps.kalman_gain().get_at(0, 0));
    assert!(gain > 0.0);
    assert_eq!(observer.gps.observation_matrix().get_at(0, Track::pos), 1.0);
    assert_eq!(
        observer.odometry.observation_matrix().get_at(0, Track::vel),
        1.0
    );
    assert_eq!(observer.odometry.gain().get_at(0, 0), 0.0);
    assert_eq!(
        observer.both.observation_matrix().as_slice(),
        &[1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        observer.scaled.observation_matrix().as_slice(),
        &[2.0, 0.0, 0.0, 2.0]
    );
    assert_eq!(observer.scaled.gain().as_slice(), &[0.0, 0.0, 0.0, 0.3]);

    let mut both_storage = full::both::new();
    let mut both = full::both::build(&mut both_storage);
    full::both::model(&mut both);
    assert_eq!(both.observation_matrix().as_slice(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(
        both.measurement_noise_covariance().as_slice(),
        &[0.2, 0.0, 0.0, 0.2]
    );

    observer.observer.predict();
    Gps { pos: 2.0 }.write(observer.gps.measurement_vector_mut().as_mut_slice());
    assert!(observer.observer.correct(&mut observer.gps).is_applied());
    assert!(Track::read(observer.observer.state_vector().as_slice()).pos > 1.0);
}

#[test]
fn a_full_observation_corrects_like_the_hand_wired_one() {
    use full::{Gps, Track, design, gps};
    use minikalman::regular::{FilterStorage, ObservationStorage};

    let (mut filter_storage, mut gps_storage) = (design::new(), gps::new());
    let mut filter = design::build(&mut filter_storage);
    let mut observation = gps::build(&mut gps_storage);
    design::model(&mut filter);
    gps::model(&mut observation);

    let mut hand_filter_storage = FilterStorage::<2, f32>::new(0.0);
    let mut hand_gps_storage = ObservationStorage::<2, 1, f32>::new(0.0);
    let mut hand_filter = hand_filter_storage.build();
    let mut hand_gps = hand_gps_storage.build();
    let model = minikalman::models::motion::constant_velocity(DT, 1.0);
    hand_filter
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    hand_filter
        .direct_process_noise_mut()
        .apply(|q| model.fill_process_noise(q));
    hand_filter.estimate_covariance_mut().make_scalar(10.0);
    Track {
        pos: 1.0,
        vel: -1.0,
    }
    .write(hand_filter.state_vector_mut().as_mut_slice());
    hand_gps.observation_matrix_mut().set_at(0, 0, 1.0);
    hand_gps
        .measurement_noise_covariance_mut()
        .set_at(0, 0, 0.25);

    for k in 0..STEPS {
        let z = Gps { pos: fix(k) };
        filter.predict();
        hand_filter.predict();
        z.write(observation.measurement_vector_mut().as_mut_slice());
        z.write(hand_gps.measurement_vector_mut().as_mut_slice());
        filter.correct(&mut observation);
        hand_filter.correct(&mut hand_gps);
    }

    assert_eq!(
        filter.state_vector().as_slice(),
        hand_filter.state_vector().as_slice()
    );
    assert_eq!(
        filter.estimate_covariance().as_slice(),
        hand_filter.estimate_covariance().as_slice()
    );
}

/// The predictor gain form, and modules named like the `lean` and `predictor` keywords.
mod predictor_form {
    use super::DT;

    minikalman::filter! {
        pub state Track(f32) { pos, vel }
        pub observation lean: Gps for Track { pos } { H = [pos]; }

        pub luenberger predictor delayed for Track {
            A = identity;
            A[pos, vel] = DT;
        } gains {
            lean { L[pos, pos] = 0.6; L[vel, pos] = 0.9; }
        }

        /// The current form, in a module named `predictor`.
        pub luenberger predictor for Track { A = identity; } gains {}
    }
}

#[test]
fn the_predictor_form_matches_the_hand_wired_observer() {
    use minikalman::regular::{FixedGainObservationStorage, PredictorObserverStorage};
    use predictor_form::{Gps, Track, delayed};

    let mut storage = delayed::new();
    let mut dsl = delayed::build(&mut storage);
    delayed::model(&mut dsl);

    let mut observer_storage = PredictorObserverStorage::<2, f32>::new(0.0);
    let mut gps_storage = FixedGainObservationStorage::<2, 1, f32, Predictor>::new(0.0);
    let mut observer = observer_storage.build();
    let mut gps = gps_storage.build();
    observer.state_transition_mut().make_identity();
    observer.state_transition_mut().set_at(0, 1, DT);
    gps.observation_matrix_mut().set_at(0, 0, 1.0);
    gps.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.6, 0.9]));

    for k in 0..STEPS {
        let z = Gps { pos: fix(k) };
        z.write(dsl.lean.measurement_vector_mut().as_mut_slice());
        z.write(gps.measurement_vector_mut().as_mut_slice());
        assert!(dsl.observer.correct(&mut dsl.lean).is_applied());
        assert!(observer.correct(&mut gps).is_applied());
        dsl.observer.predict();
        observer.predict();
    }

    let dsl_track = Track::read(dsl.observer.state_vector().as_slice());
    assert_eq!(dsl_track, Track::read(observer.state_vector().as_slice()));
    assert!(dsl_track.pos > 0.0);

    // The keyword-named modules are ordinary items.
    assert_eq!(predictor_form::predictor::STATES, 2);
    assert_eq!(predictor_form::lean::OBSERVATIONS, 1);
    let mut empty = predictor_form::predictor::new();
    let mut empty = predictor_form::predictor::build(&mut empty);
    predictor_form::predictor::model(&mut empty);
    assert_eq!(empty.observer.state_transition().get_at(1, 1), 1.0);
}

/// A fixed-point filter, with typed constants.
#[cfg(feature = "fixed")]
mod fixed_point {
    use minikalman::prelude::*;

    /// The time step.
    pub const DT: I16F16 = I16F16::lit("0.1");
    /// The measurement noise.
    pub const HALF: I16F16 = I16F16::lit("0.5");

    minikalman::filter! {
        pub state Track(I16F16) { pos, vel }
        pub observation gps: Gps for Track { pos } { H = [pos]; R = diag(HALF); }
        pub kalman tracker for Track { A = identity; A[pos, vel] = DT; P = identity; Q = diag(DT); }
        pub system tracking { tracker, gps }
    }
}

#[cfg(feature = "fixed")]
#[test]
fn a_fixed_point_filter_runs() {
    use fixed_point::{Gps, HALF, Track, tracking};

    let mut storage = tracking::new();
    let mut sys = tracking::build(&mut storage);
    tracking::model(&mut sys);
    for _ in 0..10 {
        sys.tracker.predict();
        Gps { pos: I16F16::ONE }.write(sys.gps.measurement_vector_mut().as_mut_slice());
        sys.tracker.correct(&mut sys.gps);
    }

    assert!(Track::read(sys.tracker.state_vector().as_slice()).pos > HALF);
}
