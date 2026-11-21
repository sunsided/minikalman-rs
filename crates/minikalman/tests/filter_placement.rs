//! Tests of the `filter!` placement modes: `storage = owned | boxed` against `inline`, and
//! `extern` buffers that the filter reads and writes in place.

use minikalman::prelude::*;

/// Time step, in seconds.
const DT: f32 = 0.1;

/// The number of steps every scenario runs.
const STEPS: usize = 40;

/// The inline reference: the phase-1 placement.
mod inline {
    use super::DT;

    minikalman::filter! {
        pub state Track(f32) { pos, vel }
        pub control push: Push for Track { accel } { B[pos, accel] = 0.5 * DT * DT; B[vel, accel] = DT; Q = diag(0.01); }
        pub observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
        pub kalman tracker for Track { A = identity; A[pos, vel] = DT; Q = diag(0.01); P = diag(10.0); }
        pub observation lean lean_gps: LeanGps for Track { pos } { H = [pos]; R = diag(0.5); }
        pub scratch shared for [lean_gps];
    }
}

/// The same items, owned by value.
mod owned {
    use super::DT;
    pub use super::inline::Track;

    minikalman::filter! {
        pub control push: Push for Track { accel } {
            storage = owned;
            B[pos, accel] = 0.5 * DT * DT;
            B[vel, accel] = DT;
            Q = diag(0.01);
        }
        pub observation gps: Gps for Track { pos } { storage = owned; H = [pos]; R = diag(0.5); }
        pub kalman tracker for Track {
            storage = owned;
            A = identity;
            A[pos, vel] = DT;
            Q = diag(0.01);
            P = diag(10.0);
        }
    }
}

/// The same items with `P`, `x`, `H`, `z` and `u` in buffers the caller owns.
mod external {
    use super::DT;
    pub use super::inline::{Track, shared};

    minikalman::filter! {
        pub control push: Push for Track { accel } {
            u = extern;
            B[pos, accel] = 0.5 * DT * DT;
            B[vel, accel] = DT;
            Q = diag(0.01);
        }
        pub observation gps: Gps for Track { pos } { H = extern; z = extern; H = [pos]; R = diag(0.5); }
        pub observation lean lean_gps: LeanGps for Track { pos } { R = extern; H = [pos]; R = diag(0.5); }
        pub kalman tracker for Track {
            P = extern;
            x = extern;
            A = identity;
            A[pos, vel] = DT;
            Q = diag(0.01);
            P = diag(10.0);
        }

        /// The fixed-gain observation takes `H` from the extern-`H` declaration.
        pub luenberger observer for Track { A = identity; A[pos, vel] = DT; } gains {
            gps { K[pos, pos] = 0.5; K[vel, pos] = 0.8; }
        }
    }
}

/// A deterministic, noisy position fix for step `k` under a constant acceleration of 1 m/s².
fn fix(k: usize) -> f32 {
    let t = k as f32 * DT;
    0.5 * t * t + 0.3 * (k as f32 * 1.7).sin()
}

/// The final state of the inline reference.
fn run_inline() -> [f32; 2] {
    use inline::{Gps, Push, gps, push, tracker};

    let (mut filter_storage, mut gps_storage, mut push_storage) =
        (tracker::new(), gps::new(), push::new());
    let mut filter = tracker::build(&mut filter_storage);
    let mut observation = gps::build(&mut gps_storage);
    let mut control = push::build(&mut push_storage);
    tracker::model(&mut filter);
    gps::model(&mut observation);
    push::model(&mut control);

    for k in 0..STEPS {
        Push { accel: 1.0 }.write(control.control_vector_mut().as_mut_slice());
        filter.predict().control(&mut control);
        Gps { pos: fix(k) }.write(observation.measurement_vector_mut().as_mut_slice());
        filter.correct(&mut observation);
    }

    let x = filter.state_vector().as_slice();
    [x[0], x[1]]
}

/// A tracker that owns its filter, observation and control: no lifetime, movable.
struct Tracker {
    /// The filter.
    filter: owned::tracker::Owned,
    /// The position fix.
    gps: owned::gps::Owned,
    /// The commanded acceleration.
    push: owned::push::Owned,
}

impl Tracker {
    /// Creates the tracker with the declared model.
    fn new() -> Self {
        let mut tracker = Self {
            filter: owned::tracker::new_owned(),
            gps: owned::gps::new_owned(),
            push: owned::push::new_owned(),
        };
        owned::tracker::model(&mut tracker.filter);
        owned::gps::model(&mut tracker.gps);
        owned::push::model(&mut tracker.push);
        tracker
    }

    /// Runs one step with the position fix `pos`.
    fn step(&mut self, pos: f32) {
        owned::Push { accel: 1.0 }.write(self.push.control_vector_mut().as_mut_slice());
        self.filter.predict().control(&mut self.push);
        owned::Gps { pos }.write(self.gps.measurement_vector_mut().as_mut_slice());
        self.filter.correct(&mut self.gps);
    }
}

#[test]
fn owned_matches_inline_bit_for_bit() {
    // Built in one function, moved out of it and into an array, then stepped there.
    let mut trackers = [Tracker::new()];
    let tracker = &mut trackers[0];
    for k in 0..STEPS {
        tracker.step(fix(k));
    }

    let x = tracker.filter.state_vector().as_slice();
    assert_eq!([x[0], x[1]], run_inline());
    assert!(owned::tracker::BYTES >= core::mem::size_of::<inline::tracker::Storage>());
    assert_eq!(owned::tracker::EXTERN_BYTES, 0);
}

#[test]
fn model_resets_an_owned_filter() {
    let mut filter = owned::tracker::new_owned();
    owned::tracker::model(&mut filter);
    filter.estimate_covariance_mut().set_at(0, 0, 42.0);
    owned::tracker::model(&mut filter);
    assert_eq!(filter.estimate_covariance().get_at(0, 0), 10.0);
}

#[test]
fn extern_buffers_are_used_in_place() {
    use external::{Gps, Push, gps, push, tracker};

    // Stale contents: whole-matrix statements must overwrite every entry of an extern buffer.
    let mut p = [7.0_f32; 4];
    let mut x = [0.0_f32; 2];
    let mut h = [9.0_f32; 2];
    let mut z = [0.0_f32; 1];
    let mut u = [0.0_f32; 1];
    let (mut filter_storage, mut gps_storage, mut push_storage) =
        (tracker::new(), gps::new(), push::new());

    {
        let mut filter = tracker::build(&mut filter_storage, &mut x, &mut p);
        let mut observation = gps::build(&mut gps_storage, &mut h, &mut z);
        let mut control = push::build(&mut push_storage, &mut u);
        tracker::model(&mut filter);
        gps::model(&mut observation);
        push::model(&mut control);
    }

    // model() wrote the declared values into the caller's arrays, replacing the stale ones.
    assert_eq!(p, [10.0, 0.0, 0.0, 10.0]);
    assert_eq!(h, [1.0, 0.0]);

    for k in 0..STEPS {
        u = [1.0];
        z = [fix(k)];
        let mut filter = tracker::build(&mut filter_storage, &mut x, &mut p);
        let mut observation = gps::build(&mut gps_storage, &mut h, &mut z);
        let mut control = push::build(&mut push_storage, &mut u);
        filter.predict().control(&mut control);
        filter.correct(&mut observation);
        let _ = (Gps::N, Push::N);
    }

    // The state lives in the caller's `x`, bit-identical to the inline filter.
    assert_eq!(x, run_inline());
    assert!(
        p[0] < 10.0,
        "the covariance shrank in the caller's `P`: {p:?}"
    );
}

#[test]
fn a_lean_observation_with_an_extern_r_shares_the_scratch() {
    use external::{lean_gps, shared};

    let mut r = [0.0_f32; 1];
    let (mut filter_storage, mut gps_storage, mut scratch) =
        (inline::tracker::new(), lean_gps::new(), shared::new());
    let mut filter = inline::tracker::build(&mut filter_storage);
    let mut observation = lean_gps::build(&mut gps_storage, &mut r);
    inline::tracker::model(&mut filter);
    lean_gps::model(&mut observation);

    filter.predict();
    external::LeanGps { pos: 1.0 }.write(observation.measurement_vector_mut().as_mut_slice());
    assert!(
        filter
            .correct_in(&mut observation, &mut scratch)
            .is_applied()
    );
    assert_eq!(r, [0.5]);
}

#[test]
fn the_footprint_splits_inline_and_extern() {
    use core::mem::size_of;
    use minikalman::regular::{ControlStorage, FilterStorage, ObservationStorage};

    // P (4) and x (2) are extern; A, Q and the two temporaries (4 + 4 + 2 + 4) are inline.
    assert_eq!(external::tracker::BYTES, 14 * 4);
    assert_eq!(external::tracker::EXTERN_BYTES, 6 * 4);
    assert_eq!(
        size_of::<external::tracker::Storage>(),
        external::tracker::BYTES
    );
    assert_eq!(
        external::tracker::BYTES + external::tracker::EXTERN_BYTES,
        size_of::<FilterStorage<2, f32>>()
    );
    assert_eq!(
        external::gps::BYTES + external::gps::EXTERN_BYTES,
        size_of::<ObservationStorage<2, 1, f32>>()
    );
    assert_eq!(external::gps::EXTERN_BYTES, 3 * 4);
    assert_eq!(
        external::push::BYTES + external::push::EXTERN_BYTES,
        size_of::<ControlStorage<2, 1, f32>>()
    );
    assert_eq!(external::lean_gps::EXTERN_BYTES, 4);
    assert_eq!(inline::tracker::BYTES, size_of::<FilterStorage<2, f32>>());
    assert_eq!(inline::tracker::EXTERN_BYTES, 0);
    assert_eq!(inline::shared::BYTES, size_of::<inline::shared::Storage>());
    assert_eq!(
        external::observer::BYTES,
        size_of::<external::observer::Storage>()
    );
    assert_eq!(external::observer::EXTERN_BYTES, 0);
}

#[test]
fn an_observer_takes_h_from_an_extern_h_declaration() {
    let mut storage = external::observer::new();
    let mut observer = external::observer::build(&mut storage);
    external::observer::model(&mut observer);
    assert_eq!(observer.gps.observation_matrix().as_slice(), &[1.0, 0.0]);
    assert_eq!(observer.gps.gain().as_slice(), &[0.5, 0.8]);
}

/// The same items, each matrix in its own heap allocation.
#[cfg(feature = "alloc")]
mod boxed {
    use super::DT;
    pub use super::inline::Track;

    minikalman::filter! {
        pub control push: Push for Track { accel } {
            storage = boxed;
            B[pos, accel] = 0.5 * DT * DT;
            B[vel, accel] = DT;
            Q = diag(0.01);
        }
        pub observation gps: Gps for Track { pos } { storage = boxed; H = [pos]; R = diag(0.5); }
        pub kalman tracker for Track {
            storage = boxed;
            A = identity;
            A[pos, vel] = DT;
            Q = diag(0.01);
            P = diag(10.0);
        }
    }
}

#[cfg(feature = "alloc")]
#[test]
fn boxed_matches_inline_bit_for_bit() {
    use boxed::{Gps, Push, gps, push, tracker};

    let mut filter = tracker::new_boxed();
    let mut observation = gps::new_boxed();
    let mut control = push::new_boxed();
    tracker::model(&mut filter);
    gps::model(&mut observation);
    push::model(&mut control);

    for k in 0..STEPS {
        Push { accel: 1.0 }.write(control.control_vector_mut().as_mut_slice());
        filter.predict().control(&mut control);
        Gps { pos: fix(k) }.write(observation.measurement_vector_mut().as_mut_slice());
        filter.correct(&mut observation);
    }

    let x = filter.state_vector().as_slice();
    assert_eq!([x[0], x[1]], run_inline());
    assert_eq!(
        tracker::BYTES,
        core::mem::size_of::<inline::tracker::Storage>()
    );
    assert_eq!(gps::BYTES, core::mem::size_of::<inline::gps::Storage>());
    assert_eq!(push::BYTES, core::mem::size_of::<inline::push::Storage>());
    assert_eq!(
        tracker::EXTERN_BYTES + gps::EXTERN_BYTES + push::EXTERN_BYTES,
        0
    );
}
