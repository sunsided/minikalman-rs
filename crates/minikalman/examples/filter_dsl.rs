//! Two Kalman filters declared with `filter!`, sharing one observation and one control.
//!
//! A cart on a rail is pushed with a known acceleration and its position is measured with a
//! variance of `0.25 m²`. A responsive filter (acceleration variance `1 (m/s²)²`) and a smooth
//! one (`0.01 (m/s²)²`) run side by side on the same readings. Both are declared in one
//! `filter!` block, live in one `system` storage of compile-time-fixed size, and share one
//! `gps` observation, one `throttle` control and their declarations. The steady-state gain of
//! the responsive filter then drives a fixed-gain observer declared in the same block.

#![forbid(unsafe_code)]

use minikalman::models::motion::constant_velocity;
use minikalman::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Time step in seconds.
const DT: f32 = 0.1;

/// Number of simulated steps.
const STEPS: usize = 100;

/// The commanded acceleration, in m/s².
const ACCELERATION: f32 = 0.5;

minikalman::filter! {
    /// Position (m) and velocity (m/s) of the cart.
    pub state Cart(f32) { pos, vel }

    /// The commanded acceleration, in m/s².
    pub control throttle: Throttle for Cart { accel } {
        B[pos, accel] = 0.5 * DT * DT;
        B[vel, accel] = DT;
        Q = diag(0.01);
    }

    /// A position fix with a variance of 0.25 m².
    pub observation gps: Gps for Cart { pos } {
        H = [pos];
        R = diag(0.25);
    }

    /// Trusts the measurements; reacts quickly.
    pub kalman responsive for Cart {
        A = |a| constant_velocity(DT, 1.0).fill_transition(a);
        Q = |q| constant_velocity(DT, 1.0).fill_process_noise(q);
        P = diag(10.0);
    }

    /// Trusts the model; smooths.
    pub kalman smooth for Cart {
        A = |a| constant_velocity(DT, 0.01).fill_transition(a);
        Q = |q| constant_velocity(DT, 0.01).fill_process_noise(q);
        P = diag(10.0);
    }

    /// Runs the responsive filter's steady-state gain without a covariance.
    pub luenberger fixed for Cart {
        A = |a| constant_velocity(DT, 1.0).fill_transition(a);
    } gains {
        gps {}
    }

    /// Everything in one value.
    pub system rail { responsive, smooth, fixed, gps, throttle }
}

fn main() {
    let mut storage = rail::new();
    let mut sys = rail::build(&mut storage); // wiring only
    rail::model(&mut sys); // writes the declared A, Q, P, H, R, B and Qᵤ
    println!(
        "One system storage: {} bytes, no allocation",
        core::mem::size_of::<rail::Storage>()
    );

    design_fixed_gain(&mut sys);

    let mut rng = StdRng::seed_from_u64(42);
    let noise = Normal::new(0.0, 0.5).expect("a valid standard deviation");
    let mut truth = Cart { pos: 0.0, vel: 0.0 };

    println!("\n  step   true pos  responsive    smooth     fixed");
    for step in 0..STEPS {
        truth = advance(truth);
        Throttle {
            accel: ACCELERATION,
        }
        .write(sys.throttle.control_vector_mut().as_mut_slice());
        let z = Gps {
            pos: truth.pos + noise.sample(&mut rng),
        };

        // Same control object and same observation object for both filters.
        for filter in [&mut sys.responsive, &mut sys.smooth] {
            filter.predict().control(&mut sys.throttle);
            z.write(sys.gps.measurement_vector_mut().as_mut_slice());
            filter.correct(&mut sys.gps);
        }

        sys.fixed.observer.predict().control(&mut sys.throttle);
        z.write(sys.fixed.gps.measurement_vector_mut().as_mut_slice());
        assert!(sys.fixed.observer.correct(&mut sys.fixed.gps).is_applied());

        if step % 10 == 9 {
            print_row(step, truth, &sys);
        }
    }

    let smooth = Cart::read(sys.smooth.state_vector().as_slice());
    assert!(
        (smooth.vel - truth.vel).abs() < 0.5,
        "the smooth filter tracks the velocity"
    );
}

/// Settles the responsive filter's gain on a copy of its storage and hands it to the observer.
fn design_fixed_gain(sys: &mut rail::Handle<'_>) {
    let mut design_storage = responsive::new();
    let mut design = responsive::build(&mut design_storage);
    responsive::model(&mut design);

    let iterations = design
        .settle_gain(&mut sys.gps, 1e-6, 10_000)
        .expect("the gain settles");
    sys.fixed.gps.set_gain(sys.gps.kalman_gain());
    println!(
        "Fixed gain K∞ = [{:.4}, {:.4}] after {iterations} iterations",
        sys.gps.kalman_gain().get_at(Cart::pos, Gps::pos),
        sys.gps.kalman_gain().get_at(Cart::vel, Gps::pos),
    );
}

/// The true cart one step later, under the commanded acceleration.
fn advance(cart: Cart) -> Cart {
    Cart {
        pos: cart.pos + cart.vel * DT + 0.5 * ACCELERATION * DT * DT,
        vel: cart.vel + ACCELERATION * DT,
    }
}

/// Prints the true position and the three estimates.
fn print_row(step: usize, truth: Cart, sys: &rail::Handle<'_>) {
    let pos = |x: &[f32]| Cart::read(x).pos;
    println!(
        "{:6} {:10.3} {:11.3} {:9.3} {:9.3}",
        step + 1,
        truth.pos,
        pos(sys.responsive.state_vector().as_slice()),
        pos(sys.smooth.state_vector().as_slice()),
        pos(sys.fixed.observer.state_vector().as_slice()),
    );
}
