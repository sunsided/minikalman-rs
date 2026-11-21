//! The time-step handle returned by `predict` applies controls exactly like the standalone
//! `control` calls it replaces: `x ← A·x + B₁·u₁ + B₂·u₂` and
//! `P ← A·P·Aᵀ + Q + B₁·Qᵤ₁·B₁ᵀ + B₂·Qᵤ₂·B₂ᵀ`, bit for bit.

use minikalman::prelude::*;
use minikalman::regular::{ControlStorage, FilterStorage};

/// Runs five steps of a constant-velocity model with a throttle (`B₁ = [½, 1]ᵀ`) and a
/// disturbance (`B₂ = [1, 0]ᵀ`), each with its own `u` and `Qᵤ`, and returns the final `x`
/// and `P`.
///
/// `$v` converts an `f32` literal to `$t`. The body performs one time update on the filter
/// and the two controls, bound to the three given names.
macro_rules! run {
    ($t:ty, $v:expr, |$filter:ident, $throttle:ident, $steering:ident| $body:block) => {{
        let v = $v;
        let mut filter_storage = FilterStorage::<2, $t>::new(v(0.0));
        let mut throttle_storage = ControlStorage::<2, 1, $t>::new(v(0.0));
        let mut steering_storage = ControlStorage::<2, 1, $t>::new(v(0.0));
        let mut $filter = filter_storage.build();
        let mut $throttle = throttle_storage.build();
        let mut $steering = steering_storage.build();

        $filter.state_transition_mut().apply(|a| {
            a.set_at(0, 0, v(1.0));
            a.set_at(0, 1, v(1.0));
            a.set_at(1, 1, v(1.0));
        });
        $filter.estimate_covariance_mut().make_identity();
        $filter.direct_process_noise_mut().make_scalar(v(0.125));
        $throttle.control_matrix_mut().apply(|b| {
            b.set_at(0, 0, v(0.5));
            b.set_at(1, 0, v(1.0));
        });
        $throttle.control_vector_mut().apply(|u| u[0] = v(0.75));
        $throttle.process_noise_covariance_mut().make_scalar(v(0.25));
        $steering.control_matrix_mut().apply(|b| b.set_at(0, 0, v(1.0)));
        $steering.control_vector_mut().apply(|u| u[0] = v(-0.5));
        $steering.process_noise_covariance_mut().make_scalar(v(0.5));

        for _ in 0..5 $body

        let mut x = [v(0.0); 2];
        let mut p = [v(0.0); 4];
        x.copy_from_slice($filter.state_vector().as_slice());
        p.copy_from_slice($filter.estimate_covariance().as_slice());
        (x, p)
    }};
}

/// Returns the bit patterns of `x` and `P`, so that only identical results compare equal.
fn bits((x, p): ([f32; 2], [f32; 4])) -> ([u32; 2], [u32; 4]) {
    (x.map(f32::to_bits), p.map(f32::to_bits))
}

/// The identity conversion for the `f32` runs.
fn f(v: f32) -> f32 {
    v
}

#[test]
fn handle_matches_predict_then_two_controls() {
    let split = run!(f32, f, |filter, throttle, steering| {
        filter.predict();
        filter.control(&mut throttle);
        filter.control(&mut steering);
    });
    let handle = run!(f32, f, |filter, throttle, steering| {
        filter
            .predict()
            .control(&mut throttle)
            .control(&mut steering);
    });

    assert_eq!(bits(handle), bits(split));
}

#[test]
fn handle_matches_the_closed_form_state() {
    let (x, p) = run!(f32, f, |filter, throttle, steering| {
        filter
            .predict()
            .control(&mut throttle)
            .control(&mut steering);
    });

    // From x = 0, each step adds B₁·u₁ + B₂·u₂ = [-0.125, 0.75] to A·x, so the velocity is
    // 0.75·k and the position 0.75·(0 + 1 + 2 + 3 + 4) − 5·0.125 after five steps.
    assert_eq!(x, [6.875, 3.75]);
    assert!(p[0] > 0.0 && p[3] > 0.0, "P = {p:?}");
    assert_eq!(p[1].to_bits(), p[2].to_bits(), "P is not symmetric: {p:?}");
}

#[test]
fn tuned_handle_matches_predict_tuned_then_two_controls() {
    const LAMBDA: f32 = 0.875;

    let split = run!(f32, f, |filter, throttle, steering| {
        filter.predict_tuned(LAMBDA);
        filter.control(&mut throttle);
        filter.control(&mut steering);
    });
    let handle = run!(f32, f, |filter, throttle, steering| {
        filter
            .predict_tuned(LAMBDA)
            .control(&mut throttle)
            .control(&mut steering);
    });

    assert_eq!(bits(handle), bits(split));
}

#[test]
fn handle_held_across_statements_applies_each_control() {
    let split = run!(f32, f, |filter, throttle, steering| {
        filter.predict();
        filter.control(&mut throttle);
        filter.control(&mut steering);
    });
    let held = run!(f32, f, |filter, throttle, steering| {
        let mut step = filter.predict();
        step.control(&mut throttle);
        step.control(&mut steering);
    });

    assert_eq!(bits(held), bits(split));
}

/// Predicts through the [`KalmanFilterPredict`] trait, which still returns `()`.
fn predict_via_trait<K: KalmanFilterPredict<2, f32>>(filter: &mut K) {
    let () = filter.predict();
}

#[test]
fn trait_predict_matches_the_inherent_predict() {
    let via_trait = run!(f32, f, |filter, throttle, steering| {
        predict_via_trait(&mut filter);
        filter.control(&mut throttle);
        filter.control(&mut steering);
    });
    let inherent = run!(f32, f, |filter, throttle, steering| {
        filter
            .predict()
            .control(&mut throttle)
            .control(&mut steering);
    });

    assert_eq!(bits(via_trait), bits(inherent));
}

#[cfg(feature = "fixed")]
#[test]
fn fixed_point_handle_matches_predict_then_two_controls() {
    use minikalman::prelude::I16F16;

    let v = |x: f32| I16F16::from_num(x);
    let split = run!(I16F16, v, |filter, throttle, steering| {
        filter.predict();
        filter.control(&mut throttle);
        filter.control(&mut steering);
    });
    let handle = run!(I16F16, v, |filter, throttle, steering| {
        filter
            .predict()
            .control(&mut throttle)
            .control(&mut steering);
    });

    assert_eq!(handle, split);
    assert_eq!(handle.0, [v(6.875), v(3.75)]);
}
