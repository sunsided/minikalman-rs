//! The UD filter against the regular filter, its zero-pivot handling and its fixed-point
//! behavior.

#![forbid(unsafe_code)]
#![cfg(feature = "alloc")]
#![allow(non_snake_case)]

use minikalman::prelude::*;
use minikalman::regular::builder::{KalmanFilterBuilder, KalmanFilterType, UdKalmanType};

/// Absolute tolerance for comparing `f64` results of the two filter forms.
const TOLERANCE: f64 = 1e-9;

/// The filter under test: three states in `f64`.
type Ud3 = UdKalmanType<3, 6, f64>;

/// The reference filter: three states in `f64`.
type Regular3 = KalmanFilterType<3, f64>;

/// Sets a dense, non-triangular state transition.
fn set_transition(A: &mut impl MatrixMut<3, 3, f64>) {
    A.apply(|a| {
        a.set(0, 0, 1.0);
        a.set(0, 1, 0.1);
        a.set(0, 2, 0.005);
        a.set(1, 0, -0.02);
        a.set(1, 1, 0.98);
        a.set(1, 2, 0.1);
        a.set(2, 0, 0.01);
        a.set(2, 1, -0.03);
        a.set(2, 2, 0.95);
    });
}

/// Sets a correlated initial covariance.
fn set_covariance(P: &mut impl MatrixMut<3, 3, f64>) {
    P.apply(|p| {
        p.set(0, 0, 4.0);
        p.set_symmetric(0, 1, 0.8);
        p.set_symmetric(0, 2, -0.3);
        p.set(1, 1, 2.0);
        p.set_symmetric(1, 2, 0.4);
        p.set(2, 2, 1.0);
    });
}

/// Sets a correlated process noise.
fn set_noise(Q: &mut impl MatrixMut<3, 3, f64>) {
    Q.apply(|q| {
        q.set(0, 0, 0.02);
        q.set_symmetric(0, 1, 0.005);
        q.set(1, 1, 0.05);
        q.set_symmetric(1, 2, -0.01);
        q.set(2, 2, 0.03);
    });
}

/// Builds a regular and a UD filter with the same `A`, `x`, `P` and the given `Q`.
fn build_pair(set_q: impl Fn(&mut dyn FnMut(usize, usize, f64))) -> (Regular3, Ud3) {
    let builder = KalmanFilterBuilder::<3, f64>::default();
    let mut regular = builder.build();
    let mut ud = builder.build_ud::<6>();

    set_transition(regular.state_transition_mut());
    set_transition(ud.state_transition_mut());
    set_covariance(regular.estimate_covariance_mut());
    regular.state_vector_mut().apply(|x| {
        x.set(0, 0, 1.0);
        x.set(1, 0, -0.5);
        x.set(2, 0, 0.25);
    });

    let Q = regular.direct_process_noise_mut();
    set_q(&mut |row, col, value| Q.set(row, col, value));
    assert!(ud.set_process_noise(regular.direct_process_noise()));
    assert!(ud.set_state(regular.state_vector(), regular.estimate_covariance()));

    (regular, ud)
}

/// Asserts that `x` and `P = U·D·Uᵀ` of the UD filter match the regular filter.
fn assert_matches(what: &str, ud: &Ud3, regular: &Regular3) {
    let mut x = BufferBuilder::state_vector_x::<3>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<3>().new();
    ud.state(&mut x, &mut P);

    assert_close(what, x.as_slice(), regular.state_vector().as_slice());
    assert_close(what, P.as_slice(), regular.estimate_covariance().as_slice());
    for i in 0..3 {
        assert!(ud.ud_factors().get(i, i) >= 0.0, "{what}: D[{i}] < 0");
    }
}

/// Asserts that two buffers of the same length agree to within [`TOLERANCE`].
fn assert_close(what: &str, actual: &[f64], expected: &[f64]) {
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() < TOLERANCE,
            "{what}[{i}]: {a} != {e} (actual {actual:?}, expected {expected:?})"
        );
    }
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt in the regular filter's Cholesky breaks the tolerance"
)]
fn matches_regular_filter_with_correlated_noise() {
    let (mut regular, mut ud) = build_pair(|set| {
        let mut Q = BufferBuilder::direct_process_noise_covariance_Q::<3>().new();
        set_noise(&mut Q);
        for row in 0..3 {
            for col in 0..3 {
                set(row, col, Q.get(row, col));
            }
        }
    });

    let builder = KalmanFilterBuilder::<3, f64>::default();

    // A scalar sensor mixing two states.
    let mut first = builder.observations().build::<1>();
    first.observation_matrix_mut().apply(|h| {
        h.set(0, 0, 1.0);
        h.set(0, 2, 0.5);
    });
    first.measurement_noise_covariance_mut().set(0, 0, 0.5);

    // A two-channel sensor with correlated noise.
    let mut second = builder.observations().build::<2>();
    second.observation_matrix_mut().apply(|h| {
        h.set(0, 1, 1.0);
        h.set(1, 0, 0.7);
        h.set(1, 1, -0.4);
        h.set(1, 2, 1.0);
    });
    second.measurement_noise_covariance_mut().apply(|r| {
        r.set(0, 0, 1.0);
        r.set_symmetric(0, 1, 0.3);
        r.set(1, 1, 2.0);
    });

    let mut first_regular = builder.observations().build::<1>();
    let mut second_regular = builder.observations().build::<2>();
    first
        .observation_matrix()
        .copy_to(first_regular.observation_matrix_mut());
    first
        .measurement_noise_covariance()
        .copy_to(first_regular.measurement_noise_covariance_mut());
    second
        .observation_matrix()
        .copy_to(second_regular.observation_matrix_mut());
    second
        .measurement_noise_covariance()
        .copy_to(second_regular.measurement_noise_covariance_mut());

    for step in 0..20 {
        regular.predict();
        ud.predict();
        assert_matches(&format!("predict {step}"), &ud, &regular);

        let t = f64::from(step);
        let z1 = 1.0 + 0.3 * t.sin();
        let z2 = [0.2 * t.cos(), 1.5 - 0.1 * t];
        for obs in [&mut first, &mut first_regular] {
            obs.measurement_vector_mut().set(0, 0, z1);
        }

        for obs in [&mut second, &mut second_regular] {
            obs.measurement_vector_mut().apply(|z| {
                z.set(0, 0, z2[0]);
                z.set(1, 0, z2[1]);
            });
        }

        assert!(regular.correct(&mut first_regular).is_applied());
        assert!(ud.correct(&mut first).is_applied());
        assert!((first.nis() - first_regular.nis()).abs() < TOLERANCE);
        assert_close(
            "innovation",
            first.innovation_vector().as_slice(),
            first_regular.innovation_vector().as_slice(),
        );
        assert_matches(&format!("correct first {step}"), &ud, &regular);

        assert!(regular.correct(&mut second_regular).is_applied());
        assert!(ud.correct(&mut second).is_applied());
        assert!(
            (second.nis() - second_regular.nis()).abs() < TOLERANCE,
            "NIS {} != {}",
            second.nis(),
            second_regular.nis()
        );
        assert_matches(&format!("correct second {step}"), &ud, &regular);
    }
}

#[test]
fn predict_handles_rank_deficient_process_noise() {
    // Q = g·gᵀ with g = (0.1, -0.2, 0.3): rank 1, not axis-aligned.
    let g = [0.1, -0.2, 0.3];
    let (mut regular, mut ud) = build_pair(|set| {
        for (row, gr) in g.iter().enumerate() {
            for (col, gc) in g.iter().enumerate() {
                set(row, col, gr * gc);
            }
        }
    });

    for step in 0..10 {
        regular.predict();
        ud.predict();
        assert_matches(&format!("rank-1 Q, predict {step}"), &ud, &regular);
    }
}

#[test]
fn predict_handles_zero_covariance_and_noise() {
    let builder = KalmanFilterBuilder::<3, f64>::default();
    let mut ud = builder.build_ud::<6>();
    set_transition(ud.state_transition_mut());
    ud.state_vector_mut().set(0, 0, 2.0);

    // P = 0 and Q = 0: every MWGS pivot is zero.
    ud.predict();

    let ud_factors = ud.ud_factors().as_slice();
    assert!(ud_factors.iter().all(|v| *v == 0.0), "{ud_factors:?}");
    assert_eq!(ud.state_vector().get(1, 0), -0.04);
}

#[test]
fn noise_on_one_state_only() {
    let (mut regular, mut ud) = build_pair(|set| set(2, 2, 0.04));
    for step in 0..10 {
        regular.predict();
        ud.predict();
        assert_matches(&format!("diagonal rank-1 Q, predict {step}"), &ud, &regular);
    }
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt in the regular filter's Cholesky breaks the tolerance"
)]
fn gate_rejects_without_touching_the_filter() {
    let (mut regular, mut ud) = build_pair(|_| {});
    let builder = KalmanFilterBuilder::<3, f64>::default();
    let mut obs = builder.observations().build::<2>();
    let mut obs_regular = builder.observations().build::<2>();
    for o in [&mut obs, &mut obs_regular] {
        o.observation_matrix_mut().apply(|h| {
            h.set(0, 0, 1.0);
            h.set(1, 1, 1.0);
            h.set(1, 2, 0.5);
        });
        o.measurement_noise_covariance_mut().apply(|r| {
            r.set(0, 0, 0.5);
            r.set_symmetric(0, 1, 0.1);
            r.set(1, 1, 0.8);
        });
        o.measurement_vector_mut().apply(|z| {
            z.set(0, 0, 30.0);
            z.set(1, 0, -20.0);
        });
        o.set_gate(Some(9.0));
    }

    let before_x = ud.state_vector().as_slice().to_vec();
    let before_ud = ud.ud_factors().as_slice().to_vec();
    assert_eq!(ud.correct(&mut obs), CorrectionStatus::Gated);
    assert_eq!(regular.correct(&mut obs_regular), CorrectionStatus::Gated);
    assert!((obs.nis() - obs_regular.nis()).abs() < 1e-6);
    assert_eq!(ud.state_vector().as_slice(), before_x.as_slice());
    assert_eq!(ud.ud_factors().as_slice(), before_ud.as_slice());

    // A passing gate gives the same NIS as the sequential sum without a gate.
    for o in [&mut obs, &mut obs_regular] {
        o.measurement_vector_mut().apply(|z| {
            z.set(0, 0, 1.5);
            z.set(1, 0, 0.0);
        });
    }

    let (_, mut ungated) = build_pair(|_| {});
    let mut obs_ungated = builder.observations().build::<2>();
    obs.observation_matrix()
        .copy_to(obs_ungated.observation_matrix_mut());
    obs.measurement_noise_covariance()
        .copy_to(obs_ungated.measurement_noise_covariance_mut());
    obs.measurement_vector()
        .copy_to(obs_ungated.measurement_vector_mut());

    assert!(ud.correct(&mut obs).is_applied());
    assert!(regular.correct(&mut obs_regular).is_applied());
    assert!(ungated.correct(&mut obs_ungated).is_applied());
    assert!((obs.nis() - obs_regular.nis()).abs() < TOLERANCE);
    assert!((obs.nis() - obs_ungated.nis()).abs() < TOLERANCE);
    assert_matches("gate passed", &ud, &regular);
}

#[test]
fn correct_rejects_singular_measurement_noise() {
    let (_, mut ud) = build_pair(|_| {});
    let builder = KalmanFilterBuilder::<3, f64>::default();
    let mut obs = builder.observations().build::<1>();
    obs.observation_matrix_mut().set(0, 0, 1.0);
    obs.measurement_vector_mut().set(0, 0, 1.0);

    let before = ud.ud_factors().as_slice().to_vec();
    assert_eq!(ud.correct(&mut obs), CorrectionStatus::NotPositiveDefinite);
    assert_eq!(ud.ud_factors().as_slice(), before.as_slice());

    // With a valid R, S can only be singular through invalid factors; the gate rejects that.
    obs.measurement_noise_covariance_mut().set(0, 0, 1.0);
    obs.set_gate(Some(1.0));
    ud.ud_factors_mut().set(0, 0, -10.0);
    let before = ud.ud_factors().as_slice().to_vec();
    assert_eq!(ud.correct(&mut obs), CorrectionStatus::NotPositiveDefinite);
    assert_eq!(ud.ud_factors().as_slice(), before.as_slice());
}

#[test]
fn set_state_reports_indefinite_covariance() {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut ud = builder.build_ud::<4>();
    let x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
    P.set(0, 0, 1.0);
    P.set(1, 1, -1.0);

    assert!(!ud.set_state(&x, &P));
    assert_eq!(ud.ud_factors().get(1, 1), 0.0);
    let mut Q = BufferBuilder::direct_process_noise_covariance_Q::<2>().new();
    Q.set(1, 1, -1.0);
    assert!(!ud.set_process_noise(&Q));
    assert_eq!(ud.process_noise_factors().get(1, 1), 0.0);

    // A diagonal Q can be written as its own factors.
    ud.process_noise_factors_mut().set(0, 0, 0.5);
    assert_eq!(ud.process_noise_factors().get(0, 0), 0.5);
}

#[test]
#[cfg(not(feature = "no_assert"))]
#[should_panic(expected = "TWO_STATES must be 2 × STATES")]
fn wrong_workspace_size_panics() {
    let _ = KalmanFilterBuilder::<2, f32>::default().build_ud::<3>();
}

#[cfg(feature = "fixed")]
mod fixed_point {
    //! The UD filter in `I16F16`, where the plain filter loses positive definiteness.

    use super::*;

    /// One `I16F16` constant from an `f64`.
    fn fx(value: f64) -> I16F16 {
        I16F16::from_num(value)
    }

    /// Returns whether a symmetric 2 × 2 matrix, given in `f64`, is positive semi-definite.
    fn is_psd(p00: f64, p01: f64, p11: f64) -> bool {
        p00 >= 0.0 && p11 >= 0.0 && p00 * p11 - p01 * p01 >= -1e-12
    }

    /// Runs a constant-velocity tracker with a very precise position sensor and returns
    /// whether each filter's covariance stayed positive semi-definite on every step.
    ///
    /// `dt = 0.01`, `P₀ = diag(100, 100)`, `Q = diag(0, 10⁻⁴)`, `R = 10⁻³`: the gain is close
    /// to one and `P − K·H·P` subtracts nearly equal numbers, which `I16F16` cannot resolve;
    /// the plain filter's `det P` turns negative after about 24 steps.
    fn run(steps: u32) -> (bool, bool) {
        let builder = KalmanFilterBuilder::<2, I16F16>::default();
        let mut regular = builder.build();
        let mut ud = builder.build_ud::<4>();
        let mut obs = builder.observations().build::<1>();

        regular.state_transition_mut().apply(|a| {
            a.set(0, 0, fx(1.0));
            a.set(0, 1, fx(0.01));
            a.set(1, 1, fx(1.0));
        });

        regular
            .state_transition()
            .as_matrix()
            .copy_to(ud.state_transition_mut().as_matrix_mut());
        regular.estimate_covariance_mut().set(0, 0, fx(100.0));
        regular.estimate_covariance_mut().set(1, 1, fx(100.0));
        regular.direct_process_noise_mut().set(1, 1, fx(1e-4));
        assert!(ud.set_state(regular.state_vector(), regular.estimate_covariance()));
        assert!(ud.set_process_noise(regular.direct_process_noise()));

        obs.observation_matrix_mut().set(0, 0, fx(1.0));
        obs.measurement_noise_covariance_mut().set(0, 0, fx(1e-3));

        let mut regular_psd = true;
        let mut ud_psd = true;
        let mut x = BufferBuilder::state_vector_x::<2>().new();
        let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
        for step in 0..steps {
            // Bounded motion: a sine of amplitude 10.
            let z = fx(10.0 * (f64::from(step) * 0.01).sin());
            obs.measurement_vector_mut().set(0, 0, z);

            regular.predict();
            let _ = regular.correct(&mut obs);
            let p = regular.estimate_covariance();
            regular_psd &= is_psd(
                p.get(0, 0).to_num(),
                p.get(0, 1).to_num(),
                p.get(1, 1).to_num(),
            );

            ud.predict();
            assert!(ud.correct(&mut obs).is_applied(), "step {step}");
            let factors = ud.ud_factors();
            let (d0, u01, d1) = (
                factors.get(0, 0).to_num::<f64>(),
                factors.get(0, 1).to_num::<f64>(),
                factors.get(1, 1).to_num::<f64>(),
            );
            assert!(d0 >= 0.0 && d1 >= 0.0, "step {step}: D = ({d0}, {d1})");

            // U·D·Uᵀ in f64, so that the check adds no fixed-point rounding of its own.
            ud_psd &= is_psd(d0 + u01 * u01 * d1, u01 * d1, d1);

            ud.state(&mut x, &mut P);
            let position: f64 = x.get(0, 0).to_num();
            let error = (position - z.to_num::<f64>()).abs();
            // The constant-velocity model lags the sine by up to 0.3 in f64 as well.
            assert!(error < 0.5, "step {step}: x = {position}, z = {z}");
        }

        (regular_psd, ud_psd)
    }

    #[test]
    fn ud_stays_positive_semi_definite_where_plain_filter_fails() {
        let (regular_psd, ud_psd) = run(10_000);
        assert!(ud_psd, "UD covariance lost positive semi-definiteness");
        assert!(
            !regular_psd,
            "the plain filter was expected to fail in this scenario"
        );
    }
}
