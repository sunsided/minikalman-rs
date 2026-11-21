//! The opt-in Joseph-form covariance update of the regular and extended corrections,
//! compared to the standard update on well-conditioned problems.

#![cfg(feature = "alloc")]
#![forbid(unsafe_code)]
// The comparisons need an exact square root; see `regular_joseph_matches_standard`.
#![cfg_attr(not(any(feature = "std", feature = "libm")), allow(dead_code))]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;

/// Time step of the constant-velocity test system, in seconds.
const DT: f64 = 0.1;

/// Relative tolerance between the standard and the Joseph update in `f64`: both are exact
/// for the optimal gain, so they differ by rounding only: about five ULP of 1.
const REL: f64 = 1e-15;

/// Asserts that `a` and `b` agree to [`REL`], relative to the larger magnitude (at least 1).
fn assert_close(a: f64, b: f64, what: &str) {
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!((a - b).abs() <= REL * scale, "{what}: {a} != {b}");
}

/// A deterministic position reading at step `t` of a target moving at 2 units/s.
fn reading(t: usize) -> f64 {
    let t = t as f64;
    2.0 * DT * t + 0.3 * (1.7 * t).sin()
}

/// The default is the standard update, and the setter round-trips, on both observations.
#[test]
fn default_is_standard() {
    let mut regular = RegularBuilder::<2, f64>::default()
        .observations()
        .build::<1>();
    let mut extended = ExtendedBuilder::<2, f64>::default()
        .observations()
        .build::<1>();
    assert_eq!(CovarianceUpdate::default(), CovarianceUpdate::Standard);
    assert_eq!(regular.covariance_update(), CovarianceUpdate::Standard);
    assert_eq!(extended.covariance_update(), CovarianceUpdate::Standard);

    regular.set_covariance_update(CovarianceUpdate::Joseph);
    extended.set_covariance_update(CovarianceUpdate::Joseph);
    assert_eq!(regular.covariance_update(), CovarianceUpdate::Joseph);
    assert_eq!(extended.covariance_update(), CovarianceUpdate::Joseph);
}

/// Runs a constant-velocity filter (position and velocity, position sensor) for 200 steps with
/// `update`, optionally adapting `R`, exercising the `*_with` variants on odd steps. Returns the
/// final state, the final `P` in row-major order and the final `R`.
fn regular_run(update: CovarianceUpdate, adapt: bool) -> ([f64; 2], [f64; 4], f64) {
    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut z = builder.observations().build::<1>();

    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 1, DT);
    });
    filter.direct_process_noise_mut().apply(|q| {
        q.set_at(0, 0, 1e-4);
        q.set_at(1, 1, 1e-3);
    });
    filter.estimate_covariance_mut().make_scalar(10.0);
    z.observation_matrix_mut().set_at(0, 0, 1.0);
    z.measurement_noise_covariance_mut().make_scalar(0.25);
    z.set_covariance_update(update);
    if adapt {
        z.set_noise_adaptation(Some(NoiseAdaptation {
            forgetting: 0.95,
            floor: 1e-3,
        }));
    }

    for t in 0..200 {
        filter.predict();
        z.measurement_vector_mut().set_at(0, 0, reading(t));
        let status = if t % 2 == 0 {
            filter.correct(&mut z)
        } else {
            filter.correct_with(&mut z, |_| {})
        };
        assert_eq!(status, CorrectionStatus::Applied);
    }

    let x = filter.state_vector();
    let p = filter.estimate_covariance();
    (
        [x.get_at(0, 0), x.get_at(1, 0)],
        [
            p.get_at(0, 0),
            p.get_at(0, 1),
            p.get_at(1, 0),
            p.get_at(1, 1),
        ],
        z.measurement_noise_covariance().get_at(0, 0),
    )
}

/// On a well-conditioned linear problem, the Joseph form matches the standard update, with
/// and without adapting `R`, and keeps `P` exactly symmetric.
///
/// Gated on an exact square root: micromath's approximate one makes the gain suboptimal, and
/// for a suboptimal gain the two forms differ.
#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn regular_joseph_matches_standard() {
    for adapt in [false, true] {
        let (xs, ps, rs) = regular_run(CovarianceUpdate::Standard, adapt);
        let (xj, pj, rj) = regular_run(CovarianceUpdate::Joseph, adapt);
        for i in 0..2 {
            assert_close(xs[i], xj[i], "x");
        }

        for i in 0..4 {
            assert_close(ps[i], pj[i], "P");
        }

        assert_close(rs, rj, "R");
        assert_eq!(pj[1], pj[2], "Joseph P is symmetric");
    }
}

/// The observation function `h(x) = x₀² + x₁` of the extended test system.
fn observe(x: &impl Matrix<2, 1, f64>, z: &mut impl MatrixMut<1, 1, f64>) {
    z.set_at(0, 0, x.get_at(0, 0) * x.get_at(0, 0) + x.get_at(1, 0));
}

/// Runs an extended filter with `h(x) = x₀² + x₁` for 50 steps with `update`, exercising the
/// `*_with` variants on odd steps. Returns the final state and `P` in row-major order.
fn extended_run(update: CovarianceUpdate) -> ([f64; 2], [f64; 4]) {
    let builder = ExtendedBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut z = builder.observations().build::<1>();

    filter.state_vector_mut().set_at(0, 0, 1.0);
    filter.estimate_covariance_mut().make_scalar(1.0);
    filter.direct_process_noise_mut().make_scalar(1e-3);
    filter.state_transition_jacobian_mut().make_identity();
    z.measurement_noise_covariance_mut().make_scalar(0.1);
    z.set_covariance_update(update);

    for t in 0..50 {
        filter.predict_nonlinear(|current, next| {
            next[0] = current[0];
            next[1] = current[1];
        });
        let x0 = filter.state_vector().get_at(0, 0);
        z.observation_jacobian_matrix_mut().apply(|h| {
            h.set_at(0, 0, 2.0 * x0);
            h.set_at(0, 1, 1.0);
        });
        z.measurement_vector_mut()
            .set_at(0, 0, 4.5 + 0.2 * (t as f64).sin());
        let status = if t % 2 == 0 {
            filter.correct_nonlinear(&mut z, observe)
        } else {
            filter.correct_nonlinear_with(&mut z, observe, |_| {})
        };
        assert_eq!(status, CorrectionStatus::Applied);
    }

    let x = filter.state_vector();
    let p = filter.estimate_covariance();
    (
        [x.get_at(0, 0), x.get_at(1, 0)],
        [
            p.get_at(0, 0),
            p.get_at(0, 1),
            p.get_at(1, 0),
            p.get_at(1, 1),
        ],
    )
}

/// The extended corrections with the Joseph form match the standard update.
///
/// Gated on an exact square root: micromath's approximate one makes the gain suboptimal, and
/// for a suboptimal gain the two forms differ.
#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn extended_joseph_matches_standard() {
    let (xs, ps) = extended_run(CovarianceUpdate::Standard);
    let (xj, pj) = extended_run(CovarianceUpdate::Joseph);
    for i in 0..2 {
        assert_close(xs[i], xj[i], "x");
    }

    for i in 0..4 {
        assert_close(ps[i], pj[i], "P");
    }

    assert_eq!(pj[1], pj[2], "Joseph P is symmetric");
}

/// A gated correction leaves `x` and `P` unchanged with the Joseph form too.
#[test]
fn joseph_respects_the_gate() {
    let builder = RegularBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut z = builder.observations().build::<1>();

    filter.estimate_covariance_mut().make_scalar(1.0);
    z.observation_matrix_mut().set_at(0, 0, 1.0);
    z.measurement_noise_covariance_mut().make_scalar(1.0);
    z.set_covariance_update(CovarianceUpdate::Joseph);
    z.set_gate(Some(9.0));
    z.measurement_vector_mut().set_at(0, 0, 100.0);

    assert_eq!(filter.correct(&mut z), CorrectionStatus::Gated);
    assert_eq!(filter.state_vector().get_at(0, 0), 0.0);
    assert_eq!(filter.estimate_covariance().get_at(0, 0), 1.0);
}

/// In `I16F16`, the Joseph form runs a constant-velocity filter and keeps `P` exactly
/// symmetric and its diagonal positive.
#[cfg(feature = "fixed")]
#[test]
fn fixed_point_joseph_smoke() {
    use fixed::types::I16F16;

    let builder = RegularBuilder::<2, I16F16>::default();
    let mut filter = builder.build();
    let mut z = builder.observations().build::<1>();

    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 1, I16F16::from_num(DT));
    });
    filter.direct_process_noise_mut().apply(|q| {
        q.set_at(0, 0, I16F16::from_num(1e-3));
        q.set_at(1, 1, I16F16::from_num(1e-2));
    });
    filter
        .estimate_covariance_mut()
        .make_scalar(I16F16::from_num(10));
    z.observation_matrix_mut().set_at(0, 0, I16F16::ONE);
    z.measurement_noise_covariance_mut()
        .make_scalar(I16F16::from_num(0.25));
    z.set_covariance_update(CovarianceUpdate::Joseph);

    for t in 0..200 {
        filter.predict();
        z.measurement_vector_mut()
            .set_at(0, 0, I16F16::from_num(reading(t)));
        assert_eq!(filter.correct(&mut z), CorrectionStatus::Applied);
        let p = filter.estimate_covariance();
        assert_eq!(p.get_at(0, 1), p.get_at(1, 0));
        assert!(p.get_at(0, 0) > 0 && p.get_at(1, 1) > 0);
    }

    let position = filter.state_vector().get_at(0, 0).to_num::<f64>();
    assert!(
        (position - 2.0 * DT * 199.0).abs() < 1.0,
        "position {position}"
    );
}
