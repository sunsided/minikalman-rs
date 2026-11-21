//! Analytic identities: results that must agree for mathematical reasons, checked in `f64`.
//!
//! * The EKF and the UKF reduce to the Kalman filter on a linear model (the UKF only
//!   without process noise, see the README).
//! * The information filter is the Kalman filter in another parametrization.
//! * The RTS smoother equals the batch least-squares smoother of the whole record.
//! * The filter converges to the steady state of the discrete algebraic Riccati equation.
//! * A scalar random walk has closed forms for one step and for the steady state.

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;

use crate::reference::cv;
use crate::scenarios::{self, History, Linear, CV, CV_NOISELESS, GRAVITY};
use crate::support::{assert_close, load, read, Tolerance};

/// Agreement required between two `f64` implementations of the same math.
const IDENTITY: Tolerance = if crate::support::APPROXIMATE_MATH {
    Tolerance::new(0.3, 0.1)
} else {
    Tolerance::new(1e-12, 1e-11)
};

/// Asserts that two runs recorded the same filtered states and covariances.
fn assert_same_posterior(what: &str, actual: &History, expected: &History) {
    assert_close(
        &format!("{what} x_k|k"),
        &actual.x_post,
        &expected.x_post,
        IDENTITY,
    );
    assert_close(
        &format!("{what} P_k|k"),
        &actual.p_post,
        &expected.p_post,
        IDENTITY,
    );
}

/// Runs the extended filter on a linear scenario, with `f(x) = A×x` and `h(x) = H×x`.
fn run_extended_linear<const N: usize, const M: usize>(s: &Linear) -> History {
    let builder = ExtendedBuilder::<N, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<M>();
    filter
        .state_transition_jacobian_mut()
        .apply(|a| load(a, s.a));
    filter.direct_process_noise_mut().apply(|q| load(q, s.q));
    filter.state_vector_mut().apply(|x| load(x, s.x0));
    filter.estimate_covariance_mut().apply(|p| load(p, s.p0));
    measurement
        .observation_jacobian_matrix_mut()
        .apply(|h| load(h, s.h));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, s.r));

    let mut history = History::default();
    for z in s.z.chunks_exact(M) {
        filter.predict_nonlinear(|state, next| {
            for i in 0..N {
                next[i] = (0..N).map(|j| s.a[i * N + j] * state[j]).sum();
            }
        });
        measurement.measurement_vector_mut().apply(|v| load(v, z));
        filter.correct_nonlinear(&mut measurement, |state, observation| {
            for i in 0..M {
                observation.set_row(i, (0..N).map(|j| s.h[i * N + j] * state[j]).sum());
            }
        });
        history.x_post.extend(read(filter.state_vector()));
        history.p_post.extend(read(filter.estimate_covariance()));
    }

    history
}

/// Runs the unscented filter (`S = 2N + 1` sigma points) on a linear scenario.
fn run_unscented_linear<const N: usize, const M: usize, const S: usize>(s: &Linear) -> History {
    let builder = UnscentedBuilder::<N, f64>::default();
    let mut filter = builder.build::<S>();
    let mut measurement = builder.observations().build::<M, S>();
    filter.direct_process_noise_mut().apply(|q| load(q, s.q));
    filter.state_vector_mut().apply(|x| load(x, s.x0));
    filter.estimate_covariance_mut().apply(|p| load(p, s.p0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, s.r));

    let mut history = History::default();
    for z in s.z.chunks_exact(M) {
        filter.predict_nonlinear(|next| {
            let x = read(next.as_matrix());
            for i in 0..N {
                let value = (0..N).map(|j| s.a[i * N + j] * x[j]).sum();
                next.as_matrix_mut().set(i, 0, value);
            }
        });
        measurement.measurement_vector_mut().apply(|v| load(v, z));
        filter.correct_sigma_point(&mut measurement, |sigma, observed| {
            // `sigma` is row-major `N × S`: entry `j × S + k` is state `j` of point `k`.
            for k in 0..S {
                for i in 0..M {
                    let value = (0..N).map(|j| s.h[i * N + j] * sigma[j * S + k]).sum();
                    observed.set(i, k, value);
                }
            }
        });
        history.x_post.extend(read(filter.state_vector()));
        history.p_post.extend(read(filter.estimate_covariance()));
    }

    history
}

/// With a linear `f` and `h`, the EKF computes exactly the KF.
#[test]
fn extended_equals_regular_on_linear_models() {
    assert_same_posterior(
        "ekf = kf, cv",
        &run_extended_linear::<2, 1>(&CV),
        &scenarios::run_regular::<2, 1, f64>(&CV),
    );
    assert_same_posterior(
        "ekf = kf, gravity",
        &run_extended_linear::<3, 1>(&GRAVITY),
        &scenarios::run_regular::<3, 1, f64>(&GRAVITY),
    );
}

/// With a linear `f` and `h` and `Q = 0`, the UKF computes the KF.
#[test]
fn unscented_equals_regular_on_linear_model_without_process_noise() {
    assert_same_posterior(
        "ukf = kf, cv, Q = 0",
        &run_unscented_linear::<2, 1, 5>(&CV_NOISELESS),
        &scenarios::run_regular::<2, 1, f64>(&CV_NOISELESS),
    );
}

/// The information filter recovers the KF's `x` and `P` at every step.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "micromath's approximate recip compounds over the inversions"
)]
fn information_equals_regular() {
    assert_same_posterior(
        "information = kf, cv",
        &scenarios::run_information::<2, 1, f64>(&CV),
        &scenarios::run_regular::<2, 1, f64>(&CV),
    );
    assert_same_posterior(
        "information = kf, gravity",
        &scenarios::run_information::<3, 1, f64>(&GRAVITY),
        &scenarios::run_regular::<3, 1, f64>(&GRAVITY),
    );
}

/// The RTS smoother equals the batch least-squares smoother of the whole record.
#[test]
fn rts_equals_batch_least_squares() {
    let history = scenarios::run_regular::<2, 1, f64>(&CV);
    let (x, p) = scenarios::smooth::<2, f64>(&CV, &history);
    // The batch solution inverts a 102 × 102 information matrix, which costs it about 1e-12.
    let batch = Tolerance::new(IDENTITY.abs.max(1e-10), IDENTITY.rel);
    assert_close("rts = batch x_k|N", &x, &cv::X_BATCH, batch);
    assert_close("rts = batch P_k|N", &p, &cv::P_BATCH, batch);
}

/// Steps after which the constant-velocity filter has reached its steady state in `f64`.
const SETTLE_STEPS: usize = 500;

/// The KF converges to the steady state of SciPy's `solve_discrete_are`.
#[test]
fn converges_to_the_riccati_steady_state() {
    let (mut filter, mut measurement) = scenarios::regular::<2, 1, f64>(&CV);
    let mut prior = Vec::new();
    for _ in 0..SETTLE_STEPS {
        filter.predict();
        prior = read(filter.estimate_covariance());
        filter.correct(&mut measurement);
    }

    let gain = read(measurement.kalman_gain());
    let posterior = read(filter.estimate_covariance());
    assert_close("steady P_k|k-1", &prior, &cv::P_STEADY_PRIOR, IDENTITY);
    assert_close("steady P_k|k", &posterior, &cv::P_STEADY_POST, IDENTITY);
    assert_close("steady K", &gain, &cv::K_STEADY, IDENTITY);
}

/// A scalar random walk `x_k = x_k−1 + w`, `z_k = x_k + v`, with `Q = 0.1` and `R = 1`.
const WALK: Linear = Linear {
    a: &[1.0],
    q: &[0.1],
    h: &[1.0],
    r: &[1.0],
    x0: &[0.0],
    p0: &[2.0],
    z: &[1.0],
};

/// One step of a scalar random walk matches its closed form in the KF, EKF and information filter.
#[test]
fn scalar_random_walk_one_step_closed_form() {
    let (q, r, p0, z) = (WALK.q[0], WALK.r[0], WALK.p0[0], WALK.z[0]);
    let prior = p0 + q;
    let gain = prior / (prior + r);
    let expected = History {
        x_post: vec![gain * z],
        p_post: vec![(1.0 - gain) * prior],
        ..History::default()
    };

    assert_same_posterior(
        "walk kf",
        &scenarios::run_regular::<1, 1, f64>(&WALK),
        &expected,
    );
    assert_same_posterior("walk ekf", &run_extended_linear::<1, 1>(&WALK), &expected);
    assert_same_posterior(
        "walk information",
        &scenarios::run_information::<1, 1, f64>(&WALK),
        &expected,
    );
}

/// A scalar random walk converges to the closed-form steady-state gain and variance.
#[test]
fn scalar_random_walk_steady_state_closed_form() {
    let (q, r) = (WALK.q[0], WALK.r[0]);
    // The prior variance solves p = p·r / (p + r) + q.
    let prior = (q + (q * q + 4.0 * q * r).sqrt()) / 2.0;
    let gain = prior / (prior + r);

    let (mut filter, mut measurement) = scenarios::regular::<1, 1, f64>(&WALK);
    for _ in 0..SETTLE_STEPS {
        filter.predict();
        filter.correct(&mut measurement);
    }

    assert_close(
        "walk K",
        &read(measurement.kalman_gain()),
        &[gain],
        IDENTITY,
    );
    assert_close(
        "walk P_k|k",
        &read(filter.estimate_covariance()),
        &[(1.0 - gain) * prior],
        IDENTITY,
    );
}
