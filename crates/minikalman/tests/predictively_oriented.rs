//! The predictively oriented (PrO) correction of scalar regular observations.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StudentT};

/// Newton budget used throughout.
const BUDGET: usize = 20;

/// A prior and a scalar measurement: state `x̃`, covariance `P̃` (row-major), `h`, `r` and `z`.
struct Problem<const N: usize> {
    /// Prior state vector.
    x: [f64; N],
    /// Prior covariance, row-major.
    p: [[f64; N]; N],
    /// Observation row.
    h: [f64; N],
    /// Measurement noise variance.
    r: f64,
    /// Measurement.
    z: f64,
}

/// The posterior state, covariance and the status of one correction.
struct Posterior<const N: usize> {
    /// Posterior state vector.
    x: [f64; N],
    /// Posterior covariance, row-major.
    p: [[f64; N]; N],
    /// What the correction did.
    status: CorrectionStatus,
}

/// Runs one PrO correction (or a regular one, if `pro` is `false`) on `problem`.
fn correct<const N: usize>(
    problem: &Problem<N>,
    pro: bool,
    max_iterations: usize,
    gate: Option<f64>,
) -> Posterior<N> {
    let builder = KalmanFilterBuilder::<N, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    for i in 0..N {
        filter.state_vector_mut().set_at(i, 0, problem.x[i]);
        measurement
            .observation_matrix_mut()
            .set_at(0, i, problem.h[i]);
        for (j, &value) in problem.p[i].iter().enumerate() {
            filter.estimate_covariance_mut().set_at(i, j, value);
        }
    }

    measurement
        .measurement_noise_covariance_mut()
        .set_at(0, 0, problem.r);
    measurement.measurement_vector_mut().set_at(0, 0, problem.z);
    measurement.set_gate(gate);

    let status = if pro {
        filter.correct_predictively_oriented(&mut measurement, max_iterations)
    } else {
        filter.correct(&mut measurement)
    };

    let mut posterior = Posterior {
        x: [0.0; N],
        p: [[0.0; N]; N],
        status,
    };
    for i in 0..N {
        posterior.x[i] = filter.state_vector().get_at(i, 0);
        for (j, value) in posterior.p[i].iter_mut().enumerate() {
            *value = filter.estimate_covariance().get_at(i, j);
        }
    }

    posterior
}

/// Returns `P̃·hᵀ` and `g = h·P̃·hᵀ` of `problem`.
fn prior_gain<const N: usize>(problem: &Problem<N>) -> ([f64; N], f64) {
    let mut pht = [0.0; N];
    for (i, row) in problem.p.iter().enumerate() {
        pht[i] = row.iter().zip(&problem.h).map(|(p, h)| p * h).sum();
    }

    let g = pht.iter().zip(&problem.h).map(|(p, h)| p * h).sum();
    (pht, g)
}

/// Returns `h·a`.
fn dot<const N: usize>(h: &[f64; N], a: &[f64; N]) -> f64 {
    h.iter().zip(a).map(|(h, a)| h * a).sum()
}

/// Returns `h·P·hᵀ`.
fn quadratic<const N: usize>(h: &[f64; N], p: &[[f64; N]; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        for j in 0..N {
            sum += h[i] * p[i][j] * h[j];
        }
    }

    sum
}

/// Draws a random problem with an innovation of `nu` prior standard deviations.
fn random_problem(rng: &mut StdRng, nu: f64) -> Problem<3> {
    let mut l = [[0.0; 3]; 3];
    for (i, row) in l.iter_mut().enumerate() {
        for value in row.iter_mut().take(i + 1) {
            *value = rng.gen_range(-1.0..1.0);
        }
    }

    let mut p = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            p[i][j] = (0..3).map(|k| l[i][k] * l[j][k]).sum::<f64>();
        }

        p[i][i] += 0.1;
    }

    let mut problem = Problem {
        x: [0.0; 3].map(|_| rng.gen_range(-5.0..5.0)),
        p,
        h: [0.0; 3].map(|_| rng.gen_range(-2.0..2.0)),
        r: [0.01, 1.0, 10.0][rng.gen_range(0..3)],
        z: 0.0,
    };
    let (_, g) = prior_gain(&problem);
    let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
    problem.z = dot(&problem.h, &problem.x) + sign * nu * (g + problem.r).sqrt();
    problem
}

/// The derivation's toy problem `P̃ = H = R = 1`, `z = 3` has the PrO posterior
/// `x ≈ 0.8567`, `P ≈ 1.5019`: larger than the prior, where the Kalman filter gives 1.5 and 0.5.
#[test]
fn toy_problem_inflates_past_the_prior() {
    let problem = Problem {
        x: [0.0],
        p: [[1.0]],
        h: [1.0],
        r: 1.0,
        z: 3.0,
    };

    let pro = correct(&problem, true, BUDGET, None);
    let kf = correct(&problem, false, BUDGET, None);

    assert_eq!(pro.status, CorrectionStatus::Applied);
    assert!((pro.x[0] - 0.8567).abs() < 1e-4, "x = {}", pro.x[0]);
    assert!((pro.p[0][0] - 1.5019).abs() < 1e-4, "P = {}", pro.p[0][0]);
    assert!((kf.x[0] - 1.5).abs() < 1e-12 && (kf.p[0][0] - 0.5).abs() < 1e-12);
}

/// Without innovation the state is unchanged, as in the Kalman filter, and `c = h·P·hᵀ` has
/// the closed form `(√(r² + 4gr) − r)/2`, with `P = P̃ − P̃hᵀhP̃·(g−c)/g²`.
#[test]
fn zero_innovation_matches_closed_form() {
    let mut problem = Problem {
        x: [1.0, -2.0],
        p: [[2.0, 0.5], [0.5, 1.0]],
        h: [1.0, 0.5],
        r: 0.3,
        z: 0.0,
    };
    problem.z = dot(&problem.h, &problem.x);

    let pro = correct(&problem, true, BUDGET, None);
    let (pht, g) = prior_gain(&problem);
    let r = problem.r;
    let c = ((r * r + 4.0 * g * r).sqrt() - r) / 2.0;

    assert_eq!(pro.status, CorrectionStatus::Applied);
    for i in 0..2 {
        assert!((pro.x[i] - problem.x[i]).abs() < 1e-12);
        for j in 0..2 {
            let expected = problem.p[i][j] - pht[i] * pht[j] * (g - c) / (g * g);
            assert!((pro.p[i][j] - expected).abs() < 1e-12, "P[{i}][{j}]");
        }
    }
}

/// For random problems, including very large innovations, the result satisfies the
/// stationarity equations of the PrO objective with the posterior residual
/// `v = z − h·x` and `S = h·P·hᵀ + r`:
///
/// - `x − x̃ = P̃hᵀ·v/S`
/// - `P⁻¹ = P̃⁻¹ + hᵀ·w·h` with `w = 1/S − v²/S²`, i.e. `P̃ − P = P̃hᵀhP̃·w/(1 + w·g)`
#[test]
fn satisfies_stationarity_on_random_problems() {
    let mut rng = StdRng::seed_from_u64(69);
    for &nu in &[0.0, 0.5, 1.0, 3.0, 10.0, 100.0] {
        for _ in 0..20 {
            let problem = random_problem(&mut rng, nu);
            let pro = correct(&problem, true, BUDGET, None);
            assert_eq!(pro.status, CorrectionStatus::Applied);

            let (pht, g) = prior_gain(&problem);
            let v = problem.z - dot(&problem.h, &pro.x);
            let s = quadratic(&problem.h, &pro.p) + problem.r;
            let w = 1.0 / s - v * v / (s * s);
            let shrink = w / (1.0 + w * g);
            let scale = pht.iter().fold(1.0_f64, |m, p| m.max(p.abs()));

            for i in 0..3 {
                let dx = pro.x[i] - problem.x[i];
                assert!(
                    (dx - pht[i] * v / s).abs() <= 1e-9 * scale.max(dx.abs()),
                    "nu={nu} mean[{i}]"
                );
                for j in 0..3 {
                    let dp = problem.p[i][j] - pro.p[i][j];
                    let expected = pht[i] * pht[j] * shrink;
                    assert!(
                        (dp - expected).abs() <= 1e-8 * scale * scale,
                        "nu={nu} P[{i}][{j}]: {dp} vs {expected}"
                    );
                }
            }
        }
    }
}

/// Mean squared state errors of the Kalman filter and the PrO filter.
struct Errors {
    /// Mean squared error of the regular Kalman filter.
    kf: f64,
    /// Mean squared error of the PrO filter.
    pro: f64,
}

/// Tracks a 1-D constant-velocity target (`dt = 0.1`, `Q = 0.1·I`, `R = 10`) for `steps`
/// steps over `runs` runs, with measurement noise `√10·noise`, and returns the mean squared
/// state errors of both filters.
fn track(noise: &impl Distribution<f64>, runs: usize, steps: usize, seed: u64) -> Errors {
    let mut rng = StdRng::seed_from_u64(seed);
    let process = Normal::new(0.0, 0.1_f64.sqrt()).expect("valid standard deviation");
    let mut errors = Errors { kf: 0.0, pro: 0.0 };

    for _ in 0..runs {
        let builder = KalmanFilterBuilder::<2, f64>::default();
        let mut kf = builder.build();
        let mut pro = builder.build();
        let mut kf_measurement = builder.observations().build::<1>();
        let mut pro_measurement = builder.observations().build::<1>();

        for filter in [&mut kf, &mut pro] {
            filter.state_transition_mut().make_identity();
            filter.state_transition_mut().set_at(0, 1, 0.1);
            filter.estimate_covariance_mut().make_scalar(10.0);
            filter.direct_process_noise_mut().make_scalar(0.1);
        }

        for measurement in [&mut kf_measurement, &mut pro_measurement] {
            measurement.observation_matrix_mut().set_at(0, 0, 1.0);
            measurement
                .measurement_noise_covariance_mut()
                .set_at(0, 0, 10.0);
        }

        let mut truth = [0.0_f64, 1.0];
        for _ in 0..steps {
            truth = [
                truth[0] + 0.1 * truth[1] + process.sample(&mut rng),
                truth[1] + process.sample(&mut rng),
            ];
            let z = truth[0] + 10.0_f64.sqrt() * noise.sample(&mut rng);

            kf.predict();
            pro.predict();
            kf_measurement.measurement_vector_mut().set_at(0, 0, z);
            pro_measurement.measurement_vector_mut().set_at(0, 0, z);
            assert!(kf.correct(&mut kf_measurement).is_applied());
            assert!(
                pro.correct_predictively_oriented(&mut pro_measurement, BUDGET)
                    .is_applied()
            );

            for (i, value) in truth.iter().enumerate() {
                errors.kf += (kf.state_vector().get_at(i, 0) - value).powi(2);
                errors.pro += (pro.state_vector().get_at(i, 0) - value).powi(2);
            }
        }
    }

    let samples = (runs * steps * 2) as f64;
    errors.kf /= samples;
    errors.pro /= samples;
    errors
}

/// Under heavy-tailed Student-t noise (2.01 degrees of freedom) the PrO filter tracks with a
/// clearly lower mean squared error than the Kalman filter.
#[test]
fn heavy_tails_favor_pro() {
    let noise = StudentT::new(2.01).expect("valid degrees of freedom");
    let errors = track(&noise, 10, 1000, 7);
    println!(
        "t(2.01): KF MSE {:.3}, PrO MSE {:.3}",
        errors.kf, errors.pro
    );
    assert!(errors.pro < 0.75 * errors.kf);
}

/// Under Gaussian noise the PrO filter costs a little accuracy, but stays close to the
/// optimal Kalman filter.
#[test]
fn gaussian_noise_costs_little() {
    let noise = Normal::new(0.0, 1.0).expect("valid standard deviation");
    let errors = track(&noise, 10, 1000, 8);
    println!(
        "Gaussian: KF MSE {:.3}, PrO MSE {:.3}",
        errors.kf, errors.pro
    );
    assert!(errors.kf <= errors.pro && errors.pro < 1.2 * errors.kf);
}

/// A scalar problem with `x̃ = 0`, `P̃ = 1`, `h = 1`, and the given `r` and `z`.
fn scalar(r: f64, z: f64) -> Problem<1> {
    Problem {
        x: [0.0],
        p: [[1.0]],
        h: [1.0],
        r,
        z,
    }
}

/// Without measurement noise, or with noise that is not a number, the correction is skipped
/// and the state and covariance stay unchanged.
#[test]
fn non_positive_noise_is_rejected() {
    for r in [0.0, -1.0, f64::NAN] {
        let pro = correct(&scalar(r, 3.0), true, BUDGET, None);
        assert_eq!(pro.status, CorrectionStatus::NotPositiveDefinite, "r = {r}");
        assert_eq!((pro.x[0], pro.p[0][0]), (0.0, 1.0));
    }
}

/// A measurement that carries no information about the state (`h·P̃·hᵀ = 0`) is skipped.
#[test]
fn uninformative_measurement_is_rejected() {
    let problem = Problem {
        x: [1.0, 2.0],
        p: [[1.0, 0.0], [0.0, 0.0]],
        h: [0.0, 1.0],
        r: 1.0,
        z: 5.0,
    };

    let pro = correct(&problem, true, BUDGET, None);

    assert_eq!(pro.status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!(pro.x, problem.x);
    assert_eq!(pro.p, problem.p);
}

/// The gate rejects outliers the same way as for the regular correction.
#[test]
fn gate_rejects_outliers() {
    let pro = correct(&scalar(1.0, 10.0), true, BUDGET, Some(6.63));
    assert_eq!(pro.status, CorrectionStatus::Gated);
    assert_eq!((pro.x[0], pro.p[0][0]), (0.0, 1.0));

    let pro = correct(&scalar(1.0, 1.0), true, BUDGET, Some(6.63));
    assert_eq!(pro.status, CorrectionStatus::Applied);
}

/// A zero budget still applies a valid, positive definite posterior, inflated less than the
/// converged one.
#[test]
fn zero_budget_inflates_less() {
    let problem = scalar(1.0, 3.0);
    let start = correct(&problem, true, 0, None);
    let converged = correct(&problem, true, BUDGET, None);

    assert_eq!(start.status, CorrectionStatus::Applied);
    assert!(start.p[0][0] > 0.0 && start.p[0][0] < converged.p[0][0]);
}

/// The observation can be corrected directly, without going through the filter.
#[test]
fn observation_level_call() {
    let builder = KalmanFilterBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    let mut x = BufferBuilder::state_vector_x::<1>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<1>().new();
    p.set_at(0, 0, 1.0);
    filter.estimate_covariance_mut().set_at(0, 0, 1.0);
    measurement.observation_matrix_mut().set_at(0, 0, 1.0);
    measurement
        .measurement_noise_covariance_mut()
        .set_at(0, 0, 1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 3.0);

    let status = measurement.correct_predictively_oriented(&mut x, &mut p, BUDGET);

    assert!(status.is_applied());
    assert!((x.get_at(0, 0) - 0.8567).abs() < 1e-4);
    assert!(filter.correct_predictively_oriented(&mut measurement, BUDGET) == status);
}

/// A huge outlier in `f32` leaves the state and covariance finite: the covariance grows by about
/// `|v|·√g`, and no intermediate grows faster than that.
#[test]
fn huge_outlier_stays_finite_in_f32() {
    let builder = KalmanFilterBuilder::<1, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().set_at(0, 0, 1.0);
    measurement.observation_matrix_mut().set_at(0, 0, 1.0);
    measurement
        .measurement_noise_covariance_mut()
        .set_at(0, 0, 1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 1e15);

    let status = filter.correct_predictively_oriented(&mut measurement, BUDGET);

    let x = filter.state_vector().get_at(0, 0);
    let p = filter.estimate_covariance().get_at(0, 0);
    assert!(status.is_applied());
    assert!(
        x.is_finite() && p.is_finite() && p > 1.0,
        "x = {x}, P = {p}"
    );
}
