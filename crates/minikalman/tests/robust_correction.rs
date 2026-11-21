//! The outlier-robust (Student-t, variational Bayes) correction of the regular, extended and
//! unscented filters.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// The budget of weight refreshes used throughout.
const ITERATIONS: usize = 20;

/// The chi-square 99% quantile for two degrees of freedom, used as the gate.
const GATE: f64 = 9.21;

/// The heap-allocated `M×M` workspace.
type Workspace<const M: usize> = TemporarySInvertedMatrixBufferOwnedType<M, f64>;

/// Creates a workspace for `M` measurements.
fn workspace<const M: usize>() -> Workspace<M> {
    BufferBuilder::temp_S_inv::<M>().new()
}

/// Student-t settings with `dof` degrees of freedom and the default budget.
fn student_t(dof: f64) -> RobustCorrection<f64> {
    RobustCorrection {
        weight: RobustWeight::StudentT { dof },
        iterations: ITERATIONS,
    }
}

/// Copies the state of a filter into an array.
fn state<const N: usize>(x: &impl Matrix<N, 1, f64>) -> [f64; N] {
    core::array::from_fn(|i| x.get(i, 0))
}

/// Copies a covariance into a nested array.
fn covariance<const N: usize>(p: &impl Matrix<N, N, f64>) -> [[f64; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| p.get(i, j)))
}

/// Returns whether the symmetric `a` is positive semi-definite up to `tolerance` on the
/// diagonal, by a Cholesky factorization of `a + tolerance×I`.
fn is_psd<const N: usize>(a: &[[f64; N]; N], tolerance: f64) -> bool {
    let mut l = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let mut sum = a[i][j] + if i == j { tolerance } else { 0.0 };
            sum -= (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return false;
                }

                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    true
}

/// Returns `a − b`.
fn difference<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| a[i][j] - b[i][j]))
}

/// The prior, measurement and noise of a single 2-D correction with `H = I`.
struct Problem {
    /// Prior state.
    x: [f64; 2],
    /// Prior covariance.
    p: [[f64; 2]; 2],
    /// Measurement noise covariance.
    r: [[f64; 2]; 2],
    /// Measurement.
    z: [f64; 2],
}

/// A well-conditioned problem with the given measurement.
fn problem(z: [f64; 2]) -> Problem {
    Problem {
        x: [1.0, -2.0],
        p: [[2.0, 0.3], [0.3, 1.0]],
        r: [[0.5, 0.1], [0.1, 0.8]],
        z,
    }
}

/// The outcome of one correction: posterior state, covariance, status, NIS and weight.
struct Outcome {
    /// Posterior state.
    x: [f64; 2],
    /// Posterior covariance.
    p: [[f64; 2]; 2],
    /// What the correction did.
    status: CorrectionStatus,
    /// The NIS reported by the observation.
    nis: f64,
    /// The robust weight reported by the observation.
    weight: f64,
}

/// Runs one regular correction of `problem`, robust with `robust` or plain with `None`.
fn regular(problem: &Problem, robust: Option<RobustCorrection<f64>>, gate: Option<f64>) -> Outcome {
    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    for i in 0..2 {
        filter.state_vector_mut().set_at(i, 0, problem.x[i]);
        measurement
            .measurement_vector_mut()
            .set_at(i, 0, problem.z[i]);
        for j in 0..2 {
            filter
                .estimate_covariance_mut()
                .set_at(i, j, problem.p[i][j]);
            measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, problem.r[i][j]);
        }
    }

    measurement.observation_matrix_mut().make_identity();
    measurement.set_gate(gate);

    let status = match robust {
        Some(robust) => filter.correct_robust(&mut measurement, &mut workspace::<2>(), robust),
        None => filter.correct(&mut measurement),
    };

    Outcome {
        x: state(filter.state_vector().as_matrix()),
        p: covariance(filter.estimate_covariance().as_matrix()),
        status,
        nis: measurement.nis(),
        weight: measurement.robust_weight(),
    }
}

/// Runs one robust extended correction of `problem` with the linear `h(x) = x`.
fn extended(problem: &Problem, robust: RobustCorrection<f64>) -> Outcome {
    let builder = ExtendedBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    for i in 0..2 {
        filter.state_vector_mut().set_at(i, 0, problem.x[i]);
        measurement
            .measurement_vector_mut()
            .set_at(i, 0, problem.z[i]);
        for j in 0..2 {
            filter
                .estimate_covariance_mut()
                .set_at(i, j, problem.p[i][j]);
            measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, problem.r[i][j]);
        }
    }

    measurement
        .observation_jacobian_matrix_mut()
        .make_identity();
    let status = filter.correct_nonlinear_robust(
        &mut measurement,
        |x, y| {
            y[0] = x[0];
            y[1] = x[1];
        },
        &mut workspace::<2>(),
        robust,
    );

    Outcome {
        x: state(filter.state_vector().as_matrix()),
        p: covariance(filter.estimate_covariance().as_matrix()),
        status,
        nis: measurement.nis(),
        weight: measurement.robust_weight(),
    }
}

/// Runs one robust unscented correction of `problem` with the linear `h(x) = x`.
fn unscented(problem: &Problem, robust: RobustCorrection<f64>) -> Outcome {
    let builder = UnscentedBuilder::<2, f64>::default();
    let mut filter = builder.build::<5>();
    let mut measurement = builder.observations().build::<2, 5>();
    for i in 0..2 {
        filter.state_vector_mut().set_at(i, 0, problem.x[i]);
        measurement
            .measurement_vector_mut()
            .set_at(i, 0, problem.z[i]);
        for j in 0..2 {
            filter
                .estimate_covariance_mut()
                .set_at(i, j, problem.p[i][j]);
            measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, problem.r[i][j]);
        }
    }

    let status = filter.correct_sigma_point_robust(
        &mut measurement,
        |sigma, observed| {
            for i in 0..2 {
                for k in 0..5 {
                    observed.set_at(i, k, sigma.get_at(i, k));
                }
            }
        },
        &mut workspace::<2>(),
        robust,
    );

    Outcome {
        x: state(filter.state_vector().as_matrix()),
        p: covariance(filter.estimate_covariance().as_matrix()),
        status,
        nis: measurement.nis(),
        weight: measurement.robust_weight(),
    }
}

/// Asserts that two outcomes agree to `tolerance`.
fn assert_close(a: &Outcome, b: &Outcome, tolerance: f64, what: &str) {
    for i in 0..2 {
        assert!(
            (a.x[i] - b.x[i]).abs() <= tolerance,
            "{what}: x {:?} vs {:?}",
            a.x,
            b.x
        );
        for j in 0..2 {
            assert!(
                (a.p[i][j] - b.p[i][j]).abs() <= tolerance,
                "{what}: P {:?} vs {:?}",
                a.p,
                b.p
            );
        }
    }
}

/// With very many degrees of freedom the weight is `1` and the update is the Kalman update.
#[test]
fn many_degrees_of_freedom_reproduce_the_kalman_update() {
    for z in [[1.5, -1.0], [4.0, 3.0], [30.0, -20.0]] {
        let plain = regular(&problem(z), None, None);
        let robust = regular(&problem(z), Some(student_t(1e9)), None);
        assert_eq!(robust.status, CorrectionStatus::Applied);
        assert!((robust.weight - 1.0).abs() < 1e-6, "w = {}", robust.weight);
        assert_close(&robust, &plain, 1e-6, "ν = 1e9");
        assert_eq!(robust.nis, plain.nis, "the NIS is that of the prior");
    }
}

/// An outlier gets a small weight, moves the state little and shrinks `P` less than the
/// Kalman update, but never grows it.
#[test]
fn outliers_are_discounted() {
    let prior = problem([0.0; 2]);
    let plain = regular(&problem([41.0, -2.0]), None, None);
    let robust = regular(&problem([41.0, -2.0]), Some(student_t(4.0)), None);

    assert_eq!(robust.status, CorrectionStatus::Applied);
    assert!(robust.weight < 0.02, "w = {}", robust.weight);
    assert!((robust.x[0] - prior.x[0]).abs() < 0.1 * (plain.x[0] - prior.x[0]).abs());
    assert!(is_psd(&difference(&prior.p, &robust.p), 1e-12), "P⁺ ≤ P⁻");
    assert!(is_psd(&difference(&robust.p, &plain.p), 1e-12), "P⁺ ≥ P_KF");
}

/// The extended and unscented corrections agree with the regular one for a linear
/// observation (for the unscented filter, the sigma points capture a linear `h` exactly).
#[test]
fn extended_and_unscented_match_regular_for_linear_observations() {
    for z in [[1.5, -1.0], [6.0, 0.0], [40.0, 25.0]] {
        let reference = regular(&problem(z), Some(student_t(4.0)), None);
        let ekf = extended(&problem(z), student_t(4.0));
        let ukf = unscented(&problem(z), student_t(4.0));
        for (outcome, what) in [(&ekf, "EKF"), (&ukf, "UKF")] {
            assert_eq!(outcome.status, reference.status, "{what}");
            assert!((outcome.weight - reference.weight).abs() < 1e-9, "{what}");
            assert!((outcome.nis - reference.nis).abs() < 1e-9, "{what}");
            assert_close(outcome, &reference, 1e-9, what);
        }
    }
}

/// The gate is evaluated on the prior NIS, and a gated correction leaves everything
/// unchanged, including the weight.
#[test]
fn gate_uses_the_prior_nis() {
    let prior = problem([0.0; 2]);
    let plain = regular(&problem([10.0, 0.0]), None, Some(GATE));
    let robust = regular(&problem([10.0, 0.0]), Some(student_t(4.0)), Some(GATE));

    assert_eq!(plain.status, CorrectionStatus::Gated);
    assert_eq!(robust.status, CorrectionStatus::Gated);
    assert_eq!(robust.nis, plain.nis);
    assert_eq!(robust.weight, 1.0);
    assert_eq!((robust.x, robust.p), (prior.x, prior.p));
}

/// A singular innovation covariance is rejected without touching the state.
#[test]
fn not_positive_definite_is_rejected() {
    let mut singular = problem([3.0, 1.0]);
    singular.p = [[0.0; 2]; 2];
    singular.r = [[1.0, 1.0], [1.0, 1.0]];
    let robust = regular(&singular, Some(student_t(4.0)), None);

    assert_eq!(robust.status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!((robust.x, robust.p), (singular.x, singular.p));
}

/// A budget of zero applies the plain Kalman update and reports it as not converged; a budget
/// of one refresh does not settle either.
#[test]
fn small_budgets_report_not_converged() {
    let z = [6.0, 0.0];
    let plain = regular(&problem(z), None, None);
    let zero = regular(
        &problem(z),
        Some(RobustCorrection {
            iterations: 0,
            ..student_t(4.0)
        }),
        None,
    );
    assert_eq!(zero.status, CorrectionStatus::NotConverged);
    assert!(zero.status.is_applied());
    assert_eq!(zero.weight, 1.0);
    assert_close(&zero, &plain, 1e-12, "zero budget");

    let one = regular(
        &problem(z),
        Some(RobustCorrection {
            iterations: 1,
            ..student_t(4.0)
        }),
        None,
    );
    assert_eq!(one.status, CorrectionStatus::NotConverged);
    assert!(one.weight < 1.0);
}

/// The process noise adaptation of the filter runs on robust corrections.
#[test]
fn process_noise_adaptation_runs() {
    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);
    filter.set_process_noise_adaptation(Some(NoiseAdaptation {
        forgetting: 0.5,
        floor: 0.0,
    }));
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_identity();
    measurement.measurement_vector_mut().set_at(0, 0, 2.0);

    let status = filter.correct_robust(&mut measurement, &mut workspace::<2>(), student_t(4.0));

    assert!(status.is_applied());
    let dx = filter.state_vector().get_at(0, 0);
    let q = filter.direct_process_noise().get_at(0, 0);
    assert!(
        (q - (0.05 + 0.5 * dx * dx)).abs() < 1e-12,
        "Q = {q}, Δx = {dx}"
    );
}

/// Mean squared state errors of the compared corrections in [`track`].
#[derive(Debug, Default)]
struct Errors {
    /// The regular Kalman update.
    kf: f64,
    /// The Kalman update with the NIS gate at [`GATE`].
    gated: f64,
    /// The vector predictively oriented correction.
    pro: f64,
    /// The robust correction.
    robust: f64,
    /// The fraction of robust corrections that did not converge.
    not_converged: f64,
}

/// Tracks a 2-D random walk (`Q = 0.1·I`) observed directly (`H = I`, `R = 10·I`) over `runs`
/// runs of `steps` steps. A fraction `contamination` of the measurements gets an outlier of
/// `20σ` on a random axis. Checks `P⁺ ≤ P⁻` for every robust correction.
fn track(contamination: f64, dof: f64, runs: usize, steps: usize, seed: u64) -> Errors {
    let mut rng = StdRng::seed_from_u64(seed);
    let process = Normal::new(0.0, 0.1_f64.sqrt()).expect("valid standard deviation");
    let noise = Normal::new(0.0, 10.0_f64.sqrt()).expect("valid standard deviation");
    let outlier = 20.0 * 10.0_f64.sqrt();
    let mut errors = Errors::default();
    let mut robust_workspace = workspace::<2>();
    let mut pro_workspace =
        PredictivelyOrientedWorkspace::new(workspace::<2>(), workspace::<2>(), workspace::<2>());

    for _ in 0..runs {
        let builder = RegularBuilder::<2, f64>::default();
        let mut filters = [
            builder.build(),
            builder.build(),
            builder.build(),
            builder.build(),
        ];
        let mut measurements = [
            builder.observations().build::<2>(),
            builder.observations().build::<2>(),
            builder.observations().build::<2>(),
            builder.observations().build::<2>(),
        ];
        for filter in filters.iter_mut() {
            filter.state_transition_mut().make_identity();
            filter.estimate_covariance_mut().make_scalar(10.0);
            filter.direct_process_noise_mut().make_scalar(0.1);
        }

        for measurement in measurements.iter_mut() {
            measurement.observation_matrix_mut().make_identity();
            measurement
                .measurement_noise_covariance_mut()
                .make_scalar(10.0);
        }

        measurements[1].set_gate(Some(GATE));

        let mut truth = [0.0_f64; 2];
        for _ in 0..steps {
            for value in truth.iter_mut() {
                *value += process.sample(&mut rng);
            }

            let mut z = [
                truth[0] + noise.sample(&mut rng),
                truth[1] + noise.sample(&mut rng),
            ];
            if rng.gen_bool(contamination) {
                let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                z[usize::from(rng.gen_bool(0.5))] += sign * outlier;
            }

            for (filter, measurement) in filters.iter_mut().zip(measurements.iter_mut()) {
                filter.predict();
                for (i, &value) in z.iter().enumerate() {
                    measurement.measurement_vector_mut().set_at(i, 0, value);
                }
            }

            let [kf, gated, pro, robust] = &mut filters;
            let [kf_m, gated_m, pro_m, robust_m] = &mut measurements;
            assert!(kf.correct(kf_m).is_applied());
            gated.correct(gated_m);
            assert!(
                pro.correct_predictively_oriented_vector(pro_m, &mut pro_workspace, ITERATIONS)
                    .is_applied()
            );

            let prior = covariance(robust.estimate_covariance().as_matrix());
            let status = robust.correct_robust(robust_m, &mut robust_workspace, student_t(dof));
            assert!(status.is_applied());
            if status == CorrectionStatus::NotConverged {
                errors.not_converged += 1.0;
            }

            let posterior = covariance(robust.estimate_covariance().as_matrix());
            assert!(is_psd(&difference(&prior, &posterior), 1e-9), "P⁺ ≤ P⁻");

            for (error, filter) in [
                &mut errors.kf,
                &mut errors.gated,
                &mut errors.pro,
                &mut errors.robust,
            ]
            .into_iter()
            .zip(filters.iter())
            {
                let x = state(filter.state_vector().as_matrix());
                *error += (x[0] - truth[0]).powi(2) + (x[1] - truth[1]).powi(2);
            }
        }
    }

    let samples = (runs * steps * 2) as f64;
    for error in [
        &mut errors.kf,
        &mut errors.gated,
        &mut errors.pro,
        &mut errors.robust,
    ] {
        *error /= samples;
    }

    errors.not_converged /= (runs * steps) as f64;
    errors
}

/// Under 5% contamination with 20σ outliers, the robust correction clearly beats the plain
/// Kalman update and is not worse than the gate.
#[test]
fn contaminated_noise_favors_the_robust_correction() {
    let errors = track(0.05, 4.0, 10, 1000, 11);
    println!("5% at 20σ, ν = 4: {errors:?}");
    assert!(errors.robust < 0.5 * errors.kf);
    assert!(errors.robust <= 1.05 * errors.gated);
}

/// Under Gaussian noise, `ν = 5` costs little against the optimal Kalman update.
#[test]
fn gaussian_noise_costs_little() {
    let errors = track(0.0, 5.0, 10, 1000, 12);
    println!("Gaussian, ν = 5: {errors:?}");
    assert!(errors.robust < 1.1 * errors.kf);
}

/// Range and bearing of the position `x` from the sensor at `(-50, 0)`.
fn observe(x: &[f64; 2]) -> [f64; 2] {
    let (dx, dy) = (x[0] + 50.0, x[1]);
    [dx.hypot(dy), dy.atan2(dx)]
}

/// The Jacobian of [`observe`] at `x`.
fn jacobian(x: &[f64; 2]) -> [[f64; 2]; 2] {
    let (dx, dy) = (x[0] + 50.0, x[1]);
    let r2 = dx * dx + dy * dy;
    let r = r2.sqrt();
    [[dx / r, dy / r], [-dy / r2, dx / r2]]
}

/// Writes [`observe`] of the state `x` to `y`.
fn observe_state(x: &impl Matrix<2, 1, f64>, y: &mut impl MatrixMut<2, 1, f64>) {
    let value = observe(&state(x));
    y.set_at(0, 0, value[0]);
    y.set_at(1, 0, value[1]);
}

/// Writes [`observe`] of each of the five sigma points to `observed`.
fn observe_sigma(sigma: &impl Matrix<2, 5, f64>, observed: &mut impl MatrixMut<2, 5, f64>) {
    for k in 0..5 {
        let value = observe(&[sigma.get_at(0, k), sigma.get_at(1, k)]);
        observed.set_at(0, k, value[0]);
        observed.set_at(1, k, value[1]);
    }
}

/// Tracks a 2-D random walk observed by range and bearing from a sensor at `(-50, 0)`, with
/// 5% of the ranges hit by a `+20σ` outlier, and returns the mean squared errors of the plain
/// and robust extended and unscented filters, in that order.
fn track_radar(runs: usize, steps: usize, seed: u64) -> [f64; 4] {
    let mut rng = StdRng::seed_from_u64(seed);
    let process = Normal::new(0.0, 0.1).expect("valid standard deviation");
    let (range_sigma, bearing_sigma) = (0.5, 0.01);
    let range_noise = Normal::new(0.0, range_sigma).expect("valid standard deviation");
    let bearing_noise = Normal::new(0.0, bearing_sigma).expect("valid standard deviation");

    let mut errors = [0.0; 4];
    let mut ws = workspace::<2>();
    for _ in 0..runs {
        let ekf_builder = ExtendedBuilder::<2, f64>::default();
        let mut ekfs = [ekf_builder.build(), ekf_builder.build()];
        let mut ekf_ms = [
            ekf_builder.observations().build::<2>(),
            ekf_builder.observations().build::<2>(),
        ];
        let ukf_builder = UnscentedBuilder::<2, f64>::default();
        let mut ukfs = [ukf_builder.build::<5>(), ukf_builder.build::<5>()];
        let mut ukf_ms = [
            ukf_builder.observations().build::<2, 5>(),
            ukf_builder.observations().build::<2, 5>(),
        ];
        for filter in ekfs.iter_mut() {
            filter.state_transition_jacobian_mut().make_identity();
            filter.estimate_covariance_mut().make_scalar(1.0);
            filter.direct_process_noise_mut().make_scalar(0.01);
        }

        for filter in ukfs.iter_mut() {
            filter.estimate_covariance_mut().make_scalar(1.0);
            filter.direct_process_noise_mut().make_scalar(0.01);
        }

        for r in ekf_ms
            .iter_mut()
            .map(|m| m.measurement_noise_covariance_mut())
            .chain(
                ukf_ms
                    .iter_mut()
                    .map(|m| m.measurement_noise_covariance_mut()),
            )
        {
            r.set_at(0, 0, range_sigma * range_sigma);
            r.set_at(1, 1, bearing_sigma * bearing_sigma);
        }

        let mut truth = [0.0_f64; 2];
        for _ in 0..steps {
            for value in truth.iter_mut() {
                *value += process.sample(&mut rng);
            }

            let clean = observe(&truth);
            let mut z = [
                clean[0] + range_noise.sample(&mut rng),
                clean[1] + bearing_noise.sample(&mut rng),
            ];
            if rng.gen_bool(0.05) {
                z[0] += 20.0 * range_sigma;
            }

            for (k, (filter, m)) in ekfs.iter_mut().zip(ekf_ms.iter_mut()).enumerate() {
                filter.predict_nonlinear(|x, next| {
                    next[0] = x[0];
                    next[1] = x[1];
                });
                let h = jacobian(&state(filter.state_vector().as_matrix()));
                for (i, (&value, row)) in z.iter().zip(h.iter()).enumerate() {
                    m.measurement_vector_mut().set_at(i, 0, value);
                    for (j, &entry) in row.iter().enumerate() {
                        m.observation_jacobian_matrix_mut().set_at(i, j, entry);
                    }
                }

                let status = if k == 0 {
                    filter.correct_nonlinear(m, observe_state)
                } else {
                    let robust = student_t(4.0);
                    filter.correct_nonlinear_robust(m, observe_state, &mut ws, robust)
                };
                assert!(status.is_applied());
            }

            for (k, (filter, m)) in ukfs.iter_mut().zip(ukf_ms.iter_mut()).enumerate() {
                filter.predict_nonlinear(|_| {});
                for (i, &value) in z.iter().enumerate() {
                    m.measurement_vector_mut().set_at(i, 0, value);
                }

                let status = if k == 0 {
                    filter.correct_sigma_point(m, observe_sigma)
                } else {
                    filter.correct_sigma_point_robust(m, observe_sigma, &mut ws, student_t(4.0))
                };
                assert!(status.is_applied());
            }

            let filters = [
                state(ekfs[0].state_vector().as_matrix()),
                state(ekfs[1].state_vector().as_matrix()),
                state(ukfs[0].state_vector().as_matrix()),
                state(ukfs[1].state_vector().as_matrix()),
            ];
            for (error, x) in errors.iter_mut().zip(filters) {
                *error += (x[0] - truth[0]).powi(2) + (x[1] - truth[1]).powi(2);
            }
        }
    }

    for error in errors.iter_mut() {
        *error /= (runs * steps * 2) as f64;
    }

    errors
}

/// On a range-bearing radar with range outliers, the robust extended and unscented
/// corrections beat their plain counterparts.
#[test]
fn radar_outliers_favor_the_robust_corrections() {
    let [ekf, ekf_robust, ukf, ukf_robust] = track_radar(5, 1000, 13);
    println!(
        "radar MSE: EKF {ekf:.4}, robust EKF {ekf_robust:.4}, UKF {ukf:.4}, robust UKF {ukf_robust:.4}"
    );
    assert!(ekf_robust < 0.8 * ekf);
    assert!(ukf_robust < 0.8 * ukf);
}

/// The observations' own robust corrections give the same result as the filters' methods.
#[test]
fn observations_correct_without_a_filter() {
    let z = [6.0, -1.0];
    let reference = regular(&problem(z), Some(student_t(4.0)), None);
    let prior = problem(z);
    let fill = |x: &mut StateVectorBufferOwnedType<2, f64>,
                p: &mut EstimateCovarianceMatrixBufferOwnedType<2, f64>| {
        for i in 0..2 {
            x.set_at(i, 0, prior.x[i]);
            for (j, &value) in prior.p[i].iter().enumerate() {
                p.set_at(i, j, value);
            }
        }
    };

    let builder = RegularBuilder::<2, f64>::default();
    let mut measurement = builder.observations().build::<2>();
    let ekf_builder = ExtendedBuilder::<2, f64>::default();
    let mut ekf_measurement = ekf_builder.observations().build::<2>();
    for i in 0..2 {
        measurement.measurement_vector_mut().set_at(i, 0, z[i]);
        ekf_measurement.measurement_vector_mut().set_at(i, 0, z[i]);
        for (j, &value) in prior.r[i].iter().enumerate() {
            measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, value);
            ekf_measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, value);
        }
    }

    measurement.observation_matrix_mut().make_identity();
    ekf_measurement
        .observation_jacobian_matrix_mut()
        .make_identity();

    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<2>().new();
    fill(&mut x, &mut p);
    let status = measurement.correct_robust(&mut x, &mut p, &mut workspace::<2>(), student_t(4.0));
    let outcome = Outcome {
        x: state(x.as_matrix()),
        p: covariance(p.as_matrix()),
        status,
        nis: measurement.nis(),
        weight: measurement.robust_weight(),
    };
    assert_eq!(outcome.status, reference.status);
    assert_close(&outcome, &reference, 1e-12, "regular observation");

    fill(&mut x, &mut p);
    let status = ekf_measurement.correct_nonlinear_robust(
        &mut x,
        &mut p,
        |x: &StateVectorBufferOwnedType<2, f64>, y: &mut _| {
            let y: &mut InnovationVectorBufferOwnedType<2, f64> = y;
            y.set_at(0, 0, x.get_at(0, 0));
            y.set_at(1, 0, x.get_at(1, 0));
        },
        &mut workspace::<2>(),
        student_t(4.0),
    );
    let outcome = Outcome {
        x: state(x.as_matrix()),
        p: covariance(p.as_matrix()),
        status,
        nis: ekf_measurement.nis(),
        weight: ekf_measurement.robust_weight(),
    };
    assert_close(&outcome, &reference, 1e-12, "extended observation");
}

/// The unscented robust correction honors the gate on the prior NIS.
#[test]
fn unscented_gate_uses_the_prior_nis() {
    let builder = UnscentedBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.estimate_covariance_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_identity();
    measurement.measurement_vector_mut().set_at(0, 0, 10.0);
    measurement.set_gate(Some(6.63));

    let status = filter.correct_sigma_point_robust(
        &mut measurement,
        |sigma, observed| {
            for k in 0..3 {
                observed.set_at(0, k, sigma.get_at(0, k));
            }
        },
        &mut workspace::<1>(),
        student_t(4.0),
    );

    assert_eq!(status, CorrectionStatus::Gated);
    assert!((measurement.nis() - 50.0).abs() < 1e-9);
    assert_eq!(measurement.robust_weight(), 1.0);
    assert_eq!(filter.state_vector().get_at(0, 0), 0.0);
}

/// `fix` normalizes the innovation before the weight is fitted: shifting it by `-1` equals a
/// measurement that is `1` smaller.
#[test]
fn fix_normalizes_the_innovation() {
    let reference = regular(&problem([5.0, -1.0]), Some(student_t(4.0)), None);
    let shifted = problem([6.0, -1.0]);
    let shift = |y: &mut [f64]| y[0] -= 1.0;

    let builder = RegularBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    let ekf_builder = ExtendedBuilder::<2, f64>::default();
    let mut ekf = ekf_builder.build();
    let mut ekf_measurement = ekf_builder.observations().build::<2>();
    for i in 0..2 {
        filter.state_vector_mut().set_at(i, 0, shifted.x[i]);
        ekf.state_vector_mut().set_at(i, 0, shifted.x[i]);
        measurement
            .measurement_vector_mut()
            .set_at(i, 0, shifted.z[i]);
        ekf_measurement
            .measurement_vector_mut()
            .set_at(i, 0, shifted.z[i]);
        for j in 0..2 {
            filter
                .estimate_covariance_mut()
                .set_at(i, j, shifted.p[i][j]);
            ekf.estimate_covariance_mut().set_at(i, j, shifted.p[i][j]);
            measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, shifted.r[i][j]);
            ekf_measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, shifted.r[i][j]);
        }
    }

    measurement.observation_matrix_mut().make_identity();
    ekf_measurement
        .observation_jacobian_matrix_mut()
        .make_identity();

    let status = filter.correct_robust_with(
        &mut measurement,
        &mut workspace::<2>(),
        student_t(4.0),
        shift,
    );
    assert_eq!(status, reference.status);
    let status = ekf.correct_nonlinear_robust_with(
        &mut ekf_measurement,
        |x, y| {
            y[0] = x[0];
            y[1] = x[1];
        },
        shift,
        &mut workspace::<2>(),
        student_t(4.0),
    );
    assert_eq!(status, reference.status);

    for (x, p, what) in [
        (
            state(filter.state_vector().as_matrix()),
            covariance(filter.estimate_covariance().as_matrix()),
            "regular",
        ),
        (
            state(ekf.state_vector().as_matrix()),
            covariance(ekf.estimate_covariance().as_matrix()),
            "extended",
        ),
    ] {
        let outcome = Outcome {
            x,
            p,
            status,
            nis: 0.0,
            weight: 0.0,
        };
        assert_close(&outcome, &reference, 1e-12, what);
    }
}

/// The applied posterior is the Kalman update with `R/w` for the reported weight `w`.
#[test]
fn applied_update_is_the_kalman_update_with_scaled_noise() {
    for z in [[2.0, -1.0], [6.0, 0.0], [41.0, -2.0]] {
        let robust = regular(&problem(z), Some(student_t(4.0)), None);
        assert!(robust.status.is_applied());
        assert!(robust.weight != 1.0);

        let mut scaled = problem(z);
        for row in scaled.r.iter_mut() {
            for value in row.iter_mut() {
                *value /= robust.weight;
            }
        }

        let plain = regular(&scaled, None, None);
        assert_close(&robust, &plain, 1e-9, "Kalman update with R/w");
    }
}

/// In `f32`, the default number type, the weight settles within the budget for small and
/// large innovations alike.
#[test]
fn f32_converges() {
    for (z, small) in [(1.0_f32, false), (3.0, false), (50.0, true)] {
        let builder = RegularBuilder::<2, f32>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<2>();
        let mut workspace = BufferBuilder::temp_S_inv::<2>().new();
        filter.estimate_covariance_mut().make_identity();
        measurement.observation_matrix_mut().make_identity();
        measurement
            .measurement_noise_covariance_mut()
            .make_identity();
        measurement.measurement_vector_mut().set_at(0, 0, z);

        let robust = RobustCorrection {
            weight: RobustWeight::StudentT { dof: 4.0_f32 },
            iterations: ITERATIONS,
        };
        let status = filter.correct_robust(&mut measurement, &mut workspace, robust);

        assert_eq!(status, CorrectionStatus::Applied, "z = {z}");
        assert_eq!(measurement.robust_weight() < 0.01, small, "z = {z}");
    }
}

/// The robust corrections honor the observation's covariance update: the Joseph form (the
/// default) leaves `P` exactly symmetric, and both forms agree on a well-conditioned problem.
#[test]
fn covariance_update_is_honored() {
    for z in [[2.0, -1.0], [41.0, -2.0]] {
        let prior = problem(z);
        let mut outcomes = Vec::new();
        for update in [CovarianceUpdate::Joseph, CovarianceUpdate::Standard] {
            let builder = RegularBuilder::<2, f64>::default();
            let mut filter = builder.build();
            let mut measurement = builder.observations().build::<2>();
            let ekf_builder = ExtendedBuilder::<2, f64>::default();
            let mut ekf = ekf_builder.build();
            let mut ekf_measurement = ekf_builder.observations().build::<2>();
            for i in 0..2 {
                filter.state_vector_mut().set_at(i, 0, prior.x[i]);
                ekf.state_vector_mut().set_at(i, 0, prior.x[i]);
                measurement.measurement_vector_mut().set_at(i, 0, z[i]);
                ekf_measurement.measurement_vector_mut().set_at(i, 0, z[i]);
                for (j, (&p, &r)) in prior.p[i].iter().zip(prior.r[i].iter()).enumerate() {
                    filter.estimate_covariance_mut().set_at(i, j, p);
                    ekf.estimate_covariance_mut().set_at(i, j, p);
                    measurement
                        .measurement_noise_covariance_mut()
                        .set_at(i, j, r);
                    ekf_measurement
                        .measurement_noise_covariance_mut()
                        .set_at(i, j, r);
                }
            }

            measurement.observation_matrix_mut().make_identity();
            ekf_measurement
                .observation_jacobian_matrix_mut()
                .make_identity();
            measurement.set_covariance_update(update);
            ekf_measurement.set_covariance_update(update);

            let status =
                filter.correct_robust(&mut measurement, &mut workspace::<2>(), student_t(4.0));
            assert!(status.is_applied());
            let status = ekf.correct_nonlinear_robust(
                &mut ekf_measurement,
                |x, y| {
                    y[0] = x[0];
                    y[1] = x[1];
                },
                &mut workspace::<2>(),
                student_t(4.0),
            );
            assert!(status.is_applied());

            for (x, p) in [
                (
                    state(filter.state_vector().as_matrix()),
                    covariance(filter.estimate_covariance().as_matrix()),
                ),
                (
                    state(ekf.state_vector().as_matrix()),
                    covariance(ekf.estimate_covariance().as_matrix()),
                ),
            ] {
                if update == CovarianceUpdate::Joseph {
                    assert_eq!(p[0][1], p[1][0], "Joseph keeps P exactly symmetric");
                }

                outcomes.push(Outcome {
                    x,
                    p,
                    status,
                    nis: 0.0,
                    weight: measurement.robust_weight(),
                });
            }
        }

        for outcome in &outcomes[1..] {
            assert_close(
                outcome,
                &outcomes[0],
                1e-12,
                "Joseph vs standard, regular vs extended",
            );
        }
    }
}
