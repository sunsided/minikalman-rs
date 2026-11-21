//! The predictively oriented (PrO) correction of vector regular observations.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::prelude::*;
use minikalman::regular::builder::{
    KalmanFilterBuilder, KalmanFilterObservationType, KalmanFilterType,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StudentT};

/// A row-major `R×C` matrix.
type Mat<const R: usize, const C: usize> = [[f64; C]; R];

/// The heap-allocated workspace for `M` measurements.
type Workspace<const M: usize> =
    PredictivelyOrientedWorkspace<TemporarySInvertedMatrixBufferOwnedType<M, f64>>;

/// Iteration budget for tests that require convergence.
const LARGE_BUDGET: usize = 5000;

/// Iteration budget used in the filtering tests.
const BUDGET: usize = 20;

/// A prior with `N` states and a measurement with `M` components.
struct Problem<const N: usize, const M: usize> {
    /// Prior state vector.
    x: [f64; N],
    /// Prior covariance.
    p: Mat<N, N>,
    /// Observation matrix.
    h: Mat<M, N>,
    /// Measurement noise covariance.
    r: Mat<M, M>,
    /// Measurement.
    z: [f64; M],
}

/// The posterior state, covariance and the status of one correction.
struct Posterior<const N: usize> {
    /// Posterior state vector.
    x: [f64; N],
    /// Posterior covariance.
    p: Mat<N, N>,
    /// What the correction did.
    status: CorrectionStatus,
}

/// Creates a heap-allocated workspace for `M` measurements.
fn workspace<const M: usize>() -> Workspace<M> {
    PredictivelyOrientedWorkspace::new(
        BufferBuilder::temp_S_inv::<M>().new(),
        BufferBuilder::temp_S_inv::<M>().new(),
        BufferBuilder::temp_S_inv::<M>().new(),
    )
}

/// Builds a filter and an observation holding `problem`.
fn setup<const N: usize, const M: usize>(
    problem: &Problem<N, M>,
) -> (
    KalmanFilterType<N, f64>,
    KalmanFilterObservationType<N, M, f64>,
) {
    let builder = KalmanFilterBuilder::<N, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<M>();
    for (i, row) in problem.p.iter().enumerate() {
        filter.state_vector_mut().set_at(i, 0, problem.x[i]);
        for (j, &value) in row.iter().enumerate() {
            filter.estimate_covariance_mut().set_at(i, j, value);
        }
    }

    for i in 0..M {
        measurement
            .measurement_vector_mut()
            .set_at(i, 0, problem.z[i]);
        for (j, &value) in problem.h[i].iter().enumerate() {
            measurement.observation_matrix_mut().set_at(i, j, value);
        }

        for (j, &value) in problem.r[i].iter().enumerate() {
            measurement
                .measurement_noise_covariance_mut()
                .set_at(i, j, value);
        }
    }

    (filter, measurement)
}

/// Reads the state and covariance of `filter` into a [`Posterior`] with `status`.
fn posterior<const N: usize>(
    filter: &KalmanFilterType<N, f64>,
    status: CorrectionStatus,
) -> Posterior<N> {
    let mut posterior = Posterior {
        x: [0.0; N],
        p: [[0.0; N]; N],
        status,
    };
    for (i, row) in posterior.p.iter_mut().enumerate() {
        posterior.x[i] = filter.state_vector().get_at(i, 0);
        for (j, value) in row.iter_mut().enumerate() {
            *value = filter.estimate_covariance().get_at(i, j);
        }
    }

    posterior
}

/// Runs one vector PrO correction of `problem` with the given budget.
fn correct_vector<const N: usize, const M: usize>(
    problem: &Problem<N, M>,
    max_iterations: usize,
) -> Posterior<N> {
    let (mut filter, mut measurement) = setup(problem);
    let status = filter.correct_predictively_oriented_vector(
        &mut measurement,
        &mut workspace::<M>(),
        max_iterations,
    );
    posterior(&filter, status)
}

/// Returns `a×b`.
fn mul<const R: usize, const K: usize, const C: usize>(a: &Mat<R, K>, b: &Mat<K, C>) -> Mat<R, C> {
    let mut out = [[0.0; C]; R];
    for i in 0..R {
        for j in 0..C {
            out[i][j] = (0..K).map(|k| a[i][k] * b[k][j]).sum();
        }
    }

    out
}

/// Returns `aᵀ`.
fn transpose<const R: usize, const C: usize>(a: &Mat<R, C>) -> Mat<C, R> {
    let mut out = [[0.0; R]; C];
    for i in 0..R {
        for j in 0..C {
            out[j][i] = a[i][j];
        }
    }

    out
}

/// Returns `a + scale×b`.
fn add<const R: usize, const C: usize>(a: &Mat<R, C>, scale: f64, b: &Mat<R, C>) -> Mat<R, C> {
    let mut out = *a;
    for i in 0..R {
        for j in 0..C {
            out[i][j] += scale * b[i][j];
        }
    }

    out
}

/// Returns the Frobenius norm of `a`.
fn norm<const R: usize, const C: usize>(a: &Mat<R, C>) -> f64 {
    a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt()
}

/// Returns the lower Cholesky factor of `a`, or `None` if `a` is not positive definite.
fn cholesky<const K: usize>(a: &Mat<K, K>) -> Option<Mat<K, K>> {
    let mut l = [[0.0; K]; K];
    for i in 0..K {
        for j in 0..=i {
            let sum = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if sum.is_nan() || sum <= 0.0 || sum.is_infinite() {
                    return None;
                }

                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }

    Some(l)
}

/// Returns `a⁻¹` by Gauss-Jordan elimination with partial pivoting.
fn inverse<const K: usize>(a: &Mat<K, K>) -> Mat<K, K> {
    let mut a = *a;
    let mut inv = [[0.0; K]; K];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for col in 0..K {
        let pivot = (col..K)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .expect("non-empty range");
        a.swap(col, pivot);
        inv.swap(col, pivot);

        let scale = a[col][col];
        for j in 0..K {
            a[col][j] /= scale;
            inv[col][j] /= scale;
        }

        for i in 0..K {
            if i != col {
                let factor = a[i][col];
                for j in 0..K {
                    a[i][j] -= factor * a[col][j];
                    inv[i][j] -= factor * inv[col][j];
                }
            }
        }
    }

    inv
}

/// Returns the column vector `v` as an `M×1` matrix.
fn column<const M: usize>(v: &[f64; M]) -> Mat<M, 1> {
    v.map(|value| [value])
}

/// Returns a random symmetric positive definite matrix `L×Lᵀ + floor×I`.
fn random_spd<const K: usize>(rng: &mut StdRng, floor: f64) -> Mat<K, K> {
    let mut l = [[0.0; K]; K];
    for (i, row) in l.iter_mut().enumerate() {
        for value in row.iter_mut().take(i + 1) {
            *value = rng.gen_range(-1.0..1.0);
        }
    }

    let mut a = mul(&l, &transpose(&l));
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += floor;
    }

    a
}

/// Draws a random problem whose prior innovation has a NIS of `nu²`.
fn random_problem<const N: usize, const M: usize>(rng: &mut StdRng, nu: f64) -> Problem<N, M> {
    let scale = [0.01, 1.0, 10.0][rng.gen_range(0..3)];
    let mut problem = Problem {
        x: [0.0; N].map(|_| rng.gen_range(-5.0..5.0)),
        p: random_spd::<N>(rng, 0.1),
        h: [[0.0; N]; M].map(|row| row.map(|_| rng.gen_range(-2.0..2.0))),
        r: random_spd::<M>(rng, 0.05).map(|row| row.map(|v| v * scale)),
        z: [0.0; M],
    };

    // z = H×x̃ + ν×chol(G + R)×u with a random unit vector u.
    let s = add(
        &mul(&mul(&problem.h, &problem.p), &transpose(&problem.h)),
        1.0,
        &problem.r,
    );
    let l = cholesky(&s).expect("S is positive definite");
    let mut u = [0.0; M].map(|_| rng.gen_range(-1.0..1.0));
    let length = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    u.iter_mut().for_each(|v| *v *= nu / length);
    let offset = mul(&l, &column(&u));
    let hx = mul(&problem.h, &column(&problem.x));
    for i in 0..M {
        problem.z[i] = hx[i][0] + offset[i][0];
    }

    problem
}

/// Checks the stationarity of the PrO objective at the posterior, from the public results
/// only, and returns the dimensionless residual `√tr(D×C×D×C)` with `D = F(W) − W`, the
/// quantity the solver's convergence threshold applies to.
///
/// With `G = H×P̃×Hᵀ`, `C = H×P×Hᵀ`, `S = C + R` and the posterior residual `ε = z − H×x`,
/// a Kalman-shaped posterior has `W = C⁻¹ − G⁻¹`, and `M⁻¹×v = S⁻¹×ε`, so
/// `F(W) = S⁻¹ − S⁻¹×ε×εᵀ×S⁻¹`. Also checks that `P⁻¹ = P̃⁻¹ + Hᵀ×W×H`, that
/// `x − x̃ = P̃×Hᵀ×S⁻¹×ε` and that `P` is positive definite.
fn check_posterior<const N: usize, const M: usize>(
    problem: &Problem<N, M>,
    pro: &Posterior<N>,
    context: &str,
) -> f64 {
    let ht = transpose(&problem.h);
    let g = mul(&mul(&problem.h, &problem.p), &ht);
    let c = mul(&mul(&problem.h, &pro.p), &ht);
    let s_inv = inverse(&add(&c, 1.0, &problem.r));
    let hx = mul(&problem.h, &column(&pro.x));
    let residual = add(&column(&problem.z), -1.0, &hx);
    let a = mul(&s_inv, &residual);
    let w = add(&inverse(&c), -1.0, &inverse(&g));
    let f = add(&s_inv, -1.0, &mul(&a, &transpose(&a)));

    assert!(
        cholesky(&pro.p).is_some(),
        "{context}: P not positive definite"
    );

    // P⁻¹ = P̃⁻¹ + Hᵀ×W×H
    let information = add(&inverse(&problem.p), 1.0, &mul(&mul(&ht, &w), &problem.h));
    let p_inv = inverse(&pro.p);
    let error = norm(&add(&p_inv, -1.0, &information));
    assert!(
        error <= 1e-6 * norm(&p_inv),
        "{context}: P not Kalman-shaped ({error})"
    );

    // x − x̃ = P̃×Hᵀ×S⁻¹×ε
    let expected = mul(&mul(&problem.p, &ht), &a);
    let dx = add(&column(&pro.x), -1.0, &column(&problem.x));
    let error = norm(&add(&dx, -1.0, &expected));
    assert!(
        error <= 1e-8 * norm(&expected).max(1.0),
        "{context}: mean ({error})"
    );

    let dc = mul(&add(&f, -1.0, &w), &c);
    let dcdc = mul(&dc, &dc);
    (0..M).map(|i| dcdc[i][i]).sum::<f64>().abs().sqrt()
}

/// For random problems with moderate innovations, the converged result satisfies the
/// stationarity condition `F(W) = W` and is a valid, Kalman-shaped posterior.
#[test]
fn converged_result_is_stationary() {
    let mut rng = StdRng::seed_from_u64(87);
    for &nu in &[0.0, 0.5, 1.0, 2.0, 3.0] {
        for run in 0..20 {
            let problem = random_problem::<3, 2>(&mut rng, nu);
            let pro = correct_vector(&problem, LARGE_BUDGET);
            let context = format!("nu={nu} run={run}");
            assert_eq!(pro.status, CorrectionStatus::Applied, "{context}");

            let stationarity = check_posterior(&problem, &pro, &context);
            assert!(stationarity < 1e-4, "{context}: √tr(DCDC) = {stationarity}");
        }
    }
}

/// Three measurements of four states converge to a stationary point as well.
#[test]
fn three_measurements_are_stationary() {
    let mut rng = StdRng::seed_from_u64(3);
    for &nu in &[0.0, 1.0, 2.0] {
        for run in 0..10 {
            let problem = random_problem::<4, 3>(&mut rng, nu);
            let pro = correct_vector(&problem, LARGE_BUDGET);
            let context = format!("nu={nu} run={run}");
            assert_eq!(pro.status, CorrectionStatus::Applied, "{context}");

            let stationarity = check_posterior(&problem, &pro, &context);
            assert!(stationarity < 1e-4, "{context}: √tr(DCDC) = {stationarity}");
        }
    }
}

/// Large outliers within the filtering budget are still applied as valid, Kalman-shaped
/// posteriors, whether or not the iteration converged.
#[test]
fn large_outliers_stay_valid_within_budget() {
    let mut rng = StdRng::seed_from_u64(10);
    for &nu in &[10.0, 100.0, 1e4] {
        for run in 0..20 {
            let problem = random_problem::<3, 2>(&mut rng, nu);
            let pro = correct_vector(&problem, BUDGET);
            let context = format!("nu={nu} run={run}");
            assert!(pro.status.is_applied(), "{context}: {:?}", pro.status);
            check_posterior(&problem, &pro, &context);
        }
    }
}

/// Without innovation, for `H = I` and diagonal `P̃` and `R`, every axis has the scalar closed
/// form `c = (√(r² + 4gr) − r)/2`, the KF-like root of `1/c − 1/g = 1/(c + r)`, and the state
/// stays unchanged.
#[test]
fn zero_innovation_matches_closed_form() {
    let problem = Problem::<2, 2> {
        x: [1.0, -2.0],
        p: [[2.0, 0.0], [0.0, 0.5]],
        h: [[1.0, 0.0], [0.0, 1.0]],
        r: [[0.3, 0.0], [0.0, 1.5]],
        z: [1.0, -2.0],
    };

    let pro = correct_vector(&problem, LARGE_BUDGET);

    assert_eq!(pro.status, CorrectionStatus::Applied);
    assert_eq!(pro.x, problem.x);
    for i in 0..2 {
        let (g, r) = (problem.p[i][i], problem.r[i][i]);
        let c = ((r * r + 4.0 * g * r).sqrt() - r) / 2.0;
        assert!(
            (pro.p[i][i] - c).abs() < 1e-4 * c,
            "P[{i}][{i}] = {}",
            pro.p[i][i]
        );
    }

    assert!(pro.p[0][1].abs() < 1e-12 && pro.p[1][0].abs() < 1e-12);
}

/// With a single measurement, the vector method runs the scalar Newton solver: the results are
/// bit-identical to the scalar method, for any budget.
#[test]
fn scalar_observations_use_the_scalar_path() {
    let mut rng = StdRng::seed_from_u64(1);
    for &nu in &[0.0, 1.0, 10.0, 1000.0] {
        for &budget in &[0, 3, BUDGET] {
            let problem = random_problem::<3, 1>(&mut rng, nu);

            let (mut scalar, mut scalar_measurement) = setup(&problem);
            let scalar_status =
                scalar.correct_predictively_oriented(&mut scalar_measurement, budget);

            let (mut vector, mut vector_measurement) = setup(&problem);
            let vector_status = vector.correct_predictively_oriented_vector(
                &mut vector_measurement,
                &mut workspace::<1>(),
                budget,
            );

            let context = format!("nu={nu} budget={budget}");
            assert_eq!(scalar_status, vector_status, "{context}");
            let scalar = posterior(&scalar, scalar_status);
            let vector = posterior(&vector, vector_status);
            assert_eq!(scalar.x, vector.x, "{context}");
            assert_eq!(scalar.p, vector.p, "{context}");
            assert_eq!(
                scalar_measurement.nis(),
                vector_measurement.nis(),
                "{context}"
            );
        }
    }
}

/// Mean squared state errors of the compared filters.
struct Errors {
    /// Mean squared error of the regular Kalman filter.
    kf: f64,
    /// Mean squared error of the vector PrO filter.
    pro: f64,
    /// Fraction of PrO corrections that ran out of the budget.
    not_converged: f64,
}

/// Tracks a 2-D random walk (`Q = 0.1·I`) observed directly (`H = I`, `R = 10·I`) for `steps`
/// steps over `runs` runs, with independent measurement noise `√10·noise` per axis, and returns
/// the mean squared state errors of both filters and the fraction of unconverged PrO
/// corrections.
fn track(noise: &impl Distribution<f64>, runs: usize, steps: usize, seed: u64) -> Errors {
    let mut rng = StdRng::seed_from_u64(seed);
    let process = Normal::new(0.0, 0.1_f64.sqrt()).expect("valid standard deviation");
    let mut errors = Errors {
        kf: 0.0,
        pro: 0.0,
        not_converged: 0.0,
    };
    let mut workspace = workspace::<2>();

    for _ in 0..runs {
        let builder = KalmanFilterBuilder::<2, f64>::default();
        let mut kf = builder.build();
        let mut pro = builder.build();
        let mut kf_measurement = builder.observations().build::<2>();
        let mut pro_measurement = builder.observations().build::<2>();

        for filter in [&mut kf, &mut pro] {
            filter.state_transition_mut().make_identity();
            filter.estimate_covariance_mut().make_scalar(10.0);
            filter.direct_process_noise_mut().make_scalar(0.1);
        }

        for measurement in [&mut kf_measurement, &mut pro_measurement] {
            measurement.observation_matrix_mut().make_identity();
            measurement
                .measurement_noise_covariance_mut()
                .make_scalar(10.0);
        }

        let mut truth = [0.0_f64; 2];
        for _ in 0..steps {
            for value in truth.iter_mut() {
                *value += process.sample(&mut rng);
            }

            kf.predict();
            pro.predict();
            for (i, value) in truth.iter().enumerate() {
                let z = value + 10.0_f64.sqrt() * noise.sample(&mut rng);
                kf_measurement.measurement_vector_mut().set_at(i, 0, z);
                pro_measurement.measurement_vector_mut().set_at(i, 0, z);
            }

            assert!(kf.correct(&mut kf_measurement).is_applied());
            let status = pro.correct_predictively_oriented_vector(
                &mut pro_measurement,
                &mut workspace,
                BUDGET,
            );
            assert!(status.is_applied());
            if status == CorrectionStatus::NotConverged {
                errors.not_converged += 1.0;
            }

            for (i, value) in truth.iter().enumerate() {
                errors.kf += (kf.state_vector().get_at(i, 0) - value).powi(2);
                errors.pro += (pro.state_vector().get_at(i, 0) - value).powi(2);
            }
        }
    }

    let samples = (runs * steps * 2) as f64;
    errors.kf /= samples;
    errors.pro /= samples;
    errors.not_converged /= (runs * steps) as f64;
    errors
}

/// Under heavy-tailed Student-t noise (2.01 degrees of freedom) the vector PrO filter tracks
/// with a clearly lower mean squared error than the Kalman filter.
#[test]
fn heavy_tails_favor_pro() {
    let noise = StudentT::new(2.01).expect("valid degrees of freedom");
    let errors = track(&noise, 10, 1000, 7);
    println!(
        "t(2.01): KF MSE {:.3}, PrO MSE {:.3}, unconverged {:.1}%",
        errors.kf,
        errors.pro,
        100.0 * errors.not_converged
    );
    assert!(errors.pro < 0.75 * errors.kf);
}

/// Under Gaussian noise the vector PrO filter costs a little accuracy, but stays close to the
/// optimal Kalman filter.
#[test]
fn gaussian_noise_costs_little() {
    let noise = Normal::new(0.0, 1.0).expect("valid standard deviation");
    let errors = track(&noise, 10, 1000, 8);
    println!(
        "Gaussian: KF MSE {:.3}, PrO MSE {:.3}, unconverged {:.1}%",
        errors.kf,
        errors.pro,
        100.0 * errors.not_converged
    );
    assert!(errors.kf <= errors.pro && errors.pro < 1.2 * errors.kf);
}

/// More measurements than states make `H×P×Hᵀ` singular, and a singular `R` is rejected too;
/// both leave the state and covariance unchanged.
#[test]
fn non_positive_definite_is_rejected() {
    let too_many = Problem::<2, 3> {
        x: [1.0, 2.0],
        p: [[1.0, 0.0], [0.0, 1.0]],
        h: [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        z: [3.0, 1.0, 0.0],
    };
    let pro = correct_vector(&too_many, BUDGET);
    assert_eq!(pro.status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!((pro.x, pro.p), (too_many.x, too_many.p));

    let singular_noise = Problem::<2, 2> {
        x: [1.0, 2.0],
        p: [[1.0, 0.0], [0.0, 1.0]],
        h: [[1.0, 0.0], [0.0, 1.0]],
        r: [[1.0, 1.0], [1.0, 1.0]],
        z: [3.0, 1.0],
    };
    let pro = correct_vector(&singular_noise, BUDGET);
    assert_eq!(pro.status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!((pro.x, pro.p), (singular_noise.x, singular_noise.p));
}

/// The gate rejects outliers by the NIS of the prior innovation, as for the regular correction.
#[test]
fn gate_rejects_outliers() {
    let problem = Problem::<2, 2> {
        x: [0.0, 0.0],
        p: [[1.0, 0.0], [0.0, 1.0]],
        h: [[1.0, 0.0], [0.0, 1.0]],
        r: [[1.0, 0.0], [0.0, 1.0]],
        z: [6.0, 0.0],
    };

    let (mut filter, mut measurement) = setup(&problem);
    measurement.set_gate(Some(9.21));
    let status = filter.correct_predictively_oriented_vector(
        &mut measurement,
        &mut workspace::<2>(),
        BUDGET,
    );

    assert_eq!(status, CorrectionStatus::Gated);
    assert!((measurement.nis() - 18.0).abs() < 1e-12);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(
                measurement.kalman_gain().get_at(i, j),
                0.0,
                "K is left unchanged"
            );
        }
    }

    let pro = posterior(&filter, status);
    assert_eq!((pro.x, pro.p), (problem.x, problem.p));
}

/// A zero budget applies the start. Here `W = 0` has the lower objective than `W = R⁻¹`, so
/// the covariance stays at the prior, and the state moves by `P̃×Hᵀ×(2×G + R)⁻¹×v`. The
/// reported status is [`CorrectionStatus::NotConverged`].
#[test]
fn zero_budget_applies_the_start() {
    let problem = Problem::<2, 2> {
        x: [0.0, 0.0],
        p: [[1.0, 0.0], [0.0, 1.0]],
        h: [[1.0, 0.0], [0.0, 1.0]],
        r: [[1.0, 0.0], [0.0, 1.0]],
        z: [3.0, 0.0],
    };

    let pro = correct_vector(&problem, 0);

    assert_eq!(pro.status, CorrectionStatus::NotConverged);
    assert!(pro.status.is_applied());
    assert_eq!(pro.p, problem.p);
    assert!((pro.x[0] - 1.0).abs() < 1e-12 && pro.x[1] == 0.0);
}

/// The observation can be corrected directly, without going through the filter.
#[test]
fn observation_level_call() {
    let problem = random_problem::<3, 2>(&mut StdRng::seed_from_u64(5), 1.0);
    let (mut filter, mut measurement) = setup(&problem);
    let mut x = BufferBuilder::state_vector_x::<3>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<3>().new();
    for (i, row) in problem.p.iter().enumerate() {
        x.set_at(i, 0, problem.x[i]);
        for (j, &value) in row.iter().enumerate() {
            p.set_at(i, j, value);
        }
    }

    let status = measurement.correct_predictively_oriented_vector(
        &mut x,
        &mut p,
        &mut workspace::<2>(),
        BUDGET,
    );
    let filtered = filter.correct_predictively_oriented_vector(
        &mut measurement,
        &mut workspace::<2>(),
        BUDGET,
    );

    assert!(status.is_applied());
    assert_eq!(status, filtered);
    for i in 0..3 {
        assert_eq!(x.get_at(i, 0), filter.state_vector().get_at(i, 0));
    }
}

/// The status and the posterior `x₀`, `P₀₀`, `P₀₁` and `P₁₁` of one correction in `T`, as
/// `f64`.
type Summary = (CorrectionStatus, [f64; 4]);

/// Corrects a 2-state filter with `P̃ = p·I`, `H = I` and `R = [[r, ρ], [ρ, r]]` by the
/// measurement `z`, in `T`.
fn correct_2d<T>(p: f64, r: f64, rho: f64, z: [f64; 2], budget: usize) -> Summary
where
    T: MatrixDataType + Default + Into<f64>,
{
    let value = |v: f64| T::from_f64(v).expect("representable");
    let builder = KalmanFilterBuilder::<2, T>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();
    let mut workspace = PredictivelyOrientedWorkspace::new(
        BufferBuilder::temp_S_inv::<2>().new::<T>(),
        BufferBuilder::temp_S_inv::<2>().new::<T>(),
        BufferBuilder::temp_S_inv::<2>().new::<T>(),
    );
    filter.estimate_covariance_mut().make_scalar(value(p));
    measurement.observation_matrix_mut().make_identity();
    measurement
        .measurement_noise_covariance_mut()
        .make_comatrix(value(r), value(rho));
    for (i, &z) in z.iter().enumerate() {
        measurement.measurement_vector_mut().set_at(i, 0, value(z));
    }

    let status =
        filter.correct_predictively_oriented_vector(&mut measurement, &mut workspace, budget);
    let p = filter.estimate_covariance();
    let summary = [
        filter.state_vector().get_at(0, 0).into(),
        p.get_at(0, 0).into(),
        p.get_at(0, 1).into(),
        p.get_at(1, 1).into(),
    ];
    (status, summary)
}

/// Returns whether the posterior summary is finite with a positive definite covariance.
fn is_valid(summary: &[f64; 4]) -> bool {
    let [x, p00, p01, p11] = *summary;
    summary.iter().all(|v| v.is_finite())
        && p00 > 0.0
        && p11 > 0.0
        && p00 * p11 > p01 * p01
        && x.is_finite()
}

/// In `f32`, with a prior wider than the noise (`G = 100·R`, and `G = 10⁴·R` for small
/// innovations), the result agrees with the `f64` solution of the same problem to within 1%
/// and 5%. Without step expansion, the iteration would need about `√(G/R)` steps here.
#[test]
fn f32_with_wide_prior_matches_f64() {
    for &(p, accuracy, offsets) in &[
        (1.0, 1e-2, &[0.0, 0.1, 1.0][..]),
        (1e2, 5e-2, &[0.0, 0.1][..]),
    ] {
        for &offset in offsets {
            let z = [offset * p, -0.3 * offset * p];
            let (status, single) = correct_2d::<f32>(p, 1e-2, 2e-3, z, BUDGET);
            let (reference_status, double) = correct_2d::<f64>(p, 1e-2, 2e-3, z, LARGE_BUDGET);
            let context = format!("p={p} offset={offset}: {status:?} {single:?} vs {double:?}");

            assert_eq!(reference_status, CorrectionStatus::Applied, "{context}");
            assert!(status.is_applied() && is_valid(&single), "{context}");
            for (a, b) in single.iter().zip(&double) {
                assert!((a - b).abs() <= accuracy * b.abs().max(1e-2), "{context}");
            }
        }
    }
}

/// With `f64`, `G = 10⁶·R` converges within the filtering budget thanks to step expansion.
#[test]
fn very_wide_prior_converges_quickly() {
    let (status, summary) = correct_2d::<f64>(1e4, 1e-2, 2e-3, [0.1, -0.1], BUDGET);
    let (_, reference) = correct_2d::<f64>(1e4, 1e-2, 2e-3, [0.1, -0.1], LARGE_BUDGET);
    let context = format!("{status:?} {summary:?} vs {reference:?}");
    assert_eq!(status, CorrectionStatus::Applied, "{context}");
    assert!((summary[1] / reference[1] - 1.0).abs() < 1e-4, "{context}");
}

/// Huge outliers along one axis of a diagonal problem decouple, so the scalar PrO solution is
/// an exact reference. In `f64`, even `10⁸` standard deviations come out within the budget of
/// the filtering tests. In `f32`, `W ≈ −G⁻¹` cannot be resolved finely enough beyond about
/// `10⁷` standard deviations: the result stays valid and damps the outlier, but inflates less
/// and is reported as not converged.
#[test]
fn huge_outliers_match_the_scalar_solution() {
    for &nu in &[1e2, 1e5, 1e8] {
        let z = [nu * 2.0_f64.sqrt(), 0.0];
        let (status, vector) = correct_2d::<f64>(1.0, 1.0, 0.0, z, BUDGET);
        let scalar = Problem::<1, 1> {
            x: [0.0],
            p: [[1.0]],
            h: [[1.0]],
            r: [[1.0]],
            z: [z[0]],
        };
        let (mut filter, mut measurement) = setup(&scalar);
        let reference_status = filter.correct_predictively_oriented(&mut measurement, 50);
        let reference = posterior(&filter, reference_status);
        let context = format!("nu={nu:e}: {status:?} {vector:?} vs {:?}", reference.p);

        assert!(status.is_applied() && is_valid(&vector), "{context}");
        assert!((vector[0] - reference.x[0]).abs() < 1e-2, "{context}");
        assert!(
            (vector[1] / reference.p[0][0] - 1.0).abs() < 1e-3,
            "{context}"
        );
    }

    for &z in &[1e8, 1e15] {
        let (status, single) = correct_2d::<f32>(1.0, 1.0, 0.0, [z, 0.0], BUDGET);
        let context = format!("z={z:e}: {status:?} {single:?}");
        assert_eq!(status, CorrectionStatus::NotConverged, "{context}");
        assert!(is_valid(&single) && single[1] > 1e6, "{context}");
        assert!(single[0] < 1e-6 * z, "{context}");
    }
}
