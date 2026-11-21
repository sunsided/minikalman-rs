//! The predictively oriented (PrO) correction of scalar extended observations, linearized once
//! at the prior or relinearized at the PrO mean.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StudentT};

/// Newton budget used throughout.
const BUDGET: usize = 20;

/// The correction to apply.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// The regular EKF update, `correct_nonlinear`.
    Ekf,
    /// The PrO update linearized at the prior.
    Pro,
    /// The PrO update with the given number of relinearization rounds.
    Iterated(usize),
    /// [`Mode::Pro`] through the `_with` variant, whose `fix` subtracts the given shift.
    ProWith(f64),
    /// [`Mode::Iterated`] through the `_with` variant, whose `fix` subtracts the given shift.
    IteratedWith(usize, f64),
}

/// A scalar observation `z = h(x) + noise` with its Jacobian.
struct Sensor<'a, const N: usize> {
    /// The observation function.
    h: &'a dyn Fn(&[f64; N]) -> f64,
    /// The Jacobian of `h`, a row vector.
    jacobian: &'a dyn Fn(&[f64; N]) -> [f64; N],
    /// Measurement noise variance.
    r: f64,
    /// Measurement.
    z: f64,
}

/// The posterior of one correction.
struct Posterior<const N: usize> {
    /// Posterior state vector.
    x: [f64; N],
    /// Posterior covariance, row-major.
    p: [[f64; N]; N],
    /// The observation Jacobian held by the observation on return.
    h: [f64; N],
    /// The normalized innovation squared reported by the observation.
    nis: f64,
    /// Number of `jacobian` callback invocations.
    jacobian_calls: usize,
    /// What the correction did.
    status: CorrectionStatus,
}

/// Copies a state vector into an array.
fn to_array<const N: usize>(x: &impl Matrix<N, 1, f64>) -> [f64; N] {
    core::array::from_fn(|i| x.get(i, 0))
}

/// Runs one correction of `sensor` with `mode` from the prior `(x, p)` on an extended filter.
fn correct<const N: usize>(
    x: &[f64; N],
    p: &[[f64; N]; N],
    sensor: &Sensor<N>,
    mode: Mode,
    gate: Option<f64>,
) -> Posterior<N> {
    let builder = ExtendedBuilder::<N, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    let prior_h = (sensor.jacobian)(x);
    for i in 0..N {
        filter.state_vector_mut().set_at(i, 0, x[i]);
        measurement
            .observation_jacobian_matrix_mut()
            .set_at(0, i, prior_h[i]);
        for (j, &value) in p[i].iter().enumerate() {
            filter.estimate_covariance_mut().set_at(i, j, value);
        }
    }

    measurement
        .measurement_noise_covariance_mut()
        .set_at(0, 0, sensor.r);
    measurement.measurement_vector_mut().set_at(0, 0, sensor.z);
    measurement.set_gate(gate);

    let mut jacobian_calls = 0;
    let observation = |x: &_, y: &mut _| set_scalar(y, (sensor.h)(&to_array(x)));
    let jacobian = |x: &_, h: &mut _| {
        jacobian_calls += 1;
        set_row(h, &(sensor.jacobian)(&to_array(x)));
    };
    let shift = |by: f64| move |y: &mut [f64]| y[0] -= by;
    let status = match mode {
        Mode::Ekf => filter.correct_nonlinear(&mut measurement, observation),
        Mode::Pro => {
            filter.correct_nonlinear_predictively_oriented(&mut measurement, observation, BUDGET)
        }
        Mode::Iterated(rounds) => filter.correct_nonlinear_predictively_oriented_iterated(
            &mut measurement,
            observation,
            jacobian,
            rounds,
            BUDGET,
        ),
        Mode::ProWith(by) => filter.correct_nonlinear_predictively_oriented_with(
            &mut measurement,
            observation,
            shift(by),
            BUDGET,
        ),
        Mode::IteratedWith(rounds, by) => filter
            .correct_nonlinear_predictively_oriented_iterated_with(
                &mut measurement,
                observation,
                jacobian,
                shift(by),
                rounds,
                BUDGET,
            ),
    };

    Posterior {
        x: to_array(filter.state_vector().as_matrix()),
        p: core::array::from_fn(|i| {
            core::array::from_fn(|j| filter.estimate_covariance().get_at(i, j))
        }),
        h: core::array::from_fn(|i| measurement.observation_jacobian_matrix().get_at(0, i)),
        nis: measurement.nis(),
        jacobian_calls,
        status,
    }
}

/// Writes `value` into a `1 × 1` observation vector.
fn set_scalar(y: &mut impl AsMatrixMut<1, 1, f64>, value: f64) {
    y.as_matrix_mut().set(0, 0, value);
}

/// Writes `row` into a `1 × N` observation Jacobian.
fn set_row<const N: usize>(h: &mut impl ObservationMatrixMut<1, N, f64>, row: &[f64; N]) {
    for (i, &value) in row.iter().enumerate() {
        h.set_at(0, i, value);
    }
}

/// Returns `a·b`.
fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns `P·hᵀ`.
fn mult<const N: usize>(p: &[[f64; N]; N], h: &[f64; N]) -> [f64; N] {
    core::array::from_fn(|i| dot(&p[i], h))
}

/// Draws a random positive definite `3 × 3` covariance.
fn random_covariance(rng: &mut StdRng) -> [[f64; 3]; 3] {
    let l: [[f64; 3]; 3] = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            if j <= i {
                rng.gen_range(-1.0..1.0)
            } else {
                0.0
            }
        })
    });

    core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            let sum: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                sum + 0.1
            } else {
                sum
            }
        })
    })
}

/// Runs the regular PrO correction of the linear observation `h·x` on a regular filter.
fn regular_pro(x: &[f64; 3], p: &[[f64; 3]; 3], h: &[f64; 3], r: f64, z: f64) -> Posterior<3> {
    let builder = RegularBuilder::<3, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    for i in 0..3 {
        filter.state_vector_mut().set_at(i, 0, x[i]);
        measurement.observation_matrix_mut().set_at(0, i, h[i]);
        for (j, &value) in p[i].iter().enumerate() {
            filter.estimate_covariance_mut().set_at(i, j, value);
        }
    }

    measurement
        .measurement_noise_covariance_mut()
        .set_at(0, 0, r);
    measurement.measurement_vector_mut().set_at(0, 0, z);

    let status = filter.correct_predictively_oriented(&mut measurement, BUDGET);
    Posterior {
        x: to_array(filter.state_vector().as_matrix()),
        p: core::array::from_fn(|i| {
            core::array::from_fn(|j| filter.estimate_covariance().get_at(i, j))
        }),
        h: *h,
        nis: measurement.nis(),
        jacobian_calls: 0,
        status,
    }
}

/// Asserts that two posteriors agree to a relative tolerance of `1e-12`.
fn assert_same<const N: usize>(a: &Posterior<N>, b: &Posterior<N>, what: &str) {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0);
    assert_eq!(a.status, b.status, "{what}");
    assert!(close(a.nis, b.nis), "{what}: NIS");
    for i in 0..N {
        assert!(close(a.x[i], b.x[i]), "{what}: x[{i}]");
        for j in 0..N {
            assert!(close(a.p[i][j], b.p[i][j]), "{what}: P[{i}][{j}]");
        }
    }
}

/// For a linear `h`, the extended PrO update, with or without relinearization, equals the
/// regular PrO update.
#[test]
fn linear_observation_matches_regular_pro() {
    let mut rng = StdRng::seed_from_u64(93);
    for &nu in &[0.0, 1.0, 3.0, 30.0] {
        for _ in 0..20 {
            let x: [f64; 3] = core::array::from_fn(|_| rng.gen_range(-5.0..5.0));
            let p = random_covariance(&mut rng);
            let row: [f64; 3] = core::array::from_fn(|_| rng.gen_range(-2.0..2.0));
            let r = [0.01, 1.0, 10.0][rng.gen_range(0..3)];
            let g = dot(&row, &mult(&p, &row));
            let z = dot(&row, &x) + nu * (g + r).sqrt();

            let h = |x: &[f64; 3]| dot(&row, x);
            let jacobian = |_: &[f64; 3]| row;
            let sensor = Sensor {
                h: &h,
                jacobian: &jacobian,
                r,
                z,
            };

            let expected = regular_pro(&x, &p, &row, r, z);
            assert_eq!(expected.status, CorrectionStatus::Applied);
            assert_same(&correct(&x, &p, &sensor, Mode::Pro, None), &expected, "Pro");
            assert_same(
                &correct(&x, &p, &sensor, Mode::Iterated(3), None),
                &expected,
                "Iterated",
            );
        }
    }
}

/// Beacon position of the range sensor.
const BEACON: [f64; 2] = [-1.0, 2.0];

/// Range from the beacon to the position `x`.
fn range(x: &[f64; 2]) -> f64 {
    (x[0] - BEACON[0]).hypot(x[1] - BEACON[1])
}

/// Jacobian of [`range`].
fn range_jacobian(x: &[f64; 2]) -> [f64; 2] {
    let d = range(x);
    [(x[0] - BEACON[0]) / d, (x[1] - BEACON[1]) / d]
}

/// A range sensor whose measurement is far from the prediction at the prior `(2, 6)`, where
/// the range is `5`.
fn range_sensor(z: f64) -> Sensor<'static, 2> {
    Sensor {
        h: &range,
        jacobian: &range_jacobian,
        r: 0.04,
        z,
    }
}

/// The prior state of the range problems.
const PRIOR_X: [f64; 2] = [2.0, 6.0];

/// The prior covariance of the range problems.
const PRIOR_P: [[f64; 2]; 2] = [[1.0, 0.3], [0.3, 0.5]];

/// Asserts that `(x, P)` satisfies the PrO stationarity equations of the problem linearized at
/// `m` with the Jacobian `h` there: with `v = z − h(m) − h·(x − m)` and `S = h·P·hᵀ + r`,
///
/// - `x − x̃ = P̃hᵀ·v/S`
/// - `P̃ − P = P̃hᵀhP̃·w/(1 + w·g)` with `w = 1/S − v²/S²`, i.e. `P⁻¹ = P̃⁻¹ + hᵀ·w·h`
fn assert_stationary(posterior: &Posterior<2>, sensor: &Sensor<2>, m: &[f64; 2], h: &[f64; 2]) {
    let pht = mult(&PRIOR_P, h);
    let g = dot(h, &pht);
    let dm: [f64; 2] = core::array::from_fn(|i| posterior.x[i] - m[i]);
    let v = sensor.z - (sensor.h)(m) - dot(h, &dm);
    let s = dot(h, &mult(&posterior.p, h)) + sensor.r;
    let w = 1.0 / s - v * v / (s * s);
    let shrink = w / (1.0 + w * g);

    for i in 0..2 {
        let dx = posterior.x[i] - PRIOR_X[i];
        assert!((dx - pht[i] * v / s).abs() < 1e-9, "mean[{i}]");
        for j in 0..2 {
            let dp = PRIOR_P[i][j] - posterior.p[i][j];
            assert!((dp - pht[i] * pht[j] * shrink).abs() < 1e-9, "P[{i}][{j}]");
        }
    }
}

/// Asserts that a `2 × 2` covariance is symmetric and positive definite.
fn assert_positive_definite(p: &[[f64; 2]; 2]) {
    assert!((p[0][1] - p[1][0]).abs() < 1e-12, "symmetric");
    assert!(
        p[0][0] > 0.0 && p[0][0] * p[1][1] - p[0][1] * p[1][0] > 0.0,
        "{p:?}"
    );
}

/// Linearized once at the prior, the posterior of a range measurement satisfies the
/// stationarity condition of the problem linearized there, and stays positive definite.
#[test]
fn range_satisfies_stationarity_at_the_prior() {
    for z in [5.0, 6.0, 9.0] {
        let sensor = range_sensor(z);
        let pro = correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Pro, None);

        assert_eq!(pro.status, CorrectionStatus::Applied, "z = {z}");
        assert_stationary(&pro, &sensor, &PRIOR_X, &range_jacobian(&PRIOR_X));
        assert_positive_definite(&pro.p);
    }
}

/// Relinearized at the PrO mean, the posterior of a range measurement satisfies the
/// stationarity condition of the problem linearized at the last linearization point, which
/// converges to the posterior mean itself, and stays positive definite.
#[test]
fn range_iterated_reaches_a_fixed_point() {
    for z in [5.0, 6.0, 9.0] {
        let sensor = range_sensor(z);
        let single = correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Pro, None);
        let pro = correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Iterated(30), None);

        assert_eq!(pro.status, CorrectionStatus::Applied, "z = {z}");
        assert_eq!(pro.jacobian_calls, 30);
        assert_stationary(&pro, &sensor, &pro.x, &range_jacobian(&pro.x));
        assert_positive_definite(&pro.p);
        assert!((pro.h[0] - range_jacobian(&pro.x)[0]).abs() < 1e-9);
        assert!((pro.nis - single.nis).abs() < 1e-12, "NIS of the prior");
        if z != 5.0 {
            assert!(
                (pro.x[0] - single.x[0]).abs() > 1e-3,
                "relinearization matters"
            );
        }
    }
}

/// Mean squared state errors of the three filters.
#[derive(Debug, Default)]
struct Errors {
    /// Mean squared error of the EKF.
    ekf: f64,
    /// Mean squared error of the PrO filter linearized at the prior.
    pro: f64,
    /// Mean squared error of the relinearized PrO filter.
    iterated: f64,
}

/// Beacon offset of the tracking problem: the range is `√((p + 50)² + 20²)`.
const TRACK_BEACON: [f64; 2] = [-50.0, 20.0];

/// Range and its derivative with respect to the position `p` of the tracking problem.
fn track_range(p: f64) -> (f64, f64) {
    let dx = p - TRACK_BEACON[0];
    let d = dx.hypot(TRACK_BEACON[1]);
    (d, dx / d)
}

/// Tracks a 1-D constant-velocity target (`dt = 0.1`, `Q = 0.001·I`) with a range sensor offset
/// from its track (`R = 4`) for `steps` steps over `runs` runs, with measurement noise
/// `2·noise`, and returns the mean squared state errors of the EKF and both PrO filters.
///
/// The target starts at `p = 0` and moves away from the beacon at about 2 units per second,
/// so it does not pass the beacon, where the range would be ambiguous and its slope zero.
fn track(noise: &impl Distribution<f64>, runs: usize, steps: usize, seed: u64) -> Errors {
    let mut rng = StdRng::seed_from_u64(seed);
    let process = Normal::new(0.0, 0.001_f64.sqrt()).expect("valid standard deviation");
    let mut errors = Errors::default();

    for _ in 0..runs {
        let builder = ExtendedBuilder::<2, f64>::default();
        let mut filters = [builder.build(), builder.build(), builder.build()];
        let mut measurements = [
            builder.observations().build::<1>(),
            builder.observations().build::<1>(),
            builder.observations().build::<1>(),
        ];

        for filter in &mut filters {
            filter.state_transition_jacobian_mut().make_identity();
            filter.state_transition_jacobian_mut().set_at(0, 1, 0.1);
            filter.estimate_covariance_mut().make_scalar(10.0);
            filter.direct_process_noise_mut().make_scalar(0.001);
        }

        for measurement in &mut measurements {
            measurement
                .measurement_noise_covariance_mut()
                .set_at(0, 0, 4.0);
        }

        let mut truth = [0.0_f64, 2.0];
        for _ in 0..steps {
            truth = [
                truth[0] + 0.1 * truth[1] + process.sample(&mut rng),
                truth[1] + process.sample(&mut rng),
            ];
            let z = track_range(truth[0]).0 + 2.0 * noise.sample(&mut rng);

            for (k, (filter, measurement)) in filters.iter_mut().zip(&mut measurements).enumerate()
            {
                filter.predict_nonlinear(|x, px| {
                    px.set_at(0, 0, x.get_at(0, 0) + 0.1 * x.get_at(1, 0));
                    px.set_at(1, 0, x.get_at(1, 0));
                });

                let slope = track_range(filter.state_vector().get_at(0, 0)).1;
                measurement
                    .observation_jacobian_matrix_mut()
                    .set_at(0, 0, slope);
                measurement.measurement_vector_mut().set_at(0, 0, z);

                let observation =
                    |x: &_, y: &mut _| set_scalar(y, track_range(to_array::<2>(x)[0]).0);
                let status = match k {
                    0 => filter.correct_nonlinear(measurement, observation),
                    1 => filter.correct_nonlinear_predictively_oriented(
                        measurement,
                        observation,
                        BUDGET,
                    ),
                    _ => filter.correct_nonlinear_predictively_oriented_iterated(
                        measurement,
                        observation,
                        |x, h| h.set_at(0, 0, track_range(x.get_at(0, 0)).1),
                        3,
                        BUDGET,
                    ),
                };
                assert!(status.is_applied(), "filter {k}: {status:?}");
            }

            for (i, value) in truth.iter().enumerate() {
                errors.ekf += (filters[0].state_vector().get_at(i, 0) - value).powi(2);
                errors.pro += (filters[1].state_vector().get_at(i, 0) - value).powi(2);
                errors.iterated += (filters[2].state_vector().get_at(i, 0) - value).powi(2);
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let samples = (runs * steps * 2) as f64;
    errors.ekf /= samples;
    errors.pro /= samples;
    errors.iterated /= samples;
    errors
}

/// Under heavy-tailed Student-t noise (2.01 degrees of freedom) both PrO filters track a range
/// sensor with a clearly lower mean squared error than the EKF.
#[test]
fn heavy_tails_favor_pro() {
    let noise = StudentT::new(2.01).expect("valid degrees of freedom");
    let errors = track(&noise, 20, 500, 11);
    println!("t(2.01): {errors:?}");
    assert!(errors.pro < 0.75 * errors.ekf);
    assert!(errors.iterated < 0.75 * errors.ekf);
}

/// Under Gaussian noise both PrO filters cost some accuracy (about 22% in this setup),
/// but stay close to the EKF.
#[test]
fn gaussian_noise_costs_little() {
    let noise = Normal::new(0.0, 1.0).expect("valid standard deviation");
    let errors = track(&noise, 20, 500, 12);
    println!("Gaussian: {errors:?}");
    assert!(errors.pro < 1.35 * errors.ekf);
    assert!(errors.iterated < 1.35 * errors.ekf);
}

/// The gate rejects outliers by the NIS of the prior innovation, before any relinearization,
/// and leaves the state and covariance unchanged.
#[test]
fn gate_rejects_outliers() {
    for mode in [Mode::Pro, Mode::Iterated(3)] {
        let gated = correct(&PRIOR_X, &PRIOR_P, &range_sensor(9.0), mode, Some(6.63));
        assert_eq!(gated.status, CorrectionStatus::Gated, "{mode:?}");
        assert_eq!((gated.x, gated.p), (PRIOR_X, PRIOR_P), "{mode:?}");
        assert_eq!(gated.jacobian_calls, 0, "{mode:?}");
        assert!(gated.nis > 6.63, "{mode:?}");

        let applied = correct(&PRIOR_X, &PRIOR_P, &range_sensor(5.2), mode, Some(6.63));
        assert_eq!(applied.status, CorrectionStatus::Applied, "{mode:?}");
    }

    let ekf = correct(
        &PRIOR_X,
        &PRIOR_P,
        &range_sensor(9.0),
        Mode::Ekf,
        Some(6.63),
    );
    assert_eq!(ekf.status, CorrectionStatus::Gated);
}

/// A relinearization point where the measurement carries no information (`h·P̃·hᵀ = 0`) is
/// rejected, leaving the state and covariance unchanged.
#[test]
fn uninformative_relinearization_is_rejected() {
    let h = |x: &[f64; 1]| x[0] * x[0];
    let jacobian = |x: &[f64; 1]| [2.0 * x[0]];
    // The first relinearization point is the PrO mean of the prior linearization; a Jacobian
    // that vanishes exactly there makes the second round uninformative.
    let probe = Sensor {
        h: &h,
        jacobian: &jacobian,
        r: 1.0,
        z: 0.0,
    };
    let single = correct(&[1.0], &[[1.0]], &probe, Mode::Pro, None);
    let target = single.x[0];

    let zero_at_target = move |x: &[f64; 1]| {
        if (x[0] - target).abs() < 1e-12 {
            [0.0]
        } else {
            [2.0 * x[0]]
        }
    };
    let sensor = Sensor {
        h: &h,
        jacobian: &zero_at_target,
        r: 1.0,
        z: 0.0,
    };
    let pro = correct(&[1.0], &[[1.0]], &sensor, Mode::Iterated(2), None);

    assert_eq!(pro.status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!((pro.x, pro.p), ([1.0], [[1.0]]));
}

/// Asserts that two posteriors agree to an absolute tolerance of `1e-12`.
fn assert_close<const N: usize>(a: &Posterior<N>, b: &Posterior<N>, what: &str) {
    assert_eq!(a.status, b.status, "{what}");
    assert!((a.nis - b.nis).abs() < 1e-12, "{what}: NIS");
    for i in 0..N {
        assert!((a.x[i] - b.x[i]).abs() < 1e-12, "{what}: x[{i}]");
        for j in 0..N {
            assert!((a.p[i][j] - b.p[i][j]).abs() < 1e-12, "{what}: P[{i}][{j}]");
        }
    }
}

/// The `_with` variants apply `fix` to every innovation `z − h(m)`: subtracting a constant
/// there is the same as measuring `z` minus that constant.
#[test]
fn fix_applies_to_every_innovation() {
    let shifted = correct(&PRIOR_X, &PRIOR_P, &range_sensor(5.0), Mode::Pro, None);
    let fixed = correct(
        &PRIOR_X,
        &PRIOR_P,
        &range_sensor(6.0),
        Mode::ProWith(1.0),
        None,
    );
    assert_close(&fixed, &shifted, "Pro");

    let shifted = correct(
        &PRIOR_X,
        &PRIOR_P,
        &range_sensor(5.0),
        Mode::Iterated(4),
        None,
    );
    let fixed = correct(
        &PRIOR_X,
        &PRIOR_P,
        &range_sensor(6.0),
        Mode::IteratedWith(4, 1.0),
        None,
    );
    assert_close(&fixed, &shifted, "Iterated");
    assert_eq!(fixed.jacobian_calls, 4);
}

/// Zero relinearization rounds are the single linearization, and never call `jacobian`.
#[test]
fn zero_rounds_are_the_single_linearization() {
    let sensor = range_sensor(6.0);
    let single = correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Pro, None);
    let iterated = correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Iterated(0), None);

    assert_same(&iterated, &single, "zero rounds");
    assert_eq!(iterated.jacobian_calls, 0);
}

/// Without measurement noise, or with noise that is not a number, both variants skip the
/// correction before any relinearization and leave the state and covariance unchanged.
#[test]
fn non_positive_noise_is_rejected() {
    for r in [0.0, -1.0, f64::NAN] {
        for mode in [Mode::Pro, Mode::Iterated(3)] {
            let sensor = Sensor {
                r,
                ..range_sensor(6.0)
            };
            let pro = correct(&PRIOR_X, &PRIOR_P, &sensor, mode, None);

            assert_eq!(
                pro.status,
                CorrectionStatus::NotPositiveDefinite,
                "{mode:?} r={r}"
            );
            assert_eq!((pro.x, pro.p), (PRIOR_X, PRIOR_P), "{mode:?} r={r}");
            assert_eq!(pro.jacobian_calls, 0, "{mode:?} r={r}");
        }
    }
}

/// The observation can be corrected directly, without going through the filter, and gives the
/// same result as the filter-level methods.
#[test]
fn observation_level_calls() {
    let builder = ExtendedBuilder::<2, f64>::default();
    let sensor = range_sensor(6.0);
    let observation = |x: &_, y: &mut _| set_scalar(y, range(&to_array(x)));
    let jacobian = |x: &_, h: &mut _| set_row(h, &range_jacobian(&to_array(x)));

    for rounds in [None, Some(3)] {
        let mut measurement = builder.observations().build::<1>();
        let mut x = BufferBuilder::state_vector_x::<2>().new();
        let mut p = BufferBuilder::estimate_covariance_P::<2>().new();
        for i in 0..2 {
            x.set_at(i, 0, PRIOR_X[i]);
            for (j, &value) in PRIOR_P[i].iter().enumerate() {
                p.set_at(i, j, value);
            }
        }

        set_row(
            measurement.observation_jacobian_matrix_mut(),
            &range_jacobian(&PRIOR_X),
        );
        measurement
            .measurement_noise_covariance_mut()
            .set_at(0, 0, sensor.r);
        measurement.measurement_vector_mut().set_at(0, 0, sensor.z);

        let (status, expected) = match rounds {
            None => (
                measurement.correct_nonlinear_predictively_oriented(
                    &mut x,
                    &mut p,
                    observation,
                    BUDGET,
                ),
                correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Pro, None),
            ),
            Some(rounds) => (
                measurement.correct_nonlinear_predictively_oriented_iterated(
                    &mut x,
                    &mut p,
                    observation,
                    jacobian,
                    rounds,
                    BUDGET,
                ),
                correct(&PRIOR_X, &PRIOR_P, &sensor, Mode::Iterated(rounds), None),
            ),
        };

        let direct = Posterior {
            x: to_array(x.as_matrix()),
            p: core::array::from_fn(|i| core::array::from_fn(|j| p.get_at(i, j))),
            h: [0.0; 2],
            nis: measurement.nis(),
            jacobian_calls: 0,
            status,
        };
        assert_same(&direct, &expected, "observation level");
    }
}
