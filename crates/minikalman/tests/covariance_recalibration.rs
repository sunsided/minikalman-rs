//! Covariance recalibration of the extended and unscented filters, after Jiang, Shi and
//! Moura, "Mitigating Overconfidence in Nonlinear Kalman Filters via Covariance
//! Recalibration", Automatica 194 (2026) 113295, arXiv:2407.05717.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Gravity in m/s², as in the paper's pendulum (Appendix C.4).
const G: f64 = 9.8;

/// Pendulum length in m.
const L: f64 = 1.0;

/// Pendulum mass in kg.
const M: f64 = 1.0;

/// Time step in s.
const DT: f64 = 0.01;

/// Number of filter steps per run (Table 1 of the paper).
const STEPS: usize = 100;

/// Number of Monte Carlo runs. The paper uses 10 000; fewer keep the test fast.
const RUNS: usize = 500;

/// Standard deviation of the measurement noise in N.
///
/// The paper sweeps this value and gives it only in its figures. A small value is chosen,
/// since that is where the paper reports the largest gain from recalibration.
const SIGMA_Z: f64 = 0.01;

/// Process noise variance of the angular speed. The angle gets none (Appendix C.4).
const Q_OMEGA: f64 = 1e-10;

/// The pendulum state `[ω, θ]`: angular speed in rad/s, angle in rad.
type State = [f64; 2];

/// One explicit Euler step of the pendulum (Eq. 43), both rows using the old state.
fn transition(s: State) -> State {
    let [omega, theta] = s;
    [omega - G / L * theta.sin() * DT, theta + omega * DT]
}

/// The Jacobian of [`transition`] at `s`, row-major.
fn transition_jacobian(s: State) -> [[f64; 2]; 2] {
    let theta = s[1];
    [[1.0, -G / L * theta.cos() * DT], [DT, 1.0]]
}

/// The horizontal component of the rope tension (Eq. 44).
fn observe(s: State) -> f64 {
    let [omega, theta] = s;
    M * G * theta.cos() * theta.sin() + M * L * omega * omega * theta.sin()
}

/// The Jacobian `[∂z/∂ω, ∂z/∂θ]` of [`observe`] at `s`.
fn observe_jacobian(s: State) -> [f64; 2] {
    let [omega, theta] = s;
    [
        2.0 * M * L * omega * theta.sin(),
        M * G * (2.0 * theta).cos() + M * L * omega * omega * theta.cos(),
    ]
}

/// The true trajectory and the measurements of one run, plus the initial estimate.
struct Run {
    /// The initial estimate, drawn around the true initial state with covariance `P₀`.
    initial_estimate: State,
    /// The true state after each step.
    truth: Vec<State>,
    /// The measurement after each step.
    measurements: Vec<f64>,
}

/// The initial variance `(π/18)²` of both states.
fn initial_variance() -> f64 {
    (core::f64::consts::PI / 18.0).powi(2)
}

/// Simulates one run of the pendulum from `[0, π/4]`.
fn simulate(rng: &mut StdRng) -> Run {
    let spread = Normal::new(0.0, initial_variance().sqrt()).unwrap();
    let process = Normal::new(0.0, Q_OMEGA.sqrt()).unwrap();
    let sensor = Normal::new(0.0, SIGMA_Z).unwrap();

    let start = [0.0, core::f64::consts::FRAC_PI_4];
    let initial_estimate = [start[0] + spread.sample(rng), start[1] + spread.sample(rng)];

    let mut state = start;
    let mut truth = Vec::with_capacity(STEPS);
    let mut measurements = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        state = transition(state);
        state[0] += process.sample(rng);
        truth.push(state);
        measurements.push(observe(state) + sensor.sample(rng));
    }

    Run {
        initial_estimate,
        truth,
        measurements,
    }
}

/// The statistics of one filter over all runs.
#[derive(Debug, Default)]
struct Stats {
    /// The final-state error norm of each run.
    errors: Vec<f64>,
    /// Number of corrections that were backed out.
    backed_out: usize,
    /// Number of corrections that were applied.
    applied: usize,
}

impl Stats {
    /// Records the final error and the statuses of one run.
    fn record(&mut self, estimate: State, truth: State, statuses: &[CorrectionStatus]) {
        self.errors
            .push(((estimate[0] - truth[0]).powi(2) + (estimate[1] - truth[1]).powi(2)).sqrt());
        self.backed_out += statuses
            .iter()
            .filter(|s| **s == CorrectionStatus::BackedOut)
            .count();
        self.applied += statuses.iter().filter(|s| s.is_applied()).count();
    }

    /// Root mean squared error of the final state over all runs.
    fn rmse(&self) -> f64 {
        let sum: f64 = self.errors.iter().map(|e| e * e).sum();
        (sum / self.errors.len() as f64).sqrt()
    }

    /// Median error of the final state over all runs.
    fn median(&self) -> f64 {
        let mut sorted = self.errors.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    }

    /// Fraction of corrections that were backed out.
    fn backed_out_fraction(&self) -> f64 {
        self.backed_out as f64 / (self.backed_out + self.applied) as f64
    }
}

/// Runs the extended filter over one run and returns the final estimate and the statuses.
fn run_ekf(run: &Run, recalibrate: bool) -> (State, Vec<CorrectionStatus>) {
    let builder = ExtendedBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.state_vector_mut().apply(|x| {
        x.set_at(0, 0, run.initial_estimate[0]);
        x.set_at(1, 0, run.initial_estimate[1]);
    });
    filter
        .estimate_covariance_mut()
        .make_scalar(initial_variance());
    filter.direct_process_noise_mut().set_at(0, 0, Q_OMEGA);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(SIGMA_Z * SIGMA_Z);

    let mut statuses = Vec::with_capacity(STEPS);
    for &z in &run.measurements {
        let f = transition_jacobian(state_of(filter.state_vector()));
        filter.state_transition_jacobian_mut().apply(|a| {
            for (i, row) in f.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    a.set_at(i, j, *v);
                }
            }
        });
        filter.predict_nonlinear(|current, next| {
            let [omega, theta] = transition(state_of(current));
            next[0] = omega;
            next[1] = theta;
        });

        measurement.measurement_vector_mut().set_at(0, 0, z);
        set_jacobian(
            measurement.observation_jacobian_matrix_mut(),
            state_of(filter.state_vector()),
        );
        let observation = |x: &_, y: &mut _| set_observation(y, state_of(x));
        statuses.push(if recalibrate {
            filter.correct_nonlinear_recalibrated(&mut measurement, observation, |x, h| {
                set_jacobian(h, state_of(x))
            })
        } else {
            filter.correct_nonlinear(&mut measurement, observation)
        });
    }

    (state_of(filter.state_vector()), statuses)
}

/// Runs the unscented filter over one run and returns the final estimate and the statuses.
fn run_ukf(run: &Run, recalibrate: bool) -> (State, Vec<CorrectionStatus>) {
    let builder = UnscentedBuilder::<2, f64>::default();
    let mut filter = builder.build::<5>();
    let mut measurement = builder.observations().build::<1, 5>();
    filter.state_vector_mut().apply(|x| {
        x.set_at(0, 0, run.initial_estimate[0]);
        x.set_at(1, 0, run.initial_estimate[1]);
    });
    filter
        .estimate_covariance_mut()
        .make_scalar(initial_variance());
    filter.direct_process_noise_mut().apply(|q| {
        q.set_zero();
        q.set_at(0, 0, Q_OMEGA);
    });
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(SIGMA_Z * SIGMA_Z);

    let mut statuses = Vec::with_capacity(STEPS);
    for &z in &run.measurements {
        filter.predict_nonlinear(|next| {
            let [omega, theta] = transition(state_of(next));
            next.set_at(0, 0, omega);
            next.set_at(1, 0, theta);
        });

        measurement.measurement_vector_mut().set_at(0, 0, z);
        let observation = |sigma: &_, observed: &mut _| observe_sigma_points(sigma, observed);
        statuses.push(if recalibrate {
            filter.correct_sigma_point_recalibrated(&mut measurement, observation)
        } else {
            filter.correct_sigma_point(&mut measurement, observation)
        });
    }

    (state_of(filter.state_vector()), statuses)
}

/// Reads a two-state vector.
fn state_of(x: &impl Matrix<2, 1, f64>) -> State {
    [x.get_at(0, 0), x.get_at(1, 0)]
}

/// Writes `h(s)` into a one-element observation vector.
fn set_observation(y: &mut impl MatrixMut<1, 1, f64>, s: State) {
    y.set_at(0, 0, observe(s));
}

/// Writes the observation Jacobian at `s`.
fn set_jacobian(h: &mut impl MatrixMut<1, 2, f64>, s: State) {
    let [d_omega, d_theta] = observe_jacobian(s);
    h.set_at(0, 0, d_omega);
    h.set_at(0, 1, d_theta);
}

/// Observes each of the five sigma points.
fn observe_sigma_points(sigma: &impl Matrix<2, 5, f64>, observed: &mut impl MatrixMut<1, 5, f64>) {
    for j in 0..5 {
        observed.set_at(0, j, observe([sigma.get_at(0, j), sigma.get_at(1, j)]));
    }
}

/// The final-state RMSE and back-out fraction of the four filters.
struct Results {
    /// Plain EKF.
    ekf: Stats,
    /// Recalibrated EKF.
    ekf_recalibrated: Stats,
    /// Plain UKF.
    ukf: Stats,
    /// Recalibrated UKF.
    ukf_recalibrated: Stats,
}

/// Runs all four filters on the same [`RUNS`] simulated runs.
fn monte_carlo() -> Results {
    let mut rng = StdRng::seed_from_u64(60);
    let mut results = Results {
        ekf: Stats::default(),
        ekf_recalibrated: Stats::default(),
        ukf: Stats::default(),
        ukf_recalibrated: Stats::default(),
    };

    for _ in 0..RUNS {
        let run = simulate(&mut rng);
        let truth = *run.truth.last().unwrap();
        for (stats, (estimate, statuses)) in [
            (&mut results.ekf, run_ekf(&run, false)),
            (&mut results.ekf_recalibrated, run_ekf(&run, true)),
            (&mut results.ukf, run_ukf(&run, false)),
            (&mut results.ukf_recalibrated, run_ukf(&run, true)),
        ] {
            stats.record(estimate, truth, &statuses);
        }
    }

    results
}

#[test]
fn pendulum_recalibration_lowers_the_rmse() {
    let r = monte_carlo();
    let ekf = (r.ekf.rmse(), r.ekf_recalibrated.rmse());
    let ukf = (r.ukf.rmse(), r.ukf_recalibrated.rmse());
    println!(
        "pendulum, {RUNS} runs, σz = {SIGMA_Z}: RMSE EKF {:.3e} -> {:.3e} (median {:.3e} -> {:.3e}, \
         backed out {:.1}%), UKF {:.3e} -> {:.3e} (median {:.3e} -> {:.3e}, backed out {:.1}%)",
        ekf.0,
        ekf.1,
        r.ekf.median(),
        r.ekf_recalibrated.median(),
        100.0 * r.ekf_recalibrated.backed_out_fraction(),
        ukf.0,
        ukf.1,
        r.ukf.median(),
        r.ukf_recalibrated.median(),
        100.0 * r.ukf_recalibrated.backed_out_fraction(),
    );

    for rmse in [ekf.0, ekf.1, ukf.0, ukf.1] {
        assert!(rmse.is_finite(), "RMSE = {rmse}");
    }
    assert!(ekf.1 < ekf.0, "EKF: {ekf:?}");
    assert!(ukf.1 < ukf.0, "UKF: {ukf:?}");

    // Well below, not just by a hair: the paper reports an order of magnitude or more.
    assert!(ekf.1 < ekf.0 / 10.0, "EKF: {ekf:?}");
    assert!(ukf.1 < ukf.0 / 3.0, "UKF: {ukf:?}");

    // The recalibrated filters converge in a typical run, rather than merely staying put.
    let initial = initial_variance().sqrt();
    for stats in [&r.ekf_recalibrated, &r.ukf_recalibrated] {
        let median = stats.median();
        assert!(median < initial / 10.0, "median error {median}");
    }

    // The back-out is a fallback, not the rule.
    for fraction in [
        r.ekf_recalibrated.backed_out_fraction(),
        r.ukf_recalibrated.backed_out_fraction(),
    ] {
        assert!(fraction > 0.0 && fraction < 0.5, "backed out {fraction}");
    }
}

/// The scalar observation of the paper's Eq. 27, `h(x) = x³/3 − x²/8 − x + 1.5383`.
fn cubic(x: f64) -> f64 {
    x.powi(3) / 3.0 - x * x / 8.0 - x + 1.5383
}

/// The derivative of [`cubic`].
fn cubic_slope(x: f64) -> f64 {
    x * x - x / 4.0 - 1.0
}

/// The prior variance `1.5²` of the Eq. 27 example.
const CUBIC_PRIOR_VARIANCE: f64 = 2.25;

/// The measurement variance `0.01²` of the Eq. 27 example.
const CUBIC_NOISE_VARIANCE: f64 = 1e-4;

/// The state, the variance, the NIS and the status after one correction.
type Outcome = (f64, f64, f64, CorrectionStatus);

/// How a scalar extended correction is run.
#[derive(Clone, Copy)]
enum Ekf {
    /// Through the filter's `correct_nonlinear_recalibrated`.
    Filter,
    /// Through the filter's `correct_nonlinear_recalibrated_with`, with a `fix` that zeroes
    /// the innovation.
    FilterZeroInnovation,
    /// Through the observation's inherent `correct_nonlinear_recalibrated`.
    Observation,
    /// Through [`KalmanFilterNonlinearObservationCorrectRecalibrated`] on the observation.
    ObservationTrait,
}

/// Runs one recalibrated correction of the Eq. 27 example with a scalar extended filter,
/// starting from `x = 0`, `P = p`, with measurement `z = 0` and noise variance `r`.
fn cubic_ekf(how: Ekf, p: f64, r: f64, gate: Option<f64>) -> Outcome {
    let builder = ExtendedBuilder::<1, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().make_scalar(p);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(r);
    measurement.measurement_vector_mut().set_at(0, 0, 0.0);
    measurement
        .observation_jacobian_matrix_mut()
        .set_at(0, 0, cubic_slope(0.0));
    measurement.set_gate(gate);

    let observation = |x: &_, y: &mut _| set_scalar(y, cubic(state1(x)));
    let jacobian = |x: &_, h: &mut _| set_scalar_jacobian(h, cubic_slope(state1(x)));

    // Standalone buffers for the observation-level calls.
    let mut x = BufferBuilder::state_vector_x::<1>().new();
    let mut cov = BufferBuilder::estimate_covariance_P::<1>().new();
    cov.make_scalar(p);

    let status = match how {
        Ekf::Filter => {
            filter.correct_nonlinear_recalibrated(&mut measurement, observation, jacobian)
        }
        Ekf::FilterZeroInnovation => filter.correct_nonlinear_recalibrated_with(
            &mut measurement,
            observation,
            jacobian,
            |y| y[0] = 0.0,
        ),
        Ekf::Observation => {
            let status =
                measurement.correct_nonlinear_recalibrated(&mut x, &mut cov, observation, jacobian);
            return (x.get_at(0, 0), cov.get_at(0, 0), measurement.nis(), status);
        }
        Ekf::ObservationTrait => {
            let status =
                KalmanFilterNonlinearObservationCorrectRecalibrated::correct_nonlinear_recalibrated(
                    &mut measurement,
                    &mut x,
                    &mut cov,
                    observation,
                    jacobian,
                );
            return (x.get_at(0, 0), cov.get_at(0, 0), measurement.nis(), status);
        }
    };

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
        measurement.nis(),
        status,
    )
}

/// Writes a scalar observation.
fn set_scalar(y: &mut impl MatrixMut<1, 1, f64>, value: f64) {
    y.set_at(0, 0, value);
}

/// Writes a scalar observation Jacobian.
fn set_scalar_jacobian(h: &mut impl MatrixMut<1, 1, f64>, value: f64) {
    h.set_at(0, 0, value);
}

/// Runs one correction of the Eq. 27 example with a scalar unscented filter, starting from
/// `x = 0`, `P = p`, with measurement `z = 0` and noise variance `r`. `fix` normalizes the
/// innovation and is only used by the recalibrated correction.
fn cubic_ukf(recalibrate: bool, p: f64, r: f64, gate: Option<f64>, fix: fn(&mut [f64])) -> Outcome {
    let builder = UnscentedBuilder::<1, f64>::default();
    let mut filter = builder.build::<3>();
    let mut measurement = builder.observations().build::<1, 3>();
    filter.estimate_covariance_mut().make_scalar(p);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(r);
    measurement.measurement_vector_mut().set_at(0, 0, 0.0);
    measurement.set_gate(gate);

    let observation = |sigma: &_, observed: &mut _| observe_cubic_sigma_points(sigma, observed);
    let status = if recalibrate {
        filter.correct_sigma_point_recalibrated_with(&mut measurement, observation, fix)
    } else {
        filter.correct_sigma_point(&mut measurement, observation)
    };

    (
        filter.state_vector().get_at(0, 0),
        filter.estimate_covariance().get_at(0, 0),
        measurement.nis(),
        status,
    )
}

/// Reads a one-state vector.
fn state1(x: &impl Matrix<1, 1, f64>) -> f64 {
    x.get_at(0, 0)
}

/// Observes each of the three sigma points of a one-state filter through [`cubic`].
fn observe_cubic_sigma_points(
    sigma: &impl Matrix<1, 3, f64>,
    observed: &mut impl MatrixMut<1, 3, f64>,
) {
    for j in 0..3 {
        observed.set_at(0, j, cubic(sigma.get_at(0, j)));
    }
}

/// The NIS `y²/S` of the Eq. 27 example at the prior `x = 0`, for the extended filter.
fn cubic_nis() -> f64 {
    let s = cubic_slope(0.0).powi(2) * CUBIC_PRIOR_VARIANCE + CUBIC_NOISE_VARIANCE;
    cubic(0.0).powi(2) / s
}

#[test]
fn cubic_example_is_backed_out_by_the_extended_filter() {
    // Section 2.3: the slope at the prior points away from the true state, so the
    // recalibrated covariance grows and the measurement is backed out.
    for how in [Ekf::Filter, Ekf::Observation, Ekf::ObservationTrait] {
        let (x, p, nis, status) = cubic_ekf(how, CUBIC_PRIOR_VARIANCE, CUBIC_NOISE_VARIANCE, None);
        assert_eq!(status, CorrectionStatus::BackedOut);
        assert_eq!((x, p), (0.0, CUBIC_PRIOR_VARIANCE));
        assert!((nis - cubic_nis()).abs() < 1e-12, "NIS = {nis}");
    }
}

#[test]
fn cubic_example_is_backed_out_by_the_unscented_filter() {
    let (x, p, nis, status) = cubic_ukf(
        true,
        CUBIC_PRIOR_VARIANCE,
        CUBIC_NOISE_VARIANCE,
        None,
        |_| {},
    );
    assert_eq!(status, CorrectionStatus::BackedOut);
    assert_eq!((x, p), (0.0, CUBIC_PRIOR_VARIANCE));
    assert!(nis > 0.0, "NIS = {nis}");

    // The plain filter applies the same measurement, with the same NIS.
    let (_, _, plain_nis, status) = cubic_ukf(
        false,
        CUBIC_PRIOR_VARIANCE,
        CUBIC_NOISE_VARIANCE,
        None,
        |_| {},
    );
    assert_eq!(status, CorrectionStatus::Applied);
    assert_eq!(nis, plain_nis);
}

#[test]
fn recalibrated_corrections_apply_a_small_step() {
    // With a tight prior, h is nearly linear over the step and the update is applied.
    let p = 1e-4;
    let expected_x = {
        let h = cubic_slope(0.0);
        let k = p * h / (h * h * p + CUBIC_NOISE_VARIANCE);
        -k * cubic(0.0)
    };

    let (x, cov, _, status) = cubic_ekf(Ekf::Filter, p, CUBIC_NOISE_VARIANCE, None);
    assert_eq!(status, CorrectionStatus::Applied);
    assert!((x - expected_x).abs() < 1e-12, "x = {x}");
    assert!(cov > 0.0 && cov < p, "P = {cov}");

    let (x, cov, _, status) = cubic_ukf(true, p, CUBIC_NOISE_VARIANCE, None, |_| {});
    assert_eq!(status, CorrectionStatus::Applied);
    assert!(x > 0.0 && x < 2.0 * expected_x, "x = {x}");
    assert!(cov > 0.0 && cov < p, "P = {cov}");
}

#[test]
fn recalibrated_corrections_apply_the_fix() {
    // A fix that zeroes the innovation leaves the state at the prior, while the
    // covariance still shrinks.
    let p = 1e-4;
    let (x, cov, _, status) = cubic_ekf(Ekf::FilterZeroInnovation, p, CUBIC_NOISE_VARIANCE, None);
    assert_eq!(status, CorrectionStatus::Applied);
    assert_eq!(x, 0.0);
    assert!(cov < p, "P = {cov}");

    // The unscented fix also normalizes the sigma point spread; zeroing it zeroes the gain.
    // An update that leaves the trace unchanged is applied, not backed out.
    let (x, cov, _, status) = cubic_ukf(true, p, CUBIC_NOISE_VARIANCE, None, |y| y[0] = 0.0);
    assert_eq!(status, CorrectionStatus::Applied);
    assert_eq!((x, cov), (0.0, p));
}

#[test]
fn recalibrated_corrections_respect_the_gate() {
    // The gate is checked on the first pass, at the prior.
    let gate = Some(cubic_nis() / 2.0);
    let (x, p, nis, status) = cubic_ekf(
        Ekf::Filter,
        CUBIC_PRIOR_VARIANCE,
        CUBIC_NOISE_VARIANCE,
        gate,
    );
    assert_eq!(status, CorrectionStatus::Gated);
    assert_eq!((x, p), (0.0, CUBIC_PRIOR_VARIANCE));
    assert!((nis - cubic_nis()).abs() < 1e-12, "NIS = {nis}");

    let (x, p, _, status) = cubic_ukf(
        true,
        CUBIC_PRIOR_VARIANCE,
        CUBIC_NOISE_VARIANCE,
        Some(1e-6),
        |_| {},
    );
    assert_eq!(status, CorrectionStatus::Gated);
    assert_eq!((x, p), (0.0, CUBIC_PRIOR_VARIANCE));
}

#[test]
fn recalibrated_corrections_skip_a_singular_innovation_covariance() {
    let (x, p, _, status) = cubic_ekf(Ekf::Filter, 0.0, 0.0, None);
    assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!((x, p), (0.0, 0.0));

    let (x, p, _, status) = cubic_ukf(true, 0.0, 0.0, None, |_| {});
    assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!((x, p), (0.0, 0.0));
}

/// Sets the correlated two-state prior covariance of the linear example.
fn linear_prior(p: &mut impl MatrixMut<2, 2, f64>) {
    p.set_at(0, 0, 1.0);
    p.set_at(0, 1, 0.3);
    p.set_at(1, 0, 0.3);
    p.set_at(1, 1, 2.0);
}

/// The linear observation `h(x) = x₀ + 0.5·x₁`.
fn linear(s: State) -> f64 {
    s[0] + 0.5 * s[1]
}

/// Writes the Jacobian `[1, 0.5]` of [`linear`].
fn set_linear_jacobian(h: &mut impl MatrixMut<1, 2, f64>) {
    h.set_at(0, 0, 1.0);
    h.set_at(0, 1, 0.5);
}

/// The state and the row-major covariance of a two-state filter.
type Flat = ([f64; 2], [f64; 4]);

/// Reads the state and the covariance of a two-state filter.
fn flatten(x: &impl Matrix<2, 1, f64>, p: &impl Matrix<2, 2, f64>) -> Flat {
    (
        state_of(x),
        [
            p.get_at(0, 0),
            p.get_at(0, 1),
            p.get_at(1, 0),
            p.get_at(1, 1),
        ],
    )
}

/// Runs one correction of the linear example, `z = 1`, `R = 0.1`, with the extended filter.
fn linear_ekf(recalibrate: bool) -> (Flat, CorrectionStatus) {
    let builder = ExtendedBuilder::<2, f64>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();
    filter.estimate_covariance_mut().apply(linear_prior);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(0.1);
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);
    set_linear_jacobian(measurement.observation_jacobian_matrix_mut());

    let observation = |x: &_, y: &mut _| set_scalar(y, linear(state_of(x)));
    let status = if recalibrate {
        filter.correct_nonlinear_recalibrated(&mut measurement, observation, |_, h| {
            set_linear_jacobian(h)
        })
    } else {
        filter.correct_nonlinear(&mut measurement, observation)
    };

    (
        flatten(filter.state_vector(), filter.estimate_covariance()),
        status,
    )
}

/// Runs one correction of the linear example, `z = 1`, `R = 0.1`, with the unscented filter.
fn linear_ukf(recalibrate: bool) -> (Flat, CorrectionStatus) {
    let builder = UnscentedBuilder::<2, f64>::default();
    let mut filter = builder.build::<5>();
    let mut measurement = builder.observations().build::<1, 5>();
    filter.estimate_covariance_mut().apply(linear_prior);
    measurement
        .measurement_noise_covariance_mut()
        .make_scalar(0.1);
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);

    let observation = |sigma: &_, observed: &mut _| observe_linear_sigma_points(sigma, observed);
    let status = if recalibrate {
        filter.correct_sigma_point_recalibrated(&mut measurement, observation)
    } else {
        filter.correct_sigma_point(&mut measurement, observation)
    };

    (
        flatten(filter.state_vector(), filter.estimate_covariance()),
        status,
    )
}

/// Observes each of the five sigma points of a two-state filter through [`linear`].
fn observe_linear_sigma_points(
    sigma: &impl Matrix<2, 5, f64>,
    observed: &mut impl MatrixMut<1, 5, f64>,
) {
    for j in 0..5 {
        observed.set_at(0, j, linear([sigma.get_at(0, j), sigma.get_at(1, j)]));
    }
}

/// Asserts that two filter outcomes agree to `1e-12`.
fn assert_close(a: Flat, b: Flat) {
    for (u, v) in a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)) {
        assert!((u - v).abs() < 1e-12, "{a:?} != {b:?}");
    }
}

#[test]
fn recalibration_of_a_linear_observation_changes_nothing() {
    // Section 2.3: for a linear h, the recalibrated and the conventional filter agree.
    for correct in [linear_ekf, linear_ukf] {
        let (plain, plain_status) = correct(false);
        let (recalibrated, status) = correct(true);
        assert_eq!(plain_status, CorrectionStatus::Applied);
        assert_eq!(status, CorrectionStatus::Applied);
        assert_close(plain, recalibrated);
    }
}
