//! Numerical-robustness test bench: how the estimate covariance `P` of the regular filter
//! degrades under the standard update `P ← P − K×H×P` and under the Joseph form, in `f32`,
//! `f64`, `I16F16` and `I32F32`.
//!
//! Every scenario runs each type and covariance update and records, per step, whether `P`
//! passes a Cholesky decomposition in its own type, its smallest diagonal entry, its largest
//! relative asymmetry and the NEES against an `f64` reference run (see `common/robustness.rs`).
//! It prints one report line per run; see them with
//! `cargo nextest run --features=std,fixed --test numerical_robustness --no-capture`.
//! The asserts pin what was found, so a change in the numerics shows up here.
//!
//! Fixed-point overflow panics in debug builds (the tests' default) and wraps in release
//! builds; the bench catches the panic and reports the run as overflowed.

#![cfg(all(feature = "std", feature = "fixed"))]
#![forbid(unsafe_code)]

#[path = "common/robustness.rs"]
mod robustness;

use fixed::types::{I16F16, I32F32};
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;
use robustness::{BenchType, Noise, Run, Summary, deltas, measure, num, to_f64_vector};

/// Both covariance updates, with their report labels.
const UPDATES: [(CovarianceUpdate, &str); 2] = [
    (CovarianceUpdate::Standard, "standard"),
    (CovarianceUpdate::Joseph, "joseph"),
];

/// Records the state of `filter` and the metrics of its `P`, with the NEES against step
/// `step` of `reference`, if given.
fn record_step<const N: usize, T, F>(
    filter: &F,
    reference: Option<&[[f64; N]]>,
    states: &mut Vec<[f64; N]>,
    summary: &mut Summary,
) where
    T: BenchType,
    F: KalmanFilterStateVector<N, T> + KalmanFilterEstimateCovariance<N, T>,
{
    let state = to_f64_vector(filter.state_vector().as_matrix());
    let error = reference.map_or([0.0; N], |r| {
        core::array::from_fn(|i| state[i] - r[states.len()][i])
    });

    states.push(state);
    summary.record(measure(filter.estimate_covariance().as_matrix(), error));
}

/// Scenario 1, one correction of the textbook ill-conditioned measurement: `P₀ = I₃`, two
/// nearly identical measurement rows `H = [[1,1,1],[1,1,1+δ]]` and `R = δ²×I`. As `δ²` drops
/// towards the unit roundoff of `T`, `S` becomes numerically singular and `P⁺` loses rank.
///
/// Attributed to Grewal and Andrews, *Kalman Filtering: Theory and Practice Using MATLAB*,
/// in the section on the effects of roundoff errors.
fn ill_conditioned<T: BenchType>(
    delta: f64,
    update: CovarianceUpdate,
    reference: Option<&[[f64; 3]]>,
    states: &mut Vec<[f64; 3]>,
) -> Summary {
    let builder = KalmanFilterBuilder::<3, T>::default();
    let mut filter = builder.build();
    let mut z = builder.observations().build::<2>();

    filter.estimate_covariance_mut().make_identity();
    z.set_covariance_update(update);
    z.observation_matrix_mut().apply(|h| {
        for j in 0..3 {
            h.set_at(0, j, T::one());
            h.set_at(1, j, T::one());
        }
        h.set_at(1, 2, num(1.0 + delta));
    });
    z.measurement_noise_covariance_mut()
        .make_scalar(num(delta * delta));
    z.measurement_vector_mut().apply(|v| {
        v.set_at(0, 0, num(0.75));
        v.set_at(1, 0, num(0.75 + 0.25 * delta));
    });

    let mut summary = Summary::default();
    summary.record_status(filter.correct(&mut z));
    record_step(&filter, reference, states, &mut summary);
    summary
}

/// Scenario 2, a long fixed-point run: a constant-velocity model with `P₀ = 100×I`, a precise
/// position sensor (`R = 0.01`) and a process noise of `10⁻⁶`, below the resolution of
/// `I16F16`, so that `K×H×P` comes close to `P` and stays there.
fn constant_velocity<T: BenchType>(
    steps: usize,
    update: CovarianceUpdate,
    reference: Option<&[[f64; 2]]>,
    states: &mut Vec<[f64; 2]>,
) -> Summary {
    const DT: f64 = 0.1;

    let builder = KalmanFilterBuilder::<2, T>::default();
    let mut filter = builder.build();
    let mut z = builder.observations().build::<1>();

    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 1, num(DT));
    });
    filter.direct_process_noise_mut().apply(|q| {
        q.set_at(0, 0, num(1e-6));
        q.set_at(1, 1, num(1e-6));
    });
    filter.estimate_covariance_mut().make_scalar(num(100.0));
    z.set_covariance_update(update);
    z.observation_matrix_mut().set_at(0, 0, T::one());
    z.measurement_noise_covariance_mut().make_scalar(num(0.01));

    let mut noise = Noise(0x2545_F491_4F6C_DD1D);
    let mut summary = Summary::default();
    for step in 0..steps {
        filter.predict();
        let position = 0.5 * DT * step as f64;
        z.measurement_vector_mut()
            .set_at(0, 0, num(position + 0.1 * noise.next()));
        summary.record_status(filter.correct(&mut z));
        record_step(&filter, reference, states, &mut summary);
    }

    summary
}

/// Number of scalar sensors per prediction in [`sequential_scalar`].
const SENSORS: usize = 10;

/// Scenario 3, sequential scalar updates: a 2-D constant-velocity model (`x, y, vx, vy`)
/// with [`SENSORS`] scalar position sensors along directions 18° apart, each with
/// `R = 10⁻⁴`, applied one after another between two predictions. The steady-state position
/// variance (about `10⁻⁵`) is below the resolution of `I16F16` (`2⁻¹⁶ ≈ 1.5×10⁻⁵`).
fn sequential_scalar<T: BenchType>(
    steps: usize,
    update: CovarianceUpdate,
    reference: Option<&[[f64; 4]]>,
    states: &mut Vec<[f64; 4]>,
) -> Summary {
    const DT: f64 = 0.1;

    let builder = KalmanFilterBuilder::<4, T>::default();
    let mut filter = builder.build();
    let mut sensors: Vec<_> = (0..SENSORS)
        .map(|_| builder.observations().build::<1>())
        .collect();

    filter.state_transition_mut().apply(|a| {
        a.make_identity();
        a.set_at(0, 2, num(DT));
        a.set_at(1, 3, num(DT));
    });
    filter
        .direct_process_noise_mut()
        .apply(|q| (0..4).for_each(|i| q.set_at(i, i, num(1e-5))));
    filter.estimate_covariance_mut().make_scalar(num(10.0));

    let directions: Vec<(f64, f64)> = (0..SENSORS)
        .map(|k| (k as f64 * 18f64).to_radians().sin_cos())
        .collect();
    for (sensor, &(sin, cos)) in sensors.iter_mut().zip(&directions) {
        sensor.set_covariance_update(update);
        sensor.observation_matrix_mut().apply(|h| {
            h.set_at(0, 0, num(cos));
            h.set_at(0, 1, num(sin));
        });
        sensor
            .measurement_noise_covariance_mut()
            .make_scalar(num(1e-4));
    }

    let mut noise = Noise(0xD1B5_4A32_D192_ED03);
    let mut summary = Summary::default();
    for step in 0..steps {
        filter.predict();
        let t = DT * step as f64;
        let (px, py) = (0.3 * t, -0.2 * t);
        for (sensor, &(sin, cos)) in sensors.iter_mut().zip(&directions) {
            let reading = cos * px + sin * py + 0.01 * noise.next();
            sensor.measurement_vector_mut().set_at(0, 0, num(reading));
            summary.record_status(filter.correct(sensor));
        }

        record_step(&filter, reference, states, &mut summary);
    }

    summary
}

/// The outcome of scenario 1 over a sweep of `δ`, for one type and covariance update.
#[derive(Debug, Default)]
struct Sweep {
    /// The metrics of all applied and rejected corrections; one step per `δ`.
    summary: Summary,

    /// Number of `δ` whose correction overflowed.
    overflowed: usize,

    /// The largest `δ` whose applied correction left `P` not positive definite.
    largest_not_positive_definite: Option<f64>,

    /// The largest `δ` whose correction was rejected because `S` was not positive definite.
    largest_rejected: Option<f64>,
}

impl Sweep {
    /// Runs one correction of [`ill_conditioned`] per `δ` in `deltas` in type `T`, with the
    /// NEES against the `f64` Joseph run of the same `δ`.
    fn run<T: BenchType>(deltas: &[f64], update: CovarianceUpdate) -> Self {
        let mut sweep = Self::default();
        for &delta in deltas {
            let reference =
                Run::caught(|s| ill_conditioned::<f64>(delta, CovarianceUpdate::Joseph, None, s));
            let run =
                Run::caught(|s| ill_conditioned::<T>(delta, update, Some(&reference.states), s));
            sweep.add(delta, run.summary);
        }

        sweep
    }

    /// Adds the one-step outcome of `δ`.
    fn add(&mut self, delta: f64, step: Option<Summary>) {
        let Some(step) = step else {
            self.overflowed += 1;
            return;
        };

        if step.skipped > 0 {
            self.largest_rejected.get_or_insert(delta);
        }

        if step.not_positive_definite > 0 {
            self.largest_not_positive_definite.get_or_insert(delta);
        }

        let summary = &mut self.summary;
        summary.steps += 1;
        summary.skipped += step.skipped;
        summary.not_positive_definite += step.not_positive_definite;
        summary.nees_undefined += step.nees_undefined;
        summary.min_diagonal = summary.min_diagonal.min(step.min_diagonal);
        summary.max_asymmetry = summary.max_asymmetry.max(step.max_asymmetry);
        summary.max_nees = summary.max_nees.max(step.max_nees);
    }

    /// Prints the sweep as one line of the bench report.
    fn report(&self, label: &str) {
        let delta = |d: Option<f64>| d.map_or("-".to_string(), |d| format!("{d:.1e}"));
        self.summary.report(label);
        println!(
            "{:<24} overflowed {:>3}  largest δ not PD {:>8}  largest δ rejected {:>8}",
            "",
            self.overflowed,
            delta(self.largest_not_positive_definite),
            delta(self.largest_rejected),
        );
    }
}

/// Scenario 1 over 48 values of `δ` per type.
///
/// Found: the exact `P⁺` has two eigenvalues of order `δ²`, so once `δ²` approaches the
/// roundoff of `T`, whether the computed `P⁺` passes a Cholesky decomposition is a matter of
/// rounding, and both forms leave it indefinite for some `δ` in every type. From `δ ≈ 5×10⁻⁴`
/// in `f32`, `S` is rejected as not positive definite and neither form applies the correction.
/// The gain is solved from the Cholesky factor of `S` rather than formed with `S⁻¹ ~ 1/δ²`,
/// so `I16F16` no longer overflows, and the corrected state stays close to the `f64` one
/// (`|Δx|` below `10⁻⁴` in `f32`, against about `0.4` with the explicit inverse). With that
/// gain (square-root free there), in `f32` and `f64` the standard update is indefinite for no
/// more `δ` than the Joseph form (8 and 8 of 48 in `f32`, 3 and 8 in `f64`); with the explicit
/// inverse it was 16 and
/// 7, 11 and 9. The fixed-point types are not ordered that way (`I32F32`: 27 and 11).
#[test]
fn ill_conditioned_measurement() {
    let deltas = deltas(48);
    println!("scenario 1: ill-conditioned measurement, one correction per δ");
    let mut sweeps = Vec::new();
    for (update, name) in UPDATES {
        let runs = [
            Sweep::run::<f32>(&deltas, update),
            Sweep::run::<f64>(&deltas, update),
            Sweep::run::<I16F16>(&deltas, update),
            Sweep::run::<I32F32>(&deltas, update),
        ];
        for (run, ty) in runs.iter().zip(["f32", "f64", "I16F16", "I32F32"]) {
            run.report(&format!("{ty} {name}"));
        }

        sweeps.push(runs);
    }

    let [standard, joseph] = [&sweeps[0], &sweeps[1]];
    for (standard, joseph) in standard.iter().zip(joseph) {
        let (s, j) = (&standard.summary, &joseph.summary);
        assert!(
            s.not_positive_definite > 0,
            "standard keeps P positive definite"
        );
        assert!(
            j.not_positive_definite > 0,
            "Joseph keeps P positive definite"
        );
    }

    assert!(joseph.iter().all(|run| run.summary.max_asymmetry == 0.0));
    assert!(
        standard.iter().chain(joseph).all(|run| run.overflowed == 0),
        "no type overflows"
    );
}

/// Scenario 2 for `10⁴` steps per type.
///
/// Found: no type loses positive definiteness with either form. The standard update leaves
/// `P` slightly asymmetric (relative `10⁻⁷` in `f32`, `10⁻²` in `I16F16`); the Joseph form
/// keeps it exactly symmetric.
#[test]
fn long_fixed_point_run() {
    const STEPS: usize = 10_000;

    println!("scenario 2: constant velocity, {STEPS} steps");
    let reference =
        Run::caught(|s| constant_velocity::<f64>(STEPS, CovarianceUpdate::Joseph, None, s));
    let r = Some(reference.states.as_slice());
    for (update, name) in UPDATES {
        let runs = [
            Run::caught(|s| constant_velocity::<f32>(STEPS, update, r, s)),
            Run::caught(|s| constant_velocity::<f64>(STEPS, update, r, s)),
            Run::caught(|s| constant_velocity::<I16F16>(STEPS, update, r, s)),
            Run::caught(|s| constant_velocity::<I32F32>(STEPS, update, r, s)),
        ];
        for (run, ty) in runs.iter().zip(["f32", "f64", "I16F16", "I32F32"]) {
            run.report(&format!("{ty} {name}"));
            let summary = run.completed();
            assert_eq!(summary.not_positive_definite, 0, "{ty} {name}");
            assert_eq!(summary.skipped, 0, "{ty} {name}");
            match update {
                CovarianceUpdate::Joseph => assert_eq!(summary.max_asymmetry, 0.0),
                _ if ty != "f64" => assert!(summary.max_asymmetry > 0.0, "{ty} {name}"),
                _ => {}
            }
        }
    }
}

/// Scenario 3 for `10³` predictions with [`SENSORS`] corrections each, per type.
///
/// Found: in `I16F16`, the standard update leaves `P` indefinite in some steps, and the
/// Joseph form does not fix it: it diverges (`P` indefinite from the second step on; with the
/// explicit `S⁻¹` it overflowed while computing the gain, in debug builds).
/// The cause is the resolution, not the form: the steady-state variance is below one LSB.
/// The floating-point types and `I32F32` stay positive definite with both forms.
#[test]
fn sequential_scalar_updates() {
    const STEPS: usize = 1_000;

    println!("scenario 3: {SENSORS} sequential scalar updates per prediction, {STEPS} steps");
    let reference =
        Run::caught(|s| sequential_scalar::<f64>(STEPS, CovarianceUpdate::Joseph, None, s));
    let r = Some(reference.states.as_slice());
    let mut i16f16 = Vec::new();
    for (update, name) in UPDATES {
        let runs = [
            Run::caught(|s| sequential_scalar::<f32>(STEPS, update, r, s)),
            Run::caught(|s| sequential_scalar::<f64>(STEPS, update, r, s)),
            Run::caught(|s| sequential_scalar::<I16F16>(STEPS, update, r, s)),
            Run::caught(|s| sequential_scalar::<I32F32>(STEPS, update, r, s)),
        ];
        for (run, ty) in runs.iter().zip(["f32", "f64", "I16F16", "I32F32"]) {
            run.report(&format!("{ty} {name}"));
            if ty != "I16F16" {
                assert_eq!(run.completed().not_positive_definite, 0, "{ty} {name}");
            }
        }

        let [_, _, fixed, _] = runs;
        i16f16.push(fixed.summary);
    }

    let standard = i16f16[0].expect("standard I16F16 completes");
    assert!(standard.not_positive_definite > 0);
    // The Joseph run completes (no overflow in the gain) and diverges instead.
    let joseph = i16f16[1].expect("Joseph I16F16 completes");
    assert!(joseph.not_positive_definite > 0);
}
