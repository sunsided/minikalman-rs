//! The fixed-gain (Luenberger) observer and `settle_gain`, checked against SciPy and against
//! the regular Kalman filter on the same data.
//!
//! The model is the constant-velocity tracker `motion::constant_velocity(0.1, 1.0)` with a
//! position measurement of variance `0.25`. The reference values come from SciPy's
//! `solve_discrete_are(A.T, H.T, Q, R)` (scipy 1.11.1).

#![cfg(feature = "alloc")]
#![forbid(unsafe_code)]

use minikalman::models::motion;
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use minikalman::num_traits::FromPrimitive;
use minikalman::prelude::*;
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use minikalman::regular::builder::{
    FixedGainObservationType, GatedFixedGainObservationType, KalmanFilterBuilder,
    KalmanFilterObservationType, KalmanFilterType, LuenbergerObserverType, PredictorObserverType,
};
use minikalman::regular::{
    FixedGainObservationStorage, GatedFixedGainObservationStorage, ObserverStorage,
    PredictorObserverStorage,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Time step in seconds.
const DT: f64 = 0.1;

/// Acceleration variance of the constant-velocity model, in (m/s²)².
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const ACCELERATION_VARIANCE: f64 = 1.0;

/// Measurement noise variance of the position, in m².
const R: f64 = 0.25;

/// Steady-state current-form gain `K∞` from SciPy.
const K_INF: [f64; 2] = [0.181_201_093_164_732_62, 0.180_975_015_605_498_97];

/// Steady-state predictor-form gain `L∞ = A·K∞` from SciPy.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const L_INF: [f64; 2] = [0.199_298_594_725_282_52, 0.180_975_015_605_498_97];

/// Steady-state corrected covariance `P⁺∞` from SciPy, row-major.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const P_POST: [f64; 4] = [
    0.045_300_273_291_183_16,
    0.045_243_753_901_374_75,
    0.045_243_753_901_374_75,
    0.095_124_921_972_503_88,
];

/// Steady-state innovation covariance `S∞` from SciPy.
const S_INF: f64 = 0.305_325_273_291_183_14;

/// Spectral radius of the error dynamics `(I − K∞·H)·A`, from numpy.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const RHO: f64 = 0.904_875_078_027_496_2;

/// A design-time Kalman filter and its position observation, set up for the CV model.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn design<T>(p0: f64) -> (KalmanFilterType<2, T>, KalmanFilterObservationType<2, 1, T>)
where
    T: MatrixDataType + Default + FromPrimitive,
{
    let builder = KalmanFilterBuilder::<2, T>::default();
    let mut filter = builder.build();
    let mut position = builder.observations().build::<1>();
    let model = motion::constant_velocity(of::<T>(DT), of::<T>(ACCELERATION_VARIANCE));

    filter
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    filter
        .direct_process_noise_mut()
        .apply(|q| model.fill_process_noise(q));
    filter.estimate_covariance_mut().make_scalar(of::<T>(p0));
    position.observation_matrix_mut().set_at(0, 0, T::one());
    position
        .measurement_noise_covariance_mut()
        .set_at(0, 0, of::<T>(R));
    (filter, position)
}

/// An `f64` observer and its position observation for the CV model, with no gain set yet.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn cv_observer() -> (
    LuenbergerObserverType<2, f64>,
    FixedGainObservationType<2, 1, f64>,
) {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut observer = builder.build_observer();
    let mut position = builder.observations().build_fixed_gain::<1, Current>();
    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    (observer, position)
}

/// Converts an `f64` to `T`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn of<T: FromPrimitive>(value: f64) -> T {
    T::from_f64(value).expect("representable")
}

/// A 2 × 1 matrix with the given entries, to pass a gain to the setters.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn column(values: [f64; 2]) -> impl Matrix<2, 1, f64> {
    MatrixData::new_array::<2, 1, 2, f64>(values)
}

/// Noisy position readings of a target that starts at 1 m and moves at 2 m/s.
fn readings(count: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let noise = Normal::new(0.0, R.sqrt()).expect("valid distribution");
    (0..count)
        .map(|k| 1.0 + 2.0 * DT * (k + 1) as f64 + noise.sample(&mut rng))
        .collect()
}

/// Reads a 2-vector out of a state buffer.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn state(x: &impl Matrix<2, 1, f64>) -> [f64; 2] {
    [x.get_at(0, 0), x.get_at(1, 0)]
}

/// Largest absolute entry-wise difference of two 2-vectors.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn gap(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
}

/// `settle_gain` reproduces SciPy's DARE in `f64`, in both covariance update forms and from
/// two different starting covariances.
#[test]
#[cfg(any(feature = "std", feature = "libm"))]
fn settle_gain_matches_scipy_in_f64() {
    for update in [CovarianceUpdate::Joseph, CovarianceUpdate::Standard] {
        for p0 in [1.0, 1000.0] {
            let (mut filter, mut position) = design::<f64>(p0);
            position.set_covariance_update(update);
            filter.state_vector_mut().set_at(0, 0, 7.0);

            let iterations = filter
                .settle_gain(&mut position, 1e-12, 10_000)
                .expect("detectable");
            assert!((100..200).contains(&iterations), "{update:?}: {iterations}");

            let k = position.kalman_gain();
            let p = filter.estimate_covariance();
            eprintln!(
                "{update:?}, P₀ = {p0}·I: {iterations} iterations, |ΔK| = {:e}, {:e}",
                k.get_at(0, 0) - K_INF[0],
                k.get_at(1, 0) - K_INF[1]
            );
            for (i, expected) in K_INF.iter().enumerate() {
                assert!((k.get_at(i, 0) - expected).abs() < 1e-10, "{update:?} K");
            }

            for (i, expected) in P_POST.iter().enumerate() {
                assert!(
                    (p.get_at(i / 2, i % 2) - expected).abs() < 1e-10,
                    "{update:?} P"
                );
            }

            let s = position.innovation_covariance().get_at(0, 0);
            assert!((s - S_INF).abs() < 1e-10, "{update:?} S = {s}");
            assert_eq!(filter.state_vector().get_at(0, 0), 7.0, "x is untouched");
        }
    }
}

/// `settle_gain` reproduces SciPy's DARE in `f32` to about `1e-5` relative.
#[test]
#[cfg(any(feature = "std", feature = "libm"))]
fn settle_gain_matches_scipy_in_f32() {
    let (mut filter, mut position) = design::<f32>(1.0);
    let iterations = filter
        .settle_gain(&mut position, 1e-6, 10_000)
        .expect("detectable");
    assert!(iterations < 200, "{iterations}");

    let k = position.kalman_gain();
    eprintln!("f32: {iterations} iterations, K = {:?}", k.as_slice());
    for (i, expected) in K_INF.iter().enumerate() {
        let error = (f64::from(k.get_at(i, 0)) - expected).abs();
        assert!(error < 1e-5, "K[{i}] off by {error}");
    }
}

/// The gate and the noise adaptation of the observation do not take part.
#[test]
#[cfg(any(feature = "std", feature = "libm"))]
fn settle_gain_bypasses_gate_and_adaptation() {
    let (mut filter, mut position) = design::<f64>(1.0);
    position.set_gate(Some(0.0));
    position.set_noise_adaptation(Some(NoiseAdaptation {
        forgetting: 0.5,
        floor: 1e-3,
    }));

    filter
        .settle_gain(&mut position, 1e-12, 10_000)
        .expect("detectable");
    assert!((position.kalman_gain().get_at(0, 0) - K_INF[0]).abs() < 1e-10);
    assert_eq!(position.measurement_noise_covariance().get_at(0, 0), R);
}

/// A budget too small to settle reports `NotConverged`; the first iteration never counts.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn settle_gain_reports_not_converged() {
    for budget in [0, 1, 5] {
        let (mut filter, mut position) = design::<f64>(1.0);
        assert_eq!(
            filter.settle_gain(&mut position, 1e-12, budget),
            Err(CorrectionStatus::NotConverged)
        );
    }

    // Even a tolerance that accepts any change needs a second iteration.
    let (mut filter, mut position) = design::<f64>(1.0);
    assert_eq!(filter.settle_gain(&mut position, 1.0, 10), Ok(2));
}

/// An innovation covariance that is not positive definite is reported.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn settle_gain_reports_not_positive_definite() {
    let (mut filter, mut position) = design::<f64>(1.0);
    position
        .measurement_noise_covariance_mut()
        .set_at(0, 0, -1.0);
    filter.estimate_covariance_mut().make_scalar(0.0);
    assert_eq!(
        filter.settle_gain(&mut position, 1e-12, 100),
        Err(CorrectionStatus::NotPositiveDefinite)
    );
}

/// With `K∞`, the observer converges to the Kalman filter's state on the same data, and the
/// gap closes at about `ρ` per step.
#[test]
// micromath's approximate square root keeps the filter's gain away from K∞.
#[cfg(any(feature = "std", feature = "libm"))]
fn observer_converges_to_the_kalman_filter() {
    let (mut filter, mut position) = design::<f64>(10.0);
    let (mut observer, mut fixed) = cv_observer();
    fixed.set_gain(&column(K_INF));
    filter.state_vector_mut().set_at(0, 0, 5.0);
    observer.state_vector_mut().set_at(0, 0, 5.0);

    let mut gaps = Vec::new();
    for z in readings(400) {
        filter.predict();
        position.measurement_vector_mut().set_at(0, 0, z);
        filter.correct(&mut position);

        observer.predict();
        fixed.measurement_vector_mut().set_at(0, 0, z);
        assert_eq!(observer.correct(&mut fixed), CorrectionStatus::Applied);
        gaps.push(gap(
            state(filter.state_vector()),
            state(observer.state_vector()),
        ));
    }

    assert!(gaps[0] > 0.1, "the transients differ: {}", gaps[0]);
    assert!(gaps[399] < 1e-12, "converged: {}", gaps[399]);

    let rate = (gaps[159] / gaps[59]).powf(1.0 / 100.0);
    eprintln!(
        "gaps at 1, 10, 50, 100, 200, 400: {:e} {:e} {:e} {:e} {:e} {:e}; rate {rate}",
        gaps[0], gaps[9], gaps[49], gaps[99], gaps[199], gaps[399]
    );
    assert!((rate - RHO).abs() < 0.01, "gap decays at {rate}, ρ = {RHO}");
}

/// An `f64` predictor-form observer and its position observation for the CV model, with the
/// gain `gain` set.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn cv_predictor(
    gain: [f64; 2],
) -> (
    PredictorObserverType<2, f64>,
    FixedGainObservationType<2, 1, f64, Predictor>,
) {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut observer = builder.build_predictor_observer();
    let mut position = builder.observations().build_fixed_gain::<1, Predictor>();
    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    position.set_gain(&column(gain));
    (observer, position)
}

/// The fused predictor recursion `x̂ ← A·x̂ + L·(z − H·x̂)` of the CV model with `H = [1, 0]`,
/// written with the same floating-point operations as the observer's matrix kernels.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn fused_predictor_step(x: [f64; 2], z: f64, gain: [f64; 2]) -> [f64; 2] {
    let y = z - (1.0 * x[0] + 0.0 * x[1]);
    let ax = [1.0 * x[0] + DT * x[1], 0.0 * x[0] + 1.0 * x[1]];
    [ax[0] + gain[0] * y, ax[1] + gain[1] * y]
}

/// The deferred predictor-form correction followed by `predict` is bit-for-bit the fused
/// recursion `x̂ₖ₊₁ = A·x̂ₖ + L·(zₖ − H·x̂ₖ)`, and `correct` alone leaves the prior untouched.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn deferred_predictor_correction_equals_the_fused_recursion() {
    let (mut observer, mut position) = cv_predictor(L_INF);
    let mut reference = [3.0, -1.0];
    observer.state_vector_mut().apply(|x| {
        x.set_at(0, 0, reference[0]);
        x.set_at(1, 0, reference[1]);
    });

    for z in readings(400) {
        position.measurement_vector_mut().set_at(0, 0, z);
        assert_eq!(observer.correct(&mut position), CorrectionStatus::Applied);
        assert_eq!(state(observer.state_vector()), reference, "still the prior");
        let pending = observer.pending_correction();
        assert_ne!(pending.get_at(1, 0), 0.0);

        observer.predict();
        reference = fused_predictor_step(reference, z, L_INF);
        assert_eq!(state(observer.state_vector()), reference);
        assert_eq!(observer.pending_correction().get_at(0, 0), 0.0);
    }
}

/// The predictor form with `L = A·K` runs one step ahead of the current form: its state is
/// `A·x̂⁺` of the current-form run at every step.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn predictor_form_is_current_form_one_step_ahead() {
    let (mut current, mut current_position) = cv_observer();
    let (mut predictor, mut predictor_position) = cv_predictor(L_INF);
    current_position.set_gain(&column(K_INF));

    // Both start from x̂₀ = [0, 0]; the predictor form starts from its prior A·x̂₀ = x̂₀.
    for z in readings(200) {
        current.predict();
        current_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(current.correct(&mut current_position).is_applied());

        predictor_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(predictor.correct(&mut predictor_position).is_applied());
        predictor.predict();

        let [p, v] = state(current.state_vector());
        let ahead = [p + DT * v, v];
        assert!(gap(ahead, state(predictor.state_vector())) < 1e-12);
    }
}

/// With a control input, the predictor step `correct`, `predict().control(..)` is
/// `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)`: the current-form prior `A·x̂⁺ + B·u` of the next step,
/// from a nonzero start.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn predictor_form_with_control_is_the_next_current_prior() {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut control = builder.controls().build::<1>();
    control.control_matrix_mut().apply(|b| {
        b.set_at(0, 0, 0.5 * DT * DT);
        b.set_at(1, 0, DT);
    });
    control.control_vector_mut().set_at(0, 0, 0.5);
    let x0 = [3.0, -1.0];
    let b_u = [0.25 * DT * DT, 0.5 * DT];

    let (mut current, mut current_position) = cv_observer();
    let (mut predictor, mut predictor_position) = cv_predictor(L_INF);
    current_position.set_gain(&column(K_INF));
    current.state_vector_mut().apply(|x| {
        x.set_at(0, 0, x0[0]);
        x.set_at(1, 0, x0[1]);
    });

    // The predictor form starts from the current form's first prior A·x̂₀ + B·u.
    predictor.state_vector_mut().apply(|x| {
        x.set_at(0, 0, x0[0] + DT * x0[1] + b_u[0]);
        x.set_at(1, 0, x0[1] + b_u[1]);
    });

    for z in readings(200) {
        current.predict().control(&mut control);
        assert!(
            gap(
                state(current.state_vector()),
                state(predictor.state_vector())
            ) < 1e-12
        );

        current_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(current.correct(&mut current_position).is_applied());

        predictor_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(predictor.correct(&mut predictor_position).is_applied());
        predictor.predict().control(&mut control);
    }
}

/// Two predictor-form corrections in one step add their terms, each formed at the same prior:
/// together they equal one stacked observation with `H = I` and `L = [L₁ L₂]`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn predictor_corrections_accumulate_like_a_stacked_observation() {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let (mut observer, mut position) = cv_predictor(L_INF);
    let mut velocity = builder.observations().build_fixed_gain::<1, Predictor>();
    velocity.observation_matrix_mut().set_at(0, 1, 1.0);
    velocity.set_gain(&column([0.05, 0.3]));

    let mut stacked_observer = builder.build_predictor_observer();
    let mut stacked = builder.observations().build_fixed_gain::<2, Predictor>();
    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    stacked_observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    stacked.observation_matrix_mut().make_identity();
    stacked.set_gain(&MatrixData::new_array::<2, 2, 4, f64>([
        L_INF[0], 0.05, L_INF[1], 0.3,
    ]));

    for (k, z) in readings(100).into_iter().enumerate() {
        let speed = 2.0 + 0.01 * k as f64;
        position.measurement_vector_mut().set_at(0, 0, z);
        velocity.measurement_vector_mut().set_at(0, 0, speed);
        let prior = state(observer.state_vector());
        assert!(observer.correct(&mut position).is_applied());
        assert!(observer.correct(&mut velocity).is_applied());
        assert_eq!(state(observer.state_vector()), prior);
        observer.predict();

        stacked.measurement_vector_mut().apply(|m| {
            m.set_at(0, 0, z);
            m.set_at(1, 0, speed);
        });
        assert!(stacked_observer.correct(&mut stacked).is_applied());
        stacked_observer.predict();

        let gap = gap(
            state(observer.state_vector()),
            state(stacked_observer.state_vector()),
        );
        assert!(gap < 1e-12, "step {k}: {gap}");
    }
}

/// The time-step handle applies several distinct controls: `x̂ ← A·x̂ + B₁·u₁ + B₂·u₂`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn time_update_applies_two_controls() {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut throttle = builder.controls().build::<1>();
    throttle.control_matrix_mut().apply(|b| {
        b.set_at(0, 0, 0.5 * DT * DT);
        b.set_at(1, 0, DT);
    });
    throttle.control_vector_mut().set_at(0, 0, 0.5);
    let mut wind = builder.controls().build::<2>();
    wind.control_matrix_mut().apply(|b| {
        b.set_at(0, 0, 0.1);
        b.set_at(1, 1, -0.2);
    });
    wind.control_vector_mut().apply(|u| {
        u.set_at(0, 0, 3.0);
        u.set_at(1, 0, 4.0);
    });

    let (mut observer, _) = cv_observer();
    let (mut predictor, _) = cv_predictor(L_INF);
    for x in [observer.state_vector_mut(), predictor.state_vector_mut()] {
        x.set_at(0, 0, 3.0);
        x.set_at(1, 0, -1.0);
    }

    observer.predict().control(&mut throttle).control(&mut wind);
    predictor
        .predict()
        .control(&mut throttle)
        .control(&mut wind);

    let expected = [3.0 - DT + 0.25 * DT * DT + 0.3, -1.0 + 0.5 * DT - 0.8];
    assert!(gap(state(observer.state_vector()), expected) < 1e-15);
    assert!(gap(state(predictor.state_vector()), expected) < 1e-15);
}

/// Rebuilding a predictor-form observer from its storage discards a correction left pending
/// by the previous observer, but keeps `A` and `x̂`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn predictor_storage_rebuild_clears_the_pending_correction() {
    let mut observer_storage = PredictorObserverStorage::<2, f64>::new(0.0);
    let mut position_storage = FixedGainObservationStorage::<2, 1, f64, Predictor>::new(0.0);
    {
        let mut observer = observer_storage.build();
        let mut position = position_storage.build();
        observer.state_transition_mut().make_identity();
        observer.state_vector_mut().set_at(0, 0, 1.0);
        position.observation_matrix_mut().set_at(0, 0, 1.0);
        position.set_gain(&column([0.5, 0.0]));
        position.measurement_vector_mut().set_at(0, 0, 3.0);
        assert!(observer.correct(&mut position).is_applied());
        assert_eq!(observer.pending_correction().get_at(0, 0), 1.0);
    }

    let mut observer = observer_storage.build();
    assert_eq!(state(observer.pending_correction()), [0.0, 0.0]);
    observer.predict();
    assert_eq!(state(observer.state_vector()), [1.0, 0.0]);
}

/// A storage keeps the gain across builds, in the form of its type.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn storage_rebuild_keeps_the_gain() {
    let mut position_storage = FixedGainObservationStorage::<2, 1, f64, Predictor>::new(0.0);
    position_storage.build().set_gain(&column(L_INF));

    let position = position_storage.build();
    assert_eq!(position.gain().get_at(0, 0), L_INF[0]);
}

/// The worked example of the docs: the deadbeat predictor gain `L = [2, 10]` settles the error
/// in two steps in its own form, but diverges when used as a current-form gain.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn deadbeat_gain_in_the_wrong_form_diverges() {
    let truth = |k: usize| [1.0 + 2.0 * DT * k as f64, 2.0];

    let (mut right, mut right_position) = cv_predictor([2.0, 10.0]);
    for k in 0..2 {
        right_position
            .measurement_vector_mut()
            .set_at(0, 0, truth(k)[0]);
        assert!(right.correct(&mut right_position).is_applied());
        right.predict();
    }
    assert!(gap(state(right.state_vector()), truth(2)) < 1e-12);

    let (mut wrong, mut wrong_position) = cv_observer();
    wrong_position.set_gain(&column([2.0, 10.0]));
    for k in 1..=20 {
        wrong.predict();
        wrong_position
            .measurement_vector_mut()
            .set_at(0, 0, truth(k)[0]);
        assert!(wrong.correct(&mut wrong_position).is_applied());
    }
    assert!(gap(state(wrong.state_vector()), truth(20)) > 1e3);
}

/// A control input moves the observer's state like the Kalman filter's; with the same gain,
/// both agree once the filter's gain has settled.
#[test]
// micromath's approximate square root keeps the filter's gain away from K∞.
#[cfg(any(feature = "std", feature = "libm"))]
fn control_moves_the_state_like_the_kalman_filter() {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut control = builder.controls().build::<1>();
    control.control_matrix_mut().apply(|b| {
        b.set_at(0, 0, 0.5 * DT * DT);
        b.set_at(1, 0, DT);
    });
    control.control_vector_mut().set_at(0, 0, 0.5);

    let (mut filter, mut position) = design::<f64>(10.0);
    let (mut observer, mut fixed) = cv_observer();
    fixed.set_gain(&column(K_INF));

    let mut last = f64::MAX;
    for z in readings(300) {
        filter.predict();
        filter.control(&mut control);
        position.measurement_vector_mut().set_at(0, 0, z);
        filter.correct(&mut position);

        observer.predict().control(&mut control);
        fixed.measurement_vector_mut().set_at(0, 0, z);
        assert!(observer.correct(&mut fixed).is_applied());
        last = gap(state(filter.state_vector()), state(observer.state_vector()));
    }

    assert!(last < 1e-9, "{last}");

    // From x̂ = 0, the step is x ← B·u.
    let (mut open_loop, _) = cv_observer();
    open_loop.predict().control(&mut control);
    assert_eq!(state(open_loop.state_vector()), [0.25 * DT * DT, 0.5 * DT]);
}

/// The storage types wire the same observers as the builders, and the trait impls expose them.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn storage_and_traits() {
    let mut observer_storage = ObserverStorage::<2, f64>::new(0.0);
    let mut position_storage = FixedGainObservationStorage::<2, 1, f64>::new(0.0);
    let mut observer = observer_storage.build();
    let mut position = position_storage.build();
    assert_eq!((observer.states(), position.states()), (2, 2));
    assert_eq!(position.measurements(), 1);

    generic_setup(&mut observer, &mut position);
    position.set_gain(&column([0.5, 0.0]));
    observer.predict();
    position.measurement_vector_mut().set_at(0, 0, 2.0);
    let status = observer.correct_with(&mut position, |y| y[0] *= 2.0);
    assert!(status.is_applied());
    assert_eq!(position.innovation_vector().get_at(0, 0), 4.0);
    assert_eq!(observer.state_vector().get_at(0, 0), 2.0);
    assert_eq!(position.measurement_vector().get_at(0, 0), 2.0);
    assert_eq!(position.observation_matrix().get_at(0, 0), 1.0);
    assert_eq!(observer.state_transition().get_at(0, 1), DT);

    let mut predictor_storage = PredictorObserverStorage::<2, f64>::new(0.0);
    let mut predictor_position_storage =
        FixedGainObservationStorage::<2, 1, f64, Predictor>::new(0.0);
    let mut predictor = predictor_storage.build();
    let mut predictor_position = predictor_position_storage.build();
    generic_setup(&mut predictor, &mut predictor_position);
    predictor_position.set_gain(&column([0.5, 0.0]));
    predictor_position
        .measurement_vector_mut()
        .set_at(0, 0, 2.0);
    let status = predictor.correct_with(&mut predictor_position, |y| y[0] *= 2.0);
    assert!(status.is_applied());
    assert_eq!(predictor.state_vector().get_at(0, 0), 0.0);
    assert_eq!(predictor.pending_correction().get_at(0, 0), 2.0);
    KalmanFilterPredict::predict(&mut predictor);
    assert_eq!(predictor.state_vector().get_at(0, 0), 2.0);
}

/// Sets up the CV observer through the generic traits only.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn generic_setup<O, M>(observer: &mut O, position: &mut M)
where
    O: KalmanFilterNumStates<2>
        + KalmanFilterPredict<2, f64>
        + KalmanFilterStateVectorMut<2, f64>
        + KalmanFilterStateTransitionMut<2, f64>,
    M: KalmanFilterNumObservations<1>
        + KalmanFilterObservationVectorMut<1, f64>
        + KalmanFilterObservationTransformationMut<2, 1, f64>
        + KalmanFilterObservationCorrectFixedGain<2, f64>,
{
    assert_eq!((observer.states(), position.observations()), (2, 1));
    let a = observer.state_transition_mut().as_matrix_mut();
    a.make_identity();
    a.set_at(0, 1, DT);
    position
        .observation_matrix_mut()
        .as_matrix_mut()
        .set_at(0, 0, 1.0);
    position
        .measurement_vector_mut()
        .as_matrix_mut()
        .set_at(0, 0, 0.0);
    observer.predict();
    assert_eq!(observer.state_vector().as_matrix().get_at(0, 0), 0.0);
    assert_eq!(observer.state_transition().as_matrix().get_at(0, 1), DT);
    assert_eq!(position.measurement_vector().as_matrix().get_at(0, 0), 0.0);
    assert_eq!(position.observation_matrix().as_matrix().get_at(0, 0), 1.0);
}

/// An `I16F16` observer with the `f64`-designed gain stays within the quantization bound
/// `LSB / (1 − ρ)` of the `f64` observer.
#[test]
// Compares against the `f64` observer, which needs a float backend.
#[cfg(all(
    feature = "fixed",
    any(feature = "std", feature = "libm", feature = "micromath")
))]
fn fixed_point_observer_tracks_f64() {
    use minikalman::prelude::I16F16;

    let mut observer_storage = ObserverStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut position_storage = FixedGainObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut fixed_observer = observer_storage.build();
    let mut fixed_position = position_storage.build();
    let model = motion::constant_velocity(I16F16::from_num(DT), I16F16::ONE);
    fixed_observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    fixed_position
        .observation_matrix_mut()
        .set_at(0, 0, I16F16::ONE);
    fixed_position.set_gain(&MatrixData::new_array::<2, 1, 2, I16F16>([
        I16F16::from_num(K_INF[0]),
        I16F16::from_num(K_INF[1]),
    ]));

    let (mut observer, mut position) = cv_observer();
    position.set_gain(&column(K_INF));

    let data = readings(400);
    // Seed both from the first measurement to keep the start-up innovations small.
    observer.state_vector_mut().set_at(0, 0, data[0]);
    fixed_observer
        .state_vector_mut()
        .set_at(0, 0, I16F16::from_num(data[0]));

    let lsb = 1.0 / 65536.0;
    let bound = lsb / (1.0 - RHO);
    let mut worst: f64 = 0.0;
    for (k, &z) in data.iter().enumerate().skip(1) {
        observer.predict();
        position.measurement_vector_mut().set_at(0, 0, z);
        assert!(observer.correct(&mut position).is_applied());

        fixed_observer.predict();
        fixed_position
            .measurement_vector_mut()
            .set_at(0, 0, I16F16::from_num(z));
        assert!(fixed_observer.correct(&mut fixed_position).is_applied());

        if k > 100 {
            let x = fixed_observer.state_vector();
            let fixed_state = [
                x.get_at(0, 0).to_num::<f64>(),
                x.get_at(1, 0).to_num::<f64>(),
            ];
            worst = worst.max(gap(fixed_state, state(observer.state_vector())));
        }
    }

    eprintln!("I16F16 vs f64: {worst:e} = {:.1} LSB", worst / lsb);
    assert!(worst < bound, "{worst} exceeds LSB/(1−ρ) = {bound}");
}

/// The chi-square 99% quantile for one degree of freedom, the gate of the gating tests.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
const GATE: f64 = 6.63;

/// A gate so wide (about 5σ) that no clean reading of the seeded noise exceeds it, for tests
/// that assert every clean step is applied.
const WIDE_GATE: f64 = 25.0;

/// A design filter and observation settled to `P⁺∞`, so that the next prediction gives `S∞`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn settled() -> (
    KalmanFilterType<2, f64>,
    KalmanFilterObservationType<2, 1, f64>,
) {
    let (mut filter, mut position) = design::<f64>(1.0);
    filter
        .settle_gain(&mut position, 1e-12, 10_000)
        .expect("detectable model");
    (filter, position)
}

/// An `f64` observer and a gated position observation for the CV model, with `K∞` and
/// `S∞⁻¹` taken from the settled `position` and the gate at [`GATE`].
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn gated_observer(
    position: &mut KalmanFilterObservationType<2, 1, f64>,
) -> (
    LuenbergerObserverType<2, f64>,
    GatedFixedGainObservationType<2, 1, f64>,
) {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let (observer, _) = cv_observer();
    let mut gated = builder
        .observations()
        .build_fixed_gain_gated::<1, Current>();
    gated.observation_matrix_mut().set_at(0, 0, 1.0);
    gated.set_gain(position.kalman_gain());
    gated
        .set_innovation_covariance(position.innovation_covariance_mut().as_matrix_mut())
        .expect("S∞ is positive definite");
    gated.set_gate(Some(GATE));
    (observer, gated)
}

/// `settle_gain`'s `S∞` inverts to SciPy's `1/S∞`, and the fixed-gain NIS equals the
/// Kalman filter's NIS at every step once the filter has settled.
#[test]
#[cfg(any(feature = "std", feature = "libm"))]
fn gated_nis_matches_the_kalman_filter_in_steady_state() {
    let (mut filter, mut position) = settled();
    let (mut observer, mut gated) = gated_observer(&mut position);
    let s_inv = gated
        .innovation_covariance_inverse()
        .as_matrix()
        .get_at(0, 0);
    assert!((s_inv - 1.0 / S_INF).abs() < 1e-9, "S∞⁻¹ = {s_inv}");
    assert_eq!(gated.nis(), 0.0);
    // No gate: one rejection would widen the Kalman filter's S, but not S∞.
    gated.set_gate(None);

    let mut sum = 0.0;
    let mut worst: f64 = 0.0;
    let data = readings(400);
    for &z in &data {
        filter.predict();
        position.measurement_vector_mut().set_at(0, 0, z);
        let kalman = filter.correct(&mut position);

        observer.predict();
        gated.measurement_vector_mut().set_at(0, 0, z);
        assert_eq!(observer.correct(&mut gated), kalman);
        worst = worst.max((gated.nis() - position.nis()).abs());
        sum += gated.nis();
    }

    let mean = sum / data.len() as f64;
    eprintln!("NIS gap {worst:e}, mean NIS {mean}");
    assert!(worst < 1e-9, "NIS differs by {worst}");
    assert!((mean - 1.0).abs() < 0.25, "mean NIS {mean} is far from 1");
}

/// An outlier exceeds the gate in both gain forms, is skipped with `Gated` and leaves the
/// state untouched; in the predictor form, it leaves no pending correction, so the following
/// `predict` is a pure model step.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn outliers_are_gated_and_leave_the_state_unchanged() {
    let (mut observer, mut gated) = gated_observer(&mut settled().1);
    let (mut predictor, _) = cv_predictor(L_INF);
    let mut gated_predictor = KalmanFilterBuilder::<2, f64>::default()
        .observations()
        .build_fixed_gain_gated::<1, Predictor>();
    gated_predictor.observation_matrix_mut().set_at(0, 0, 1.0);
    gated_predictor.set_gain(&column(L_INF));
    gated_predictor
        .set_innovation_covariance_inverse(&MatrixData::new_array::<1, 1, 1, f64>([1.0 / S_INF]));
    gated.set_gate(Some(WIDE_GATE));
    gated_predictor.set_gate(Some(WIDE_GATE));

    let data = readings(100);
    for (k, &reading) in data.iter().enumerate() {
        let outlier = k == 60;
        let z = if outlier { reading + 10.0 } else { reading };

        observer.predict();
        let prior = state(observer.state_vector());
        gated.measurement_vector_mut().set_at(0, 0, z);
        let status = observer.correct(&mut gated);

        let before = state(predictor.state_vector());
        gated_predictor.measurement_vector_mut().set_at(0, 0, z);
        let predictor_status = predictor.correct(&mut gated_predictor);
        assert_eq!(state(predictor.state_vector()), before);

        if outlier {
            assert_eq!(status, CorrectionStatus::Gated);
            assert_eq!(state(observer.state_vector()), prior);
            assert!(gated.nis() > 300.0, "outlier NIS {}", gated.nis());

            assert_eq!(predictor_status, CorrectionStatus::Gated);
            assert!(gated_predictor.nis() > 300.0);
            assert_eq!(state(predictor.pending_correction()), [0.0, 0.0]);
            predictor.predict();
            let model_step = [
                1.0 * before[0] + DT * before[1],
                0.0 * before[0] + 1.0 * before[1],
            ];
            assert_eq!(state(predictor.state_vector()), model_step);
        } else if k > 10 {
            assert_eq!(status, CorrectionStatus::Applied, "step {k}");
            assert_eq!(predictor_status, CorrectionStatus::Applied, "step {k}");
        }

        if !outlier {
            predictor.predict();
        }
    }
}

/// With a wrong initial state, a gate that is on from the first step rejects every
/// measurement for good, since `S∞` never grows; turning it on after the transient works.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn start_up_gate_locks_out_and_delayed_gate_converges() {
    let (mut locked, mut locked_position) = gated_observer(&mut settled().1);
    let (mut delayed, mut delayed_position) = gated_observer(&mut settled().1);
    locked.state_vector_mut().set_at(0, 0, -5.0);
    delayed.state_vector_mut().set_at(0, 0, -5.0);
    delayed_position.set_gate(None);

    let mut rejected = 0;
    let mut false_alarms = 0;
    let data = readings(400);
    for (k, &z) in data.iter().enumerate() {
        locked.predict();
        locked_position.measurement_vector_mut().set_at(0, 0, z);
        if locked.correct(&mut locked_position) == CorrectionStatus::Gated {
            rejected += 1;
        }

        if k == 100 {
            delayed_position.set_gate(Some(GATE));
        }

        delayed.predict();
        delayed_position.measurement_vector_mut().set_at(0, 0, z);
        if delayed.correct(&mut delayed_position) == CorrectionStatus::Gated {
            false_alarms += 1;
        }
    }

    // A 99% gate rejects about 3 of the 300 good readings once the observer has settled.
    eprintln!("delayed gate rejected {false_alarms} of 300");
    assert!(false_alarms <= 9, "{false_alarms} rejections");
    assert_eq!(rejected, data.len(), "the gate locked out");
    let truth = [1.0 + 2.0 * DT * data.len() as f64, 2.0];
    assert!(gap(state(delayed.state_vector()), truth) < 1.0);
    assert!(gap(state(locked.state_vector()), truth) > 10.0);
}

/// A gate of `None` still computes the NIS.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn open_gate_computes_the_nis() {
    let (mut observer, mut gated) = gated_observer(&mut settled().1);
    gated.measurement_vector_mut().set_at(0, 0, 100.0);
    assert_eq!(gated.nis(), 0.0);
    assert_eq!(gated.gate(), Some(GATE));
    gated.set_gate(None);
    assert_eq!(observer.correct(&mut gated), CorrectionStatus::Applied);
    assert!(gated.nis() > 1000.0);
}

/// A covariance that is not positive definite is refused and keeps the stored `S∞⁻¹`.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn non_positive_definite_innovation_covariance_is_refused() {
    let (_, mut gated) = gated_observer(&mut settled().1);
    let before = gated
        .innovation_covariance_inverse()
        .as_matrix()
        .get_at(0, 0);
    let mut bad = MatrixData::new_array::<1, 1, 1, f64>([-1.0]);
    assert_eq!(
        gated.set_innovation_covariance(&mut bad),
        Err(CorrectionStatus::NotPositiveDefinite)
    );
    assert_eq!(
        gated
            .innovation_covariance_inverse()
            .as_matrix()
            .get_at(0, 0),
        before
    );
}

/// The gated storage keeps `S∞⁻¹` across builds, but not the threshold or the NIS.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
#[test]
fn gated_storage_keeps_the_inverse_but_not_the_threshold() {
    let mut storage = GatedFixedGainObservationStorage::<2, 1, f64>::new(0.0);
    {
        let mut position = storage.build();
        position.set_innovation_covariance_inverse(&MatrixData::new_array::<1, 1, 1, f64>([
            1.0 / S_INF
        ]));
        position.set_gate(Some(GATE));
    }

    let position = storage.build();
    assert_eq!(position.gate(), None);
    assert_eq!(position.nis(), 0.0);
    assert_eq!(
        position
            .innovation_covariance_inverse()
            .as_matrix()
            .get_at(0, 0),
        1.0 / S_INF
    );
}

/// An `I16F16` gated observer on statically allocated storage passes the readings and
/// gates a 10 m outlier, whose NIS of about 330 fits the integer range.
#[test]
#[cfg(feature = "fixed")]
fn fixed_point_gate_rejects_outliers() {
    use minikalman::prelude::I16F16;

    let mut observer_storage = ObserverStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut position_storage = GatedFixedGainObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut observer = observer_storage.build();
    let mut position = position_storage.build();
    let model = motion::constant_velocity(I16F16::from_num(DT), I16F16::ONE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, I16F16::ONE);
    position.set_gain(&MatrixData::new_array::<2, 1, 2, I16F16>([
        I16F16::from_num(K_INF[0]),
        I16F16::from_num(K_INF[1]),
    ]));
    let mut s = MatrixData::new_array::<1, 1, 1, I16F16>([I16F16::from_num(S_INF)]);
    position
        .set_innovation_covariance(&mut s)
        .expect("positive definite");
    position.set_gate(Some(I16F16::from_num(WIDE_GATE)));

    let data = readings(200);
    observer
        .state_vector_mut()
        .set_at(0, 0, I16F16::from_num(data[0]));
    for (k, &reading) in data.iter().enumerate().skip(1) {
        let outlier = k == 150;
        let z = if outlier { reading + 10.0 } else { reading };
        observer.predict();
        let prior = observer.state_vector().get_at(0, 0);
        position
            .measurement_vector_mut()
            .set_at(0, 0, I16F16::from_num(z));
        let status = observer.correct(&mut position);

        if outlier {
            assert_eq!(status, CorrectionStatus::Gated);
            assert_eq!(observer.state_vector().get_at(0, 0), prior);
            assert!(position.nis() > I16F16::from_num(300));
        } else if k > 20 {
            assert_eq!(status, CorrectionStatus::Applied, "step {k}");
        }
    }
}
