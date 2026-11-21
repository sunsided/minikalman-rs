//! The fixed-gain (Luenberger) observer and `settle_gain`, checked against SciPy and against
//! the regular Kalman filter on the same data.
//!
//! The model is the constant-velocity tracker `motion::constant_velocity(0.1, 1.0)` with a
//! position measurement of variance `0.25`. The reference values come from SciPy's
//! `solve_discrete_are(A.T, H.T, Q, R)` (scipy 1.11.1).

#![cfg(feature = "alloc")]
#![forbid(unsafe_code)]

use minikalman::models::motion;
use minikalman::num_traits::FromPrimitive;
use minikalman::prelude::*;
use minikalman::regular::builder::{
    FixedGainObservationType, KalmanFilterBuilder, KalmanFilterObservationType, KalmanFilterType,
    LuenbergerObserverType,
};
use minikalman::regular::{FixedGainObservationStorage, ObserverStorage};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Time step in seconds.
const DT: f64 = 0.1;

/// Acceleration variance of the constant-velocity model, in (m/s²)².
const ACCELERATION_VARIANCE: f64 = 1.0;

/// Measurement noise variance of the position, in m².
const R: f64 = 0.25;

/// Steady-state current-form gain `K∞` from SciPy.
const K_INF: [f64; 2] = [0.181_201_093_164_732_62, 0.180_975_015_605_498_97];

/// Steady-state predictor-form gain `L∞ = A·K∞` from SciPy.
const L_INF: [f64; 2] = [0.199_298_594_725_282_52, 0.180_975_015_605_498_97];

/// Steady-state corrected covariance `P⁺∞` from SciPy, row-major.
const P_POST: [f64; 4] = [
    0.045_300_273_291_183_16,
    0.045_243_753_901_374_75,
    0.045_243_753_901_374_75,
    0.095_124_921_972_503_88,
];

/// Steady-state innovation covariance `S∞` from SciPy.
const S_INF: f64 = 0.305_325_273_291_183_14;

/// Spectral radius of the error dynamics `(I − K∞·H)·A`, from numpy.
const RHO: f64 = 0.904_875_078_027_496_2;

/// A design-time Kalman filter and its position observation, set up for the CV model.
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
fn cv_observer() -> (
    LuenbergerObserverType<2, f64>,
    FixedGainObservationType<2, 1, f64>,
) {
    let builder = KalmanFilterBuilder::<2, f64>::default();
    let mut observer = builder.build_observer();
    let mut position = builder.observations().build_fixed_gain::<1>();
    let model = motion::constant_velocity(DT, ACCELERATION_VARIANCE);
    observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    position.observation_matrix_mut().set_at(0, 0, 1.0);
    (observer, position)
}

/// Converts an `f64` to `T`.
fn of<T: FromPrimitive>(value: f64) -> T {
    T::from_f64(value).expect("representable")
}

/// A 2 × 1 matrix with the given entries, to pass a gain to the setters.
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
fn state(x: &impl Matrix<2, 1, f64>) -> [f64; 2] {
    [x.get_at(0, 0), x.get_at(1, 0)]
}

/// Largest absolute entry-wise difference of two 2-vectors.
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
    fixed.set_gain_current(&column(K_INF));
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

/// The predictor form with `L = A·K` runs one step ahead of the current form: its state is
/// `A·x̂⁺` of the current-form run at every step.
#[test]
fn predictor_form_is_current_form_one_step_ahead() {
    let (mut current, mut current_position) = cv_observer();
    let (mut predictor, mut predictor_position) = cv_observer();
    current_position.set_gain_current(&column(K_INF));
    predictor_position.set_gain_predictor(&column(L_INF));
    assert_eq!(predictor_position.gain_form(), GainForm::Predictor);

    // Both start from x̂₀ = [0, 0]; the predictor form starts from its prior A·x̂₀ = x̂₀.
    for z in readings(200) {
        current.predict();
        current_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(current.correct(&mut current_position).is_applied());

        predictor_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(predictor
            .predict_from_measurement(&mut predictor_position)
            .is_applied());

        let [p, v] = state(current.state_vector());
        let ahead = [p + DT * v, v];
        assert!(gap(ahead, state(predictor.state_vector())) < 1e-12);
    }
}

/// With a control input, the predictor step followed by `control` is
/// `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)`: the current-form prior `A·x̂⁺ + B·u` of the next step,
/// from a nonzero start.
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
    let (mut predictor, mut predictor_position) = cv_observer();
    current_position.set_gain_current(&column(K_INF));
    predictor_position.set_gain_predictor(&column(L_INF));
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
        current.predict();
        current.control(&mut control);
        assert!(
            gap(
                state(current.state_vector()),
                state(predictor.state_vector())
            ) < 1e-12
        );

        current_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(current.correct(&mut current_position).is_applied());

        predictor_position.measurement_vector_mut().set_at(0, 0, z);
        assert!(predictor
            .predict_from_measurement(&mut predictor_position)
            .is_applied());
        predictor.control(&mut control);
    }
}

/// A storage keeps the gain across builds, and `build` states its form.
#[test]
fn storage_rebuild_keeps_the_stated_form() {
    let mut position_storage = FixedGainObservationStorage::<2, 1, f64>::new(0.0);
    position_storage
        .build(GainForm::Current)
        .set_gain_predictor(&column(L_INF));

    let position = position_storage.build(GainForm::Predictor);
    assert_eq!(position.gain_form(), GainForm::Predictor);
    assert_eq!(position.gain().get_at(0, 0), L_INF[0]);
}

/// A gain is only run in the recursion of its form; the other one is refused.
#[test]
fn gain_form_mismatch_is_refused() {
    let (mut observer, mut position) = cv_observer();
    position.measurement_vector_mut().set_at(0, 0, 3.0);

    position.set_gain_predictor(&column(L_INF));
    assert_eq!(
        observer.correct(&mut position),
        CorrectionStatus::GainFormMismatch
    );
    assert!(!CorrectionStatus::GainFormMismatch.is_applied());

    position.set_gain_current(&column(K_INF));
    assert_eq!(
        observer.predict_from_measurement(&mut position),
        CorrectionStatus::GainFormMismatch
    );
    assert_eq!(state(observer.state_vector()), [0.0, 0.0]);
    assert_eq!(position.gain().get_at(0, 0), K_INF[0]);
}

/// The worked example of the docs: the deadbeat predictor gain `L = [2, 10]` settles the error
/// in two steps in its own form, but diverges when used as a current-form gain.
#[test]
fn deadbeat_gain_in_the_wrong_form_diverges() {
    let truth = |k: usize| [1.0 + 2.0 * DT * k as f64, 2.0];

    let (mut right, mut right_position) = cv_observer();
    right_position.set_gain_predictor(&column([2.0, 10.0]));
    for k in 0..2 {
        right_position
            .measurement_vector_mut()
            .set_at(0, 0, truth(k)[0]);
        assert!(right
            .predict_from_measurement(&mut right_position)
            .is_applied());
    }
    assert!(gap(state(right.state_vector()), truth(2)) < 1e-12);

    let (mut wrong, mut wrong_position) = cv_observer();
    wrong_position.set_gain_current(&column([2.0, 10.0]));
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
    fixed.set_gain_current(&column(K_INF));

    let mut last = f64::MAX;
    for z in readings(300) {
        filter.predict();
        filter.control(&mut control);
        position.measurement_vector_mut().set_at(0, 0, z);
        filter.correct(&mut position);

        observer.predict();
        observer.control(&mut control);
        fixed.measurement_vector_mut().set_at(0, 0, z);
        assert!(observer.correct(&mut fixed).is_applied());
        last = gap(state(filter.state_vector()), state(observer.state_vector()));
    }

    assert!(last < 1e-9, "{last}");

    // Without a correction, the control alone is x ← x + B·u.
    let (mut open_loop, _) = cv_observer();
    open_loop.control(&mut control);
    assert_eq!(state(open_loop.state_vector()), [0.25 * DT * DT, 0.5 * DT]);
}

/// The storage types wire the same observer as the builders, and the trait impls expose it.
#[test]
fn storage_and_traits() {
    let mut observer_storage = ObserverStorage::<2, f64>::new(0.0);
    let mut position_storage = FixedGainObservationStorage::<2, 1, f64>::new(0.0);
    let mut observer = observer_storage.build();
    let mut position = position_storage.build(GainForm::Current);
    assert_eq!(position.gain_form(), GainForm::Current);
    assert_eq!((observer.states(), position.states()), (2, 2));
    assert_eq!(position.measurements(), 1);

    generic_setup(&mut observer, &mut position);
    position.set_gain_current(&column([0.5, 0.0]));
    observer.predict();
    position.measurement_vector_mut().set_at(0, 0, 2.0);
    let status = observer.correct_with(&mut position, |y| y[0] *= 2.0);
    assert!(status.is_applied());
    assert_eq!(position.innovation_vector().get_at(0, 0), 4.0);
    assert_eq!(observer.state_vector().get_at(0, 0), 2.0);
    assert_eq!(position.measurement_vector().get_at(0, 0), 2.0);
    assert_eq!(position.observation_matrix().get_at(0, 0), 1.0);
    assert_eq!(observer.state_transition().get_at(0, 1), DT);

    position.set_gain_predictor(&column([0.5, 0.0]));
    let status = observer.predict_from_measurement_with(&mut position, |y| y[0] = 0.0);
    assert!(status.is_applied());
    assert_eq!(observer.state_vector().get_at(0, 0), 2.0);
}

/// Sets up the CV observer through the generic traits only.
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
    assert_eq!(position.gain_form(), GainForm::Current);
}

/// An `I16F16` observer with the `f64`-designed gain stays within the quantization bound
/// `LSB / (1 − ρ)` of the `f64` observer.
#[test]
#[cfg(feature = "fixed")]
fn fixed_point_observer_tracks_f64() {
    use minikalman::prelude::I16F16;

    let mut observer_storage = ObserverStorage::<2, I16F16>::new(I16F16::ZERO);
    let mut position_storage = FixedGainObservationStorage::<2, 1, I16F16>::new(I16F16::ZERO);
    let mut fixed_observer = observer_storage.build();
    let mut fixed_position = position_storage.build(GainForm::Current);
    let model = motion::constant_velocity(I16F16::from_num(DT), I16F16::ONE);
    fixed_observer
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    fixed_position
        .observation_matrix_mut()
        .set_at(0, 0, I16F16::ONE);
    fixed_position.set_gain_current(&MatrixData::new_array::<2, 1, 2, I16F16>([
        I16F16::from_num(K_INF[0]),
        I16F16::from_num(K_INF[1]),
    ]));

    let (mut observer, mut position) = cv_observer();
    position.set_gain_current(&column(K_INF));

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
