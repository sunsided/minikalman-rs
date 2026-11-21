//! The square-root covariance filter against the regular filter, and its handling of
//! rejected corrections and singular process noise.

#![forbid(unsafe_code)]
#![cfg(feature = "alloc")]
#![allow(non_snake_case)]

use minikalman::matrix::MatrixDataArray;
use minikalman::prelude::*;
use minikalman::regular::builder::{
    KalmanFilterBuilder, KalmanFilterObservationType, KalmanFilterType, SquareRootKalmanType,
};
use minikalman::smoother::RtsSmoother;

/// A two-state regular filter with owned buffers.
type Regular<T> = KalmanFilterType<2, T>;

/// A two-state square-root filter with owned buffers.
type SquareRoot<T> = SquareRootKalmanType<2, 4, T>;

/// A two-state observation of `M` measurements with owned buffers.
type Observation<const M: usize, T> = KalmanFilterObservationType<2, M, T>;

/// Converts an `f64` literal to `T`.
fn t<T: MatrixDataType>(value: f64) -> T {
    T::from_f64(value).unwrap()
}

/// Sets a constant-velocity state transition with `dt = 0.5`.
fn set_transition<T: MatrixDataType>(A: &mut impl MatrixMut<2, 2, T>) {
    A.apply(|a| {
        a.set(0, 0, T::one());
        a.set(0, 1, t(0.5));
        a.set(1, 0, T::zero());
        a.set(1, 1, T::one());
    });
}

/// Writes the correlated process noise `Q` used by most tests.
fn set_noise<T: MatrixDataType>(Q: &mut impl MatrixMut<2, 2, T>) {
    Q.apply(|q| {
        q.set(0, 0, t(0.05));
        q.set_symmetric(0, 1, t(0.01));
        q.set(1, 1, t(0.1));
    });
}

/// Builds a regular and a square-root filter with the same `A`, `x` and correlated `P`, and
/// the square-root filter's `Lq = chol(Q)` for the `Q` of [`set_noise`].
fn make_filters<T>() -> (Regular<T>, SquareRoot<T>)
where
    T: MatrixDataType + Default,
{
    let builder = KalmanFilterBuilder::<2, T>::default();
    let mut regular = builder.build();
    let mut square_root = builder.build_square_root::<4>();

    set_transition(regular.state_transition_mut());
    set_transition(square_root.state_transition_mut());
    set_noise(regular.direct_process_noise_mut());
    assert!(square_root.set_process_noise(regular.direct_process_noise()));

    regular.state_vector_mut().apply(|x| {
        x.set(0, 0, T::one());
        x.set(1, 0, t(-0.5));
    });
    regular.estimate_covariance_mut().apply(|p| {
        p.set(0, 0, t(4.0));
        p.set_symmetric(0, 1, t(0.5));
        p.set(1, 1, T::one());
    });
    assert!(square_root.set_state(regular.state_vector(), regular.estimate_covariance()));

    (regular, square_root)
}

/// A position sensor that also sees half the velocity.
fn first_sensor<T>() -> Observation<1, T>
where
    T: MatrixDataType + Default,
{
    let mut sensor = KalmanFilterBuilder::<2, T>::default()
        .observations()
        .build::<1>();
    sensor.observation_matrix_mut().apply(|h| {
        h.set(0, 0, T::one());
        h.set(0, 1, t(0.5));
    });
    sensor.measurement_noise_covariance_mut().set(0, 0, t(0.5));
    sensor
}

/// A two-channel sensor with correlated noise.
fn second_sensor<T>() -> Observation<2, T>
where
    T: MatrixDataType + Default,
{
    let mut sensor = KalmanFilterBuilder::<2, T>::default()
        .observations()
        .build::<2>();
    sensor.observation_matrix_mut().apply(|h| {
        h.set(0, 1, T::one());
        h.set(1, 0, T::one());
        h.set(1, 1, T::one());
    });
    sensor.measurement_noise_covariance_mut().apply(|r| {
        r.set(0, 0, T::one());
        r.set_symmetric(0, 1, t(0.2));
        r.set(1, 1, t(2.0));
    });
    sensor
}

/// Asserts that two buffers agree entry by entry to within `tolerance`.
fn assert_close<T>(what: &str, actual: &[T], expected: &[T], tolerance: f64)
where
    T: Copy + Into<f64>,
{
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let (a, e) = (a.into(), e.into());
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance,
            "{what}[{i}]: {a} != {e}"
        );
    }
}

/// Asserts that `x` and `P = S×Sᵀ` of the square-root filter match the regular filter.
fn assert_matches<T>(what: &str, square_root: &SquareRoot<T>, regular: &Regular<T>, tol: f64)
where
    T: MatrixDataType + Default + Into<f64>,
{
    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
    square_root.state(&mut x, &mut P);

    let regular_x = regular.state_vector().as_slice();
    assert_close(what, x.as_slice(), regular_x, tol);
    let regular_P = regular.estimate_covariance().as_slice();
    assert_close(what, P.as_slice(), regular_P, tol);
}

/// Corrects both filters with `sensor` and asserts that the status, NIS, innovation, `x`
/// and `P` agree.
fn correct_both<const M: usize, T>(
    what: &str,
    sensor: &mut Observation<M, T>,
    regular: &mut Regular<T>,
    square_root: &mut SquareRoot<T>,
    tolerance: f64,
) where
    T: MatrixDataType + Default + Into<f64>,
{
    let expected = regular.correct(sensor);
    let nis = sensor.nis();
    let innovation = sensor.innovation_vector().as_slice().to_vec();

    assert_eq!(square_root.correct(sensor), expected, "{what}");
    assert_close(what, &[sensor.nis()], &[nis], tolerance);
    assert_close(
        what,
        sensor.innovation_vector().as_slice(),
        &innovation,
        tolerance,
    );
    assert_matches(what, square_root, regular, tolerance);
}

/// Runs three predict and correct steps with a scalar and a correlated two-channel sensor.
fn matches_regular_filter<T>(tolerance: f64)
where
    T: MatrixDataType + Default + Into<f64>,
{
    let (mut regular, mut square_root) = make_filters::<T>();
    let mut first = first_sensor::<T>();
    let mut second = second_sensor::<T>();

    for (step, (z1, z2)) in [(3.0, [0.8, 3.5]), (3.6, [1.1, 4.4]), (4.0, [0.9, 5.2])]
        .into_iter()
        .enumerate()
    {
        first.measurement_vector_mut().set(0, 0, t(z1));
        second.measurement_vector_mut().apply(|z| {
            z.set(0, 0, t(z2[0]));
            z.set(1, 0, t(z2[1]));
        });

        let what = format!("step {step}");
        correct_both(&what, &mut first, &mut regular, &mut square_root, tolerance);
        correct_both(
            &what,
            &mut second,
            &mut regular,
            &mut square_root,
            tolerance,
        );

        // The predict starts from the non-triangular factor the Potter updates leave behind.
        regular.predict();
        assert!(square_root.predict());
        assert_matches(&what, &square_root, &regular, tolerance);
        let s = square_root.covariance_factor();
        assert!(
            s.get(0, 1) == T::zero(),
            "S is lower triangular after predict"
        );
    }
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn matches_the_regular_filter_in_f64() {
    matches_regular_filter::<f64>(1e-12);
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn matches_the_regular_filter_in_f32() {
    matches_regular_filter::<f32>(1e-4);
}

/// Sets the free-fall transition with `dt = 1`: position, velocity and acceleration.
fn set_gravity_transition<T: MatrixDataType>(A: &mut impl MatrixMut<3, 3, T>) {
    A.make_identity();
    A.set(0, 1, T::one());
    A.set(0, 2, t(0.5));
    A.set(1, 2, T::one());
}

/// Runs the free-fall setup of `tests/gravity.rs` (position measurements of a falling object,
/// with a third state for the acceleration) on both filters and compares `x` and `P`.
fn gravity_matches<T>(tolerance: f64)
where
    T: MatrixDataType + Default + Into<f64>,
{
    const DISTANCE: [f64; 15] = [
        0.0, 4.905, 19.62, 44.145, 78.48, 122.63, 176.58, 240.35, 313.92, 397.31, 490.5, 593.51,
        706.32, 828.94, 961.38,
    ];
    const NOISE: [f64; 15] = [
        0.13442, 0.45847, -0.56471, 0.21554, 0.079691, -0.32692, -0.1084, 0.085656, 0.8946,
        0.69236, -0.33747, 0.75873, 0.18135, -0.015764, 0.17869,
    ];

    let builder = KalmanFilterBuilder::<3, T>::default();
    let mut regular = builder.build();
    let mut square_root = builder.build_square_root::<6>();
    let mut sensor = builder.observations().build::<1>();

    set_gravity_transition(regular.state_transition_mut());
    set_gravity_transition(square_root.state_transition_mut());
    regular.estimate_covariance_mut().make_scalar(t(0.5));
    regular.estimate_covariance_mut().set(2, 2, T::one());
    regular.direct_process_noise_mut().make_scalar(t(1e-4));
    assert!(square_root.set_state(regular.state_vector(), regular.estimate_covariance()));
    assert!(square_root.set_process_noise(regular.direct_process_noise()));
    sensor.observation_matrix_mut().set(0, 0, T::one());
    sensor.measurement_noise_covariance_mut().set(0, 0, t(0.5));

    for (distance, noise) in DISTANCE.iter().zip(NOISE) {
        regular.predict();
        assert!(square_root.predict());

        sensor
            .measurement_vector_mut()
            .set(0, 0, t(distance + noise));
        regular.correct(&mut sensor);
        assert!(square_root.correct(&mut sensor).is_applied());
    }

    let mut P = BufferBuilder::estimate_covariance_P::<3>().new();
    let mut x = BufferBuilder::state_vector_x::<3>().new();
    square_root.state(&mut x, &mut P);
    let regular_x = regular.state_vector().as_slice();
    assert_close("x", x.as_slice(), regular_x, tolerance);
    let regular_P = regular.estimate_covariance().as_slice();
    assert_close("P", P.as_slice(), regular_P, tolerance);

    let acceleration: f64 = x.get(2, 0).into();
    assert!(
        (acceleration - 9.807).abs() < 0.5,
        "acceleration {acceleration}"
    );
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn matches_the_regular_filter_on_the_gravity_example_in_f64() {
    gravity_matches::<f64>(1e-9);
}

/// Positions reach about 1000, where the `f32` spacing is `6e-5`.
#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn matches_the_regular_filter_on_the_gravity_example_in_f32() {
    gravity_matches::<f32>(1e-2);
}

/// Copies `x` and `S` of the square-root filter, to check that a rejected step left them
/// untouched.
fn snapshot<T: MatrixDataType + Default>(square_root: &SquareRoot<T>) -> Vec<T> {
    let mut values = square_root.state_vector().as_slice().to_vec();
    values.extend_from_slice(square_root.covariance_factor().as_slice());
    values
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn gated_corrections_match_the_regular_filter_and_change_nothing() {
    let (mut regular, mut square_root) = make_filters::<f64>();
    let mut sensor = first_sensor::<f64>();
    sensor.set_gate(Some(3.0));
    sensor.measurement_vector_mut().set(0, 0, 50.0);

    let before = snapshot(&square_root);
    assert_eq!(regular.correct(&mut sensor), CorrectionStatus::Gated);
    let nis = sensor.nis();
    assert_eq!(square_root.correct(&mut sensor), CorrectionStatus::Gated);

    assert!((sensor.nis() - nis).abs() < 1e-9 && nis > 3.0);
    assert_eq!(snapshot(&square_root), before);
    assert!((sensor.innovation_vector().get(0, 0) - (50.0 - 0.75)).abs() < 1e-12);
}

#[test]
fn non_positive_definite_noise_changes_nothing() {
    let (_, mut square_root) = make_filters::<f64>();
    let mut sensor = first_sensor::<f64>();
    sensor.measurement_vector_mut().set(0, 0, 2.0);
    let before = snapshot(&square_root);

    // H×P×Hᵀ + R is positive definite, but R = 0 cannot be decorrelated.
    sensor.measurement_noise_covariance_mut().set(0, 0, 0.0);
    let status = square_root.correct(&mut sensor);
    assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!(
        sensor.nis(),
        0.0,
        "the NIS is only stored once R is factored"
    );

    // With H = 0 as well, H×P×Hᵀ + R = 0 has no Cholesky factor either.
    sensor.observation_matrix_mut().set_zero();
    let status = square_root.correct(&mut sensor);
    assert_eq!(status, CorrectionStatus::NotPositiveDefinite);
    assert_eq!(snapshot(&square_root), before);
}

/// Runs both filters with a process noise that `cholesky_decompose_lower` rejects, set as a
/// factor `Lq` with `Lq×Lqᵀ = Q`.
fn singular_process_noise_matches(lq: [f64; 4]) {
    let (mut regular, mut square_root) = make_filters::<f64>();
    let mut sensor = first_sensor::<f64>();

    square_root
        .process_noise_factor_mut()
        .as_mut_slice()
        .copy_from_slice(&lq);
    regular.direct_process_noise_mut().apply(|q| {
        for i in 0..2 {
            for j in 0..2 {
                q.set(i, j, (0..2).map(|k| lq[i * 2 + k] * lq[j * 2 + k]).sum());
            }
        }
    });
    assert!(!square_root.set_process_noise(regular.direct_process_noise()));
    assert_eq!(square_root.process_noise_factor().as_slice(), &lq);

    for (step, z) in [3.0, 3.6, 4.0, 4.1, 5.0].into_iter().enumerate() {
        regular.predict();
        assert!(square_root.predict());
        sensor.measurement_vector_mut().set(0, 0, z);
        let what = format!("Lq {lq:?}, step {step}");
        correct_both(&what, &mut sensor, &mut regular, &mut square_root, 1e-12);
    }
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn zero_and_singular_process_noise_match_the_regular_filter() {
    singular_process_noise_matches([0.0; 4]);
    singular_process_noise_matches([0.0, 0.0, 0.0, 0.3]);
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn state_round_trips_and_rejects_non_positive_definite_covariances() {
    let (regular, mut square_root) = make_filters::<f64>();
    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
    square_root.state(&mut x, &mut P);
    assert_close("x", x.as_slice(), regular.state_vector().as_slice(), 1e-12);
    assert_close(
        "P",
        P.as_slice(),
        regular.estimate_covariance().as_slice(),
        1e-12,
    );
    assert_eq!(square_root.covariance_factor().get(0, 1), 0.0);
    assert_eq!(square_root.states(), 2);

    let before = snapshot(&square_root);
    P.set(1, 1, -1.0);
    assert!(!square_root.set_state(&x, &P));
    assert_eq!(snapshot(&square_root), before);
}

#[test]
#[should_panic(expected = "TWO_STATES must be 2 × STATES")]
#[cfg(not(feature = "no_assert"))]
fn rejects_a_wrong_prearray_size() {
    let _ = KalmanFilterBuilder::<2, f32>::default().build_square_root::<3>();
}

/// A buffer of `[x̂, P]` states, one per step.
type History = (
    Vec<MatrixDataArray<2, 1, 2, f64>>,
    Vec<MatrixDataArray<2, 2, 4, f64>>,
);

/// Appends the current `x` and `P` of a filter to `history`.
fn record(history: &mut History, x: &[f64], P: &[f64]) {
    history.0.push(MatrixDataArray::new([x[0], x[1]]));
    history
        .1
        .push(MatrixDataArray::new([P[0], P[1], P[2], P[3]]));
}

/// Smooths the recorded `filtered` history with its `predicted` counterpart.
fn smooth(A: &impl Matrix<2, 2, f64>, filtered: &mut History, predicted: &History) {
    let square = || MatrixDataArray::<2, 2, 4, f64>::new([0.0; 4]);
    let mut smoother =
        RtsSmoother::new(square(), square(), square(), MatrixDataArray::new([0.0; 2]));
    let (x, p) = filtered;
    assert!(smoother.smooth(A, x, p, &predicted.0, &predicted.1));
}

#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn state_feeds_the_rts_smoother() {
    let (mut regular, mut square_root) = make_filters::<f64>();
    let mut sensor = first_sensor::<f64>();
    let (mut regular_filtered, mut regular_predicted) = (History::default(), History::default());
    let (mut filtered, mut predicted) = (History::default(), History::default());
    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();

    for z in [3.0, 3.6, 4.0, 4.1] {
        regular.predict();
        assert!(square_root.predict());
        record(
            &mut regular_predicted,
            regular.state_vector().as_slice(),
            regular.estimate_covariance().as_slice(),
        );
        square_root.state(&mut x, &mut P);
        record(&mut predicted, x.as_slice(), P.as_slice());

        sensor.measurement_vector_mut().set(0, 0, z);
        regular.correct(&mut sensor);
        assert!(square_root.correct(&mut sensor).is_applied());
        record(
            &mut regular_filtered,
            regular.state_vector().as_slice(),
            regular.estimate_covariance().as_slice(),
        );
        square_root.state(&mut x, &mut P);
        record(&mut filtered, x.as_slice(), P.as_slice());
    }

    smooth(
        regular.state_transition(),
        &mut regular_filtered,
        &regular_predicted,
    );
    smooth(square_root.state_transition(), &mut filtered, &predicted);
    for k in 0..4 {
        assert_close(
            "x",
            filtered.0[k].as_slice(),
            regular_filtered.0[k].as_slice(),
            1e-10,
        );
        assert_close(
            "P",
            filtered.1[k].as_slice(),
            regular_filtered.1[k].as_slice(),
            1e-10,
        );
    }
}

/// Runs `steps` constant-velocity steps with a position measurement whose variance `1e-4` is
/// twelve orders of magnitude below the initial variance `1e8`, and returns `P` after each
/// correction: the regular filter's, then the square-root filter's.
fn run_tiny_measurement_noise<T>(steps: usize) -> (Vec<Vec<T>>, Vec<Vec<T>>)
where
    T: MatrixDataType + Default,
{
    let builder = KalmanFilterBuilder::<2, T>::default();
    let mut regular = builder.build();
    let mut square_root = builder.build_square_root::<4>();
    let mut sensor = builder.observations().build::<1>();

    let transition = [T::one(), T::one(), T::zero(), T::one()];
    regular
        .state_transition_mut()
        .as_mut_slice()
        .copy_from_slice(&transition);
    square_root
        .state_transition_mut()
        .as_mut_slice()
        .copy_from_slice(&transition);
    regular.estimate_covariance_mut().make_scalar(t(1e8));
    square_root.covariance_factor_mut().make_scalar(t(1e4));
    regular.direct_process_noise_mut().make_scalar(t(1e-9));
    square_root
        .process_noise_factor_mut()
        .make_scalar(t(1e-9f64.sqrt()));
    sensor.observation_matrix_mut().set(0, 0, T::one());
    sensor.measurement_noise_covariance_mut().set(0, 0, t(1e-4));

    let mut x = BufferBuilder::state_vector_x::<2>().new();
    let mut P = BufferBuilder::estimate_covariance_P::<2>().new();
    let (mut regular_P, mut square_root_P) = (Vec::new(), Vec::new());
    for k in 0..steps {
        regular.predict();
        assert!(square_root.predict());
        sensor.measurement_vector_mut().set(0, 0, t(0.5 * k as f64));
        regular.correct(&mut sensor);
        assert!(square_root.correct(&mut sensor).is_applied());

        regular_P.push(regular.estimate_covariance().as_slice().to_vec());
        square_root.state(&mut x, &mut P);
        square_root_P.push(P.as_slice().to_vec());
    }

    (regular_P, square_root_P)
}

/// Returns `true` if the `2×2` matrix `P` has a Cholesky factor.
fn is_positive_definite(P: &[f32]) -> bool {
    let mut copy = [P[0], P[1], P[2], P[3]];
    MatrixData::new_mut::<2, 2, f32>(&mut copy).cholesky_decompose_lower()
}

/// In `f32`, `P − K×H×P` cancels catastrophically when `R ≪ H×P×Hᵀ`: the regular filter's
/// position variance becomes negative in the first correction. The square-root filter keeps
/// `P` positive definite in every step and close to the `f64` result.
#[test]
#[cfg_attr(
    feature = "micromath",
    ignore = "micromath's approximate sqrt compounds"
)]
fn keeps_the_covariance_positive_definite_where_the_regular_filter_fails() {
    let (regular, square_root) = run_tiny_measurement_noise::<f32>(50);
    let (reference, _) = run_tiny_measurement_noise::<f64>(50);

    assert!(regular[0][0] < 0.0, "regular P = {:?}", regular[0]);
    for (k, P) in square_root.iter().enumerate() {
        assert!(is_positive_definite(P), "step {k}: P = {P:?}");
    }

    let (last, expected) = (square_root[49][0], reference[49][0]);
    assert!(
        (f64::from(last) / expected - 1.0).abs() < 0.01,
        "{last} vs {expected}"
    );
}

#[test]
fn predict_rejects_non_finite_noise_and_changes_nothing() {
    let (_, mut square_root) = make_filters::<f64>();
    square_root
        .process_noise_factor_mut()
        .set(1, 1, f64::INFINITY);
    let before = snapshot(&square_root);

    assert!(!square_root.predict());
    assert_eq!(snapshot(&square_root), before);
}
