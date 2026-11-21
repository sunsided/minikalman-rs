//! The fixed-lag smoother must match the fixed-interval RTS smoother over the same window.

#![cfg(feature = "alloc")]
#![forbid(unsafe_code)]

use minikalman::matrix::MatrixDataArray;
use minikalman::models::motion;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;
use minikalman::smoother::{FixedLagSmoother, RtsSmoother};

/// A `[position, velocity]` state vector.
type Vector = MatrixDataArray<2, 1, 2, f32>;

/// A 2×2 covariance or scratch matrix.
type Square = MatrixDataArray<2, 2, 4, f32>;

/// Fixed-lag smoother over a 2-state model with array buffers and a window of `N` steps.
type Smoother<const N: usize> =
    FixedLagSmoother<2, N, f32, Vector, Square, Square, Square, Square, Vector>;

/// Noisy position measurements of a track moving at about one unit per step.
const MEASUREMENTS: [f32; 9] = [0.1, 1.2, 1.9, 3.1, 3.8, 5.3, 5.9, 7.2, 7.8];

/// A recorded filter run, indexed by step.
struct History {
    /// The state transition matrix `A`, constant over the run.
    a: Square,
    /// Filtered states `x_k|k`.
    x: Vec<Vector>,
    /// Filtered covariances `P_k|k`.
    p: Vec<Square>,
    /// Predicted states `x_k|k−1`.
    x_predicted: Vec<Vector>,
    /// Predicted covariances `P_k|k−1`.
    p_predicted: Vec<Square>,
}

/// Runs a constant-velocity filter over [`MEASUREMENTS`] and records every step.
fn record() -> History {
    let builder = KalmanFilterBuilder::<2, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<1>();

    let model = motion::constant_velocity(1.0, 0.1);
    filter
        .state_transition_mut()
        .apply(|a| model.fill_transition(a));
    filter
        .direct_process_noise_mut()
        .apply(|q| model.fill_process_noise(q));
    filter
        .estimate_covariance_mut()
        .apply(|p| p.make_scalar(10.0));
    measurement
        .observation_matrix_mut()
        .apply(|h| h.set(0, 0, 1.0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| r.set(0, 0, 1.0));

    let steps = MEASUREMENTS.len();
    let mut history = History {
        a: Square::new([0.0; 4]),
        x: vec![Vector::new([0.0; 2]); steps],
        p: vec![Square::new([0.0; 4]); steps],
        x_predicted: vec![Vector::new([0.0; 2]); steps],
        p_predicted: vec![Square::new([0.0; 4]); steps],
    };
    filter
        .state_transition()
        .inspect(|a| a.copy_to(&mut history.a));

    for (k, z) in MEASUREMENTS.into_iter().enumerate() {
        filter.predict();
        filter
            .state_vector()
            .inspect(|v| v.copy_to(&mut history.x_predicted[k]));
        filter
            .estimate_covariance()
            .inspect(|m| m.copy_to(&mut history.p_predicted[k]));

        measurement
            .measurement_vector_mut()
            .apply(|v| v.set(0, 0, z));
        filter.correct(&mut measurement);
        filter
            .state_vector()
            .inspect(|v| v.copy_to(&mut history.x[k]));
        filter
            .estimate_covariance()
            .inspect(|m| m.copy_to(&mut history.p[k]));
    }

    history
}

/// Creates an RTS smoother with zeroed scratch buffers.
fn rts() -> RtsSmoother<2, f32, Square, Square, Square, Vector> {
    RtsSmoother::new(
        Square::new([0.0; 4]),
        Square::new([0.0; 4]),
        Square::new([0.0; 4]),
        Vector::new([0.0; 2]),
    )
}

/// Creates an empty fixed-lag smoother with a window of `N` steps.
fn fixed_lag<const N: usize>() -> Smoother<N> {
    FixedLagSmoother::new(rts(), Vector::new([0.0; 2]), Square::new([0.0; 4]))
}

/// Smooths the steps `0..=t` of `history` with the fixed-interval smoother and returns step `k`.
fn rts_estimate(history: &History, t: usize, k: usize) -> (Vector, Square) {
    let mut x = history.x[..=t].to_vec();
    let mut p = history.p[..=t].to_vec();
    assert!(rts().smooth(
        &history.a,
        &mut x,
        &mut p,
        &history.x_predicted[..=t],
        &history.p_predicted[..=t],
    ));

    (x[k].clone(), p[k].clone())
}

/// Pushes every step of `history` and checks each full-window estimate against the
/// fixed-interval smoother over the same steps.
fn assert_matches_rts<const N: usize>(history: &History) {
    let mut smoother = fixed_lag::<N>();

    for t in 0..history.x.len() {
        let smoothed = smoother.push(
            &history.a,
            &history.x[t],
            &history.p[t],
            &history.x_predicted[t],
            &history.p_predicted[t],
        );
        if t + 1 < N {
            assert_eq!(smoothed, None, "window of {N} not full at step {t}");
            continue;
        }

        assert_eq!(smoothed, Some(true));
        let (x, p) = rts_estimate(history, t, t + 1 - N);
        assert_eq!(
            smoother.state().as_slice(),
            x.as_slice(),
            "x, N = {N}, t = {t}"
        );
        assert_eq!(
            smoother.covariance().as_slice(),
            p.as_slice(),
            "P, N = {N}, t = {t}"
        );
    }
}

/// With the window covering the whole run, the single estimate is the smoothed first step.
#[test]
fn window_over_whole_run_matches_rts() {
    let history = record();
    assert_matches_rts::<{ MEASUREMENTS.len() }>(&history);
}

/// A window shorter than the run wraps around its ring buffer several times.
#[test]
fn ring_wraparound_matches_rts() {
    let history = record();
    assert_matches_rts::<2>(&history);
    assert_matches_rts::<3>(&history);
    assert_matches_rts::<4>(&history);
}

/// A window of one step returns the filtered estimate unchanged.
#[test]
fn window_of_one_returns_filtered_values() {
    let history = record();
    assert_matches_rts::<1>(&history);
}

/// Nothing is smoothed until the window is full.
#[test]
fn returns_none_until_window_is_full() {
    let history = record();
    let mut smoother = fixed_lag::<4>();

    let results: Vec<_> = (0..5)
        .map(|t| {
            smoother.push(
                &history.a,
                &history.x[t],
                &history.p[t],
                &history.x_predicted[t],
                &history.p_predicted[t],
            )
        })
        .collect();

    assert_eq!(results, [None, None, None, Some(true), Some(true)]);
}
