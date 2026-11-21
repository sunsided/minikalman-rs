//! `predict_tuned(λ)` scales the propagated covariance by `1/λ²` before adding `Q`.

#![cfg(feature = "alloc")]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;

const LAMBDA: f32 = 0.5;

/// With `A = I`, `P = I` and `Q = 0.1·I`, one tuned prediction gives `P = I/λ² + Q = 4.1·I`.
fn assert_tuned<F>(filter: &F)
where
    F: KalmanFilterEstimateCovariance<2, f32>,
{
    let p = filter.estimate_covariance();
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j {
                1.0 / (LAMBDA * LAMBDA) + 0.1
            } else {
                0.0
            };
            let actual = p.as_matrix().get_at(i, j);
            assert!((actual - expected).abs() < 1e-5, "P[{i}][{j}] = {actual}");
        }
    }
}

#[test]
fn regular_predict_tuned_scales_covariance() {
    let builder = RegularBuilder::<2, f32>::default();
    let mut filter = builder.build();
    filter.state_transition_mut().make_identity();
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);

    filter.predict_tuned(LAMBDA);

    assert_tuned(&filter);
}

#[test]
fn extended_predict_tuned_scales_covariance() {
    let builder = ExtendedBuilder::<2, f32>::default();
    let mut filter = builder.build();
    filter.state_transition_jacobian_mut().make_identity();
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);

    filter.predict_tuned_nonlinear(LAMBDA, |x, next| {
        x.as_matrix().copy_to(next.as_matrix_mut())
    });

    assert_tuned(&filter);
}
