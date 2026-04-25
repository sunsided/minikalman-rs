#![cfg(feature = "std")]
#![forbid(unsafe_code)]

mod common;

use common::*;
use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder;

const NUM_STATES: usize = 2;
const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
const NUM_OBSERVATIONS: usize = 2;

#[test]
fn ukf_compute_predicted_measurement_uniform_sigma() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Set state and covariance
    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 5.0);
        vec.set_at(1, 0, 3.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);

    // Predict to populate sigma points
    filter.predict_nonlinear(|_next| {});

    // Set measurement z to match expected sigma mean
    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 5.0);
        vec.set_at(1, 0, 3.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.1);
    });

    // Correct with identity observation: h(x) = [x[0], x[1]]
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            observed.set_at(0, j, sigma_pred[j]);
            observed.set_at(1, j, sigma_pred[NUM_SIGMA + j]);
        }
    });

    // State should remain close to prior since z matches prediction
    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    for (i, &v) in state.iter().enumerate() {
        assert!(v.is_finite(), "State[{}] is non-finite: {}", i, v);
    }

    // P should be reduced but still positive
    let mut p_data = [0.0f64; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });
    assert!(p_data[0] > 0.0, "P[0,0] should be positive");
    assert!(p_data[3] > 0.0, "P[1,1] should be positive");
}

#[test]
fn ukf_compute_predicted_measurement_constant_sigma() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 42.0);
        vec.set_at(1, 0, 7.0);
    });
    filter.estimate_covariance_mut().make_scalar(1e-10); // Near-zero covariance
    filter.direct_process_noise_mut().make_scalar(1e-10);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 42.0);
        vec.set_at(1, 0, 7.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.01);
        mat.set_at(1, 1, 0.01);
    });

    // Nonlinear observation: h(x) = [x[0]^2 / 42, x[1]]
    // When x[0]=42, h(x) = [42, x[1]]
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0 * x0 / 42.0);
            observed.set_at(1, j, x1);
        }
    });

    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    // Should be close to prior since z matches
    assert!(
        (state[0] - 42.0).abs() < 1.0,
        "State[0] = {}, expected close to 42.0",
        state[0]
    );
    assert!(
        (state[1] - 7.0).abs() < 1.0,
        "State[1] = {}, expected close to 7.0",
        state[1]
    );
}

#[test]
fn ukf_compute_innovation_covariance_symmetric() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, 1.0);
    });
    filter.estimate_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 1.0);
        mat.set_at(0, 1, 0.2);
        mat.set_at(1, 0, 0.2);
        mat.set_at(1, 1, 0.5);
    });
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, 1.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.5);
        mat.set_at(0, 1, 0.1);
        mat.set_at(1, 0, 0.1);
        mat.set_at(1, 1, 0.3);
    });

    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0 + x1);
            observed.set_at(1, j, x0 - x1);
        }
    });

    // Verify symmetry of updated P
    let mut p_data = [0.0f64; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });

    assert!(
        (p_data[1] - p_data[2]).abs() < EPS_RELAXED as f64,
        "P lost symmetry: P[0,1]={}, P[1,0]={}",
        p_data[1],
        p_data[2]
    );
}

#[test]
fn ukf_compute_innovation_covariance_zero_variance() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 5.0);
        vec.set_at(1, 0, 10.0);
    });
    filter.estimate_covariance_mut().make_scalar(1e-12);
    filter.direct_process_noise_mut().make_scalar(1e-12);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 5.0);
        vec.set_at(1, 0, 10.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.25);
        mat.set_at(0, 1, 0.05);
        mat.set_at(1, 0, 0.05);
        mat.set_at(1, 1, 0.36);
    });

    // Identity observation with zero variance sigma points
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            observed.set_at(0, j, sigma_pred[j]);
            observed.set_at(1, j, sigma_pred[NUM_SIGMA + j]);
        }
    });

    // State should barely change
    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    assert!(
        (state[0] - 5.0).abs() < 0.01,
        "State[0] = {}, expected ~5.0",
        state[0]
    );
    assert!(
        (state[1] - 10.0).abs() < 0.01,
        "State[1] = {}, expected ~10.0",
        state[1]
    );
}

#[test]
fn ukf_compute_cross_covariance_basic() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, 0.5);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.5);
        vec.set_at(1, 0, 0.8);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.1);
    });

    // Observation with correlation: h(x) = [x[0] * 2, x[1] * 3]
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0 * 2.0);
            observed.set_at(1, j, x1 * 3.0);
        }
    });

    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    for (i, &v) in state.iter().enumerate() {
        assert!(v.is_finite(), "State[{}] is non-finite: {}", i, v);
    }

    // State should have moved toward measurement
    assert!(
        (state[0] - 1.0).abs() > 0.01 || (state[1] - 0.5).abs() > 0.01,
        "State should have been corrected"
    );
}

#[test]
fn ukf_compute_cross_covariance_zero_cross() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 3.0);
        vec.set_at(1, 0, 6.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 3.0);
        vec.set_at(1, 0, 6.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.1);
    });

    // Constant observation regardless of sigma points => zero cross-cov
    filter.correct_sigma_point(&mut measurement, |_sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            observed.set_at(0, j, 3.0);
            observed.set_at(1, j, 6.0);
        }
    });

    // State should remain essentially unchanged (no information from measurement)
    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    assert!(
        (state[0] - 3.0).abs() < 0.01,
        "State[0] = {}, expected ~3.0",
        state[0]
    );
    assert!(
        (state[1] - 6.0).abs() < 0.01,
        "State[1] = {}, expected ~6.0",
        state[1]
    );
}

#[test]
fn ukf_correct_with_weights_weighted_average() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 18.0);
        vec.set_at(1, 0, 9.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 18.0);
        vec.set_at(1, 0, 9.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.1);
    });

    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0);
            observed.set_at(1, j, x1);
        }
    });

    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    assert!(state[0].is_finite());
    assert!(state[1].is_finite());

    let mut p_data = [0.0f64; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });
    assert!(p_data[0] > 0.0);
    assert!(p_data[3] > 0.0);
}

#[test]
fn ukf_correct_with_weights_high_center_weight() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // High alpha => more weight spread, low alpha => center-weighted
    filter.set_alpha(0.01);
    filter.set_beta(2.0);
    filter.set_kappa(0.0);

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 50.0);
        vec.set_at(1, 0, 25.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 50.0);
        vec.set_at(1, 0, 25.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 1.0);
        mat.set_at(1, 1, 1.0);
    });

    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0);
            observed.set_at(1, j, x1);
        }
    });

    // Should produce finite results
    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    assert!(state[0].is_finite());
    assert!(state[1].is_finite());
    assert!(filter
        .estimate_covariance()
        .inspect(|mat| mat.get_at(0, 0))
        .is_finite());
}

#[test]
fn ukf_predicted_measurement_asymmetric_sigma() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Asymmetric covariance to create asymmetric sigma points
    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, 1.0);
    });
    filter.estimate_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 4.0);
        mat.set_at(0, 1, 1.0);
        mat.set_at(1, 0, 1.0);
        mat.set_at(1, 1, 1.0);
    });
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, 1.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.01);
        mat.set_at(1, 1, 0.01);
    });

    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            observed.set_at(0, j, sigma_pred[j]);
            observed.set_at(1, j, sigma_pred[NUM_SIGMA + j]);
        }
    });

    // State should remain close to prior since z matches predicted mean
    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    assert!(
        (state[0] - 2.0).abs() < 0.5,
        "State[0] = {}, expected close to 2.0",
        state[0]
    );
    assert!(
        (state[1] - 1.0).abs() < 0.5,
        "State[1] = {}, expected close to 1.0",
        state[1]
    );
}

#[test]
fn ukf_all_functions_finite_with_varied_inputs() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.5);
        vec.set_at(1, 0, -0.5);
    });
    filter.estimate_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 3.0);
        mat.set_at(0, 1, 0.5);
        mat.set_at(1, 0, 0.5);
        mat.set_at(1, 1, 2.0);
    });
    filter.direct_process_noise_mut().make_scalar(0.05);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, std::f64::consts::PI);
        vec.set_at(1, 0, -2.71);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 2.0);
        mat.set_at(0, 1, 0.3);
        mat.set_at(1, 0, 0.3);
        mat.set_at(1, 1, 1.5);
    });

    // Nonlinear observation
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0.sin() + x1.cos());
            observed.set_at(1, j, x0 * x1);
        }
    });

    // Verify state is finite
    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });
    for (i, &v) in state.iter().enumerate() {
        assert!(v.is_finite(), "State[{}] is non-finite: {}", i, v);
    }

    // Verify P is finite
    let mut p_data = [0.0f64; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });
    for (i, &v) in p_data.iter().enumerate() {
        assert!(v.is_finite(), "P[{}] is non-finite: {}", i, v);
    }
}

#[test]
fn ukf_innovation_covariance_with_nonlinear_obs() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f64>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, 2.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.1);

    filter.predict_nonlinear(|_next| {});

    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, -1.0);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.5);
        mat.set_at(1, 1, 0.5);
    });

    // Nonlinear: h(x) = [x[0]^2, x[0]*x[1]]
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            let x1 = sigma_pred[NUM_SIGMA + j];
            observed.set_at(0, j, x0 * x0);
            observed.set_at(1, j, x0 * x1);
        }
    });

    let mut state = [0.0f64; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    // State should be finite and somewhat corrected
    for (i, &v) in state.iter().enumerate() {
        assert!(v.is_finite(), "State[{}] is non-finite: {}", i, v);
    }
}
