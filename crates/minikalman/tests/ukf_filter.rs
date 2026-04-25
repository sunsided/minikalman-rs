#![cfg(feature = "std")]
#![forbid(unsafe_code)]

mod common;

use common::*;
use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder;

const NUM_STATES: usize = 3;
const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
const NUM_OBSERVATIONS: usize = 2;

// ============================================================================
// Step 1: Basic predict_nonlinear test
// ============================================================================

#[test]
fn ukf_predict_nonlinear_state_changes() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();

    // Set initial state [1.0, 0.5, 0.2]
    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, 0.5);
        vec.set_at(2, 0, 0.2);
    });

    // Set P = identity
    filter.estimate_covariance_mut().make_identity();

    // Set Q = small scalar
    filter.direct_process_noise_mut().make_scalar(0.01);

    // Predict with constant-velocity model: x' = x + v*dt
    let dt = 1.0f32;
    filter.predict_nonlinear(|next| {
        let x = next.as_matrix().get_at(0, 0);
        let v = next.as_matrix().get_at(1, 0);
        next.as_matrix_mut().set_at(0, 0, x + v * dt);
        // v stays the same, other states pass through
    });

    // Verify state is finite
    let mut state = [0.0f32; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    for (i, &v) in state.iter().enumerate() {
        assert!(v.is_finite(), "State[{}] is non-finite: {}", i, v);
    }

    // Expected: x[0] = 1.0 + 0.5*1.0 = 1.5, x[1] = 0.5, x[2] = 0.2
    assert!(
        (state[0] - 1.5).abs() < 0.1,
        "State[0] = {}, expected ~1.5",
        state[0]
    );

    // Verify P is symmetric
    let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });
    assert!(
        is_symmetric(&p_data, NUM_STATES, EPS_RELAXED),
        "P lost symmetry after predict"
    );
}

// ============================================================================
// Step 2: Full predict + correct cycle
// ============================================================================

#[test]
fn ukf_predict_correct_cycle() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Set initial state
    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, 1.0);
        vec.set_at(2, 0, 0.5);
    });

    // Set P = identity
    filter.estimate_covariance_mut().make_identity();

    // Set Q = small scalar
    filter.direct_process_noise_mut().make_scalar(0.01);

    // Set R = diagonal
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.1);
    });

    // Set measurement z = [2.0, 4.0] (expecting h(x) = [x[0], x[0]^2])
    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, 4.0);
    });

    // Predict with identity transition
    filter.predict_nonlinear(|_next| {});

    // Correct with nonlinear observation h(x) = [x[0], x[0]^2]
    filter.correct_sigma_point(&mut measurement, |_sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = _sigma_pred[j];
            observed.set_at(0, j, x0);
            observed.set_at(1, j, x0 * x0);
        }
    });

    // Verify state is finite
    let mut state = [0.0f32; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    for (i, &v) in state.iter().enumerate() {
        assert!(
            v.is_finite(),
            "State[{}] is non-finite after correct: {}",
            i,
            v
        );
    }

    // Verify P is symmetric
    let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });
    assert!(
        is_symmetric(&p_data, NUM_STATES, EPS_RELAXED),
        "P lost symmetry after correct"
    );
}

// ============================================================================
// Step 3: Parameter getter/setter tests
// ============================================================================

#[test]
fn ukf_parameter_getters_return_defaults() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let filter = builder.build::<NUM_SIGMA>();

    // Builder defaults: alpha=1, beta=2, kappa=1
    assert_eq!(filter.alpha(), 1.0f32);
    assert_eq!(filter.beta(), 2.0f32);
    assert_eq!(filter.kappa(), 1.0f32);

    // lambda = alpha^2 * (n + kappa) - n = 1 * (3 + 1) - 3 = 1
    let expected_lambda = 1.0f32 * (3.0 + 1.0) - 3.0;
    assert!(
        (filter.lambda() - expected_lambda).abs() < 1e-6,
        "lambda = {}, expected {}",
        filter.lambda(),
        expected_lambda
    );
}

#[test]
fn ukf_parameter_setters_update_values() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();

    filter.set_alpha(0.5);
    filter.set_beta(3.0);
    filter.set_kappa(0.0);

    assert_eq!(filter.alpha(), 0.5f32);
    assert_eq!(filter.beta(), 3.0f32);
    assert_eq!(filter.kappa(), 0.0f32);

    // lambda = 0.25 * (3 + 0) - 3 = 0.75 - 3 = -2.25
    let expected_lambda = 0.5 * 0.5 * (3.0 + 0.0) - 3.0;
    assert!(
        (filter.lambda() - expected_lambda).abs() < 1e-6,
        "lambda = {}, expected {}",
        filter.lambda(),
        expected_lambda
    );
}

// ============================================================================
// Step 4: sigma_propagated accessor test
// ============================================================================

#[test]
fn ukf_sigma_propagated_accessor() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();

    // Before predict, sigma_propagated should be all zeros (default)
    let sigma_prop = filter.sigma_propagated();
    let mut all_zero = true;
    minikalman::matrix::AsMatrix::as_matrix(sigma_prop).inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_SIGMA {
                if mat.get_at(i, j) != 0.0 {
                    all_zero = false;
                }
            }
        }
    });
    assert!(all_zero, "sigma_propagated should be zero before predict");

    // After predict, it should contain propagated values
    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, 0.0);
        vec.set_at(2, 0, 0.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);

    filter.predict_nonlinear(|_next| {});

    // sigma_propagated should now have non-zero values
    let sigma_prop = filter.sigma_propagated();
    let mut any_nonzero = false;
    minikalman::matrix::AsMatrix::as_matrix(sigma_prop).inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_SIGMA {
                if mat.get_at(i, j) != 0.0 {
                    any_nonzero = true;
                }
            }
        }
    });
    assert!(
        any_nonzero,
        "sigma_propagated should be non-zero after predict"
    );
}

// ============================================================================
// Step 5: UnscentedObservation direct method tests
// ============================================================================

#[test]
fn ukf_observation_accessors() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Test measurement_vector_mut / measurement_vector
    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.5);
        vec.set_at(1, 0, 2.5);
    });

    let mut z = [0.0f32; NUM_OBSERVATIONS];
    measurement.measurement_vector().inspect(|vec| {
        for (i, slot) in z.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });
    assert!((z[0] - 1.5).abs() < 1e-6);
    assert!((z[1] - 2.5).abs() < 1e-6);

    // Test measurement_noise_covariance_mut / measurement_noise_covariance
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.5);
        mat.set_at(1, 1, 0.3);
    });

    let mut r_data = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
    measurement.measurement_noise_covariance().inspect(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_OBSERVATIONS {
                r_data[i * NUM_OBSERVATIONS + j] = mat.get_at(i, j);
            }
        }
    });
    assert!((r_data[0] - 0.5).abs() < 1e-6);
    assert!((r_data[3] - 0.3).abs() < 1e-6);
}

#[test]
fn ukf_observation_correct_nonlinear() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Set up filter
    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, 0.5);
        vec.set_at(2, 0, 0.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.01);

    // Set up measurement
    measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 1.0);
        vec.set_at(1, 0, 0.25);
    });
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.01);
    });

    // Predict first (required to populate sigma points)
    filter.predict_nonlinear(|_next| {});

    // Now manually run sigma point observation and correct
    // This exercises correct_with_observed -> correct_with_weights internally
    filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
        for j in 0..NUM_SIGMA {
            let x0 = sigma_pred[j];
            observed.set_at(0, j, x0);
            observed.set_at(1, j, x0 * x0);
        }
    });

    // State should be finite
    let mut state = [0.0f32; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });
    for (i, &v) in state.iter().enumerate() {
        assert!(v.is_finite(), "State[{}] non-finite: {}", i, v);
    }
}

// ============================================================================
// Additional: Trait implementation verification
// ============================================================================

#[test]
fn ukf_implements_sigma_point_predict() {
    fn assert_predict<T: KalmanFilterSigmaPointPredict<NUM_STATES, f32>>(_: &T) {}

    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let filter = builder.build::<NUM_SIGMA>();
    assert_predict(&filter);
}

#[test]
fn ukf_implements_sigma_point_correct() {
    fn assert_correct<T: KalmanFilterSigmaPointCorrect<NUM_STATES, NUM_SIGMA, f32>>(_: &T) {}

    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let filter = builder.build::<NUM_SIGMA>();
    assert_correct(&filter);
}

// ============================================================================
// Additional: Multiple predict-correct cycles stability
// ============================================================================

#[test]
fn ukf_multiple_cycles_stability() {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    filter.state_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 0.0);
        vec.set_at(1, 0, 1.0);
        vec.set_at(2, 0, 0.0);
    });
    filter.estimate_covariance_mut().make_identity();
    filter.direct_process_noise_mut().make_scalar(0.001);

    measurement.measurement_noise_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.1);
        mat.set_at(1, 1, 0.1);
    });

    for step in 0..5 {
        // Predict: x[0] += x[1]
        filter.predict_nonlinear(|next| {
            let x0 = next.as_matrix().get_at(0, 0);
            let x1 = next.as_matrix().get_at(1, 0);
            next.as_matrix_mut().set_at(0, 0, x0 + x1 * 0.1);
        });

        // Set measurement
        measurement.measurement_vector_mut().apply(|vec| {
            vec.set_at(0, 0, (step as f32) * 0.1);
            vec.set_at(1, 0, 1.0);
        });

        // Correct
        filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
            for j in 0..NUM_SIGMA {
                let x0 = sigma_pred[j];
                let x1 = sigma_pred[NUM_SIGMA + j];
                observed.set_at(0, j, x0);
                observed.set_at(1, j, x1);
            }
        });
    }

    // Verify state remains finite
    let mut state = [0.0f32; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });
    for (i, &v) in state.iter().enumerate() {
        assert!(
            v.is_finite(),
            "State[{}] diverged after multiple cycles: {}",
            i,
            v
        );
    }

    // Verify P symmetry
    let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
    filter.estimate_covariance().inspect(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                p_data[i * NUM_STATES + j] = mat.get_at(i, j);
            }
        }
    });
    assert!(
        is_symmetric(&p_data, NUM_STATES, EPS_RELAXED),
        "P lost symmetry after cycles"
    );
}
