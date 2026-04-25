#![cfg(feature = "std")]
#![forbid(unsafe_code)]

mod common;

use common::*;
use minikalman::prelude::*;
use minikalman::unscented::builder::KalmanFilterBuilder;
use proptest::prelude::*;

const NUM_STATES: usize = 3;
const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
const NUM_OBSERVATIONS: usize = 2;

const FILTER_CASES: u32 = 64;

// ============================================================================
// Helper: build a UKF filter with given matrices
// ============================================================================

fn build_ukf_filter(
    state_init: [f32; NUM_STATES],
    p_init: [f32; NUM_STATES * NUM_STATES],
    q_init: [f32; NUM_STATES * NUM_STATES],
    r_init: [f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS],
) -> (
    minikalman::unscented::builder::KalmanFilterType<NUM_STATES, NUM_SIGMA, f32>,
    minikalman::unscented::builder::KalmanFilterObservationType<
        NUM_STATES,
        NUM_OBSERVATIONS,
        NUM_SIGMA,
        f32,
    >,
) {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();

    // Set state
    filter.state_vector_mut().apply(|vec| {
        for (i, &v) in state_init.iter().enumerate() {
            vec.set_at(i, 0, v);
        }
    });

    // Set P
    filter.estimate_covariance_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, p_init[i * NUM_STATES + j]);
            }
        }
    });

    // Set Q
    filter.direct_process_noise_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, q_init[i * NUM_STATES + j]);
            }
        }
    });

    // Set R
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_OBSERVATIONS {
                mat.set_at(i, j, r_init[i * NUM_OBSERVATIONS + j]);
            }
        }
    });

    (filter, measurement)
}

// ============================================================================
// Property 1: predict + correct leaves state finite
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn ukf_predict_correct_leaves_state_finite(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        spd_q in spd_matrix_3x3(),
        spd_r in spd_matrix_2x2(),
        meas_vals in small_fixed_vec::<NUM_OBSERVATIONS>(),
    ) {
        let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
        p_data.copy_from_slice(spd_p.as_slice());
        let mut q_data = [0.0f32; NUM_STATES * NUM_STATES];
        q_data.copy_from_slice(spd_q.as_slice());
        let mut r_data = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        r_data.copy_from_slice(spd_r.as_slice());

        let (mut filter, mut measurement) = build_ukf_filter(
            state_vals, p_data, q_data, r_data,
        );

        // Set measurement
        measurement.measurement_vector_mut().apply(|vec| {
            for (i, &v) in meas_vals.iter().enumerate() {
                vec.set_at(i, 0, v);
            }
        });

        // Predict with a simple nonlinear transition
        filter.predict_nonlinear(|next| {
            let x0 = next.as_matrix().get_at(0, 0);
            let x1 = next.as_matrix().get_at(1, 0);
            // x0' = x0 + 0.1 * x1 (simple coupling)
            next.as_matrix_mut().set_at(0, 0, x0 + 0.1 * x1);
        });

        // Correct with nonlinear observation h(x) = [x[0], x[1]]
        filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
            for j in 0..NUM_SIGMA {
                observed.set_at(0, j, sigma_pred[j]);
                observed.set_at(1, j, sigma_pred[NUM_SIGMA + j]);
            }
        });

        // Verify state is finite
        let mut final_state = [0.0f32; NUM_STATES];
        filter.state_vector().inspect(|vec| {
            for (i, slot) in final_state.iter_mut().enumerate() {
                *slot = vec.get_at(i, 0);
            }
        });

        for (i, &v) in final_state.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "State[{}] is non-finite after predict+correct: {}", i, v
            );
        }
    }
}

// ============================================================================
// Property 2: Covariance symmetry after cycles
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn ukf_covariance_symmetry(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        spd_q in spd_matrix_3x3(),
        spd_r in spd_matrix_2x2(),
        num_cycles in 1usize..8,
        meas_vals in small_finite_vec(NUM_OBSERVATIONS * 8),
    ) {
        let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
        p_data.copy_from_slice(spd_p.as_slice());
        let mut q_data = [0.0f32; NUM_STATES * NUM_STATES];
        q_data.copy_from_slice(spd_q.as_slice());
        let mut r_data = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        r_data.copy_from_slice(spd_r.as_slice());

        let (mut filter, mut measurement) = build_ukf_filter(
            state_vals, p_data, q_data, r_data,
        );

        for cycle in 0..num_cycles {
            // Predict
            filter.predict_nonlinear(|next| {
                let x0 = next.as_matrix().get_at(0, 0);
                let x1 = next.as_matrix().get_at(1, 0);
                next.as_matrix_mut().set_at(0, 0, x0 + 0.1 * x1);
            });

            // Set measurement
            let offset = cycle * NUM_OBSERVATIONS;
            if offset + NUM_OBSERVATIONS <= meas_vals.len() {
                measurement.measurement_vector_mut().apply(|vec| {
                    for i in 0..NUM_OBSERVATIONS {
                        vec.set_at(i, 0, meas_vals[offset + i]);
                    }
                });
            }

            // Correct
            filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
                for j in 0..NUM_SIGMA {
                    observed.set_at(0, j, sigma_pred[j]);
                    observed.set_at(1, j, sigma_pred[NUM_SIGMA + j]);
                }
            });
        }

        // Check P symmetry
        let mut p_check = [0.0f32; NUM_STATES * NUM_STATES];
        filter.estimate_covariance().inspect(|mat| {
            for i in 0..NUM_STATES {
                for j in 0..NUM_STATES {
                    p_check[i * NUM_STATES + j] = mat.get_at(i, j);
                }
            }
        });

        prop_assert!(
            is_symmetric(&p_check, NUM_STATES, EPS_RELAXED),
            "Covariance lost symmetry after {} cycles", num_cycles
        );
    }
}

// ============================================================================
// Property 3: Covariance diagonal non-negative
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn ukf_covariance_diagonal_nonnegative(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        spd_q in spd_matrix_3x3(),
        spd_r in spd_matrix_2x2(),
        num_cycles in 1usize..6,
        meas_vals in small_finite_vec(NUM_OBSERVATIONS * 8),
    ) {
        let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
        p_data.copy_from_slice(spd_p.as_slice());
        let mut q_data = [0.0f32; NUM_STATES * NUM_STATES];
        q_data.copy_from_slice(spd_q.as_slice());
        let mut r_data = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        r_data.copy_from_slice(spd_r.as_slice());

        let (mut filter, mut measurement) = build_ukf_filter(
            state_vals, p_data, q_data, r_data,
        );

        for cycle in 0..num_cycles {
            filter.predict_nonlinear(|next| {
                let x0 = next.as_matrix().get_at(0, 0);
                let x1 = next.as_matrix().get_at(1, 0);
                next.as_matrix_mut().set_at(0, 0, x0 + 0.1 * x1);
            });

            let offset = cycle * NUM_OBSERVATIONS;
            if offset + NUM_OBSERVATIONS <= meas_vals.len() {
                measurement.measurement_vector_mut().apply(|vec| {
                    for i in 0..NUM_OBSERVATIONS {
                        vec.set_at(i, 0, meas_vals[offset + i]);
                    }
                });
            }

            filter.correct_sigma_point(&mut measurement, |sigma_pred, observed| {
                for j in 0..NUM_SIGMA {
                    observed.set_at(0, j, sigma_pred[j]);
                    observed.set_at(1, j, sigma_pred[NUM_SIGMA + j]);
                }
            });

            // Check diagonal non-negativity
            let mut p_check = [0.0f32; NUM_STATES * NUM_STATES];
            filter.estimate_covariance().inspect(|mat| {
                for i in 0..NUM_STATES {
                    for j in 0..NUM_STATES {
                        p_check[i * NUM_STATES + j] = mat.get_at(i, j);
                    }
                }
            });

            for i in 0..NUM_STATES {
                let diag = p_check[i * NUM_STATES + i];
                prop_assert!(
                    diag > -EPS_RELAXED,
                    "Covariance diagonal[{}] = {} went negative at cycle {}",
                    i, diag, cycle
                );
            }
        }
    }
}
