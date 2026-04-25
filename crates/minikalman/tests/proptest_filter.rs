#![cfg(feature = "std")]
#![forbid(unsafe_code)]

mod common;

use common::*;
use minikalman::extended::builder::KalmanFilterBuilder as ExtendedKalmanFilterBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;
use proptest::prelude::*;

const NUM_STATES: usize = 3;
const NUM_CONTROLS: usize = 1;
const NUM_OBSERVATIONS: usize = 2;

// Reduced case count for filter tests since each case iterates.
const FILTER_CASES: u32 = 64;

// ============================================================================
// Helper: build a well-conditioned regular filter with given matrices
// ============================================================================

fn build_regular_filter(
    state_init: [f32; NUM_STATES],
    p_init: [f32; NUM_STATES * NUM_STATES],
    a: [f32; NUM_STATES * NUM_STATES],
    q: [f32; NUM_STATES * NUM_STATES],
    h: [f32; NUM_OBSERVATIONS * NUM_STATES],
    r: [f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS],
) -> (
    minikalman::regular::builder::KalmanFilterType<NUM_STATES, f32>,
    minikalman::regular::builder::KalmanFilterControlType<NUM_STATES, NUM_CONTROLS, f32>,
    minikalman::regular::builder::KalmanFilterObservationType<NUM_STATES, NUM_OBSERVATIONS, f32>,
) {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut control = builder.controls().build::<NUM_CONTROLS>();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // Set state
    filter.state_vector_mut().apply(|vec| {
        for (i, &v) in state_init.iter().enumerate() {
            vec.set_at(i, 0, v);
        }
    });

    // Set A
    filter.state_transition_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, a[i * NUM_STATES + j]);
            }
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
                mat.set_at(i, j, q[i * NUM_STATES + j]);
            }
        }
    });

    // Set control (identity-like, small)
    control.control_matrix_mut().apply(|mat| {
        mat.set_at(0, 0, 1.0);
    });
    control.control_vector_mut().set_at(0, 0, 0.0);
    control
        .process_noise_covariance_mut()
        .apply(|mat| mat.make_scalar(0.01));

    // Set H
    measurement.observation_matrix_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, h[i * NUM_STATES + j]);
            }
        }
    });

    // Set R
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_OBSERVATIONS {
                mat.set_at(i, j, r[i * NUM_OBSERVATIONS + j]);
            }
        }
    });

    (filter, control, measurement)
}

// ============================================================================
// Covariance symmetry after predict/correct cycles
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn covariance_symmetry_after_cycles(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        spd_q in spd_matrix_3x3(),
        h_vals in small_fixed_vec::<{ NUM_OBSERVATIONS * NUM_STATES }>(),
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

        let (mut filter, mut _control, mut measurement) = build_regular_filter(
            state_vals, p_data,
            // A = identity for stability
            { let mut id = [0.0f32; 9]; id[0]=1.0; id[4]=1.0; id[8]=1.0; id },
            q_data,
            h_vals, r_data,
        );

        for cycle in 0..num_cycles {
            filter.predict();

            // Set measurement
            let offset = cycle * NUM_OBSERVATIONS;
            if offset + NUM_OBSERVATIONS <= meas_vals.len() {
                measurement.measurement_vector_mut().apply(|vec| {
                    for i in 0..NUM_OBSERVATIONS {
                        vec.set_at(i, 0, meas_vals[offset + i]);
                    }
                });
                filter.correct(&mut measurement);
            }
        }

        // Check symmetry of P
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
            "Covariance matrix lost symmetry after {} cycles", num_cycles
        );
    }
}

// ============================================================================
// Covariance diagonal stays non-negative
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn covariance_diagonal_nonnegative(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        spd_q in spd_matrix_3x3(),
        h_vals in small_fixed_vec::<{ NUM_OBSERVATIONS * NUM_STATES }>(),
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

        let (mut filter, mut _control, mut measurement) = build_regular_filter(
            state_vals, p_data,
            { let mut id = [0.0f32; 9]; id[0]=1.0; id[4]=1.0; id[8]=1.0; id },
            q_data,
            h_vals, r_data,
        );

        for cycle in 0..num_cycles {
            filter.predict();

            let offset = cycle * NUM_OBSERVATIONS;
            if offset + NUM_OBSERVATIONS <= meas_vals.len() {
                measurement.measurement_vector_mut().apply(|vec| {
                    for i in 0..NUM_OBSERVATIONS {
                        vec.set_at(i, 0, meas_vals[offset + i]);
                    }
                });
                filter.correct(&mut measurement);
            }

            // Check diagonal non-negativity after each step
            let mut p_check = [0.0f32; NUM_STATES * NUM_STATES];
            filter.estimate_covariance().inspect(|mat| {
                for i in 0..NUM_STATES {
                    for j in 0..NUM_STATES {
                        p_check[i * NUM_STATES + j] = mat.get_at(i, j);
                    }
                }
            });

            // Diagonal should be non-negative (allow small negative due to numerical error)
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

// ============================================================================
// No divergence on zero-input: A=I, Q=0, z=H*x should keep state bounded
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn no_divergence_zero_input(
        initial_state in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        // H with small values
        h_vals in small_fixed_vec::<{ NUM_OBSERVATIONS * NUM_STATES }>(),
        spd_r in spd_matrix_2x2(),
        num_steps in 1usize..20,
    ) {
        let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
        p_data.copy_from_slice(spd_p.as_slice());
        let q_data = [0.0f32; NUM_STATES * NUM_STATES]; // zero process noise
        let mut r_data = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        r_data.copy_from_slice(spd_r.as_slice());

        let (mut filter, mut _control, mut measurement) = build_regular_filter(
            initial_state, p_data,
            { let mut id = [0.0f32; 9]; id[0]=1.0; id[4]=1.0; id[8]=1.0; id },
            q_data,
            h_vals, r_data,
        );

        // Compute z = H * x_initial (perfect measurement, no noise)
        let mut z = [0.0f32; NUM_OBSERVATIONS];
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_STATES {
                z[i] += h_vals[i * NUM_STATES + j] * initial_state[j];
            }
        }

        // Store initial state
        let initial_state_vec = initial_state.to_vec();

        for _step in 0..num_steps {
            filter.predict();

            measurement.measurement_vector_mut().apply(|vec| {
                for (i, &v) in z.iter().enumerate() {
                    vec.set_at(i, 0, v);
                }
            });
            filter.correct(&mut measurement);
        }

        // State should remain finite
        let mut final_state = [0.0f32; NUM_STATES];
        filter.state_vector().inspect(|vec| {
            for (i, slot) in final_state.iter_mut().enumerate() {
                *slot = vec.get_at(i, 0);
            }
        });

        for (i, &v) in final_state.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "State[{}] diverged to non-finite value: {}", i, v
            );
        }

        // With A=I, Q=0, and perfect measurements, the state should converge
        // toward something reasonable (not blow up). We check that it stays
        // within a reasonable bound of the initial state.
        for (i, (&v, &init)) in final_state.iter().zip(initial_state_vec.iter()).enumerate() {
            let diff = (v - init).abs();
            prop_assert!(
                diff < 1e4,
                "State[{}] drifted too far from initial: {} vs {} (diff={})",
                i, v, init, diff
            );
        }
    }
}

// ============================================================================
// Predict + correct on SPD inputs leaves state finite
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn predict_correct_leaves_state_finite(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        h_vals in small_fixed_vec::<{ NUM_OBSERVATIONS * NUM_STATES }>(),
        spd_r in spd_matrix_2x2(),
        meas_vals in small_fixed_vec::<NUM_OBSERVATIONS>(),
    ) {
        let mut p_data = [0.0f32; NUM_STATES * NUM_STATES];
        p_data.copy_from_slice(spd_p.as_slice());
        let q_data = [0.0f32; NUM_STATES * NUM_STATES];
        let mut r_data = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        r_data.copy_from_slice(spd_r.as_slice());

        let (mut filter, mut _control, mut measurement) = build_regular_filter(
            state_vals, p_data,
            { let mut id = [0.0f32; 9]; id[0]=1.0; id[4]=1.0; id[8]=1.0; id },
            q_data,
            h_vals, r_data,
        );

        // Set measurement
        measurement.measurement_vector_mut().apply(|vec| {
            for (i, &v) in meas_vals.iter().enumerate() {
                vec.set_at(i, 0, v);
            }
        });

        // Run one predict + correct cycle
        filter.predict();
        filter.correct(&mut measurement);

        // Verify state is finite after correct.
        let mut final_state = [0.0f32; NUM_STATES];
        filter.state_vector().inspect(|vec| {
            for (i, slot) in final_state.iter_mut().enumerate() {
                *slot = vec.get_at(i, 0);
            }
        });

        for (i, &v) in final_state.iter().enumerate() {
            prop_assert!(
                v.is_finite(),
                "State[{}] is non-finite after correct", i
            );
        }
    }
}

// ============================================================================
// Extended filter: nonlinear predict/correct doesn't panic
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(FILTER_CASES))]

    #[test]
    fn extended_filter_nonlinear_no_panic(
        state_vals in small_fixed_vec::<NUM_STATES>(),
        spd_p in spd_matrix_3x3(),
        spd_q in spd_matrix_3x3(),
        h_vals in small_fixed_vec::<{ NUM_OBSERVATIONS * NUM_STATES }>(),
        spd_r in spd_matrix_2x2(),
        num_steps in 1usize..5,
    ) {
        let builder = ExtendedKalmanFilterBuilder::<NUM_STATES, f32>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

        // Set state
        filter.state_vector_mut().apply(|vec| {
            for (i, &v) in state_vals.iter().enumerate() {
                vec.set_at(i, 0, v);
            }
        });

        // Set P
        filter.estimate_covariance_mut().apply(|mat| {
            for i in 0..NUM_STATES {
                for j in 0..NUM_STATES {
                    mat.set_at(i, j, spd_p.as_slice()[i * NUM_STATES + j]);
                }
            }
        });

        // Set Q
        filter.direct_process_noise_mut().apply(|mat| {
            for i in 0..NUM_STATES {
                for j in 0..NUM_STATES {
                    mat.set_at(i, j, spd_q.as_slice()[i * NUM_STATES + j]);
                }
            }
        });

        // Set H
        measurement.observation_jacobian_matrix_mut().apply(|mat| {
            for i in 0..NUM_OBSERVATIONS {
                for j in 0..NUM_STATES {
                    mat.set_at(i, j, h_vals[i * NUM_STATES + j]);
                }
            }
        });

        // Set R
        measurement.measurement_noise_covariance_mut().apply(|mat| {
            for i in 0..NUM_OBSERVATIONS {
                for j in 0..NUM_OBSERVATIONS {
                    mat.set_at(i, j, spd_r.as_slice()[i * NUM_OBSERVATIONS + j]);
                }
            }
        });

        for _step in 0..num_steps {
            // Nonlinear prediction: clamp to bounded values
            filter.predict_nonlinear(|current, next| {
                for i in 0..NUM_STATES {
                    let val = current.get_at(i, 0);
                    next.set_at(i, 0, val.clamp(-100.0, 100.0) * 0.99);
                }
            });

            // Set measurement
            measurement.measurement_vector_mut().apply(|vec| {
                for i in 0..NUM_OBSERVATIONS {
                    vec.set_at(i, 0, 0.0);
                }
            });

            // Nonlinear correction: bounded observation function
            filter.correct_nonlinear::<_, _, NUM_OBSERVATIONS>(&mut measurement, |state, obs| {
                for i in 0..NUM_OBSERVATIONS {
                    let val = if i < state.rows() { state.get_at(i, 0) } else { 0.0 };
                    obs.set_at(i, 0, val.sin().clamp(-100.0, 100.0));
                }
            });
        }

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
                "Extended filter state[{}] is non-finite", i
            );
        }
    }
}
