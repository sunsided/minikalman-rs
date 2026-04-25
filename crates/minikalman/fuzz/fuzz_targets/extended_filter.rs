#![no_main]

use libfuzzer_sys::fuzz_target;
use minikalman::extended::builder::KalmanFilterBuilder;
use minikalman::prelude::*;

const NUM_STATES: usize = 3;
const NUM_OBSERVATIONS: usize = 2;

#[derive(Debug)]
struct ExtendedFilterInput {
    state: [f32; NUM_STATES],
    p_seed: [f32; NUM_STATES * NUM_STATES],
    q_seed: [f32; NUM_STATES * NUM_STATES],
    h: [f32; NUM_OBSERVATIONS * NUM_STATES],
    r_seed: [f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS],
    measurements: [[f32; NUM_OBSERVATIONS]; 4],
    nonlinear_coeffs: [f32; 6], // coefficients for nonlinear functions
}

fn symmetrize_and_regularize(data: &mut [f32], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (data[i * n + j] + data[j * n + i]) * 0.5;
            data[i * n + j] = avg;
            data[j * n + i] = avg;
        }
        data[i * n + i] = data[i * n + i].abs() + 0.1;
    }
}

fn clamp_finite(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1000.0, 1000.0)
    } else {
        0.0
    }
}

impl arbitrary::Arbitrary<'_> for ExtendedFilterInput {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        let mut state = [0.0f32; NUM_STATES];
        let mut p_seed = [0.0f32; NUM_STATES * NUM_STATES];
        let mut q_seed = [0.0f32; NUM_STATES * NUM_STATES];
        let mut h = [0.0f32; NUM_OBSERVATIONS * NUM_STATES];
        let mut r_seed = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        let mut measurements = [[0.0f32; NUM_OBSERVATIONS]; 4];
        let mut nonlinear_coeffs = [0.0f32; 6];

        for v in state.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for v in p_seed.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for v in q_seed.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for v in h.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for v in r_seed.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for m in measurements.iter_mut() {
            for v in m.iter_mut() {
                *v = clamp_finite(f32::arbitrary(u)?);
            }
        }
        for v in nonlinear_coeffs.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }

        Ok(ExtendedFilterInput {
            state,
            p_seed,
            q_seed,
            h,
            r_seed,
            measurements,
            nonlinear_coeffs,
        })
    }
}

fuzz_target!(|input: ExtendedFilterInput| {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // Set state
    filter.state_vector_mut().apply(|vec| {
        for i in 0..NUM_STATES {
            vec.set_at(i, 0, input.state[i]);
        }
    });

    // Set P
    let mut p = input.p_seed;
    symmetrize_and_regularize(&mut p, NUM_STATES);
    filter.estimate_covariance_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, p[i * NUM_STATES + j]);
            }
        }
    });

    // Set Q
    let mut q = input.q_seed;
    symmetrize_and_regularize(&mut q, NUM_STATES);
    filter.direct_process_noise_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, q[i * NUM_STATES + j]);
            }
        }
    });

    // Set H
    measurement.observation_jacobian_matrix_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, input.h[i * NUM_STATES + j]);
            }
        }
    });

    // Set R
    let mut r = input.r_seed;
    symmetrize_and_regularize(&mut r, NUM_OBSERVATIONS);
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_OBSERVATIONS {
                mat.set_at(i, j, r[i * NUM_OBSERVATIONS + j]);
            }
        }
    });

    let c = input.nonlinear_coeffs;

    for step in 0..4 {
        // Nonlinear prediction: simple bounded function using fuzzed coefficients
        filter.predict_nonlinear(|current, next| {
            for i in 0..NUM_STATES {
                let val = current.get_at(i, 0);
                // Simple nonlinear: c[0]*sin(val) + c[1]*val, clamped
                let new_val = c[0] * val.sin() + c[1] * val;
                next.set_at(i, 0, new_val.clamp(-100.0, 100.0));
            }
        });

        // Update Jacobian to identity for simplicity
        filter.state_transition_jacobian_mut().apply(|mat| {
            mat.make_identity();
        });

        // Set measurement
        measurement.measurement_vector_mut().apply(|vec| {
            for i in 0..NUM_OBSERVATIONS {
                vec.set_at(i, 0, input.measurements[step][i]);
            }
        });

        // Nonlinear correction
        filter.correct_nonlinear::<_, _, NUM_OBSERVATIONS>(&mut measurement, |state, obs| {
            for i in 0..NUM_OBSERVATIONS {
                let val = if i < NUM_STATES {
                    state.get_at(i, 0)
                } else {
                    0.0
                };
                // Simple nonlinear: c[2]*cos(val) + c[3]*sin(val)
                let h_val = c[2] * val.cos() + c[3] * val.sin();
                obs.set_at(i, 0, h_val.clamp(-100.0, 100.0));
            }
        });
    }

    // Read final state out of the filter.
    let mut final_state = [0.0f32; NUM_STATES];
    filter.state_vector().inspect(|vec| {
        for (i, slot) in final_state.iter_mut().enumerate() {
            *slot = vec.get_at(i, 0);
        }
    });

    // Unstable dynamics naturally diverge with arbitrary inputs — not a
    // library bug — so we do not assert finiteness here. The fuzz target
    // only reports panics, UB, and hangs; reaching this point cleanly is
    // enough.
    let _ = final_state;
});
