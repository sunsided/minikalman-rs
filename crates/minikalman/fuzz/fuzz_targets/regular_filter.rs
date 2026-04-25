#![no_main]

use libfuzzer_sys::fuzz_target;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder;

const NUM_STATES: usize = 3;
const NUM_CONTROLS: usize = 1;
const NUM_OBSERVATIONS: usize = 2;

#[derive(Debug)]
struct FilterInput {
    state: [f32; NUM_STATES],
    p_seed: [f32; NUM_STATES * NUM_STATES],
    a: [f32; NUM_STATES * NUM_STATES],
    q_seed: [f32; NUM_STATES * NUM_STATES],
    h: [f32; NUM_OBSERVATIONS * NUM_STATES],
    r_seed: [f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS],
    measurements: [[f32; NUM_OBSERVATIONS]; 4],
    controls: [f32; 4],
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

impl arbitrary::Arbitrary<'_> for FilterInput {
    fn arbitrary(u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        let mut state = [0.0f32; NUM_STATES];
        let mut p_seed = [0.0f32; NUM_STATES * NUM_STATES];
        let mut a = [0.0f32; NUM_STATES * NUM_STATES];
        let mut q_seed = [0.0f32; NUM_STATES * NUM_STATES];
        let mut h = [0.0f32; NUM_OBSERVATIONS * NUM_STATES];
        let mut r_seed = [0.0f32; NUM_OBSERVATIONS * NUM_OBSERVATIONS];
        let mut measurements = [[0.0f32; NUM_OBSERVATIONS]; 4];
        let mut controls = [0.0f32; 4];

        for v in state.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for v in p_seed.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }
        for v in a.iter_mut() {
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
        for v in controls.iter_mut() {
            *v = clamp_finite(f32::arbitrary(u)?);
        }

        Ok(FilterInput {
            state,
            p_seed,
            a,
            q_seed,
            h,
            r_seed,
            measurements,
            controls,
        })
    }
}

fuzz_target!(|input: FilterInput| {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut control = builder.controls().build::<NUM_CONTROLS>();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();

    // Set state
    filter.state_vector_mut().apply(|vec| {
        for i in 0..NUM_STATES {
            vec.set_at(i, 0, input.state[i]);
        }
    });

    // Set A
    let a = input.a;
    filter.state_transition_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, a[i * NUM_STATES + j]);
            }
        }
    });

    // Set P (symmetrized + regularized)
    let mut p = input.p_seed;
    symmetrize_and_regularize(&mut p, NUM_STATES);
    filter.estimate_covariance_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, p[i * NUM_STATES + j]);
            }
        }
    });

    // Set Q (symmetrized + regularized)
    let mut q = input.q_seed;
    symmetrize_and_regularize(&mut q, NUM_STATES);
    filter.direct_process_noise_mut().apply(|mat| {
        for i in 0..NUM_STATES {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, q[i * NUM_STATES + j]);
            }
        }
    });

    // Set control
    control
        .control_matrix_mut()
        .apply(|mat| mat.set_at(0, 0, 1.0));
    control.control_vector_mut().set_at(0, 0, input.controls[0]);
    control
        .process_noise_covariance_mut()
        .apply(|mat| mat.make_scalar(0.01));

    // Set H
    measurement.observation_matrix_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_STATES {
                mat.set_at(i, j, input.h[i * NUM_STATES + j]);
            }
        }
    });

    // Set R (symmetrized + regularized)
    let mut r = input.r_seed;
    symmetrize_and_regularize(&mut r, NUM_OBSERVATIONS);
    measurement.measurement_noise_covariance_mut().apply(|mat| {
        for i in 0..NUM_OBSERVATIONS {
            for j in 0..NUM_OBSERVATIONS {
                mat.set_at(i, j, r[i * NUM_OBSERVATIONS + j]);
            }
        }
    });

    // Run filter steps
    for step in 0..4 {
        filter.predict();
        control
            .control_vector_mut()
            .set_at(0, 0, input.controls[step]);
        filter.control(&mut control);

        measurement.measurement_vector_mut().apply(|vec| {
            for i in 0..NUM_OBSERVATIONS {
                vec.set_at(i, 0, input.measurements[step][i]);
            }
        });
        filter.correct(&mut measurement);
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
