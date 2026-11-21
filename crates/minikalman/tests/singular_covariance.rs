//! Filters must stay finite when a covariance cannot be decomposed, and reject
//! a sigma point count that does not match the number of states.

#![cfg(feature = "std")]
#![forbid(unsafe_code)]

use minikalman::extended::builder::KalmanFilterBuilder as ExtendedBuilder;
use minikalman::prelude::*;
use minikalman::regular::builder::KalmanFilterBuilder as RegularBuilder;
use minikalman::unscented::builder::KalmanFilterBuilder as UnscentedBuilder;
use proptest::prelude::*;

const NUM_STATES: usize = 3;
const NUM_OBSERVATIONS: usize = 2;
const NUM_SIGMA: usize = 2 * NUM_STATES + 1;

/// Reads the state vector and the estimate covariance of any of the filters.
macro_rules! snapshot {
    ($filter:expr) => {{
        let mut x = [0.0f32; NUM_STATES];
        let mut p = [0.0f32; NUM_STATES * NUM_STATES];
        $filter.state_vector().inspect(|v| {
            for (i, slot) in x.iter_mut().enumerate() {
                *slot = v.get_at(i, 0);
            }
        });
        $filter.estimate_covariance().inspect(|m| {
            for (k, slot) in p.iter_mut().enumerate() {
                *slot = m.get_at(k / NUM_STATES, k % NUM_STATES);
            }
        });
        (x, p)
    }};
}

proptest! {
    /// With P = 0 and R = 0 the innovation covariance S is exactly zero. The update
    /// must then leave the state untouched instead of turning it into NaN.
    #[test]
    fn regular_update_with_zero_innovation_covariance_keeps_state(
        x in prop::array::uniform3(-100.0f32..100.0),
        h in prop::array::uniform6(-10.0f32..10.0),
        z in prop::array::uniform2(-100.0f32..100.0),
    ) {
        let builder = RegularBuilder::<NUM_STATES, f32>::default();
        let mut filter = builder.build();
        let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
        filter.state_vector_mut().apply(|v| {
            for (i, &value) in x.iter().enumerate() {
                v.set_at(i, 0, value);
            }
        });
        measurement.observation_matrix_mut().apply(|m| {
            for (k, &value) in h.iter().enumerate() {
                m.set_at(k / NUM_STATES, k % NUM_STATES, value);
            }
        });
        measurement.measurement_vector_mut().apply(|v| {
            v.set_at(0, 0, z[0]);
            v.set_at(1, 0, z[1]);
        });

        filter.correct(&mut measurement);

        let (after, p) = snapshot!(filter);
        prop_assert_eq!(after, x);
        prop_assert!(p.iter().all(|&v| v == 0.0));
    }

    /// A positive semi-definite P (some variances zero) cannot be Cholesky-decomposed.
    /// Sigma points then fall back to the diagonal; an identity prediction keeps the
    /// variances and stays finite.
    #[test]
    fn unscented_predict_with_semidefinite_covariance_stays_finite(
        x in prop::array::uniform3(-100.0f32..100.0),
        variances in prop::array::uniform3(prop_oneof![Just(0.0f32), 0.1f32..10.0]),
    ) {
        let builder = UnscentedBuilder::<NUM_STATES, f32>::default();
        let mut filter = builder.build::<NUM_SIGMA>();
        filter.state_vector_mut().apply(|v| {
            for (i, &value) in x.iter().enumerate() {
                v.set_at(i, 0, value);
            }
        });
        filter.estimate_covariance_mut().apply(|m| {
            for (i, &value) in variances.iter().enumerate() {
                m.set_at(i, i, value);
            }
        });

        filter.predict_nonlinear(|_| {});

        let (after, p) = snapshot!(filter);
        prop_assert!(after.iter().chain(p.iter()).all(|v| v.is_finite()));
        for i in 0..NUM_STATES {
            prop_assert!((after[i] - x[i]).abs() <= 1e-3 * (1.0 + x[i].abs()), "x[{}] = {}", i, after[i]);
            prop_assert!((p[i * NUM_STATES + i] - variances[i]).abs() <= 1e-3 * (1.0 + variances[i]), "P[{i}][{i}] = {}", p[i * NUM_STATES + i]);
        }
    }
}

#[test]
fn extended_update_with_zero_innovation_covariance_keeps_state() {
    let builder = ExtendedBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    filter.state_vector_mut().set_at(0, 0, 5.0);
    measurement
        .observation_jacobian_matrix_mut()
        .set_at(0, 0, 1.0);
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);

    filter.correct_nonlinear(&mut measurement, |state, observation| {
        observation.set_at(0, 0, state.get_at(0, 0));
    });

    let (x, p) = snapshot!(filter);
    assert_eq!(x, [5.0, 0.0, 0.0]);
    assert!(p.iter().all(|&v| v == 0.0));
}

#[test]
fn unscented_update_with_zero_innovation_covariance_keeps_state() {
    let builder = UnscentedBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build::<NUM_SIGMA>();
    let mut measurement = builder
        .observations()
        .build::<NUM_OBSERVATIONS, NUM_SIGMA>();
    filter.state_vector_mut().set_at(0, 0, 5.0);
    measurement.measurement_vector_mut().set_at(0, 0, 1.0);

    // P = 0 and R = 0: all sigma points coincide, so S = 0.
    filter.predict_nonlinear(|_| {});
    filter.correct_sigma_point(&mut measurement, |sigma, observed| {
        for j in 0..NUM_SIGMA {
            observed.set_at(0, j, sigma.get_at(0, j));
            observed.set_at(1, j, sigma.get_at(1, j));
        }
    });

    let (x, p) = snapshot!(filter);
    assert_eq!(x, [5.0, 0.0, 0.0]);
    assert!(p.iter().all(|v| v.is_finite()));
}

#[test]
#[cfg(not(feature = "no_assert"))]
#[should_panic(expected = "NUM_SIGMA must be 2 × STATES + 1")]
fn unscented_filter_rejects_wrong_sigma_count() {
    let builder = UnscentedBuilder::<NUM_STATES, f32>::default();
    let _ = builder.build::<9>();
}

#[test]
#[cfg(not(feature = "no_assert"))]
#[should_panic(expected = "NUM_SIGMA must be 2 × STATES + 1")]
fn unscented_observation_rejects_wrong_sigma_count() {
    let builder = UnscentedBuilder::<NUM_STATES, f32>::default();
    let _ = builder.observations().build::<NUM_OBSERVATIONS, 5>();
}
