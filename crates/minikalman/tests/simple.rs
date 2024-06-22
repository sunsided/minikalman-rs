//! Tracking a free-falling object.
//!
//! This example sets up a filter to estimate the acceleration of a free-falling object
//! under earth conditions (i.e. a ≈ 9.807 m/s²) through position observations only.

#![forbid(unsafe_code)]

use assert_float_eq::*;

use minikalman::builder::regular::*;
use minikalman::prelude::*;

pub struct TestFilter {
    pub filter: KalmanFilterType<3, f32>,
    pub control: KalmanFilterControlType<3, 1, f32>,
    pub measurement: KalmanFilterObservationType<3, 2, f32>,
}

pub fn create_test_filter(delta_t: f32) -> TestFilter {
    let builder = KalmanFilterBuilder::<3, f32>::default();
    let mut filter = builder.build();

    let controls = builder.controls();
    let mut control = controls.build::<1>();

    let measurements = builder.observations();
    let mut measurement = measurements.build::<2>();

    // Simple model of linear motion.
    filter.state_transition_mut().apply(|mat| {
        mat.set_at(0, 0, 1.0);
        mat.set_at(0, 1, delta_t);
        mat.set_at(0, 2, 0.5 * delta_t * delta_t);

        mat.set_at(1, 0, 0.0);
        mat.set_at(1, 1, 1.0);
        mat.set_at(1, 2, delta_t);

        mat.set_at(2, 0, 0.0);
        mat.set_at(2, 1, 0.0);
        mat.set_at(2, 2, 1.0);
    });

    // No state estimate so far.
    filter.state_vector_mut().set_zero();

    // Estimate covariance is identity.
    filter.estimate_covariance_mut().make_scalar(0.1);

    // Control input directly affects the acceleration and, over the time step,
    // the velocity and position.
    control.control_matrix_mut().apply(|mat| {
        mat.set_at(0, 0, 0.5 * delta_t * delta_t);
        mat.set_at(1, 0, delta_t);
        mat.set_at(2, 0, 1.0);
    });

    // No control inputs now.
    control.control_vector_mut().set_zero();

    // Process noise is almost-identity.
    control
        .process_noise_covariance_mut()
        .make_comatrix(1.0, 0.1);

    // The measurement is both directly observing position and an average of the states.
    measurement.observation_matrix_mut().set_all(1.0 / 3.0);
    measurement.observation_matrix_mut().apply(|mat| {
        mat.set_at(0, 0, 1.0); // measure position directly
        mat.set_at(0, 1, 0.0);
        mat.set_at(0, 2, 0.0);
    });

    // Measurement noise covariance is identity.
    measurement
        .measurement_noise_covariance_mut()
        .make_identity();

    TestFilter {
        filter,
        control,
        measurement,
    }
}

#[test]
fn simple_filter() {
    let mut example = create_test_filter(1.0);

    // The estimate covariance still is scalar.
    assert!(example
        .filter
        .estimate_covariance()
        .inspect(|mat| (0..3).into_iter().all(|i| { mat.get_at(i, i) == 0.1 })));

    // Since our initial state is zero, any number of prediction steps keeps the filter unchanged.
    for _ in 0..10 {
        example.filter.predict();
    }

    // All states are zero.
    assert!(example
        .filter
        .state_vector()
        .as_ref()
        .iter()
        .all(|&x| x == 0.0));

    // The estimate covariance has changed.
    example.filter.estimate_covariance().inspect(|mat| {
        assert_f32_near!(mat.get_at(0, 0), 260.1);
        assert_f32_near!(mat.get_at(1, 1), 10.1);
        assert_f32_near!(mat.get_at(2, 2), 0.1);
    });

    // The measurement is zero.
    example
        .measurement
        .measurement_vector_mut()
        .set_at(0, 0, 0.0);

    // Apply a measurement of the unchanged state.
    example.filter.correct(&mut example.measurement);

    // All states are still zero.
    assert!(example
        .filter
        .state_vector()
        .as_ref()
        .iter()
        .all(|&x| x == 0.0));

    // The estimate covariance has improved.
    example.filter.estimate_covariance().inspect(|mat| {
        assert!(mat.get_at(0, 0) < 1.0);
        assert!(mat.get_at(1, 1) < 0.2);
        assert!(mat.get_at(2, 2) < 0.01);
    });

    // Set an input.
    example.control.control_vector_mut().set_at(0, 0, 1.0);

    // Predict and apply an input.
    example.filter.predict();
    example.filter.control(&mut example.control);

    // All states are still zero.
    example.filter.state_vector().inspect(|vec| {
        assert_eq!(
            vec.get_at(0, 0),
            0.5,
            "incorrect position after control input"
        );
        assert_eq!(
            vec.get_at(1, 0),
            1.0,
            "incorrect velocity after control input"
        );
        assert_eq!(
            vec.get_at(2, 0),
            1.0,
            "incorrect acceleration after control input"
        );
    });

    // Predict without input.
    example.filter.predict();

    // All states are still zero.
    example.filter.state_vector().inspect(|vec| {
        assert_eq!(vec.get_at(0, 0), 2.0, "incorrect position");
        assert_eq!(vec.get_at(1, 0), 2.0, "incorrect velocity");
        assert_eq!(vec.get_at(2, 0), 1.0, "incorrect acceleration");
    });

    // The estimate covariance has worsened.
    example.filter.estimate_covariance().inspect(|mat| {
        assert!(mat.get_at(0, 0) > 6.2);
        assert!(mat.get_at(1, 1) > 4.2);
        assert!(mat.get_at(2, 2) > 1.0);
    });

    // Set a new measurement
    example.measurement.measurement_vector_mut().apply(|vec| {
        vec.set_at(0, 0, 2.0);
        vec.set_at(1, 0, (2.0 + 2.0 + 1.0) / 3.0);
    });

    // Apply a measurement of the state.
    example.filter.correct(&mut example.measurement);

    // The estimate covariance has improved.
    example.filter.estimate_covariance().inspect(|mat| {
        assert!(mat.get_at(0, 0) < 1.0);
        assert!(mat.get_at(1, 1) < 1.0);
        assert!(mat.get_at(2, 2) < 0.4);
    });
}
