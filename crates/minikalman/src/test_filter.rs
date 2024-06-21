use crate::builder::KalmanFilterBuilder;
use crate::prelude::*;

pub struct TestFilter {
    pub filter: KalmanFilterType<3, f32>,
    pub control: KalmanFilterControlType<3, 1, f32>,
    pub measurement: KalmanFilterObservationType<3, 2, f32>,
}

#[allow(unused)]
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
