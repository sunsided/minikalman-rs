use crate::builder::KalmanFilterBuilder;
use crate::prelude::*;

pub struct TestFilter {
    pub filter: KalmanFilterType<3, f32>,
    pub control: KalmanFilterControlType<3, 1, f32>,
    pub measurement: KalmanFilterObservationType<3, 1, f32>,
}

#[allow(unused)]
pub fn create_test_filter(delta_t: f32) -> TestFilter {
    let builder = KalmanFilterBuilder::<3, f32>::default();
    let mut filter = builder.build();

    let controls = builder.controls();
    let mut control = controls.build::<1>();

    let measurements = builder.observations();
    let mut measurement = measurements.build::<1>();

    // Simple model of linear motion.
    filter.state_transition_apply(|mat| {
        mat.set(0, 0, 1.0);
        mat.set(0, 1, delta_t);
        mat.set(0, 2, 0.5 * delta_t * delta_t);

        mat.set(1, 0, 0.0);
        mat.set(1, 1, 1.0);
        mat.set(1, 2, delta_t);

        mat.set(2, 0, 0.0);
        mat.set(2, 1, 0.0);
        mat.set(2, 2, 1.0);
    });

    // No state estimate so far.
    filter.state_vector_mut().set_zero();

    // Estimate covariance is identity.
    filter.estimate_covariance_mut().make_scalar(0.1);

    // Control input directly affects the acceleration and, over the time step,
    // the velocity and position.
    control.control_matrix_apply(|mat| {
        mat.set(0, 0, 0.5 * delta_t * delta_t);
        mat.set(1, 0, delta_t);
        mat.set(2, 0, 1.0);
    });

    // No control inputs now.
    control.control_vector_mut().set_zero();

    // Process noise is identity.
    control.process_noise_covariance_mut().make_identity();

    // The measurement is an average of the states.
    measurement.observation_matrix_mut().set_all(1.0 / 3.0);

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
