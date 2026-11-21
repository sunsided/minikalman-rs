//! Estimates the gravity constant from position observations of a falling object, with
//! `I16F16` fixed-point numbers and all filter buffers in one `static`.

use lazy_static::lazy_static;
use minikalman::prelude::*;
use minikalman::regular::{FilterStorage, ObservationStorage};
use static_cell::ConstStaticCell;

minikalman::state! {
    /// The state of the falling object.
    struct Fall(I16F16) {
        /// Position, in meters.
        s,
        /// Velocity, in meters per second.
        v,
        /// Acceleration, in meters per second squared.
        g,
    }
}

/// The number of states.
const NUM_STATES: usize = Fall::N;

/// The number of observations: the position.
const NUM_OBSERVATIONS: usize = 1;

/// All buffers of the filter and its observation.
///
/// The measurement vector `z` is part of the observation storage, so it can no longer be
/// declared with the wrong buffer macro (this example used `impl_buffer_Q_control!` for it).
struct Storage {
    /// The filter buffers: `A`, `x`, `P`, `Q` and the temporaries.
    filter: FilterStorage<NUM_STATES, I16F16>,
    /// The observation buffers: `H`, `z`, `R`, `y`, `S`, `K` and the temporaries.
    measurement: ObservationStorage<NUM_STATES, NUM_OBSERVATIONS, I16F16>,
}

/// The filter's memory, fixed at compile time.
///
/// `I16F16::ZERO` is a constant with all bits zero, so the storage lands in `.bss` and is
/// never built on the stack or moved. [`ConstStaticCell::try_take`] hands out its only
/// `&'static mut`; a second or re-entrant call gets `None` instead of a second mutable alias.
static STORAGE: ConstStaticCell<Storage> = ConstStaticCell::new(Storage {
    filter: FilterStorage::new(I16F16::ZERO),
    measurement: ObservationStorage::new(I16F16::ZERO),
});

lazy_static! {

    /// Observations.
    ///
    /// MATLAB source:
    /// ```matlab
    /// s = s + v*T + g*0.5*T^2;
    /// v = v + g*T;
    /// ```
    static ref REAL_DISTANCE: [I16F16; 15] = [
        I16F16::from_num(0.0),
        I16F16::from_num(4.905),
        I16F16::from_num(19.62),
        I16F16::from_num(44.145),
        I16F16::from_num(78.48),
        I16F16::from_num(122.63),
        I16F16::from_num(176.58),
        I16F16::from_num(240.35),
        I16F16::from_num(313.92),
        I16F16::from_num(397.31),
        I16F16::from_num(490.5),
        I16F16::from_num(593.51),
        I16F16::from_num(706.32),
        I16F16::from_num(828.94),
        I16F16::from_num(961.38),
    ];

    /// Observation noise with variance 0.5
    ///
    /// MATLAB source:
    /// ```matlab
    /// noise = 0.5^2*randn(15,1);
    /// ```
    static ref OBSERVATION_ERROR: [I16F16; 15] = [
        I16F16::from_num(0.13442),
        I16F16::from_num(0.45847),
        I16F16::from_num(-0.56471),
        I16F16::from_num(0.21554),
        I16F16::from_num(0.079691),
        I16F16::from_num(-0.32692),
        I16F16::from_num(-0.1084),
        I16F16::from_num(0.085656),
        I16F16::from_num(0.8946),
        I16F16::from_num(0.69236),
        I16F16::from_num(-0.33747),
        I16F16::from_num(0.75873),
        I16F16::from_num(0.18135),
        I16F16::from_num(-0.015764),
        I16F16::from_num(0.17869),
    ];

}

/// Runs the filter over the recorded observations and returns the estimated gravity constant.
///
/// Returns `None` if the storage was already taken, i.e. on every call after the first.
pub fn predict_gravity() -> Option<I16F16> {
    let storage = STORAGE.try_take()?;
    let mut filter = storage.filter.build();
    let mut measurement = storage.measurement.build();

    // Set initial state.
    initialize_state_vector(filter.state_vector_mut());
    initialize_state_transition_matrix(filter.state_transition_mut());
    initialize_state_covariance_matrix(filter.estimate_covariance_mut());
    initialize_position_measurement_transformation_matrix(measurement.observation_matrix_mut());
    initialize_position_measurement_process_noise_matrix(
        measurement.measurement_noise_covariance_mut(),
    );

    // Filter!
    for (real, error) in REAL_DISTANCE.iter().zip(OBSERVATION_ERROR.iter()) {
        filter.predict();
        measurement
            .measurement_vector_mut()
            .apply(|z| z[0] = *real + *error);
        filter.correct(&mut measurement);
    }

    Some(Fall::read(filter.state_vector().as_slice()).g)
}

/// Initializes the state vector with initial assumptions.
fn initialize_state_vector(filter: &mut impl StateVectorMut<NUM_STATES, I16F16>) {
    filter.as_matrix_mut().apply(|state| {
        Fall {
            s: I16F16::ZERO,
            v: I16F16::ZERO,
            g: I16F16::from_num(6.0),
        }
        .write(state.as_mut_slice());
    });
}

/// Initializes the state transition matrix.
///
/// This sets up the differential equations for
/// ```math
/// s₁ = 1×s₀ + T×v₀ + 0.5×T²×g₀
/// v₁ = 1×v₀ + T×g₀
/// g₁ = 1×g₀
/// ```
fn initialize_state_transition_matrix(
    filter: &mut impl StateTransitionMatrixMut<NUM_STATES, I16F16>,
) {
    filter.as_matrix_mut().apply(|a| {
        // Time constant.
        const T: I16F16 = I16F16::ONE;

        a.make_identity();
        a.set_at(Fall::s, Fall::v, T);
        a.set_at(Fall::s, Fall::g, I16F16::from_num(0.5) * T * T);
        a.set_at(Fall::v, Fall::g, T);
    });
}

/// Initializes the system covariance matrix.
///
/// This defines how different states (linearly) influence each other
/// over time. In this setup we claim that position, velocity and acceleration
/// are linearly independent.
fn initialize_state_covariance_matrix(
    filter: &mut impl EstimateCovarianceMatrix<NUM_STATES, I16F16>,
) {
    filter.as_matrix_mut().apply(|p| {
        p.set_at(Fall::s, Fall::s, I16F16::from_num(0.1));
        p.set_at(Fall::v, Fall::v, I16F16::ONE);
        p.set_at(Fall::g, Fall::g, I16F16::ONE);
    });
}

/// Initializes the measurement transformation matrix.
///
/// This matrix describes how a single measurement is obtained from the
/// state vector. In our case, we directly observe one of the states, namely position.
/// ```math
/// z = 1×s + 0×v + 0×g
/// ```
fn initialize_position_measurement_transformation_matrix(
    measurement: &mut impl ObservationMatrixMut<NUM_OBSERVATIONS, NUM_STATES, I16F16>,
) {
    measurement
        .as_matrix_mut()
        .apply(|h| h.set_at(0, Fall::s, I16F16::ONE));
}

/// Initializes the measurement noise / uncertainty matrix.
///
/// This matrix describes the measurement covariances as well as the
/// individual variation components. It is the measurement counterpart
/// of the state covariance matrix.
fn initialize_position_measurement_process_noise_matrix(
    measurement: &mut impl MeasurementNoiseCovarianceMatrix<NUM_OBSERVATIONS, I16F16>,
) {
    measurement
        .as_matrix_mut()
        .apply(|r| r.set_at(0, 0, I16F16::from_num(0.5)));
}
