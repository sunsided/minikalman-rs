//! EKF example for estimating the state of charge (SoC) of a battery cell.
//!
//! The cell is modelled as a first-order RC equivalent circuit: an open-circuit voltage source
//! `OCV(SoC)`, a series resistance `R0`, and one RC pair (`R1` ∥ `C1`) for the polarization.
//! The filter state is `x = [SoC, V_rc]`, the measured cell current `I` (positive when
//! discharging) is a known input, and only the terminal voltage `V` is observed.
//!
//! ## Model
//!
//! With the time step `Δt`, the capacity `Q` in ampere-seconds and `τ = R1·C1`:
//!
//! ```text
//! SoC'  = SoC − I·Δt/Q
//! V_rc' = V_rc·e^(−Δt/τ) + R1·(1 − e^(−Δt/τ))·I
//! V     = OCV(SoC) − V_rc − R0·I
//! ```
//!
//! The prediction is linear in the state, so its Jacobian is constant:
//! `F = [[1, 0], [0, e^(−Δt/τ)]]`. The observation is nonlinear through the OCV curve,
//! and its Jacobian is `H = [dOCV/dSoC, −1]`.
//!
//! The extended filter has no separate control step, so the current enters the nonlinear
//! state transition closure as a captured input.
//!
//! ## Simulation
//!
//! A true cell is discharged with a pulsed current. The filter sees the current and the
//! terminal voltage with sensor noise and starts from a wrong SoC. The example exits with a
//! non-zero status if the estimate has not converged to the true SoC by the end.

#![forbid(unsafe_code)]

use std::process::ExitCode;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use minikalman::extended::builder::{
    KalmanFilterBuilder, KalmanFilterObservationType, KalmanFilterType,
};
use minikalman::prelude::*;

/// Number of filter states: SoC (unitless, `0..=1`) and RC voltage (V).
const NUM_STATES: usize = 2;

/// Number of observations: the terminal voltage (V).
const NUM_OBSERVATIONS: usize = 1;

/// The extended filter over [`NUM_STATES`] states with owned buffers.
type Filter = KalmanFilterType<NUM_STATES, f32>;

/// The terminal voltage measurement with owned buffers.
type Measurement = KalmanFilterObservationType<NUM_STATES, NUM_OBSERVATIONS, f32>;

/// Simulation time step in seconds.
const DELTA_T: f32 = 1.0;

/// Number of simulated steps (30 minutes).
const NUM_STEPS: usize = 1800;

/// Print the estimate every this many steps once the filter has settled.
const PRINT_EVERY: usize = 120;

/// Print the estimate more often during the first this many steps to show convergence.
const SETTLING_STEPS: usize = 30;

/// Cell capacity in ampere-seconds (2.5 Ah).
const CAPACITY: f32 = 2.5 * 3600.0;

/// Series resistance in ohms.
const R0: f32 = 0.05;

/// Polarization resistance of the RC pair in ohms.
const R1: f32 = 0.03;

/// Polarization capacitance of the RC pair in farads.
const C1: f32 = 1000.0;

/// True SoC at the start of the simulation.
const TRUE_INITIAL_SOC: f32 = 0.9;

/// The filter's (deliberately wrong) initial SoC guess.
const ESTIMATED_INITIAL_SOC: f32 = 0.5;

/// Standard deviation of the current sensor noise in amperes.
const CURRENT_NOISE_STD: f32 = 0.05;

/// Standard deviation of the voltage sensor noise in volts.
const VOLTAGE_NOISE_STD: f32 = 0.01;

/// Largest final SoC error that still counts as converged.
const SOC_TOLERANCE: f32 = 0.02;

/// Fixed seed so CI runs are reproducible.
const SEED: u64 = 42;

/// Runs the simulation and checks that the filter converged.
fn main() -> ExitCode {
    let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    let mut truth = Cell::new(TRUE_INITIAL_SOC);
    let mut sensors = Sensors::new();

    initialize_filter(&mut filter);
    measurement.measurement_noise_covariance_mut().set_at(
        0,
        0,
        VOLTAGE_NOISE_STD * VOLTAGE_NOISE_STD,
    );

    for step in 1..=NUM_STEPS {
        let current = load_current(step);
        truth.step(current);

        let measured_current = sensors.current(current);
        let measured_voltage = sensors.voltage(truth.terminal_voltage(current));

        predict(&mut filter, measured_current);
        correct(
            &mut filter,
            &mut measurement,
            measured_current,
            measured_voltage,
        );

        if should_print(step) {
            print_state(step, &filter, &truth);
        }
    }

    check_convergence(&filter, &truth)
}

/// Returns whether to print the estimate at the given step.
///
/// Prints the first step, every 5 steps while settling, then every [`PRINT_EVERY`] steps.
fn should_print(step: usize) -> bool {
    step == 1 || (step <= SETTLING_STEPS && step % 5 == 0) || step % PRINT_EVERY == 0
}

/// The true cell state that the filter tries to estimate.
struct Cell {
    /// State of charge, unitless in `0..=1`.
    soc: f32,
    /// Voltage across the RC pair in volts.
    v_rc: f32,
}

impl Cell {
    /// Creates a relaxed cell (no polarization) at the given SoC.
    fn new(soc: f32) -> Self {
        Self { soc, v_rc: 0.0 }
    }

    /// Advances the cell by one time step with the given current in amperes.
    fn step(&mut self, current: f32) {
        let [soc, v_rc] = transition(self.soc, self.v_rc, current);
        self.soc = soc;
        self.v_rc = v_rc;
    }

    /// Returns the noise-free terminal voltage in volts for the given current.
    fn terminal_voltage(&self, current: f32) -> f32 {
        terminal_voltage(self.soc, self.v_rc, current)
    }
}

/// Noisy current and voltage sensors.
struct Sensors {
    /// Seeded random source for reproducible noise.
    rng: StdRng,
    /// Current sensor noise in amperes.
    current_noise: Normal<f32>,
    /// Voltage sensor noise in volts.
    voltage_noise: Normal<f32>,
}

impl Sensors {
    /// Creates the sensors with the configured noise levels and [`SEED`].
    fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(SEED),
            current_noise: Normal::new(0.0, CURRENT_NOISE_STD).expect("valid std"),
            voltage_noise: Normal::new(0.0, VOLTAGE_NOISE_STD).expect("valid std"),
        }
    }

    /// Returns a noisy reading of the true current in amperes.
    fn current(&mut self, current: f32) -> f32 {
        current + self.current_noise.sample(&mut self.rng)
    }

    /// Returns a noisy reading of the true terminal voltage in volts.
    fn voltage(&mut self, voltage: f32) -> f32 {
        voltage + self.voltage_noise.sample(&mut self.rng)
    }
}

/// Returns the discharge current in amperes at the given step.
///
/// Alternates one minute at 1C (2.5 A) with one minute at a light 0.2C load (0.5 A).
fn load_current(step: usize) -> f32 {
    if (step / 60) % 2 == 0 { 2.5 } else { 0.5 }
}

/// Returns the RC decay factor `e^(−Δt/τ)` for one time step.
fn rc_decay() -> f32 {
    (-DELTA_T / (R1 * C1)).exp()
}

/// Advances `[SoC, V_rc]` by one time step with the given current in amperes.
fn transition(soc: f32, v_rc: f32, current: f32) -> [f32; 2] {
    let decay = rc_decay();

    [
        soc - current * DELTA_T / CAPACITY,
        v_rc * decay + R1 * (1.0 - decay) * current,
    ]
}

/// Returns the terminal voltage in volts for the given state and current.
fn terminal_voltage(soc: f32, v_rc: f32, current: f32) -> f32 {
    ocv(soc) - v_rc - R0 * current
}

/// Open-circuit voltage in volts as a function of SoC.
///
/// A cubic fit that rises monotonically from 3.0 V at empty to 4.2 V at full, flatter in the
/// middle like a typical lithium-ion cell.
fn ocv(soc: f32) -> f32 {
    3.0 + 1.6 * soc - 1.6 * soc * soc + 1.2 * soc * soc * soc
}

/// Derivative of [`ocv`] with respect to SoC, in volts per unit SoC.
fn ocv_slope(soc: f32) -> f32 {
    1.6 - 3.2 * soc + 3.6 * soc * soc
}

/// Sets the wrong initial guess, its uncertainty, the process noise and the constant Jacobian.
fn initialize_filter(filter: &mut Filter) {
    filter.state_vector_mut().apply(|vec| {
        vec.set_row(0, ESTIMATED_INITIAL_SOC);
        vec.set_row(1, 0.0);
    });

    filter.estimate_covariance_mut().apply(|mat| {
        mat.set_at(0, 0, 0.3 * 0.3);
        mat.set_at(1, 1, 0.01 * 0.01);
    });

    filter.direct_process_noise_mut().apply(|mat| {
        mat.set_at(0, 0, 1e-8);
        mat.set_at(1, 1, 1e-6);
    });

    filter.state_transition_jacobian_mut().apply(|mat| {
        mat.make_identity();
        mat.set_at(1, 1, rc_decay());
    });
}

/// Predicts the next state from the measured current in amperes.
fn predict(filter: &mut Filter, current: f32) {
    filter.predict_nonlinear(|state, next| {
        let [soc, v_rc] = transition(state.get_row(0), state.get_row(1), current);
        next.set_row(0, soc);
        next.set_row(1, v_rc);
    });
}

/// Corrects the estimate with a terminal voltage reading in volts.
fn correct(filter: &mut Filter, measurement: &mut Measurement, current: f32, voltage: f32) {
    let soc = filter.state_vector().get_row(0);

    measurement.measurement_vector_mut().set_row(0, voltage);
    measurement.observation_jacobian_matrix_mut().apply(|mat| {
        mat.set_at(0, 0, ocv_slope(soc));
        mat.set_at(0, 1, -1.0);
    });

    filter.correct_nonlinear(measurement, |state, observation| {
        let voltage = terminal_voltage(state.get_row(0), state.get_row(1), current);
        observation.set_row(0, voltage);
    });
}

/// Prints the true and estimated SoC with the estimate's standard deviation.
fn print_state(step: usize, filter: &Filter, truth: &Cell) {
    let soc = filter.state_vector().get_row(0);
    let std_soc = filter.estimate_covariance().get_at(0, 0).sqrt();

    println!(
        "t={step:4} s  true SoC={:5.1}%  estimate={:5.1}% ± {:.2}%  error={:+.2}%",
        truth.soc * 100.0,
        soc * 100.0,
        std_soc * 100.0,
        (soc - truth.soc) * 100.0,
    );
}

/// Returns success if the final SoC estimate is within [`SOC_TOLERANCE`] of the truth.
fn check_convergence(filter: &Filter, truth: &Cell) -> ExitCode {
    let error = (filter.state_vector().get_row(0) - truth.soc).abs();

    if error <= SOC_TOLERANCE {
        println!("Converged: final SoC error {:.2}%", error * 100.0);
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "Did not converge: final SoC error {:.2}% exceeds {:.2}%",
            error * 100.0,
            SOC_TOLERANCE * 100.0
        );
        ExitCode::FAILURE
    }
}
