//! The reference scenarios, set up and run through each filter of the crate.
//!
//! Every runner takes its model and measurements from [`reference`](crate::reference), so the
//! crate sees exactly the inputs the independent implementation saw.

use minikalman::extended::builder::{
    KalmanFilterBuilder as ExtendedBuilder, KalmanFilterObservationType as ExtendedObservation,
    KalmanFilterType as Extended,
};
use minikalman::matrix::{MatrixData, MatrixDataBoxed};
use minikalman::prelude::*;
use minikalman::regular::builder::{
    KalmanFilterBuilder as RegularBuilder, KalmanFilterObservationType as RegularObservation,
    KalmanFilterType as Regular,
};
use minikalman::smoother::RtsSmoother;
use minikalman::unscented::builder::{
    KalmanFilterBuilder as UnscentedBuilder, KalmanFilterObservationType as UnscentedObservation,
    KalmanFilterType as Unscented,
};

use crate::reference::{cv, gravity, pendulum, radar};
use crate::support::{load, read, Real};

/// A linear Gaussian scenario: model matrices (row-major), initial estimate and measurements.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    /// State transition matrix `A`.
    pub a: &'static [f64],
    /// Process noise covariance `Q`.
    pub q: &'static [f64],
    /// Observation matrix `H`.
    pub h: &'static [f64],
    /// Measurement noise covariance `R`.
    pub r: &'static [f64],
    /// Initial state estimate `x₀`.
    pub x0: &'static [f64],
    /// Initial estimate covariance `P₀`.
    pub p0: &'static [f64],
    /// Measurements, `OBSERVATIONS` values per step.
    pub z: &'static [f64],
}

/// The constant-velocity scenario.
pub const CV: Linear = Linear {
    a: &cv::A,
    q: &cv::Q,
    h: &cv::H,
    r: &cv::R,
    x0: &cv::X0,
    p0: &cv::P0,
    z: &cv::Z,
};

/// The free-fall scenario.
pub const GRAVITY: Linear = Linear {
    a: &gravity::A,
    q: &gravity::Q,
    h: &gravity::H,
    r: &gravity::R,
    x0: &gravity::X0,
    p0: &gravity::P0,
    z: &gravity::Z,
};

/// The constant-velocity scenario without process noise, where the UKF is exact.
pub const CV_NOISELESS: Linear = Linear { q: &[0.0; 4], ..CV };

/// What a run recorded at every step, each flattened row-major and concatenated over steps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    /// Predicted states `x_k|k−1`.
    pub x_prior: Vec<f64>,
    /// Predicted covariances `P_k|k−1`.
    pub p_prior: Vec<f64>,
    /// Filtered states `x_k|k`.
    pub x_post: Vec<f64>,
    /// Filtered covariances `P_k|k`.
    pub p_post: Vec<f64>,
    /// Normalized innovation squared of each correction (empty for the information filter).
    pub nis: Vec<f64>,
}

impl History {
    /// The bit patterns of every recorded value, for exact determinism checks.
    pub fn bits(&self) -> Vec<u64> {
        [
            &self.x_prior,
            &self.p_prior,
            &self.x_post,
            &self.p_post,
            &self.nis,
        ]
        .iter()
        .flat_map(|v| v.iter().map(|x| x.to_bits()))
        .collect()
    }
}

/// Builds a regular filter and its observation for a linear scenario.
pub fn regular<const N: usize, const M: usize, T: Real>(
    s: &Linear,
) -> (Regular<N, T>, RegularObservation<N, M, T>) {
    let builder = RegularBuilder::<N, T>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<M>();

    filter.state_transition_mut().apply(|a| load(a, s.a));
    filter.direct_process_noise_mut().apply(|q| load(q, s.q));
    filter.state_vector_mut().apply(|x| load(x, s.x0));
    filter.estimate_covariance_mut().apply(|p| load(p, s.p0));
    measurement.observation_matrix_mut().apply(|h| load(h, s.h));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, s.r));

    (filter, measurement)
}

/// Runs the regular filter over a linear scenario, predicting then correcting each step.
pub fn run_regular<const N: usize, const M: usize, T: Real>(s: &Linear) -> History {
    let (mut filter, mut measurement) = regular::<N, M, T>(s);
    let mut history = History::default();
    for z in s.z.chunks_exact(M) {
        filter.predict();
        record_prior::<N, T>(
            &mut history,
            filter.state_vector(),
            filter.estimate_covariance(),
        );

        measurement.measurement_vector_mut().apply(|v| load(v, z));
        filter.correct(&mut measurement);
        record_post::<N, T>(
            &mut history,
            filter.state_vector(),
            filter.estimate_covariance(),
        );
        history.nis.push(measurement.nis().f64());
    }

    history
}

/// Runs the information filter over a linear scenario and records the recovered `x` and `P`.
///
/// # Panics
/// Panics if a step reports a matrix that is not positive definite.
pub fn run_information<const N: usize, const M: usize, T: Real>(s: &Linear) -> History {
    let builder = RegularBuilder::<N, T>::default();
    let mut info = builder.build_information();
    let mut measurement = builder.observations().build::<M>();
    let mut x = BufferBuilder::state_vector_x::<N>().new();
    let mut p = BufferBuilder::estimate_covariance_P::<N>().new();

    info.state_transition_mut().apply(|a| load(a, s.a));
    info.process_noise_mut().apply(|q| load(q, s.q));
    measurement.observation_matrix_mut().apply(|h| load(h, s.h));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, s.r));
    x.apply(|v| load(v, s.x0));
    p.apply(|m| load(m, s.p0));
    assert!(info.set_state(&x, &p), "P₀ not invertible");

    let mut history = History::default();
    for z in s.z.chunks_exact(M) {
        assert!(info.predict(), "predict: Y or Q not invertible");
        measurement.measurement_vector_mut().apply(|v| load(v, z));
        assert!(info.fuse(&mut measurement), "fuse: R not invertible");
        assert!(info.state(&mut x, &mut p), "state: Y not invertible");
        record_post::<N, T>(&mut history, &x, &p);
    }

    history
}

/// Smooths a regular-filter history with the RTS smoother; returns `(x_k|N, P_k|N)`.
///
/// # Panics
/// Panics if a predicted covariance could not be inverted.
pub fn smooth<const N: usize, T: Real>(s: &Linear, history: &History) -> (Vec<f64>, Vec<f64>) {
    let mut x = boxed::<N, 1, T>(&history.x_post);
    let mut p = boxed::<N, N, T>(&history.p_post);
    let x_prior = boxed::<N, 1, T>(&history.x_prior);
    let p_prior = boxed::<N, N, T>(&history.p_prior);
    let mut a = MatrixData::new::<N, N, T>(T::zero());
    load(&mut a, s.a);

    let mut smoother = RtsSmoother::new(
        MatrixData::new::<N, N, T>(T::zero()),
        MatrixData::new::<N, N, T>(T::zero()),
        MatrixData::new::<N, N, T>(T::zero()),
        MatrixData::new::<N, 1, T>(T::zero()),
    );
    assert!(
        smoother.smooth(&a, &mut x, &mut p, &x_prior, &p_prior),
        "a predicted covariance was not invertible"
    );

    (
        x.iter().flat_map(read::<N, 1, T>).collect(),
        p.iter().flat_map(read::<N, N, T>).collect(),
    )
}

/// Splits a flat per-step history into one owned `R × C` matrix per step.
fn boxed<const R: usize, const C: usize, T: Real>(flat: &[f64]) -> Vec<MatrixDataBoxed<R, C, T>> {
    flat.chunks_exact(R * C)
        .map(|values| {
            let mut m = MatrixData::new::<R, C, T>(T::zero());
            load(&mut m, values);
            m
        })
        .collect()
}

/// Builds the extended filter and range-bearing observation of the radar scenario.
pub fn extended_radar<T: Real>() -> (Extended<4, T>, ExtendedObservation<4, 2, T>) {
    let builder = ExtendedBuilder::<4, T>::default();
    let mut filter = builder.build();
    let mut measurement = builder.observations().build::<2>();

    filter
        .state_transition_jacobian_mut()
        .apply(|a| load(a, &radar::A));
    filter
        .direct_process_noise_mut()
        .apply(|q| load(q, &radar::Q));
    filter.state_vector_mut().apply(|x| load(x, &radar::X0));
    filter
        .estimate_covariance_mut()
        .apply(|p| load(p, &radar::P0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, &radar::R));

    (filter, measurement)
}

/// One radar step: [`radar_predict`], then [`radar_correct`] with `z`.
pub fn radar_step<T: Real>(
    filter: &mut Extended<4, T>,
    measurement: &mut ExtendedObservation<4, 2, T>,
    z: &[f64],
) -> CorrectionStatus {
    radar_predict(filter);
    radar_correct(filter, measurement, z)
}

/// Predicts the radar target through its linear motion.
pub fn radar_predict<T: Real>(filter: &mut Extended<4, T>) {
    filter.predict_nonlinear(|state, next| {
        for i in 0..4 {
            next[i] = (0..4).fold(T::zero(), |sum, j| {
                sum + T::of(radar::A[i * 4 + j]) * state[j]
            });
        }
    });
}

/// Linearizes the range-bearing observation at the current state and corrects with `z`.
pub fn radar_correct<T: Real>(
    filter: &mut Extended<4, T>,
    measurement: &mut ExtendedObservation<4, 2, T>,
    z: &[f64],
) -> CorrectionStatus {
    let prior = read(filter.state_vector());
    measurement
        .observation_jacobian_matrix_mut()
        .apply(|h| load(h, &radar_jacobian(&prior)));
    measurement.measurement_vector_mut().apply(|v| load(v, z));
    filter.correct_nonlinear(measurement, |state, observation| {
        let [range, bearing] = range_bearing(&[state.get_row(0).f64(), state.get_row(1).f64()]);
        observation.set_row(0, T::of(range));
        observation.set_row(1, T::of(bearing));
    })
}

/// Runs the extended filter over the radar scenario.
pub fn run_extended_radar<T: Real>() -> History {
    let (mut filter, mut measurement) = extended_radar::<T>();
    let mut history = History::default();
    for z in radar::Z.chunks_exact(2) {
        radar_step(&mut filter, &mut measurement, z);
        record_post::<4, T>(
            &mut history,
            filter.state_vector(),
            filter.estimate_covariance(),
        );
        history.nis.push(measurement.nis().f64());
    }

    history
}

/// Range and bearing of a position `[x, y]` as seen from [`radar::RADAR`].
pub fn range_bearing(position: &[f64]) -> [f64; 2] {
    let dx = position[0] - radar::RADAR[0];
    let dy = position[1] - radar::RADAR[1];
    [dx.hypot(dy), dy.atan2(dx)]
}

/// Jacobian of [`range_bearing`] with respect to `[px, py, vx, vy]`, row-major.
fn radar_jacobian(state: &[f64]) -> [f64; 8] {
    let dx = state[0] - radar::RADAR[0];
    let dy = state[1] - radar::RADAR[1];
    let r2 = dx * dx + dy * dy;
    let r = r2.sqrt();
    [dx / r, dy / r, 0.0, 0.0, -dy / r2, dx / r2, 0.0, 0.0]
}

/// Builds the unscented filter and observation of the pendulum scenario.
pub fn unscented_pendulum<T: Real>() -> (Unscented<2, 5, T>, UnscentedObservation<2, 1, 5, T>) {
    let builder = UnscentedBuilder::<2, T>::default();
    let mut filter = builder.build::<5>();
    let mut measurement = builder.observations().build::<1, 5>();

    filter.set_alpha(T::of(pendulum::ALPHA));
    filter.set_beta(T::of(pendulum::BETA));
    filter.set_kappa(T::of(pendulum::KAPPA));
    filter
        .direct_process_noise_mut()
        .apply(|q| load(q, &pendulum::Q));
    filter.state_vector_mut().apply(|x| load(x, &pendulum::X0));
    filter
        .estimate_covariance_mut()
        .apply(|p| load(p, &pendulum::P0));
    measurement
        .measurement_noise_covariance_mut()
        .apply(|r| load(r, &pendulum::R));

    (filter, measurement)
}

/// One pendulum step: [`pendulum_predict`], then [`pendulum_correct`] with `z`.
pub fn pendulum_step<T: Real>(
    filter: &mut Unscented<2, 5, T>,
    measurement: &mut UnscentedObservation<2, 1, 5, T>,
    z: &[f64],
) -> CorrectionStatus {
    pendulum_predict(filter);
    pendulum_correct(filter, measurement, z)
}

/// Propagates the sigma points through [`pendulum_transition`].
pub fn pendulum_predict<T: Real>(filter: &mut Unscented<2, 5, T>) {
    filter.predict_nonlinear(|next| {
        let state = [
            next.as_matrix().get(0, 0).f64(),
            next.as_matrix().get(1, 0).f64(),
        ];
        let [angle, rate] = pendulum_transition(state);
        next.as_matrix_mut().set(0, 0, T::of(angle));
        next.as_matrix_mut().set(1, 0, T::of(rate));
    });
}

/// Corrects on the bob's horizontal position `z`.
pub fn pendulum_correct<T: Real>(
    filter: &mut Unscented<2, 5, T>,
    measurement: &mut UnscentedObservation<2, 1, 5, T>,
    z: &[f64],
) -> CorrectionStatus {
    measurement.measurement_vector_mut().apply(|v| load(v, z));
    filter.correct_sigma_point(measurement, |sigma, observed| {
        for j in 0..5 {
            // Row-major, so entry `j` is the angle (row 0) of sigma point `j`.
            let angle = sigma[j].f64();
            observed.set(0, j, T::of(pendulum::LENGTH * angle.sin()));
        }
    })
}

/// One explicit Euler step of the frictionless pendulum `[angle, rate]`.
pub fn pendulum_transition([angle, rate]: [f64; 2]) -> [f64; 2] {
    [
        angle + rate * pendulum::DT,
        rate - pendulum::G_OVER_L * angle.sin() * pendulum::DT,
    ]
}

/// Runs the unscented filter over the pendulum scenario.
pub fn run_unscented_pendulum<T: Real>() -> History {
    let (mut filter, mut measurement) = unscented_pendulum::<T>();
    let mut history = History::default();
    for z in pendulum::Z.iter() {
        pendulum_step(&mut filter, &mut measurement, core::slice::from_ref(z));
        record_post::<2, T>(
            &mut history,
            filter.state_vector(),
            filter.estimate_covariance(),
        );
        history.nis.push(measurement.nis().f64());
    }

    history
}

/// Records the predicted state and covariance.
fn record_prior<const N: usize, T: Real>(
    history: &mut History,
    x: &impl Matrix<N, 1, T>,
    p: &impl Matrix<N, N, T>,
) {
    history.x_prior.extend(read(x));
    history.p_prior.extend(read(p));
}

/// Records the filtered state and covariance.
fn record_post<const N: usize, T: Real>(
    history: &mut History,
    x: &impl Matrix<N, 1, T>,
    p: &impl Matrix<N, N, T>,
) {
    history.x_post.extend(read(x));
    history.p_post.extend(read(p));
}
