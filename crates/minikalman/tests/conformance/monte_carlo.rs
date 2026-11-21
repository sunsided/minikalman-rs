//! Monte Carlo runs of the scenarios against simulated truth, in `f64`.
//!
//! Each run draws its own initial truth from `N(x₀, P₀)`, process noise from the filter's own
//! `Q` and measurement noise from its `R`, so a correct filter is consistent by construction.
//! The run count and step count come from [`chi2`](crate::reference::chi2), where the
//! generator also put the matching chi-square bounds.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use minikalman::prelude::*;

use crate::reference::{chi2, cv, pendulum, radar};
use crate::scenarios::{self, CV};
use crate::support::{cholesky, load, mahalanobis, read};

/// Seed used when `MINIKALMAN_CONFORMANCE_SEED` is not set; the baselines were taken with it.
pub const DEFAULT_SEED: u64 = 0x6b61_6c6d_616e;

/// The seed of this test run: `MINIKALMAN_CONFORMANCE_SEED` if set, else [`DEFAULT_SEED`].
///
/// # Panics
/// Panics if the variable is set but not an unsigned integer.
pub fn seed() -> u64 {
    std::env::var("MINIKALMAN_CONFORMANCE_SEED").map_or(DEFAULT_SEED, |s| {
        s.parse().expect("MINIKALMAN_CONFORMANCE_SEED is a u64")
    })
}

/// Accuracy and consistency of a filter over all runs.
#[derive(Clone, Debug)]
pub struct Metrics {
    /// NEES averaged over the runs, one entry per step.
    pub nees: Vec<f64>,
    /// NEES averaged over all runs and steps.
    pub anees: f64,
    /// NIS averaged over all runs and steps.
    pub nis: f64,
    /// Root mean squared error of the position states over all runs and steps.
    pub rmse: f64,
}

/// Sums the per-step errors of all runs.
struct Accumulator {
    /// Sum of the NEES over runs, per step.
    nees: Vec<f64>,
    /// Sum of the NIS over all runs and steps.
    nis: f64,
    /// Sum of the squared position errors over all runs and steps.
    squared_error: f64,
    /// Indices of the states that count as position for the RMSE.
    position: &'static [usize],
}

impl Accumulator {
    /// Starts an empty accumulator for [`chi2::STEPS`] steps.
    fn new(position: &'static [usize]) -> Self {
        Self {
            nees: vec![0.0; chi2::STEPS],
            nis: 0.0,
            squared_error: 0.0,
            position,
        }
    }

    /// Records step `k` of a run: the truth, the estimate, its covariance and the NIS.
    fn record(&mut self, k: usize, truth: &[f64], x: &[f64], p: &[f64], nis: f64) {
        let error: Vec<f64> = truth.iter().zip(x).map(|(t, e)| t - e).collect();
        self.nees[k] += mahalanobis(p, &error);
        self.nis += nis;
        self.squared_error += self
            .position
            .iter()
            .map(|&i| error[i] * error[i])
            .sum::<f64>();
    }

    /// Averages the sums into [`Metrics`].
    fn finish(self) -> Metrics {
        let runs = chi2::RUNS as f64;
        let samples = runs * chi2::STEPS as f64;
        let nees: Vec<f64> = self.nees.iter().map(|s| s / runs).collect();

        Metrics {
            anees: nees.iter().sum::<f64>() / chi2::STEPS as f64,
            nees,
            nis: self.nis / samples,
            rmse: (self.squared_error / (samples * self.position.len() as f64)).sqrt(),
        }
    }
}

/// A seeded source of correlated Gaussian vectors.
struct Gaussian(StdRng);

impl Gaussian {
    /// The generator of run `run` under `seed`; every run gets its own stream.
    fn new(seed: u64, run: usize) -> Self {
        Self(StdRng::seed_from_u64(seed ^ ((run as u64) << 40)))
    }

    /// Draws from `N(mean, cov)` for a positive definite row-major `cov`.
    fn draw(&mut self, mean: &[f64], cov: &[f64]) -> Vec<f64> {
        let n = mean.len();
        let l = cholesky(cov, n).expect("positive definite covariance");
        let white: Vec<f64> = (0..n).map(|_| StandardNormal.sample(&mut self.0)).collect();
        (0..n)
            .map(|i| mean[i] + (0..=i).map(|j| l[i * n + j] * white[j]).sum::<f64>())
            .collect()
    }

    /// Draws zero-mean noise with covariance `cov`.
    fn noise(&mut self, cov: &[f64], n: usize) -> Vec<f64> {
        self.draw(&vec![0.0; n], cov)
    }
}

/// Asserts that a correction was applied, so a skipped one cannot pass the statistics.
///
/// # Panics
/// Panics, naming the run and step, for any other [`CorrectionStatus`].
fn assert_applied(status: CorrectionStatus, run: usize, k: usize) {
    assert_eq!(status, CorrectionStatus::Applied, "run {run}, step {k}");
}

/// Row-major `n × m` matrix times vector.
fn mul(a: &[f64], x: &[f64]) -> Vec<f64> {
    a.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum())
        .collect()
}

/// Element-wise sum of two vectors.
fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

/// The regular filter on the constant-velocity model.
pub fn regular_cv(seed: u64) -> Metrics {
    let mut acc = Accumulator::new(&[0]);
    for run in 0..chi2::RUNS {
        let mut rng = Gaussian::new(seed, run);
        let (mut filter, mut measurement) = scenarios::regular::<2, 1, f64>(&CV);
        let mut truth = rng.draw(&cv::X0, &cv::P0);
        for k in 0..chi2::STEPS {
            truth = add(&mul(&cv::A, &truth), &rng.noise(&cv::Q, 2));
            let z = add(&mul(&cv::H, &truth), &rng.noise(&cv::R, 1));

            filter.predict();
            measurement.measurement_vector_mut().apply(|v| load(v, &z));
            assert_applied(filter.correct(&mut measurement), run, k);
            let x = read(filter.state_vector());
            let p = read(filter.estimate_covariance());
            acc.record(k, &truth, &x, &p, measurement.nis());
        }
    }

    acc.finish()
}

/// The extended filter on the radar model.
pub fn extended_radar(seed: u64) -> Metrics {
    let mut acc = Accumulator::new(&[0, 1]);
    for run in 0..chi2::RUNS {
        let mut rng = Gaussian::new(seed, run);
        let (mut filter, mut measurement) = scenarios::extended_radar::<f64>();
        let mut truth = rng.draw(&radar::X0, &radar::P0);
        for k in 0..chi2::STEPS {
            truth = add(&mul(&radar::A, &truth), &rng.noise(&radar::Q, 4));
            let z = add(&scenarios::range_bearing(&truth), &rng.noise(&radar::R, 2));

            assert_applied(
                scenarios::radar_step(&mut filter, &mut measurement, &z),
                run,
                k,
            );
            let x = read(filter.state_vector());
            let p = read(filter.estimate_covariance());
            acc.record(k, &truth, &x, &p, measurement.nis());
        }
    }

    acc.finish()
}

/// The unscented filter on the pendulum model.
pub fn unscented_pendulum(seed: u64) -> Metrics {
    let mut acc = Accumulator::new(&[0]);
    for run in 0..chi2::RUNS {
        let mut rng = Gaussian::new(seed, run);
        let (mut filter, mut measurement) = scenarios::unscented_pendulum::<f64>();
        let mut truth = rng.draw(&pendulum::X0, &pendulum::P0);
        for k in 0..chi2::STEPS {
            let next = scenarios::pendulum_transition([truth[0], truth[1]]);
            truth = add(&next, &rng.noise(&pendulum::Q, 2));
            let bob = [pendulum::LENGTH * truth[0].sin()];
            let z = add(&bob, &rng.noise(&pendulum::R, 1));

            assert_applied(
                scenarios::pendulum_step(&mut filter, &mut measurement, &z),
                run,
                k,
            );
            let x = read(filter.state_vector());
            let p = read(filter.estimate_covariance());
            acc.record(k, &truth, &x, &p, measurement.nis());
        }
    }

    acc.finish()
}
