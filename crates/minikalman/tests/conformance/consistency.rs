//! Statistical consistency: NEES and NIS over Monte Carlo runs inside chi-square bounds.
//!
//! For a consistent filter the estimation error `e = x − x̂` has covariance `P`, so the NEES
//! `eᵀP⁻¹e` of `n` states is `χ²(n)` and its average over `R` independent runs at one step is
//! `χ²(R·n)/R`. Innovations are white, so the NIS of `m` measurements averaged over all `R`
//! runs and `K` steps is `χ²(R·K·m)/(R·K)`. NEES is correlated over time, so it is tested per
//! step at the steps in [`chi2::NEES_STEPS`], Bonferroni-corrected.
//!
//! Every test fails a correct filter with probability [`chi2::ALPHA`] for a random seed. The
//! seed is fixed ([`DEFAULT_SEED`](crate::monte_carlo::DEFAULT_SEED)), so CI is deterministic;
//! set `MINIKALMAN_CONFORMANCE_SEED` to try others.
//!
//! The EKF and UKF scenarios are mildly nonlinear, where the bounds hold only approximately:
//! a failure there is a strong hint, not a proof, of a defect.

use crate::monte_carlo::{self, seed, Metrics};
use crate::reference::chi2;

/// Asserts the run-averaged NEES and the overall NIS of `metrics` are inside their bounds.
fn assert_consistent(what: &str, metrics: &Metrics, nees: [f64; 2], nis: [f64; 2]) {
    eprintln!("{what}: {metrics:?}");
    for &step in &chi2::NEES_STEPS {
        let value = metrics.nees[step - 1];
        assert!(
            (nees[0]..=nees[1]).contains(&value),
            "{what}: NEES {value} at step {step} outside {nees:?} (α = {}, seed {})",
            chi2::ALPHA,
            seed()
        );
    }

    assert!(
        (nis[0]..=nis[1]).contains(&metrics.nis),
        "{what}: NIS {} outside {nis:?} (α = {}, seed {})",
        metrics.nis,
        chi2::ALPHA,
        seed()
    );
}

/// The KF on constant velocity has NEES and NIS inside the chi-square bounds.
#[test]
fn regular_constant_velocity_is_consistent() {
    let metrics = monte_carlo::regular_cv(seed());
    assert_consistent("kf cv", &metrics, chi2::NEES2, chi2::NIS1);
}

/// The EKF on the radar has NEES and NIS inside the chi-square bounds.
#[test]
fn extended_radar_is_consistent() {
    let metrics = monte_carlo::extended_radar(seed());
    assert_consistent("ekf radar", &metrics, chi2::NEES4, chi2::NIS2);
}

/// The UKF on the pendulum has NEES and NIS inside the chi-square bounds.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "micromath's approximate sqrt leaves the UKF underconfident (NEES 1.07 at step 20)"
)]
fn unscented_pendulum_is_consistent() {
    let metrics = monte_carlo::unscented_pendulum(seed());
    assert_consistent("ukf pendulum", &metrics, chi2::NEES2, chi2::NIS1);
}
