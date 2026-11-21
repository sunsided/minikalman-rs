//! Regression baselines: accuracy metrics of the Monte Carlo scenarios, pinned to a band.
//!
//! Unlike the reference vectors, these numbers were produced by this crate (at the commit
//! that added them, in `f64` with [`DEFAULT_SEED`]). They are not proof of correctness; the
//! other categories are. They catch a change that keeps every identity intact but makes a
//! filter measurably less accurate or less consistent, e.g. a reordered sum that loses
//! precision or a changed default.
//!
//! A metric outside its band fails in both directions, so that an improvement is also
//! looked at and the baseline updated on purpose: rerun with `--nocapture`, copy the printed
//! value and say why in the commit.

use crate::monte_carlo::{self, DEFAULT_SEED, Metrics};

/// Allowed relative deviation from a baseline. The runs are deterministic, so this only has
/// to absorb last-bit differences of `sin` and `atan2` across platforms.
const BAND: f64 = 5e-3;

/// Pinned metrics of one scenario.
struct Baseline {
    /// Position RMSE.
    rmse: f64,
    /// NEES averaged over all runs and steps.
    anees: f64,
    /// NIS averaged over all runs and steps.
    nis: f64,
}

/// Asserts that each metric is within [`BAND`] of its baseline.
fn assert_baseline(what: &str, metrics: &Metrics, baseline: &Baseline) {
    eprintln!("{what}: {metrics:?}");
    for (name, value, pinned) in [
        ("RMSE", metrics.rmse, baseline.rmse),
        ("ANEES", metrics.anees, baseline.anees),
        ("NIS", metrics.nis, baseline.nis),
    ] {
        let deviation = (value - pinned) / pinned;
        assert!(
            deviation.abs() <= BAND,
            "{what}: {name} {value} deviates {:+.3} % from baseline {pinned}",
            deviation * 100.0
        );
    }
}

/// The KF on constant velocity keeps its pinned RMSE, ANEES and NIS.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "baselines are taken with exact sqrt"
)]
fn regular_constant_velocity_baseline() {
    let baseline = Baseline {
        rmse: 0.256_368_264_807_235_4,
        anees: 1.809_670_443_792_896,
        nis: 1.001_115_961_211_442,
    };
    assert_baseline("kf cv", &monte_carlo::regular_cv(DEFAULT_SEED), &baseline);
}

/// The EKF on the radar keeps its pinned RMSE, ANEES and NIS.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "baselines are taken with exact sqrt"
)]
fn extended_radar_baseline() {
    let baseline = Baseline {
        rmse: 0.159_673_736_178_120_86,
        anees: 4.038_341_001_306_62,
        nis: 1.948_906_652_676_147_2,
    };
    assert_baseline(
        "ekf radar",
        &monte_carlo::extended_radar(DEFAULT_SEED),
        &baseline,
    );
}

/// The UKF on the pendulum keeps its pinned RMSE, ANEES and NIS.
#[test]
#[cfg_attr(
    all(feature = "micromath", not(any(feature = "std", feature = "libm"))),
    ignore = "baselines are taken with exact sqrt"
)]
fn unscented_pendulum_baseline() {
    let baseline = Baseline {
        rmse: 0.077_152_096_076_862_96,
        anees: 1.897_934_701_479_972_8,
        nis: 0.989_327_794_297_73,
    };
    assert_baseline(
        "ukf pendulum",
        &monte_carlo::unscented_pendulum(DEFAULT_SEED),
        &baseline,
    );
}
