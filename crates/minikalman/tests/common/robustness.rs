//! Metrics of the numerical-robustness test bench: how far an estimate covariance `P` has
//! degraded from a symmetric positive definite matrix, and how far a run has drifted from an
//! `f64` reference run.
//!
//! The bench (`tests/numerical_robustness.rs`) includes this module with
//! `#[path = "common/robustness.rs"]`. Factored filters (square-root, UD) can include it the
//! same way and record their own runs with [`Summary::record`].

#![allow(dead_code)]

use minikalman::matrix::MatrixDataBoxed;
use minikalman::num_traits::ToPrimitive;
use minikalman::prelude::*;

/// The metrics of one estimate covariance `P`, taken after a correction.
#[derive(Debug, Clone, Copy)]
pub struct StepMetrics {
    /// Whether `cholesky_decompose_lower` succeeds on a copy of `P` in the filter's own type.
    pub positive_definite: bool,

    /// The smallest diagonal entry of `P`, converted to `f64`.
    pub min_diagonal: f64,

    /// The largest relative asymmetry `|Pij − Pji| / √(|Pii×Pjj|)` over all `i ≠ j`; `0` for an
    /// exactly symmetric `P` and where a diagonal entry is zero.
    ///
    /// Scaled by the diagonal rather than by `max(|Pij|, |Pji|)`, because off-diagonal entries
    /// that are zero in exact arithmetic come out as rounding noise of either sign, which the
    /// latter would report as an asymmetry of `2`.
    pub max_asymmetry: f64,

    /// The normalized estimation error squared `eᵀ×P⁻¹×e` of the state error `e` against the
    /// `f64` reference run, with `P` converted to `f64`. `NaN` if that `P` is not positive
    /// definite in `f64`.
    pub nees: f64,
}

/// Measures `P` and the error `error = x − x_ref` of a run of `N` states in type `T`.
pub fn measure<const N: usize, T>(p: &impl Matrix<N, N, T>, error: [f64; N]) -> StepMetrics
where
    T: MatrixDataType + ToPrimitive,
{
    StepMetrics {
        positive_definite: is_positive_definite(p),
        min_diagonal: (0..N)
            .map(|i| to_f64(p.get(i, i)))
            .fold(f64::INFINITY, f64::min),
        max_asymmetry: max_asymmetry(p),
        nees: nees(&to_f64_matrix(p), error),
    }
}

/// Runs `cholesky_decompose_lower` on a copy of `p` in type `T`.
fn is_positive_definite<const N: usize, T>(p: &impl Matrix<N, N, T>) -> bool
where
    T: MatrixDataType,
{
    let mut copy = MatrixDataBoxed::<N, N, T>::from(p.as_slice().to_vec());
    copy.cholesky_decompose_lower()
}

/// Returns the largest relative asymmetry of `p`; see [`StepMetrics::max_asymmetry`].
fn max_asymmetry<const N: usize, T>(p: &impl Matrix<N, N, T>) -> f64
where
    T: MatrixDataType + ToPrimitive,
{
    let mut worst = 0.0f64;
    for i in 0..N {
        for j in (i + 1)..N {
            let (a, b) = (to_f64(p.get(i, j)), to_f64(p.get(j, i)));
            let scale = (to_f64(p.get(i, i)) * to_f64(p.get(j, j))).abs().sqrt();
            if scale > 0.0 {
                worst = worst.max((a - b).abs() / scale);
            }
        }
    }

    worst
}

/// Returns `eᵀ×P⁻¹×e` through a Cholesky decomposition of `p`, or `NaN` if `p` is not
/// positive definite.
fn nees<const N: usize>(p: &MatrixDataBoxed<N, N, f64>, error: [f64; N]) -> f64 {
    let mut l = MatrixDataBoxed::<N, N, f64>::from(p.as_slice().to_vec());
    if !l.cholesky_decompose_lower() {
        return f64::NAN;
    }

    // Forward substitution L×w = e; then eᵀ×P⁻¹×e = wᵀ×w.
    let mut w = [0.0; N];
    for i in 0..N {
        let partial: f64 = (0..i).map(|k| l.get(i, k) * w[k]).sum();
        w[i] = (error[i] - partial) / l.get(i, i);
    }

    w.iter().map(|v| v * v).sum()
}

/// Converts a value of the filter's type to `f64`.
pub fn to_f64<T: ToPrimitive>(value: T) -> f64 {
    value.to_f64().expect("every bench type converts to f64")
}

/// Converts an `N × N` matrix of the filter's type to `f64`.
pub fn to_f64_matrix<const N: usize, T>(p: &impl Matrix<N, N, T>) -> MatrixDataBoxed<N, N, f64>
where
    T: MatrixDataType + ToPrimitive,
{
    MatrixDataBoxed::from(p.as_slice().iter().map(|&v| to_f64(v)).collect::<Vec<_>>())
}

/// Reads an `N`-state vector of the filter's type as `f64`.
pub fn to_f64_vector<const N: usize, T>(x: &impl Matrix<N, 1, T>) -> [f64; N]
where
    T: MatrixDataType + ToPrimitive,
{
    core::array::from_fn(|i| to_f64(x.get(i, 0)))
}

/// The metrics of a whole run, aggregated over its steps.
#[derive(Debug, Clone, Copy)]
pub struct Summary {
    /// Number of recorded steps.
    pub steps: usize,

    /// Number of steps whose `P` failed the Cholesky check.
    pub not_positive_definite: usize,

    /// The first step whose `P` failed the Cholesky check, if any.
    pub first_failure: Option<usize>,

    /// The smallest diagonal entry seen in any step.
    pub min_diagonal: f64,

    /// The largest relative asymmetry seen in any step.
    pub max_asymmetry: f64,

    /// The largest finite NEES seen in any step; `0` if there is none.
    pub max_nees: f64,

    /// Number of steps whose NEES is `NaN`, i.e. whose `P` is not positive definite even in
    /// `f64`. Fewer of these than [`not_positive_definite`](Self::not_positive_definite) means
    /// that `P` is only too ill-conditioned for the Cholesky check in the filter's own type.
    pub nees_undefined: usize,

    /// Number of corrections that were not applied, e.g. because `S` was not positive
    /// definite.
    pub skipped: usize,
}

impl Default for Summary {
    fn default() -> Self {
        Self {
            steps: 0,
            not_positive_definite: 0,
            first_failure: None,
            min_diagonal: f64::INFINITY,
            max_asymmetry: 0.0,
            max_nees: 0.0,
            nees_undefined: 0,
            skipped: 0,
        }
    }
}

impl Summary {
    /// Adds the metrics of one step.
    pub fn record(&mut self, metrics: StepMetrics) {
        if !metrics.positive_definite {
            self.not_positive_definite += 1;
            self.first_failure.get_or_insert(self.steps);
        }

        self.min_diagonal = self.min_diagonal.min(metrics.min_diagonal);
        self.max_asymmetry = self.max_asymmetry.max(metrics.max_asymmetry);
        if metrics.nees.is_finite() {
            self.max_nees = self.max_nees.max(metrics.nees);
        } else {
            self.nees_undefined += 1;
        }

        self.steps += 1;
    }

    /// Counts a correction that returned anything but [`CorrectionStatus::Applied`].
    pub fn record_status(&mut self, status: CorrectionStatus) {
        if !status.is_applied() {
            self.skipped += 1;
        }
    }

    /// Prints the summary as one line of the bench report.
    pub fn report(&self, label: &str) {
        println!(
            "{label:<24} steps {:>6}  not PD {:>5} (first {:>5}, NaN NEES {:>5})  \
             min diag {:>10.3e}  max asym {:>8.2e}  max NEES {:>8.2e}  skipped {:>5}",
            self.steps,
            self.not_positive_definite,
            self.first_failure
                .map_or("-".to_string(), |s| s.to_string()),
            self.nees_undefined,
            self.min_diagonal,
            self.max_asymmetry,
            self.max_nees,
            self.skipped,
        );
    }
}
