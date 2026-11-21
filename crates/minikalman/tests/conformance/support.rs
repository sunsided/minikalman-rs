//! Number-type plumbing, tolerances and matrix checks shared by the conformance tests.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Sub};

use minikalman::matrix::{Matrix, MatrixDataType, MatrixMut};
use minikalman::num_traits::{FromPrimitive, ToPrimitive};

/// Whether the crate's `sqrt` and `recip` come from micromath's fast approximations.
///
/// The feature selection in `matrix/data_type.rs` prefers `std`, then `libm`, then
/// micromath, so micromath only takes effect when neither of the others is enabled.
pub const APPROXIMATE_MATH: bool = cfg!(all(
    feature = "micromath",
    not(any(feature = "std", feature = "libm"))
));

/// A number type the filters run on, convertible to and from `f64` for comparisons.
pub trait Real:
    MatrixDataType
    + Default
    + Copy
    + Debug
    + PartialOrd
    + FromPrimitive
    + ToPrimitive
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// Short name used in failure messages.
    const NAME: &'static str;

    /// How closely a filter in this type must reproduce an `f64` reference.
    const REFERENCE: Tolerance;

    /// Relative slack of the covariance invariants: a few units of the type's resolution,
    /// independent of the math backend.
    const COVARIANCE: f64;

    /// Converts a reference value into this type.
    fn of(value: f64) -> Self {
        Self::from_f64(value).expect("reference value fits the number type")
    }

    /// Converts this value to `f64`.
    fn f64(self) -> f64 {
        self.to_f64().expect("finite value")
    }
}

impl Real for f64 {
    const NAME: &'static str = "f64";
    const COVARIANCE: f64 = 1e-12;
    const REFERENCE: Tolerance = if APPROXIMATE_MATH {
        // micromath computes the f64 square root through an approximate f32 one; its errors
        // compound to about 0.2 over the reference runs.
        Tolerance::new(0.3, 0.1)
    } else {
        Tolerance::new(1e-12, 1e-11)
    };
}

impl Real for f32 {
    const NAME: &'static str = "f32";
    const COVARIANCE: f64 = 1e-5;
    const REFERENCE: Tolerance = if APPROXIMATE_MATH {
        Tolerance::new(0.3, 0.1)
    } else {
        Tolerance::new(5e-5, 1e-5)
    };
}

#[cfg(feature = "fixed")]
impl Real for fixed::types::I32F32 {
    const NAME: &'static str = "I32F32";
    const COVARIANCE: f64 = 1e-7;
    const REFERENCE: Tolerance = Tolerance::new(1e-7, 1e-7);
}

#[cfg(feature = "fixed")]
impl Real for fixed::types::I16F16 {
    const NAME: &'static str = "I16F16";
    const COVARIANCE: f64 = 1e-3;
    const REFERENCE: Tolerance = Tolerance::new(1e-3, 1e-3);
}

/// A mixed absolute and relative bound: `|actual − expected| ≤ abs + rel × |expected|`.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// Absolute part, which dominates near zero.
    pub abs: f64,
    /// Relative part, which dominates for large values.
    pub rel: f64,
}

impl Tolerance {
    /// Creates a tolerance from its absolute and relative parts.
    pub const fn new(abs: f64, rel: f64) -> Self {
        Self { abs, rel }
    }

    /// Whether `actual` is within this tolerance of `expected`.
    pub fn accepts(self, actual: f64, expected: f64) -> bool {
        actual.is_finite() && (actual - expected).abs() <= self.abs + self.rel * expected.abs()
    }
}

/// Asserts that every entry of `actual` is within `tolerance` of `expected`.
///
/// # Panics
/// Panics, naming the first offending index and its error, if the lengths differ or an
/// entry is outside the tolerance.
pub fn assert_close(what: &str, actual: &[f64], expected: &[f64], tolerance: Tolerance) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    // Shown with `--nocapture`, to see how much headroom a tolerance has.
    eprintln!(
        "{what}: max error {:e} ({tolerance:?})",
        max_error(actual, expected)
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            tolerance.accepts(a, e),
            "{what}[{i}]: {a} vs reference {e} (error {:e}, {tolerance:?}); max error {:e}",
            (a - e).abs(),
            max_error(actual, expected)
        );
    }
}

/// Largest absolute difference between two equally long slices.
pub fn max_error(actual: &[f64], expected: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f64::max)
}

/// Fills a matrix from row-major `f64` values.
pub fn load<const R: usize, const C: usize, T: Real>(
    matrix: &mut impl MatrixMut<R, C, T>,
    values: &[f64],
) {
    assert_eq!(values.len(), R * C, "row-major value count");
    for (i, &v) in values.iter().enumerate() {
        matrix.set(i / C, i % C, T::of(v));
    }
}

/// Reads a matrix into row-major `f64` values.
pub fn read<const R: usize, const C: usize, T: Real>(matrix: &impl Matrix<R, C, T>) -> Vec<f64> {
    (0..R * C).map(|i| matrix.get(i / C, i % C).f64()).collect()
}

/// Lower Cholesky factor of a row-major `n × n` matrix, or `None` if it is not positive
/// definite.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            let v = a[i * n + j] - dot;
            if i == j {
                if v <= 0.0 {
                    return None;
                }

                l[i * n + i] = v.sqrt();
            } else {
                l[i * n + j] = v / l[j * n + j];
            }
        }
    }

    Some(l)
}

/// `eᵀ × A⁻¹ × e` for a positive definite row-major `A`, solved through its Cholesky factor.
///
/// # Panics
/// Panics if `A` is not positive definite.
pub fn mahalanobis(a: &[f64], e: &[f64]) -> f64 {
    let n = e.len();
    let l = cholesky(a, n).expect("positive definite covariance");
    let mut y = vec![0.0; n];
    for i in 0..n {
        let dot: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (e[i] - dot) / l[i * n + i];
    }

    y.iter().map(|v| v * v).sum()
}

/// Asserts that a row-major `n × n` covariance is symmetric and positive semidefinite.
///
/// Symmetry is checked to `epsilon` times the largest diagonal entry (at least 1).
/// Semidefiniteness is checked by a Cholesky decomposition of `P + ε·I` with `ε` that same
/// scale.
///
/// # Panics
/// Panics with `what` if either property fails.
pub fn assert_covariance(what: &str, p: &[f64], n: usize, epsilon: f64) {
    let scale = (0..n).map(|i| p[i * n + i].abs()).fold(1.0, f64::max);
    let eps = epsilon * scale;
    for i in 0..n {
        for j in 0..i {
            let gap = (p[i * n + j] - p[j * n + i]).abs();
            assert!(
                gap <= eps,
                "{what}: P[{i}][{j}] asymmetric by {gap:e}: {p:?}"
            );
        }
    }

    let mut shifted = p.to_vec();
    for i in 0..n {
        shifted[i * n + i] += eps;
    }

    assert!(
        cholesky(&shifted, n).is_some(),
        "{what}: P not positive semidefinite: {p:?}"
    );
}

/// Splits a flat per-step history into one slice per step.
pub fn steps(flat: &[f64], width: usize) -> impl Iterator<Item = &[f64]> {
    flat.chunks_exact(width)
}
