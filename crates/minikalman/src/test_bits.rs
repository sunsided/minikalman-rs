//! Bit-exact comparisons for tests that pin rewritten kernels to the results of the old ones,
//! and deviation measurements for kernels whose rounding changed on purpose.

/// The bit pattern of a scalar, so that tests can demand bit-identical results, including the
/// sign of zero.
pub trait ToBits: Copy + core::fmt::Debug {
    /// Returns the raw bits, widened to `u64`.
    fn to_bits_u64(self) -> u64;

    /// Returns the value as `f64`, for tests that measure deviations rather than demand
    /// identical bits.
    fn to_f64(self) -> f64;
}

impl ToBits for f32 {
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl ToBits for f64 {
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }

    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(feature = "fixed")]
impl ToBits for fixed::types::I16F16 {
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits() as u32)
    }

    fn to_f64(self) -> f64 {
        self.to_num()
    }
}

#[cfg(feature = "fixed")]
impl ToBits for fixed::types::I32F32 {
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }

    fn to_f64(self) -> f64 {
        self.to_num()
    }
}

/// Asserts that two slices hold the same bit patterns.
pub fn assert_bits_eq<T: ToBits>(actual: &[T], expected: &[T], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits_u64(),
            e.to_bits_u64(),
            "{what}[{i}]: {a:?} != {e:?}"
        );
    }
}

/// A deterministic test value in `[-1, 1)` for entry `i` of a matrix seeded with `seed`.
pub fn test_value(seed: u32, i: usize) -> f64 {
    let mut h = (u64::from(seed) << 32 | i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Returns the largest relative deviation `|a − e| / (1 + |e|)` between two slices, or
/// infinity if any deviation is not finite (`f64::max` would drop a `NaN`).
pub fn max_deviation<T: ToBits>(actual: &[T], expected: &[T]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "length");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a.to_f64() - e.to_f64()).abs() / (1.0 + e.to_f64().abs()))
        .map(finite_or_infinity)
        .fold(0.0, f64::max)
}

/// Returns `value`, or infinity if it is `NaN` or infinite.
pub fn finite_or_infinity(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        f64::INFINITY
    }
}

/// A `NaN` or infinite deviation is reported as infinite, not dropped by the maximum.
#[test]
fn non_finite_deviations_are_infinite() {
    assert_eq!(max_deviation(&[f64::NAN, 1.0], &[0.0, 1.0]), f64::INFINITY);
    assert_eq!(max_deviation(&[f64::INFINITY], &[0.0]), f64::INFINITY);
    assert_eq!(max_deviation(&[1.0f32], &[1.0f32]), 0.0);
}
