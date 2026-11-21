//! Bit-exact comparisons for tests that pin rewritten kernels to the results of the old ones.

/// The bit pattern of a scalar, so that tests can demand bit-identical results, including the
/// sign of zero.
pub trait ToBits: Copy + core::fmt::Debug {
    /// Returns the raw bits, widened to `u64`.
    fn to_bits_u64(self) -> u64;
}

impl ToBits for f32 {
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl ToBits for f64 {
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
}

#[cfg(feature = "fixed")]
impl ToBits for fixed::types::I16F16 {
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits() as u32)
    }
}

#[cfg(feature = "fixed")]
impl ToBits for fixed::types::I32F32 {
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
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
