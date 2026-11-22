use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use num_traits::{FromPrimitive, One, Zero};

/// Collection of numeric traits required for calculations in the Kalman filter.
/// This trait will be auto-implemented to the majority of relevant types.
pub trait MatrixDataTypeBase:
    Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Copy
    + Zero
    + One
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
    + core::iter::Sum<Self>
    + PartialOrd<Self>
    + FromPrimitive
{
}

/// Trait identifying the data type used by the Kalman Filter matrix operations.
pub trait MatrixDataType: MatrixDataTypeBase {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        Self::one() / self
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self;

    /// Whether the Kalman gain of the regular and extended observations may factor innovation
    /// covariances of up to three measurements as `L×D×Lᵀ`, without square roots.
    ///
    /// That factorization divides by the pivots `Dᵢ`, the squares of the Cholesky pivots, so
    /// its intermediate values span twice the dynamic range of the Cholesky solve. That is
    /// harmless for floating-point types (`true` for `f32` and `f64`), but overflows a
    /// fixed-point type near its resolution, so the default, used by the fixed-point types, is
    /// `false`: they keep the Cholesky solve.
    const SQUARE_ROOT_FREE_GAIN: bool = false;

    /// The value a sum starts from: an additive identity that is exact for every addend, i.e.
    /// `additive_identity() + x` has the bits of `x` for every `x`.
    ///
    /// For floating-point types this is `−0.0`: `−0.0 + x = x` holds for every `x`, including
    /// `+0.0` and `−0.0`, so the compiler can drop the first addition of a dot product.
    /// `+0.0` is not exact (`+0.0 + (−0.0) = +0.0`), so starting from it forces that addition.
    /// The default, used by the fixed-point types, where `−0` does not exist, is
    /// [`Zero::zero`](num_traits::Zero::zero).
    #[inline(always)]
    fn additive_identity() -> Self {
        Self::zero()
    }
}

/// Auto-implementation of [`MatrixDataTypeBase`].
impl<T> MatrixDataTypeBase for T where
    T: Add<T, Output = T>
        + AddAssign<T>
        + Sub<T, Output = T>
        + SubAssign<T>
        + Copy
        + Zero
        + One
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Neg<Output = T>
        + core::iter::Sum<T>
        + PartialOrd<T>
        + FromPrimitive
{
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
impl MatrixDataType for f32 {
    const SQUARE_ROOT_FREE_GAIN: bool = true;

    /// `−0.0`, the exact additive identity; see [`MatrixDataType::additive_identity`].
    #[inline(always)]
    fn additive_identity() -> Self {
        -0.0
    }

    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.recip()
        }
        #[cfg(all(feature = "libm", not(feature = "std")))]
        {
            1.0 / self
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            micromath::F32Ext::recip(self)
        }
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(all(
            feature = "libm",
            not(feature = "std"),
            feature = "unsafe",
            target_arch = "arm",
            target_abi = "eabihf"
        ))]
        {
            hardware_sqrtf(self)
        }
        #[cfg(all(
            feature = "libm",
            not(feature = "std"),
            not(all(feature = "unsafe", target_arch = "arm", target_abi = "eabihf"))
        ))]
        {
            libm::sqrtf(self)
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            micromath::F32Ext::sqrt(self)
        }
    }
}

/// Square root with the FPU's `vsqrt.f32` on Arm hard-float targets.
///
/// `libm::sqrtf` has no Arm32 path and runs ~80 instructions in software; core's `f32::sqrt`
/// is not stable. Every VFP version has `vsqrt.f32`, and the `eabihf` ABI implies a VFP. Both
/// are IEEE 754 correctly rounded, so the results are bit-identical (NaN payloads aside).
#[cfg(all(
    feature = "libm",
    not(feature = "std"),
    feature = "unsafe",
    target_arch = "arm",
    target_abi = "eabihf"
))]
#[inline]
fn hardware_sqrtf(x: f32) -> f32 {
    let bits: u32;

    // SAFETY: the block only computes in registers: no memory, no stack. It borrows `s0` and
    // restores it, because rustc 1.85 (the MSRV) rejects S-register operands on the M4F
    // ("register class `sreg` requires the `vfp2` target feature"). `vsqrt.f32` sets FPSCR exception flags,
    // hence no `preserves_flags`.
    unsafe {
        core::arch::asm!(
            "vmov {saved}, s0",
            "vmov s0, {x}",
            "vsqrt.f32 s0, s0",
            "vmov {root}, s0",
            "vmov s0, {saved}",
            x = in(reg) x.to_bits(),
            root = lateout(reg) bits,
            saved = out(reg) _,
            options(pure, nomem, nostack)
        );
    }
    f32::from_bits(bits)
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
impl MatrixDataType for f64 {
    const SQUARE_ROOT_FREE_GAIN: bool = true;

    /// `−0.0`, the exact additive identity; see [`MatrixDataType::additive_identity`].
    #[inline(always)]
    fn additive_identity() -> Self {
        -0.0
    }

    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.recip()
        }
        #[cfg(all(feature = "libm", not(feature = "std")))]
        {
            1.0 / self
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            1.0 / self
        }
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(all(feature = "libm", not(feature = "std")))]
        {
            libm::sqrt(self)
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            micromath::F32Ext::sqrt(self as f32) as f64
        }
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "fixed")))]
#[cfg(feature = "fixed")]
impl MatrixDataType for fixed::types::I16F16 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        self.recip()
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        self.sqrt()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "fixed")))]
#[cfg(feature = "fixed")]
impl MatrixDataType for fixed::types::I32F32 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        self.recip()
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        self.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::MatrixDataType;

    /// `additive_identity() + x` must have the bits of `x` for every `x`.
    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn float_additive_identity_is_exact() {
        let start = <f32 as MatrixDataType>::additive_identity();
        assert_eq!(start.to_bits(), (-0.0f32).to_bits());
        for x in [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
        ] {
            assert_eq!((start + x).to_bits(), x.to_bits(), "{x:?}");
        }
        assert!((start + f32::NAN).is_nan());

        let start = <f64 as MatrixDataType>::additive_identity();
        assert_eq!(start.to_bits(), (-0.0f64).to_bits());
        for x in [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::NEG_INFINITY,
        ] {
            assert_eq!((start + x).to_bits(), x.to_bits(), "{x:?}");
        }
    }

    /// Fixed point has no `−0`: the additive identity is plain zero.
    #[cfg(feature = "fixed")]
    #[test]
    fn fixed_additive_identity_is_zero() {
        use fixed::types::{I16F16, I32F32};

        assert_eq!(<I16F16 as MatrixDataType>::additive_identity().to_bits(), 0);
        assert_eq!(<I32F32 as MatrixDataType>::additive_identity().to_bits(), 0);
        assert_eq!((-I16F16::ZERO).to_bits(), 0);
        for x in [I16F16::ZERO, I16F16::MIN, I16F16::MAX, I16F16::DELTA] {
            assert_eq!(<I16F16 as MatrixDataType>::additive_identity() + x, x);
        }
    }

    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn f32_recip_and_sqrt() {
        let x: f32 = 2.0;
        let r = <f32 as MatrixDataType>::recip(x);
        assert!((r - 0.5).abs() < 1e-3);

        let s = <f32 as MatrixDataType>::square_root(x);
        assert!(s > 1.0 && s < 2.0);

        let y: f32 = 4.0;
        let sy = <f32 as MatrixDataType>::square_root(y);
        assert!(sy > 1.0 && sy < 3.0);
    }

    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn f64_recip_and_sqrt() {
        let x: f64 = 2.0;
        let r = <f64 as MatrixDataType>::recip(x);
        assert!((r - 0.5).abs() < 1e-3);

        let s = <f64 as MatrixDataType>::square_root(x);
        assert!(s > 1.0 && s < 2.0);

        let y: f64 = 4.0;
        let sy = <f64 as MatrixDataType>::square_root(y);
        assert!(sy > 1.0 && sy < 3.0);
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn i16f16_recip_and_sqrt() {
        use fixed::types::I16F16;

        let x = I16F16::from_num(2);
        let r = <I16F16 as MatrixDataType>::recip(x);
        let expected_recip = I16F16::from_num(0.5);
        let diff = (r - expected_recip).abs();
        assert!(diff < I16F16::from_num(0.001));

        let y = I16F16::from_num(4);
        let s = <I16F16 as MatrixDataType>::square_root(y);
        let expected_sqrt = I16F16::from_num(2);
        let diff = (s - expected_sqrt).abs();
        assert!(diff < I16F16::from_num(0.001));
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn i32f32_recip_and_sqrt() {
        use fixed::types::I32F32;

        let x = I32F32::from_num(2);
        let r = <I32F32 as MatrixDataType>::recip(x);
        let expected_recip = I32F32::from_num(0.5);
        let diff = (r - expected_recip).abs();
        assert!(diff < I32F32::from_num(0.0001));

        let y = I32F32::from_num(4);
        let s = <I32F32 as MatrixDataType>::square_root(y);
        let expected_sqrt = I32F32::from_num(2);
        let diff = (s - expected_sqrt).abs();
        assert!(diff < I32F32::from_num(0.0001));
    }
}
