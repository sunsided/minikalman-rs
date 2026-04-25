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
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.recip()
        }
        #[cfg(all(feature = "libm", not(any(feature = "std", feature = "micromath"))))]
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
        #[cfg(all(feature = "libm", not(any(feature = "std", feature = "micromath"))))]
        {
            libm::sqrtf(self)
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            micromath::F32Ext::sqrt(self)
        }
    }
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
impl MatrixDataType for f64 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.recip()
        }
        #[cfg(all(feature = "libm", not(any(feature = "std", feature = "micromath"))))]
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
        #[cfg(all(feature = "libm", not(any(feature = "std", feature = "micromath"))))]
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

    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn f32_recip_and_sqrt() {
        let x: f32 = 2.0;
        let r = x.recip();
        assert!((r - 0.5).abs() < 1e-3);

        let s = x.square_root();
        assert!(s > 1.0 && s < 2.0);

        let y: f32 = 4.0;
        let sy = y.square_root();
        assert!(sy > 1.0 && sy < 3.0);
    }

    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn f64_recip_and_sqrt() {
        let x: f64 = 2.0;
        let r = x.recip();
        assert!((r - 0.5).abs() < 1e-3);

        let s = x.square_root();
        assert!(s > 1.0 && s < 2.0);

        let y: f64 = 4.0;
        let sy = y.square_root();
        assert!(sy > 1.0 && sy < 3.0);
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn i16f16_recip_and_sqrt() {
        use fixed::types::I16F16;

        let x = I16F16::from_num(2);
        let r = x.recip();
        let expected_recip = I16F16::from_num(0.5);
        let diff = (r - expected_recip).abs();
        assert!(diff < I16F16::from_num(0.001));

        let y = I16F16::from_num(4);
        let s = y.square_root();
        let expected_sqrt = I16F16::from_num(2);
        let diff = (s - expected_sqrt).abs();
        assert!(diff < I16F16::from_num(0.001));
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn i32f32_recip_and_sqrt() {
        use fixed::types::I32F32;

        let x = I32F32::from_num(2);
        let r = x.recip();
        let expected_recip = I32F32::from_num(0.5);
        let diff = (r - expected_recip).abs();
        assert!(diff < I32F32::from_num(0.0001));

        let y = I32F32::from_num(4);
        let s = y.square_root();
        let expected_sqrt = I32F32::from_num(2);
        let diff = (s - expected_sqrt).abs();
        assert!(diff < I32F32::from_num(0.0001));
    }
}
