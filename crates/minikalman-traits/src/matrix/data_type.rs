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

#[cfg(any(feature = "std", feature = "libm"))]
impl MatrixDataType for f32 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        #[cfg(feature = "libm")]
        {
            1.0 / self
        }
        #[cfg(feature = "std")]
        {
            self.recip()
        }
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        #[cfg(feature = "libm")]
        {
            libm::sqrtf(self)
        }
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
impl MatrixDataType for f64 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        #[cfg(feature = "libm")]
        {
            1.0 / self
        }
        #[cfg(feature = "std")]
        {
            self.recip()
        }
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        #[cfg(feature = "libm")]
        {
            libm::sqrt(self)
        }
        #[cfg(feature = "std")]
        {
            self.sqrt()
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
