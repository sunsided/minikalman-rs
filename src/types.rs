use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
#[cfg(feature = "fixed")]
use fixed::types::{I16F16, I32F32};
#[cfg(feature = "float")]
use num_traits::Float;
use num_traits::{FromPrimitive, One, Zero};

/// A type that captures the value range of `i16` but is known to be
/// fastest on the target platform.
#[cfg(not(feature = "stdint"))]
pub type FastInt16 = i16;

/// A type that captures the value range of `i16` but is known to be
/// fastest on the target platform.
#[cfg(feature = "stdint")]
pub type FastInt16 = stdint::int_fast16_t;

/// A type that captures the value range of `i32` but is known to be
/// fastest on the target platform.
#[cfg(not(feature = "stdint"))]
pub type FastInt32 = i32;

/// A type that captures the value range of `i32` but is known to be
/// fastest on the target platform.
#[cfg(feature = "stdint")]
pub type FastInt32 = stdint::int_fast32_t;

/// A type that captures the value range of `u16` but is known to be
/// fastest on the target platform.
#[cfg(not(feature = "stdint"))]
pub type FastUInt16 = u16;

/// A type that captures the value range of `u16` but is known to be
/// fastest on the target platform.
#[cfg(feature = "stdint")]
pub type FastUInt16 = stdint::uint_fast16_t;

/// A type that captures the value range of `u8` but is known to be
/// fastest on the target platform.
#[cfg(not(feature = "stdint"))]
pub type FastUInt8 = u8;

/// A type that captures the value range of `u8` but is known to be
/// fastest on the target platform.
#[cfg(feature = "stdint")]
pub type FastUInt8 = stdint::uint_fast8_t;

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

/// Trait identifying the data type used by the Kalman Filter matrix operations.
pub trait MatrixDataType: MatrixDataTypeBase {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        Self::one() / self
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self;
}

#[cfg_attr(docsrs, doc(cfg(feature = "float")))]
#[cfg(feature = "float")]
impl MatrixDataType for f32 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        self.recip()
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        self.sqrt()
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "float")))]
#[cfg(feature = "float")]
impl MatrixDataType for f64 {
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
impl MatrixDataType for I16F16 {
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
impl MatrixDataType for I32F32 {
    /// Calculates the reciprocal (inverse) of a number, i.e. `1/self`.
    fn recip(self) -> Self {
        self.recip()
    }

    /// Calculates the square root of a number.
    fn square_root(self) -> Self {
        self.sqrt()
    }
}
