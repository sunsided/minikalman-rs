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
