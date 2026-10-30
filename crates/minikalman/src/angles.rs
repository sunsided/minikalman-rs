//! Helpers for quantities that wrap around, such as headings and bearings.
//!
//! Use these to normalize innovations in
//! [`correct_with`](crate::regular::RegularKalman::correct_with) and
//! [`correct_nonlinear_with`](crate::extended::ExtendedKalman::correct_nonlinear_with).
//! They only need remainder and comparison on `T`, so they also work for fixed-point types.

use core::ops::Rem;

use crate::matrix::MatrixDataTypeBase;

/// Wraps an angle in radians into `(-π, π]`.
///
/// ```
/// use minikalman::angles::wrap_pi;
/// use core::f32::consts::PI;
///
/// assert!((wrap_pi(1.5 * PI) - -0.5 * PI).abs() < 1e-6);
/// assert!((wrap_pi(-PI) - PI).abs() < 1e-6);
/// ```
pub fn wrap_pi<T>(angle: T) -> T
where
    T: MatrixDataTypeBase + Rem<Output = T>,
{
    wrap_half_turn(angle, T::from_f64(core::f64::consts::PI).expect("π fits T"))
}

/// Wraps an angle in degrees into `(-180, 180]`.
///
/// ```
/// use minikalman::angles::wrap_180;
///
/// assert_eq!(wrap_180(1.0_f32 - 359.0), 2.0);
/// assert_eq!(wrap_180(-180.0_f32), 180.0);
/// ```
pub fn wrap_180<T>(angle: T) -> T
where
    T: MatrixDataTypeBase + Rem<Output = T>,
{
    wrap_half_turn(angle, T::from_u8(180).expect("180 fits T"))
}

/// Wraps `angle` into `(-half_turn, half_turn]`.
fn wrap_half_turn<T>(angle: T, half_turn: T) -> T
where
    T: MatrixDataTypeBase + Rem<Output = T>,
{
    let turn = half_turn + half_turn;
    let mut shifted = (angle + half_turn) % turn;
    if shifted <= T::zero() {
        shifted += turn;
    }
    shifted - half_turn
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn wraps_degrees() {
        for (input, expected) in [
            (0.0, 0.0),
            (180.0, 180.0),
            (-180.0, 180.0),
            (181.0, -179.0),
            (-181.0, 179.0),
            (359.0, -1.0),
            (720.0 + 10.0, 10.0),
            (-720.0 - 10.0, -10.0),
        ] {
            assert_eq!(wrap_180(input), expected, "wrap_180({input})");
        }
    }

    #[test]
    fn wraps_radians() {
        assert!((wrap_pi(1.5 * PI) + 0.5 * PI).abs() < 1e-12);
        assert!((wrap_pi(-1.5 * PI) - 0.5 * PI).abs() < 1e-12);
        assert!((wrap_pi(PI) - PI).abs() < 1e-12);
        assert!((wrap_pi(-PI) - PI).abs() < 1e-12);
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn wraps_fixed_point() {
        use fixed::types::I16F16;

        let wrapped = wrap_180(I16F16::from_num(1) - I16F16::from_num(359));
        assert_eq!(wrapped, I16F16::from_num(2));
    }
}
