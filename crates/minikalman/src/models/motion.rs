//! Common motion models.
//!
//! Filters whose time step changes from sample to sample have to rebuild the state
//! transition and process noise matrices before every prediction. The models in this
//! module describe a single axis and fill both matrices from the time step:
//!
//! ```
//! # #[cfg(feature = "alloc")] {
//! use minikalman::models::motion;
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build();
//!
//! for dt in [0.1, 0.12, 0.09] {
//!     let model = motion::constant_velocity(dt, 0.5);
//!     filter.state_transition_mut().apply(|a| model.fill_transition(a));
//!     filter.direct_process_noise_mut().apply(|noise| model.fill_process_noise(noise));
//!     filter.predict();
//! }
//! # }
//! ```
//!
//! ## Process noise
//!
//! The models use the *discrete* white noise model: the highest-order derivative
//! (acceleration for [`constant_velocity`], jerk for [`constant_acceleration`]) is
//! constant during a step and changes by a normally distributed amount between
//! steps. Its variance is passed per step, e.g. as `acceleration_variance` in
//! units of (m/s²)² for a position in metres.
//!
//! This is not the continuous white noise model, which takes a spectral density
//! instead and yields `∆t³/3, ∆t²/2, ∆t` terms. If that model is added later, it
//! will get its own constructor next to the existing ones; the model types keep
//! their fields private so that this does not break existing callers.

use crate::matrix::{MatrixDataTypeBase, MatrixMut};

/// Creates a constant-velocity model for the state `[position, velocity]`.
///
/// ## Arguments
/// * `dt` - The time step.
/// * `acceleration_variance` - The variance of the (piecewise constant) acceleration
///   acting on the object during one step.
pub fn constant_velocity<T>(dt: T, acceleration_variance: T) -> ConstantVelocity<T> {
    ConstantVelocity {
        dt,
        acceleration_variance,
    }
}

/// Creates a constant-acceleration model for the state `[position, velocity, acceleration]`.
///
/// ## Arguments
/// * `dt` - The time step.
/// * `jerk_variance` - The variance of the (piecewise constant) jerk, i.e. the change
///   in acceleration, during one step.
pub fn constant_acceleration<T>(dt: T, jerk_variance: T) -> ConstantAcceleration<T> {
    ConstantAcceleration { dt, jerk_variance }
}

/// A constant-velocity model for the state `[position, velocity]`.
///
/// ```text
/// A = | 1  ∆t |      Q = σ²ₐ | ∆t⁴/4  ∆t³/2 |
///     | 0   1 |              | ∆t³/2  ∆t²   |
/// ```
///
/// where `σ²ₐ` is the acceleration variance. See [`constant_velocity`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantVelocity<T> {
    dt: T,
    acceleration_variance: T,
}

impl<T: MatrixDataTypeBase> ConstantVelocity<T> {
    /// Fills the state transition matrix `A`.
    pub fn fill_transition<A>(&self, transition: &mut A)
    where
        A: MatrixMut<2, 2, T> + ?Sized,
    {
        let (zero, one) = (T::zero(), T::one());
        transition.set_at(0, 0, one);
        transition.set_at(0, 1, self.dt);
        transition.set_at(1, 0, zero);
        transition.set_at(1, 1, one);
    }

    /// Fills the process noise covariance matrix `Q`.
    pub fn fill_process_noise<Q>(&self, process_noise: &mut Q)
    where
        Q: MatrixMut<2, 2, T> + ?Sized,
    {
        let dt = self.dt;
        let g = [dt * dt * half(), dt];
        fill_outer_product(process_noise, self.acceleration_variance, &g);
    }
}

/// A constant-acceleration model for the state `[position, velocity, acceleration]`.
///
/// ```text
///     | 1  ∆t  ∆t²/2 |             | ∆t⁴/4  ∆t³/2  ∆t²/2 |
/// A = | 0   1  ∆t    |    Q = σ²ⱼ | ∆t³/2  ∆t²    ∆t    |
///     | 0   0   1    |             | ∆t²/2  ∆t     1     |
/// ```
///
/// where `σ²ⱼ` is the jerk variance. See [`constant_acceleration`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantAcceleration<T> {
    dt: T,
    jerk_variance: T,
}

impl<T: MatrixDataTypeBase> ConstantAcceleration<T> {
    /// Fills the state transition matrix `A`.
    pub fn fill_transition<A>(&self, transition: &mut A)
    where
        A: MatrixMut<3, 3, T> + ?Sized,
    {
        let (zero, one, dt) = (T::zero(), T::one(), self.dt);
        transition.set_at(0, 0, one);
        transition.set_at(0, 1, dt);
        transition.set_at(0, 2, dt * dt * half());
        transition.set_at(1, 0, zero);
        transition.set_at(1, 1, one);
        transition.set_at(1, 2, dt);
        transition.set_at(2, 0, zero);
        transition.set_at(2, 1, zero);
        transition.set_at(2, 2, one);
    }

    /// Fills the process noise covariance matrix `Q`.
    pub fn fill_process_noise<Q>(&self, process_noise: &mut Q)
    where
        Q: MatrixMut<3, 3, T> + ?Sized,
    {
        let dt = self.dt;
        let g = [dt * dt * half(), dt, T::one()];
        fill_outer_product(process_noise, self.jerk_variance, &g);
    }
}

fn half<T: MatrixDataTypeBase>() -> T {
    T::one() / (T::one() + T::one())
}

/// Sets `Q = σ²·g·gᵀ`.
fn fill_outer_product<const N: usize, T, Q>(process_noise: &mut Q, variance: T, g: &[T; N])
where
    T: MatrixDataTypeBase,
    Q: MatrixMut<N, N, T> + ?Sized,
{
    for (row, &gr) in g.iter().enumerate() {
        for (col, &gc) in g.iter().enumerate() {
            process_noise.set_at(row, col, variance * gr * gc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use assert_float_eq::*;

    #[test]
    fn constant_velocity_matrices() {
        let model = constant_velocity(0.5_f32, 2.0);
        let mut a = MatrixData::new_array::<2, 2, 4, f32>([9.0; 4]);
        let mut q = MatrixData::new_array::<2, 2, 4, f32>([9.0; 4]);
        model.fill_transition(&mut a);
        model.fill_process_noise(&mut q);

        assert_eq!(a.as_slice(), &[1.0, 0.5, 0.0, 1.0]);
        // σ²ₐ·[dt⁴/4, dt³/2; dt³/2, dt²] with dt = 0.5, σ²ₐ = 2
        assert_f32_near!(q.get_at(0, 0), 2.0 * 0.0625 / 4.0);
        assert_f32_near!(q.get_at(0, 1), 2.0 * 0.125 / 2.0);
        assert_f32_near!(q.get_at(1, 0), 2.0 * 0.125 / 2.0);
        assert_f32_near!(q.get_at(1, 1), 2.0 * 0.25);
    }

    #[test]
    fn constant_acceleration_matrices() {
        let model = constant_acceleration(2.0_f32, 1.0);
        let mut a = MatrixData::new_array::<3, 3, 9, f32>([9.0; 9]);
        let mut q = MatrixData::new_array::<3, 3, 9, f32>([9.0; 9]);
        model.fill_transition(&mut a);
        model.fill_process_noise(&mut q);

        assert_eq!(a.as_slice(), &[1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
        assert_eq!(q.as_slice(), &[4.0, 4.0, 2.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0]);
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn constant_velocity_fixed_point() {
        use fixed::types::I16F16;

        let model = constant_velocity(I16F16::from_num(0.5), I16F16::from_num(2.0));
        let mut a = MatrixData::new_array::<2, 2, 4, I16F16>([I16F16::ZERO; 4]);
        model.fill_transition(&mut a);
        assert_eq!(a.get_at(0, 1), I16F16::from_num(0.5));
    }
}
