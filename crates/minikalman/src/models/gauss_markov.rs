//! First-order Gauss-Markov processes.
//!
//! Sensor bias drift and colored (band-limited) noise are usually modelled by augmenting the
//! state with a first-order Gauss-Markov state `b`, which decays towards zero with the time
//! constant `τ` while being driven by white noise:
//!
//! ```text
//! bₖ₊₁ = a·bₖ + wₖ,    a = exp(−∆t/τ),    Var(wₖ) = q = σ²·(1 − a²)
//! ```
//!
//! `σ²` is the steady-state variance of `b`: the process noise keeps the bias at that
//! variance for any time step. A [`FirstOrder`] model fills the diagonal entries of `A` and
//! `Q` that belong to the bias state and leaves all other entries untouched.
//!
//! ## Example
//!
//! A 1D position filter with the state `[position, bias]`. A biased sensor measures
//! `position + bias` every step; an unbiased reference fix arrives every tenth step and
//! makes the bias observable.
//!
//! ```
//! # #[cfg(all(feature = "alloc", any(feature = "std", feature = "libm", feature = "micromath")))] {
//! use minikalman::models::gauss_markov;
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//!
//! const POSITION: usize = 0;
//! const BIAS: usize = 1;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build();
//! let mut biased = builder.observations().build::<1>();
//! let mut reference = builder.observations().build::<1>();
//!
//! biased.observation_matrix_mut().apply(|h| {
//!     h.set_at(0, POSITION, 1.0);
//!     h.set_at(0, BIAS, 1.0);
//! });
//! reference.observation_matrix_mut().apply(|h| h.set_at(0, POSITION, 1.0));
//!
//! // The bias drifts with a time constant of 30 s and a standard deviation of 0.2 m.
//! let bias = gauss_markov::first_order(0.1, 30.0, 0.2 * 0.2);
//! filter.state_transition_mut().apply(|a| {
//!     a.set_at(POSITION, POSITION, 1.0);
//!     bias.fill_transition(a, BIAS);
//! });
//! filter.direct_process_noise_mut().apply(|q| {
//!     q.set_at(POSITION, POSITION, 0.01);
//!     bias.fill_process_noise(q, BIAS);
//! });
//!
//! for step in 0..50 {
//!     filter.predict();
//!
//!     biased.measurement_vector_mut().apply(|z| z[0] = 10.25);
//!     filter.correct(&mut biased);
//!
//!     if step % 10 == 0 {
//!         reference.measurement_vector_mut().apply(|z| z[0] = 10.0);
//!         filter.correct(&mut reference);
//!     }
//! }
//!
//! let state = filter.state_vector();
//! assert!((state.get_row(POSITION) - 10.0).abs() < 0.1);
//! assert!((state.get_row(BIAS) - 0.25).abs() < 0.1);
//! # }
//! ```
//!
//! ## Fixed-point types
//!
//! [`first_order`] needs an exponential, which is available for `f32` and `f64` (see
//! [`Exponential`]). Fixed-point types have none; compute `a` ahead of time and use
//! [`from_coefficient`] instead:
//!
//! ```
//! # #[cfg(feature = "fixed")] {
//! use fixed::types::I16F16;
//! use minikalman::models::gauss_markov;
//!
//! let (dt, tau) = (0.1_f64, 30.0_f64);
//! let a = I16F16::from_num((-dt / tau).exp());
//! let bias = gauss_markov::from_coefficient(a, I16F16::from_num(0.04));
//! # }
//! ```

use crate::matrix::{MatrixDataTypeBase, MatrixMut};

/// Creates a first-order Gauss-Markov model from its time constant.
///
/// ## Arguments
/// * `dt` - The time step.
/// * `time_constant` - The correlation time `τ`, in the same unit as `dt`. Must be positive.
/// * `variance` - The steady-state variance `σ²` of the process.
pub fn first_order<T: Exponential>(dt: T, time_constant: T, variance: T) -> FirstOrder<T> {
    from_coefficient((-(dt / time_constant)).exponential(), variance)
}

/// Creates a first-order Gauss-Markov model from a precomputed coefficient `a = exp(−∆t/τ)`.
///
/// Use this for types without an [`Exponential`] implementation, such as fixed-point types,
/// or when the time step is constant and `a` can be computed once.
///
/// ## Arguments
/// * `coefficient` - The transition coefficient `a`, in `[0, 1)` for a stable process.
/// * `variance` - The steady-state variance `σ²` of the process.
pub fn from_coefficient<T>(coefficient: T, variance: T) -> FirstOrder<T> {
    FirstOrder {
        coefficient,
        variance,
    }
}

/// A first-order Gauss-Markov model for a single state.
///
/// ```text
/// A[i,i] = a      Q[i,i] = σ²·(1 − a²)
/// ```
///
/// where `a = exp(−∆t/τ)` and `σ²` is the steady-state variance. See [`first_order`] and
/// [`from_coefficient`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstOrder<T> {
    /// The transition coefficient `a = exp(−∆t/τ)`.
    coefficient: T,
    /// The steady-state variance `σ²`.
    variance: T,
}

impl<T: MatrixDataTypeBase> FirstOrder<T> {
    /// Sets the transition entry `A[index, index] = a` of the Gauss-Markov state.
    ///
    /// All other entries are left untouched; the rest of row `index` should be zero.
    pub fn fill_transition<const N: usize, A>(&self, transition: &mut A, index: usize)
    where
        A: MatrixMut<N, N, T> + ?Sized,
    {
        transition.set_at(index, index, self.coefficient);
    }

    /// Sets the process noise entry `Q[index, index] = σ²·(1 − a²)` of the Gauss-Markov state.
    ///
    /// All other entries are left untouched.
    pub fn fill_process_noise<const N: usize, Q>(&self, process_noise: &mut Q, index: usize)
    where
        Q: MatrixMut<N, N, T> + ?Sized,
    {
        let a = self.coefficient;
        process_noise.set_at(index, index, self.variance * (T::one() - a * a));
    }
}

/// The exponential function `eˣ`, needed by [`first_order`].
///
/// Implemented for `f32` and `f64` when `std`, `libm` or `micromath` is enabled. Implement it
/// for other types, or use [`from_coefficient`].
pub trait Exponential: MatrixDataTypeBase {
    /// Calculates `eˣ` for `x = self`.
    fn exponential(self) -> Self;
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
impl Exponential for f32 {
    fn exponential(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.exp()
        }
        #[cfg(all(feature = "libm", not(feature = "std")))]
        {
            libm::expf(self)
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            micromath::F32Ext::exp(self)
        }
    }
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
impl Exponential for f64 {
    fn exponential(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.exp()
        }
        #[cfg(all(feature = "libm", not(feature = "std")))]
        {
            libm::exp(self)
        }
        #[cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
        {
            micromath::F32Ext::exp(self as f32) as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn first_order_f32_matrices() {
        let model = first_order(1.0_f32, 2.0, 4.0);
        let mut a = MatrixData::new_array::<3, 3, 9, f32>([9.0; 9]);
        let mut q = MatrixData::new_array::<3, 3, 9, f32>([9.0; 9]);
        model.fill_transition(&mut a, 2);
        model.fill_process_noise(&mut q, 2);

        // a = exp(−1/2), q = 4·(1 − exp(−1)); loose bounds for micromath's approximate exp
        assert!((a.get_at(2, 2) - 0.606_530_66).abs() < 1e-3);
        assert!((q.get_at(2, 2) - 2.528_482_3).abs() < 1e-2);
        assert_eq!(&a.as_slice()[..8], &[9.0; 8]);
        assert_eq!(&q.as_slice()[..8], &[9.0; 8]);
    }

    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    #[test]
    fn first_order_f64_matrices() {
        let model = first_order(0.1_f64, 30.0, 0.04);
        let mut a = MatrixData::new_array::<1, 1, 1, f64>([0.0]);
        let mut q = MatrixData::new_array::<1, 1, 1, f64>([0.0]);
        model.fill_transition(&mut a, 0);
        model.fill_process_noise(&mut q, 0);

        // a = exp(−0.1/30) ≈ 0.996_672, q = 0.04·(1 − a²)
        assert!((a.get_at(0, 0) - 0.996_672_2).abs() < 1e-5);
        assert!((q.get_at(0, 0) - 0.000_265_8).abs() < 1e-6);
    }

    #[cfg(feature = "fixed")]
    #[test]
    fn from_coefficient_fixed_point() {
        use fixed::types::I16F16;

        let model = from_coefficient(I16F16::from_num(0.5), I16F16::from_num(2.0));
        let mut a = MatrixData::new_array::<2, 2, 4, I16F16>([I16F16::ZERO; 4]);
        let mut q = MatrixData::new_array::<2, 2, 4, I16F16>([I16F16::ZERO; 4]);
        model.fill_transition(&mut a, 1);
        model.fill_process_noise(&mut q, 1);

        assert_eq!(a.get_at(1, 1), I16F16::from_num(0.5));
        assert_eq!(q.get_at(1, 1), I16F16::from_num(1.5));
        assert_eq!(a.get_at(0, 0), I16F16::ZERO);
    }
}
