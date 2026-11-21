//! Shared problems, an `f64` reference filter and thin wrappers around each crate's filter.
//!
//! Every wrapper implements [`Filter`], so the agreement check and the benchmarks drive all
//! crates through the same calls: `predict`, `correct` with one measurement vector, and
//! `restore` to a snapshot. The wrappers only forward to each crate's own API.

#![allow(non_snake_case)]

pub mod kalman_filters_impl;
pub mod kfilter_impl;
pub mod minikalman_impl;
pub mod problem;
pub mod reference;
pub mod registry;
pub mod smelly_impl;

pub use problem::{Model, Problem};

/// The scalar types every crate under test accepts: `f32` and `f64`.
pub trait Real:
    num_traits::Float
    + minikalman::prelude::MatrixDataType
    + na34::RealField
    + na35::RealField
    + kalman_filters::KalmanScalar
    + Default
    + core::fmt::Debug
    + 'static
{
    /// Converts from `f64`.
    fn c(v: f64) -> Self;
    /// Converts to `f64`.
    fn f(self) -> f64;
    /// The type name used in benchmark ids.
    const NAME: &'static str;
}

impl Real for f32 {
    #[inline(always)]
    fn c(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn f(self) -> f64 {
        self as f64
    }
    const NAME: &'static str = "f32";
}

impl Real for f64 {
    #[inline(always)]
    fn c(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn f(self) -> f64 {
        self
    }
    const NAME: &'static str = "f64";
}

/// One filter under test, driven through its crate's own API.
pub trait Filter {
    /// The scalar type.
    type T: Real;
    /// Time update.
    fn predict(&mut self);
    /// Measurement update with `z` (length `M`). Panics if the crate rejects the update.
    fn correct(&mut self, z: &[Self::T]);
    /// The state estimate, as `f64`.
    fn state(&self) -> Vec<f64>;
    /// The covariance, row-major, as `f64`. Empty for filters without a covariance.
    fn covariance(&self) -> Vec<f64>;
    /// Sets the state and covariance (row-major) to a snapshot.
    fn restore(&mut self, x: &[f64], p: &[f64]);
}

/// Converts a slice of `f64` to `T`.
pub fn conv<T: Real>(v: &[f64]) -> Vec<T> {
    v.iter().map(|&x| T::c(x)).collect()
}

/// The constant-velocity motion of the radar problem: `next = F·x` for `x = [p…, v…]`.
#[inline(always)]
pub fn cv_step<T: Real>(dt: T, x: &[T], next: &mut [T]) {
    let d = x.len() / 2;
    for i in 0..d {
        next[i] = x[i] + dt * x[i + d];
        next[i + d] = x[i + d];
    }
}

/// The radar measurement `h(x) = [range, bearing]` of a 2-D constant-velocity state.
#[inline(always)]
pub fn radar_h<T: Real>(x: &[T]) -> [T; 2] {
    let (px, py) = (x[0], x[1]);
    [
        num_traits::Float::sqrt(px * px + py * py),
        num_traits::Float::atan2(py, px),
    ]
}

/// The Jacobian of [`radar_h`], row-major 2×4.
#[inline(always)]
pub fn radar_jacobian<T: Real>(x: &[T]) -> [T; 8] {
    let (px, py) = (x[0], x[1]);
    let r2 = px * px + py * py;
    let r = num_traits::Float::sqrt(r2);
    let z = T::zero();
    [px / r, py / r, z, z, -py / r2, px / r2, z, z]
}
