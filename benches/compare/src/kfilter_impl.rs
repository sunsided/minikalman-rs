//! kfilter 0.5.1 (nalgebra 0.34, const-generic, `no_std`).
//!
//! Linear problems use the documented `Kalman1M::new` (linear system, linear measurement).
//! `Kalman1M::new` starts with `P = Q`, so `P0` is written through `covariance_mut`.
//!
//! The radar uses `Kalman` with a linear system and a user-defined measurement type that
//! implements `Measurement` + `LinearisableMeasurement` and supplies the Jacobian before each
//! update.

use kfilter::kalman::{Kalman, Kalman1MLinearNoInput, KalmanFilter, KalmanPredict, KalmanUpdate};
use kfilter::measurement::{LinearisableMeasurement, Measurement};
use kfilter::system::{LinearNoInputSystem, System};
use na34::{SMatrix, SVector};

use crate::{Filter, Problem, Real, radar_h, radar_jacobian};

/// Row-major `f64` slice to a nalgebra 0.34 matrix.
pub fn mat<const R: usize, const C: usize, T: Real>(v: &[f64]) -> SMatrix<T, R, C> {
    SMatrix::from_row_iterator(v.iter().map(|&x| T::c(x)))
}

/// nalgebra 0.34 matrix to a row-major `f64` vector.
pub fn unmat<const R: usize, const C: usize, T: Real>(m: &SMatrix<T, R, C>) -> Vec<f64> {
    let mut out = Vec::with_capacity(R * C);
    for i in 0..R {
        for j in 0..C {
            out.push(m[(i, j)].f());
        }
    }
    out
}

/// kfilter's linear filter with a linear measurement.
pub struct Linear<const N: usize, const M: usize, T: Real> {
    k: Kalman1MLinearNoInput<T, N, M>,
}

impl<const N: usize, const M: usize, T: Real> Linear<N, M, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let mut k = Kalman1MLinearNoInput::new(
            mat::<N, N, T>(&p.F),
            mat::<N, N, T>(&p.Q),
            mat::<M, N, T>(&p.H),
            mat::<M, M, T>(&p.R),
            mat::<N, 1, T>(&p.x0),
        );
        *k.covariance_mut() = mat::<N, N, T>(&p.P0);
        Self { k }
    }
}

impl<const N: usize, const M: usize, T: Real> Filter for Linear<N, M, T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict().unwrap();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.k
            .update(SVector::<T, M>::from_column_slice(z))
            .unwrap();
    }
    fn state(&self) -> Vec<f64> {
        unmat(self.k.state())
    }
    fn covariance(&self) -> Vec<f64> {
        unmat(self.k.covariance())
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        *self.k.system_mut().state_mut() = mat::<N, 1, T>(x);
        *self.k.covariance_mut() = mat::<N, N, T>(p);
    }
}

/// The radar measurement for kfilter: `h(x)` and a Jacobian refreshed before each update.
#[derive(Clone, Debug)]
pub struct RadarMeasurement<T: Real> {
    z: SVector<T, 2>,
    H: SMatrix<T, 2, 4>,
    H_t: SMatrix<T, 4, 2>,
    R: SMatrix<T, 2, 2>,
}

impl<T: Real> Measurement<T, 4, 2> for RadarMeasurement<T> {
    fn covariance(&self) -> &SMatrix<T, 2, 2> {
        &self.R
    }
    fn measurement(&self) -> &SVector<T, 2> {
        &self.z
    }
    fn set_measurement(&mut self, z: SVector<T, 2>) {
        self.z = z;
    }
    #[inline(always)]
    fn predict(&self, x: &SVector<T, 4>) -> SVector<T, 2> {
        let h = radar_h(x.as_slice());
        SVector::<T, 2>::new(h[0], h[1])
    }
}

impl<T: Real> LinearisableMeasurement<T, 4, 2> for RadarMeasurement<T> {
    fn observation(&self) -> &SMatrix<T, 2, 4> {
        &self.H
    }
    fn observation_transpose(&self) -> &SMatrix<T, 4, 2> {
        &self.H_t
    }
}

/// kfilter's filter with a linear system and the radar measurement (an EKF update).
pub struct Radar<T: Real> {
    k: Kalman<T, 4, 0, LinearNoInputSystem<T, 4>>,
    m: RadarMeasurement<T>,
}

impl<T: Real> Radar<T> {
    /// Builds the filter for the radar problem `p`.
    pub fn new(p: &Problem) -> Self {
        let k = Kalman::new(
            mat::<4, 4, T>(&p.F),
            mat::<4, 4, T>(&p.Q),
            mat::<4, 1, T>(&p.x0),
            mat::<4, 4, T>(&p.P0),
        );
        let m = RadarMeasurement {
            z: SVector::zeros(),
            H: SMatrix::zeros(),
            H_t: SMatrix::zeros(),
            R: mat::<2, 2, T>(&p.R),
        };
        Self { k, m }
    }
}

impl<T: Real> Filter for Radar<T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict().unwrap();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.m
            .set_measurement(SVector::<T, 2>::from_column_slice(z));
        let j = radar_jacobian(self.k.state().as_slice());
        self.m.H = SMatrix::<T, 2, 4>::from_row_slice(&j);
        self.m.H_t = self.m.H.transpose();
        self.k.update(&self.m).unwrap();
    }
    fn state(&self) -> Vec<f64> {
        unmat(self.k.state())
    }
    fn covariance(&self) -> Vec<f64> {
        unmat(self.k.covariance())
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        *self.k.system_mut().state_mut() = mat::<4, 1, T>(x);
        *self.k.covariance_mut() = mat::<4, 4, T>(p);
    }
}
