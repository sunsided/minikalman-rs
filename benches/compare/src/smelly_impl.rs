//! smelly 0.2.0 (nalgebra 0.35; `no_std` + `alloc`).
//!
//! `SKalmanFilter` / `SExtendedKalmanFilter` are the fixed-size filters, `KalmanFilter` the
//! dynamically sized one. The dynamic filter's `update` takes a `&DVector`; the wrapper keeps
//! one preallocated `DVector` and copies each measurement into it, as a caller would.

use na35::{DMatrix, DVector, SMatrix, SVector};
use smelly::{KalmanFilter, SExtendedKalmanFilter, SKalmanFilter};

use crate::{Filter, Problem, Real, radar_h, radar_jacobian};

/// Row-major `f64` slice to a nalgebra 0.35 static matrix.
pub fn mat<const R: usize, const C: usize, T: Real>(v: &[f64]) -> SMatrix<T, R, C> {
    SMatrix::from_row_iterator(v.iter().map(|&x| T::c(x)))
}

/// Row-major `f64` slice to a nalgebra 0.35 dynamic matrix.
pub fn dmat<T: Real>(r: usize, c: usize, v: &[f64]) -> DMatrix<T> {
    DMatrix::from_row_iterator(r, c, v.iter().map(|&x| T::c(x)))
}

fn unmat<T: Real>(r: usize, c: usize, get: impl Fn(usize, usize) -> T) -> Vec<f64> {
    let mut out = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            out.push(get(i, j).f());
        }
    }
    out
}

/// smelly's fixed-size linear filter.
pub struct Static<const N: usize, const M: usize, T: Real> {
    k: SKalmanFilter<T, N, M>,
}

impl<const N: usize, const M: usize, T: Real> Static<N, M, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let mut k = SKalmanFilter::<T, N, M>::new();
        k.set_transition(mat(&p.F)).unwrap();
        k.set_process_noise(mat(&p.Q)).unwrap();
        k.set_measurement(mat(&p.H)).unwrap();
        k.set_measurement_noise(mat(&p.R)).unwrap();
        k.set_state(mat(&p.x0)).unwrap();
        k.set_covariance(mat(&p.P0)).unwrap();
        Self { k }
    }
}

impl<const N: usize, const M: usize, T: Real> Filter for Static<N, M, T>
where
    na35::Const<M>: na35::DimMin<na35::Const<M>, Output = na35::Const<M>>,
{
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict().unwrap();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.k
            .update(&SVector::<T, M>::from_column_slice(z))
            .unwrap();
    }
    fn state(&self) -> Vec<f64> {
        unmat(N, 1, |i, _| self.k.state()[i])
    }
    fn covariance(&self) -> Vec<f64> {
        unmat(N, N, |i, j| self.k.covariance()[(i, j)])
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        self.k.set_state(mat(x)).unwrap();
        self.k.set_covariance(mat(p)).unwrap();
    }
}

/// smelly's dynamically sized linear filter.
pub struct Dynamic<T: Real> {
    k: KalmanFilter<T>,
    z: DVector<T>,
    n: usize,
}

impl<T: Real> Dynamic<T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let (n, m) = (p.n, p.m);
        let mut k = KalmanFilter::<T>::new(n, m);
        k.set_transition(dmat(n, n, &p.F)).unwrap();
        k.set_process_noise(dmat(n, n, &p.Q)).unwrap();
        k.set_measurement(dmat(m, n, &p.H)).unwrap();
        k.set_measurement_noise(dmat(m, m, &p.R)).unwrap();
        k.set_state(DVector::from_iterator(n, p.x0.iter().map(|&v| T::c(v))))
            .unwrap();
        k.set_covariance(dmat(n, n, &p.P0)).unwrap();
        Self {
            k,
            z: DVector::zeros(m),
            n,
        }
    }
}

impl<T: Real> Filter for Dynamic<T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict().unwrap();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.z.as_mut_slice().copy_from_slice(z);
        self.k.update(&self.z).unwrap();
    }
    fn state(&self) -> Vec<f64> {
        unmat(self.n, 1, |i, _| self.k.state()[i])
    }
    fn covariance(&self) -> Vec<f64> {
        unmat(self.n, self.n, |i, j| self.k.covariance()[(i, j)])
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        let n = self.n;
        self.k
            .set_state(DVector::from_iterator(n, x.iter().map(|&v| T::c(v))))
            .unwrap();
        self.k.set_covariance(dmat(n, n, p)).unwrap();
    }
}

/// smelly's fixed-size extended filter on the radar problem. Its `predict` is linear
/// (`x ← F·x`), which matches the constant-velocity motion here.
pub struct Radar<T: Real> {
    k: SExtendedKalmanFilter<T, 4, 2>,
}

impl<T: Real> Radar<T> {
    /// Builds the filter for the radar problem `p`.
    pub fn new(p: &Problem) -> Self {
        let mut k = SExtendedKalmanFilter::<T, 4, 2>::new();
        k.set_transition(mat(&p.F)).unwrap();
        k.set_process_noise(mat(&p.Q)).unwrap();
        k.set_measurement_noise(mat(&p.R)).unwrap();
        k.set_state(mat(&p.x0)).unwrap();
        k.set_covariance(mat(&p.P0)).unwrap();
        Self { k }
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
        self.k
            .update(
                &SVector::<T, 2>::from_column_slice(z),
                |x| {
                    let h = radar_h(x.as_slice());
                    SVector::<T, 2>::new(h[0], h[1])
                },
                |x| SMatrix::<T, 2, 4>::from_row_slice(&radar_jacobian(x.as_slice())),
            )
            .unwrap();
    }
    fn state(&self) -> Vec<f64> {
        unmat(4, 1, |i, _| self.k.state()[i])
    }
    fn covariance(&self) -> Vec<f64> {
        unmat(4, 4, |i, j| self.k.covariance()[(i, j)])
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        self.k.set_state(mat(x)).unwrap();
        self.k.set_covariance(mat(p)).unwrap();
    }
}
