//! kalman_filters 1.0.1 (`std`; optional `nalgebra` 0.34 for the static filter).
//!
//! `KalmanFilter` (built with the documented `KalmanFilterBuilder`) is the dynamically sized
//! filter. `StaticKalmanFilter<T, M, N>` (feature
//! `nalgebra`; measurements first) is the fixed-size variant. The extended filter takes a
//! `NonlinearSystem` implementation that returns `Vec`s; it is built with
//! `ExtendedKalmanFilterBuilder` and `JacobianStrategy::Analytical`.

use kalman_filters::filter::StaticKalmanFilter;
use kalman_filters::{
    ExtendedKalmanFilter, ExtendedKalmanFilterBuilder, JacobianStrategy, KalmanFilter,
    KalmanFilterBuilder, NonlinearSystem,
};
use na34::SVector;

use crate::kfilter_impl::{mat, unmat};
use crate::{Filter, Problem, Real, conv, radar_h, radar_jacobian};

/// The dynamically sized (`Vec`) linear filter.
pub struct Dynamic<T: Real> {
    k: KalmanFilter<T>,
}

impl<T: Real> Dynamic<T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let k = KalmanFilterBuilder::new(p.n, p.m)
            .initial_state(conv(&p.x0))
            .initial_covariance(conv(&p.P0))
            .transition_matrix(conv(&p.F))
            .process_noise(conv(&p.Q))
            .observation_matrix(conv(&p.H))
            .measurement_noise(conv(&p.R))
            .build()
            .unwrap();
        Self { k }
    }
}

impl<T: Real> Filter for Dynamic<T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.k.update(z).unwrap();
    }
    fn state(&self) -> Vec<f64> {
        self.k.state().iter().map(|v| v.f()).collect()
    }
    fn covariance(&self) -> Vec<f64> {
        self.k.covariance().iter().map(|v| v.f()).collect()
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        self.k.x = conv(x);
        self.k.P = conv(p);
    }
}

/// The fixed-size (nalgebra) linear filter.
pub struct Static<const N: usize, const M: usize, T: Real> {
    k: StaticKalmanFilter<T, M, N>,
}

impl<const N: usize, const M: usize, T: Real> Static<N, M, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let k = StaticKalmanFilter::<T, M, N>::new(
            mat::<N, 1, T>(&p.x0),
            mat::<N, N, T>(&p.P0),
            mat::<N, N, T>(&p.F),
            mat::<N, N, T>(&p.Q),
            mat::<M, N, T>(&p.H),
            mat::<M, M, T>(&p.R),
        );
        Self { k }
    }
}

impl<const N: usize, const M: usize, T: Real> Filter for Static<N, M, T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.k
            .update(&SVector::<T, M>::from_column_slice(z))
            .unwrap();
    }
    fn state(&self) -> Vec<f64> {
        unmat(self.k.state())
    }
    fn covariance(&self) -> Vec<f64> {
        unmat(self.k.covariance())
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        self.k.x = mat::<N, 1, T>(x);
        self.k.P = mat::<N, N, T>(p);
    }
}

/// The radar model as a kalman_filters `NonlinearSystem`.
pub struct RadarSystem<T> {
    F: Vec<T>,
}

impl<T: Real> NonlinearSystem<T> for RadarSystem<T> {
    fn state_transition(&self, state: &[T], _control: Option<&[T]>, _dt: T) -> Vec<T> {
        (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| self.F[i * 4 + j] * state[j])
                    .fold(T::zero(), |a, b| a + b)
            })
            .collect()
    }
    fn measurement(&self, state: &[T]) -> Vec<T> {
        radar_h(state).to_vec()
    }
    fn state_jacobian(&self, _state: &[T], _control: Option<&[T]>, _dt: T) -> Vec<T> {
        self.F.clone()
    }
    fn measurement_jacobian(&self, state: &[T]) -> Vec<T> {
        radar_jacobian(state).to_vec()
    }
    fn state_dim(&self) -> usize {
        4
    }
    fn measurement_dim(&self) -> usize {
        2
    }
}

/// The extended filter on the radar problem.
pub struct Radar<T: Real> {
    k: ExtendedKalmanFilter<T, RadarSystem<T>>,
}

impl<T: Real> Radar<T> {
    /// Builds the filter for the radar problem `p`.
    pub fn new(p: &Problem) -> Self {
        let k = ExtendedKalmanFilterBuilder::new(RadarSystem { F: conv(&p.F) })
            .initial_state(conv(&p.x0))
            .initial_covariance(conv(&p.P0))
            .process_noise(conv(&p.Q))
            .measurement_noise(conv(&p.R))
            .dt(T::c(p.dt))
            .jacobian_strategy(JacobianStrategy::Analytical)
            .build()
            .unwrap();
        Self { k }
    }
}

impl<T: Real> Filter for Radar<T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.k.predict();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        self.k.update(z).unwrap();
    }
    fn state(&self) -> Vec<f64> {
        self.k.state().iter().map(|v| v.f()).collect()
    }
    fn covariance(&self) -> Vec<f64> {
        self.k.covariance().iter().map(|v| v.f()).collect()
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        self.k.x = conv(x);
        self.k.P = conv(p);
    }
}
