//! minikalman: storage types, boxed builders, lean observation + scratch, UD, square root,
//! the fixed-gain (Luenberger) observer and the extended filter.
//!
//! The storage-backed variants borrow a storage value for the filter's lifetime. To keep the
//! filter in a plain struct, the benchmarks leak one boxed storage per filter at setup; the
//! per-step code is the same as with a storage on the stack or in a `static`.

use minikalman::extended as mext;
use minikalman::prelude::*;
use minikalman::regular::{self as mreg, CorrectionScratch, CovarianceUpdate};

use crate::{Filter, Problem, Real, cv_step, radar_h, radar_jacobian};

/// Writes row-major `v` into an `R×C` matrix.
pub fn fill<const R: usize, const C: usize, T: Real>(m: &mut impl MatrixMut<R, C, T>, v: &[f64]) {
    assert_eq!(v.len(), R * C);
    for i in 0..R {
        for j in 0..C {
            m.set(i, j, T::c(v[i * C + j]));
        }
    }
}

/// Reads an `R×C` matrix row-major as `f64`.
pub fn read<const R: usize, const C: usize, T: Real>(m: &impl Matrix<R, C, T>) -> Vec<f64> {
    let mut out = Vec::with_capacity(R * C);
    for i in 0..R {
        for j in 0..C {
            out.push(m.get(i, j).f());
        }
    }
    out
}

fn leak<S>(s: S) -> &'static mut S {
    Box::leak(Box::new(s))
}

/// The regular filter on the storage types, Joseph form (the default) or standard form.
pub struct Regular<const N: usize, const M: usize, T: Real> {
    f: mreg::BorrowedKalman<'static, N, T>,
    o: mreg::BorrowedObservation<'static, N, M, T>,
}

impl<const N: usize, const M: usize, T: Real> Regular<N, M, T> {
    /// Builds the filter for `p`; `standard` selects `CovarianceUpdate::Standard`.
    pub fn new(p: &Problem, standard: bool) -> Self {
        let mut f = leak(mreg::FilterStorage::<N, T>::new(T::zero())).build();
        let mut o = leak(mreg::ObservationStorage::<N, M, T>::new(T::zero())).build();
        f.state_transition_mut().apply(|a| fill(a, &p.F));
        f.direct_process_noise_mut().apply(|q| fill(q, &p.Q));
        fill(f.state_vector_mut(), &p.x0);
        fill(f.estimate_covariance_mut(), &p.P0);
        o.observation_matrix_mut().apply(|h| fill(h, &p.H));
        o.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        if standard {
            o.set_covariance_update(CovarianceUpdate::Standard);
        }
        Self { f, o }
    }
}

/// Shared `Filter` body for `RegularKalman` wrappers with an `f` filter field.
macro_rules! regular_common {
    () => {
        #[inline(always)]
        fn predict(&mut self) {
            let _ = self.f.predict();
        }
        fn state(&self) -> Vec<f64> {
            read(self.f.state_vector())
        }
        fn covariance(&self) -> Vec<f64> {
            read(self.f.estimate_covariance())
        }
        fn restore(&mut self, x: &[f64], p: &[f64]) {
            fill(self.f.state_vector_mut(), x);
            fill(self.f.estimate_covariance_mut(), p);
        }
    };
}

/// Writes `z` into an observation's measurement vector.
macro_rules! set_z {
    ($o:expr, $z:expr, $M:expr) => {{
        let zv = $o.measurement_vector_mut();
        for i in 0..$M {
            zv.set(i, 0, $z[i]);
        }
    }};
}

impl<const N: usize, const M: usize, T: Real> Filter for Regular<N, M, T> {
    type T = T;
    regular_common!();
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, M);
        assert!(self.f.correct(&mut self.o).is_applied());
    }
}

/// The regular filter from the `alloc` builder (boxed buffers), Joseph form.
pub struct Boxed<const N: usize, const M: usize, T: Real> {
    f: mreg::builder::KalmanFilterType<N, T>,
    o: mreg::builder::KalmanFilterObservationType<N, M, T>,
}

impl<const N: usize, const M: usize, T: Real> Boxed<N, M, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let b = mreg::builder::KalmanFilterBuilder::<N, T>::default();
        let mut f = b.build();
        let mut o = b.observations().build::<M>();
        f.state_transition_mut().apply(|a| fill(a, &p.F));
        f.direct_process_noise_mut().apply(|q| fill(q, &p.Q));
        fill(f.state_vector_mut(), &p.x0);
        fill(f.estimate_covariance_mut(), &p.P0);
        o.observation_matrix_mut().apply(|h| fill(h, &p.H));
        o.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        Self { f, o }
    }
}

impl<const N: usize, const M: usize, T: Real> Filter for Boxed<N, M, T> {
    type T = T;
    regular_common!();
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, M);
        assert!(self.f.correct(&mut self.o).is_applied());
    }
}

/// The regular filter with a lean observation and a shared `CorrectionScratch` (`correct_in`).
pub struct Lean<const N: usize, const M: usize, T: Real> {
    f: mreg::BorrowedKalman<'static, N, T>,
    o: mreg::BorrowedLeanObservation<'static, N, M, T>,
    s: &'static mut CorrectionScratch<N, M, T>,
}

impl<const N: usize, const M: usize, T: Real> Lean<N, M, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let mut f = leak(mreg::FilterStorage::<N, T>::new(T::zero())).build();
        let mut o = leak(mreg::LeanObservationStorage::<N, M, T>::new(T::zero())).build();
        let s = leak(CorrectionScratch::<N, M, T>::new(T::zero()));
        f.state_transition_mut().apply(|a| fill(a, &p.F));
        f.direct_process_noise_mut().apply(|q| fill(q, &p.Q));
        fill(f.state_vector_mut(), &p.x0);
        fill(f.estimate_covariance_mut(), &p.P0);
        o.observation_matrix_mut().apply(|h| fill(h, &p.H));
        o.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        Self { f, o, s }
    }
}

impl<const N: usize, const M: usize, T: Real> Filter for Lean<N, M, T> {
    type T = T;
    regular_common!();
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, M);
        assert!(self.f.correct_in(&mut self.o, self.s).is_applied());
    }
}

/// The UD-factored filter (Bierman/Thornton) from the builder. `W = 2·N`.
pub struct Ud<const N: usize, const M: usize, const W: usize, T: Real> {
    f: mreg::builder::UdKalmanType<N, W, T>,
    o: mreg::builder::KalmanFilterObservationType<N, M, T>,
}

impl<const N: usize, const M: usize, const W: usize, T: Real> Ud<N, M, W, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let b = mreg::builder::KalmanFilterBuilder::<N, T>::default();
        let mut f = b.build_ud::<W>();
        let mut o = b.observations().build::<M>();
        f.state_transition_mut().apply(|a| fill(a, &p.F));
        let mut q = BufferBuilder::direct_process_noise_covariance_Q::<N>().new();
        fill(&mut q, &p.Q);
        assert!(f.set_process_noise(&q));
        o.observation_matrix_mut().apply(|h| fill(h, &p.H));
        o.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        let mut w = Self { f, o };
        w.restore(&p.x0, &p.P0);
        w
    }
}

impl<const N: usize, const M: usize, const W: usize, T: Real> Filter for Ud<N, M, W, T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        self.f.predict();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, M);
        assert!(self.f.correct(&mut self.o).is_applied());
    }
    fn state(&self) -> Vec<f64> {
        let mut x = BufferBuilder::state_vector_x::<N>().new();
        let mut p = BufferBuilder::estimate_covariance_P::<N>().new();
        self.f.state(&mut x, &mut p);
        read(&x)
    }
    fn covariance(&self) -> Vec<f64> {
        let mut x = BufferBuilder::state_vector_x::<N>().new();
        let mut p = BufferBuilder::estimate_covariance_P::<N>().new();
        self.f.state(&mut x, &mut p);
        read(&p)
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        let mut xb = BufferBuilder::state_vector_x::<N>().new();
        let mut pb = BufferBuilder::estimate_covariance_P::<N>().new();
        fill(&mut xb, x);
        fill(&mut pb, p);
        assert!(self.f.set_state(&xb, &pb));
    }
}

/// The square-root covariance filter from the builder. `W = 2·N`.
pub struct SquareRoot<const N: usize, const M: usize, const W: usize, T: Real> {
    f: mreg::builder::SquareRootKalmanType<N, W, T>,
    o: mreg::builder::KalmanFilterObservationType<N, M, T>,
}

impl<const N: usize, const M: usize, const W: usize, T: Real> SquareRoot<N, M, W, T> {
    /// Builds the filter for `p`.
    pub fn new(p: &Problem) -> Self {
        let b = mreg::builder::KalmanFilterBuilder::<N, T>::default();
        let mut f = b.build_square_root::<W>();
        let mut o = b.observations().build::<M>();
        f.state_transition_mut().apply(|a| fill(a, &p.F));
        let mut q = BufferBuilder::direct_process_noise_covariance_Q::<N>().new();
        fill(&mut q, &p.Q);
        assert!(f.set_process_noise(&q));
        o.observation_matrix_mut().apply(|h| fill(h, &p.H));
        o.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        let mut w = Self { f, o };
        w.restore(&p.x0, &p.P0);
        w
    }
}

impl<const N: usize, const M: usize, const W: usize, T: Real> Filter for SquareRoot<N, M, W, T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        assert!(self.f.predict());
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, M);
        assert!(self.f.correct(&mut self.o).is_applied());
    }
    fn state(&self) -> Vec<f64> {
        let mut x = BufferBuilder::state_vector_x::<N>().new();
        let mut p = BufferBuilder::estimate_covariance_P::<N>().new();
        self.f.state(&mut x, &mut p);
        read(&x)
    }
    fn covariance(&self) -> Vec<f64> {
        let mut x = BufferBuilder::state_vector_x::<N>().new();
        let mut p = BufferBuilder::estimate_covariance_P::<N>().new();
        self.f.state(&mut x, &mut p);
        read(&p)
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        let mut xb = BufferBuilder::state_vector_x::<N>().new();
        let mut pb = BufferBuilder::estimate_covariance_P::<N>().new();
        fill(&mut xb, x);
        fill(&mut pb, p);
        assert!(self.f.set_state(&xb, &pb));
    }
}

/// The current-form fixed-gain (Luenberger) observer on storage types, with the steady-state
/// gain `K∞` from `settle_gain` on the same problem.
pub struct Observer<const N: usize, const M: usize, T: Real> {
    f: mreg::BorrowedLuenbergerObserver<'static, N, T>,
    o: mreg::BorrowedFixedGainObservation<'static, N, M, T, Current>,
    /// The settled gain, row-major `N×M`.
    pub gain: Vec<f64>,
}

impl<const N: usize, const M: usize, T: Real> Observer<N, M, T> {
    /// Settles `K∞` for `p` and builds the observer.
    pub fn new(p: &Problem) -> Self {
        let b = mreg::builder::KalmanFilterBuilder::<N, T>::default();
        let mut design = b.build();
        let mut obs = b.observations().build::<M>();
        design.state_transition_mut().apply(|a| fill(a, &p.F));
        design.direct_process_noise_mut().apply(|q| fill(q, &p.Q));
        fill(design.estimate_covariance_mut(), &p.P0);
        obs.observation_matrix_mut().apply(|h| fill(h, &p.H));
        obs.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        design
            .settle_gain(&mut obs, T::c(1e-6), 10_000)
            .expect("detectable model");
        let gain = read(obs.kalman_gain());

        let mut f = leak(mreg::ObserverStorage::<N, T>::new(T::zero())).build();
        let mut o = leak(mreg::FixedGainObservationStorage::<N, M, T, Current>::new(
            T::zero(),
        ))
        .build();
        f.state_transition_mut().apply(|a| fill(a, &p.F));
        fill(f.state_vector_mut(), &p.x0);
        o.observation_matrix_mut().apply(|h| fill(h, &p.H));
        o.set_gain(obs.kalman_gain());
        Self { f, o, gain }
    }
}

impl<const N: usize, const M: usize, T: Real> Filter for Observer<N, M, T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        let _ = self.f.predict();
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, M);
        assert!(self.f.correct(&mut self.o).is_applied());
    }
    fn state(&self) -> Vec<f64> {
        read(self.f.state_vector())
    }
    fn covariance(&self) -> Vec<f64> {
        Vec::new()
    }
    fn restore(&mut self, x: &[f64], _p: &[f64]) {
        fill(self.f.state_vector_mut(), x);
    }
}

/// The extended filter on storage types for the radar problem (`N = 4`, `M = 2`).
pub struct Extended<T: Real> {
    f: mext::BorrowedKalman<'static, 4, T>,
    o: mext::BorrowedObservation<'static, 4, 2, T>,
    dt: T,
}

impl<T: Real> Extended<T> {
    /// Builds the filter for the radar problem `p`.
    pub fn new(p: &Problem) -> Self {
        let mut f = leak(mext::FilterStorage::<4, T>::new(T::zero())).build();
        let mut o = leak(mext::ObservationStorage::<4, 2, T>::new(T::zero())).build();
        // The motion is linear, so its Jacobian is constant and set once.
        f.state_transition_jacobian_mut().apply(|a| fill(a, &p.F));
        f.direct_process_noise_mut().apply(|q| fill(q, &p.Q));
        fill(f.state_vector_mut(), &p.x0);
        fill(f.estimate_covariance_mut(), &p.P0);
        o.measurement_noise_covariance_mut()
            .apply(|r| fill(r, &p.R));
        Self {
            f,
            o,
            dt: T::c(p.dt),
        }
    }
}

impl<T: Real> Filter for Extended<T> {
    type T = T;
    #[inline(always)]
    fn predict(&mut self) {
        let dt = self.dt;
        let _ = self
            .f
            .predict_nonlinear(|x, next| cv_step(dt, x.as_slice(), next.as_mut_slice()));
    }
    #[inline(always)]
    fn correct(&mut self, z: &[T]) {
        set_z!(self.o, z, 2);
        let j = radar_jacobian(self.f.state_vector().as_slice());
        self.o.observation_jacobian_matrix_mut().apply(|h| {
            for i in 0..2 {
                for k in 0..4 {
                    h.set(i, k, j[i * 4 + k]);
                }
            }
        });
        let status = self.f.correct_nonlinear(&mut self.o, |x, obs| {
            let h = radar_h(x.as_slice());
            obs.set(0, 0, h[0]);
            obs.set(1, 0, h[1]);
        });
        assert!(status.is_applied());
    }
    fn state(&self) -> Vec<f64> {
        read(self.f.state_vector())
    }
    fn covariance(&self) -> Vec<f64> {
        read(self.f.estimate_covariance())
    }
    fn restore(&mut self, x: &[f64], p: &[f64]) {
        fill(self.f.state_vector_mut(), x);
        fill(self.f.estimate_covariance_mut(), p);
    }
}
