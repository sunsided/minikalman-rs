//! Every filter under test for a problem, behind `dyn Filter` (for the checks, not timing).

use crate::{
    Filter, Problem, Real, kalman_filters_impl as kfs, kfilter_impl as kf, minikalman_impl as mk,
    smelly_impl as sm,
};

/// A named constructor of one filter.
pub type Ctor<T> = (&'static str, fn(&Problem) -> Box<dyn Filter<T = T>>);

/// Constructors of every linear filter for `N` states and `M` measurements.
/// `W = 2·N` is the UD / square-root workspace width.
pub fn linear<const N: usize, const M: usize, const W: usize, T: Real>() -> Vec<Ctor<T>>
where
    na35::Const<M>: na35::DimMin<na35::Const<M>, Output = na35::Const<M>>,
{
    vec![
        ("minikalman/storage_joseph", |p| {
            Box::new(mk::Regular::<N, M, T>::new(p, false))
        }),
        ("minikalman/storage_standard", |p| {
            Box::new(mk::Regular::<N, M, T>::new(p, true))
        }),
        ("minikalman/boxed_joseph", |p| {
            Box::new(mk::Boxed::<N, M, T>::new(p))
        }),
        ("minikalman/lean_scratch_joseph", |p| {
            Box::new(mk::Lean::<N, M, T>::new(p))
        }),
        ("minikalman/ud", |p| Box::new(mk::Ud::<N, M, W, T>::new(p))),
        ("minikalman/square_root", |p| {
            Box::new(mk::SquareRoot::<N, M, W, T>::new(p))
        }),
        ("kfilter/kalman1m", |p| {
            Box::new(kf::Linear::<N, M, T>::new(p))
        }),
        ("smelly/static", |p| Box::new(sm::Static::<N, M, T>::new(p))),
        ("smelly/dynamic", |p| Box::new(sm::Dynamic::<T>::new(p))),
        ("kalman_filters/dynamic", |p| {
            Box::new(kfs::Dynamic::<T>::new(p))
        }),
        ("kalman_filters/static", |p| {
            Box::new(kfs::Static::<N, M, T>::new(p))
        }),
    ]
}

/// Constructors of every extended filter for the radar problem.
pub fn radar<T: Real>() -> Vec<Ctor<T>> {
    vec![
        ("minikalman/extended_joseph", |p| {
            Box::new(mk::Extended::<T>::new(p))
        }),
        ("kfilter/kalman_custom_measurement", |p| {
            Box::new(kf::Radar::<T>::new(p))
        }),
        ("smelly/static_ekf", |p| Box::new(sm::Radar::<T>::new(p))),
        ("kalman_filters/ekf", |p| Box::new(kfs::Radar::<T>::new(p))),
    ]
}

/// All constructors for `p`.
pub fn for_problem<T: Real>(p: &Problem) -> Vec<Ctor<T>> {
    match p.name {
        "kf_2x1" => linear::<2, 1, 4, T>(),
        "kf_4x2" => linear::<4, 2, 8, T>(),
        "kf_6x3" => linear::<6, 3, 12, T>(),
        "ekf_radar_4x2" => radar::<T>(),
        other => panic!("unknown problem {other}"),
    }
}

/// The fixed-gain observer for `p` (linear problems only), with its settled gain.
pub fn observer<T: Real>(p: &Problem) -> (Box<dyn Filter<T = T>>, Vec<f64>) {
    fn make<const N: usize, const M: usize, T: Real>(
        p: &Problem,
    ) -> (Box<dyn Filter<T = T>>, Vec<f64>) {
        let o = mk::Observer::<N, M, T>::new(p);
        let g = o.gain.clone();
        (Box::new(o), g)
    }
    match p.name {
        "kf_2x1" => make::<2, 1, T>(p),
        "kf_4x2" => make::<4, 2, T>(p),
        "kf_6x3" => make::<6, 3, T>(p),
        other => panic!("no observer for {other}"),
    }
}

/// The problems: three linear trackers and the radar.
pub fn problems() -> Vec<Problem> {
    vec![
        Problem::linear(1),
        Problem::linear(2),
        Problem::linear(3),
        Problem::radar(),
    ]
}
