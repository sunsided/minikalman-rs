//! Criterion timings of predict, correct and a full predict+correct step.
#![allow(non_snake_case)]
//!
//! Every filter is driven through the same loop: the filter starts from the reference
//! filter's state after `WARMUP` steps and is restored to it every `LAP` iterations, the
//! measurements cycle through the `LAP` measurements after the warm-up, and the filter and
//! each measurement pass through `black_box`. The restore is inside the timed loop for every
//! filter (amortised over `LAP` iterations).

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use kalman_compare::problem::{LAP, WARMUP};
use kalman_compare::reference::Reference;
use kalman_compare::{
    Filter, Problem, Real, conv, kalman_filters_impl as kfs, kfilter_impl as kf,
    minikalman_impl as mk, smelly_impl as sm,
};

#[derive(Clone, Copy)]
enum Mode {
    Predict,
    Correct,
    Step,
}

/// The reference state after the warm-up, and the measurements after it.
fn snapshot<T: Real>(p: &Problem) -> (Vec<f64>, Vec<f64>, Vec<Vec<T>>) {
    let mut r = Reference::new(p);
    for z in &p.zs[..WARMUP] {
        r.predict();
        r.correct(z);
    }
    let zs = p.zs[WARMUP..WARMUP + LAP]
        .iter()
        .map(|z| conv::<T>(z))
        .collect();
    (r.x, r.P, zs)
}

fn run<F: Filter>(
    g: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    p: &Problem,
    make: impl Fn(&Problem) -> F,
) {
    let (x, P, zs) = snapshot::<F::T>(p);
    for (mode, label) in [
        (Mode::Predict, "predict"),
        (Mode::Correct, "correct"),
        (Mode::Step, "step"),
    ] {
        g.bench_function(format!("{name}/{label}"), |b| {
            b.iter_custom(|iters| {
                let mut f = black_box(make(p));
                f.restore(&x, &P);
                let start = Instant::now();
                for i in 0..iters as usize {
                    let k = i % LAP;
                    if k == 0 {
                        f.restore(&x, &P);
                    }
                    match mode {
                        Mode::Predict => f.predict(),
                        Mode::Correct => f.correct(black_box(&zs[k])),
                        Mode::Step => {
                            f.predict();
                            f.correct(black_box(&zs[k]));
                        }
                    }
                    black_box(&mut f);
                }
                start.elapsed()
            })
        });
    }
}

fn linear<const N: usize, const M: usize, const W: usize, T: Real>(c: &mut Criterion, p: &Problem)
where
    na35::Const<M>: na35::DimMin<na35::Const<M>, Output = na35::Const<M>>,
{
    let mut g = c.benchmark_group(format!("{}/{}", p.name, T::NAME));
    run(&mut g, "minikalman/storage_joseph", p, |p| {
        mk::Regular::<N, M, T>::new(p, false)
    });
    run(&mut g, "minikalman/storage_standard", p, |p| {
        mk::Regular::<N, M, T>::new(p, true)
    });
    run(
        &mut g,
        "minikalman/boxed_joseph",
        p,
        mk::Boxed::<N, M, T>::new,
    );
    run(
        &mut g,
        "minikalman/lean_scratch_joseph",
        p,
        mk::Lean::<N, M, T>::new,
    );
    run(&mut g, "minikalman/ud", p, mk::Ud::<N, M, W, T>::new);
    run(
        &mut g,
        "minikalman/square_root",
        p,
        mk::SquareRoot::<N, M, W, T>::new,
    );
    run(
        &mut g,
        "minikalman/observer_fixed_gain",
        p,
        mk::Observer::<N, M, T>::new,
    );
    run(&mut g, "kfilter/kalman1m", p, kf::Linear::<N, M, T>::new);
    run(&mut g, "smelly/static", p, sm::Static::<N, M, T>::new);
    run(&mut g, "smelly/dynamic", p, sm::Dynamic::<T>::new);
    run(
        &mut g,
        "kalman_filters/static",
        p,
        kfs::Static::<N, M, T>::new,
    );
    run(&mut g, "kalman_filters/dynamic", p, kfs::Dynamic::<T>::new);
    g.finish();
}

fn radar<T: Real>(c: &mut Criterion, p: &Problem) {
    let mut g = c.benchmark_group(format!("{}/{}", p.name, T::NAME));
    run(
        &mut g,
        "minikalman/extended_joseph",
        p,
        mk::Extended::<T>::new,
    );
    run(
        &mut g,
        "kfilter/kalman_custom_measurement",
        p,
        kf::Radar::<T>::new,
    );
    run(&mut g, "smelly/static_ekf", p, sm::Radar::<T>::new);
    run(&mut g, "kalman_filters/ekf", p, kfs::Radar::<T>::new);
    g.finish();
}

fn all(c: &mut Criterion) {
    let (p2, p4, p6, pr) = (
        Problem::linear(1),
        Problem::linear(2),
        Problem::linear(3),
        Problem::radar(),
    );
    linear::<2, 1, 4, f32>(c, &p2);
    linear::<2, 1, 4, f64>(c, &p2);
    linear::<4, 2, 8, f32>(c, &p4);
    linear::<4, 2, 8, f64>(c, &p4);
    linear::<6, 3, 12, f32>(c, &p6);
    linear::<6, 3, 12, f64>(c, &p6);
    radar::<f32>(c, &pr);
    radar::<f64>(c, &pr);
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_millis(800))
        .sample_size(40)
        .without_plots()
}

criterion_group! { name = benches; config = config(); targets = all }
criterion_main!(benches);
