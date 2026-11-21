//! Correctness gate and the non-timing metrics.
//!
//! 1. Agreement: every filter runs the same predict/correct sequence as the `f64` reference;
//!    the run fails (exit code 1) if any filter leaves the tolerance.
//! 2. Heap allocations and bytes per predict+correct step (counting global allocator).
//! 3. Memory: live heap bytes after construction, and `size_of` of the filter value.
//!
//! Output: a JSON document on stdout, a human summary on stderr.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use kalman_compare::problem::{LAP, WARMUP};
use kalman_compare::reference::{FixedGainReference, Reference};
use kalman_compare::registry::{self, Ctor};
use kalman_compare::{Filter, Model, Problem, Real, conv};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static FREED_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(l.size(), Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        FREED_BYTES.fetch_add(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(new, Relaxed);
        FREED_BYTES.fetch_add(l.size(), Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn snapshot() -> (usize, usize, usize) {
    (
        ALLOCS.load(Relaxed),
        ALLOC_BYTES.load(Relaxed),
        FREED_BYTES.load(Relaxed),
    )
}

/// Relative error `|a − b| / (1 + |b|)`, maximised over the entries.
fn err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
        .fold(0.0, f64::max)
}

/// Tolerances on the state and covariance error for `T`.
fn tolerance<T: Real>() -> (f64, f64) {
    if T::NAME == "f64" {
        (1e-9, 1e-9)
    } else {
        (2e-3, 2e-3)
    }
}

fn check<T: Real>(p: &Problem, ctors: &[Ctor<T>], out: &mut Vec<String>) -> bool {
    let mut ok = true;
    let (tol_x, tol_p) = tolerance::<T>();
    let steps = WARMUP + LAP;
    for (name, ctor) in ctors {
        let mut reference = Reference::new(p);
        let mut f = ctor(p);
        let (mut ex, mut ep) = (0.0f64, 0.0f64);
        for k in 0..steps {
            reference.predict();
            f.predict();
            let z = &p.zs[k];
            reference.correct(z);
            f.correct(&conv::<T>(z));
            ex = ex.max(err(&f.state(), &reference.state()));
            ep = ep.max(err(&f.covariance(), &reference.covariance()));
        }
        let pass = ex <= tol_x && ep <= tol_p;
        ok &= pass;

        // Allocations per step, after the agreement run warmed every buffer up.
        let z: Vec<T> = conv(&p.zs[WARMUP]);
        let before = snapshot();
        const STEPS: usize = 100;
        for _ in 0..STEPS {
            f.predict();
            f.correct(&z);
        }
        let after = snapshot();
        let allocs = (after.0 - before.0) as f64 / STEPS as f64;
        let bytes = (after.1 - before.1) as f64 / STEPS as f64;

        // Live heap after construction.
        let b = snapshot();
        let g = ctor(p);
        let a = snapshot();
        let live = (a.1 - b.1) as isize - (a.2 - b.2) as isize;
        drop(g);

        eprintln!(
            "{:<14} {:<4} {:<36} max_err_x={ex:.2e} max_err_P={ep:.2e} {} allocs/step={allocs:.1} bytes/step={bytes:.0} live_heap={live}",
            p.name,
            T::NAME,
            name,
            if pass { "PASS" } else { "FAIL" },
        );
        out.push(format!(
            "{{\"problem\":\"{}\",\"type\":\"{}\",\"filter\":\"{}\",\"err_x\":{ex:e},\"err_P\":{ep:e},\"pass\":{pass},\"allocs_per_step\":{allocs},\"bytes_per_step\":{bytes},\"live_heap_after_build\":{live}}}",
            p.name,
            T::NAME,
            name
        ));
    }

    // The fixed-gain observer against a fixed-gain recursion with the same gain, and its
    // gain against the reference filter's converged gain.
    if p.model == Model::Linear {
        let (mut f, gain) = registry::observer::<T>(p);
        let mut r = FixedGainReference {
            p: p.clone(),
            x: p.x0.clone(),
            K: gain.clone(),
        };
        let mut ex = 0.0f64;
        for k in 0..steps {
            f.predict();
            f.correct(&conv::<T>(&p.zs[k]));
            r.step(&p.zs[k]);
            ex = ex.max(err(&f.state(), &r.x));
        }
        let mut converged = Reference::new(p);
        for k in 0..5000 {
            converged.predict();
            converged.correct(&p.zs[k % p.zs.len()]);
        }
        let eg = err(&gain, &converged.K);
        let (tol_x, _) = tolerance::<T>();
        // settle_gain stops once successive gains change by < 1e-6, so its gain is accurate to ~1e-5.
        let pass = ex <= tol_x && eg <= 1e-5;
        ok &= pass;
        let before = snapshot();
        let z: Vec<T> = conv(&p.zs[WARMUP]);
        for _ in 0..100 {
            f.predict();
            f.correct(&z);
        }
        let allocs = (snapshot().0 - before.0) as f64 / 100.0;
        eprintln!(
            "{:<14} {:<4} {:<36} max_err_x={ex:.2e} gain_err={eg:.2e} {} allocs/step={allocs:.1}",
            p.name,
            T::NAME,
            "minikalman/observer_fixed_gain",
            if pass { "PASS" } else { "FAIL" }
        );
        out.push(format!(
            "{{\"problem\":\"{}\",\"type\":\"{}\",\"filter\":\"minikalman/observer_fixed_gain\",\"err_x\":{ex:e},\"err_gain\":{eg:e},\"pass\":{pass},\"allocs_per_step\":{allocs}}}",
            p.name,
            T::NAME
        ));
    }
    ok
}

fn main() {
    sizes();
    let mut out = Vec::new();
    let mut ok = true;
    for p in registry::problems() {
        ok &= check::<f64>(&p, &registry::for_problem::<f64>(&p), &mut out);
        ok &= check::<f32>(&p, &registry::for_problem::<f32>(&p), &mut out);
    }
    println!("[\n{}\n]", out.join(",\n"));
    if !ok {
        eprintln!("AGREEMENT CHECK FAILED");
        std::process::exit(1);
    }
    eprintln!("all filters agree with the reference");
}

/// `size_of` of each crate's filter values for the 4-state, 2-measurement problem.
#[allow(dead_code)]
pub fn sizes() {
    use core::mem::size_of;
    use minikalman::regular as r;
    macro_rules! p {
        ($t:ty) => {
            eprintln!("size_of {:<70} {:>5} B", stringify!($t), size_of::<$t>());
        };
    }
    p!(r::FilterStorage<4, f32>);
    p!(r::ObservationStorage<4, 2, f32>);
    p!(r::LeanObservationStorage<4, 2, f32>);
    p!(r::CorrectionScratch<4, 2, f32>);
    p!(r::ObserverStorage<4, f32>);
    p!(r::FixedGainObservationStorage<4, 2, f32, minikalman::prelude::Current>);
    p!(minikalman::extended::FilterStorage<4, f32>);
    p!(minikalman::extended::ObservationStorage<4, 2, f32>);
    p!(r::BorrowedKalman<'static, 4, f32>);
    p!(r::BorrowedObservation<'static, 4, 2, f32>);
    p!(r::builder::KalmanFilterType<4, f32>);
    p!(kfilter::kalman::Kalman1MLinearNoInput<f32, 4, 2>);
    p!(smelly::SKalmanFilter<f32, 4, 2>);
    p!(smelly::SExtendedKalmanFilter<f32, 4, 2>);
    p!(smelly::KalmanFilter<f32>);
    p!(kalman_filters::filter::StaticKalmanFilter<f32, 2, 4>);
    p!(kalman_filters::KalmanFilter<f32>);
}
