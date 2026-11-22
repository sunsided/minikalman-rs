//! Per-operation instruction counts of minikalman on Cortex-M, measured in Renode.
//!
//! Each measured operation runs between a call to [`mk_begin`] and a call to [`mk_end`].
//! `renode/bench.py` hooks both symbols and logs the number of instructions the emulated core
//! executed in between, together with the operation's name and its status. Setup (building
//! the filters, writing the model, warm-up steps, generating measurements) runs outside the
//! markers. The filters run a realistic sequence (predict, then correct with a fresh
//! measurement) and every iteration is bracketed separately, so data-dependent paths (soft
//! float on the M0+) see the same values as in a real run.
#![no_std]
#![no_main]
// The navigation-only binaries (`nav`, `nav-libm`) leave the generic workloads unused.
#![cfg_attr(
    not(any(feature = "f32", feature = "i16f16", feature = "i32f32")),
    allow(unused)
)]

#[cfg(any(feature = "nav", feature = "nav-libm"))]
mod nav;
#[cfg(any(feature = "nav", feature = "nav-libm"))]
mod nav_bench;

use core::hint::black_box;
use cortex_m_rt::entry;
#[cfg(feature = "i16f16")]
use fixed::types::I16F16;
#[cfg(feature = "i32f32")]
use fixed::types::I32F32;
#[cfg(feature = "f32")]
use minikalman::extended as mext;
use minikalman::prelude::*;
use minikalman::regular::{self as mreg, CovarianceUpdate};
use minikalman::square_root::SquareRootKalman;
use minikalman::ud::UdKalman;

/// Unmeasured steps before the measured ones, so `P` is off its initial value.
const WARMUP: usize = 8;
/// Measured iterations per operation.
const ITERATIONS: usize = 16;
/// Time step of every model.
const DT: f32 = 0.1;

/// Start of a measured section; hooked by `renode/bench.py`.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn mk_begin() {
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// End of a measured section named by the UTF-8 bytes at `name`; `ok` is the op's status.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn mk_end(name: *const u8, len: usize, ok: bool) {
    black_box((name, len, ok));
}

/// End of the run; the hook writes the results and quits Renode.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn mk_done() -> ! {
    loop {
        black_box(());
    }
}

/// Reached on panic; the hook reports a failure and quits Renode.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn mk_fail() -> ! {
    // A body unlike `mk_done`'s, or LTO folds both into one address and a panic reads as DONE.
    loop {
        black_box(0xFA11_u32);
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    mk_fail()
}

/// Runs `f` between the markers and reports it as `name`.
#[inline(always)]
fn measure(name: &'static str, f: impl FnOnce() -> bool) {
    mk_begin();
    let ok = f();
    mk_end(name.as_ptr(), name.len(), ok);
}

/// Converts an `f32` model constant into the filter's element type.
fn c<T: MatrixDataType>(v: f32) -> T {
    T::from_f32(v).unwrap()
}

/// Writes `f(row, column)` into every entry of `m`.
fn fill<const R: usize, const C: usize, T: MatrixDataType>(
    m: &mut impl MatrixMut<R, C, T>,
    f: impl Fn(usize, usize) -> f32,
) {
    for i in 0..R {
        for j in 0..C {
            m.set(i, j, c(f(i, j)));
        }
    }
}

/// A deterministic uniform sequence in `[-0.5, 0.5)` (xorshift32).
struct Noise(u32);

impl Noise {
    /// Next sample.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }
}

/// `F` of a constant-velocity model with `n / 2` dimensions, state `[p…, v…]`.
fn cv_f(n: usize, i: usize, j: usize) -> f32 {
    let d = n / 2;
    if i == j {
        1.0
    } else if i < d && j == i + d {
        DT
    } else {
        0.0
    }
}

/// `Q` of a constant-velocity model (continuous white-noise acceleration, intensity 0.5).
fn cv_q(n: usize, i: usize, j: usize) -> f32 {
    let (d, q) = (n / 2, 0.5);
    let (pi, pj) = (i < d, j < d);
    if i % d != j % d {
        0.0
    } else if pi && pj {
        q * DT * DT * DT / 3.0
    } else if pi != pj {
        q * DT * DT / 2.0
    } else {
        q * DT
    }
}

/// `H` measuring the positions.
fn cv_h(i: usize, j: usize) -> f32 {
    if i == j { 1.0 } else { 0.0 }
}

/// `R` with mild correlations.
fn cv_r(i: usize, j: usize) -> f32 {
    if i == j { 0.25 + 0.05 * i as f32 } else { 0.05 }
}

/// `P₀ = 4·I`.
fn cv_p0(i: usize, j: usize) -> f32 {
    if i == j { 4.0 } else { 0.0 }
}

/// `x₀`: positions off the truth by 0.5, velocities zero.
fn cv_x0(n: usize, i: usize) -> f32 {
    if i < n / 2 { 1.5 + i as f32 } else { 0.0 }
}

/// Position measurement `i` at step `k` of the constant-velocity truth, with noise.
fn cv_z(i: usize, k: usize, noise: &mut Noise) -> f32 {
    let (p, v) = (1.0 + i as f32, 0.5 - 0.2 * i as f32);
    p + v * DT * k as f32 + noise.next()
}

/// Sets up a regular filter and observation for the constant-velocity problem.
fn setup_regular<const N: usize, const M: usize, T: MatrixDataType>(
    f: &mut mreg::BorrowedKalman<'_, N, T>,
    o: &mut mreg::BorrowedObservation<'_, N, M, T>,
    standard: bool,
) {
    f.state_transition_mut()
        .apply(|a| fill(a, |i, j| cv_f(N, i, j)));
    f.direct_process_noise_mut()
        .apply(|q| fill(q, |i, j| cv_q(N, i, j)));
    fill(f.state_vector_mut(), |i, _| cv_x0(N, i));
    fill(f.estimate_covariance_mut(), cv_p0);
    o.observation_matrix_mut().apply(|h| fill(h, cv_h));
    o.measurement_noise_covariance_mut()
        .apply(|r| fill(r, cv_r));
    if standard {
        o.set_covariance_update(CovarianceUpdate::Standard);
    }
}

/// `predict` of the regular filter (its own symbol for the objdump cross-check).
#[inline(never)]
fn kf_predict<const N: usize, T: MatrixDataType>(f: &mut mreg::BorrowedKalman<'_, N, T>) -> bool {
    let _ = f.predict();
    true
}

/// `correct` of the regular filter (its own symbol for the objdump cross-check).
#[inline(never)]
fn kf_correct<const N: usize, const M: usize, T: MatrixDataType>(
    f: &mut mreg::BorrowedKalman<'_, N, T>,
    o: &mut mreg::BorrowedObservation<'_, N, M, T>,
) -> bool {
    f.correct(o).is_applied()
}

/// Measures every linear filter variant of one problem size and element type.
///
/// A macro rather than a generic function: the UD and square-root workspaces are
/// `N × 2N` arrays, which need concrete sizes on stable Rust.
macro_rules! linear {
    ($fn_name:ident, $label:literal, $n:literal, $m:literal, $t:ty) => {
        #[inline(never)]
        fn $fn_name() {
            type T = $t;
            const N: usize = $n;
            const M: usize = $m;
            macro_rules! name {
                ($op:literal) => {
                    concat!($label, "/", $op)
                };
            }

            // Joseph form (the default): predict and correct measured separately.
            let mut fs_j = mreg::FilterStorage::<N, T>::new(c(0.0));
            let mut os_j = mreg::ObservationStorage::<N, M, T>::new(c(0.0));
            let (mut f_j, mut o_j) = (fs_j.build(), os_j.build());
            setup_regular(&mut f_j, &mut o_j, false);
            // Standard form: correct measured, predict not.
            let mut fs_s = mreg::FilterStorage::<N, T>::new(c(0.0));
            let mut os_s = mreg::ObservationStorage::<N, M, T>::new(c(0.0));
            let (mut f_s, mut o_s) = (fs_s.build(), os_s.build());
            setup_regular(&mut f_s, &mut o_s, true);
            // Joseph form: predict + correct as one step.
            let mut fs_t = mreg::FilterStorage::<N, T>::new(c(0.0));
            let mut os_t = mreg::ObservationStorage::<N, M, T>::new(c(0.0));
            let (mut f_t, mut o_t) = (fs_t.build(), os_t.build());
            setup_regular(&mut f_t, &mut o_t, false);

            // UD filter.
            impl_buffer_A!(mut ud_a, N, T, c(0.0));
            impl_buffer_x!(mut ud_x, N, T, c(0.0));
            impl_buffer_P!(mut ud_ud, N, T, c(0.0));
            impl_buffer_Q_direct!(mut ud_q, N, T, c(0.0));
            impl_buffer_ud_W!(mut ud_w, N, T, c(0.0));
            impl_buffer_ud_D!(mut ud_d, N, T, c(0.0));
            let mut ud = UdKalman::<N, { 2 * N }, T, _, _, _, _, _, _>::new(
                ud_a, ud_x, ud_ud, ud_q, ud_w, ud_d,
            );
            let mut os_u = mreg::ObservationStorage::<N, M, T>::new(c(0.0));
            let mut o_u = os_u.build();
            o_u.observation_matrix_mut().apply(|h| fill(h, cv_h));
            o_u.measurement_noise_covariance_mut()
                .apply(|r| fill(r, cv_r));

            // Square-root filter.
            impl_buffer_A!(mut sr_a, N, T, c(0.0));
            impl_buffer_x!(mut sr_x, N, T, c(0.0));
            impl_buffer_P!(mut sr_s, N, T, c(0.0));
            impl_buffer_Q_direct!(mut sr_q, N, T, c(0.0));
            impl_buffer_temp_prearray!(mut sr_pre, N, T, c(0.0));
            let mut sr = SquareRootKalman::<N, { 2 * N }, T, _, _, _, _, _>::new(
                sr_a, sr_x, sr_s, sr_q, sr_pre,
            );
            let mut os_r = mreg::ObservationStorage::<N, M, T>::new(c(0.0));
            let mut o_r = os_r.build();
            o_r.observation_matrix_mut().apply(|h| fill(h, cv_h));
            o_r.measurement_noise_covariance_mut()
                .apply(|r| fill(r, cv_r));
            {
                impl_buffer_x!(mut x0, N, T, c(0.0));
                impl_buffer_P!(mut p0, N, T, c(0.0));
                impl_buffer_Q_direct!(mut q, N, T, c(0.0));
                fill(&mut x0, |i, _| cv_x0(N, i));
                fill(&mut p0, cv_p0);
                fill(&mut q, |i, j| cv_q(N, i, j));
                ud.state_transition_mut()
                    .apply(|a| fill(a, |i, j| cv_f(N, i, j)));
                sr.state_transition_mut()
                    .apply(|a| fill(a, |i, j| cv_f(N, i, j)));
                assert!(ud.set_process_noise(&q) && ud.set_state(&x0, &p0));
                assert!(sr.set_process_noise(&q) && sr.set_state(&x0, &p0));
            }

            // Fixed-gain observer, with the gain of a design filter after the warm-up.
            let mut fs_d = mreg::FilterStorage::<N, T>::new(c(0.0));
            let mut os_d = mreg::ObservationStorage::<N, M, T>::new(c(0.0));
            let (mut f_d, mut o_d) = (fs_d.build(), os_d.build());
            setup_regular(&mut f_d, &mut o_d, false);
            let mut obs_s = mreg::ObserverStorage::<N, T>::new(c(0.0));
            let mut obs_os = mreg::FixedGainObservationStorage::<N, M, T, Current>::new(c(0.0));
            let (mut obs, mut obs_o) = (obs_s.build(), obs_os.build());
            obs.state_transition_mut()
                .apply(|a| fill(a, |i, j| cv_f(N, i, j)));
            fill(obs.state_vector_mut(), |i, _| cv_x0(N, i));
            obs_o.observation_matrix_mut().apply(|h| fill(h, cv_h));

            let mut noise = Noise(0x9E37_79B9 ^ N as u32);
            for k in 0..WARMUP {
                let mut zs = [0.0f32; M];
                for z in zs.iter_mut().enumerate() {
                    *z.1 = cv_z(z.0, k, &mut noise);
                }
                macro_rules! put {
                    ($o:expr) => {
                        $o.measurement_vector_mut().apply(|z| fill(z, |i, _| zs[i]))
                    };
                }
                let _ = f_j.predict();
                put!(o_j);
                assert!(f_j.correct(&mut o_j).is_applied());
                let _ = f_s.predict();
                put!(o_s);
                assert!(f_s.correct(&mut o_s).is_applied());
                let _ = f_t.predict();
                put!(o_t);
                assert!(f_t.correct(&mut o_t).is_applied());
                let _ = f_d.predict();
                put!(o_d);
                assert!(f_d.correct(&mut o_d).is_applied());
                ud.predict();
                put!(o_u);
                assert!(ud.correct(&mut o_u).is_applied());
                assert!(sr.predict());
                put!(o_r);
                assert!(sr.correct(&mut o_r).is_applied());
                let _ = obs.predict();
                put!(obs_o);
                let _ = obs.correct(&mut obs_o);
            }
            obs_o.set_gain(o_d.kalman_gain());

            // Hide the filters from the optimizer, so nothing is specialized on the model.
            let (f_j, o_j) = black_box((&mut f_j, &mut o_j));
            let (f_s, o_s) = black_box((&mut f_s, &mut o_s));
            let (f_t, o_t) = black_box((&mut f_t, &mut o_t));
            let (ud, o_u) = black_box((&mut ud, &mut o_u));
            let (sr, o_r) = black_box((&mut sr, &mut o_r));
            let (obs, obs_o) = black_box((&mut obs, &mut obs_o));

            for k in WARMUP..WARMUP + ITERATIONS {
                let mut zs = [0.0f32; M];
                for z in zs.iter_mut().enumerate() {
                    *z.1 = cv_z(z.0, k, &mut noise);
                }
                macro_rules! put {
                    ($o:expr) => {
                        $o.measurement_vector_mut().apply(|z| fill(z, |i, _| zs[i]))
                    };
                }
                measure(name!("predict"), || kf_predict(f_j));
                put!(o_j);
                measure(name!("correct_joseph"), || kf_correct(f_j, o_j));

                let _ = f_s.predict();
                put!(o_s);
                measure(name!("correct_standard"), || kf_correct(f_s, o_s));

                put!(o_t);
                measure(name!("step_joseph"), || {
                    let _ = f_t.predict();
                    f_t.correct(o_t).is_applied()
                });

                put!(obs_o);
                measure(name!("observer_step"), || {
                    let _ = obs.predict();
                    obs.correct(obs_o).is_applied()
                });

                measure(name!("ud_predict"), || {
                    ud.predict();
                    true
                });
                put!(o_u);
                measure(name!("ud_correct"), || ud.correct(o_u).is_applied());

                measure(name!("sqrt_predict"), || sr.predict());
                put!(o_r);
                measure(name!("sqrt_correct"), || sr.correct(o_r).is_applied());
            }
        }
    };
}

#[cfg(feature = "f32")]
linear!(f32_2x1, "f32/kf_2x1", 2, 1, f32);
#[cfg(feature = "f32")]
linear!(f32_4x2, "f32/kf_4x2", 4, 2, f32);
#[cfg(feature = "f32")]
linear!(f32_6x3, "f32/kf_6x3", 6, 3, f32);
#[cfg(feature = "i16f16")]
linear!(i16f16_2x1, "I16F16/kf_2x1", 2, 1, I16F16);
#[cfg(feature = "i16f16")]
linear!(i16f16_4x2, "I16F16/kf_4x2", 4, 2, I16F16);
#[cfg(feature = "i16f16")]
linear!(i16f16_6x3, "I16F16/kf_6x3", 6, 3, I16F16);
#[cfg(feature = "i32f32")]
linear!(i32f32_2x1, "I32F32/kf_2x1", 2, 1, I32F32);
#[cfg(feature = "i32f32")]
linear!(i32f32_4x2, "I32F32/kf_4x2", 4, 2, I32F32);
#[cfg(feature = "i32f32")]
linear!(i32f32_6x3, "I32F32/kf_6x3", 6, 3, I32F32);

/// The 4-state, 2-measurement range–bearing radar EKF (`f32`), as in `benches/compare`.
#[cfg(feature = "f32")]
#[inline(never)]
fn ekf_radar() {
    let mut fs = mext::FilterStorage::<4, f32>::new(0.0);
    let mut os = mext::ObservationStorage::<4, 2, f32>::new(0.0);
    let (mut f, mut o) = (fs.build(), os.build());
    // The motion is linear, so its Jacobian is constant and set once.
    f.state_transition_jacobian_mut()
        .apply(|a| fill(a, |i, j| cv_f(4, i, j)));
    f.direct_process_noise_mut()
        .apply(|q| fill(q, |i, j| cv_q(4, i, j)));
    fill(f.state_vector_mut(), |i, _| [10.5, 4.5, 0.0, 0.0][i]);
    fill(
        f.estimate_covariance_mut(),
        |i, j| if i == j { 1.0 } else { 0.0 },
    );
    o.measurement_noise_covariance_mut()
        .apply(|r| fill(r, |i, j| [[0.01, 0.0], [0.0, 0.0001]][i][j]));

    let mut noise = Noise(0xD1B5_4A32);
    let mut z = |k: usize| {
        let t = DT * k as f32;
        let (px, py) = (10.0 + t, 5.0 + 0.5 * t);
        let r = (px * px + py * py).square_root();
        [
            r + 0.1 * noise.next(),
            libm::atan2f(py, px) + 0.01 * noise.next(),
        ]
    };

    /// Predict with the constant-velocity motion as the nonlinear transition.
    fn predict(f: &mut mext::BorrowedKalman<'_, 4, f32>) -> bool {
        f.predict_nonlinear(|x, next| {
            for i in 0..2 {
                next[i] = x[i] + DT * x[i + 2];
                next[i + 2] = x[i + 2];
            }
        });
        true
    }
    /// Correct with the range–bearing model: Jacobian at the prior, then `h(x)`.
    fn correct(
        f: &mut mext::BorrowedKalman<'_, 4, f32>,
        o: &mut mext::BorrowedObservation<'_, 4, 2, f32>,
    ) -> bool {
        let (px, py) = (f.state_vector().get(0, 0), f.state_vector().get(1, 0));
        let r2 = px * px + py * py;
        let r = r2.square_root();
        o.observation_jacobian_matrix_mut().apply(|h| {
            h.set(0, 0, px / r);
            h.set(0, 1, py / r);
            h.set(1, 0, -py / r2);
            h.set(1, 1, px / r2);
        });
        f.correct_nonlinear(o, |x, obs| {
            let (px, py) = (x.get(0, 0), x.get(1, 0));
            obs.set(0, 0, (px * px + py * py).square_root());
            obs.set(1, 0, libm::atan2f(py, px));
        })
        .is_applied()
    }

    for k in 0..WARMUP {
        let zk = z(k);
        predict(&mut f);
        o.measurement_vector_mut().apply(|m| fill(m, |i, _| zk[i]));
        assert!(correct(&mut f, &mut o));
    }
    let (f, o) = black_box((&mut f, &mut o));
    for k in WARMUP..WARMUP + ITERATIONS {
        let zk = z(k);
        measure("f32/ekf_radar_4x2/predict", || predict(f));
        o.measurement_vector_mut().apply(|m| fill(m, |i, _| zk[i]));
        measure("f32/ekf_radar_4x2/correct_joseph", || correct(f, o));
    }
}

/// The cost of single arithmetic operations, to read the filter counts against.
///
/// Each one loads its operands from and stores its result to the stack (`black_box`); `copy`
/// is that overhead alone, to subtract.
#[inline(never)]
fn primitives() {
    macro_rules! prims {
        ($t:ty, $label:literal) => {{
            let (a, b): ($t, $t) = black_box((c(1.2345), c(0.678)));
            macro_rules! prim {
                ($op:literal, $e:expr) => {
                    measure(concat!($label, "/prim/", $op), || {
                        black_box($e);
                        true
                    })
                };
            }
            for _ in 0..4 {
                prim!("copy", black_box(a));
                prim!("add", black_box(a) + black_box(b));
                prim!("mul", black_box(a) * black_box(b));
                prim!("div", black_box(a) / black_box(b));
                prim!("sqrt", black_box(a).square_root());
            }
        }};
    }
    #[cfg(feature = "f32")]
    prims!(f32, "f32");
    #[cfg(feature = "i16f16")]
    prims!(I16F16, "I16F16");
    #[cfg(feature = "i32f32")]
    prims!(I32F32, "I32F32");
}

/// Fill pattern of the unused stack, see [`paint_stack`].
const STACK_PAINT: u32 = 0xA5A5_A5A5;

unsafe extern "C" {
    /// End of `.bss`/`.data` (cortex-m-rt): the stack must never grow below this.
    static mut __sheap: u32;
}

/// Fills the free RAM between the statics and the current stack pointer (minus a margin)
/// with [`STACK_PAINT`], so [`report_stack`] can find the deepest stack use.
#[inline(never)]
fn paint_stack() {
    let sp = black_box(0u8);
    let top = core::ptr::addr_of!(sp) as usize - 256;
    let mut p = core::ptr::addr_of_mut!(__sheap);
    while (p as usize) < top {
        // SAFETY: the words between the statics and the live stack are unused here.
        unsafe {
            p.write_volatile(STACK_PAINT);
            p = p.add(1);
        }
    }
}

/// Reports how many bytes above the statics the stack never wrote, so the run can tell
/// whether the stack reached the statics (which would corrupt them silently).
#[inline(never)]
fn report_stack() {
    let mut p = core::ptr::addr_of_mut!(__sheap);
    let mut free = 0usize;
    // SAFETY: reads within RAM between the statics and the stack.
    while unsafe { p.read_volatile() } == STACK_PAINT {
        free += 4;
        p = unsafe { p.add(1) };
    }
    mk_stack(free);
}

/// Reports the number of stack bytes never written; hooked by `renode/bench.py`.
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn mk_stack(free_bytes: usize) {
    black_box(free_bytes);
}

/// Fails the run unless minikalman's `f32` square root matches `libm::sqrtf` bit for bit
/// (NaN for NaN) on the edge cases and on 65 552 bit patterns spread over all of `u32`.
#[cfg(feature = "sqrt-check")]
fn sqrt_check() {
    let edges = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::INFINITY,
        f32::from_bits(1),
    ];
    let stride = (0..=u32::MAX / 65_521).map(|i| f32::from_bits(i * 65_521));

    for x in edges.into_iter().chain(stride) {
        let (hw, sw) = (black_box(x).square_root(), libm::sqrtf(black_box(x)));
        if !(hw.to_bits() == sw.to_bits() || hw.is_nan() && sw.is_nan()) {
            mk_fail();
        }
    }

    measure("sqrt_check/ok", || true);
}

#[entry]
fn main() -> ! {
    paint_stack();
    // The marker overhead, subtracted from every other count.
    for _ in 0..4 {
        measure("calib/empty", || black_box(true));
    }
    #[cfg(feature = "sqrt-check")]
    sqrt_check();
    primitives();
    #[cfg(feature = "f32")]
    {
        f32_2x1();
        f32_4x2();
        f32_6x3();
        ekf_radar();
    }
    #[cfg(feature = "i16f16")]
    {
        i16f16_2x1();
        i16f16_4x2();
        i16f16_6x3();
    }
    #[cfg(feature = "i32f32")]
    {
        i32f32_2x1();
        i32f32_4x2();
        i32f32_6x3();
    }
    #[cfg(any(feature = "nav", feature = "nav-libm"))]
    {
        nav_bench::attitude();
        nav_bench::gnss_ins();
    }
    report_stack();
    mk_done()
}
