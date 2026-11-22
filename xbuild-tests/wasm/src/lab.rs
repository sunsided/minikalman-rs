//! Shared plumbing of the lab experiments: the output buffers and a seeded noise source.
//!
//! Every experiment is one batch export, `<name>_run(knobs…) -> u32`, that simulates a
//! whole run, writes it to the trace buffer and returns the number of steps. The trace is
//! channel-major: channel `c` of a run with `n` steps occupies `[c·n, (c + 1)·n)`. Scalar
//! results (RMSE, counts) go to the summary buffer. JS reads both through [`trace_ptr`] and
//! [`summary_ptr`] as a `Float32Array` over the module's memory, created after each run.
//!
//! The buffers are fixed-size statics, so no experiment allocates.

use std::cell::RefCell;

/// Number of `f32` values in the trace buffer (256 KiB).
pub const TRACE_CAPACITY: usize = 65_536;

/// Number of `f32` values in the summary buffer.
pub const SUMMARY_CAPACITY: usize = 64;

thread_local! {
    /// The per-step output of the last run, channel-major.
    static TRACE: RefCell<[f32; TRACE_CAPACITY]> = const { RefCell::new([0.0; TRACE_CAPACITY]) };

    /// The scalar results of the last run.
    static SUMMARY: RefCell<[f32; SUMMARY_CAPACITY]> =
        const { RefCell::new([0.0; SUMMARY_CAPACITY]) };
}

/// Returns the address of the trace buffer in the module's memory.
#[unsafe(no_mangle)]
pub extern "C" fn trace_ptr() -> *const f32 {
    TRACE.with(|t| t.borrow().as_ptr())
}

/// Returns the address of the summary buffer in the module's memory.
#[unsafe(no_mangle)]
pub extern "C" fn summary_ptr() -> *const f32 {
    SUMMARY.with(|s| s.borrow().as_ptr())
}

/// Returns the first `N` summary values of the last run.
#[cfg(test)]
pub fn summary<const N: usize>() -> [f32; N] {
    SUMMARY.with(|s| core::array::from_fn(|i| s.borrow()[i]))
}

/// Returns step `k` of `channel` of the last run with `steps` steps.
#[cfg(test)]
pub fn trace_value(steps: usize, channel: usize, k: usize) -> f32 {
    TRACE.with(|t| t.borrow()[channel * steps + k])
}

/// A channel-major view of the trace buffer for a run of `steps` steps.
pub struct Trace<'a> {
    /// The trace buffer.
    data: &'a mut [f32; TRACE_CAPACITY],
    /// The number of steps of the run.
    steps: usize,
}

impl Trace<'_> {
    /// Writes `value` to step `k` of `channel`.
    pub fn set(&mut self, channel: usize, k: usize, value: f32) {
        debug_assert!(k < self.steps);
        self.data[channel * self.steps + k] = value;
    }

    /// Reads step `k` of `channel`.
    pub fn get(&self, channel: usize, k: usize) -> f32 {
        self.data[channel * self.steps + k]
    }
}

/// Runs `f` with the trace buffer laid out for `channels` channels of `steps` steps and the
/// cleared summary buffer, and returns `steps`.
///
/// ## Panics
/// Panics if `channels × steps` exceeds [`TRACE_CAPACITY`].
pub fn record(channels: usize, steps: usize, f: impl FnOnce(&mut Trace<'_>, &mut [f32])) -> u32 {
    assert!(channels * steps <= TRACE_CAPACITY, "trace too long");
    TRACE.with(|t| {
        SUMMARY.with(|s| {
            let mut summary = s.borrow_mut();
            summary.fill(0.0);
            let mut data = t.borrow_mut();
            f(
                &mut Trace {
                    data: &mut data,
                    steps,
                },
                &mut summary[..],
            );
        });
    });
    u32::try_from(steps).unwrap_or(u32::MAX)
}

/// Clamps a knob from JS into `[min, max]`; NaN becomes `min`.
pub fn knob(value: f32, min: f32, max: f32) -> f32 {
    if value.is_nan() {
        min
    } else {
        value.clamp(min, max)
    }
}

/// A seeded xorshift64 noise source, so a run is reproducible and the module needs no
/// imports for randomness.
pub struct Noise(u64);

impl Noise {
    /// Creates a source from `seed`; every seed, including `0`, gives a valid state.
    pub fn new(seed: u32) -> Self {
        let seed = u64::from(seed);
        let mut noise = Self(0x9E37_79B9_7F4A_7C15 ^ (seed << 32 | seed));
        // Discard the first outputs, which are poorly mixed for small seeds.
        for _ in 0..8 {
            noise.uniform();
        }
        noise
    }

    /// Returns a uniform sample in `[0, 1)`.
    #[allow(clippy::cast_precision_loss)]
    pub fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns a standard normal sample (Box-Muller).
    pub fn gaussian(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    }

    /// Returns `true` with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.uniform() < p
    }
}

/// Returns the root mean square of `errors`, or NaN for none.
#[allow(clippy::cast_precision_loss)]
pub fn rms(errors: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = errors.fold((0.0, 0usize), |(s, n), e| (s + e * e, n + 1));
    (sum / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_reproducible_and_standard_normal() {
        let mut a = Noise::new(7);
        let mut b = Noise::new(7);
        let samples: Vec<f64> = (0..20_000).map(|_| a.gaussian()).collect();
        assert!(samples.iter().all(|&s| s == b.gaussian()));
        let mean = samples.iter().sum::<f64>() / 20_000.0;
        let var = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / 20_000.0;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "{mean} {var}"
        );
    }

    #[test]
    fn knobs_are_clamped() {
        assert_eq!(knob(f32::NAN, 1.0, 2.0), 1.0);
        assert_eq!(knob(5.0, 1.0, 2.0), 2.0);
        assert_eq!(knob(-5.0, 1.0, 2.0), 1.0);
    }
}
