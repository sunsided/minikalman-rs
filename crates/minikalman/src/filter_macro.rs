//! The [`filter!`](crate::filter) macro: a declaration language for states, observations,
//! controls, scratch, filters and systems of them, over the storage types.

pub mod support;

/// Declares named states, observations, controls, scratch, filters and systems of them, and
/// generates their storage and wiring.
///
/// `filter!` is a thin layer over the storage types. Every item becomes a module holding the
/// library's own storage type, a `const fn new()`, a `build()` that wires the library's
/// borrowed filter type to the storage, and a `model()` that writes the declared values. It
/// generates no `predict` or `correct` wrapper: you call the library API on what `build()`
/// returns, so every method of those types (controls through
/// [`TimeUpdate`](crate::regular::TimeUpdate), `correct_in`, `settle_gain`, the observer's
/// typed gain forms) works unchanged.
///
/// The examples below build up from the smallest filter. The [reference](#reference) at the
/// end has the grammar and the expansion of each item.
///
/// ## The smallest filter
///
/// One state, one observation and one Kalman filter. `build` wires each part to its storage,
/// `model` writes the declared matrices, and from there it is the library API:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.01); }
///     kalman tracker for Track {
///         A = identity;
///         A[pos, vel] = 0.1;
///         Q[vel, vel] = 0.01;
///         P = diag(10.0);
///     }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     let (mut filter_storage, mut gps_storage) = (tracker::new(), gps::new());
///     let mut filter = tracker::build(&mut filter_storage);
///     let mut gps = gps::build(&mut gps_storage);
///     tracker::model(&mut filter);
///     gps::model(&mut gps);
///
///     for pos in [0.1, 0.2, 0.32, 0.4, 0.49, 0.61, 0.7, 0.79] {
///         filter.predict();
///         Gps { pos }.write(gps.measurement_vector_mut().as_mut_slice());
///         filter.correct(&mut gps);
///     }
///
///     let track = Track::read(filter.state_vector().as_slice());
///     assert!(track.vel > 0.5 && track.vel < 1.5);
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// Invoke `filter!` at module level, not in a function body: each generated module imports
/// its parent with `use super::*`, which is how the statements see your constants and
/// functions.
///
/// ## A control, and two filters sharing it
///
/// A `system` puts every part in one value, and its handle has one field per part. The fields
/// are disjoint borrows, so the same observation and control serve both filters, one call at
/// a time:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::models::motion::constant_velocity;
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// const DT: f32 = 0.1;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///
///     /// Commanded acceleration.
///     control throttle: Throttle for Track { accel } {
///         B[pos, accel] = 0.5 * DT * DT;
///         B[vel, accel] = DT;
///         Q = diag(0.01);
///     }
///
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///
///     /// Reacts quickly.
///     kalman fast for Track {
///         A = |a| constant_velocity(DT, 1.0).fill_transition(a);
///         Q = |q| constant_velocity(DT, 1.0).fill_process_noise(q);
///         P = identity;
///     }
///
///     /// Smooths.
///     kalman slow for Track {
///         A = |a| constant_velocity(DT, 0.01).fill_transition(a);
///         Q = |q| constant_velocity(DT, 0.01).fill_process_noise(q);
///         P = identity;
///     }
///
///     system tracking { fast, slow, gps, throttle }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     let mut storage = tracking::new();
///     let mut sys = tracking::build(&mut storage);
///     tracking::model(&mut sys); // every member's model
///
///     for k in 0..50 {
///         let t = k as f32 * DT;
///         Throttle { accel: 1.0 }.write(sys.throttle.control_vector_mut().as_mut_slice());
///         for filter in [&mut sys.fast, &mut sys.slow] {
///             filter.predict().control(&mut sys.throttle);
///             Gps { pos: 0.5 * t * t }.write(sys.gps.measurement_vector_mut().as_mut_slice());
///             filter.correct(&mut sys.gps);
///         }
///     }
///
///     let fast = Track::read(sys.fast.state_vector().as_slice());
///     assert!((fast.vel - 4.9).abs() < 0.5);
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// Each correction rewrites `y`, `S` and `K` in the shared observation, so afterwards they
/// describe the last filter corrected. If you read the NIS or the gain per filter, use one
/// observation per filter.
///
/// ## A fixed-gain observer with a settled gain
///
/// A `luenberger` block runs a [`LuenbergerObserver`](crate::regular::LuenbergerObserver), with
/// no covariance and a fixed gain per observation. The observation's `H` comes from its
/// `observation` declaration. Here the gain comes from
/// [`settle_gain`](crate::regular::RegularKalman::settle_gain) on a Kalman filter. The empty
/// gain block `gps {}` tells `model` to leave that gain alone:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::models::motion::constant_velocity;
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// const DT: f32 = 0.1;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.25); }
///
///     /// The design filter: only run to settle the gain.
///     kalman design for Track {
///         A = |a| constant_velocity(DT, 1.0).fill_transition(a);
///         Q = |q| constant_velocity(DT, 1.0).fill_process_noise(q);
///         P = identity;
///     }
///
///     /// The run-time observer: `A`, `x` and one temporary, no covariance.
///     luenberger steady for Track {
///         A = |a| constant_velocity(DT, 1.0).fill_transition(a);
///     } gains {
///         gps {}
///     }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     let (mut design_storage, mut gps_storage) = (design::new(), gps::new());
///     let mut design = design::build(&mut design_storage);
///     let mut gps = gps::build(&mut gps_storage);
///     design::model(&mut design);
///     gps::model(&mut gps);
///     design.settle_gain(&mut gps, 1e-6, 10_000).expect("the gain settles");
///
///     let mut storage = steady::new();
///     let mut observer = steady::build(&mut storage);
///     observer.gps.set_gain(gps.kalman_gain());
///     steady::model(&mut observer); // writes A and H; the empty gain block keeps K∞
///
///     observer.observer.predict();
///     Gps { pos: 1.0 }.write(observer.gps.measurement_vector_mut().as_mut_slice());
///     assert!(observer.observer.correct(&mut observer.gps).is_applied());
///     assert!(Track::read(observer.observer.state_vector().as_slice()).pos > 0.0);
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// You can also write the gain itself, as `gps { K[pos, pos] = 0.5; K[vel, pos] = 0.8; }`.
/// The gain's name depends on the form, by design:
///
/// * A current-form block (`luenberger`) takes `K`, for `x̂⁺ = x̂⁻ + K·(z − H·x̂⁻)`.
/// * A predictor-form block (`luenberger predictor`) takes `L`, for
///   `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)`.
///
/// The names follow the library's own. The other name is a compile error, which catches
/// [the gain convention trap](crate::regular::LuenbergerObserver#the-gain-convention-trap):
/// a gain written in one form and run in the other.
///
/// ## Lean observations and a shared scratch, to save memory
///
/// An `observation` owns its correction temporaries by default, so it works with every
/// correction: `correct`, `correct_with_gain`, `settle_gain` and the robust variants.
/// `observation lean` stores only `H`, `z`, `R`, `y`, `S` and `K`. It corrects through
/// `correct_in`, which borrows the temporaries from a `scratch` sized for the largest
/// observation the scratch lists. One scratch serves every lean observation it lists, in
/// every filter over the same state.
///
/// Here the two lean observations and their scratch take 176 bytes instead of 212. Every
/// further lean observation saves its temporaries again:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::regular::ObservationStorage;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation lean gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///     observation lean odometry: Odometry for Track { pos, vel } { H = identity; R = diag(0.1); }
///     scratch shared for [gps, odometry];
///     kalman tracker for Track { A = identity; A[pos, vel] = 0.1; P = identity; Q = diag(0.01); }
///     system tracking { tracker, gps, odometry, shared }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     use core::mem::size_of;
///     let lean = size_of::<gps::Storage>() + size_of::<odometry::Storage>()
///         + size_of::<shared::Storage>();
///     let full = size_of::<ObservationStorage<2, 1, f32>>()
///         + size_of::<ObservationStorage<2, 2, f32>>();
///     assert_eq!((lean, full), (176, 212));
///
///     let mut storage = tracking::new();
///     let mut sys = tracking::build(&mut storage);
///     tracking::model(&mut sys);
///
///     sys.tracker.predict();
///     Gps { pos: 1.0 }.write(sys.gps.measurement_vector_mut().as_mut_slice());
///     assert!(sys.tracker.correct_in(&mut sys.gps, sys.shared).is_applied());
///     Odometry { pos: 1.0, vel: 0.5 }.write(sys.odometry.measurement_vector_mut().as_mut_slice());
///     assert!(sys.tracker.correct_in(&mut sys.odometry, sys.shared).is_applied());
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// ## Placement: on the stack or in a `static`
///
/// `filter!` creates no `static` and, unless an item says `storage = boxed;`, allocates
/// nothing. You place one value, and `new()` is a
/// `const fn` that zeroes it, so a `static` of it lands in `.bss`. For a local, write
/// `let mut storage = tracking::new();` as above. A `static` can be initialized directly:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///     kalman tracker for Track { A = identity; P = identity; }
///     system tracking { tracker, gps }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// static STORAGE: tracking::Storage = tracking::new();
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     // 20 + 17 f32: FilterStorage<2> and ObservationStorage<2, 1>, nothing else.
///     assert_eq!(core::mem::size_of_val(&STORAGE), 37 * 4);
///     assert_eq!(tracking::BYTES, 37 * 4);
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// To get a `&'static mut` without `static mut`, go through a take-once cell such as
/// [`static_cell::ConstStaticCell`](https://docs.rs/static_cell) (a crate of its own, not a
/// dependency of this one):
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use static_cell::ConstStaticCell;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///     kalman tracker for Track { A = identity; P = identity; }
///     system tracking { tracker, gps }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// static STORAGE: ConstStaticCell<tracking::Storage> = ConstStaticCell::new(tracking::new());
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     let storage: &'static mut tracking::Storage = STORAGE.take(); // once; .bss, nothing moved
///     let mut sys = tracking::build(storage);
///     tracking::model(&mut sys);
///     assert!(STORAGE.try_take().is_none()); // a second take is refused
///
///     sys.tracker.predict();
///     Gps { pos: 1.0 }.write(sys.gps.measurement_vector_mut().as_mut_slice());
///     sys.tracker.correct(&mut sys.gps);
///     assert!(Track::read(sys.tracker.state_vector().as_slice()).pos > 0.0);
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// An RTIC `#[local]` resource and a `Box::new(tracking::new())` work the same way. You can
/// also place each item's `Storage` on its own, e.g. the scratch on the stack and the filters
/// in a separate RAM section.
///
/// ## An owned filter as a struct field
///
/// On the desktop, a handle that borrows its storage is awkward: it has a lifetime, so it
/// cannot simply be a field of your own type. `storage = owned;` gives a `kalman`,
/// `observation` or `control` item an `Owned` type that holds its buffers by value. It has
/// no lifetime, it moves like any value, and `model` works on it:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { storage = owned; H = [pos]; R = diag(0.01); }
///     kalman tracker for Track {
///         storage = owned;
///         A = identity;
///         A[pos, vel] = 0.1;
///         Q[vel, vel] = 0.01;
///         P = diag(10.0);
///     }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// /// Your own type: the filter and its observation are plain fields.
/// struct Tracker {
///     filter: tracker::Owned,
///     gps: gps::Owned,
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// impl Tracker {
///     fn new() -> Self {
///         let mut tracker = Self { filter: tracker::new_owned(), gps: gps::new_owned() };
///         tracker::model(&mut tracker.filter);
///         gps::model(&mut tracker.gps);
///         tracker
///     }
///
///     fn update(&mut self, pos: f32) -> Track {
///         self.filter.predict();
///         Gps { pos }.write(self.gps.measurement_vector_mut().as_mut_slice());
///         self.filter.correct(&mut self.gps);
///         Track::read(self.filter.state_vector().as_slice())
///     }
/// }
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     let mut tracker = Tracker::new();
///     let mut track = Track { pos: 0.0, vel: 0.0 };
///     for pos in [0.1, 0.2, 0.32, 0.4, 0.49, 0.61, 0.7, 0.79] {
///         track = tracker.update(pos);
///     }
///
///     assert!(track.vel > 0.5 && track.vel < 1.5);
///     assert!(tracker::BYTES >= 20 * 4); // the buffers plus the filter's bookkeeping
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// `new_owned()` returns the value, so it passes through the stack (`BYTES`) on its way into
/// your struct. It is not a `const fn`, so an `Owned` cannot initialize a `static`; for a
/// `static`, use the inline `Storage`. For a large filter, `storage = boxed;` (with the `alloc` feature) gives a
/// `Boxed` type and `new_boxed()`, which allocates each matrix directly on the heap; only the
/// handle, a few pointers, is on the stack:
///
/// ```
/// # fn main() {
/// #     #[cfg(all(feature = "alloc", any(feature = "std", feature = "libm", feature = "micromath")))]
/// #     boxed::run();
/// # }
/// # #[cfg(all(feature = "alloc", any(feature = "std", feature = "libm", feature = "micromath")))]
/// # mod boxed {
/// use minikalman::prelude::*;
///
/// minikalman::filter! {
///     state Big(f64) { a, b, c, d, e, f, g, h }
///     kalman big for Big { storage = boxed; A = identity; P = identity; }
/// }
///
/// pub fn run() {
///     let mut filter = big::new_boxed();
///     big::model(&mut filter);
///     filter.predict();
///     assert_eq!(big::BYTES, (4 * 64 + 2 * 8) * 8); // on the heap
/// }
/// # }
/// ```
///
/// ## Squeezing RAM: `extern` buffers, lean observations, one scratch
///
/// On a small target you may already own the memory for some matrices: a DMA region, an arena
/// shared with another task, a section in faster RAM. `M = extern;` leaves `M` out of
/// `Storage`, and `build` takes it as `&mut [Scalar; N]`, with `N` derived from the declared
/// shape. The filter reads and writes your buffer in place; nothing is copied.
///
/// Placement and value are independent. `P = extern; P = diag(10.0);` still makes `model`
/// write `10·I` into your buffer; an extern matrix without a declared value keeps whatever
/// your buffer holds.
///
/// With a lean observation and one shared scratch, the filter below keeps 56 bytes inline and
/// borrows 24 bytes of `P` and `x` from the application. `BYTES` and `EXTERN_BYTES` are
/// constants, so the footprint can be checked at compile time:
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use minikalman::prelude::*;
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// use static_cell::ConstStaticCell;
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation lean gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///     scratch shared for [gps];
///     kalman tracker for Track {
///         x = extern; // placement statements come first
///         P = extern;
///         A = identity;
///         A[pos, vel] = 0.1;
///         Q = diag(0.01);
///         P = diag(10.0); // written into the caller's buffer by `model`
///     }
/// }
///
/// // A, Q and two temporaries inline; P and x lent by the application.
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// const _: () = assert!(tracker::BYTES == 14 * 4 && tracker::EXTERN_BYTES == 6 * 4);
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// /// Memory the application owns, e.g. shared with a logger task or placed by the linker.
/// static P: ConstStaticCell<[f32; 4]> = ConstStaticCell::new([0.0; 4]);
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// static X: ConstStaticCell<[f32; 2]> = ConstStaticCell::new([0.0; 2]);
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// static FILTER: ConstStaticCell<tracker::Storage> = ConstStaticCell::new(tracker::new());
///
/// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
/// fn main() {
///     let (p, x) = (P.take(), X.take());
///     let (mut gps_storage, mut scratch) = (gps::new(), shared::new());
///     let mut gps = gps::build(&mut gps_storage);
///     gps::model(&mut gps);
///
///     {
///         let mut filter = tracker::build(FILTER.take(), &mut *x, &mut *p);
///         tracker::model(&mut filter); // writes A, Q and P = 10·I
///         for pos in [0.1, 0.2, 0.32, 0.4, 0.49, 0.61, 0.7, 0.79] {
///             filter.predict();
///             Gps { pos }.write(gps.measurement_vector_mut().as_mut_slice());
///             filter.correct_in(&mut gps, &mut scratch);
///         }
///     }
///
///     // The estimate is in the application's buffers.
///     let track = Track::read(x);
///     assert!(track.vel > 0.5 && track.vel < 1.5);
///     assert!(p[0] < 10.0);
/// }
/// # #[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
/// # fn main() {}
/// ```
///
/// What goes through the stack, so you can budget for it:
///
/// * `new()` is a `const fn`. In a `static` it costs nothing at run time. As a local it is
///   `BYTES` on the stack, and `Box::new(module::new())` may build it on the stack before
///   moving it.
/// * `new_owned()` returns `BYTES` by value; `new_boxed()` allocates each matrix directly.
/// * `build()` copies nothing: the handle holds slices into the storage and your buffers.
/// * `model()` writes in place, with two exceptions. A vector statement `x = Track { .. }`
///   builds the `Track` value (`N` scalars) before writing it. A `luenberger` gain block builds
///   its `STATES × OBSERVATIONS` gain on the stack and copies it in with `set_gain`, because
///   the fixed-gain observation has no mutable gain accessor.
/// * There are no hidden temporaries: the correction temporaries are in a full observation's
///   `Storage` or in the shared `scratch`, and both count in `BYTES`.
///
/// ## Fixed point
///
/// Any [`MatrixDataType`](crate::matrix::MatrixDataType) scalar that implements
/// [`num_traits::ConstZero`] works. With the `fixed` feature, that needs `fixed` 1.28 or
/// newer. The statements take values as written, so declare typed constants:
///
/// ```
/// # fn main() {
/// #     #[cfg(feature = "fixed")]
/// #     fixed_point::run();
/// # }
/// # #[cfg(feature = "fixed")]
/// # mod fixed_point {
/// use minikalman::prelude::*;
///
/// const DT: I16F16 = I16F16::lit("0.1");
/// const HALF: I16F16 = I16F16::lit("0.5");
///
/// minikalman::filter! {
///     state Track(I16F16) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(HALF); }
///     kalman tracker for Track { A = identity; A[pos, vel] = DT; P = identity; Q = diag(DT); }
///     system tracking { tracker, gps }
/// }
///
/// pub fn run() {
///     let mut storage = tracking::new();
///     let mut sys = tracking::build(&mut storage);
///     tracking::model(&mut sys);
///
///     sys.tracker.predict();
///     Gps { pos: I16F16::ONE }.write(sys.gps.measurement_vector_mut().as_mut_slice());
///     sys.tracker.correct(&mut sys.gps);
///     assert!(Track::read(sys.tracker.state_vector().as_slice()).pos > HALF);
/// }
/// # }
/// ```
///
/// ## What a mistake looks like
///
/// Shapes are never written as numbers: they come from the `state!` entry counts, and matrix
/// entries are addressed only by name. Anything left over is checked at compile time.
///
/// A typo in an index name is rustc's own error, pointing at the name:
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     kalman bad for Track { A[pos, vz] = 0.1; }
/// }
/// fn main() {}
/// ```
///
/// ```text
/// error[E0599]: no associated function or constant named `vz` found for struct `Track` in the current scope
///   |
/// 3 |     kalman bad for Track { A[pos, vz] = 0.1; }
///   |                                   ^^ associated function or constant not found in `Track`
/// ```
///
/// An `H = [...]` that does not name exactly one state per measurement fails a const assert,
/// which also fires under `cargo check`:
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation bad: Bad for Track { pos } { H = [pos, vel]; }
/// }
/// fn main() {}
/// ```
///
/// ```text
/// error[E0080]: evaluation panicked: `H = [...]` must name exactly one state per measurement
/// ```
///
/// The other checked mistakes:
///
/// * A matrix of another block fails with "filter!: `R` is not a matrix of a `kalman` block":
///
///   ```compile_fail
///   minikalman::filter! {
///       state Track(f32) { pos, vel }
///       kalman bad for Track { R = diag(0.5); }
///   }
///   fn main() {}
///   ```
///
/// * A gain in the wrong form fails with "filter!: a current-form gain is `K`; `L` is the gain
///   of a `luenberger predictor` block", and the reverse for `K` in a predictor-form block:
///
///   ```compile_fail
///   minikalman::filter! {
///       state Track(f32) { pos, vel }
///       observation gps: Gps for Track { pos } { H = [pos]; }
///       luenberger bad for Track { A = identity; } gains { gps { L[pos, pos] = 0.5; } }
///   }
///   fn main() {}
///   ```
///
/// * An observation of another state type fails on the library's trait bound: E0277,
///   `KalmanFilterObservationCorrectFilter<2, f32>` is not satisfied for an observation of 3
///   states:
///
///   ```compile_fail
///   minikalman::filter! {
///       state Track(f32) { pos, vel }
///       state Pose(f32) { x, y, heading }
///       observation compass: Compass for Pose { heading } { H = [heading]; }
///       kalman fast for Track { A = identity; }
///   }
///
///   fn main() {
///       let (mut filter_storage, mut compass_storage) = (fast::new(), compass::new());
///       let mut filter = fast::build(&mut filter_storage);
///       let mut compass = compass::build(&mut compass_storage);
///       filter.correct(&mut compass);
///   }
///   ```
///
/// * A misspelled keyword fails with "filter!: expected `state`, `observation`, …, found: …".
///   Syntax errors point at the whole block, not at the bad token; the `found:` text shows
///   where parsing stopped.
///
/// * An `extern` buffer of the wrong length is a type error at the call: "expected an array
///   with a size of 4, found one with a size of 3":
///
///   ```compile_fail
///   minikalman::filter! {
///       state Track(f32) { pos, vel }
///       kalman k for Track { P = extern; }
///   }
///
///   fn main() {
///       let (mut storage, mut p) = (k::new(), [0.0_f32; 3]);
///       let _filter = k::build(&mut storage, &mut p);
///   }
///   ```
///
/// * `extern` together with `storage = owned` (or `boxed`) fails with "filter!: `extern`
///   buffers need `storage = inline`: an owned value borrowing external memory would carry a
///   lifetime again":
///
///   ```compile_fail
///   minikalman::filter! {
///       state Track(f32) { pos, vel }
///       kalman k for Track { storage = owned; P = extern; }
///   }
///   fn main() {}
///   ```
///
/// * The other placement mistakes are compile errors too:
///   * `observation lean` with `storage = owned` or `boxed`;
///   * a placement statement after a model statement, or inside a `luenberger` block;
///   * an unknown `storage` mode;
///   * an `extern` matrix the block does not have (`R = extern;` in a `kalman`);
///   * an owned, boxed or extern item as a `system` member ("filter!: `k` cannot be a system
///     member: only inline items without `extern` buffers can", next to rustc's own errors).
///
///   ```compile_fail
///   minikalman::filter! {
///       state Track(f32) { pos, vel }
///       observation gps: Gps for Track { pos } { H = [pos]; }
///       kalman k for Track { storage = owned; }
///       system s { k, gps }
///   }
///   fn main() {}
///   ```
///
/// ## How it works
///
/// ### `build()` wires, `model()` writes
///
/// `build()` is pure wiring, exactly like the storage types' own `build()`. It writes nothing,
/// so whatever the storage held stays. `model()` writes the declared values (`A`, `Q`, `P`,
/// `x₀`, `H`, `R`, `B`, `Qᵤ`, `u`, `z` and the gains) and nothing else:
///
/// * Call it once after the first `build()`. `system::model` calls the model of every member.
/// * Call it again to reset a filter to its declared model and initial values.
/// * It writes only what is declared. `P = identity;` sets every entry of `P`, while
///   `A[pos, vel] = DT;` sets one entry and leaves the others as they are. Entries you never
///   declare, such as an `x` without `x = …`, keep their value, which is zero after `new()`.
/// * In a `luenberger` block, `model()` writes each observation's `H` from its declaration
///   (`fill_h`). It writes the gain only when the gain block is non-empty.
///
/// The declared values are code that runs at start-up, not data. The storage stays zeroed, so
/// a `static` of it lands in `.bss`. Whole-matrix statements (`identity`, `diag`, `[..]`)
/// overwrite the entire matrix, `extern` buffers included.
///
/// ### Placement modes
///
/// A `kalman`, `observation` or `control` block may start with placement statements, before
/// any model statement:
///
/// * `storage = inline;` is the default: `Storage`, `new()`, `build(&mut Storage) ->
///   Handle<'_>`.
/// * `storage = owned;` generates `Owned` and `new_owned()`: the buffers held by value, no
///   lifetime, no allocation.
/// * `storage = boxed;` (needs the `alloc` feature) generates `Boxed` and `new_boxed()`: each
///   buffer in its own heap allocation.
/// * `M = extern;` leaves `M` out of `Storage`; `build` then takes it after the storage, as
///   `&mut [Scalar; N]`, in the order `A, x, P, Q` (kalman), `H, z, R` (observation) or
///   `B, u, Q` (control), whatever order you declared them in. `extern` needs
///   `storage = inline`. Only the length is type-checked, so buffers of the same length can
///   be swapped silently: `A`, `P` and `Q` always, `H` and `R` when there are as many
///   measurements as states, `B` and `Q` when there are as many controls as states.
///
/// `model` works in every mode. Only inline items without `extern` buffers can be `system`
/// members, since a system is one placed value wired in one `build`; wire the others
/// yourself, or keep the owned values in your own struct.
///
/// ### Sharing declarations
///
/// Items refer to each other only by Rust paths (`gps::OBSERVATIONS`, `gps::fill_h`,
/// `fast::Storage`). Rustc therefore does the name resolution and the shape checks, and you
/// can split items across `filter!` blocks, modules and crates: declare the state,
/// observations and controls once, and the filters wherever they are used.
///
/// * A `for` clause takes any path, resolved from inside the generated module. Write a name in
///   scope or an absolute path (`kalman third for crate::shared::Track { … }`), not `super::…`.
/// * `scratch`, `luenberger … gains` and `system` name their members as modules in scope, so
///   `use` the shared ones first (`use crate::shared::{gps, throttle};`).
/// * A `luenberger` block shares the *declaration* of an observation (its value type and its
///   `H`), not the object. Its fixed-gain observations are a different library type.
///
/// ### Memory
///
/// Every module exports `BYTES`, the bytes of the value you place (`size_of` of its `Storage`
/// or `Owned`; for `Boxed`, the heap bytes of its matrices), and `EXTERN_BYTES`, the bytes of
/// the `extern` buffers it borrows. A `system` exports both for all its members.
///
/// Every `Storage` is the library storage type, a plain struct of them, or (with `extern`
/// buffers) a struct of the inline matrices, checked at compile time to be exactly `BYTES`. A `luenberger` and a
/// `system` storage carry a compile-time assert that their size is exactly the sum of their
/// parts. The assert can fail for a `system` whose members mix scalars of different alignment,
/// e.g. `f32` and `f64`, when the struct needs padding; use one `system` per scalar. The
/// generated code contains no `unsafe`, no `static` and no allocation (except for
/// `storage = boxed;`), only paths you could
/// write by hand; `cargo expand` shows it.
///
/// ## Reference
///
/// ### Grammar
///
/// ```text
/// filter!     := item*
/// item        := attr* vis? ( state | observation | control | scratch
///                           | kalman | luenberger | system )
///
/// state       := 'state' Name '(' Scalar ')' '{' field,* '}'
/// observation := 'observation' 'lean'? mod ':' Name 'for' StatePath '{' field,* '}' block
/// control     := 'control' mod ':' Name 'for' StatePath '{' field,* '}' block
/// scratch     := 'scratch' mod 'for' '[' obsmod,+ ']' ';'
/// kalman      := 'kalman' mod 'for' StatePath block
/// luenberger  := 'luenberger' 'predictor'? mod 'for' StatePath '{' stmt* '}'
///                'gains' '{' ( obsmod '{' stmt* '}' )* '}'
/// system      := 'system' mod '{' member,+ '}'
///
/// field       := attr* ident                    -- doc comments allowed, as in state!
/// block       := '{' placement* stmt* '}'       -- kalman, observation, control
/// placement   := 'storage' '=' ( 'inline' | 'owned' | 'boxed' ) ';'
///              | M '=' 'extern' ';'
/// stmt        := M '=' 'identity' ';'
///              | M '=' 'diag' '(' expr ')' ';'
///              | M '[' row ',' col ']' '=' expr ';'
///              | M '=' '[' state,* ']' ';'      -- H only: row i selects state i
///              | M '=' '|' ident '|' expr ';'   -- any filler, e.g. models::*
///              | v '=' expr ';'                 -- vectors x, z, u: a Name { .. } value
/// ```
///
/// ### Matrices per block
///
/// | Block | Matrix | Rows × columns named by |
/// |---|---|---|
/// | `kalman` | `A`, `P`, `Q` / vector `x` | state × state / state |
/// | `luenberger` | `A` / vector `x` | state × state / state |
/// | `luenberger … gains { gps { … } }` | `K` (current form), `L` (`predictor` form) | state × `gps::Value` |
/// | `observation` | `H` / `R` / vector `z` | measurement × state / measurement × measurement / measurement |
/// | `control` | `B` / `Q` / vector `u` | state × control / control × control / control |
///
/// The role of the matrix picks the namespace, so you never qualify a name. In
/// `A[pos, vel] = DT;` both names are constants of the state. In `H[pos, pos]` the row is
/// `Gps::pos` and the column `Track::pos`. In `B[vel, accel]` the column is `Throttle::accel`.
///
/// ### What each item expands to
///
/// | Item | Module contents |
/// |---|---|
/// | `state Track(f32) { pos, vel }` | no module: a [`state!`](crate::state) struct `Track` |
/// | `observation gps: Gps for Track { pos } { … }` | `Storage` = [`ObservationStorage`](crate::regular::ObservationStorage), `Handle` = [`BorrowedObservation`](crate::regular::BorrowedObservation), `Value`, `STATES`, `OBSERVATIONS`, `fill_h`; plus a `state!` struct `Gps` |
/// | `observation lean gps: …` | the same over [`LeanObservationStorage`](crate::regular::LeanObservationStorage) and [`BorrowedLeanObservation`](crate::regular::BorrowedLeanObservation) |
/// | `control throttle: Throttle for Track { accel } { … }` | `Storage` = [`ControlStorage`](crate::regular::ControlStorage), `Handle` = [`BorrowedControl`](crate::regular::BorrowedControl), `Value`, `STATES`, `CONTROLS`; plus a `state!` struct `Throttle` |
/// | `scratch shared for [gps, lidar];` | `Storage` = [`CorrectionScratch`](crate::regular::CorrectionScratch) sized for the largest, `Handle` = `&mut Storage`, `MAX_OBSERVATIONS` |
/// | `kalman fast for Track { … }` | `Storage` = [`FilterStorage`](crate::regular::FilterStorage), `Handle` = [`BorrowedKalman`](crate::regular::BorrowedKalman) |
/// | `luenberger steady for Track { … } gains { gps { … } }` | struct `Storage { observer, gps }` of [`ObserverStorage`](crate::regular::ObserverStorage) and [`FixedGainObservationStorage`](crate::regular::FixedGainObservationStorage), struct `Handle { observer, gps }`, `Form` |
/// | `luenberger predictor steady …` | the same over [`PredictorObserverStorage`](crate::regular::PredictorObserverStorage) and [`Predictor`](crate::prelude::Predictor)-form observations |
/// | `system tracking { fast, gps, … }` | struct `Storage` of the members' `Storage`, struct `Handle` of their `Handle` |
/// | `kalman fast for Track { P = extern; … }` (also `observation`, `control`) | struct `Storage` of the inline matrices, `new()`, `build(&mut Storage, P: &mut [Scalar; N]) -> Handle<'_>`; `Handle` as without `extern` |
/// | `kalman fast for Track { storage = owned; … }` (also `observation`, `control`) | `Owned` (buffers by value, no lifetime), `new_owned()` |
/// | `kalman fast for Track { storage = boxed; … }` (also `observation`, `control`) | `Boxed` (the `alloc` builder types), `new_boxed()` |
///
/// Every module (all items but `state`) has `BYTES`, `EXTERN_BYTES` and `model`, and all but
/// `system` also have `Scalar` and `STATES`. Inline modules have `pub const fn new() -> Storage` and `pub fn build(&mut
/// Storage, …) -> Handle<'_>`; `model` takes `&mut Handle<'_>`, `&mut Owned` or `&mut Boxed`.
///
/// ### Limits
///
/// * So far, only the regular Kalman filter and the Luenberger observer. Wire the other
///   filters by hand from their storage types.
/// * Matrices are set by the statement forms above; there are no matrix literals.
/// * Closures stay at the call site: `predict_nonlinear` and friends are not generated. Inside
///   a closure body, write named indices as `m.set_at(Track::pos, Track::vel, v)`.
/// * Statements and `for` paths expand inside the generated module:
///   * They see the parent's items (through `use super::*`), not function locals.
///   * `self::` and `super::` are relative to the generated module.
///   * The module's own names (`Storage`, `Handle`, `Scalar`, `STATES`, `new`, `build`,
///     `model`, `fill_h`, …) shadow parent items of the same name.
/// * Put `#[cfg]` on the `filter!` invocation, not on an item inside it.
/// * The statements of one item expand recursively, about 119 per item at the default
///   `recursion_limit`.
/// * `storage =` and `extern` apply to `kalman`, `observation` (`extern` also when lean) and
///   `control` items; `luenberger` and `scratch` are always inline.
#[macro_export]
macro_rules! filter {
    () => {};

    (
        $(# $attr:tt)*
        $vis:vis state $name:ident ($scalar:ty) { $($fields:tt)* }
        $($rest:tt)*
    ) => {
        $crate::state! { $(# $attr)* $vis struct $name($scalar) { $($fields)* } }

        impl $crate::__dsl::Named for $name {
            type Scalar = $scalar;
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis observation lean $module:ident : $value:ident for $state:path
            { $($fields:tt)* } { $($model:tt)* }
        $($rest:tt)*
    ) => {
        $crate::__filter_value! { [$(# $attr)*] $vis $value $state { $($fields)* } }
        $crate::__filter_scan! {
            observation { $vis $module $value lean $state }
            inline [inline inline inline] ; $($model)*
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis observation $module:ident : $value:ident for $state:path
            { $($fields:tt)* } { $($model:tt)* }
        $($rest:tt)*
    ) => {
        $crate::__filter_value! { [$(# $attr)*] $vis $value $state { $($fields)* } }
        $crate::__filter_scan! {
            observation { $vis $module $value full $state }
            inline [inline inline inline] ; $($model)*
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis control $module:ident : $value:ident for $state:path
            { $($fields:tt)* } { $($model:tt)* }
        $($rest:tt)*
    ) => {
        $crate::__filter_value! { [$(# $attr)*] $vis $value $state { $($fields)* } }
        $crate::__filter_scan! {
            control { $vis $module $value $state }
            inline [inline inline inline] ; $($model)*
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis scratch $module:ident for [ $first:ident $(, $obs:ident)* $(,)? ] ;
        $($rest:tt)*
    ) => {
        $(# $attr)*
        $vis mod $module {
            #[allow(unused_imports)]
            use super::*;

            /// The scalar type, that of the first observation listed.
            pub type Scalar = $first::Scalar;
            /// The number of states, that of the first observation listed.
            pub const STATES: usize = $first::STATES;
            /// The size of the largest observation this scratch serves.
            pub const MAX_OBSERVATIONS: usize =
                $crate::__dsl::max(&[$first::OBSERVATIONS $(, $obs::OBSERVATIONS)*]);
            /// The library scratch, lent to every `correct_in` of the listed observations.
            pub type Storage = $crate::regular::CorrectionScratch<STATES, MAX_OBSERVATIONS, Scalar>;
            /// A borrow of the scratch, passed to `correct_in`.
            pub type Handle<'a> = &'a mut Storage;
            /// The bytes of [`Storage`], the value you place.
            pub const BYTES: usize = ::core::mem::size_of::<Storage>();
            /// A scratch has no `extern` buffers.
            pub const EXTERN_BYTES: usize = 0;
            #[doc(hidden)]
            pub const SYSTEM_MEMBER: bool = true;

            const _: () = {
                $( ::core::assert!(
                    $obs::STATES == STATES,
                    "a scratch serves observations of one state type only"
                ); )*
            };

            /// Creates zeroed scratch; `const`, so it can initialize a `static`.
            pub const fn new() -> Storage {
                Storage::new(<Scalar as $crate::num_traits::ConstZero>::ZERO)
            }

            /// Borrows the scratch.
            pub fn build(storage: &mut Storage) -> Handle<'_> {
                storage
            }

            /// Does nothing: a scratch has no model. Exists so a `system` can call every
            /// member's `model`.
            pub fn model(_scratch: &mut Handle<'_>) {}
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis kalman $module:ident for $state:path { $($model:tt)* }
        $($rest:tt)*
    ) => {
        $crate::__filter_scan! {
            kalman { [$(# $attr)*] $vis $module $state }
            inline [inline inline inline inline] ; $($model)*
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis luenberger predictor $module:ident for $state:path { $($model:tt)* }
            gains { $( $obs:ident { $($gain:tt)* } )* }
        $($rest:tt)*
    ) => {
        $crate::__filter_luenberger! {
            [$(# $attr)*] $vis $module $state
            [PredictorObserverStorage BorrowedPredictorObserver Predictor predictor_gain]
            { $($model)* } { $( $obs { $($gain)* } )* }
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis luenberger $module:ident for $state:path { $($model:tt)* }
            gains { $( $obs:ident { $($gain:tt)* } )* }
        $($rest:tt)*
    ) => {
        $crate::__filter_luenberger! {
            [$(# $attr)*] $vis $module $state
            [ObserverStorage BorrowedLuenbergerObserver Current current_gain]
            { $($model)* } { $( $obs { $($gain)* } )* }
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis system $module:ident { $($member:ident),+ $(,)? }
        $($rest:tt)*
    ) => {
        $(# $attr)*
        $vis mod $module {
            #[allow(unused_imports)]
            use super::*;

            /// The storage of every member in one value: one `static`, one stack slot or one
            /// `Box`. Exactly the sum of the members' storage, checked at compile time.
            pub struct Storage {
                $(
                    #[doc = ::core::concat!("Storage of `", ::core::stringify!($member), "`.")]
                    pub $member: $member::Storage,
                )+
            }

            const _: () = {
                $( ::core::assert!(
                    $member::SYSTEM_MEMBER,
                    ::core::concat!(
                        "filter!: `", ::core::stringify!($member), "` cannot be a system member: ",
                        "only inline items without `extern` buffers can"
                    )
                ); )+
            };

            /// The bytes of [`Storage`], the value you place: the sum of the members' `BYTES`.
            pub const BYTES: usize = ::core::mem::size_of::<Storage>();
            /// The bytes of the members' `extern` buffers; `0`, since a member with `extern`
            /// buffers cannot be part of a system.
            pub const EXTERN_BYTES: usize = 0 $(+ $member::EXTERN_BYTES)+;
            #[doc(hidden)]
            pub const SYSTEM_MEMBER: bool = true;

            const _: () = ::core::assert!(
                ::core::mem::size_of::<Storage>()
                    == 0 $(+ ::core::mem::size_of::<$member::Storage>())+,
                "the system storage must be exactly the sum of its members (mixed scalars can pad it)"
            );

            /// Every member, wired. The fields are disjoint, so `sys.fast.correct(&mut
            /// sys.gps)` borrows each one separately.
            pub struct Handle<'a> {
                $(
                    #[doc = ::core::concat!("The wired `", ::core::stringify!($member), "`.")]
                    pub $member: $member::Handle<'a>,
                )+
            }

            /// Creates zeroed storage for every member; `const`, so it can initialize a `static`.
            pub const fn new() -> Storage {
                Storage { $( $member: $member::new(), )+ }
            }

            /// Wires every member over its storage. Writes nothing; see [`model`].
            pub fn build(storage: &mut Storage) -> Handle<'_> {
                Handle { $( $member: $member::build(&mut storage.$member), )+ }
            }

            /// Writes the declared model of every member, e.g. to reset the system.
            pub fn model(handle: &mut Handle<'_>) {
                $( $member::model(&mut handle.$member); )+
            }
        }

        $crate::filter! { $($rest)* }
    };

    ($($bad:tt)+) => {
        ::core::compile_error!(::core::concat!(
            "filter!: expected `state`, `observation`, `control`, `scratch`, `kalman`, ",
            "`luenberger` or `system`, found: ",
            ::core::stringify!($($bad)+)
        ));
    };
}

/// Declares the named value type of an `observation` or `control` item: a `state!` struct
/// whose scalar is that of the state.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_value {
    ([$(# $attr:tt)*] $vis:vis $value:ident $state:path { $($fields:tt)* }) => {
        $crate::state! {
            $(# $attr)*
            $vis struct $value(<$state as $crate::__dsl::Named>::Scalar) { $($fields)* }
        }

        impl $crate::__dsl::Named for $value {
            type Scalar = <$state as $crate::__dsl::Named>::Scalar;
        }
    };
}

/// Reads the leading placement statements of a model block, `storage = …;` and
/// `M = extern;`, then expands the item with the remaining model statements. The slots record,
/// per matrix in builder order, whether it is `inline` or `extern`. Only the leading
/// statements are scanned, so the scan costs one recursion step per placement statement.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_scan {
    ($kind:ident $info:tt $mode:ident $slots:tt ; storage = $new:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { $kind $info $new $slots ; $($rest)* }
    };
    ($kind:ident $info:tt $mode:ident $slots:tt ; $m:ident = extern ; $($rest:tt)*) => {
        $crate::__filter_mark! { $kind $m $slots $info $mode ; $($rest)* }
    };
    ($kind:ident $info:tt $mode:ident $slots:tt ; $($model:tt)*) => {
        $crate::__filter_item! { $kind $mode $slots $info { $($model)* } }
    };
}

/// Marks one matrix of a slot list as `extern`, then continues the scan.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_mark {
    (kalman A [$a:ident $x:ident $p:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { kalman $info $mode [extern $x $p $q] ; $($rest)* }
    };
    (kalman x [$a:ident $x:ident $p:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { kalman $info $mode [$a extern $p $q] ; $($rest)* }
    };
    (kalman P [$a:ident $x:ident $p:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { kalman $info $mode [$a $x extern $q] ; $($rest)* }
    };
    (kalman Q [$a:ident $x:ident $p:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { kalman $info $mode [$a $x $p extern] ; $($rest)* }
    };
    (observation H [$h:ident $z:ident $r:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { observation $info $mode [extern $z $r] ; $($rest)* }
    };
    (observation z [$h:ident $z:ident $r:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { observation $info $mode [$h extern $r] ; $($rest)* }
    };
    (observation R [$h:ident $z:ident $r:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { observation $info $mode [$h $z extern] ; $($rest)* }
    };
    (control B [$b:ident $u:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { control $info $mode [extern $u $q] ; $($rest)* }
    };
    (control u [$b:ident $u:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { control $info $mode [$b extern $q] ; $($rest)* }
    };
    (control Q [$b:ident $u:ident $q:ident] $info:tt $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_scan! { control $info $mode [$b $u extern] ; $($rest)* }
    };
    ($kind:ident $m:ident $slots:tt $info:tt $mode:ident ; $($rest:tt)*) => {
        ::core::compile_error!(::core::concat!(
            "filter!: `", ::core::stringify!($m), "` cannot be `extern` in a `",
            ::core::stringify!($kind), "` block"
        ));
    };
}

/// Expands a `kalman`, `observation` or `control` item once its placement is known: the
/// parts every placement shares, then [`__filter_place`] for the storage.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_item {
    (kalman $mode:ident $slots:tt { [$(# $attr:tt)*] $vis:vis $module:ident $state:path } { $($model:tt)* }) => {
        $(# $attr)*
        $vis mod $module {
            #[allow(unused_imports)]
            use super::*;
            #[allow(unused_imports)]
            use $crate::matrix::{MatrixMut as _, RowMajorSequentialDataMut as _, SquareMatrixMut as _};

            /// The scalar type, that of the state.
            pub type Scalar = <$state as $crate::__dsl::Named>::Scalar;
            /// The number of states.
            pub const STATES: usize = <$state>::N;

            $crate::__filter_place! { kalman $mode $slots [$state] { $($model)* } }
        }
    };

    (observation $mode:ident $slots:tt { $vis:vis $module:ident $value:ident $flavour:ident $state:path } { $($model:tt)* }) => {
        #[doc = ::core::concat!("Storage and model of the `", ::core::stringify!($value), "` observation.")]
        $vis mod $module {
            #[allow(unused_imports)]
            use super::*;
            #[allow(unused_imports)]
            use $crate::matrix::{MatrixMut as _, RowMajorSequentialDataMut as _, SquareMatrixMut as _};

            /// The named measurement vector.
            pub type Value = $value;
            /// The scalar type, that of the state.
            pub type Scalar = <$state as $crate::__dsl::Named>::Scalar;
            /// The number of states.
            pub const STATES: usize = <$state>::N;
            /// The number of measurements.
            pub const OBSERVATIONS: usize = <$value>::N;

            $crate::__filter_stmts!(hcheck [$state, $value]; $($model)*);

            /// Writes the declared `H` into any `OBSERVATIONS × STATES` matrix, e.g. the `H` of
            /// a fixed-gain observation of the same sensor.
            pub fn fill_h<M>(h: &mut M)
            where
                M: $crate::matrix::MatrixMut<OBSERVATIONS, STATES, Scalar> + ?Sized,
            {
                let _ = &h;
                $crate::__filter_stmts!(hmodel [h $state, $value]; $($model)*);
            }

            $crate::__filter_place! { observation $mode $slots [$state, $value, $flavour] { $($model)* } }
        }
    };

    (control $mode:ident $slots:tt { $vis:vis $module:ident $value:ident $state:path } { $($model:tt)* }) => {
        #[doc = ::core::concat!("Storage and model of the `", ::core::stringify!($value), "` control.")]
        $vis mod $module {
            #[allow(unused_imports)]
            use super::*;
            #[allow(unused_imports)]
            use $crate::matrix::{MatrixMut as _, RowMajorSequentialDataMut as _, SquareMatrixMut as _};

            /// The named control vector.
            pub type Value = $value;
            /// The scalar type, that of the state.
            pub type Scalar = <$state as $crate::__dsl::Named>::Scalar;
            /// The number of states.
            pub const STATES: usize = <$state>::N;
            /// The number of control inputs.
            pub const CONTROLS: usize = <$value>::N;

            $crate::__filter_place! { control $mode $slots [$state, $value] { $($model)* } }
        }
    };
}

/// The storage, handle, constructors and `model` of an item in its placement mode.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_place {
    // Kalman filters.
    (kalman inline [inline inline inline inline] $ctx:tt $model:tt) => {
        /// The library storage: `A`, `x`, `P`, `Q` and two temporaries.
        pub type Storage = $crate::regular::FilterStorage<STATES, Scalar>;
        $crate::__filter_inline_alias!();
        $crate::__filter_kalman_handle!($ctx $model);
    };
    (kalman inline [$a:ident $x:ident $p:ident $q:ident] $ctx:tt $model:tt) => {
        $crate::__filter_lower! {
            [storage 'a [$crate::regular::RegularKalmanBuilder::new::<STATES, Scalar>] "`A`, `x`, `P`, `Q`"]
            {} {} {} {} {} {} ;
            ($a A a_buffer { STATES * STATES }) ($x x x_buffer { STATES }) ($p P p_buffer { STATES * STATES })
            ($q Q q_buffer { STATES * STATES }) (inline predicted_x predicted_x { STATES })
            (inline temp_P temp_P { STATES * STATES })
        }
        $crate::__filter_kalman_handle!($ctx $model);
    };
    (kalman owned [inline inline inline inline] $ctx:tt $model:tt) => {
        /// The filter with its buffers held by value: movable, no lifetime, no allocation.
        pub type Owned = $crate::__dsl::OwnedKalman<STATES, { STATES * STATES }, Scalar>;
        /// The bytes of an [`Owned`] filter, buffers and bookkeeping.
        pub const BYTES: usize = ::core::mem::size_of::<Owned>();
        /// No buffer is `extern`.
        pub const EXTERN_BYTES: usize = 0;

        #[doc(hidden)]
        pub const SYSTEM_MEMBER: bool = false;

        /// Creates a zeroed filter by value (`BYTES` on the stack until you move it).
        pub fn new_owned() -> Owned {
            $crate::__dsl::owned_kalman(<Scalar as $crate::num_traits::ConstZero>::ZERO)
        }

        $crate::__filter_model!(kalman $ctx (Owned) $model);
    };
    (kalman boxed [inline inline inline inline] $ctx:tt $model:tt) => {
        $crate::__filter_if_alloc! {
            /// The filter with each buffer in its own heap allocation.
            pub type Boxed = $crate::regular::builder::KalmanFilterType<STATES, Scalar>;
            /// The heap bytes of the matrix data of a [`Boxed`] filter.
            pub const BYTES: usize = (4 * STATES * STATES + 2 * STATES) * ::core::mem::size_of::<Scalar>();
            /// No buffer is `extern`.
            pub const EXTERN_BYTES: usize = 0;

            #[doc(hidden)]
            pub const SYSTEM_MEMBER: bool = false;

            /// Allocates a zeroed filter, each matrix directly on the heap.
            pub fn new_boxed() -> Boxed {
                $crate::regular::builder::KalmanFilterBuilder::<STATES, Scalar>::new().build()
            }

            $crate::__filter_model!(kalman $ctx (Boxed) $model);
        }
    };

    // Observations.
    (observation inline [inline inline inline] [$state:path, $value:ident, full] $model:tt) => {
        /// The library storage: `H`, `z`, `R`, `y`, `S`, `K` and the correction temporaries.
        pub type Storage = $crate::regular::ObservationStorage<STATES, OBSERVATIONS, Scalar>;
        $crate::__filter_inline_alias!();
        /// The observation [`build`] returns.
        pub type Handle<'a> = $crate::regular::BorrowedObservation<'a, STATES, OBSERVATIONS, Scalar>;
        $crate::__filter_model!(observation [$state, $value] (Handle<'_>) $model);
    };
    (observation inline [inline inline inline] [$state:path, $value:ident, lean] $model:tt) => {
        /// The library storage: `H`, `z`, `R`, `y`, `S` and `K`; correct it with `correct_in`
        /// and a `scratch`.
        pub type Storage = $crate::regular::LeanObservationStorage<STATES, OBSERVATIONS, Scalar>;
        $crate::__filter_inline_alias!();
        /// The observation [`build`] returns.
        pub type Handle<'a> = $crate::regular::BorrowedLeanObservation<'a, STATES, OBSERVATIONS, Scalar>;
        $crate::__filter_model!(observation [$state, $value] (Handle<'_>) $model);
    };
    (observation inline [$h:ident $z:ident $r:ident] [$state:path, $value:ident, full] $model:tt) => {
        $crate::__filter_lower! {
            [storage 'a [$crate::regular::RegularObservationBuilder::new::<STATES, OBSERVATIONS, Scalar>] "`H`, `z`, `R`"]
            {} {} {} {} {} {} ;
            ($h H h_buffer { OBSERVATIONS * STATES }) ($z z z_buffer { OBSERVATIONS })
            ($r R r_buffer { OBSERVATIONS * OBSERVATIONS }) (inline y y { OBSERVATIONS })
            (inline S S { OBSERVATIONS * OBSERVATIONS }) (inline K K { STATES * OBSERVATIONS })
            (inline temp_S_inv temp_S_inv { OBSERVATIONS * OBSERVATIONS })
            (inline temp_HP temp_HP { OBSERVATIONS * STATES }) (inline temp_PHt temp_PHt { STATES * OBSERVATIONS })
            (inline temp_KHP temp_KHP { STATES * STATES })
        }
        /// The observation [`build`] returns.
        pub type Handle<'a> = $crate::regular::BorrowedObservation<'a, STATES, OBSERVATIONS, Scalar>;
        $crate::__filter_model!(observation [$state, $value] (Handle<'_>) $model);
    };
    (observation inline [$h:ident $z:ident $r:ident] [$state:path, $value:ident, lean] $model:tt) => {
        $crate::__filter_lower! {
            [storage 'a [$crate::regular::RegularObservationBuilder::new_lean::<STATES, OBSERVATIONS, Scalar>] "`H`, `z`, `R`"]
            {} {} {} {} {} {} ;
            ($h H h_buffer { OBSERVATIONS * STATES }) ($z z z_buffer { OBSERVATIONS })
            ($r R r_buffer { OBSERVATIONS * OBSERVATIONS }) (inline y y { OBSERVATIONS })
            (inline S S { OBSERVATIONS * OBSERVATIONS }) (inline K K { STATES * OBSERVATIONS })
        }
        /// The observation [`build`] returns; correct it with `correct_in` and a `scratch`.
        pub type Handle<'a> = $crate::regular::BorrowedLeanObservation<'a, STATES, OBSERVATIONS, Scalar>;
        $crate::__filter_model!(observation [$state, $value] (Handle<'_>) $model);
    };
    (observation owned [inline inline inline] [$state:path, $value:ident, full] $model:tt) => {
        /// The observation with its buffers held by value: movable, no lifetime, no allocation.
        pub type Owned = $crate::__dsl::OwnedObservation<
            STATES,
            OBSERVATIONS,
            { OBSERVATIONS * STATES },
            { OBSERVATIONS * OBSERVATIONS },
            { STATES * STATES },
            Scalar,
        >;
        /// The bytes of an [`Owned`] observation, buffers and bookkeeping.
        pub const BYTES: usize = ::core::mem::size_of::<Owned>();
        /// No buffer is `extern`.
        pub const EXTERN_BYTES: usize = 0;

        #[doc(hidden)]
        pub const SYSTEM_MEMBER: bool = false;

        /// Creates a zeroed observation by value (`BYTES` on the stack until you move it).
        pub fn new_owned() -> Owned {
            $crate::__dsl::owned_observation(<Scalar as $crate::num_traits::ConstZero>::ZERO)
        }

        $crate::__filter_model!(observation [$state, $value] (Owned) $model);
    };
    (observation boxed [inline inline inline] [$state:path, $value:ident, full] $model:tt) => {
        $crate::__filter_if_alloc! {
            /// The observation with each buffer in its own heap allocation.
            pub type Boxed =
                $crate::regular::builder::KalmanFilterObservationType<STATES, OBSERVATIONS, Scalar>;
            /// The heap bytes of the matrix data of a [`Boxed`] observation.
            pub const BYTES: usize = (4 * OBSERVATIONS * STATES
                + 3 * OBSERVATIONS * OBSERVATIONS
                + 2 * OBSERVATIONS
                + STATES * STATES)
                * ::core::mem::size_of::<Scalar>();
            /// No buffer is `extern`.
            pub const EXTERN_BYTES: usize = 0;

            #[doc(hidden)]
            pub const SYSTEM_MEMBER: bool = false;

            /// Allocates a zeroed observation, each matrix directly on the heap.
            pub fn new_boxed() -> Boxed {
                $crate::regular::builder::KalmanFilterObservationBuilder::<STATES, Scalar>::new()
                    .build::<OBSERVATIONS>()
            }

            $crate::__filter_model!(observation [$state, $value] (Boxed) $model);
        }
    };
    (observation $mode:ident [inline inline inline] [$state:path, $value:ident, lean] $model:tt) => {
        ::core::compile_error!(::core::concat!(
            "filter!: `observation lean` needs `storage = inline`, not `",
            ::core::stringify!($mode), "`: it corrects through a borrowed `scratch` anyway"
        ));
    };

    // Controls.
    (control inline [inline inline inline] [$state:path, $value:ident] $model:tt) => {
        /// The library storage: `B`, `u`, `Q` and `B×Q`.
        pub type Storage = $crate::regular::ControlStorage<STATES, CONTROLS, Scalar>;
        $crate::__filter_inline_alias!();
        $crate::__filter_control_handle!([$state, $value] $model);
    };
    (control inline [$b:ident $u:ident $q:ident] [$state:path, $value:ident] $model:tt) => {
        $crate::__filter_lower! {
            [storage 'a [$crate::regular::ControlBuilder::new::<STATES, CONTROLS, Scalar>] "`B`, `u`, `Q`"]
            {} {} {} {} {} {} ;
            ($b B b_buffer { STATES * CONTROLS }) ($u u u_buffer { CONTROLS }) ($q Q q_buffer { CONTROLS * CONTROLS })
            (inline temp_BQ temp_BQ { STATES * CONTROLS })
        }
        $crate::__filter_control_handle!([$state, $value] $model);
    };
    (control owned [inline inline inline] [$state:path, $value:ident] $model:tt) => {
        /// The control with its buffers held by value: movable, no lifetime, no allocation.
        pub type Owned = $crate::__dsl::OwnedControl<
            STATES,
            CONTROLS,
            { STATES * CONTROLS },
            { CONTROLS * CONTROLS },
            Scalar,
        >;
        /// The bytes of an [`Owned`] control, buffers and bookkeeping.
        pub const BYTES: usize = ::core::mem::size_of::<Owned>();
        /// No buffer is `extern`.
        pub const EXTERN_BYTES: usize = 0;

        #[doc(hidden)]
        pub const SYSTEM_MEMBER: bool = false;

        /// Creates a zeroed control by value (`BYTES` on the stack until you move it).
        pub fn new_owned() -> Owned {
            $crate::__dsl::owned_control(<Scalar as $crate::num_traits::ConstZero>::ZERO)
        }

        $crate::__filter_model!(control [$state, $value] (Owned) $model);
    };
    (control boxed [inline inline inline] [$state:path, $value:ident] $model:tt) => {
        $crate::__filter_if_alloc! {
            /// The control with each buffer in its own heap allocation.
            pub type Boxed = $crate::regular::builder::KalmanFilterControlType<STATES, CONTROLS, Scalar>;
            /// The heap bytes of the matrix data of a [`Boxed`] control.
            pub const BYTES: usize = (2 * STATES * CONTROLS + CONTROLS * CONTROLS + CONTROLS)
                * ::core::mem::size_of::<Scalar>();
            /// No buffer is `extern`.
            pub const EXTERN_BYTES: usize = 0;

            #[doc(hidden)]
            pub const SYSTEM_MEMBER: bool = false;

            /// Allocates a zeroed control, each matrix directly on the heap.
            pub fn new_boxed() -> Boxed {
                $crate::regular::builder::KalmanFilterControlBuilder::<STATES, Scalar>::new()
                    .build::<CONTROLS>()
            }

            $crate::__filter_model!(control [$state, $value] (Boxed) $model);
        }
    };

    // Unsupported combinations.
    ($kind:ident inline $slots:tt $ctx:tt $model:tt) => {
        ::core::compile_error!("filter!: unsupported placement");
    };
    ($kind:ident owned $slots:tt $ctx:tt $model:tt) => {
        ::core::compile_error!(
            "filter!: `extern` buffers need `storage = inline`: an owned value borrowing external memory would carry a lifetime again"
        );
    };
    ($kind:ident boxed $slots:tt $ctx:tt $model:tt) => {
        ::core::compile_error!(
            "filter!: `extern` buffers need `storage = inline`: a boxed value borrowing external memory would carry a lifetime again"
        );
    };
    ($kind:ident $mode:ident $slots:tt $ctx:tt $model:tt) => {
        ::core::compile_error!(::core::concat!(
            "filter!: `storage` is `inline`, `owned` or `boxed`, found `",
            ::core::stringify!($mode), "`"
        ));
    };
}

/// The footprint constants, `new` and `build` of an item whose `Storage` is the library type.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_inline_alias {
    () => {
        /// The bytes of [`Storage`], the value you place.
        pub const BYTES: usize = ::core::mem::size_of::<Storage>();
        /// No buffer is `extern`.
        pub const EXTERN_BYTES: usize = 0;

        #[doc(hidden)]
        pub const SYSTEM_MEMBER: bool = true;

        /// Creates zeroed storage; `const`, so it can initialize a `static`.
        pub const fn new() -> Storage {
            Storage::new(<Scalar as $crate::num_traits::ConstZero>::ZERO)
        }

        /// Wires the item over `storage`. Writes nothing; see [`model`].
        pub fn build(storage: &mut Storage) -> Handle<'_> {
            storage.build()
        }
    };
}

/// The borrowed handle and `model` of an inline `kalman` item.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_kalman_handle {
    ($ctx:tt $model:tt) => {
        /// The filter [`build`] returns.
        pub type Handle<'a> = $crate::regular::BorrowedKalman<'a, STATES, Scalar>;
        $crate::__filter_model!(kalman $ctx (Handle<'_>) $model);
    };
}

/// The borrowed handle and `model` of an inline `control` item.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_control_handle {
    ($ctx:tt $model:tt) => {
        /// The control [`build`] returns.
        pub type Handle<'a> = $crate::regular::BorrowedControl<'a, STATES, CONTROLS, Scalar>;
        $crate::__filter_model!(control $ctx (Handle<'_>) $model);
    };
}

/// The `model` function of an item, over the given handle type.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_model {
    (kalman [$state:path] ($target:ty) { $($model:tt)* }) => {
        /// Writes the declared `A`, `Q`, `P` and `x` into the filter, e.g. to reset it.
        pub fn model(filter: &mut $target) {
            let _ = &filter;
            $crate::__filter_stmts!(kalman [filter $state]; $($model)*);
        }
    };
    (observation [$state:path, $value:ident] ($target:ty) { $($model:tt)* }) => {
        /// Writes the declared `H`, `R` and `z` into the observation.
        pub fn model(observation: &mut $target) {
            let _ = &observation;
            $crate::__filter_stmts!(observation [observation $state, $value]; $($model)*);
        }
    };
    (control [$state:path, $value:ident] ($target:ty) { $($model:tt)* }) => {
        /// Writes the declared `B`, `Q` and `u` into the control.
        pub fn model(control: &mut $target) {
            let _ = &control;
            $crate::__filter_stmts!(control [control $state, $value]; $($model)*);
        }
    };
}

/// Lowers a slot list into the inline `Storage` struct, `new`, and a `build` that takes each
/// `extern` matrix as a `&mut [Scalar; N]`. Fields and parameters cannot come from macro
/// calls, so they are accumulated as tokens and emitted in one step.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_lower {
    (
        $ctx:tt {$($fields:tt)*} {$($inits:tt)*} {$($params:tt)*} {$($args:tt)*}
        {$($inline:tt)*} {$($extern:tt)*} ;
    ) => {
        $crate::__filter_lowered! {
            $ctx {$($fields)*} {$($inits)*} {$($params)*} {$($args)*} {$($inline)*} {$($extern)*}
        }
    };
    (
        [$st:ident $lt:lifetime $ctor:tt $names:literal]
        {$($fields:tt)*} {$($inits:tt)*} {$($params:tt)*} {$($args:tt)*}
        {$($inline:tt)*} {$($extern:tt)*} ;
        (inline $name:ident $param:ident $len:tt) $($rest:tt)*
    ) => {
        $crate::__filter_lower! {
            [$st $lt $ctor $names]
            {
                $($fields)*
                #[doc = ::core::concat!("`", ::core::stringify!($name), "`, held inline.")]
                pub $name: [Scalar; $len],
            }
            { $($inits)* $name: [<Scalar as $crate::num_traits::ConstZero>::ZERO; $len], }
            { $($params)* }
            { $($args)* $st.$name.as_mut_slice().into(), }
            { $($inline)* + $len }
            { $($extern)* }
            ; $($rest)*
        }
    };
    (
        [$st:ident $lt:lifetime $ctor:tt $names:literal]
        {$($fields:tt)*} {$($inits:tt)*} {$($params:tt)*} {$($args:tt)*}
        {$($inline:tt)*} {$($extern:tt)*} ;
        (extern $name:ident $param:ident $len:tt) $($rest:tt)*
    ) => {
        $crate::__filter_lower! {
            [$st $lt $ctor $names]
            { $($fields)* }
            { $($inits)* }
            { $($params)* $param: &$lt mut [Scalar; $len], }
            { $($args)* $param.as_mut_slice().into(), }
            { $($inline)* }
            { $($extern)* + $len }
            ; $($rest)*
        }
    };
}

/// Emits the lowered `Storage`, footprint constants, `new` and `build`.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_lowered {
    (
        [$st:ident $lt:lifetime [$($ctor:tt)*] $names:literal]
        {$($fields:tt)*} {$($inits:tt)*} {$($params:tt)*} {$($args:tt)*}
        {$($inline:tt)*} {$($extern:tt)*}
    ) => {
        /// The inline part of the storage: every matrix not declared `extern`, and the
        /// temporaries. Exactly [`BYTES`], checked at compile time.
        #[allow(non_snake_case)]
        pub struct Storage {
            $($fields)*
        }

        /// The bytes of [`Storage`], the value you place.
        pub const BYTES: usize = (0 $($inline)*) * ::core::mem::size_of::<Scalar>();
        /// The bytes of the `extern` buffers [`build`] borrows.
        pub const EXTERN_BYTES: usize = (0 $($extern)*) * ::core::mem::size_of::<Scalar>();

        const _: () = ::core::assert!(
            ::core::mem::size_of::<Storage>() == BYTES,
            "the inline storage must be exactly its matrices"
        );

        #[doc(hidden)]
        pub const SYSTEM_MEMBER: bool = false;

        /// Creates zeroed storage; `const`, so it can initialize a `static`.
        pub const fn new() -> Storage {
            Storage { $($inits)* }
        }

        #[doc = ::core::concat!(
            "Wires the item over `storage` and the `extern` buffers, which come after it in the order ",
            $names, " (those declared `extern`), whatever order they were declared in. Only the ",
            "length is type-checked, so pass same-length buffers in the right order. ",
            "Writes nothing; see [`model`]."
        )]
        #[allow(non_snake_case)]
        pub fn build<$lt>($st: &$lt mut Storage, $($params)*) -> Handle<$lt> {
            $($ctor)*($($args)*)
        }
    };
}

/// Expands its input when minikalman has the `alloc` feature, for `storage = boxed`.
#[cfg(feature = "alloc")]
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_if_alloc {
    ($($items:tt)*) => { $($items)* };
}

/// Rejects `storage = boxed` when minikalman lacks the `alloc` feature.
#[cfg(not(feature = "alloc"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_if_alloc {
    ($($items:tt)*) => {
        ::core::compile_error!("filter!: `storage = boxed;` needs minikalman's `alloc` feature");
    };
}

/// Expands a `luenberger` item of [`filter!`](crate::filter) in the given gain form.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_luenberger {
    (
        [$(# $attr:tt)*] $vis:vis $module:ident $state:path
        [$storage:ident $handle:ident $form:ident $gain_kind:ident]
        { $($model:tt)* } { $( $obs:ident { $($gain:tt)* } )* }
    ) => {
        $(# $attr)*
        $vis mod $module {
            #[allow(unused_imports)]
            use super::*;
            #[allow(unused_imports)]
            use $crate::matrix::{MatrixMut as _, RowMajorSequentialDataMut as _, SquareMatrixMut as _};

            /// The scalar type, that of the state.
            pub type Scalar = <$state as $crate::__dsl::Named>::Scalar;
            /// The number of states.
            pub const STATES: usize = <$state>::N;
            /// The gain form of every observation of this observer.
            pub type Form = $crate::prelude::$form;

            /// The observer's storage and one fixed-gain observation storage per sensor.
            pub struct Storage {
                /// `A`, `x` and the predicted-state temporary.
                pub observer: $crate::regular::$storage<STATES, Scalar>,
                $(
                    #[doc = ::core::concat!("`H`, `z`, `y` and the gain of `", ::core::stringify!($obs), "`.")]
                    pub $obs: $crate::regular::FixedGainObservationStorage<STATES, { $obs::OBSERVATIONS }, Scalar, Form>,
                )*
            }

            const _: () = ::core::assert!(
                ::core::mem::size_of::<Storage>()
                    == ::core::mem::size_of::<$crate::regular::$storage<STATES, Scalar>>()
                    $(+ ::core::mem::size_of::<
                        $crate::regular::FixedGainObservationStorage<STATES, { $obs::OBSERVATIONS }, Scalar, Form>
                    >())*,
                "the observer storage must be exactly the sum of its parts"
            );

            /// The bytes of [`Storage`], the value you place.
            pub const BYTES: usize = ::core::mem::size_of::<Storage>();
            /// An observer has no `extern` buffers.
            pub const EXTERN_BYTES: usize = 0;
            #[doc(hidden)]
            pub const SYSTEM_MEMBER: bool = true;

            /// The observer and its fixed-gain observations.
            pub struct Handle<'a> {
                /// The observer.
                pub observer: $crate::regular::$handle<'a, STATES, Scalar>,
                $(
                    #[doc = ::core::concat!("The fixed-gain `", ::core::stringify!($obs), "` observation.")]
                    pub $obs: $crate::regular::BorrowedFixedGainObservation<'a, STATES, { $obs::OBSERVATIONS }, Scalar, Form>,
                )*
            }

            /// Creates zeroed storage; `const`, so it can initialize a `static`.
            pub const fn new() -> Storage {
                const ZERO: Scalar = <Scalar as $crate::num_traits::ConstZero>::ZERO;
                Storage {
                    observer: $crate::regular::$storage::new(ZERO),
                    $( $obs: $crate::regular::FixedGainObservationStorage::new(ZERO), )*
                }
            }

            /// Wires the observer and its observations over `storage`. Writes nothing beyond
            /// what the library's `build` does; see [`model`].
            pub fn build(storage: &mut Storage) -> Handle<'_> {
                Handle {
                    observer: storage.observer.build(),
                    $( $obs: storage.$obs.build(), )*
                }
            }

            /// Writes the declared `A` and `x`, each observation's `H` (from its declaration)
            /// and each non-empty declared gain.
            pub fn model(handle: &mut Handle<'_>) {
                let observer = &mut handle.observer;
                let _ = &observer;
                $crate::__filter_stmts!(luenberger [observer $state]; $($model)*);
                $(
                    handle.$obs.observation_matrix_mut().apply(|h| $obs::fill_h(h));
                    $crate::__filter_gain!($gain_kind [handle.$obs] $state, $obs; $($gain)*);
                )*
            }
        }
    };
}

/// Writes a declared fixed gain into an observation, or nothing for an empty gain block.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_gain {
    ($kind:ident [$($target:tt)*] $state:path, $obs:ident;) => {};
    ($kind:ident [$($target:tt)*] $state:path, $obs:ident; $($gain:tt)+) => {{
        let mut gain = $crate::matrix::MatrixData::new_array::<
            STATES,
            { $obs::OBSERVATIONS },
            { STATES * $obs::OBSERVATIONS },
            Scalar,
        >([<Scalar as $crate::num_traits::ConstZero>::ZERO; STATES * $obs::OBSERVATIONS]);
        $crate::__filter_stmts!($kind [gain $state, $obs::Value]; $($gain)+);
        $($target)*.set_gain(&gain);
    }};
}

/// Splits a model block into statements and routes each to its matrix.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_stmts {
    ($kind:ident $ctx:tt ;) => {};
    // Placement statements: read by `__filter_scan`, nothing to write here.
    ($kind:ident $ctx:tt ; storage = $mode:ident ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx storage (placement));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident = extern ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (placement));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident = identity ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (identity));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident = diag ( $e:expr ) ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (diag ($e)));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident [ $r:ident , $c:ident ] = $e:expr ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (set $r $c ($e)));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident = [ $($s:ident),* $(,)? ] ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (select [$($s)*]));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident = | $arg:ident | $body:expr ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (apply $arg ($body)));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $m:ident = $e:expr ; $($rest:tt)*) => {
        $crate::__filter_route!($kind $ctx $m (value ($e)));
        $crate::__filter_stmts!($kind $ctx ; $($rest)*);
    };
    ($kind:ident $ctx:tt ; $($bad:tt)+) => {
        ::core::compile_error!(::core::concat!(
            "filter!: expected `M = identity;`, `M = diag(v);`, `M[row, col] = v;`, ",
            "`H = [state, ...];`, `M = |m| ...;` or `x = value;`, found: ",
            ::core::stringify!($($bad)+)
        ));
    };
}

/// Maps a matrix name in a block of a given kind to its accessor and its row and column names.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_route {
    ($kind:ident $ctx:tt $m:ident (placement)) => {
        ::core::compile_error!(
            "filter!: `storage = …;` and `M = extern;` come first in a `kalman`, `observation` or `control` block, before the model statements; a `luenberger` block and its gains take neither"
        );
    };

    (kalman [$f:ident $s:path] A $op:tt) => { $crate::__filter_mat!(($f.state_transition_mut()) [$s] [$s] $op) };
    (kalman [$f:ident $s:path] Q $op:tt) => { $crate::__filter_mat!(($f.direct_process_noise_mut()) [$s] [$s] $op) };
    (kalman [$f:ident $s:path] P $op:tt) => { $crate::__filter_mat!(($f.estimate_covariance_mut()) [$s] [$s] $op) };
    (kalman [$f:ident $s:path] x $op:tt) => { $crate::__filter_vec!(($f.state_vector_mut()) [$s] $op) };

    (luenberger [$f:ident $s:path] A $op:tt) => { $crate::__filter_mat!(($f.state_transition_mut()) [$s] [$s] $op) };
    (luenberger [$f:ident $s:path] x $op:tt) => { $crate::__filter_vec!(($f.state_vector_mut()) [$s] $op) };

    (observation [$o:ident $s:path, $z:path] H $op:tt) => { $crate::__filter_mat!(($o.observation_matrix_mut()) [$z] [$s] $op) };
    (observation [$o:ident $s:path, $z:path] R $op:tt) => { $crate::__filter_mat!(($o.measurement_noise_covariance_mut()) [$z] [$z] $op) };
    (observation [$o:ident $s:path, $z:path] z $op:tt) => { $crate::__filter_vec!(($o.measurement_vector_mut()) [$z] $op) };

    // Only `H`, for `fill_h`; the other statements were routed (and checked) by `observation`.
    // `fill_h` takes any `MatrixMut`, which has no `make_identity`; write the diagonal by hand.
    (hmodel [$h:ident $s:path, $z:path] H (identity)) => {
        $crate::__filter_mat!(($h) [$z] [$s] (comatrix ($crate::num_traits::One::one())))
    };
    (hmodel [$h:ident $s:path, $z:path] H (diag ($e:expr))) => {
        $crate::__filter_mat!(($h) [$z] [$s] (comatrix ($e)))
    };
    (hmodel [$h:ident $s:path, $z:path] H $op:tt) => { $crate::__filter_mat!(($h) [$z] [$s] $op) };
    (hmodel $ctx:tt $m:ident $op:tt) => {};

    // Item-level checks, so `cargo check` and rust-analyzer report them too.
    (hcheck [$s:path, $z:path] H (select [$($x:ident)*])) => {
        const _: () = ::core::assert!(
            [$(::core::stringify!($x)),*].len() == <$z>::N,
            "`H = [...]` must name exactly one state per measurement"
        );
    };
    (hcheck $ctx:tt $m:ident $op:tt) => {};

    (control [$c:ident $s:path, $u:path] B $op:tt) => { $crate::__filter_mat!(($c.control_matrix_mut()) [$s] [$u] $op) };
    (control [$c:ident $s:path, $u:path] Q $op:tt) => { $crate::__filter_mat!(($c.process_noise_covariance_mut()) [$u] [$u] $op) };
    (control [$c:ident $s:path, $u:path] u $op:tt) => { $crate::__filter_vec!(($c.control_vector_mut()) [$u] $op) };

    (current_gain [$g:ident $s:path, $z:path] K $op:tt) => { $crate::__filter_mat!(($g) [$s] [$z] $op) };
    (current_gain $ctx:tt L $op:tt) => {
        ::core::compile_error!(
            "filter!: a current-form gain is `K`; `L` is the gain of a `luenberger predictor` block"
        )
    };
    (predictor_gain [$g:ident $s:path, $z:path] L $op:tt) => { $crate::__filter_mat!(($g) [$s] [$z] $op) };
    (predictor_gain $ctx:tt K $op:tt) => {
        ::core::compile_error!(
            "filter!: a predictor-form gain is `L`; `K` is the gain of a current-form `luenberger` block"
        )
    };

    ($kind:ident $ctx:tt $m:ident $op:tt) => {
        ::core::compile_error!(::core::concat!(
            "filter!: `", ::core::stringify!($m), "` is not a matrix of a `",
            ::core::stringify!($kind), "` block"
        ))
    };
}

/// One statement on a matrix whose rows and columns are named by two `state!` types.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_mat {
    (($m:expr) [$r:path] [$c:path] (identity)) => {
        $m.make_identity()
    };
    (($m:expr) [$r:path] [$c:path] (diag ($e:expr))) => {
        $m.make_scalar($e)
    };
    (($m:expr) [$r:path] [$c:path] (comatrix ($e:expr))) => {{
        let (matrix, diagonal) = ($m, $e);
        for row in 0..<$r>::N {
            for col in 0..<$c>::N {
                let value = if row == col { diagonal } else { $crate::num_traits::Zero::zero() };
                matrix.set_at(row, col, value);
            }
        }
    }};
    (($m:expr) [$r:path] [$c:path] (set $row:ident $col:ident ($e:expr))) => {
        $m.set_at(<$r>::$row, <$c>::$col, $e)
    };
    (($m:expr) [$r:path] [$c:path] (select [$($s:ident)*])) => {{
        let matrix = $m;
        // A whole-matrix assignment: clear stale entries so `model()` is a true reset.
        matrix.clear();
        let mut row = 0_usize;
        $(
            matrix.set_at(row, <$c>::$s, $crate::num_traits::One::one());
            row += 1;
        )*
        let _ = row;
    }};
    (($m:expr) [$r:path] [$c:path] (apply $arg:ident ($body:expr))) => {{
        let _ = $m.apply(|$arg| $body);
    }};
    (($m:expr) [$r:path] [$c:path] (value ($e:expr))) => {
        ::core::compile_error!(
            "filter!: set a matrix with `identity`, `diag(v)`, `[row, col] = v`, `[..]` or `|m| ...`"
        )
    };
}

/// One statement on a vector named by a `state!` type.
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_vec {
    (($v:expr) [$t:path] (value ($e:expr))) => {{
        let value: $t = $e;
        value.write($v.as_mut_slice());
    }};
    (($v:expr) [$t:path] $op:tt) => {
        ::core::compile_error!("filter!: set a vector with `x = Name { field: value, ... };`")
    };
}
