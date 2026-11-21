//! The [`filter!`](crate::filter) macro: a declaration language for states, observations,
//! controls, scratch, filters and systems of them, over the storage types.

/// Support items for the code [`filter!`](crate::filter) generates. Not part of the public API.
pub mod support {
    /// The scalar type of a named vector declared in a [`filter!`](crate::filter) block.
    ///
    /// Implemented by the generated `state`, observation and control value types, so items
    /// declared `for` a state find its scalar without the user repeating it.
    pub trait Named {
        /// The scalar of every entry, e.g. `f32`.
        type Scalar;
    }

    /// The largest value of a list, `0` for an empty one: the `MAX_OBSERVATIONS` of a
    /// `scratch` item.
    #[must_use]
    pub const fn max(values: &[usize]) -> usize {
        let mut max = 0;
        let mut i = 0;
        while i < values.len() {
            if values[i] > max {
                max = values[i];
            }

            i += 1;
        }

        max
    }

    #[cfg(test)]
    mod tests {
        use super::max;

        #[test]
        fn max_of_a_list() {
            assert_eq!(max(&[]), 0);
            assert_eq!(max(&[3]), 3);
            assert_eq!(max(&[1, 4, 2]), 4);
        }
    }
}

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
/// ```
/// use minikalman::models::motion::constant_velocity;
/// use minikalman::prelude::*;
///
/// const DT: f32 = 0.1;
///
/// minikalman::filter! {
///     /// Position and velocity on a line.
///     pub state Track(f32) { pos, vel }
///
///     /// A position fix.
///     pub observation gps: Gps for Track { pos } {
///         H = [pos];
///         R = diag(0.5);
///     }
///
///     /// Commanded acceleration.
///     pub control throttle: Throttle for Track { accel } {
///         B[pos, accel] = 0.5 * DT * DT;
///         B[vel, accel] = DT;
///         Q = diag(0.01);
///     }
///
///     pub kalman tracker for Track {
///         A = |a| constant_velocity(DT, 1.0).fill_transition(a);
///         Q = |q| constant_velocity(DT, 1.0).fill_process_noise(q);
///         P = identity;
///     }
///
///     /// Everything in one value.
///     pub system tracking { tracker, gps, throttle }
/// }
///
/// fn main() {
///     let mut storage = tracking::new(); // or a `static`, see below
///     let mut sys = tracking::build(&mut storage); // wiring only
///     tracking::model(&mut sys); // writes A, Q, P, H, R, B and Qᵤ
///
///     for k in 0..50 {
///         let t = k as f32 * DT;
///         Throttle { accel: 1.0 }.write(sys.throttle.control_vector_mut().as_mut_slice());
///         sys.tracker.predict().control(&mut sys.throttle);
///         Gps { pos: 0.5 * t * t }.write(sys.gps.measurement_vector_mut().as_mut_slice());
///         sys.tracker.correct(&mut sys.gps);
///     }
///
///     let track = Track::read(sys.tracker.state_vector().as_slice());
///     assert!((track.vel - 4.9).abs() < 0.5);
/// }
/// ```
///
/// Invoke it at module level, not in a function body: each generated module imports its
/// parent with `use super::*`, which is how the statements see your constants and functions
/// (`DT`, `constant_velocity` above).
///
/// ## Grammar
///
/// ```text
/// filter!     := item*
/// item        := attr* vis? ( state | observation | control | scratch
///                           | kalman | luenberger | system )
///
/// state       := 'state' Name '(' Scalar ')' '{' field,* '}'
/// observation := 'observation' 'lean'? mod ':' Name 'for' StatePath
///                '{' field,* '}' '{' stmt* '}'
/// control     := 'control' mod ':' Name 'for' StatePath '{' field,* '}' '{' stmt* '}'
/// scratch     := 'scratch' mod 'for' '[' obsmod,+ ']' ';'
/// kalman      := 'kalman' mod 'for' StatePath '{' stmt* '}'
/// luenberger  := 'luenberger' 'predictor'? mod 'for' StatePath '{' stmt* '}'
///                'gains' '{' ( obsmod '{' stmt* '}' )* '}'
/// system      := 'system' mod '{' member,+ '}'
///
/// field       := attr* ident                    -- doc comments allowed, as in state!
/// stmt        := M '=' 'identity' ';'
///              | M '=' 'diag' '(' expr ')' ';'
///              | M '[' row ',' col ']' '=' expr ';'
///              | M '=' '[' state,* ']' ';'      -- H only: row i selects state i
///              | M '=' '|' ident '|' expr ';'   -- any filler, e.g. models::*
///              | v '=' expr ';'                 -- vectors x, z, u: a Name { .. } value
/// ```
///
/// The matrices each block accepts, and the names that index their rows and columns:
///
/// | Block | Matrix | Rows × columns named by |
/// |---|---|---|
/// | `kalman` | `A`, `P`, `Q` / vector `x` | state × state / state |
/// | `luenberger` | `A` / vector `x` | state × state / state |
/// | `luenberger … gains { gps { … } }` | `K` (current form), `L` (`predictor` form) | state × `gps::Value` |
/// | `observation` | `H` / `R` / vector `z` | measurement × state / measurement × measurement / measurement |
/// | `control` | `B` / `Q` / vector `u` | state × control / control × control / control |
///
/// So in `A[pos, vel] = DT;` both names are constants of the state, in `H[pos, pos]` the row
/// is `Gps::pos` and the column `Track::pos`, and in `B[vel, accel]` the column is
/// `Throttle::accel`. The role of the matrix picks the namespace; you never qualify a name.
///
/// ## What each item expands to
///
/// | Item | Module contents |
/// |---|---|
/// | `state Track(f32) { pos, vel }` | no module: a [`state!`](crate::state) struct `Track` |
/// | `observation gps: Gps for Track { pos } { … }` | `Storage` = [`ObservationStorage`](crate::regular::ObservationStorage), `Handle` = [`BorrowedObservation`](crate::regular::BorrowedObservation), `Value`, `STATES`, `OBSERVATIONS`, `fill_h`; plus a `state!` struct `Gps` |
/// | `observation lean gps: …` | the same over [`LeanObservationStorage`](crate::regular::LeanObservationStorage) and [`BorrowedLeanObservation`](crate::regular::BorrowedLeanObservation) |
/// | `control throttle: Throttle for Track { accel } { … }` | `Storage` = [`ControlStorage`](crate::regular::ControlStorage), `Handle` = [`BorrowedControl`](crate::regular::BorrowedControl), `Value`, `STATES`, `CONTROLS`; plus a `state!` struct `Throttle` |
/// | `scratch shared for [gps, lidar];` | `Storage` = [`CorrectionScratch`](crate::regular::CorrectionScratch) sized for the largest, `Handle` = `&mut Storage`, `MAX_OBSERVATIONS` |
/// | `kalman fast for Track { … }` | `Storage` = [`FilterStorage`](crate::regular::FilterStorage), `Handle` = [`BorrowedKalman`](crate::regular::BorrowedKalman) |
/// | `luenberger steady for Track { … } gains { gps { … } }` | struct `Storage { observer, gps }` of [`ObserverStorage`](crate::regular::ObserverStorage) and [`FixedGainObservationStorage`](crate::regular::FixedGainObservationStorage), struct `Handle { observer, gps }` |
/// | `luenberger predictor steady …` | the same over [`PredictorObserverStorage`](crate::regular::PredictorObserverStorage) and [`Predictor`](crate::prelude::Predictor)-form observations |
/// | `system tracking { fast, gps, … }` | struct `Storage` of the members' `Storage`, struct `Handle` of their `Handle` |
///
/// Every module (all items but `state`) has `Scalar`, `STATES`, `pub const fn new() ->
/// Storage`, `pub fn build(&mut Storage) -> Handle<'_>` and `pub fn model(&mut Handle<'_>)`.
/// The generated code contains no `unsafe`, no `static` and no allocation, only paths you
/// could write by hand; `cargo expand` shows it.
///
/// ## `build()` wires, `model()` writes
///
/// `build()` is pure wiring, exactly like the storage types' own `build()`: it writes nothing,
/// so what the storage held stays. `model()` writes the declared values (`A`, `Q`, `P`, `x₀`,
/// `H`, `R`, `B`, `Qᵤ`, `u`, `z` and the gains) and nothing else:
///
/// * Call it once after the first `build()`; `system::model` calls the model of every member.
/// * Call it again to reset a filter to its declared model and initial values.
/// * It writes only what is declared: `P = identity;` sets every entry of `P`, while
///   `A[pos, vel] = DT;` sets one entry and leaves the others as they are. Entries you never
///   declare (e.g. an `x` without `x = …`) keep their value, which is zero after `new()`.
/// * In a `luenberger` block, `model()` writes each observation's `H` from its declaration
///   (`fill_h`) and the gain only when the gain block is non-empty, so `gains { gps {} }`
///   keeps a gain you set yourself, e.g. one from `settle_gain`.
///
/// The declared values are code that runs at start-up, not data: the storage stays zeroed,
/// so a `static` of it lands in `.bss`.
///
/// ## Observations: full or lean
///
/// An `observation` owns its temporaries by default, so it works with every correction:
/// `correct`, `correct_with_gain`, `settle_gain` and the robust variants. `observation lean`
/// stores only `H`, `z`, `R`, `y`, `S` and `K`; correct it through `correct_in` with a
/// `scratch` that lists it. One scratch serves every lean observation it lists, of every
/// filter over the same state:
///
/// ```
/// use minikalman::prelude::*;
///
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation lean gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///     observation lean odometry: Odometry for Track { pos, vel } {
///         H = identity;
///         R = diag(0.1);
///     }
///     scratch shared for [gps, odometry];
///     kalman tracker for Track { A = identity; A[pos, vel] = 0.1; P = identity; Q = diag(0.01); }
///     system tracking { tracker, gps, odometry, shared }
/// }
///
/// fn main() {
///     let mut storage = tracking::new();
///     let mut sys = tracking::build(&mut storage);
///     tracking::model(&mut sys);
///     assert_eq!(shared::MAX_OBSERVATIONS, 2);
///
///     sys.tracker.predict();
///     Gps { pos: 1.0 }.write(sys.gps.measurement_vector_mut().as_mut_slice());
///     assert!(sys.tracker.correct_in(&mut sys.gps, sys.shared).is_applied());
/// }
/// ```
///
/// ## Sharing
///
/// * **Objects.** A `system` handle is a struct of disjoint `pub` fields, so
///   `sys.fast.correct(&mut sys.gps)` borrows `sys.fast` and `sys.gps` separately, and the
///   same observation and control can serve several filters over the same state, one call at
///   a time. Each correction rewrites `y`, `S` and `K` in the observation, so they describe the
///   last filter corrected; use one observation per filter when you read the NIS or the gain
///   per filter.
/// * **Declarations.** Items refer to each other only by Rust paths (`gps::OBSERVATIONS`,
///   `gps::fill_h`, `fast::Storage`), so rustc does the name resolution and the shape checks,
///   and items can be split across `filter!` blocks, modules and crates. Declare the state,
///   observations and controls once, and the filters wherever they are used: a `for` clause
///   takes any path, resolved from inside the generated module, so write a name in scope or
///   an absolute path (`kalman third for crate::shared::Track { … }`, not `super::…`), while `scratch`,
///   `luenberger … gains` and `system` name their members as modules in scope, so `use` the
///   shared ones first (`use crate::shared::{gps, throttle};`).
/// * A `luenberger` block shares the *declaration* of an observation (its value type and its
///   `H`), not the object: its fixed-gain observations are a different library type.
///
/// ## Memory and placement
///
/// Every `Storage` is the library storage type or a plain struct of them. A `luenberger` and
/// a `system` storage carry a compile-time assert that their size is exactly the sum of their
/// parts. It can fail for a `system` whose members mix scalars of different alignment, e.g.
/// `f32` and `f64`, when the struct needs padding; use one `system` per scalar.
///
/// `filter!` creates no `static` and allocates nothing: you place one value, on the stack as
/// above or in a `static` through a take-once cell such as `static_cell::ConstStaticCell` (not
/// a dependency of this crate) rather than a `static mut`:
///
/// ```ignore
/// static STORAGE: ConstStaticCell<tracking::Storage> = ConstStaticCell::new(tracking::new());
///
/// let mut sys = tracking::build(STORAGE.take()); // .bss, nothing moved at runtime
/// tracking::model(&mut sys);
/// ```
///
/// An RTIC `#[local]` resource or a `Box::new(tracking::new())` work the same way. Each item's
/// `Storage` can also be placed on its own, e.g. the scratch on the stack and the filters in
/// a separate RAM section.
///
/// ## Fixed-gain observers
///
/// A `luenberger` block runs a [`LuenbergerObserver`](crate::regular::LuenbergerObserver): no
/// covariance, a fixed gain per observation, written in the current form `K` by default or in
/// the predictor form `L` with `luenberger predictor`. Writing `L` in a current-form block (or
/// `K` in a predictor-form one) is a compile error, as is correcting with a gain of the other
/// form. The observation's `H` comes from its `observation` declaration:
///
/// ```
/// use minikalman::prelude::*;
///
/// const DT: f32 = 0.1;
///
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; R = diag(0.5); }
///
///     luenberger steady for Track {
///         A = identity;
///         A[pos, vel] = DT;
///     } gains {
///         gps { K[pos, pos] = 0.5; K[vel, pos] = 0.8; }
///     }
/// }
///
/// fn main() {
///     let mut storage = steady::new();
///     let mut observer = steady::build(&mut storage);
///     steady::model(&mut observer);
///
///     observer.observer.predict();
///     Gps { pos: 1.0 }.write(observer.gps.measurement_vector_mut().as_mut_slice());
///     assert!(observer.observer.correct(&mut observer.gps).is_applied());
///     assert_eq!(Track::read(observer.observer.state_vector().as_slice()).pos, 0.5);
/// }
/// ```
///
/// ## Fixed point
///
/// Any [`MatrixDataType`](crate::matrix::MatrixDataType) scalar that implements
/// [`num_traits::ConstZero`] works; for the `fixed` feature, that needs `fixed` 1.28 or newer.
/// The statements take the values as written, so write typed constants:
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
/// ## Compile-time checks
///
/// Shapes are never written as numbers: they come from the `state!` entry counts, and matrix
/// entries are addressed only by name. What remains is checked at compile time.
///
/// An unknown index name is rustc's own E0599, "no … named `vz` found for struct `Track`",
/// pointing at the name:
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     kalman bad for Track { A[pos, vz] = 0.1; }
/// }
/// fn main() {}
/// ```
///
/// An `H = [...]` that does not name one state per measurement fails with "`H = [...]` must
/// name exactly one state per measurement", also under `cargo check`:
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation bad: Bad for Track { pos } { H = [pos, vel]; }
/// }
/// fn main() {}
/// ```
///
/// A matrix of another block fails with "filter!: `R` is not a matrix of a `kalman` block":
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     kalman bad for Track { R = diag(0.5); }
/// }
/// fn main() {}
/// ```
///
/// A gain written in the other form fails with "filter!: a current-form gain is `K`; `L` is
/// the gain of a `luenberger predictor` block" (and the reverse):
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     observation gps: Gps for Track { pos } { H = [pos]; }
///     luenberger bad for Track { A = identity; } gains { gps { L[pos, pos] = 0.5; } }
/// }
/// fn main() {}
/// ```
///
/// An observation of another state type fails on the library's trait bound
/// (`KalmanFilterObservationCorrectFilter<2, f32>` is not satisfied for an observation of 3
/// states):
///
/// ```compile_fail
/// minikalman::filter! {
///     state Track(f32) { pos, vel }
///     state Pose(f32) { x, y, heading }
///     observation compass: Compass for Pose { heading } { H = [heading]; }
///     kalman fast for Track { A = identity; }
/// }
///
/// fn main() {
///     let (mut filter_storage, mut compass_storage) = (fast::new(), compass::new());
///     let mut filter = fast::build(&mut filter_storage);
///     let mut compass = compass::build(&mut compass_storage);
///     filter.correct(&mut compass);
/// }
/// ```
///
/// A misspelled keyword fails with "filter!: expected `state`, `observation`, …, found: …".
/// Syntax errors point at the whole block rather than at the bad token; the `found:` text
/// shows where parsing stopped.
///
/// ## Limits
///
/// * Only the regular Kalman filter and the Luenberger observer, so far; the other filters
///   are wired by hand from their storage types.
/// * Matrices are set by the statement forms above; there are no matrix literals.
/// * Closures stay at the call site (`predict_nonlinear` and friends are not generated), and
///   named indices inside a closure body are written as `m.set_at(Track::pos, Track::vel, v)`.
/// * Statements and `for` paths expand inside the generated module: they see the parent's
///   items (through `use super::*`), not function locals, and `self::` / `super::` are
///   relative to the generated module. Its own names (`Storage`, `Handle`, `Scalar`, `STATES`,
///   `new`, `build`, `model`, `fill_h`, …) shadow parent items of the same name.
/// * Put `#[cfg]` on the `filter!` invocation, not on an item inside it.
/// * The statements of one item expand recursively, about 120 per item at the default
///   `recursion_limit`.
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
        $crate::__filter_observation! {
            [$(# $attr)*] $vis $module $value $state
            [LeanObservationStorage BorrowedLeanObservation]
            "`H`, `z`, `R`, `y`, `S` and `K`; correct it with `correct_in` and a `scratch`."
            { $($fields)* } { $($model)* }
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis observation $module:ident : $value:ident for $state:path
            { $($fields:tt)* } { $($model:tt)* }
        $($rest:tt)*
    ) => {
        $crate::__filter_observation! {
            [$(# $attr)*] $vis $module $value $state
            [ObservationStorage BorrowedObservation]
            "`H`, `z`, `R`, `y`, `S`, `K` and the correction temporaries."
            { $($fields)* } { $($model)* }
        }

        $crate::filter! { $($rest)* }
    };

    (
        $(# $attr:tt)*
        $vis:vis control $module:ident : $value:ident for $state:path
            { $($fields:tt)* } { $($model:tt)* }
        $($rest:tt)*
    ) => {
        $crate::state! {
            $(# $attr)*
            $vis struct $value(<$state as $crate::__dsl::Named>::Scalar) { $($fields)* }
        }

        impl $crate::__dsl::Named for $value {
            type Scalar = <$state as $crate::__dsl::Named>::Scalar;
        }

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
            /// The library storage: `B`, `u`, `Q` and `B×Q`.
            pub type Storage = $crate::regular::ControlStorage<STATES, CONTROLS, Scalar>;
            /// The control [`build`] returns.
            pub type Handle<'a> = $crate::regular::BorrowedControl<'a, STATES, CONTROLS, Scalar>;

            /// Creates zeroed storage; `const`, so it can initialize a `static`.
            pub const fn new() -> Storage {
                Storage::new(<Scalar as $crate::num_traits::ConstZero>::ZERO)
            }

            /// Wires the control over `storage`. Writes nothing; see [`model`].
            pub fn build(storage: &mut Storage) -> Handle<'_> {
                storage.build()
            }

            /// Writes the declared `B`, `Q` and `u` into the control.
            pub fn model(control: &mut Handle<'_>) {
                let _ = &control;
                $crate::__filter_stmts!(control [control $state, $value]; $($model)*);
            }
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
            /// The library storage: `A`, `x`, `P`, `Q` and two temporaries.
            pub type Storage = $crate::regular::FilterStorage<STATES, Scalar>;
            /// The filter [`build`] returns.
            pub type Handle<'a> = $crate::regular::BorrowedKalman<'a, STATES, Scalar>;

            /// Creates zeroed storage; `const`, so it can initialize a `static`.
            pub const fn new() -> Storage {
                Storage::new(<Scalar as $crate::num_traits::ConstZero>::ZERO)
            }

            /// Wires the filter over `storage`. Writes nothing; see [`model`].
            pub fn build(storage: &mut Storage) -> Handle<'_> {
                storage.build()
            }

            /// Writes the declared `A`, `Q`, `P` and `x` into the filter, e.g. to reset it.
            pub fn model(filter: &mut Handle<'_>) {
                let _ = &filter;
                $crate::__filter_stmts!(kalman [filter $state]; $($model)*);
            }
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

/// Expands an `observation` item of [`filter!`](crate::filter) over the given storage and
/// handle types (full or lean).
#[doc(hidden)]
#[macro_export]
macro_rules! __filter_observation {
    (
        [$(# $attr:tt)*] $vis:vis $module:ident $value:ident $state:path
        [$storage:ident $handle:ident] $contents:literal
        { $($fields:tt)* } { $($model:tt)* }
    ) => {
        $crate::state! {
            $(# $attr)*
            $vis struct $value(<$state as $crate::__dsl::Named>::Scalar) { $($fields)* }
        }

        impl $crate::__dsl::Named for $value {
            type Scalar = <$state as $crate::__dsl::Named>::Scalar;
        }

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
            #[doc = ::core::concat!("The library storage: ", $contents)]
            pub type Storage = $crate::regular::$storage<STATES, OBSERVATIONS, Scalar>;
            /// The observation [`build`] returns.
            pub type Handle<'a> = $crate::regular::$handle<'a, STATES, OBSERVATIONS, Scalar>;

            /// Creates zeroed storage; `const`, so it can initialize a `static`.
            pub const fn new() -> Storage {
                Storage::new(<Scalar as $crate::num_traits::ConstZero>::ZERO)
            }

            /// Wires the observation over `storage`. Writes nothing; see [`model`].
            pub fn build(storage: &mut Storage) -> Handle<'_> {
                storage.build()
            }

            $crate::__filter_stmts!(hcheck [$state, $value]; $($model)*);

            /// Writes the declared `H`, `R` and `z` into the observation.
            pub fn model(observation: &mut Handle<'_>) {
                let _ = &observation;
                $crate::__filter_stmts!(observation [observation $state, $value]; $($model)*);
            }

            /// Writes the declared `H` into any `OBSERVATIONS × STATES` matrix, e.g. the `H` of
            /// a fixed-gain observation of the same sensor.
            pub fn fill_h<M>(h: &mut M)
            where
                M: $crate::matrix::MatrixMut<OBSERVATIONS, STATES, Scalar> + ?Sized,
            {
                let _ = &h;
                $crate::__filter_stmts!(hmodel [h $state, $value]; $($model)*);
            }
        }
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
