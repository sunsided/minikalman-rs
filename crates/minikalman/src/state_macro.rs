//! The [`state!`](crate::state) macro for named state and observation vectors.

/// Declares a named state or observation vector: a `Copy` struct with one field per entry, the
/// entry count, index constants, and copy-in/copy-out helpers for vectors and sigma point matrices.
///
/// ```
/// minikalman::state! {
///     /// Position and velocity in the plane.
///     pub struct Radar(f32) {
///         /// East position, in meters.
///         x,
///         /// North position, in meters.
///         y,
///         vx,
///         vy,
///     }
/// }
///
/// assert_eq!(Radar::N, 4);
/// assert_eq!((Radar::x, Radar::y, Radar::vx, Radar::vy), (0, 1, 2, 3));
///
/// let mut vector = [0.0_f32; Radar::N];
/// Radar { x: 1.0, y: 2.0, vx: 3.0, vy: 4.0 }.write(&mut vector);
/// assert_eq!(vector, [1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(Radar::read(&vector).vy, 4.0);
/// ```
///
/// The macro expands to:
///
/// * the struct, with `pub` fields of the given scalar type and `Clone`, `Copy`, `Debug` and
///   `PartialEq`;
/// * `N`, the number of entries, usable as a const generic argument, as in
///   `FilterStorage<{ Radar::N }, f32>`;
/// * one `usize` constant per field holding its index (`Radar::vx == 2`), for matrix entries such
///   as `h.set_at(RadarZ::bearing, Radar::x, value)`;
/// * `read` and `write` for vectors such as `x` and `z` (pass `as_ref()` / `as_mut()` of
///   a state vector buffer, or the closure arguments of the nonlinear filters);
/// * `read_column` and `write_column` for column `j` of a row-major `N × cols` matrix, the layout of
///   the unscented filter's sigma points.
///
/// Copying in and out costs nothing after inlining for small states and needs no `unsafe`.
///
/// ## Using it in a model
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use minikalman::extended::builder::KalmanFilterBuilder;
/// use minikalman::prelude::*;
///
/// minikalman::state! { struct Track(f32) { position, velocity } }
/// minikalman::state! { struct Range(f32) { distance } }
///
/// const DT: f32 = 0.1;
///
/// let builder = KalmanFilterBuilder::<{ Track::N }, f32>::default();
/// let mut filter = builder.build();
/// let mut measurement = builder.observations().build::<{ Range::N }>();
///
/// Track { position: 0.0, velocity: 1.0 }.write(filter.state_vector_mut().as_mut());
/// filter.estimate_covariance_mut().make_identity();
/// filter.state_transition_jacobian_mut().apply(|a| {
///     a.make_identity();
///     a.set_at(Track::position, Track::velocity, DT);
/// });
/// measurement
///     .observation_jacobian_matrix_mut()
///     .apply(|h| h.set_at(Range::distance, Track::position, 1.0));
/// measurement.measurement_noise_covariance_mut().make_scalar(0.5);
///
/// filter.predict_nonlinear(|x, next| {
///     let s = Track::read(x.as_ref());
///     Track { position: s.position + s.velocity * DT, ..s }.write(next.as_mut());
/// });
/// Range { distance: 0.12 }.write(measurement.measurement_vector_mut().as_mut());
/// filter.correct_nonlinear(&mut measurement, |x, z| {
///     Range { distance: Track::read(x.as_ref()).position }.write(z.as_mut());
/// });
///
/// let track = Track::read(filter.state_vector().as_ref());
/// assert!(track.position > 0.0);
/// # }
/// ```
///
/// ## Syntax
///
/// `$vis struct Name(Scalar) { field, ... }`, with optional attributes and doc comments on the
/// struct and on each field. The scalar comes in parentheses because `macro_rules!` cannot
/// follow a type with a brace group. Any `Copy` scalar works, e.g. `I16F16` with the `fixed`
/// feature.
///
/// ## Conditional compilation
///
/// Attributes inside the macro apply to the struct or field only, so they are for docs,
/// derives and lints. `#[cfg]` and `#[cfg_attr]` inside the macro would remove a struct or
/// field while the generated constants and helpers still refer to it, so the macro rejects
/// them with a compile error. Put the `#[cfg]` on the macro invocation instead; it gates the
/// whole expansion:
///
/// ```
/// #[cfg(feature = "std")]
/// minikalman::state! { pub struct Radar(f32) { x, y } }
///
/// #[cfg(feature = "std")]
/// assert_eq!(Radar::N, 2);
/// ```
///
/// ```compile_fail
/// minikalman::state! { #[cfg(feature = "std")] pub struct Radar(f32) { x, y } }
/// ```
///
/// ```compile_fail
/// minikalman::state! { pub struct Radar(f32) { x, #[cfg(feature = "std")] y } }
/// ```
///
/// ## Limits
///
/// * The field names `N`, `read`, `write`, `read_column` and `write_column` are reserved, since the
///   index constants share the namespace of the generated items.
/// * The index constants are the field names as written (`Radar::vx`), so the expansion allows
///   `non_upper_case_globals`.
/// * There is no per-field metadata such as initial values, variances or angle wrapping.
///
/// ## Panics
///
/// `read` and `write` panic when the slice has fewer than `N` entries, and `read_column` and
/// `write_column` when an index `field * cols + j` is out of bounds.
#[macro_export]
macro_rules! state {
    (
        $(# $attr:tt)*
        $vis:vis struct $name:ident ($scalar:ty) $fields:tt
    ) => {
        $crate::state!(@define [$(# $attr)*] $vis $name ($scalar) $fields $fields);
    };
    (
        @define [$(# $attr:tt)*] $vis:vis $name:ident ($scalar:ty)
        { $( $(# $field_attr:tt)* $field:ident ),+ $(,)? }
        $all:tt
    ) => {
        $( $crate::state!(@reject_cfg $attr); )*
        $( $( $crate::state!(@reject_cfg $field_attr); )* )+

        $(# $attr)*
        #[derive(
            ::core::clone::Clone,
            ::core::marker::Copy,
            ::core::fmt::Debug,
            ::core::cmp::PartialEq,
        )]
        $vis struct $name {
            $( $(# $field_attr)* pub $field: $scalar, )+
        }

        #[allow(non_upper_case_globals, dead_code)]
        impl $name {
            /// The number of entries.
            pub const N: usize = [$(::core::stringify!($field)),+].len();

            $(
                #[doc = ::core::concat!("The index of `", ::core::stringify!($field), "`.")]
                pub const $field: usize = $crate::state!(@index $all $field);
            )+

            /// Reads the entries from the first `N` elements of a vector.
            ///
            /// # Panics
            /// Panics when `vector` has fewer than `N` elements.
            #[inline(always)]
            pub fn read(vector: &[$scalar]) -> Self {
                Self { $( $field: vector[Self::$field], )+ }
            }

            /// Writes the entries to the first `N` elements of a vector.
            ///
            /// # Panics
            /// Panics when `vector` has fewer than `N` elements.
            #[inline(always)]
            pub fn write(&self, vector: &mut [$scalar]) {
                $( vector[Self::$field] = self.$field; )+
            }

            /// Reads column `j` of a row-major `N × cols` matrix, such as a sigma point matrix.
            ///
            /// # Panics
            /// Panics when an index `field × cols + j` is out of bounds.
            #[inline(always)]
            pub fn read_column<M>(matrix: &M, cols: usize, j: usize) -> Self
            where
                M: ::core::ops::Index<usize, Output = $scalar> + ?Sized,
            {
                Self { $( $field: matrix[Self::$field * cols + j], )+ }
            }

            /// Writes column `j` of a row-major `N × cols` matrix, such as the observed sigma
            /// points.
            ///
            /// # Panics
            /// Panics when an index `field × cols + j` is out of bounds.
            #[inline(always)]
            pub fn write_column<M>(&self, matrix: &mut M, cols: usize, j: usize)
            where
                M: ::core::ops::IndexMut<usize, Output = $scalar> + ?Sized,
            {
                $( matrix[Self::$field * cols + j] = self.$field; )+
            }
        }
    };
    (@reject_cfg [cfg $($rest:tt)*]) => {
        ::core::compile_error!(
            "`#[cfg]` inside `state!` is not supported; put it on the `state!` invocation"
        );
    };
    (@reject_cfg [cfg_attr $($rest:tt)*]) => {
        ::core::compile_error!(
            "`#[cfg_attr]` inside `state!` is not supported; put it on the `state!` invocation"
        );
    };
    (@reject_cfg [$($rest:tt)*]) => {};
    (@index { $( $(# $field_attr:tt)* $all:ident ),+ $(,)? } $field:ident) => {{
        #[allow(non_camel_case_types, dead_code)]
        enum Index { $($all),+ }
        Index::$field as usize
    }};
}
