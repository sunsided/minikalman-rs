//! Zero-copy named views of state and measurement vectors, behind the `derive` feature.
//!
//! [`state!`](crate::state) copies entries in and out of a vector. This module instead views
//! the vector's memory as a `#[repr(C)]` struct derived with [`UniformArray`], so reads and
//! writes through the struct's fields go straight to the filter's buffer:
//!
//! ```
//! # #[cfg(feature = "unsafe")] {
//! use minikalman::prelude::*;
//! use minikalman::regular::FilterStorage;
//! use minikalman::UniformArray;
//!
//! /// Height, velocity and acceleration of a falling object.
//! #[derive(UniformArray)]
//! #[repr(C)]
//! struct Fall {
//!     height: f32,
//!     velocity: f32,
//!     acceleration: f32,
//! }
//!
//! minikalman::named_vector!(Fall);
//!
//! let mut storage = FilterStorage::<3, f32>::new(0.0);
//! let mut filter = storage.build();
//!
//! filter.state_vector_mut().view_mut::<Fall>().acceleration = 9.81;
//! assert_eq!(filter.state_vector().as_slice(), &[0.0, 0.0, 9.81]);
//! assert_eq!(filter.state_vector().view::<Fall>().acceleration, 9.81);
//! # }
//! ```
//!
//! ## What the defining crate needs
//!
//! The derive emits `Fall::from_slice` and `Fall::from_mut_slice`, the `unsafe` casts behind
//! the view, only when the crate that defines `Fall` enables its own `unsafe` feature
//! (`#[uniform_array(safety_gate = "...")]` picks another name). Declare it in that crate:
//!
//! ```toml
//! [dependencies]
//! minikalman = { version = "0.8", features = ["derive"] }
//!
//! [features]
//! default = ["unsafe"]
//! unsafe = []
//! ```
//!
//! minikalman's own `unsafe` feature is unrelated, and minikalman itself does no cast: it
//! calls the derived functions through [`NamedVector`], which [`named_vector!`](macro@crate::named_vector)
//! implements. Without the gate, `named_vector!` fails to compile with "no associated
//! function or constant named `from_slice`". With the gate, the derive rejects a struct without
//! `#[repr(C)]` or `#[repr(transparent)]`, and a struct whose fields differ in type.
//!
//! The examples below set `safety_gate = "derive"` because a doctest's features are
//! minikalman's, and `derive` is the one always on when this module exists. In your crate, the
//! gate must name a feature that crate declares and enables: keep the default `unsafe` from the
//! setup above, or declare a feature with the name you pass.
//!
//! A struct without `#[repr(C)]` fails to compile:
//!
//! ```compile_fail
//! #[derive(minikalman::UniformArray)]
//! #[uniform_array(safety_gate = "derive")]
//! struct Fall {
//!     height: f32,
//!     velocity: f32,
//! }
//! ```
//!
//! The same struct with `#[repr(C)]` compiles:
//!
//! ```
//! #[derive(minikalman::UniformArray)]
//! #[uniform_array(safety_gate = "derive")]
//! #[repr(C)]
//! struct Fall {
//!     height: f32,
//!     velocity: f32,
//! }
//!
//! minikalman::named_vector!(Fall);
//! ```
//!
//! ## Length
//!
//! A view of a vector whose length differs from the struct's field count fails to build. The
//! check is a constant evaluated when the call is instantiated, so `cargo build` and
//! `cargo test` report it, but `cargo check` may not:
//!
//! ```compile_fail
//! # use minikalman::prelude::*;
//! #[derive(minikalman::UniformArray)]
//! #[uniform_array(safety_gate = "derive")]
//! #[repr(C)]
//! struct Fall {
//!     height: f32,
//!     velocity: f32,
//! }
//!
//! minikalman::named_vector!(Fall);
//!
//! let mut storage = minikalman::regular::FilterStorage::<3, f32>::new(0.0);
//! let filter = storage.build();
//! let _ = filter.state_vector().view::<Fall>().height;
//! ```
//!
//! With two states, the view compiles:
//!
//! ```
//! # use minikalman::prelude::*;
//! #[derive(minikalman::UniformArray)]
//! #[uniform_array(safety_gate = "derive")]
//! #[repr(C)]
//! struct Fall {
//!     height: f32,
//!     velocity: f32,
//! }
//!
//! minikalman::named_vector!(Fall);
//!
//! let mut storage = minikalman::regular::FilterStorage::<2, f32>::new(0.0);
//! let filter = storage.build();
//! assert_eq!(filter.state_vector().view::<Fall>().height, 0.0);
//! ```
//!
//! ## Together with `state!`
//!
//! [`state!`](crate::state) stays the safe, copying route and provides the index constants
//! for matrix entries. For the view, declare a derived struct next to it with the same field
//! names, and use `Radar::vx` from `state!` for matrices and `view::<RadarView>().vx` for the
//! vector. Passing `#[derive(UniformArray)]` into `state!` does not compile yet: the derive's
//! type check loses hygiene when the field types come from a `macro_rules!` expansion.
//!
//! ## Limits
//!
//! * Views cover column vectors (`N × 1`), such as state, measurement and control vectors and
//!   the arguments of the nonlinear filters' closures. Sigma point columns are strided, so they
//!   stay with `state!`'s `read_column` and `write_column`.
//! * The derive and [`ensure_uniform_type`](macro@crate::ensure_uniform_type) need Rust 1.88, above the crate's MSRV of 1.85.

use crate::matrix::{RowMajorSequentialData, RowMajorSequentialDataMut};

pub use ensure_uniform_type::ensure_uniform_type;
pub use uniform_array_derive::UniformArray;

/// A struct that can stand in for a vector of `T` without copying.
///
/// Implement it with [`named_vector!`](macro@crate::named_vector) for a struct derived with
/// [`UniformArray`]. The trait is safe: implementations only borrow the slice they are given.
pub trait NamedVector<T>: Sized {
    /// The number of entries; for a derived struct, its field count.
    const N: usize = size_of::<Self>() / size_of::<T>();

    /// Views exactly [`N`](Self::N) entries as this struct.
    ///
    /// # Panics
    /// The derived implementation panics when `vector` does not have `N` entries.
    fn from_vector(vector: &[T]) -> &Self;

    /// Views exactly [`N`](Self::N) entries as this struct, mutably.
    ///
    /// # Panics
    /// The derived implementation panics when `vector` does not have `N` entries.
    fn from_vector_mut(vector: &mut [T]) -> &mut Self;
}

/// Views a column vector, such as a filter's state vector, as a [`NamedVector`].
pub trait VectorView<const N: usize, T>: RowMajorSequentialData<N, 1, T> {
    /// Borrows the vector as `V`.
    ///
    /// Fails to build when `V` has a different number of entries than the vector.
    #[inline(always)]
    fn view<'a, V>(&'a self) -> &'a V
    where
        V: NamedVector<T>,
        T: 'a,
    {
        const { assert!(V::N == N, "the struct and the vector differ in length") };
        V::from_vector(&self.as_slice()[..N])
    }
}

impl<const N: usize, T, X> VectorView<N, T> for X where X: RowMajorSequentialData<N, 1, T> + ?Sized {}

/// Views a mutable column vector, such as a filter's state vector, as a [`NamedVector`].
pub trait VectorViewMut<const N: usize, T>: RowMajorSequentialDataMut<N, 1, T> {
    /// Borrows the vector mutably as `V`; writes to its fields go to the vector.
    ///
    /// Fails to build when `V` has a different number of entries than the vector.
    #[inline(always)]
    fn view_mut<'a, V>(&'a mut self) -> &'a mut V
    where
        V: NamedVector<T>,
        T: 'a,
    {
        const { assert!(V::N == N, "the struct and the vector differ in length") };
        V::from_vector_mut(&mut self.as_mut_slice()[..N])
    }
}

impl<const N: usize, T, X> VectorViewMut<N, T> for X where
    X: RowMajorSequentialDataMut<N, 1, T> + ?Sized
{
}

/// Implements [`NamedVector`] for a struct derived with [`UniformArray`], through the derived
/// `from_slice` and `from_mut_slice`.
///
/// The derive emits those only when the defining crate enables the derive's safety gate
/// (its `unsafe` feature by default); without it, this macro fails to compile. See the
/// [`named_vector`](mod@crate::named_vector) module for an example.
#[macro_export]
macro_rules! named_vector {
    ($ty:ty) => {
        impl $crate::NamedVector<<$ty as ::core::ops::Index<usize>>::Output> for $ty {
            #[inline(always)]
            fn from_vector(vector: &[<$ty as ::core::ops::Index<usize>>::Output]) -> &Self {
                <$ty>::from_slice(vector)
            }

            #[inline(always)]
            fn from_vector_mut(
                vector: &mut [<$ty as ::core::ops::Index<usize>>::Output],
            ) -> &mut Self {
                <$ty>::from_mut_slice(vector)
            }
        }
    };
}
