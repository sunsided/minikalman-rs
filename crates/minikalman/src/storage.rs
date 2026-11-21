//! Const-generic storage for the filter buffers, without allocation and without macros.
//!
//! Each storage type keeps the buffers of one filter part (filter, observation, control,
//! smoother) as nested arrays `[[T; COLS]; ROWS]`. A nested array has exactly the bytes of the
//! flat `[T; ROWS * COLS]` buffer that the `impl_buffer_*` macros declare. `build(&mut self)`
//! flattens each array with the safe `as_flattened_mut` and wires the existing buffer types over
//! [`MatrixDataMut`](crate::matrix::MatrixDataMut) slices into the storage, so the resulting
//! filter borrows the storage and moves no arrays.
//!
//! The types are re-exported from the filter modules, e.g. [`crate::regular::FilterStorage`].

pub(crate) mod extended;
pub(crate) mod information;
mod layout;
pub(crate) mod luenberger;
pub(crate) mod regular;
pub(crate) mod smoother;
pub(crate) mod unscented;
