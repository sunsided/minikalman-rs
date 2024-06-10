// only enables the `doc_cfg` feature when
// the `docsrs` configuration attribute is defined
#![cfg_attr(docsrs, feature(doc_cfg))]
// Enable `no_std` if the `no_std` crate feature is enabled.
#![cfg_attr(not(any(feature = "std", feature = "alloc")), no_std)]
// Forbid unsafe code.
#![forbid(unsafe_code)]
// Attempt to disable allocations.
#![cfg_attr(not(feature = "alloc"), forbid(box_pointers))]

#[cfg(any(feature = "std", feature = "alloc"))]
extern crate alloc;

pub mod kalman;
pub mod matrix;
