//! Conformance suite: checks that the filters compute what the math says they compute.
//!
//! See `README.md` next to this file for the categories, the pass criteria and how to
//! regenerate the reference vectors. Run it with `cargo test -p minikalman --test conformance`.

#![cfg(feature = "alloc")]
#![forbid(unsafe_code)]

mod baselines;
mod consistency;
mod identities;
mod invariants;
mod monte_carlo;
#[rustfmt::skip]
mod reference;
mod reference_vectors;
mod scenarios;
mod support;
