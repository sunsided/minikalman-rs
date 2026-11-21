//! Ready-made system models.
//!
//! * [`motion`] - State transition and process noise for common motion models.
//! * [`gauss_markov`] - First-order Gauss-Markov processes for sensor bias and colored noise.

pub mod gauss_markov;
pub mod motion;
