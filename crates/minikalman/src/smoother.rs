//! # Rauch-Tung-Striebel (RTS) Smoother
//!
//! A filter estimates the state at step `k` from the measurements up to `k`. When the whole
//! run is available, e.g. for logged trajectories or calibration runs, the
//! [`RtsSmoother`] refines every estimate with the measurements that came after it by running
//! a backward pass over the recorded filter history.
//!
//! For `k = N−2` down to `0`, with `x̂ₖ = x_k|k` and `Pₖ = P_k|k` the filtered values and
//! `x̂⁻ₖ₊₁ = x_k+1|k` and `P⁻ₖ₊₁ = P_k+1|k` the predicted values of the next step:
//!
//! ```text
//! Cₖ     = Pₖ × Aᵀ × (P⁻ₖ₊₁)⁻¹
//! x_k|N  = x̂ₖ + Cₖ × (x_k+1|N − x̂⁻ₖ₊₁)
//! P_k|N  = Pₖ + Cₖ × (P_k+1|N − P⁻ₖ₊₁) × Cₖᵀ
//! ```
//!
//! The last step is already smoothed (`x_N−1|N = x_N−1|N−1`).
//!
//! ## Example
//!
//! Record the predicted state after `predict()` (and any `control()`) and the filtered state
//! after `correct()`, then smooth the history in place:
//!
//! ```
//! # #[cfg(feature = "alloc")] {
//! use minikalman::matrix::MatrixDataArray;
//! use minikalman::models::motion;
//! use minikalman::prelude::*;
//! use minikalman::regular::builder::KalmanFilterBuilder;
//! use minikalman::smoother::RtsSmoother;
//!
//! const N: usize = 4;
//!
//! type Vector = MatrixDataArray<2, 1, 2, f32>;
//! type Square = MatrixDataArray<2, 2, 4, f32>;
//!
//! let builder = KalmanFilterBuilder::<2, f32>::default();
//! let mut filter = builder.build();
//! let mut measurement = builder.observations().build::<1>();
//!
//! let model = motion::constant_velocity(1.0, 0.1);
//! filter.state_transition_mut().apply(|a| model.fill_transition(a));
//! filter.direct_process_noise_mut().apply(|q| model.fill_process_noise(q));
//! filter.estimate_covariance_mut().apply(|p| p.make_scalar(10.0));
//! measurement.observation_matrix_mut().apply(|h| h.set(0, 0, 1.0));
//! measurement.measurement_noise_covariance_mut().apply(|r| r.set(0, 0, 1.0));
//!
//! let mut x = vec![Vector::new([0.0; 2]); N];
//! let mut p = vec![Square::new([0.0; 4]); N];
//! let mut x_predicted = x.clone();
//! let mut p_predicted = p.clone();
//!
//! for (k, z) in [0.1, 1.2, 1.9, 3.1].into_iter().enumerate() {
//!     filter.predict();
//!     filter.state_vector().inspect(|v| v.copy_to(&mut x_predicted[k]));
//!     filter.estimate_covariance().inspect(|m| m.copy_to(&mut p_predicted[k]));
//!
//!     measurement.measurement_vector_mut().apply(|v| v.set(0, 0, z));
//!     filter.correct(&mut measurement);
//!     filter.state_vector().inspect(|v| v.copy_to(&mut x[k]));
//!     filter.estimate_covariance().inspect(|m| m.copy_to(&mut p[k]));
//! }
//!
//! let mut smoother = RtsSmoother::new(
//!     Square::new([0.0; 4]),
//!     Square::new([0.0; 4]),
//!     Square::new([0.0; 4]),
//!     Vector::new([0.0; 2]),
//! );
//! let all_smoothed = filter.state_transition().inspect(|a| {
//!     smoother.smooth(a, &mut x, &mut p, &x_predicted, &p_predicted)
//! });
//! assert!(all_smoothed);
//! # }
//! ```

use crate::matrix::{Matrix, MatrixDataType, MatrixMut, SquareMatrix};

/// Runs the Rauch-Tung-Striebel backward pass over a recorded filter history.
///
/// Holds only the scratch buffers of the pass, so it does not allocate and can be reused for
/// several runs. The state transition matrix `A` is constant over the run, matching the
/// [`RegularKalman`](crate::regular::RegularKalman) filter, which also keeps a single `A`.
pub struct RtsSmoother<const STATES: usize, T, Gain, Inv, Temp, TempX> {
    /// The smoother gain `Cₖ = Pₖ × Aᵀ × (P⁻ₖ₊₁)⁻¹`, `STATES × STATES`.
    gain: Gain,
    /// The inverse `(P⁻ₖ₊₁)⁻¹`, later reused for `Cₖ × (P_k+1|N − P⁻ₖ₊₁)`.
    inverse: Inv,
    /// Scratch for the Cholesky decomposition, `Pₖ × Aᵀ` and `P_k+1|N − P⁻ₖ₊₁`.
    temp: Temp,
    /// Scratch for the state difference `x_k+1|N − x̂⁻ₖ₊₁`.
    temp_x: TempX,
    /// Ties the element type to the buffers.
    _phantom: core::marker::PhantomData<T>,
}

impl<const STATES: usize, T, Gain, Inv, Temp, TempX> RtsSmoother<STATES, T, Gain, Inv, Temp, TempX>
where
    Gain: MatrixMut<STATES, STATES, T>,
    Inv: MatrixMut<STATES, STATES, T>,
    Temp: MatrixMut<STATES, STATES, T>,
    TempX: MatrixMut<STATES, 1, T>,
    T: MatrixDataType,
{
    /// Creates a smoother from its scratch buffers: three `STATES × STATES` matrices for the
    /// gain, the inverse and a temporary, and one `STATES × 1` vector.
    pub fn new(gain: Gain, inverse: Inv, temp: Temp, temp_x: TempX) -> Self {
        Self {
            gain,
            inverse,
            temp,
            temp_x,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Smooths a recorded filter history in place.
    ///
    /// All slices are indexed by step `k = 0..N`:
    /// * `x[k]`, `p[k]` — the filtered `x_k|k`, `P_k|k`, recorded after `correct()`.
    ///   Overwritten with the smoothed `x_k|N`, `P_k|N`.
    /// * `x_predicted[k]`, `p_predicted[k]` — the predicted `x_k|k−1`, `P_k|k−1` of the same
    ///   step, recorded after `predict()` (and any `control()`) but before `correct()`.
    ///   Entry `0` is not read.
    ///
    /// A history of zero or one step is left unchanged.
    ///
    /// ## Singular predictions
    /// If a predicted covariance `P⁻ₖ₊₁` is not positive definite, it cannot be inverted. That
    /// step keeps its filtered `x̂ₖ` and `Pₖ`, and the pass continues from there to the earlier
    /// steps. This mirrors how a correction is skipped when `S = H×P×Hᵀ + R` is singular.
    ///
    /// ## Returns
    /// `true` if every step was smoothed, `false` if at least one step was skipped.
    ///
    /// ## Panics
    /// Panics if the slices do not all have the same length.
    #[allow(non_snake_case)]
    pub fn smooth<A, X, P, XP, PP>(
        &mut self,
        A: &A,
        x: &mut [X],
        p: &mut [P],
        x_predicted: &[XP],
        p_predicted: &[PP],
    ) -> bool
    where
        A: Matrix<STATES, STATES, T>,
        X: MatrixMut<STATES, 1, T>,
        P: MatrixMut<STATES, STATES, T>,
        XP: Matrix<STATES, 1, T>,
        PP: Matrix<STATES, STATES, T>,
    {
        let n = x.len();
        assert_eq!(p.len(), n, "filtered covariance history length");
        assert_eq!(x_predicted.len(), n, "predicted state history length");
        assert_eq!(p_predicted.len(), n, "predicted covariance history length");

        let mut all_smoothed = true;
        for k in (1..n).rev() {
            let (x_head, x_tail) = x.split_at_mut(k);
            let (p_head, p_tail) = p.split_at_mut(k);
            all_smoothed &= self.step(
                A,
                &mut x_head[k - 1],
                &mut p_head[k - 1],
                &x_tail[0],
                &p_tail[0],
                &x_predicted[k],
                &p_predicted[k],
            );
        }

        all_smoothed
    }

    /// Smooths one step `k` from the smoothed step `k+1` (`x_next`, `p_next`) and its
    /// prediction (`x_pred`, `p_pred`). Returns `false`, leaving `x` and `p` unchanged, if
    /// `p_pred` is not positive definite.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn step(
        &mut self,
        A: &impl Matrix<STATES, STATES, T>,
        x: &mut impl MatrixMut<STATES, 1, T>,
        p: &mut impl MatrixMut<STATES, STATES, T>,
        x_next: &impl Matrix<STATES, 1, T>,
        p_next: &impl Matrix<STATES, STATES, T>,
        x_pred: &impl Matrix<STATES, 1, T>,
        p_pred: &impl Matrix<STATES, STATES, T>,
    ) -> bool {
        // inverse = (P⁻ₖ₊₁)⁻¹
        p_pred.copy_to(&mut self.temp);
        if !self.temp.cholesky_decompose_lower() {
            return false;
        }
        self.temp.invert_l_cholesky(&mut self.inverse);

        // C = Pₖ × Aᵀ × inverse
        p.mult_transb(A, &mut self.temp);
        self.temp.mult(&self.inverse, &mut self.gain);

        // x += C × (x_next − x_pred)
        x_next.sub(x_pred, &mut self.temp_x);
        self.gain.multadd_rowvector(&self.temp_x, x);

        // P += C × (P_next − P_pred) × Cᵀ
        p_next.sub(p_pred, &mut self.temp);
        self.gain.mult(&self.temp, &mut self.inverse);
        self.inverse.multadd_transb(&self.gain, p);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::{MatrixData, MatrixDataArray};

    /// A 1×1 matrix.
    type Scalar = MatrixDataArray<1, 1, 1, f32>;

    /// Creates a scalar smoother with zeroed scratch buffers.
    fn scalar_smoother() -> RtsSmoother<1, f32, Scalar, Scalar, Scalar, Scalar> {
        RtsSmoother::new(s(0.0), s(0.0), s(0.0), s(0.0))
    }

    /// Wraps a value into a 1×1 matrix.
    fn s(value: f32) -> Scalar {
        MatrixData::new_array::<1, 1, 1, f32>([value])
    }

    /// Scalar random walk `x₁ = x₀ + w`, `z₁ = x₁ + v` with prior `x₀ ~ N(m₀, P₀)`, no measurement
    /// at step 0 and `P₀ = q = r = ½`. The Bayes posterior of `x₀` given `z₁` has mean
    /// `m₀ + P₀/(P₀+q+r) × (z₁ − m₀) = m₀ + (z₁ − m₀)/3` and variance `P₀(q+r)/(P₀+q+r) = 1/3`.
    #[test]
    fn random_walk_matches_closed_form_posterior() {
        let (m0, z1) = (2.0, 5.0);

        // Step 1 prediction: x⁻ = m₀, P⁻ = P₀ + q = 1. Correction: K = 1/1.5 = 2/3.
        // P⁻ = 1 keeps micromath's approximate sqrt and recip exact.
        let x1 = m0 + 2.0 / 3.0 * (z1 - m0);
        let p1 = 1.0 / 3.0;

        let mut x = [s(m0), s(x1)];
        let mut p = [s(0.5), s(p1)];
        let x_pred = [s(0.0), s(m0)];
        let p_pred = [s(0.0), s(1.0)];

        assert!(scalar_smoother().smooth(&s(1.0), &mut x, &mut p, &x_pred, &p_pred));

        assert!((x[0].get(0, 0) - (m0 + (z1 - m0) / 3.0)).abs() < 1e-3);
        assert!((p[0].get(0, 0) - 1.0 / 3.0).abs() < 1e-3);
        assert!((x[1].get(0, 0) - x1).abs() < 1e-6);
        assert!((p[1].get(0, 0) - p1).abs() < 1e-6);
    }

    /// A zero predicted covariance cannot be inverted; the step keeps its filtered values.
    #[test]
    fn singular_prediction_keeps_filtered_values() {
        let mut x = [s(1.0), s(3.0)];
        let mut p = [s(0.0), s(0.0)];
        let x_pred = [s(0.0), s(1.0)];
        let p_pred = [s(0.0), s(0.0)];

        assert!(!scalar_smoother().smooth(&s(1.0), &mut x, &mut p, &x_pred, &p_pred));

        assert_eq!(x[0].get(0, 0), 1.0);
        assert_eq!(p[0].get(0, 0), 0.0);
        assert_eq!(x[1].get(0, 0), 3.0);
    }

    /// Empty and single-step histories are left unchanged.
    #[test]
    fn short_histories_are_unchanged() {
        let mut smoother = scalar_smoother();
        let (mut x, mut p, none): ([Scalar; 0], [Scalar; 0], [Scalar; 0]) = ([], [], []);
        assert!(smoother.smooth(&s(1.0), &mut x, &mut p, &none, &none));

        let mut x = [s(4.0)];
        let mut p = [s(2.0)];
        assert!(smoother.smooth(&s(1.0), &mut x, &mut p, &[s(0.0)], &[s(0.0)]));
        assert_eq!(x[0].get(0, 0), 4.0);
        assert_eq!(p[0].get(0, 0), 2.0);
    }
}
