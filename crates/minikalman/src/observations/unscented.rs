//! # Observations for Unscented Kalman Filters.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{
    AsMatrix, AsMatrixMut, Matrix, MatrixDataType, MatrixMut, RowMajorSequentialDataMut,
    SquareMatrix,
};

/// Kalman Filter measurement structure for Unscented Kalman Filter.
#[allow(non_snake_case)]
pub struct UnscentedObservation<
    const STATES: usize,
    const OBSERVATIONS: usize,
    const NUM_SIGMA: usize,
    T,
    SigmaObserved,
    CrossCov,
    Z,
    R,
    Y,
    S,
    K,
    TempSInv,
    TempP,
> {
    z: Z,
    R: R,
    y: Y,
    S: S,
    K: K,
    temp_S_inv: TempSInv,
    sigma_observed: SigmaObserved,
    cross_covariance: CrossCov,
    temp_P: TempP,
    _phantom: PhantomData<T>,
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
    UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
{
    pub const fn states(&self) -> usize {
        STATES
    }

    pub const fn observations(&self) -> usize {
        OBSERVATIONS
    }

    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn new(
        z: Z,
        R: R,
        y: Y,
        S: S,
        K: K,
        temp_S_inv: TempSInv,
        sigma_observed: SigmaObserved,
        cross_covariance: CrossCov,
        temp_P: TempP,
    ) -> Self {
        Self {
            z,
            R,
            y,
            S,
            K,
            temp_S_inv,
            sigma_observed,
            cross_covariance,
            temp_P,
            _phantom: PhantomData,
        }
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
    UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    Z: MeasurementVector<OBSERVATIONS, T>,
{
    #[inline(always)]
    pub fn measurement_vector(&self) -> &Z {
        &self.z
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
    UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
{
    #[inline(always)]
    pub fn measurement_vector_mut(&mut self) -> &mut Z {
        &mut self.z
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
    UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    #[inline(always)]
    pub fn measurement_noise_covariance(&self) -> &R {
        &self.R
    }

    #[inline(always)]
    pub fn measurement_noise_covariance_mut(&mut self) -> &mut R {
        &mut self.R
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
    UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    T: MatrixDataType
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::FromPrimitive
        + Default,
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    SigmaObserved: SigmaObservedMatrix<OBSERVATIONS, NUM_SIGMA, T>,
    CrossCov: CrossCovarianceMatrix<STATES, OBSERVATIONS, T>,
    TempP: TempSigmaPMatrix<STATES, T>,
{
    #[allow(non_snake_case)]
    pub fn correct_nonlinear<X, P, SP>(&mut self, x: &mut X, P: &mut P, sigma_propagated: &SP)
    where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
    {
        let z_pred = self.compute_predicted_measurement();
        self.compute_innovation_covariance(&z_pred);
        self.compute_cross_covariance(&z_pred, sigma_propagated);
        self.compute_kalman_gain();
        self.apply_correction(x, P, &z_pred, &mut |_: &mut [T]| {});
    }

    /// Correct using explicit weights (for proper UKF weighted averaging).
    #[allow(non_snake_case)]
    pub fn correct_with_weights<X, P, SP, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_propagated: &SP,
        weights: &W,
        lambda: T,
    ) where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
    {
        self.correct_with_weights_residuals(x, P, sigma_propagated, weights, lambda, |_| {}, |_| {})
    }

    /// Correct using explicit weights, normalizing state and measurement differences
    /// before they are used.
    ///
    /// * `fix_state` normalizes differences between states (one entry per state).
    /// * `fix_measurement` normalizes differences between measurements (one entry per
    ///   measurement).
    ///
    /// Means are taken relative to the centre sigma point, `x̄ = Σw·σ₀ + Σ wₖ·fix(σₖ − σ₀)`,
    /// so values on either side of a wrap point average correctly. Without normalization
    /// this equals the plain weighted mean.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub fn correct_with_weights_residuals<X, P, SP, W, NX, NZ>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_propagated: &SP,
        weights: &W,
        lambda: T,
        mut fix_state: NX,
        mut fix_measurement: NZ,
    ) where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        NX: FnMut(&mut [T]),
        NZ: FnMut(&mut [T]),
    {
        let n = T::from_usize(STATES).unwrap();
        let w0_m = lambda / (n + lambda);
        let z_pred = anchored_mean(
            self.sigma_observed.as_matrix(),
            weights,
            w0_m,
            &mut fix_measurement,
        );
        self.compute_innovation_covariance_weighted(&z_pred, weights, &mut fix_measurement);
        self.compute_cross_covariance_weighted(
            &z_pred,
            sigma_propagated,
            weights,
            w0_m,
            &mut fix_state,
            &mut fix_measurement,
        );
        self.compute_kalman_gain();
        self.apply_correction(x, P, &z_pred, &mut fix_measurement);
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_predicted_measurement(&mut self) -> [T; OBSERVATIONS] {
        let num_sigma = NUM_SIGMA;
        let sigma_obs = self.sigma_observed.as_matrix();

        let mut z_pred = [T::default(); OBSERVATIONS];
        for i in 0..OBSERVATIONS {
            let mut sum = T::default();
            for j in 0..num_sigma {
                sum += sigma_obs.get(i, j);
            }
            z_pred[i] = sum / T::from_usize(num_sigma).unwrap();
        }
        z_pred
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_innovation_covariance_weighted<W, NZ>(
        &mut self,
        z_pred: &[T; OBSERVATIONS],
        weights: &W,
        fix_measurement: &mut NZ,
    ) where
        W: AsMatrix<NUM_SIGMA, 1, T>,
        NZ: FnMut(&mut [T]),
    {
        let sigma_obs = self.sigma_observed.as_matrix();
        let S = self.S.as_matrix_mut();
        let R = self.R.as_matrix();
        let w = weights.as_matrix();

        S.set_zero();
        let mut dz = [T::default(); OBSERVATIONS];
        for k in 0..NUM_SIGMA {
            for i in 0..OBSERVATIONS {
                dz[i] = sigma_obs.get(i, k) - z_pred[i];
            }
            fix_measurement(&mut dz);

            let w_val = w.get(k, 0);
            for i in 0..OBSERVATIONS {
                for j in 0..OBSERVATIONS {
                    S.set(i, j, S.get(i, j) + w_val * dz[i] * dz[j]);
                }
            }
        }

        R.add_inplace_b(S);
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_cross_covariance_weighted<SP, W, NX, NZ>(
        &mut self,
        z_pred: &[T; OBSERVATIONS],
        sigma_propagated: &SP,
        weights: &W,
        w0_m: T,
        fix_state: &mut NX,
        fix_measurement: &mut NZ,
    ) where
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        NX: FnMut(&mut [T]),
        NZ: FnMut(&mut [T]),
    {
        let cross_cov = self.cross_covariance.as_matrix_mut();
        let sigma_obs = self.sigma_observed.as_matrix();
        let sigma_prop = sigma_propagated.as_matrix();
        let w = weights.as_matrix();

        let x_mean = anchored_mean(sigma_prop, weights, w0_m, fix_state);

        cross_cov.set_zero();
        let mut dx = [T::default(); STATES];
        let mut dz = [T::default(); OBSERVATIONS];
        for k in 0..NUM_SIGMA {
            for i in 0..STATES {
                dx[i] = sigma_prop.get(i, k) - x_mean[i];
            }
            fix_state(&mut dx);
            for j in 0..OBSERVATIONS {
                dz[j] = sigma_obs.get(j, k) - z_pred[j];
            }
            fix_measurement(&mut dz);

            let w_val = w.get(k, 0);
            for i in 0..STATES {
                for j in 0..OBSERVATIONS {
                    cross_cov.set(i, j, cross_cov.get(i, j) + w_val * dx[i] * dz[j]);
                }
            }
        }
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_innovation_covariance(&mut self, z_pred: &[T; OBSERVATIONS]) {
        let num_sigma = NUM_SIGMA;
        let sigma_obs = self.sigma_observed.as_matrix();
        let S = self.S.as_matrix_mut();
        let R = self.R.as_matrix();

        for i in 0..OBSERVATIONS {
            for j in 0..OBSERVATIONS {
                let mut sum = T::default();
                for k in 0..num_sigma {
                    let diff_i = sigma_obs.get(i, k) - z_pred[i];
                    let diff_j = sigma_obs.get(j, k) - z_pred[j];
                    sum += diff_i * diff_j;
                }
                S.set(i, j, sum / T::from_usize(num_sigma).unwrap());
            }
        }

        R.add_inplace_b(S);
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_cross_covariance<SP>(&mut self, z_pred: &[T; OBSERVATIONS], sigma_propagated: &SP)
    where
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
    {
        let num_sigma = NUM_SIGMA;
        let cross_cov = self.cross_covariance.as_matrix_mut();
        let sigma_obs = self.sigma_observed.as_matrix();
        let sigma_prop = sigma_propagated.as_matrix();

        // Compute mean of propagated sigma points (simple average)
        let mut x_mean = [T::default(); STATES];
        for i in 0..STATES {
            let mut sum = T::default();
            for k in 0..num_sigma {
                sum += sigma_prop.get(i, k);
            }
            x_mean[i] = sum / T::from_usize(num_sigma).unwrap();
        }

        for i in 0..STATES {
            for j in 0..OBSERVATIONS {
                let mut sum = T::default();
                for k in 0..num_sigma {
                    let diff_x = sigma_prop.get(i, k) - x_mean[i];
                    let diff_z = sigma_obs.get(j, k) - z_pred[j];
                    sum += diff_x * diff_z;
                }
                cross_cov.set(i, j, sum / T::from_usize(num_sigma).unwrap());
            }
        }
    }

    #[allow(non_snake_case)]
    fn compute_kalman_gain(&mut self) {
        let S = self.S.as_matrix_mut();
        let S_inv = self.temp_S_inv.as_matrix_mut();
        let cross_cov = self.cross_covariance.as_matrix();
        let K = self.K.as_matrix_mut();

        S.cholesky_decompose_lower();
        S.invert_l_cholesky(S_inv);

        cross_cov.mult_transb(S_inv, K);
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn apply_correction<XVec, PCov, NZ>(
        &mut self,
        x: &mut XVec,
        P: &mut PCov,
        z_pred: &[T; OBSERVATIONS],
        fix_measurement: &mut NZ,
    ) where
        XVec: StateVectorMut<STATES, T>,
        PCov: EstimateCovarianceMatrix<STATES, T>,
        NZ: FnMut(&mut [T]),
    {
        let z = self.z.as_matrix();
        let y = self.y.as_matrix_mut();
        let K = self.K.as_matrix();

        for i in 0..OBSERVATIONS {
            y.set(i, 0, z.get(i, 0) - z_pred[i]);
        }
        fix_measurement(y.as_mut_slice());

        let x_mat = x.as_matrix_mut();
        for i in 0..STATES {
            let mut correction = T::default();
            for j in 0..OBSERVATIONS {
                correction += K.get(i, j) * y.get(j, 0);
            }
            x_mat.set(i, 0, x_mat.get(i, 0) + correction);
        }

        let cross_cov = self.cross_covariance.as_matrix();
        let K = self.K.as_matrix();
        let P_mat = P.as_matrix_mut();
        let temp_P = self.temp_P.as_matrix_mut();

        cross_cov.mult_transb(K, temp_P);
        P_mat.sub_inplace_a(temp_P);
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterNumStates<STATES>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
{
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterNumObservations<OBSERVATIONS>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
{
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterMeasurementVector<OBSERVATIONS, T>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    Z: MeasurementVector<OBSERVATIONS, T>,
{
    type MeasurementVector = Z;

    #[inline(always)]
    fn measurement_vector(&self) -> &Self::MeasurementVector {
        self.measurement_vector()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
{
    type MeasurementVectorMut = Z;

    #[inline(always)]
    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut {
        self.measurement_vector_mut()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterMeasurementNoiseCovariance<OBSERVATIONS, T>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    type MeasurementNoiseCovarianceMatrix = R;

    #[inline(always)]
    fn measurement_noise_covariance(&self) -> &Self::MeasurementNoiseCovarianceMatrix {
        self.measurement_noise_covariance()
    }
}

impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterMeasurementNoiseCovarianceMut<OBSERVATIONS, T>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
{
    type MeasurementNoiseCovarianceMatrixMut = R;

    #[inline(always)]
    fn measurement_noise_covariance_mut(
        &mut self,
    ) -> &mut Self::MeasurementNoiseCovarianceMatrixMut {
        self.measurement_noise_covariance_mut()
    }
}

#[allow(non_snake_case)]
impl<
        const STATES: usize,
        const OBSERVATIONS: usize,
        const NUM_SIGMA: usize,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    > KalmanFilterUnscentedObservationCorrectFilter<STATES, OBSERVATIONS, NUM_SIGMA, T>
    for UnscentedObservation<
        STATES,
        OBSERVATIONS,
        NUM_SIGMA,
        T,
        SigmaObserved,
        CrossCov,
        Z,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempP,
    >
where
    T: MatrixDataType
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Div<Output = T>
        + num_traits::FromPrimitive
        + Default,
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
    R: MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    S: InnovationCovarianceMatrix<OBSERVATIONS, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    TempSInv: TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>,
    SigmaObserved:
        SigmaObservedMatrix<OBSERVATIONS, NUM_SIGMA, T> + AsMatrixMut<OBSERVATIONS, NUM_SIGMA, T>,
    CrossCov: CrossCovarianceMatrix<STATES, OBSERVATIONS, T>,
    TempP: TempSigmaPMatrix<STATES, T>,
{
    type ObservedSigmaPoints = SigmaObserved;

    fn correct_with_observed_residuals<X, P, SP, F, W, NX, NZ>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_propagated: &SP,
        weights: &W,
        lambda: T,
        mut observation: F,
        fix_state: NX,
        fix_measurement: NZ,
    ) where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        F: FnMut(&SP, &mut Self::ObservedSigmaPoints),
        NX: FnMut(&mut [T]),
        NZ: FnMut(&mut [T]),
    {
        observation(sigma_propagated, &mut self.sigma_observed);
        self.correct_with_weights_residuals(
            x,
            P,
            sigma_propagated,
            weights,
            lambda,
            fix_state,
            fix_measurement,
        );
    }
}

/// Weighted mean of the columns of `points`, taken relative to the first (centre) column:
/// `Σw·c₀ + Σₖ wₖ·fix(cₖ − c₀)`.
///
/// `weights` holds the covariance weights; the centre column uses the mean weight `w0_m`
/// instead. Without normalization, this equals the plain weighted mean `Σₖ wₖ·cₖ`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn anchored_mean<const ROWS: usize, const COLS: usize, T, M, W, N>(
    points: &M,
    weights: &W,
    w0_m: T,
    fix: &mut N,
) -> [T; ROWS]
where
    T: MatrixDataType + Default,
    M: Matrix<ROWS, COLS, T>,
    W: AsMatrix<COLS, 1, T>,
    N: FnMut(&mut [T]),
{
    let w = weights.as_matrix();

    let mut weight_sum = w0_m;
    let mut offset = [T::default(); ROWS];
    let mut diff = [T::default(); ROWS];
    for k in 1..COLS {
        let w_val = w.get(k, 0);
        weight_sum += w_val;
        for i in 0..ROWS {
            diff[i] = points.get(i, k) - points.get(i, 0);
        }
        fix(&mut diff);
        for i in 0..ROWS {
            offset[i] += w_val * diff[i];
        }
    }

    let mut mean = [T::default(); ROWS];
    for i in 0..ROWS {
        mean[i] = weight_sum * points.get(i, 0) + offset[i];
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::angles;
    use crate::buffers::types::{SigmaPropagatedMatrixBuffer, SigmaWeightsVectorBuffer};
    use crate::matrix::{MatrixData, RowMajorSequentialData};

    #[test]
    fn test_dummy() {
        assert_eq!(2 * 3 + 1, 7);
    }

    #[test]
    fn anchored_mean_without_fix_is_weighted_mean() {
        let points = MatrixData::new_array::<1, 3, 3, f64>([1.0, 2.0, 4.0]);
        // The first weight is the covariance weight and must be ignored for the mean.
        let weights = SigmaWeightsVectorBuffer::<3, f64, _>::from([99.0, 0.4, 0.4]);
        let mean = anchored_mean(&points, &weights, 0.2, &mut |_: &mut [f64]| {});
        assert!((mean[0] - (0.2 * 1.0 + 0.4 * 2.0 + 0.4 * 4.0)).abs() < 1e-12);
    }

    #[test]
    fn anchored_mean_wraps_across_pi() {
        // 3.0 and -3.0 rad are 0.28 rad apart across π, not 6 rad apart across 0.
        let points = MatrixData::new_array::<1, 3, 3, f64>([3.0, -3.0, 3.1]);
        let third = 1.0 / 3.0;
        let weights = SigmaWeightsVectorBuffer::<3, f64, _>::from([third, third, third]);

        let wrapped = anchored_mean(&points, &weights, third, &mut |d: &mut [f64]| {
            d[0] = angles::wrap_pi(d[0])
        });
        let expected = 3.0 + (2.0 * core::f64::consts::PI - 6.0 + 0.1) / 3.0;
        assert!((wrapped[0] - expected).abs() < 1e-12, "{}", wrapped[0]);

        let plain = anchored_mean(&points, &weights, third, &mut |_: &mut [f64]| {});
        assert!(
            (plain[0] - (3.0 - 3.0 + 3.1) / 3.0).abs() < 1e-12,
            "{}",
            plain[0]
        );
    }

    /// Five sigma points around `x = (1, 2)` and the weights for `λ = 0`, i.e. `w0_m = 0`.
    #[allow(clippy::type_complexity)]
    fn sigma_points() -> (
        SigmaPropagatedMatrixBuffer<2, 5, f64, crate::matrix::MatrixDataArray<2, 5, 10, f64>>,
        SigmaWeightsVectorBuffer<5, f64, crate::matrix::MatrixDataArray<5, 1, 5, f64>>,
    ) {
        let sigma = SigmaPropagatedMatrixBuffer::from([
            1.0, 1.5, 1.0, 0.5, 1.0, // x
            2.0, 2.0, 2.5, 2.0, 1.5, // y
        ]);
        let weights = SigmaWeightsVectorBuffer::from([2.0, 0.25, 0.25, 0.25, 0.25]);
        (sigma, weights)
    }

    /// Observes `x + y` for each sigma point.
    fn observe_sum<SP, O>(sigma: &SP, observed: &mut O)
    where
        SP: AsMatrix<2, 5, f64>,
        O: AsMatrixMut<1, 5, f64>,
    {
        let sigma = sigma.as_matrix();
        let observed = observed.as_matrix_mut();
        for k in 0..5 {
            observed.set_at(0, k, sigma.get_at(0, k) + sigma.get_at(1, k));
        }
    }

    #[cfg(feature = "alloc")]
    #[allow(non_snake_case)]
    fn correct_once(
        correct: impl FnOnce(
            &mut crate::kalman_builder::unscented::KalmanFilterObservationType<2, 1, 5, f64>,
            &mut crate::buffers::types::StateVectorBuffer<
                2,
                f64,
                crate::matrix::MatrixDataArray<2, 1, 2, f64>,
            >,
            &mut crate::buffers::types::EstimateCovarianceMatrixBuffer<
                2,
                f64,
                crate::matrix::MatrixDataArray<2, 2, 4, f64>,
            >,
        ),
    ) -> ([f64; 2], [f64; 4]) {
        use crate::buffers::types::{EstimateCovarianceMatrixBuffer, StateVectorBuffer};
        use crate::kalman_builder::unscented::KalmanFilterBuilder;

        let builder = KalmanFilterBuilder::<2, f64>::default();
        let mut measurement = builder.observations().build::<1, 5>();
        measurement.measurement_vector_mut().set_at(0, 0, 3.2);
        measurement
            .measurement_noise_covariance_mut()
            .set_at(0, 0, 0.1);

        let mut x = StateVectorBuffer::from([1.0, 2.0]);
        let mut P = EstimateCovarianceMatrixBuffer::from([0.125, 0.0, 0.0, 0.125]);
        correct(&mut measurement, &mut x, &mut P);

        let x = x.as_matrix();
        let P = P.as_matrix();
        (
            [x.get_at(0, 0), x.get_at(1, 0)],
            [
                P.get_at(0, 0),
                P.get_at(0, 1),
                P.get_at(1, 0),
                P.get_at(1, 1),
            ],
        )
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[allow(non_snake_case)]
    fn all_weighted_entry_points_agree_without_fix() {
        let (sigma, weights) = sigma_points();

        let residuals = correct_once(|m, x, P| {
            m.correct_with_observed_residuals(
                x,
                P,
                &sigma,
                &weights,
                0.0,
                observe_sum,
                |_| {},
                |_| {},
            )
        });
        let observed = correct_once(|m, x, P| {
            m.correct_with_observed(x, P, &sigma, &weights, 0.0, observe_sum)
        });
        let weighted = correct_once(|m, x, P| {
            observe_sum(&sigma, &mut m.sigma_observed);
            m.correct_with_weights(x, P, &sigma, &weights, 0.0)
        });

        assert_eq!(residuals, observed);
        assert_eq!(residuals, weighted);

        // The update pulls x + y from 3.0 towards the measurement 3.2 and shrinks P.
        let ([x, y], [pxx, _, _, pyy]) = residuals;
        assert!(x + y > 3.0 && x + y < 3.2, "x + y = {}", x + y);
        assert!(pxx < 0.125 && pyy < 0.125);
    }

    #[cfg(feature = "alloc")]
    #[test]
    #[allow(non_snake_case)]
    fn unweighted_correction_moves_towards_measurement() {
        let (sigma, _) = sigma_points();
        let ([x, y], [pxx, _, _, pyy]) = correct_once(|m, x, P| {
            observe_sum(&sigma, &mut m.sigma_observed);
            m.correct_nonlinear(x, P, &sigma)
        });
        assert!(x + y > 3.0 && x + y < 3.2, "x + y = {}", x + y);
        assert!(pxx < 0.125 && pyy < 0.125);
    }
}
