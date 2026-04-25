//! # Observations for Unscented Kalman Filters.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{AsMatrix, AsMatrixMut, Matrix, MatrixDataType, MatrixMut, SquareMatrix};

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
        self.apply_correction(x, P, &z_pred);
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
        let n = T::from_usize(STATES).unwrap();
        let w0_m = lambda / (n + lambda);
        let z_pred = self.compute_predicted_measurement_weighted(weights, w0_m);
        self.compute_innovation_covariance_weighted(&z_pred, weights);
        self.compute_cross_covariance_weighted(&z_pred, sigma_propagated, weights, w0_m);
        self.compute_kalman_gain();
        self.apply_correction(x, P, &z_pred);
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
    fn compute_predicted_measurement_weighted<W: AsMatrix<NUM_SIGMA, 1, T>>(
        &mut self,
        weights: &W,
        w0_m: T,
    ) -> [T; OBSERVATIONS] {
        let sigma_obs = self.sigma_observed.as_matrix();
        let w = weights.as_matrix();
        let mut z_pred = [T::default(); OBSERVATIONS];
        for i in 0..OBSERVATIONS {
            let mut sum = T::default();
            for j in 0..NUM_SIGMA {
                let w_val = if j == 0 { w0_m } else { w.get(j, 0) };
                sum += w_val * sigma_obs.get(i, j);
            }
            z_pred[i] = sum;
        }
        z_pred
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_innovation_covariance_weighted<W: AsMatrix<NUM_SIGMA, 1, T>>(
        &mut self,
        z_pred: &[T; OBSERVATIONS],
        weights: &W,
    ) {
        let sigma_obs = self.sigma_observed.as_matrix();
        let S = self.S.as_matrix_mut();
        let R = self.R.as_matrix();
        let w = weights.as_matrix();

        for i in 0..OBSERVATIONS {
            for j in 0..OBSERVATIONS {
                let mut sum = T::default();
                for k in 0..NUM_SIGMA {
                    let diff_i = sigma_obs.get(i, k) - z_pred[i];
                    let diff_j = sigma_obs.get(j, k) - z_pred[j];
                    sum += w.get(k, 0) * diff_i * diff_j;
                }
                S.set(i, j, sum);
            }
        }

        R.add_inplace_b(S);
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn compute_cross_covariance_weighted<SP, W>(
        &mut self,
        z_pred: &[T; OBSERVATIONS],
        sigma_propagated: &SP,
        weights: &W,
        w0_m: T,
    ) where
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
    {
        let cross_cov = self.cross_covariance.as_matrix_mut();
        let sigma_obs = self.sigma_observed.as_matrix();
        let sigma_prop = sigma_propagated.as_matrix();
        let w = weights.as_matrix();

        let mut x_mean = [T::default(); STATES];
        for i in 0..STATES {
            let mut sum = T::default();
            for k in 0..NUM_SIGMA {
                let w_val = if k == 0 { w0_m } else { w.get(k, 0) };
                sum += w_val * sigma_prop.get(i, k);
            }
            x_mean[i] = sum;
        }

        for i in 0..STATES {
            for j in 0..OBSERVATIONS {
                let mut sum = T::default();
                for k in 0..NUM_SIGMA {
                    let diff_x = sigma_prop.get(i, k) - x_mean[i];
                    let diff_z = sigma_obs.get(j, k) - z_pred[j];
                    sum += w.get(k, 0) * diff_x * diff_z;
                }
                cross_cov.set(i, j, sum);
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
    fn apply_correction<XVec, PCov>(
        &mut self,
        x: &mut XVec,
        P: &mut PCov,
        z_pred: &[T; OBSERVATIONS],
    ) where
        XVec: StateVectorMut<STATES, T>,
        PCov: EstimateCovarianceMatrix<STATES, T>,
    {
        let z = self.z.as_matrix();
        let y = self.y.as_matrix_mut();
        let K = self.K.as_matrix();

        for i in 0..OBSERVATIONS {
            y.set(i, 0, z.get(i, 0) - z_pred[i]);
        }

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

    fn correct_with_observed<X, P, SP, F, W>(
        &mut self,
        x: &mut X,
        P: &mut P,
        sigma_propagated: &SP,
        weights: &W,
        lambda: T,
        mut observation: F,
    ) where
        X: StateVectorMut<STATES, T>,
        P: EstimateCovarianceMatrix<STATES, T>,
        SP: AsMatrix<STATES, NUM_SIGMA, T>,
        W: AsMatrix<NUM_SIGMA, 1, T>,
        F: FnMut(&SP, &mut Self::ObservedSigmaPoints),
    {
        observation(sigma_propagated, &mut self.sigma_observed);
        self.correct_with_weights(x, P, sigma_propagated, weights, lambda);
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_dummy() {
        assert_eq!(2 * 3 + 1, 7);
    }
}
