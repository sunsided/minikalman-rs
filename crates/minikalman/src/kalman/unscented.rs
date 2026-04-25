//! # Unscented Kalman Filter
//!
//! The Unscented Kalman Filter (UKF) uses sigma points to propagate mean and covariance
//! estimates through nonlinear transformations, avoiding the need for Jacobian calculations.

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};
use core::marker::PhantomData;

/// Kalman Filter structure for Unscented Kalman Filter.
#[allow(non_snake_case)]
pub struct UnscentedKalman<
    const STATES: usize,
    const NUM_SIGMA: usize,
    T,
    X,
    P,
    Q,
    PX,
    SigmaPoints,
    SigmaWeights,
    SigmaPredicted,
    TempSigmaP,
> {
    x: X,
    P: P,
    Q: Q,
    predicted_x: PX,

    sigma_points: SigmaPoints,
    sigma_weights: SigmaWeights,
    sigma_predicted: SigmaPredicted,
    temp_sigma_P: TempSigmaP,

    alpha: T,
    beta: T,
    kappa: T,

    _phantom: PhantomData<T>,
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
{
    pub const fn states(&self) -> usize {
        STATES
    }

    pub const fn num_sigma_points(&self) -> usize {
        NUM_SIGMA
    }

    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn new(
        x: X,
        P: P,
        Q: Q,
        predicted_x: PX,
        sigma_points: SigmaPoints,
        sigma_weights: SigmaWeights,
        sigma_predicted: SigmaPredicted,
        temp_sigma_P: TempSigmaP,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Self {
        Self {
            x,
            P,
            Q,
            predicted_x,
            sigma_points,
            sigma_weights,
            sigma_predicted,
            temp_sigma_P,
            alpha,
            beta,
            kappa,
            _phantom: PhantomData,
        }
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    X: StateVector<STATES, T>,
{
    #[inline(always)]
    pub fn state_vector(&self) -> &X {
        &self.x
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    X: StateVectorMut<STATES, T>,
{
    #[inline(always)]
    #[doc(alias = "kalman_get_state_vector")]
    pub fn state_vector_mut(&mut self) -> &mut X {
        &mut self.x
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    #[inline(always)]
    #[doc(alias = "system_covariance")]
    pub fn estimate_covariance(&self) -> &P {
        &self.P
    }

    #[inline(always)]
    #[doc(alias = "system_covariance_mut")]
    #[doc(alias = "kalman_get_system_covariance")]
    pub fn estimate_covariance_mut(&mut self) -> &mut P {
        &mut self.P
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
{
    #[inline(always)]
    pub fn direct_process_noise(&self) -> &Q {
        &self.Q
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    Q: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
{
    #[inline(always)]
    pub fn direct_process_noise_mut(&mut self) -> &mut Q {
        &mut self.Q
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    T: MatrixDataType
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Div<Output = T>
        + From<f64>
        + PartialOrd,
{
    #[inline(always)]
    pub fn alpha(&self) -> T {
        self.alpha
    }

    #[inline(always)]
    pub fn beta(&self) -> T {
        self.beta
    }

    #[inline(always)]
    pub fn kappa(&self) -> T {
        self.kappa
    }

    #[inline(always)]
    pub fn lambda(&self) -> T {
        let n = T::from(STATES as f64);
        self.alpha * self.alpha * (n + self.kappa) - n
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
    UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    T: MatrixDataType
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + core::ops::Div<Output = T>
        + From<f64>
        + PartialOrd
        + Default,
    X: StateVectorMut<STATES, T>,
    P: EstimateCovarianceMatrix<STATES, T>,
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    SigmaPoints: SigmaPointMatrix<STATES, NUM_SIGMA, T>,
    SigmaWeights: SigmaWeightsVectorMut<NUM_SIGMA, T>,
    SigmaPredicted: SigmaPredictedMatrix<STATES, NUM_SIGMA, T>,
    TempSigmaP: TempSigmaPMatrix<STATES, T>,
{
    pub fn set_alpha(&mut self, alpha: T) {
        self.alpha = alpha;
    }

    pub fn set_beta(&mut self, beta: T) {
        self.beta = beta;
    }

    pub fn set_kappa(&mut self, kappa: T) {
        self.kappa = kappa;
    }

    #[allow(non_snake_case)]
    fn compute_weights(&mut self) {
        let n = T::from(STATES as f64);
        let lambda = self.lambda();
        let n_lambda = n + lambda;

        let w = self.sigma_weights.as_matrix_mut();

        let w0_m = lambda / n_lambda;
        let w0_c = lambda / n_lambda + (T::from(1.0) - self.alpha * self.alpha + self.beta);
        let wi = T::from(1.0) / (T::from(2.0) * n_lambda);

        w.set(0, 0, w0_m);
        w.set(0, 0, w0_c);
        for i in 1..NUM_SIGMA {
            w.set(i, 0, wi);
        }
    }

    #[allow(non_snake_case)]
    fn generate_sigma_points(&mut self) {
        let n = T::from(STATES as f64);
        let lambda = self.lambda();
        let scale = (n + lambda).square_root();

        let P = self.P.as_matrix_mut();
        let sigma = self.sigma_points.as_matrix_mut();
        let temp = self.temp_sigma_P.as_matrix_mut();

        P.copy_to(temp);
        temp.cholesky_decompose_lower();

        let x = self.x.as_matrix();
        for i in 0..STATES {
            sigma.set(i, 0, x.get(i, 0));
        }

        for j in 0..STATES {
            for i in 0..STATES {
                let val = temp.get(i, j);
                sigma.set(i, j + 1, x.get(i, 0) + scale * val);
                sigma.set(i, j + 1 + STATES, x.get(i, 0) - scale * val);
            }
        }

        for i in 0..STATES {
            for j in 0..STATES {
                P.set(i, j, temp.get(i, j));
            }
        }
    }

    #[allow(non_snake_case)]
    fn predict_sigma_points<F>(&mut self, mut state_transition: F)
    where
        F: FnMut(&X, &mut PX),
    {
        let sigma = self.sigma_points.as_matrix();
        let sigma_pred = self.sigma_predicted.as_matrix_mut();
        let _x = self.x.as_matrix();

        for j in 0..(2 * STATES + 1) {
            for i in 0..STATES {
                self.predicted_x.as_matrix_mut().set(i, 0, sigma.get(i, j));
            }

            state_transition(&self.x, &mut self.predicted_x);

            for i in 0..STATES {
                sigma_pred.set(i, j, self.predicted_x.as_matrix().get(i, 0));
            }
        }
    }

    #[allow(non_snake_case)]
    fn reconstruct_prediction(&mut self) {
        let w = self.sigma_weights.as_matrix();

        let sigma_pred = self.sigma_predicted.as_matrix();
        let x = self.x.as_matrix_mut();
        let P = self.P.as_matrix_mut();

        for i in 0..STATES {
            x.set(i, 0, T::default());
            for j in 0..NUM_SIGMA {
                let w_val = w.get(j, 0);
                let sp = sigma_pred.get(i, j);
                x.set(i, 0, x.get(i, 0) + w_val * sp);
            }
        }

        for i in 0..STATES {
            for j in 0..STATES {
                P.set(i, j, T::default());
            }
        }

        for k in 0..NUM_SIGMA {
            let w_val = w.get(k, 0);
            for i in 0..STATES {
                for j in 0..STATES {
                    let diff_i = sigma_pred.get(i, k) - x.get(i, 0);
                    let diff_j = sigma_pred.get(j, k) - x.get(j, 0);
                    let val = P.get(i, j) + w_val * diff_i * diff_j;
                    P.set(i, j, val);
                }
            }
        }

        let Q = self.Q.as_matrix();
        Q.add_inplace_b(P);
    }

    pub fn predict_nonlinear<F>(&mut self, state_transition: F)
    where
        F: FnMut(&X, &mut PX),
    {
        self.compute_weights();
        self.generate_sigma_points();
        self.predict_sigma_points(state_transition);
        self.reconstruct_prediction();
    }

    pub fn correct_nonlinear<M, F, const OBSERVATIONS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
    ) where
        M: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBSERVATIONS, NUM_SIGMA, T>,
        F: FnMut(&X, &mut M::ObservedSigmaPoints),
    {
        measurement.correct_nonlinear(&mut self.x, &mut self.P, observation);
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterNumStates<STATES>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
{
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterStateVector<STATES, T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    X: StateVector<STATES, T>,
{
    type StateVector = X;

    #[inline(always)]
    fn state_vector(&self) -> &Self::StateVector {
        self.state_vector()
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterStateVectorMut<STATES, T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    X: StateVectorMut<STATES, T>,
{
    type StateVectorMut = X;

    #[inline(always)]
    fn state_vector_mut(&mut self) -> &mut Self::StateVectorMut {
        self.state_vector_mut()
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterEstimateCovariance<STATES, T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    type EstimateCovarianceMatrix = P;

    #[inline(always)]
    fn estimate_covariance(&self) -> &Self::EstimateCovarianceMatrix {
        self.estimate_covariance()
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterEstimateCovarianceMut<STATES, T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    P: EstimateCovarianceMatrix<STATES, T>,
{
    type EstimateCovarianceMatrixMut = P;

    #[inline(always)]
    fn estimate_covariance_mut(&mut self) -> &mut Self::EstimateCovarianceMatrixMut {
        self.estimate_covariance_mut()
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterDirectProcessNoiseCovariance<STATES, T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
{
    type ProcessNoiseCovarianceMatrix = Q;

    #[inline(always)]
    fn direct_process_noise(&self) -> &Self::ProcessNoiseCovarianceMatrix {
        self.direct_process_noise()
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterDirectProcessNoiseMut<STATES, T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    Q: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
{
    type ProcessNoiseCovarianceMatrixMut = Q;

    #[inline(always)]
    fn direct_process_noise_mut(&mut self) -> &mut Self::ProcessNoiseCovarianceMatrixMut {
        self.direct_process_noise_mut()
    }
}

impl<
        const STATES: usize,
        const NUM_SIGMA: usize,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    > KalmanFilterUnscentedParams<T>
    for UnscentedKalman<
        STATES,
        NUM_SIGMA,
        T,
        X,
        P,
        Q,
        PX,
        SigmaPoints,
        SigmaWeights,
        SigmaPredicted,
        TempSigmaP,
    >
where
    T: MatrixDataType
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + From<f64>,
{
    fn alpha(&self) -> T {
        self.alpha
    }

    fn beta(&self) -> T {
        self.beta
    }

    fn kappa(&self) -> T {
        self.kappa
    }

    fn lambda(&self, n: usize) -> T {
        let nn = T::from(n as f64);
        self.alpha * self.alpha * (nn + self.kappa) - nn
    }
}

#[cfg(test)]
mod tests {
    use crate::test_dummies::make_dummy_filter_ukf;

    #[test]
    fn builder_simple() {
        let filter = make_dummy_filter_ukf();
        assert_eq!(filter.states(), 3);
        assert_eq!(filter.num_sigma_points(), 7);
    }
}
