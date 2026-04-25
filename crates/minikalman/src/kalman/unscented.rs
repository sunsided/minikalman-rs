//! # Unscented Kalman Filter
//!
//! The Unscented Kalman Filter (UKF) uses sigma points to propagate mean and covariance
//! estimates through nonlinear transformations, avoiding the need for Jacobian calculations.

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};
use crate::prelude::AsMatrixMut;
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

    /// Returns a reference to the predicted sigma points buffer.
    pub fn sigma_predicted(&self) -> &SigmaPredicted {
        &self.sigma_predicted
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
        + num_traits::FromPrimitive
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
        let n = T::from_usize(STATES).unwrap_or(T::zero());
        self.alpha * self.alpha * (n + self.kappa) - n
    }
    pub fn set_alpha(&mut self, v: T) {
        self.alpha = v;
    }
    pub fn set_beta(&mut self, v: T) {
        self.beta = v;
    }
    pub fn set_kappa(&mut self, v: T) {
        self.kappa = v;
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
        + num_traits::FromPrimitive
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
    #[allow(non_snake_case)]
    fn compute_weights(&mut self) {
        let n = T::from_usize(STATES).unwrap_or(T::zero());
        let lambda = self.lambda();
        let n_lambda = n + lambda;
        let w = self.sigma_weights.as_matrix_mut();
        let w0_c =
            lambda / n_lambda + (T::from_usize(1).unwrap() - self.alpha * self.alpha + self.beta);
        let wi = T::from_usize(1).unwrap() / (T::from_usize(2).unwrap() * n_lambda);
        w.set(0, 0, w0_c);
        for i in 1..NUM_SIGMA {
            w.set(i, 0, wi);
        }
    }

    #[allow(non_snake_case)]
    fn generate_sigma_points(&mut self) {
        let n = T::from_usize(STATES).unwrap_or(T::zero());
        let lambda = self.lambda();
        let scale = (n + lambda).square_root();
        let P = self.P.as_matrix_mut();
        let sigma = self.sigma_points.as_matrix_mut();
        let temp = self.temp_sigma_P.as_matrix_mut();

        // P.copy_to(temp) saves original P, then Cholesky overwrites temp with L
        // But we need both L and original P. Use a copy approach:
        // First copy P to temp, then Cholesky temp in-place
        // After sigma generation, we don't need to restore P since it wasn't modified

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
        // Note: P was NOT modified by copy_to, so no restoration needed
    }

    #[allow(non_snake_case)]
    fn predict_sigma_points<F>(&mut self, mut state_transition: F)
    where
        F: FnMut(&mut PX),
    {
        let sigma = self.sigma_points.as_matrix();
        let sigma_pred = self.sigma_predicted.as_matrix_mut();
        for j in 0..NUM_SIGMA {
            // Load sigma point into predicted_x
            for i in 0..STATES {
                self.predicted_x.as_matrix_mut().set(i, 0, sigma.get(i, j));
            }
            // Propagate sigma point through state transition (in-place)
            state_transition(&mut self.predicted_x);
            // Store result
            for i in 0..STATES {
                sigma_pred.set(i, j, self.predicted_x.as_matrix().get(i, 0));
            }
        }
    }

    #[allow(non_snake_case)]
    fn reconstruct_prediction(&mut self) {
        let lambda = self.lambda();
        let n = T::from_usize(STATES).unwrap();
        let w0_m = lambda / (n + lambda);
        let w = self.sigma_weights.as_matrix();
        let sigma_pred = self.sigma_predicted.as_matrix();
        let x = self.x.as_matrix_mut();
        let P = self.P.as_matrix_mut();
        for i in 0..STATES {
            x.set(i, 0, T::default());
            for j in 0..NUM_SIGMA {
                let w_val = if j == 0 { w0_m } else { w.get(j, 0) };
                x.set(i, 0, x.get(i, 0) + w_val * sigma_pred.get(i, j));
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
                    P.set(i, j, P.get(i, j) + w_val * diff_i * diff_j);
                }
            }
        }
        let Q = self.Q.as_matrix();
        Q.add_inplace_b(P);
    }

    /// Nonlinear prediction using sigma points.
    pub fn predict_nonlinear<F>(&mut self, state_transition: F)
    where
        F: FnMut(&mut PX),
    {
        self.compute_weights();
        self.generate_sigma_points();
        self.predict_sigma_points(state_transition);
        self.reconstruct_prediction();
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
        + num_traits::FromPrimitive,
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
        let nn = T::from_usize(n).unwrap_or(T::zero());
        self.alpha * self.alpha * (nn + self.kappa) - nn
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
    > KalmanFilterSigmaPointPredict<STATES, T>
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
        + core::ops::Div<Output = T>
        + num_traits::FromPrimitive
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
    type NextStateVector = PX;

    fn predict_sigma_point<F>(&mut self, state_transition: F)
    where
        F: FnMut(&mut PX),
    {
        self.predict_nonlinear(state_transition)
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
    > KalmanFilterSigmaPointCorrect<STATES, NUM_SIGMA, T>
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
        + core::ops::Div<Output = T>
        + num_traits::FromPrimitive
        + PartialOrd
        + Default,
    X: StateVectorMut<STATES, T>,
    P: EstimateCovarianceMatrix<STATES, T>,
    Q: DirectProcessNoiseCovarianceMatrix<STATES, T>,
    PX: PredictedStateEstimateVector<STATES, T>,
    SigmaPoints: SigmaPointMatrix<STATES, NUM_SIGMA, T>,
    SigmaWeights: SigmaWeightsVectorMut<NUM_SIGMA, T>,
    SigmaPredicted: SigmaPredictedMatrix<STATES, NUM_SIGMA, T> + AsMatrixMut<STATES, NUM_SIGMA, T>,
    TempSigmaP: TempSigmaPMatrix<STATES, T>,
{
    type SigmaPredicted = SigmaPredicted;
    fn correct_sigma_point<M, F, const OBS: usize>(&mut self, measurement: &mut M, observation: F)
    where
        M: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBS, NUM_SIGMA, T>,
        F: FnMut(&SigmaPredicted, &mut M::ObservedSigmaPoints),
    {
        let lambda = self.lambda();
        measurement.correct_with_weights(
            &mut self.x,
            &mut self.P,
            &self.sigma_predicted,
            &self.sigma_weights,
            lambda,
            observation,
        )
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
