//! # Unscented Kalman Filter
//!
//! The Unscented Kalman Filter (UKF) uses sigma points to propagate mean and covariance
//! estimates through nonlinear transformations, avoiding the need for Jacobian calculations.

use crate::kalman::process_noise::{correct_adapting_process_noise, ProcessNoiseAdapter};
use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut};
use crate::observations::unscented::anchored_mean;
use crate::observations::NoiseAdaptation;
use crate::prelude::{AsMatrix, AsMatrixMut};
use core::marker::PhantomData;

/// Kalman Filter structure for Unscented Kalman Filter.
///
#[doc = diagram!("unscented-loop.mmd")]
///
/// # Choosing α, β and κ
///
/// The parameters set where the sigma points go and how they are weighted. With
/// `n = STATES`, the filter computes
///
/// - `λ = α²·(n + κ) − n`,
/// - the sigma points `x` and `x ± √(n + λ)·[√P]ᵢ` for each column `i` of the Cholesky
///   factor of `P`,
/// - the mean weights `W₀ᵐ = λ / (n + λ)` and `Wᵢ = 1 / (2·(n + λ))`,
/// - the covariance weights `W₀ᶜ = λ / (n + λ) + (1 − α² + β)` and the same `Wᵢ`.
///
/// `n + λ = α²·(n + κ)` must be positive. The weights are recomputed on every prediction,
/// so values changed with [`set_alpha`](Self::set_alpha), [`set_beta`](Self::set_beta) and
/// [`set_kappa`](Self::set_kappa) take effect on the next [`predict_nonlinear`](Self::predict_nonlinear).
///
/// | Setting | α | β | κ | λ | `√(n + λ)` | `W₀ᵐ` | `W₀ᶜ` | `Wᵢ` |
/// |---|---|---|---|---|---|---|---|---|
/// | [`KalmanFilterBuilder`](crate::unscented::builder::KalmanFilterBuilder) default | 1 | 2 | 1 | 1 | `√(n + 1)` | `1 / (n + 1)` | `1 / (n + 1) + 2` | `1 / (2·(n + 1))` |
/// | Cubature Kalman filter (CKF) | 1 | 0 | 0 | 0 | `√n` | 0 | 0 | `1 / (2n)` |
/// | Classic scaled UKF | 10⁻³ | 2 | 0 | `≈ −n` | `10⁻³·√n` | `1 − 10⁶` | `≈ 4 − 10⁶` | `10⁶ / (2n)` |
///
/// - **Default (α = 1, β = 2, κ = 1)**: the points sit at `√(n + 1)` standard deviations,
///   all weights are positive, and `β = 2` adds the extra centre weight that is optimal for
///   a Gaussian prior.
/// - **Cubature Kalman filter (α = 1, β = 0, κ = 0)**: the third-degree spherical-radial
///   cubature rule. The `2n` outer points sit at `±√n` standard deviations with equal
///   weights `1 / (2n)`; the centre point gets zero weight in both the mean and the
///   covariance. No weight is negative, so the predicted covariance stays positive
///   semi-definite. Kean, Shen and Oates ([arXiv:2610.14773](https://arxiv.org/abs/2610.14773))
///   evaluate a predictively oriented (PrO) UKF variant on the Lorenz96 model. With these
///   settings it "offers consistent improvements, which become substantial under strong
///   nonlinearity", while "the default unscented parameters are unreliable".
/// - **Classic scaled UKF (α = 10⁻³, β = 2, κ = 0)**: the points sit very close to the mean,
///   and the centre weights are about `−10⁶`. The mean and covariance then come from the
///   difference of large numbers, which loses most of an `f32`'s precision. These are the
///   "default unscented parameters" of the paper above. Avoid them, in particular with
///   `f32` or fixed-point types.
///
/// Which setting tracks best depends on the model, so compare them on recorded or simulated
/// data before you change them.
///
/// ## Example
///
/// Configure a cubature Kalman filter:
///
/// ```
/// # #[cfg(feature = "alloc")] {
/// use minikalman::prelude::*;
/// use minikalman::unscented::builder::KalmanFilterBuilder;
///
/// const NUM_STATES: usize = 2;
/// const NUM_SIGMA: usize = 2 * NUM_STATES + 1;
///
/// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
/// let mut filter = builder.build::<NUM_SIGMA>();
///
/// filter.set_alpha(1.0);
/// filter.set_beta(0.0);
/// filter.set_kappa(0.0);
/// assert_eq!(filter.lambda(), 0.0);
///
/// // With equal positive weights, a linear transition reproduces the covariance exactly.
/// filter.estimate_covariance_mut().make_identity();
/// filter.predict_nonlinear(|_x| {});
///
/// let p = filter.estimate_covariance().as_matrix();
/// # // With the approximate `micromath` square root, the variances come out as 1.125.
/// # if cfg!(any(feature = "std", feature = "libm")) {
/// assert!((p.get_at(0, 0) - 1.0).abs() < 1e-6);
/// assert!((p.get_at(1, 1) - 1.0).abs() < 1e-6);
/// assert!(p.get_at(0, 1).abs() < 1e-6);
/// # }
/// # }
/// ```
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
    SigmaPropagated,
    TempSigmaP,
> {
    x: X,
    P: P,
    Q: Q,
    predicted_x: PX,
    sigma_points: SigmaPoints,
    sigma_weights: SigmaWeights,
    sigma_propagated: SigmaPropagated,
    temp_sigma_P: TempSigmaP,
    alpha: T,
    beta: T,
    kappa: T,
    state_residual: Option<fn(&mut [T])>,
    /// Whether `sigma_propagated` no longer describes the current `x` and `P`, i.e. a
    /// correction ran since the last prediction (or no prediction ran yet).
    sigma_points_stale: bool,
    /// Online adaptation of `Q` from the state corrections; `None` (the default) disables it.
    ///
    /// See [`set_process_noise_adaptation`](Self::set_process_noise_adaptation).
    process_noise_adaptation: Option<ProcessNoiseAdapter<STATES, T, Q>>,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        sigma_propagated: SigmaPropagated,
        temp_sigma_P: TempSigmaP,
        alpha: T,
        beta: T,
        kappa: T,
    ) -> Self {
        #[cfg(not(feature = "no_assert"))]
        assert_eq!(
            NUM_SIGMA,
            2 * STATES + 1,
            "NUM_SIGMA must be 2 × STATES + 1"
        );
        Self {
            x,
            P,
            Q,
            predicted_x,
            sigma_points,
            sigma_weights,
            sigma_propagated,
            temp_sigma_P,
            alpha,
            beta,
            kappa,
            state_residual: None,
            sigma_points_stale: true,
            process_noise_adaptation: None,
            _phantom: PhantomData,
        }
    }

    /// Returns a reference to the propagated sigma points buffer.
    pub fn sigma_propagated(&self) -> &SigmaPropagated {
        &self.sigma_propagated
    }

    /// Sets the function that normalizes differences between states, e.g. sigma point
    /// minus mean.
    ///
    /// Use this when a state wraps around, such as a heading. The function receives the
    /// difference as a slice (one entry per state) and can wrap it back into range:
    ///
    /// ```
    /// # #[cfg(feature = "alloc")] {
    /// # use minikalman::prelude::*;
    /// # use minikalman::unscented::builder::KalmanFilterBuilder;
    /// use minikalman::angles;
    ///
    /// # let builder = KalmanFilterBuilder::<3, f32>::default();
    /// # let mut filter = builder.build::<7>();
    /// // State: x, y, heading
    /// filter.set_state_residual(|dx| dx[2] = angles::wrap_pi(dx[2]));
    /// # }
    /// ```
    ///
    /// It applies to the state means and residuals in both [`predict_nonlinear`](Self::predict_nonlinear)
    /// and [`correct_sigma_point_with`](KalmanFilterSigmaPointCorrect::correct_sigma_point_with).
    /// Means are taken relative to the centre sigma point, so sigma points on either side
    /// of the wrap point average correctly. The resulting state itself is not wrapped; do
    /// that on [`state_vector_mut`](Self::state_vector_mut) if you need it in range.
    pub fn set_state_residual(&mut self, fix: fn(&mut [T])) {
        self.state_residual = Some(fix);
    }

    /// Calls the state residual function, if one is set.
    fn fix_state(residual: Option<fn(&mut [T])>) -> impl FnMut(&mut [T]) {
        move |dx: &mut [T]| {
            if let Some(fix) = residual {
                fix(dx)
            }
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
        TempSigmaP,
    >
where
    Q: DirectProcessNoiseCovarianceMatrixMut<STATES, T>,
{
    #[inline(always)]
    pub fn direct_process_noise_mut(&mut self) -> &mut Q {
        &mut self.Q
    }

    /// Returns the online adaptation of the process noise `Q`, if enabled with
    /// [`set_process_noise_adaptation`](Self::set_process_noise_adaptation).
    #[inline(always)]
    pub fn process_noise_adaptation(&self) -> Option<NoiseAdaptation<T>>
    where
        T: MatrixDataType,
    {
        self.process_noise_adaptation
            .map(|adapter| adapter.settings())
    }

    /// Enables or disables the online adaptation of the process noise `Q`.
    ///
    #[doc = diagram!("unscented-correct.mmd", "adapt")]
    ///
    /// With `Some`, every applied sigma point correction, including the recalibrated one,
    /// replaces `Q` by `α×Q + (1−α)×Δx×Δxᵀ` from the state correction `Δx = x⁺ − x⁻`, then
    /// raises every diagonal entry to at least the floor; `Q` stays symmetric positive
    /// semi-definite. The estimate is unbiased only for a steady state that the prediction
    /// without `Q` leaves unchanged, e.g. a random walk; otherwise it is biased, and with a
    /// steady-state gain it has rank one along `K`. `None` (the
    /// default) leaves `Q` untouched.
    ///
    /// See [`NoiseAdaptation`] for the forgetting factor, the method and its limits.
    #[inline(always)]
    pub fn set_process_noise_adaptation(&mut self, adaptation: Option<NoiseAdaptation<T>>)
    where
        T: MatrixDataType,
    {
        self.process_noise_adaptation = adaptation.map(ProcessNoiseAdapter::new);
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
    SigmaPropagated: SigmaPropagatedMatrix<STATES, NUM_SIGMA, T>,
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

        // Decompose a copy, so that P itself stays untouched.
        P.copy_to(temp);
        if !temp.cholesky_decompose_lower() {
            // shortcut: if P is not positive definite (e.g. P = 0 or a zero variance), spread
            // the sigma points along the axes by √Pᵢᵢ and drop the correlations; a failed
            // decomposition would leave partial rows and the untouched upper triangle behind.
            for i in 0..STATES {
                for j in 0..STATES {
                    let p = P.get(i, i);
                    let val = if i == j && p > T::zero() {
                        p.square_root()
                    } else {
                        T::zero()
                    };
                    temp.set(i, j, val);
                }
            }
        }

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
    }

    #[allow(non_snake_case)]
    fn predict_sigma_points<F>(&mut self, mut state_transition: F)
    where
        F: FnMut(&mut PX),
    {
        let sigma = self.sigma_points.as_matrix();
        let sigma_prop = self.sigma_propagated.as_matrix_mut();
        for j in 0..NUM_SIGMA {
            // Load sigma point into predicted_x
            for i in 0..STATES {
                self.predicted_x.as_matrix_mut().set(i, 0, sigma.get(i, j));
            }
            // Propagate sigma point through state transition (in-place)
            state_transition(&mut self.predicted_x);
            // Store result
            for i in 0..STATES {
                sigma_prop.set(i, j, self.predicted_x.as_matrix().get(i, 0));
            }
        }
    }

    #[allow(non_snake_case, clippy::needless_range_loop)]
    fn reconstruct_prediction(&mut self) {
        let lambda = self.lambda();
        let n = T::from_usize(STATES).unwrap();
        let w0_m = lambda / (n + lambda);
        let mut fix_state = Self::fix_state(self.state_residual);

        let w = self.sigma_weights.as_matrix();
        let sigma_prop = self.sigma_propagated.as_matrix();
        let x = self.x.as_matrix_mut();
        let P = self.P.as_matrix_mut();

        let mean = anchored_mean(sigma_prop, &self.sigma_weights, w0_m, &mut fix_state);
        for i in 0..STATES {
            x.set(i, 0, mean[i]);
        }

        P.set_zero();
        let mut dx = [T::default(); STATES];
        for k in 0..NUM_SIGMA {
            for i in 0..STATES {
                dx[i] = sigma_prop.get(i, k) - mean[i];
            }
            fix_state(&mut dx);

            let w_val = w.get(k, 0);
            for i in 0..STATES {
                for j in 0..STATES {
                    P.set(i, j, P.get(i, j) + w_val * dx[i] * dx[j]);
                }
            }
        }
        let Q = self.Q.as_matrix();
        Q.add_inplace_b(P);
    }

    /// Nonlinear prediction using sigma points.
    ///
    #[doc = diagram!("unscented-loop.mmd", "sigma", "propagate", "predict")]
    ///
    /// `state_transition` is called once per sigma point and transforms it in place.
    pub fn predict_nonlinear<F>(&mut self, state_transition: F)
    where
        F: FnMut(&mut PX),
    {
        self.compute_weights();
        self.generate_sigma_points();
        self.predict_sigma_points(state_transition);
        self.reconstruct_prediction();
        self.sigma_points_stale = false;
    }

    /// Performs the nonlinear correction step using sigma points, with covariance
    /// recalibration.
    ///
    #[doc = diagram!("unscented-correct.mmd", "recalibrate")]
    ///
    /// The gain and the state update are the same as in
    /// [`correct_sigma_point`](KalmanFilterSigmaPointCorrect::correct_sigma_point). The
    /// covariance is then recalibrated from the sigma points shifted to the updated state,
    /// which keeps the filter from becoming overconfident when the prior is poor; if that
    /// would increase the trace of `P`, the measurement is backed out and
    /// [`CorrectionStatus::BackedOut`] is returned. See
    /// [`KalmanFilterUnscentedObservationCorrectRecalibrated`] for the algorithm and the
    /// reference (Jiang, Shi and Moura, Automatica 2026,
    /// [arXiv:2407.05717](https://arxiv.org/abs/2407.05717)).
    ///
    /// `observation` is called twice when the first pass is applied, once per set of sigma
    /// points.
    pub fn correct_sigma_point_recalibrated<M, F, const OBS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectRecalibrated<STATES, OBS, NUM_SIGMA, T>,
        F: FnMut(&SigmaPropagated, &mut M::ObservedSigmaPoints),
        SigmaPropagated: AsMatrixMut<STATES, NUM_SIGMA, T>,
    {
        self.correct_sigma_point_recalibrated_with(measurement, observation, |_| {})
    }

    /// Performs the nonlinear correction step using sigma points, with covariance
    /// recalibration, normalizing differences between measurements before they are used.
    ///
    #[doc = diagram!("unscented-correct.mmd", "recalibrate")]
    ///
    /// See [`correct_sigma_point_recalibrated`](Self::correct_sigma_point_recalibrated) and
    /// [`correct_sigma_point_with`](KalmanFilterSigmaPointCorrect::correct_sigma_point_with).
    pub fn correct_sigma_point_recalibrated_with<M, F, N, const OBS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectRecalibrated<STATES, OBS, NUM_SIGMA, T>,
        F: FnMut(&SigmaPropagated, &mut M::ObservedSigmaPoints),
        N: FnMut(&mut [T]),
        SigmaPropagated: AsMatrixMut<STATES, NUM_SIGMA, T>,
    {
        self.refresh_sigma_points();
        let lambda = self.lambda();
        let fix_state = Self::fix_state(self.state_residual);
        let status = correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| {
                measurement.correct_with_observed_recalibrated_residuals(
                    x,
                    &mut self.P,
                    &mut self.sigma_propagated,
                    &self.sigma_weights,
                    lambda,
                    observation,
                    fix_state,
                    fix,
                )
            },
        );
        // Also after a back-out: the second pass leaves the sigma points shifted.
        self.sigma_points_stale = true;
        status
    }

    /// Performs an outlier-robust sigma point correction step with Student-t measurement
    /// noise.
    ///
    #[doc = diagram!("unscented-loop.mmd", "observe", "correct")]
    ///
    /// The gain is `K = Pxz×(Pzz + R/w)⁻¹`, where the weight `w` is fitted to the innovation
    /// by a few variational Bayes iterations (Piché, Särkkä, Hartikainen, IEEE MLSP 2012) on
    /// the predicted measurement covariance `Pzz`. Outliers get a small weight and move the
    /// state little; `P` never grows past the prior. See
    /// [`RegularObservation::correct_robust_with`](crate::regular::RegularObservation::correct_robust_with)
    /// for the update, its status, and how it relates to the predictively oriented correction.
    /// This is the only robust update of the unscented filter besides the
    /// [gate](crate::unscented::UnscentedObservation::set_gate).
    ///
    /// ## Difference to Piché et al.
    /// The paper recomputes the residual statistics `D` from posterior sigma points in every
    /// iteration. Here they come from the prior sigma points, which observes `h` only once:
    /// `z − ẑ⁺ = R×(w×Pzz + R)⁻¹×y` and `Pzz⁺ = Pzz − Pzz×(Pzz + R/w)⁻¹×Pzz`, which is exact
    /// for a linear `h` and the statistical linearization of the sigma points otherwise.
    ///
    /// The observation's `R` is not adapted; the filter's
    /// [process noise adaptation](Self::set_process_noise_adaptation) runs on applied
    /// corrections, including [`CorrectionStatus::NotConverged`].
    ///
    /// ## Arguments
    /// * `measurement` - The measurement.
    /// * `observation` - Writes the images of the sigma points under `h`.
    /// * `workspace` - One `OBS × OBS` scratch matrix.
    /// * `robust` - The noise model and the iteration budget, e.g. `20`.
    pub fn correct_sigma_point_robust<M, F, W, const OBS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectRobust<STATES, OBS, NUM_SIGMA, T>,
        F: FnMut(&SigmaPropagated, &mut M::ObservedSigmaPoints),
        W: TemporaryResidualCovarianceInvertedMatrix<OBS, T>,
        SigmaPropagated: AsMatrixMut<STATES, NUM_SIGMA, T>,
    {
        self.correct_sigma_point_robust_with(measurement, observation, |_| {}, workspace, robust)
    }

    /// Performs an outlier-robust sigma point correction step with Student-t measurement
    /// noise, normalizing differences between measurements before they are used.
    ///
    /// See [`correct_sigma_point_robust`](Self::correct_sigma_point_robust) and
    /// [`correct_sigma_point_with`](KalmanFilterSigmaPointCorrect::correct_sigma_point_with).
    pub fn correct_sigma_point_robust_with<M, F, N, W, const OBS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
        fix: N,
        workspace: &mut W,
        robust: crate::prelude::RobustCorrection<T>,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectRobust<STATES, OBS, NUM_SIGMA, T>,
        F: FnMut(&SigmaPropagated, &mut M::ObservedSigmaPoints),
        N: FnMut(&mut [T]),
        W: TemporaryResidualCovarianceInvertedMatrix<OBS, T>,
        SigmaPropagated: AsMatrixMut<STATES, NUM_SIGMA, T>,
    {
        self.refresh_sigma_points();
        let lambda = self.lambda();
        let fix_state = Self::fix_state(self.state_residual);
        let status = correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| {
                measurement.correct_with_observed_robust_residuals(
                    x,
                    &mut self.P,
                    &self.sigma_propagated,
                    &self.sigma_weights,
                    lambda,
                    observation,
                    fix_state,
                    fix,
                    workspace,
                    robust,
                )
            },
        );
        self.sigma_points_stale = true;
        status
    }

    /// Redraws the sigma points from the current state and covariance, without
    /// propagating them, if a correction already used the current set.
    ///
    /// A correction updates `x` and `P` but not the sigma points. A second correction
    /// before the next prediction (e.g. from another sensor) would otherwise reuse the
    /// prior's spread and shrink `P` by that much again, until it is no longer positive.
    fn refresh_sigma_points(&mut self) {
        if !self.sigma_points_stale {
            return;
        }

        self.compute_weights();
        self.generate_sigma_points();
        let sigma = self.sigma_points.as_matrix();
        let sigma_prop = self.sigma_propagated.as_matrix_mut();
        for j in 0..NUM_SIGMA {
            for i in 0..STATES {
                sigma_prop.set(i, j, sigma.get(i, j));
            }
        }
        self.sigma_points_stale = false;
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
        SigmaPropagated,
        TempSigmaP,
    > KalmanFilterUnscentedParamsMut<T>
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
        SigmaPropagated,
        TempSigmaP,
    >
where
    T: MatrixDataType
        + Copy
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>
        + core::ops::Sub<Output = T>
        + num_traits::FromPrimitive
        + PartialOrd,
{
    fn set_alpha(&mut self, v: T) {
        self.alpha = v;
    }
    fn set_beta(&mut self, v: T) {
        self.beta = v;
    }
    fn set_kappa(&mut self, v: T) {
        self.kappa = v;
    }
    fn set_state_residual(&mut self, fix: fn(&mut [T])) {
        self.state_residual = Some(fix);
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
        SigmaPropagated,
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
        SigmaPropagated,
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
    SigmaPropagated: SigmaPropagatedMatrix<STATES, NUM_SIGMA, T>,
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
        SigmaPropagated,
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
        SigmaPropagated,
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
    SigmaPropagated: SigmaPropagatedMatrix<STATES, NUM_SIGMA, T> + AsMatrix<STATES, NUM_SIGMA, T>,
    TempSigmaP: TempSigmaPMatrix<STATES, T>,
{
    type SigmaPropagated = SigmaPropagated;
    fn correct_sigma_point_with<M, F, N, const OBS: usize>(
        &mut self,
        measurement: &mut M,
        observation: F,
        fix: N,
    ) -> CorrectionStatus
    where
        M: KalmanFilterUnscentedObservationCorrectFilter<STATES, OBS, NUM_SIGMA, T>,
        F: FnMut(&SigmaPropagated, &mut M::ObservedSigmaPoints),
        N: FnMut(&mut [T]),
    {
        self.refresh_sigma_points();
        let lambda = self.lambda();
        let fix_state = Self::fix_state(self.state_residual);
        let status = correct_adapting_process_noise(
            self.process_noise_adaptation,
            &mut self.x,
            &mut self.Q,
            |x| {
                measurement.correct_with_observed_residuals(
                    x,
                    &mut self.P,
                    &self.sigma_propagated,
                    &self.sigma_weights,
                    lambda,
                    observation,
                    fix_state,
                    fix,
                )
            },
        );
        self.sigma_points_stale = true;
        status
    }
}

#[cfg(test)]
mod tests {
    use crate::kalman::*;
    use crate::test_dummies::{make_dummy_filter_ukf, make_dummy_observation_ukf, Dummy};
    use crate::unscented::UnscentedKalman;

    #[test]
    fn builder_simple() {
        let filter = make_dummy_filter_ukf();
        assert_eq!(filter.states(), 3);
        assert_eq!(filter.num_sigma_points(), 7);
    }

    #[test]
    fn state_vector_accessors() {
        let filter = make_dummy_filter_ukf();
        let _ = filter.state_vector();

        let mut filter = make_dummy_filter_ukf();
        let _ = filter.state_vector_mut();
    }

    #[test]
    fn estimate_covariance_accessors() {
        let filter = make_dummy_filter_ukf();
        let _ = filter.estimate_covariance();

        let mut filter = make_dummy_filter_ukf();
        let _ = filter.estimate_covariance_mut();
    }

    #[test]
    fn direct_process_noise_accessors() {
        let filter = make_dummy_filter_ukf();
        let _ = filter.direct_process_noise();

        let mut filter = make_dummy_filter_ukf();
        let _ = filter.direct_process_noise_mut();
    }

    #[test]
    fn sigma_propagated_accessor() {
        let filter = make_dummy_filter_ukf();
        let _ = filter.sigma_propagated();
    }

    #[test]
    fn parameter_getters_setters() {
        let mut filter = make_dummy_filter_ukf();

        assert_eq!(filter.alpha(), 1e-3);
        assert_eq!(filter.beta(), 2.0);
        assert_eq!(filter.kappa(), 0.0);

        filter.set_alpha(0.5);
        filter.set_beta(3.0);
        filter.set_kappa(1.0);

        assert_eq!(filter.alpha(), 0.5);
        assert_eq!(filter.beta(), 3.0);
        assert_eq!(filter.kappa(), 1.0);

        let n = 3usize;
        let expected_lambda = 0.5 * 0.5 * (n as f32 + 1.0) - n as f32;
        assert!((filter.lambda() - expected_lambda).abs() < 1e-6);
    }

    #[test]
    fn trait_num_states() {
        fn check<T: KalmanFilterNumStates<3>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);
    }

    #[test]
    fn trait_state_vector() {
        fn check<T: KalmanFilterStateVector<3, f32>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);
    }

    #[test]
    fn trait_state_vector_mut() {
        fn check<T: KalmanFilterStateVectorMut<3, f32>>(_: &mut T) {}
        let mut filter = make_dummy_filter_ukf();
        check(&mut filter);
    }

    #[test]
    fn trait_estimate_covariance() {
        fn check<T: KalmanFilterEstimateCovariance<3, f32>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);
    }

    #[test]
    fn trait_estimate_covariance_mut() {
        fn check<T: KalmanFilterEstimateCovarianceMut<3, f32>>(_: &mut T) {}
        let mut filter = make_dummy_filter_ukf();
        check(&mut filter);
    }

    #[test]
    fn trait_direct_process_noise() {
        fn check<T: KalmanFilterDirectProcessNoiseCovariance<3, f32>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);
    }

    #[test]
    fn trait_direct_process_noise_mut() {
        fn check<T: KalmanFilterDirectProcessNoiseMut<3, f32>>(_: &mut T) {}
        let mut filter = make_dummy_filter_ukf();
        check(&mut filter);
    }

    #[test]
    fn trait_unscented_params() {
        fn check<T: KalmanFilterUnscentedParams<f32>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);

        let lambda = KalmanFilterUnscentedParams::lambda(&filter, 3);
        let expected = 1e-3 * 1e-3 * (3.0 + 0.0) - 3.0;
        assert!((lambda - expected).abs() < 1e-6);
    }

    #[test]
    fn trait_unscented_params_mut() {
        fn check<T: KalmanFilterUnscentedParamsMut<f32>>(_: &mut T) {}
        let mut filter = make_dummy_filter_ukf();
        check(&mut filter);
    }

    #[allow(clippy::type_complexity)]
    #[test]
    fn new_constructor() {
        let filter: UnscentedKalman<
            3,
            7,
            f32,
            Dummy<f32, 3, 1>,
            Dummy<f32, 3, 3>,
            Dummy<f32, 3, 3>,
            Dummy<f32, 3, 1>,
            Dummy<f32, 3, 7>,
            Dummy<f32, 7, 1>,
            Dummy<f32, 3, 7>,
            Dummy<f32, 3, 3>,
        > = UnscentedKalman::new(
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            1.0,
            2.0,
            0.5,
        );
        assert_eq!(filter.alpha(), 1.0);
        assert_eq!(filter.beta(), 2.0);
        assert_eq!(filter.kappa(), 0.5);
    }

    #[test]
    fn trait_sigma_point_predict() {
        fn check<T: KalmanFilterSigmaPointPredict<3, f32>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);
    }

    #[test]
    fn observation_accessors() {
        let obs = make_dummy_observation_ukf();
        assert_eq!(obs.states(), 3);
        assert_eq!(obs.observations(), 2);
    }

    #[test]
    fn observation_measurement_accessors() {
        let mut obs = make_dummy_observation_ukf();
        let _ = obs.measurement_vector();
        let _ = obs.measurement_vector_mut();
    }

    #[test]
    fn observation_noise_accessors() {
        let mut obs = make_dummy_observation_ukf();
        let _ = obs.measurement_noise_covariance();
        let _ = obs.measurement_noise_covariance_mut();
    }

    #[test]
    fn trait_observation_num_states() {
        fn check<T: KalmanFilterNumStates<3>>(_: &T) {}
        let obs = make_dummy_observation_ukf();
        check(&obs);
    }

    #[test]
    fn trait_observation_num_observations() {
        fn check<T: KalmanFilterNumObservations<2>>(_: &T) {}
        let obs = make_dummy_observation_ukf();
        check(&obs);
    }

    #[test]
    fn trait_observation_measurement_vector() {
        fn check<T: KalmanFilterMeasurementVector<2, f32>>(_: &T) {}
        let obs = make_dummy_observation_ukf();
        check(&obs);
    }

    #[test]
    fn trait_observation_measurement_vector_mut() {
        fn check<T: KalmanFilterObservationVectorMut<2, f32>>(_: &mut T) {}
        let mut obs = make_dummy_observation_ukf();
        check(&mut obs);
    }

    #[test]
    fn trait_observation_noise_covariance() {
        fn check<T: KalmanFilterMeasurementNoiseCovariance<2, f32>>(_: &T) {}
        let obs = make_dummy_observation_ukf();
        check(&obs);
    }

    #[test]
    fn trait_observation_noise_covariance_mut() {
        fn check<T: KalmanFilterMeasurementNoiseCovarianceMut<2, f32>>(_: &mut T) {}
        let mut obs = make_dummy_observation_ukf();
        check(&mut obs);
    }

    #[test]
    fn trait_sigma_point_correct() {
        fn check<T: KalmanFilterSigmaPointCorrect<3, 7, f32>>(_: &T) {}
        let filter = make_dummy_filter_ukf();
        check(&filter);
    }
}
