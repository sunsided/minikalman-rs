//! # Fixed-gain observations for the Luenberger observer.

use core::marker::PhantomData;

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixDataType, MatrixMut, RowMajorSequentialDataMut};

/// A measurement with a fixed gain for the [`LuenbergerObserver`](crate::regular::LuenbergerObserver).
///
/// Holds only the observation matrix `H`, the measurement `z`, the innovation `y` and the gain,
/// plus the [`GainForm`] the gain is written in: `OBSERVATIONS·STATES·2 + OBSERVATIONS·2`
/// scalars. There is no `R`, `S` or covariance temporary, so the correction costs
/// `2·OBSERVATIONS·STATES` multiply-adds, with no square root, division or data-dependent
/// branch.
///
/// Set the gain with [`set_gain_current`](Self::set_gain_current) (a Kalman-style gain `K`,
/// e.g. from [`RegularKalman::settle_gain`](crate::regular::RegularKalman::settle_gain)) or
/// [`set_gain_predictor`](Self::set_gain_predictor) (a textbook observer gain `L`, e.g. from
/// `place` or python-control `dlqe`). The setter records the form, and the observer refuses to
/// run a gain in the recursion of the other form; see
/// [the gain convention trap](crate::regular::LuenbergerObserver#the-gain-convention-trap).
///
/// There is no innovation gate yet
/// ([#144](https://github.com/sunsided/minikalman-rs/issues/144)).
#[allow(non_snake_case)]
pub struct FixedGainObservation<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K> {
    /// Observation matrix `H` (`OBSERVATIONS × STATES`).
    H: H,

    /// Measurement vector `z` (`OBSERVATIONS × 1`).
    z: Z,

    /// Innovation vector `y = z − H·x̂` of the last correction (`OBSERVATIONS × 1`).
    y: Y,

    /// The fixed gain (`STATES × OBSERVATIONS`), `K` or `L` as [`form`](Self::form) says.
    K: K,

    /// The convention the gain in [`K`](Self::K) is written in.
    form: GainForm,

    /// Marks the scalar type.
    _phantom: PhantomData<T>,
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
{
    /// Creates a fixed-gain observation from its buffers.
    ///
    /// `form` states the convention of the gain that `K` already holds; with a zeroed `K`,
    /// either form is harmless until a gain is set with
    /// [`set_gain_current`](Self::set_gain_current) or
    /// [`set_gain_predictor`](Self::set_gain_predictor).
    ///
    /// ## Arguments
    /// * `H` - The observation matrix (`OBSERVATIONS` × `STATES`).
    /// * `z` - The measurement vector (`OBSERVATIONS` × `1`).
    /// * `y` - The innovation vector (`OBSERVATIONS` × `1`).
    /// * `K` - The gain (`STATES` × `OBSERVATIONS`).
    /// * `form` - The convention the gain in `K` is written in.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// use minikalman::prelude::*;
    /// use minikalman::regular::FixedGainObservation;
    ///
    /// impl_buffer_H!(mut H, 1, 2, f32, 0.0);
    /// impl_buffer_z!(mut z, 1, f32, 0.0);
    /// impl_buffer_y!(mut y, 1, f32, 0.0);
    /// impl_buffer_K!(mut K, 2, 1, f32, 0.0);
    ///
    /// let mut position = FixedGainObservation::<2, 1, f32, _, _, _, _>::new(H, z, y, K, GainForm::Current);
    /// position.observation_matrix_mut().set_at(0, 0, 1.0);
    /// position.set_gain_current(&MatrixData::new_array::<2, 1, 2, f32>([0.181201, 0.180975]));
    /// ```
    #[allow(non_snake_case)]
    pub fn new(H: H, z: Z, y: Y, K: K, form: GainForm) -> Self {
        Self {
            H,
            z,
            y,
            K,
            form,
            _phantom: PhantomData,
        }
    }

    /// Returns the number of states.
    #[inline(always)]
    pub const fn states(&self) -> usize {
        STATES
    }

    /// Returns the number of measurements.
    #[inline(always)]
    pub const fn measurements(&self) -> usize {
        OBSERVATIONS
    }

    /// Returns the convention the stored gain is written in.
    #[inline(always)]
    pub fn gain_form(&self) -> GainForm {
        self.form
    }

    /// Gets a reference to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector(&self) -> &Z {
        &self.z
    }

    /// Gets a mutable reference to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_mut(&mut self) -> &mut Z {
        &mut self.z
    }

    /// Gets a reference to the observation matrix H.
    #[inline(always)]
    pub fn observation_matrix(&self) -> &H {
        &self.H
    }

    /// Gets a mutable reference to the observation matrix H.
    #[inline(always)]
    pub fn observation_matrix_mut(&mut self) -> &mut H {
        &mut self.H
    }

    /// Gets a reference to the innovation vector `y = z − H·x̂` of the last correction.
    #[inline(always)]
    pub fn innovation_vector(&self) -> &Y {
        &self.y
    }

    /// Gets a reference to the stored gain; [`gain_form`](Self::gain_form) says whether it is
    /// a current-form `K` or a predictor-form `L`.
    ///
    /// There is deliberately no mutable access: the gain is only written together with its
    /// form, by [`set_gain_current`](Self::set_gain_current) or
    /// [`set_gain_predictor`](Self::set_gain_predictor).
    #[inline(always)]
    pub fn gain(&self) -> &K {
        &self.K
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
where
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    T: Copy,
{
    /// Stores a gain `K` in the **current (filtered) form** `x̂⁺ = x̂⁻ + K·(z − H·x̂⁻)`.
    ///
    /// This is the crate's convention and the Kalman gain: copy it from
    /// [`RegularObservation::kalman_gain`](crate::regular::RegularObservation::kalman_gain)
    /// after [`RegularKalman::settle_gain`](crate::regular::RegularKalman::settle_gain), or from
    /// MATLAB `kalman`'s `Mx`. Run it with
    /// [`LuenbergerObserver::predict`](crate::regular::LuenbergerObserver::predict) then
    /// [`correct`](crate::regular::LuenbergerObserver::correct). The error dynamics are
    /// `(I − K·H)·A`.
    ///
    /// A pole placement gives `K` directly when designed on the pair `(A, H·A)`, e.g.
    /// `K = place(A.T, (H @ A).T, poles).T`, because `(I − K·H)·A = A − K·(H·A)`.
    ///
    /// ## Arguments
    /// * `gain` - The gain `K` (`STATES` × `OBSERVATIONS`).
    pub fn set_gain_current<G>(&mut self, gain: &G)
    where
        G: Matrix<STATES, OBSERVATIONS, T>,
    {
        gain.copy_to(self.K.as_matrix_mut());
        self.form = GainForm::Current;
    }

    /// Stores a gain `L` in the **predictor (textbook Luenberger) form**
    /// `x̂ₖ₊₁ = A·x̂ₖ + B·uₖ + L·(zₖ − H·x̂ₖ)`.
    ///
    /// This is what `place(A', C')'`, `acker` and python-control `dlqe` return. Run it with
    /// [`LuenbergerObserver::predict_from_measurement`](crate::regular::LuenbergerObserver::predict_from_measurement),
    /// which is the whole step; do not call `predict` in the same step. The error dynamics are
    /// `A − L·H`, the same poles as the current form with `K` where `L = A·K`.
    ///
    /// ## Arguments
    /// * `gain` - The gain `L` (`STATES` × `OBSERVATIONS`).
    pub fn set_gain_predictor<G>(&mut self, gain: &G)
    where
        G: Matrix<STATES, OBSERVATIONS, T>,
    {
        gain.copy_to(self.K.as_matrix_mut());
        self.form = GainForm::Predictor;
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    KalmanFilterObservationCorrectFixedGain<STATES, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
    Z: MeasurementVector<OBSERVATIONS, T>,
    Y: InnovationVector<OBSERVATIONS, T>,
    K: KalmanGainMatrix<STATES, OBSERVATIONS, T>,
    T: MatrixDataType,
{
    #[inline(always)]
    fn gain_form(&self) -> GainForm {
        self.form
    }

    fn compute_innovation<X, N>(&mut self, x: &X, mut fix: N)
    where
        X: Matrix<STATES, 1, T>,
        N: FnMut(&mut [T]),
    {
        let y = self.y.as_matrix_mut();
        self.H.as_matrix().mult_rowvector(x, y);
        self.z.as_matrix().sub_inplace_b(y);
        fix(y.as_mut_slice());
    }

    #[inline(always)]
    fn add_gain_times_innovation<V>(&self, target: &mut V)
    where
        V: MatrixMut<STATES, 1, T>,
    {
        self.K
            .as_matrix()
            .multadd_rowvector(self.y.as_matrix(), target);
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K> KalmanFilterNumStates<STATES>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    KalmanFilterNumObservations<OBSERVATIONS>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
{
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    KalmanFilterMeasurementVector<OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
where
    Z: MeasurementVector<OBSERVATIONS, T>,
{
    type MeasurementVector = Z;

    #[inline(always)]
    fn measurement_vector(&self) -> &Self::MeasurementVector {
        self.measurement_vector()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    KalmanFilterObservationVectorMut<OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
where
    Z: MeasurementVectorMut<OBSERVATIONS, T>,
{
    type MeasurementVectorMut = Z;

    #[inline(always)]
    fn measurement_vector_mut(&mut self) -> &mut Self::MeasurementVectorMut {
        self.measurement_vector_mut()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    KalmanFilterObservationTransformation<STATES, OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
where
    H: ObservationMatrix<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationMatrix = H;

    #[inline(always)]
    fn observation_matrix(&self) -> &Self::ObservationTransformationMatrix {
        self.observation_matrix()
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T, H, Z, Y, K>
    KalmanFilterObservationTransformationMut<STATES, OBSERVATIONS, T>
    for FixedGainObservation<STATES, OBSERVATIONS, T, H, Z, Y, K>
where
    H: ObservationMatrixMut<OBSERVATIONS, STATES, T>,
{
    type ObservationTransformationMatrixMut = H;

    #[inline(always)]
    fn observation_matrix_mut(&mut self) -> &mut Self::ObservationTransformationMatrixMut {
        self.observation_matrix_mut()
    }
}
