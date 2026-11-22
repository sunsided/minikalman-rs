use core::marker::PhantomData;

use crate::matrix::{MatrixData, MatrixDataBoxed, MatrixDataType};

use crate::buffers::builder::*;

use crate::information::InformationFilter;
use crate::kalman::{Current, GainForm};
use crate::regular::{
    Control, ControlBuilder, FixedGainGate, FixedGainObservation, LuenbergerObserver,
    PendingCorrection, RegularKalman, RegularKalmanBuilder, RegularObservation,
    RegularObservationBuilder,
};
use crate::square_root::SquareRootKalman;
use crate::ud::UdKalman;

/// A simple builder for [`RegularKalman`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`Control`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterControlBuilder<const STATES: usize, T>(PhantomData<T>);

/// A simple builder for [`RegularObservation`] instances.
#[derive(Copy, Clone)]
pub struct KalmanFilterObservationBuilder<const STATES: usize, T>(PhantomData<T>);

impl<const STATES: usize, T> Default for KalmanFilterBuilder<STATES, T> {
    fn default() -> Self {
        KalmanFilterBuilder::new()
    }
}

/// The type of Kalman filters with owned buffers.
///
/// See also the [`KalmanFilter`](crate::kalman::KalmanFilter) trait.
pub type KalmanFilterType<const STATES: usize, T> = RegularKalman<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    EstimateCovarianceMatrixBufferOwnedType<STATES, T>,
    DirectProcessNoiseCovarianceMatrixBufferOwnedType<STATES, T>,
    TemporaryStatePredictionVectorBufferOwnedType<STATES, T>,
    TemporaryStateMatrixBufferOwnedType<STATES, T>,
>;

/// The type of information filters with owned buffers.
///
/// See also [`KalmanFilterBuilder::build_information`].
pub type InformationFilterType<const STATES: usize, T> = InformationFilter<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    EstimateCovarianceMatrixBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    DirectProcessNoiseCovarianceMatrixBufferOwnedType<STATES, T>,
    TemporaryStatePredictionVectorBufferOwnedType<STATES, T>,
    TemporaryStateMatrixBufferOwnedType<STATES, T>,
    TemporaryStateMatrixBufferOwnedType<STATES, T>,
>;

/// The type of UD filters with owned buffers.
///
/// See also [`KalmanFilterBuilder::build_ud`].
pub type UdKalmanType<const STATES: usize, const TWO_STATES: usize, T> = UdKalman<
    STATES,
    TWO_STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    EstimateCovarianceMatrixBufferOwnedType<STATES, T>,
    DirectProcessNoiseCovarianceMatrixBufferOwnedType<STATES, T>,
    MatrixDataBoxed<STATES, TWO_STATES, T>,
    MatrixDataBoxed<TWO_STATES, 1, T>,
>;

/// The type of square-root covariance filters with owned buffers.
///
/// `TWO_STATES` must be `2 × STATES`. See also [`KalmanFilterBuilder::build_square_root`].
pub type SquareRootKalmanType<const STATES: usize, const TWO_STATES: usize, T> = SquareRootKalman<
    STATES,
    TWO_STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    EstimateCovarianceMatrixBufferOwnedType<STATES, T>,
    DirectProcessNoiseCovarianceMatrixBufferOwnedType<STATES, T>,
    MatrixDataBoxed<TWO_STATES, STATES, T>,
>;

/// The type of [`Current`]-form fixed-gain (Luenberger) observers with owned buffers.
///
/// See also [`KalmanFilterBuilder::build_observer`].
pub type LuenbergerObserverType<const STATES: usize, T> = LuenbergerObserver<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    TemporaryStatePredictionVectorBufferOwnedType<STATES, T>,
>;

/// The type of [`Predictor`](crate::kalman::Predictor)-form fixed-gain (Luenberger) observers
/// with owned buffers.
///
/// See also [`KalmanFilterBuilder::build_predictor_observer`].
pub type PredictorObserverType<const STATES: usize, T> = LuenbergerObserver<
    STATES,
    T,
    SystemMatrixMutBufferOwnedType<STATES, T>,
    StateVectorBufferOwnedType<STATES, T>,
    TemporaryStatePredictionVectorBufferOwnedType<STATES, T>,
    PendingCorrection<StateVectorBufferOwnedType<STATES, T>>,
>;

/// The type of fixed-gain observations of the [`GainForm`] `F` with owned buffers.
///
/// See also [`KalmanFilterObservationBuilder::build_fixed_gain`].
pub type FixedGainObservationType<const STATES: usize, const OBSERVATIONS: usize, T, F = Current> =
    FixedGainObservation<
        STATES,
        OBSERVATIONS,
        T,
        ObservationMatrixBufferOwnedType<OBSERVATIONS, STATES, T>,
        ObservationVectorBufferOwnedType<OBSERVATIONS, T>,
        InnovationVectorBufferOwnedType<OBSERVATIONS, T>,
        KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
        F,
    >;

/// The type of gated fixed-gain observations of the [`GainForm`] `F` with owned buffers.
///
/// See also [`KalmanFilterObservationBuilder::build_fixed_gain_gated`].
pub type GatedFixedGainObservationType<
    const STATES: usize,
    const OBSERVATIONS: usize,
    T,
    F = Current,
> = FixedGainObservation<
    STATES,
    OBSERVATIONS,
    T,
    ObservationMatrixBufferOwnedType<OBSERVATIONS, STATES, T>,
    ObservationVectorBufferOwnedType<OBSERVATIONS, T>,
    InnovationVectorBufferOwnedType<OBSERVATIONS, T>,
    KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
    F,
    FixedGainGate<
        OBSERVATIONS,
        T,
        InnovationResidualCovarianceMatrixBufferOwnedType<OBSERVATIONS, T>,
    >,
>;

impl<const STATES: usize, T> KalmanFilterBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    /// const NUM_OBSERVATIONS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// let mut filter = builder.build();
    /// let mut control = builder.controls().build::<NUM_CONTROLS>();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    /// # }
    /// ```
    ///
    /// See also [`KalmanFilterControlBuilder`] and [`KalmanFilterObservationBuilder`] for further information.
    pub fn build(&self) -> KalmanFilterType<STATES, T>
    where
        T: MatrixDataType + Default,
    {
        // System buffers.
        let state_vector = BufferBuilder::state_vector_x::<STATES>().new();
        let system_matrix = BufferBuilder::system_matrix_A::<STATES>().new();
        let system_covariance = BufferBuilder::estimate_covariance_P::<STATES>().new();
        let process_noise = BufferBuilder::direct_process_noise_covariance_Q::<STATES>().new();

        // Filter temporaries.
        let temp_x = BufferBuilder::state_prediction_temp_x::<STATES>().new();
        let temp_p = BufferBuilder::temp_system_covariance_P::<STATES>().new();

        RegularKalmanBuilder::new::<STATES, T>(
            system_matrix,
            state_vector,
            system_covariance,
            process_noise,
            temp_x,
            temp_p,
        )
    }

    /// Builds a new [`Current`]-form fixed-gain (Luenberger) observer using heap allocated
    /// buffers.
    ///
    /// It holds only `A`, `x` and one temporary vector; pair it with observations from
    /// [`KalmanFilterObservationBuilder::build_fixed_gain`] in the [`Current`] form.
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f32>::default();
    /// let mut observer = builder.build_observer();
    /// let mut position = builder.observations().build_fixed_gain::<1, Current>();
    /// # }
    /// ```
    pub fn build_observer(&self) -> LuenbergerObserverType<STATES, T>
    where
        T: MatrixDataType + Default,
    {
        LuenbergerObserver::new(
            BufferBuilder::system_matrix_A::<STATES>().new(),
            BufferBuilder::state_vector_x::<STATES>().new(),
            BufferBuilder::state_prediction_temp_x::<STATES>().new(),
        )
    }

    /// Builds a new [`Predictor`](crate::kalman::Predictor)-form fixed-gain (Luenberger)
    /// observer using heap allocated buffers.
    ///
    /// Like [`build_observer`](Self::build_observer), plus the zeroed pending correction that
    /// its corrections defer to the next prediction.
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f32>::default();
    /// let mut observer = builder.build_predictor_observer();
    /// let mut position = builder.observations().build_fixed_gain::<1, Predictor>();
    /// # }
    /// ```
    pub fn build_predictor_observer(&self) -> PredictorObserverType<STATES, T>
    where
        T: MatrixDataType + Default,
    {
        LuenbergerObserver::new_predictor(
            BufferBuilder::system_matrix_A::<STATES>().new(),
            BufferBuilder::state_vector_x::<STATES>().new(),
            BufferBuilder::state_prediction_temp_x::<STATES>().new(),
            BufferBuilder::state_vector_x::<STATES>().new(),
        )
    }

    /// Builds a new information filter using heap allocated buffers.
    ///
    /// The filter starts with zero information (`Y = 0`, `ŷ = 0`). Observations built with
    /// [`observations`](Self::observations) can be fused into it with
    /// [`InformationFilter::fuse`].
    pub fn build_information(&self) -> InformationFilterType<STATES, T>
    where
        T: MatrixDataType + Default,
    {
        InformationFilter::new(
            BufferBuilder::system_matrix_A::<STATES>().new(),
            BufferBuilder::estimate_covariance_P::<STATES>().new(),
            BufferBuilder::state_vector_x::<STATES>().new(),
            BufferBuilder::direct_process_noise_covariance_Q::<STATES>().new(),
            BufferBuilder::state_prediction_temp_x::<STATES>().new(),
            BufferBuilder::temp_system_covariance_P::<STATES>().new(),
            BufferBuilder::temp_system_covariance_P::<STATES>().new(),
        )
    }

    /// Builds a new UD filter using heap allocated buffers.
    ///
    /// `TWO_STATES` must be `2 × STATES`. The filter starts with `x = 0`, `P = 0` and `Q = 0`;
    /// set them with [`UdKalman::set_state`] and [`UdKalman::set_process_noise`]. Observations
    /// built with [`observations`](Self::observations) are applied with [`UdKalman::correct`].
    ///
    /// ## Panics
    /// Panics if `TWO_STATES` is not `2 × STATES`, unless the `no_assert` feature is enabled.
    pub fn build_ud<const TWO_STATES: usize>(&self) -> UdKalmanType<STATES, TWO_STATES, T>
    where
        T: MatrixDataType + Default,
    {
        UdKalman::new(
            BufferBuilder::system_matrix_A::<STATES>().new(),
            BufferBuilder::state_vector_x::<STATES>().new(),
            BufferBuilder::estimate_covariance_P::<STATES>().new(),
            BufferBuilder::direct_process_noise_covariance_Q::<STATES>().new(),
            MatrixData::new_boxed::<STATES, TWO_STATES, T, _>(alloc::vec![
                T::default();
                STATES * TWO_STATES
            ]),
            MatrixData::new_boxed::<TWO_STATES, 1, T, _>(alloc::vec![T::default(); TWO_STATES]),
        )
    }

    /// Builds a new square-root covariance filter using heap allocated buffers.
    ///
    /// `TWO_STATES` is the row count of the prearray and must be `2 × STATES`. The filter
    /// starts with `x = 0`, `S = 0` and `Lq = 0`; set them with
    /// [`SquareRootKalman::set_state`] and [`SquareRootKalman::set_process_noise`], or
    /// directly as factors. Observations built with [`observations`](Self::observations) are
    /// applied with [`SquareRootKalman::correct`].
    ///
    /// ## Panics
    /// If `TWO_STATES` is not `2 × STATES` (unless the `no_assert` feature is enabled).
    pub fn build_square_root<const TWO_STATES: usize>(
        &self,
    ) -> SquareRootKalmanType<STATES, TWO_STATES, T>
    where
        T: MatrixDataType + Default,
    {
        SquareRootKalman::new(
            BufferBuilder::system_matrix_A::<STATES>().new(),
            BufferBuilder::state_vector_x::<STATES>().new(),
            BufferBuilder::estimate_covariance_P::<STATES>().new(),
            BufferBuilder::direct_process_noise_covariance_Q::<STATES>().new(),
            MatrixDataBoxed::new(alloc::vec![T::default(); TWO_STATES * STATES]),
        )
    }

    /// Convenience function to return a [`KalmanFilterControlBuilder`].
    pub fn controls(&self) -> KalmanFilterControlBuilder<STATES, T> {
        Default::default()
    }

    /// Convenience function to return a [`KalmanFilterObservationBuilder`].
    pub fn observations(&self) -> KalmanFilterObservationBuilder<STATES, T> {
        Default::default()
    }
}

impl<const STATES: usize, T> Default for KalmanFilterControlBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The type of Kalman filter controls with owned buffers.
///
/// See also the [`KalmanFilterControl`](crate::kalman::KalmanFilterControl) trait.
pub type KalmanFilterControlType<const STATES: usize, const CONTROLS: usize, T> = Control<
    STATES,
    CONTROLS,
    T,
    ControlMatrixBufferOwnedType<STATES, CONTROLS, T>,
    ControlVectorBufferOwnedType<CONTROLS, T>,
    ControlProcessNoiseCovarianceMatrixBufferOwnedType<CONTROLS, T>,
    TemporaryBQMatrixBufferOwnedType<STATES, CONTROLS, T>,
>;

impl<const STATES: usize, T> KalmanFilterControlBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterControlBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter control input using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_CONTROLS: usize = 2;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut control = builder.controls().build::<NUM_CONTROLS>();
    /// # }
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterObservationBuilder`] for further information.
    pub fn build<const CONTROLS: usize>(&self) -> KalmanFilterControlType<STATES, CONTROLS, T>
    where
        T: MatrixDataType + Default,
    {
        // Control buffers.
        let control_vector = BufferBuilder::control_vector_u::<CONTROLS>().new();
        let control_matrix = BufferBuilder::control_matrix_B::<STATES, CONTROLS>().new();
        let control_covariance =
            BufferBuilder::control_process_noise_covariance_Q::<CONTROLS>().new();

        // Control temporaries.
        let temp_bq = BufferBuilder::temp_BQ::<STATES, CONTROLS>().new();

        ControlBuilder::new::<STATES, CONTROLS, T>(
            control_matrix,
            control_vector,
            control_covariance,
            temp_bq,
        )
    }
}

impl<const STATES: usize, T> Default for KalmanFilterObservationBuilder<STATES, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// The type of Kalman filter measurement / observation with owned buffers.
///
/// See also the [`KalmanFilterObservation`](crate::kalman::KalmanFilterObservation) trait.
pub type KalmanFilterObservationType<const STATES: usize, const OBSERVATIONS: usize, T> =
    RegularObservation<
        STATES,
        OBSERVATIONS,
        T,
        ObservationMatrixBufferOwnedType<OBSERVATIONS, STATES, T>,
        ObservationVectorBufferOwnedType<OBSERVATIONS, T>,
        MeasurementNoiseCovarianceBufferOwnedType<OBSERVATIONS, T>,
        InnovationVectorBufferOwnedType<OBSERVATIONS, T>,
        InnovationResidualCovarianceMatrixBufferOwnedType<OBSERVATIONS, T>,
        KalmanGainMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
        TemporarySInvertedMatrixBufferOwnedType<OBSERVATIONS, T>,
        TemporaryHPMatrixBufferOwnedType<OBSERVATIONS, STATES, T>,
        TemporaryPHtMatrixBufferOwnedType<STATES, OBSERVATIONS, T>,
        TemporaryKHPMatrixBufferOwnedType<STATES, T>,
    >;

impl<const STATES: usize, T> KalmanFilterObservationBuilder<STATES, T> {
    /// Creates a new [`KalmanFilterObservationBuilder`] instance.
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Builds a new Kalman filter measurements using heap allocated buffers.
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// const NUM_STATES: usize = 3;
    /// const NUM_OBSERVATIONS: usize = 5;
    ///
    /// let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
    /// // let mut filter = builder.build();
    /// let mut measurement = builder.observations().build::<NUM_OBSERVATIONS>();
    /// # }
    /// ```
    ///
    /// See also [`KalmanFilterBuilder`] and [`KalmanFilterControlBuilder`] for further information.
    pub fn build<const OBSERVATIONS: usize>(
        &self,
    ) -> KalmanFilterObservationType<STATES, OBSERVATIONS, T>
    where
        T: MatrixDataType + Default,
    {
        // Observation buffers.
        let measurement_vector = BufferBuilder::measurement_vector_z::<OBSERVATIONS>().new();
        let observation_matrix =
            BufferBuilder::observation_matrix_H::<OBSERVATIONS, STATES>().new();
        let observation_covariance =
            BufferBuilder::observation_covariance_R::<OBSERVATIONS>().new();
        let innovation_vector = BufferBuilder::innovation_vector_y::<OBSERVATIONS>().new();
        let residual_covariance_matrix =
            BufferBuilder::innovation_covariance_S::<OBSERVATIONS>().new();
        let kalman_gain = BufferBuilder::kalman_gain_K::<STATES, OBSERVATIONS>().new();

        // Observation temporaries.
        let temp_s_inverted = BufferBuilder::temp_S_inv::<OBSERVATIONS>().new();
        let temp_hp = BufferBuilder::temp_HP::<OBSERVATIONS, STATES>().new();
        let temp_pht = BufferBuilder::temp_PHt::<STATES, OBSERVATIONS>().new();
        let temp_khp = BufferBuilder::temp_KHP::<STATES>().new();

        RegularObservationBuilder::new::<STATES, OBSERVATIONS, T>(
            observation_matrix,
            measurement_vector,
            observation_covariance,
            innovation_vector,
            residual_covariance_matrix,
            kalman_gain,
            temp_s_inverted,
            temp_hp,
            temp_pht,
            temp_khp,
        )
    }

    /// Builds a new fixed-gain observation for a
    /// [`LuenbergerObserver`] using heap allocated buffers.
    ///
    /// The gain, in the [`GainForm`] `F`, starts at zero; set it with
    /// [`set_gain`](FixedGainObservation::set_gain).
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f32>::default();
    /// let mut position = builder.observations().build_fixed_gain::<1, Current>();
    /// position.set_gain(&MatrixData::new_array::<2, 1, 2, f32>([0.181201, 0.180975]));
    /// # }
    /// ```
    pub fn build_fixed_gain<const OBSERVATIONS: usize, F>(
        &self,
    ) -> FixedGainObservationType<STATES, OBSERVATIONS, T, F>
    where
        T: MatrixDataType + Default,
        F: GainForm,
    {
        FixedGainObservation::new(
            BufferBuilder::observation_matrix_H::<OBSERVATIONS, STATES>().new(),
            BufferBuilder::measurement_vector_z::<OBSERVATIONS>().new(),
            BufferBuilder::innovation_vector_y::<OBSERVATIONS>().new(),
            BufferBuilder::kalman_gain_K::<STATES, OBSERVATIONS>().new(),
        )
    }

    /// Builds a new fixed-gain observation with an innovation gate, using heap allocated
    /// buffers.
    ///
    /// Like [`build_fixed_gain`](Self::build_fixed_gain), plus a zeroed `S∞⁻¹`; fill it with
    /// [`set_innovation_covariance`](FixedGainObservation::set_innovation_covariance) and set
    /// a threshold with [`set_gate`](FixedGainObservation::set_gate).
    ///
    /// ## Example
    /// ```rust
    /// # #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))] {
    /// use minikalman::prelude::*;
    /// use minikalman::regular::builder::KalmanFilterBuilder;
    ///
    /// let builder = KalmanFilterBuilder::<2, f32>::default();
    /// let mut position = builder.observations().build_fixed_gain_gated::<1, Current>();
    /// position.set_gate(Some(6.63));
    /// # }
    /// ```
    pub fn build_fixed_gain_gated<const OBSERVATIONS: usize, F>(
        &self,
    ) -> GatedFixedGainObservationType<STATES, OBSERVATIONS, T, F>
    where
        T: MatrixDataType + Default,
        F: GainForm,
    {
        self.build_fixed_gain()
            .with_gate(BufferBuilder::innovation_covariance_S::<OBSERVATIONS>().new())
    }
}

#[cfg(all(test, any(feature = "std", feature = "libm", feature = "micromath")))]
mod tests {
    use super::*;
    use crate::kalman::{KalmanFilter, KalmanFilterControl, KalmanFilterObservation};

    const NUM_STATES: usize = 3; // height, upwards velocity, upwards acceleration
    const NUM_CONTROLS: usize = 1; // constant velocity
    const NUM_OBSERVATIONS: usize = 1; // position

    fn accept_filter<F, T>(_filter: F)
    where
        F: KalmanFilter<NUM_STATES, T>,
    {
    }

    fn accept_control<I, T>(_control: I)
    where
        I: KalmanFilterControl<NUM_STATES, NUM_CONTROLS, T>,
    {
    }

    fn accept_observation<M, T>(_measurement: M)
    where
        M: KalmanFilterObservation<NUM_STATES, NUM_OBSERVATIONS, T>,
    {
    }

    #[test]
    fn kalman_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let filter = builder.build();
        assert_eq!(filter.states(), NUM_STATES);
        accept_filter(filter);
    }

    #[test]
    fn control_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let control = builder.controls().build::<NUM_CONTROLS>();
        assert_eq!(control.states(), NUM_STATES);
        assert_eq!(control.controls(), NUM_CONTROLS);
        accept_control(control);
    }

    #[test]
    fn measurement_builder() {
        let builder = KalmanFilterBuilder::<NUM_STATES, f32>::default();
        let measurement = builder.observations().build::<NUM_OBSERVATIONS>();
        assert_eq!(measurement.states(), NUM_STATES);
        assert_eq!(measurement.measurements(), NUM_OBSERVATIONS);
        accept_observation(measurement);
    }
}
