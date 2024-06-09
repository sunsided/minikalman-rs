use core::marker::PhantomData;

use crate::filter_traits::*;
use crate::{FastUInt8, MatrixDataType};

/// A builder for a Kalman filter [`Measurement`] instances.
pub struct MeasurementBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP> {
    _phantom: (
        PhantomData<Z>,
        PhantomData<H>,
        PhantomData<R>,
        PhantomData<Y>,
        PhantomData<S>,
        PhantomData<K>,
        PhantomData<TempSInv>,
        PhantomData<TempHP>,
        PhantomData<TempPHt>,
        PhantomData<TempKHP>,
    ),
}

impl<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    MeasurementBuilder<H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
    /// Initializes a measurement.
    ///
    /// ## Arguments
    /// * `num_states` - The number of states tracked by the filter.
    /// * `num_measurements` - The number of measurements available to the filter.
    /// * `H` - The measurement transformation matrix (`num_measurements` × `num_states`).
    /// * `z` - The measurement vector (`num_measurements` × `1`).
    /// * `R` - The process noise / measurement uncertainty (`num_measurements` × `num_measurements`).
    /// * `y` - The innovation (`num_measurements` × `1`).
    /// * `S` - The residual covariance (`num_measurements` × `num_measurements`).
    /// * `K` - The Kalman gain (`num_states` × `num_measurements`).
    /// * `S_inv` - The temporary vector for predicted states (`num_states` × `1`).
    /// * `temp_HP` - The temporary matrix for H×P calculation (`num_measurements` × `num_states`).
    /// * `temp_PHt` - The temporary matrix for P×H' calculation (`num_states` × `num_measurements`).
    /// * `temp_KHP` - The temporary matrix for K×H×P calculation (`num_states` × `num_states`).
    /// ## Example
    ///
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// // Measurement buffers.
    /// let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    ///
    /// // Measurement temporaries.
    /// let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    ///
    /// let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
    ///     gravity_H,
    ///     gravity_z,
    ///     gravity_R,
    ///     gravity_y,
    ///     gravity_S,
    ///     gravity_K,
    ///     gravity_temp_S_inv,
    ///     gravity_temp_HP,
    ///     gravity_temp_PHt,
    ///     gravity_temp_KHP,
    /// );
    /// ```
    ///
    /// See also [`Kalman::new_direct`](crate::Kalman::new_direct) for setting up the Kalman filter itself.
    #[allow(non_snake_case, unused)]
    pub fn new<const STATES: usize, const MEASUREMENTS: usize, T>(
        H: H,
        z: Z,
        R: R,
        y: Y,
        S: S,
        K: K,
        temp_S_inv: TempSInv,
        temp_HP: TempHP,
        temp_PHt: TempPHt,
        temp_KHP: TempKHP,
    ) -> Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
    where
        T: MatrixDataType,
        Z: MeasurementVector<MEASUREMENTS, T>,
        H: MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>,
        R: MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
        Y: InnovationVector<MEASUREMENTS, T>,
        S: ResidualCovarianceMatrix<MEASUREMENTS, T>,
        K: KalmanGainMatrix<STATES, MEASUREMENTS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>,
        TempHP: TemporaryHPMatrix<MEASUREMENTS, STATES, T>,
        TempPHt: TemporaryPHTMatrix<STATES, MEASUREMENTS, T>,
        TempKHP: TemporaryKHPMatrix<STATES, T>,
    {
        Measurement::<STATES, MEASUREMENTS, T, _, _, _, _, _, _, _, _, _, _> {
            z,
            H,
            R,
            y,
            S,
            K,
            temp_S_inv,
            temp_HP,
            temp_PHt,
            temp_KHP,
            _phantom: PhantomData::default(),
        }
    }
}

/// Kalman Filter measurement structure. See [`MeasurementBuilder`] for construction.
#[allow(non_snake_case, unused)]
pub struct Measurement<
    const STATES: usize,
    const MEASUREMENTS: usize,
    T,
    H,
    Z,
    R,
    Y,
    S,
    K,
    TempSInv,
    TempHP,
    TempPHt,
    TempKHP,
> {
    /// Measurement vector.
    pub(crate) z: Z,

    /// Measurement transformation matrix.
    ///
    /// See also [`R`].
    pub(crate) H: H,

    /// Process noise covariance matrix.
    ///
    /// See also [`A`].
    pub(crate) R: R,

    /// Innovation vector.
    pub(crate) y: Y,

    /// Residual covariance matrix.
    pub(crate) S: S,

    /// Kalman gain matrix.
    pub(crate) K: K,

    /// S-Sized temporary matrix  (number of measurements × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_KHP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`tmp_HP`] (if it is not aliased with [`temp_PHt`]).
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_PHt`].
    pub(crate) temp_S_inv: TempSInv,

    /// H-Sized temporary matrix  (number of measurements × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_KHP`].
    pub(crate) temp_HP: TempHP,

    /// P×H'-Sized (H'-Sized) temporary matrix  (number of states × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`tmp_HP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_KHP`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_S_inv`].
    pub(crate) temp_PHt: TempPHt,

    /// P-Sized temporary matrix  (number of states × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`tmp_HP`].
    pub(crate) temp_KHP: TempKHP,

    _phantom: PhantomData<T>,
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
        T,
        Z,
        H,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempHP,
        TempKHP,
        TempPHt,
    > Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
{
    /// Returns then number of measurements.
    pub const fn measurements() -> FastUInt8 {
        MEASUREMENTS as _
    }

    /// Returns then number of states.
    pub const fn states() -> FastUInt8 {
        STATES as _
    }

    /// Gets a reference to the measurement vector z.
    #[inline(always)]
    pub fn measurement_vector_ref(&self) -> &Z {
        &self.z
    }

    /// Gets a mutable reference to the measurement vector z.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_vector")]
    pub fn measurement_vector_mut(&mut self) -> &mut Z {
        &mut self.z
    }

    /// Applies a function to the measurement vector z.
    ///
    /// ## Example
    /// ```
    /// # #![allow(non_snake_case)]
    /// # use minikalman::*;
    /// # const NUM_STATES: usize = 3;
    /// # const NUM_INPUTS: usize = 0;
    /// # const NUM_MEASUREMENTS: usize = 1;
    /// # // System buffers.
    /// # let gravity_x = create_buffer_x!(NUM_STATES);
    /// # let gravity_A = create_buffer_A!(NUM_STATES);
    /// # let gravity_P = create_buffer_P!(NUM_STATES);
    /// #
    /// # // Input buffers.
    /// # let gravity_u = create_buffer_u!(NUM_INPUTS);
    /// # let gravity_B = create_buffer_B!(NUM_STATES, NUM_INPUTS);
    /// # let gravity_Q = create_buffer_Q!(NUM_INPUTS);
    /// #
    /// # // Filter temporaries.
    /// # let gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    /// # let gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    /// # let gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
    /// #
    /// # let mut filter = KalmanBuilder::new::<NUM_STATES, NUM_INPUTS, f32>(
    /// #     gravity_A,
    /// #     gravity_x,
    /// #     gravity_B,
    /// #     gravity_u,
    /// #     gravity_P,
    /// #     gravity_Q,
    /// #     gravity_temp_x,
    /// #     gravity_temp_P,
    /// #     gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # let gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// # let gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// # let gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// # let gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// # let gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    /// #
    /// # // Measurement temporaries.
    /// # let gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// # let gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// # let gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    /// #
    /// # let mut measurement = MeasurementBuilder::new::<NUM_STATES, NUM_MEASUREMENTS, f32>(
    /// #     gravity_H,
    /// #     gravity_z,
    /// #     gravity_R,
    /// #     gravity_y,
    /// #     gravity_S,
    /// #     gravity_K,
    /// #     gravity_temp_S_inv,
    /// #     gravity_temp_HP,
    /// #     gravity_temp_PHt,
    /// #     gravity_temp_KHP,
    /// # );
    /// #
    /// # const REAL_DISTANCE: &[f32] = &[0.0, 0.0, 0.0];
    /// # const MEASUREMENT_ERROR: &[f32] = &[0.0, 0.0, 0.0];
    /// #
    /// for t in 0..REAL_DISTANCE.len() {
    ///     // Prediction.
    ///     filter.predict();
    ///
    ///     // Measure ...
    ///     let m = REAL_DISTANCE[t] + MEASUREMENT_ERROR[t];
    ///     measurement.measurement_vector_apply(|z| z[0] = m);
    ///
    ///     // Update.
    ///     filter.correct(&mut measurement);
    /// }
    /// ```
    #[inline(always)]
    pub fn measurement_vector_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Z),
    {
        f(&mut self.z)
    }

    /// Gets a reference to the measurement transformation matrix H.
    #[inline(always)]
    pub fn measurement_transformation_ref(&self) -> &H {
        &self.H
    }

    /// Gets a reference to the process noise matrix R.
    #[inline(always)]
    pub fn process_noise_ref(&self) -> &R {
        &self.R
    }

    /// Gets a mutable reference to the process noise matrix R.
    #[inline(always)]
    #[doc(alias = "kalman_get_process_noise")]
    pub fn process_noise_mut(&mut self) -> &mut R {
        &mut self.R
    }

    /// Applies a function to the process noise matrix R.
    #[inline(always)]
    pub fn process_noise_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut R),
    {
        f(&mut self.R)
    }
}

impl<
        const STATES: usize,
        const MEASUREMENTS: usize,
        T,
        Z,
        H,
        R,
        Y,
        S,
        K,
        TempSInv,
        TempHP,
        TempPHt,
        TempKHP,
    > Measurement<STATES, MEASUREMENTS, T, H, Z, R, Y, S, K, TempSInv, TempHP, TempPHt, TempKHP>
where
    H: MeasurementTransformationMatrixMut<MEASUREMENTS, STATES, T>,
{
    /// Gets a mutable reference to the measurement transformation matrix H.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_transformation")]
    pub fn measurement_transformation_mut(&mut self) -> &mut H {
        &mut self.H
    }

    /// Applies a function to the measurement transformation matrix H.
    #[inline(always)]
    pub fn measurement_transformation_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut H),
    {
        f(&mut self.H)
    }
}

#[cfg(test)]
mod tests {
    use core::ops::{Index, IndexMut};

    use crate::filter_traits::{MeasurementTransformationMatrixMut, MeasurementVectorMut};
    use crate::matrix_traits::{Matrix, MatrixMut};

    use super::*;

    #[test]
    fn builder_simple() {
        let _filter = MeasurementBuilder::new::<3, 1, f32>(
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
            Dummy::default(),
        );
    }

    #[derive(Default)]
    struct Dummy<T>(DummyMatrix<T>, PhantomData<T>);

    #[derive(Default)]
    struct DummyMatrix<T>(PhantomData<T>);

    impl<const STATES: usize, T> MeasurementVector<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<const STATES: usize, T> MeasurementVectorMut<STATES, T> for Dummy<T> {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        MeasurementTransformationMatrix<MEASUREMENTS, STATES, T> for Dummy<T>
    {
        type Target = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        MeasurementTransformationMatrixMut<MEASUREMENTS, STATES, T> for Dummy<T>
    {
        type TargetMut = DummyMatrix<T>;

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> MeasurementProcessNoiseCovarianceMatrix<MEASUREMENTS, T>
        for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::TargetMut {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> InnovationVector<MEASUREMENTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> ResidualCovarianceMatrix<MEASUREMENTS, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, const MEASUREMENTS: usize, T>
        KalmanGainMatrix<STATES, MEASUREMENTS, T> for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, T> TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>
        for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        TemporaryHPMatrix<MEASUREMENTS, STATES, T> for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, T> TemporaryKHPMatrix<STATES, T> for Dummy<T> {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<const STATES: usize, const MEASUREMENTS: usize, T>
        TemporaryPHTMatrix<STATES, MEASUREMENTS, T> for Dummy<T>
    {
        type Target = DummyMatrix<T>;
        type TargetMut = DummyMatrix<T>;

        fn as_matrix(&self) -> &Self::Target {
            &self.0
        }

        fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
            &mut self.0
        }
    }

    impl<T> AsRef<[T]> for DummyMatrix<T> {
        fn as_ref(&self) -> &[T] {
            todo!()
        }
    }

    impl<T> AsMut<[T]> for DummyMatrix<T> {
        fn as_mut(&mut self) -> &mut [T] {
            todo!()
        }
    }

    impl<T> Index<usize> for DummyMatrix<T> {
        type Output = T;

        fn index(&self, _index: usize) -> &Self::Output {
            todo!()
        }
    }

    impl<T> IndexMut<usize> for DummyMatrix<T> {
        fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
            todo!()
        }
    }

    impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix<T> {}
    impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T> for DummyMatrix<T> {}
}
