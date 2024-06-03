use crate::types::FastUInt8;
use crate::Matrix;
use num_traits::Float;

/// Kalman Filter measurement structure.
#[allow(non_snake_case, unused)]
pub struct Measurement<'a, const STATES: usize, const MEASUREMENTS: usize, T = f32> {
    /// Measurement vector.
    pub(crate) z: Matrix<'a, MEASUREMENTS, 1, T>,
    /// Measurement transformation matrix.
    ///
    /// See also [`R`].
    pub(crate) H: Matrix<'a, MEASUREMENTS, STATES, T>,
    /// Process noise covariance matrix.
    ///
    /// See also [`A`].
    pub(crate) R: Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>,
    /// Innovation vector.
    pub(crate) y: Matrix<'a, MEASUREMENTS, 1, T>,
    /// Residual covariance matrix.
    pub(crate) S: Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>,
    /// Kalman gain matrix.
    pub(crate) K: Matrix<'a, STATES, MEASUREMENTS, T>,

    /// Temporary storage.
    pub(crate) temporary: MeasurementTemporary<'a, STATES, MEASUREMENTS, T>,
}

#[allow(non_snake_case)]
pub(crate) struct MeasurementTemporary<'a, const STATES: usize, const MEASUREMENTS: usize, T = f32>
{
    /// S-Sized temporary matrix  (number of measurements × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`KHP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`HP`] (if it is not aliased with [`PHt`]).
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`PHt`].
    pub(crate) S_inv: Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>,
    /// H-Sized temporary matrix  (number of measurements × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`KHP`].
    pub(crate) HP: Matrix<'a, MEASUREMENTS, STATES, T>,
    /// P-Sized temporary matrix  (number of states × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`HP`].
    pub(crate) KHP: Matrix<'a, STATES, STATES, T>,
    /// P×H'-Sized (H'-Sized) temporary matrix  (number of states × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`HP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`KHP`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`S_inv`].
    pub(crate) PHt: Matrix<'a, STATES, MEASUREMENTS, T>,
}

impl<'a, const STATES: usize, const MEASUREMENTS: usize, T>
    Measurement<'a, STATES, MEASUREMENTS, T>
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
    /// let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    ///
    /// // Measurement temporaries.
    /// let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    ///
    /// let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS>::new_direct(
    ///     &mut gravity_H,
    ///     &mut gravity_z,
    ///     &mut gravity_R,
    ///     &mut gravity_y,
    ///     &mut gravity_S,
    ///     &mut gravity_K,
    ///     &mut gravity_temp_S_inv,
    ///     &mut gravity_temp_HP,
    ///     &mut gravity_temp_PHt,
    ///     &mut gravity_temp_KHP,
    /// );
    /// ```
    ///
    /// See also [`Kalman::new_direct`](crate::Kalman::new_direct) for setting up the Kalman filter itself.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc(alias = "kalman_measurement_initialize")]
    pub fn new_direct(
        H: &'a mut [T],
        z: &'a mut [T],
        R: &'a mut [T],
        y: &'a mut [T],
        S: &'a mut [T],
        K: &'a mut [T],
        S_inv: &'a mut [T],
        temp_HP: &'a mut [T],
        temp_PHt: &'a mut [T],
        temp_KHP: &'a mut [T],
    ) -> Self
    where
        T: Float,
    {
        Self::new(
            Matrix::<MEASUREMENTS, STATES, T>::new(H),
            Matrix::<MEASUREMENTS, 1, T>::new(z),
            Matrix::<MEASUREMENTS, MEASUREMENTS, T>::new(R),
            Matrix::<MEASUREMENTS, 1, T>::new(y),
            Matrix::<MEASUREMENTS, MEASUREMENTS, T>::new(S),
            Matrix::<STATES, MEASUREMENTS, T>::new(K),
            Matrix::<MEASUREMENTS, MEASUREMENTS, T>::new(S_inv),
            Matrix::<MEASUREMENTS, STATES, T>::new(temp_HP),
            Matrix::<STATES, MEASUREMENTS, T>::new(temp_PHt),
            Matrix::<STATES, STATES, T>::new(temp_KHP),
        )
    }

    /// Initializes a measurement.
    ///
    /// ## Arguments
    /// * `H` - The measurement transformation matrix (`num_measurements` × `num_states`).
    /// * `z` - The measurement vector (`num_measurements` × `1`).
    /// * `R` - The process noise / measurement uncertainty (`num_measurements` × `num_measurements`).
    /// * `y` - The innovation (`num_measurements` × `1`).
    /// * `S` - The residual covariance (`num_measurements` × `num_measurements`).
    /// * `K` - The Kalman gain (`num_states` × `num_measurements`).
    /// * `S_inv` - The temporary matrix for S-inverted (`num_measurements` × `num_measurements`).
    /// * `temp_HP` - The temporary matrix for H×P calculation (`num_measurements` × `num_states`).
    /// * `temp_PHt` - The temporary matrix for P×H' calculation (`num_states` × `num_measurements`).
    /// * `temp_KHP` - The temporary matrix for K×H×P calculation (`num_states` × `num_states`).
    ///
    /// ## Example
    /// See [`Measurement::new_direct`] for an example.
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc(alias = "kalman_measurement_initialize")]
    pub fn new(
        H: Matrix<'a, MEASUREMENTS, STATES, T>,
        z: Matrix<'a, MEASUREMENTS, 1, T>,
        R: Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>,
        y: Matrix<'a, MEASUREMENTS, 1, T>,
        S: Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>,
        K: Matrix<'a, STATES, MEASUREMENTS, T>,
        S_inv: Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>,
        temp_HP: Matrix<'a, MEASUREMENTS, STATES, T>,
        temp_PHt: Matrix<'a, STATES, MEASUREMENTS, T>,
        temp_KHP: Matrix<'a, STATES, STATES, T>,
    ) -> Self
    where
        T: Float,
    {
        debug_assert_eq!(
            H.rows(), MEASUREMENTS as FastUInt8,
            "The measurement transformation matrix H requires {} rows and {} columns (i.e. measurements × states)",
            MEASUREMENTS, STATES
        );
        debug_assert_eq!(
            H.cols(), STATES as FastUInt8,
            "The measurement transformation matrix H requires {} rows and {} columns (i.e. measurements × states)",
            MEASUREMENTS, STATES
        );

        debug_assert_eq!(
            z.rows(),
            MEASUREMENTS as FastUInt8,
            "The measurement vector z requires {} rows and 1 column (i.e. measurements × 1)",
            MEASUREMENTS
        );
        debug_assert_eq!(
            z.cols(),
            1,
            "The measurement vector z requires {} rows and 1 column (i.e. measurements × 1)",
            MEASUREMENTS
        );

        debug_assert_eq!(
            R.rows(), MEASUREMENTS as FastUInt8,
            "The process noise / measurement uncertainty matrix R requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, MEASUREMENTS
        );
        debug_assert_eq!(
            R.cols(), MEASUREMENTS as FastUInt8,
            "The process noise / measurement uncertainty matrix R requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, MEASUREMENTS
        );

        debug_assert_eq!(
            y.rows(),
            MEASUREMENTS as FastUInt8,
            "The innovation vector y requires {} rows and 1 column (i.e. measurements × 1)",
            MEASUREMENTS
        );
        debug_assert_eq!(
            y.cols(),
            1,
            "The innovation vector y requires {} rows and 1 column (i.e. measurements × 1)",
            MEASUREMENTS
        );

        debug_assert_eq!(
            S.rows(), MEASUREMENTS as FastUInt8,
            "The residual covariance matrix S requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, MEASUREMENTS
        );
        debug_assert_eq!(
            S.cols(), MEASUREMENTS as FastUInt8,
            "The residual covariance S requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, MEASUREMENTS
        );

        debug_assert_eq!(
            K.rows(),
            STATES as FastUInt8,
            "The Kalman gain matrix S requires {} rows and {} columns (i.e. states × measurements)",
            STATES,
            MEASUREMENTS
        );
        debug_assert_eq!(
            K.cols(),
            MEASUREMENTS as FastUInt8,
            "The Kalman gain matrix K requires {} rows and {} columns (i.e. states × measurements)",
            STATES,
            MEASUREMENTS
        );

        debug_assert_eq!(
            S_inv.rows(), MEASUREMENTS as FastUInt8,
            "The temporary S-inverted matrix requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, MEASUREMENTS
        );
        debug_assert_eq!(
            S_inv.cols(), MEASUREMENTS as FastUInt8,
            "The temporary S-inverted matrix requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, MEASUREMENTS
        );

        debug_assert_eq!(
            temp_HP.rows(), MEASUREMENTS as FastUInt8,
            "The temporary H×P calculation matrix requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, STATES
        );
        debug_assert_eq!(
            temp_HP.cols(), STATES as FastUInt8,
            "The temporary H×P calculation matrix requires {} rows and {} columns (i.e. measurements × measurements)",
            MEASUREMENTS, STATES
        );

        debug_assert_eq!(
            temp_PHt.rows(), STATES as FastUInt8,
            "The temporary P×H' calculation matrix requires {} rows and {} columns (i.e. states × measurements)",
            STATES, MEASUREMENTS
        );
        debug_assert_eq!(
            temp_PHt.cols(), MEASUREMENTS as FastUInt8,
            "The temporary P×H' calculation matrix requires {} rows and {} columns (i.e. states × measurements)",
            STATES, MEASUREMENTS
        );

        debug_assert_eq!(
            temp_KHP.rows(), STATES as FastUInt8,
            "The temporary K×H×P calculation matrix requires {} rows and {} columns (i.e. states × states)",
            STATES, STATES
        );
        debug_assert_eq!(
            temp_KHP.cols(), STATES as FastUInt8,
            "The temporary K×H×P calculation matrix requires {} rows and {} columns (i.e. states × states)",
            STATES, STATES
        );

        Self {
            H,
            R,
            z,
            K,
            S,
            y,
            temporary: MeasurementTemporary {
                S_inv,
                HP: temp_HP,
                PHt: temp_PHt,
                KHP: temp_KHP,
            },
        }
    }

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
    pub fn measurement_vector_ref(&self) -> &Matrix<'_, MEASUREMENTS, 1, T> {
        &self.z
    }

    /// Gets a mutable reference to the measurement vector z.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_vector")]
    pub fn measurement_vector_mut(&'a mut self) -> &'a mut Matrix<'_, MEASUREMENTS, 1, T> {
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
    /// # let mut gravity_x = create_buffer_x!(NUM_STATES);
    /// # let mut gravity_A = create_buffer_A!(NUM_STATES);
    /// # let mut gravity_P = create_buffer_P!(NUM_STATES);
    /// #
    /// # // Input buffers.
    /// # let mut gravity_u = create_buffer_u!(0);
    /// # let mut gravity_B = create_buffer_B!(0, 0);
    /// # let mut gravity_Q = create_buffer_Q!(0);
    /// #
    /// # // Filter temporaries.
    /// # let mut gravity_temp_x = create_buffer_temp_x!(NUM_STATES);
    /// # let mut gravity_temp_P = create_buffer_temp_P!(NUM_STATES);
    /// # let mut gravity_temp_BQ = create_buffer_temp_BQ!(NUM_STATES, NUM_INPUTS);
    /// #
    /// # let mut filter = Kalman::<NUM_STATES, NUM_INPUTS>::new_direct(
    /// #     &mut gravity_A,
    /// #     &mut gravity_x,
    /// #     &mut gravity_B,
    /// #     &mut gravity_u,
    /// #     &mut gravity_P,
    /// #     &mut gravity_Q,
    /// #     &mut gravity_temp_x,
    /// #     &mut gravity_temp_P,
    /// #     &mut gravity_temp_BQ,
    /// #  );
    /// #
    /// # // Measurement buffers.
    /// # let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
    /// # let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
    /// # let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
    /// # let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
    /// # let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);
    /// #
    /// # // Measurement temporaries.
    /// # let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
    /// # let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
    /// # let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
    /// # let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);
    /// #
    /// # let mut measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS>::new_direct(
    /// #     &mut gravity_H,
    /// #     &mut gravity_z,
    /// #     &mut gravity_R,
    /// #     &mut gravity_y,
    /// #     &mut gravity_S,
    /// #     &mut gravity_K,
    /// #     &mut gravity_temp_S_inv,
    /// #     &mut gravity_temp_HP,
    /// #     &mut gravity_temp_PHt,
    /// #     &mut gravity_temp_KHP,
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
        F: FnMut(&mut Matrix<'a, MEASUREMENTS, 1, T>),
    {
        f(&mut self.z)
    }

    /// Gets a reference to the measurement transformation matrix H.
    #[inline(always)]
    pub fn measurement_transformation_ref(&self) -> &Matrix<'_, MEASUREMENTS, STATES, T> {
        &self.H
    }

    /// Gets a mutable reference to the measurement transformation matrix H.
    #[inline(always)]
    #[doc(alias = "kalman_get_measurement_transformation")]
    pub fn measurement_transformation_mut(
        &'a mut self,
    ) -> &'a mut Matrix<'_, MEASUREMENTS, STATES, T> {
        &mut self.H
    }

    /// Applies a function to the measurement transformation matrix H.
    #[inline(always)]
    pub fn measurement_transformation_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a, MEASUREMENTS, STATES, T>),
    {
        f(&mut self.H)
    }

    /// Gets a reference to the process noise matrix R.
    #[inline(always)]
    pub fn process_noise_ref(&self) -> &Matrix<'_, MEASUREMENTS, MEASUREMENTS, T> {
        &self.R
    }

    /// Gets a mutable reference to the process noise matrix R.
    #[inline(always)]
    #[doc(alias = "kalman_get_process_noise")]
    pub fn process_noise_mut(&'a mut self) -> &'a mut Matrix<'_, MEASUREMENTS, MEASUREMENTS, T> {
        &mut self.R
    }

    /// Applies a function to the process noise matrix R.
    #[inline(always)]
    pub fn process_noise_apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Matrix<'a, MEASUREMENTS, MEASUREMENTS, T>),
    {
        f(&mut self.R)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        create_buffer_H, create_buffer_K, create_buffer_R, create_buffer_S, create_buffer_temp_HP,
        create_buffer_temp_KHP, create_buffer_temp_PHt, create_buffer_temp_S_inv, create_buffer_y,
        create_buffer_z,
    };

    #[test]
    #[allow(non_snake_case)]
    fn test() {
        const NUM_STATES: usize = 3;
        const NUM_MEASUREMENTS: usize = 1;

        // Measurement buffers.
        let mut gravity_z = create_buffer_z!(NUM_MEASUREMENTS);
        let mut gravity_H = create_buffer_H!(NUM_MEASUREMENTS, NUM_STATES);
        let mut gravity_R = create_buffer_R!(NUM_MEASUREMENTS);
        let mut gravity_y = create_buffer_y!(NUM_MEASUREMENTS);
        let mut gravity_S = create_buffer_S!(NUM_MEASUREMENTS);
        let mut gravity_K = create_buffer_K!(NUM_STATES, NUM_MEASUREMENTS);

        // Measurement temporaries.
        let mut gravity_temp_S_inv = create_buffer_temp_S_inv!(NUM_MEASUREMENTS);
        let mut gravity_temp_HP = create_buffer_temp_HP!(NUM_MEASUREMENTS, NUM_STATES);
        let mut gravity_temp_PHt = create_buffer_temp_PHt!(NUM_STATES, NUM_MEASUREMENTS);
        let mut gravity_temp_KHP = create_buffer_temp_KHP!(NUM_STATES);

        gravity_z[0] = 1.0;
        gravity_H[0] = 2.0;
        gravity_R[0] = 3.0;
        gravity_y[0] = 4.0;
        gravity_S[0] = 5.0;
        gravity_K[0] = 6.0;

        let measurement = Measurement::<NUM_STATES, NUM_MEASUREMENTS>::new_direct(
            &mut gravity_H,
            &mut gravity_z,
            &mut gravity_R,
            &mut gravity_y,
            &mut gravity_S,
            &mut gravity_K,
            &mut gravity_temp_S_inv,
            &mut gravity_temp_HP,
            &mut gravity_temp_PHt,
            &mut gravity_temp_KHP,
        );

        assert_eq!(measurement.measurement_vector_ref().data[0], 1.0);
        assert_eq!(measurement.measurement_transformation_ref().data[0], 2.0);
        assert_eq!(measurement.process_noise_ref().data[0], 3.0);

        // Legacy
        assert_eq!(
            Measurement::<NUM_STATES, NUM_MEASUREMENTS>::measurements(),
            NUM_MEASUREMENTS as FastUInt8
        );
        assert_eq!(
            Measurement::<NUM_STATES, NUM_MEASUREMENTS>::states(),
            NUM_STATES as FastUInt8
        );
    }
}
