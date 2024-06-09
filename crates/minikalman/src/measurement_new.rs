use core::marker::PhantomData;

use crate::more_traits::{
    InnovationVector, InputCovarianceMatrix, InputCovarianceMatrixMut, InputMatrix, InputMatrixMut,
    InputVector, InputVectorMut, KalmanGainMatrix, MeasurementTransformationMatrix,
    MeasurementTransformationMatrixMut, MeasurementVector, ProcessNoiseCovarianceMatrix,
    ResidualCovarianceMatrix, StatePredictionVector, StateVector, SystemCovarianceMatrix,
    SystemMatrix, SystemMatrixMut, TemporaryBQMatrix, TemporaryHPMatrix, TemporaryKHPMatrix,
    TemporaryPHTMatrix, TemporaryResidualCovarianceInvertedMatrix, TemporaryStateMatrix,
};
use crate::{FastUInt8, MatrixDataType};

/// A builder for a Kalman filter measurements.
pub struct MeasurementBuilder<Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt> {
    _phantom: (
        PhantomData<Z>,
        PhantomData<H>,
        PhantomData<R>,
        PhantomData<Y>,
        PhantomData<S>,
        PhantomData<K>,
        PhantomData<TempSInv>,
        PhantomData<TempHP>,
        PhantomData<TempKHP>,
        PhantomData<TempPHt>,
    ),
}

impl<Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt>
    MeasurementBuilder<Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt>
{
    #[allow(non_snake_case, unused)]
    pub fn new<const STATES: usize, const MEASUREMENTS: usize, T>(
        z: Z,
        H: H,
        R: R,
        y: Y,
        S: S,
        K: K,
        temp_S_inv: TempSInv,
        temp_HP: TempHP,
        temp_KHP: TempKHP,
        temp_PHt: TempPHt,
    ) -> Measurement<STATES, MEASUREMENTS, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt>
    where
        T: MatrixDataType,
        Z: MeasurementVector<MEASUREMENTS, T>,
        H: MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>,
        R: ProcessNoiseCovarianceMatrix<MEASUREMENTS, T>,
        Y: InnovationVector<MEASUREMENTS, T>,
        S: ResidualCovarianceMatrix<MEASUREMENTS, T>,
        K: KalmanGainMatrix<STATES, MEASUREMENTS, T>,
        TempSInv: TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>,
        TempHP: TemporaryHPMatrix<MEASUREMENTS, STATES, T>,
        TempKHP: TemporaryKHPMatrix<STATES, T>,
        TempPHt: TemporaryPHTMatrix<STATES, MEASUREMENTS, T>,
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
            temp_KHP,
            temp_PHt,
            _phantom: PhantomData::default(),
        }
    }
}

/// Kalman Filter structure.
#[allow(non_snake_case, unused)]
pub struct Measurement<
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

    /// P-Sized temporary matrix  (number of states × number of states).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_S_inv`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_PHt`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`tmp_HP`].
    pub(crate) temp_KHP: TempKHP,

    /// P×H'-Sized (H'-Sized) temporary matrix  (number of states × number of measurements).
    ///
    /// - The backing field for this temporary MAY be aliased with temporary [`tmp_HP`].
    /// - The backing field for this temporary MAY be aliased with temporary [`temp_KHP`].
    /// - The backing field for this temporary MUST NOT be aliased with temporary [`temp_S_inv`].
    pub(crate) temp_PHt: TempPHt,

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
    > Measurement<STATES, MEASUREMENTS, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt>
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
        TempKHP,
        TempPHt,
    > Measurement<STATES, MEASUREMENTS, T, Z, H, R, Y, S, K, TempSInv, TempHP, TempKHP, TempPHt>
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
    use crate::more_matrix_traits::{Matrix, MatrixMut, SquareMatrix, SquareMatrixMut};
    use crate::more_traits::{MeasurementTransformationMatrixMut, MeasurementVectorMut};

    use super::*;

    #[test]
    fn builder_simple() {
        let filter = MeasurementBuilder::new::<3, 1, f32>(
            Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy, Dummy,
        );
    }

    struct Dummy;
    struct DummyMatrix;

    impl<const STATES: usize, T> MeasurementVector<STATES, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<STATES, 1, T> {
            DummyMatrix
        }
    }

    impl<const STATES: usize, T> MeasurementVectorMut<STATES, T> for Dummy {
        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, 1, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        MeasurementTransformationMatrix<MEASUREMENTS, STATES, T> for Dummy
    {
        fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, STATES, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        MeasurementTransformationMatrixMut<MEASUREMENTS, STATES, T> for Dummy
    {
        fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, STATES, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, T> ProcessNoiseCovarianceMatrix<MEASUREMENTS, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, MEASUREMENTS, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, MEASUREMENTS, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, T> InnovationVector<MEASUREMENTS, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, 1, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, 1, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, T> ResidualCovarianceMatrix<MEASUREMENTS, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, MEASUREMENTS, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, MEASUREMENTS, T> {
            DummyMatrix
        }
    }

    impl<const STATES: usize, const MEASUREMENTS: usize, T>
        KalmanGainMatrix<STATES, MEASUREMENTS, T> for Dummy
    {
        fn as_matrix(&self) -> impl Matrix<STATES, MEASUREMENTS, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, MEASUREMENTS, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, T> TemporaryResidualCovarianceInvertedMatrix<MEASUREMENTS, T>
        for Dummy
    {
        fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, MEASUREMENTS, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, MEASUREMENTS, T> {
            DummyMatrix
        }
    }

    impl<const MEASUREMENTS: usize, const STATES: usize, T>
        TemporaryHPMatrix<MEASUREMENTS, STATES, T> for Dummy
    {
        fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, STATES, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, STATES, T> {
            DummyMatrix
        }
    }

    impl<const STATES: usize, T> TemporaryKHPMatrix<STATES, T> for Dummy {
        fn as_matrix(&self) -> impl Matrix<STATES, STATES, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, STATES, T> {
            DummyMatrix
        }
    }

    impl<const STATES: usize, const MEASUREMENTS: usize, T>
        TemporaryPHTMatrix<STATES, MEASUREMENTS, T> for Dummy
    {
        fn as_matrix(&self) -> impl Matrix<STATES, MEASUREMENTS, T> {
            DummyMatrix
        }

        fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, MEASUREMENTS, T> {
            DummyMatrix
        }
    }

    impl<T> AsRef<[T]> for DummyMatrix {
        fn as_ref(&self) -> &[T] {
            todo!()
        }
    }

    impl<T> AsMut<[T]> for DummyMatrix {
        fn as_mut(&mut self) -> &mut [T] {
            todo!()
        }
    }

    impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix {}
    impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T> for DummyMatrix {}
}
