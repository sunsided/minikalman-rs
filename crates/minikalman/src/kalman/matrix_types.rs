use crate::matrix::{Matrix, MatrixMut};
use core::ops::{Index, IndexMut};

/// State vector.
///
/// Immutable variant. For a mutable variant, see [`StateVectorMut`].
pub trait StateVector<const STATES: usize, T = f32>: AsRef<[T]> + Index<usize, Output = T> {
    type Target: Matrix<STATES, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// State vector.
///
/// Mutable variant. For an immutable variant, see [`StateVector`].
pub trait StateVectorMut<const STATES: usize, T = f32>:
    StateVector<STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<STATES, 1, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the state vector x.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// System matrix.
///
/// Immutable variant. For a mutable variant, see [`SystemMatrixMut`].
pub trait SystemMatrix<const STATES: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<STATES, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// System matrix.
///
/// Mutable variant. For an immutable variant, see [`SystemMatrix`].
pub trait SystemMatrixMut<const STATES: usize, T = f32>:
    SystemMatrix<STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<STATES, STATES, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the state transition matrix A.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// System covariance matrix.
///
/// Always mutable.
pub trait SystemCovarianceMatrix<const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, STATES, T>;
    type TargetMut: MatrixMut<STATES, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the system covariance matrix P.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Control vector.
///
/// Immutable variant. For a mutable variant, see [`ControlVectorMut`].
pub trait ControlVector<const CONTROLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<CONTROLS, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Control vector.
///
/// Mutable variant. For an immutable variant, see [`ControlVector`].
pub trait ControlVectorMut<const CONTROLS: usize, T = f32>:
    ControlVector<CONTROLS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<CONTROLS, 1, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the control vector u.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Control matrix.
///
/// Immutable variant. For a mutable variant, see [`ControlMatrixMut`].
pub trait ControlMatrix<const STATES: usize, const CONTROLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<STATES, CONTROLS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Control matrix.
///
/// Mutable variant. For an immutable variant, see [`ControlMatrix`].
pub trait ControlMatrixMut<const STATES: usize, const CONTROLS: usize, T = f32>:
    ControlMatrix<STATES, CONTROLS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<STATES, CONTROLS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the control transition matrix B.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Control covariance matrix.
///
/// Immutable variant. For a mutable variant, see [`ControlCovarianceMatrixMut`].
pub trait ControlCovarianceMatrix<const CONTROLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<CONTROLS, CONTROLS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Control covariance matrix.
///
/// Mutable variant. For an immutable variant, see [`ControlCovarianceMatrix`].
pub trait ControlCovarianceMatrixMut<const CONTROLS: usize, T = f32>:
    ControlCovarianceMatrix<CONTROLS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<CONTROLS, CONTROLS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the control covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_control_covariance")]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// x-sized temporary vector.
///
/// Always mutable.
pub trait StatePredictionVector<const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, 1, T>;
    type TargetMut: MatrixMut<STATES, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// P-Sized temporary matrix (number of states × number of states).
///
/// Always mutable.
pub trait TemporaryStateMatrix<const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, STATES, T>;
    type TargetMut: MatrixMut<STATES, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// B×Q-sized temporary matrix (number of states × number of controls).
///
/// Always mutable.
pub trait TemporaryBQMatrix<const STATES: usize, const CONTROLS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, CONTROLS, T>;
    type TargetMut: MatrixMut<STATES, CONTROLS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Measurement vector.
///
/// Immutable variant. For a mutable variant, see [`MeasurementVectorMut`].
pub trait MeasurementVector<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Measurement vector.
///
/// Mutable variant. For a immutable variant, see [`MeasurementVector`].
pub trait MeasurementVectorMut<const OBSERVATIONS: usize, T = f32>:
    MeasurementVector<OBSERVATIONS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<OBSERVATIONS, 1, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the measurement vector z.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Measurement transformation matrix.
///
/// Immutable variant. For a mutable variant, see [`MeasurementObservationMatrixMut`].
pub trait MeasurementObservationMatrix<const OBSERVATIONS: usize, const STATES: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Measurement transformation matrix.
///
/// Mutable variant. For a immutable variant, see [`MeasurementObservationMatrix`].
pub trait MeasurementObservationMatrixMut<const OBSERVATIONS: usize, const STATES: usize, T = f32>:
    MeasurementObservationMatrix<OBSERVATIONS, STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<OBSERVATIONS, STATES, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the measurement transformation matrix H.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Measurement process noise covariance matrix.
///
/// Always mutable.
pub trait MeasurementProcessNoiseCovarianceMatrix<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the measurement process noise covariance matrix R.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Innovation vector.
///
/// Always mutable.
pub trait InnovationVector<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, 1, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Residual covariance matrix.
///
/// Always mutable.
pub trait ResidualCovarianceMatrix<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Kalman Gain matrix.
///
/// Always mutable.
pub trait KalmanGainMatrix<const STATES: usize, const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<STATES, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Temporary residual covariance-inverted matrix.
///
/// Always mutable.
pub trait TemporaryResidualCovarianceInvertedMatrix<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Temporary measurement transformation matrix.
///
/// Always mutable.
pub trait TemporaryHPMatrix<const OBSERVATIONS: usize, const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, STATES, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Temporary system covariance matrix.
///
/// Always mutable.
pub trait TemporaryKHPMatrix<const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, STATES, T>;
    type TargetMut: MatrixMut<STATES, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// P×H'-Sized (H'-Sized) temporary matrix  (number of states × number of measurements).
///
/// Always mutable.
pub trait TemporaryPHTMatrix<const STATES: usize, const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<STATES, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}
