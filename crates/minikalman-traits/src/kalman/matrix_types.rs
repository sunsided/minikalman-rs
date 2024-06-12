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

/// Input vector.
///
/// Immutable variant. For a mutable variant, see [`InputVectorMut`].
pub trait InputVector<const INPUTS: usize, T = f32>: AsRef<[T]> + Index<usize, Output = T> {
    type Target: Matrix<INPUTS, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Input vector.
///
/// Mutable variant. For an immutable variant, see [`InputVector`].
pub trait InputVectorMut<const INPUTS: usize, T = f32>:
    InputVector<INPUTS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<INPUTS, 1, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the input vector u.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Input matrix.
///
/// Immutable variant. For a mutable variant, see [`InputMatrixMut`].
pub trait InputMatrix<const STATES: usize, const INPUTS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<STATES, INPUTS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Input matrix.
///
/// Mutable variant. For an immutable variant, see [`InputMatrix`].
pub trait InputMatrixMut<const STATES: usize, const INPUTS: usize, T = f32>:
    InputMatrix<STATES, INPUTS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<STATES, INPUTS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the input transition matrix B.
    #[inline(always)]
    fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::TargetMut),
    {
        f(self.as_matrix_mut())
    }
}

/// Input covariance matrix.
///
/// Immutable variant. For a mutable variant, see [`InputCovarianceMatrixMut`].
pub trait InputCovarianceMatrix<const INPUTS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<INPUTS, INPUTS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Input covariance matrix.
///
/// Mutable variant. For an immutable variant, see [`InputCovarianceMatrix`].
pub trait InputCovarianceMatrixMut<const INPUTS: usize, T = f32>:
    InputCovarianceMatrix<INPUTS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<INPUTS, INPUTS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;

    /// Applies a function to the input covariance matrix Q.
    #[inline(always)]
    #[doc(alias = "kalman_get_input_covariance")]
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

/// B×Q-sized temporary matrix (number of states × number of inputs).
///
/// Always mutable.
pub trait TemporaryBQMatrix<const STATES: usize, const INPUTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, INPUTS, T>;
    type TargetMut: MatrixMut<STATES, INPUTS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Measurement vector.
///
/// Immutable variant. For a mutable variant, see [`MeasurementVectorMut`].
pub trait MeasurementVector<const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Measurement vector.
///
/// Mutable variant. For a immutable variant, see [`MeasurementVector`].
pub trait MeasurementVectorMut<const MEASUREMENTS: usize, T = f32>:
    MeasurementVector<MEASUREMENTS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<MEASUREMENTS, 1, T>;

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
pub trait MeasurementObservationMatrix<const MEASUREMENTS: usize, const STATES: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Measurement transformation matrix.
///
/// Mutable variant. For a immutable variant, see [`MeasurementObservationMatrix`].
pub trait MeasurementObservationMatrixMut<const MEASUREMENTS: usize, const STATES: usize, T = f32>:
    MeasurementObservationMatrix<MEASUREMENTS, STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<MEASUREMENTS, STATES, T>;

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
pub trait MeasurementProcessNoiseCovarianceMatrix<const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, MEASUREMENTS, T>;
    type TargetMut: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;

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
pub trait InnovationVector<const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, 1, T>;
    type TargetMut: MatrixMut<MEASUREMENTS, 1, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Residual covariance matrix.
///
/// Always mutable.
pub trait ResidualCovarianceMatrix<const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, MEASUREMENTS, T>;
    type TargetMut: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Kalman Gain matrix.
///
/// Always mutable.
pub trait KalmanGainMatrix<const STATES: usize, const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, MEASUREMENTS, T>;
    type TargetMut: MatrixMut<STATES, MEASUREMENTS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Temporary residual covariance-inverted matrix.
///
/// Always mutable.
pub trait TemporaryResidualCovarianceInvertedMatrix<const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, MEASUREMENTS, T>;
    type TargetMut: MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Temporary measurement transformation matrix.
///
/// Always mutable.
pub trait TemporaryHPMatrix<const MEASUREMENTS: usize, const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<MEASUREMENTS, STATES, T>;
    type TargetMut: MatrixMut<MEASUREMENTS, STATES, T>;

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
pub trait TemporaryPHTMatrix<const STATES: usize, const MEASUREMENTS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, MEASUREMENTS, T>;
    type TargetMut: MatrixMut<STATES, MEASUREMENTS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}
