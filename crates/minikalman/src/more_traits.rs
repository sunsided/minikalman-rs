use crate::more_matrix_traits::{Matrix, MatrixMut, SquareMatrix, SquareMatrixMut};

/// State vector.
///
/// Always mutable
pub trait StateVector<const STATES: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, 1, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, 1, T>;
}

/// System matrix.
///
/// Immutable variant. For a mutable variant, see [`SystemMatrixMut`].
pub trait SystemMatrix<const STATES: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, STATES, T>;
}

/// System matrix.
///
/// Mutable variant. For an immutable variant, see [`SystemMatrix`].
pub trait SystemMatrixMut<const STATES: usize, T = f32>: SystemMatrix<STATES, T> {
    fn as_matrix_mut(&self) -> impl MatrixMut<STATES, STATES, T>;
}

/// System covariance matrix.
///
/// Always mutable.
pub trait SystemCovarianceMatrix<const STATES: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, STATES, T>;
    fn as_matrix_mut(&self) -> impl MatrixMut<STATES, STATES, T>;
}

/// Input vector.
///
/// Immutable variant. For a mutable variant, see [`InputVectorMut`].
pub trait InputVector<const INPUTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<INPUTS, 1, T>;
}

/// Input vector.
///
/// Mutable variant. For an immutable variant, see [`InputVector`].
pub trait InputVectorMut<const INPUTS: usize, T = f32>: InputVector<INPUTS, T> {
    fn as_matrix_mut(&self) -> impl MatrixMut<INPUTS, 1, T>;
}

/// Input matrix.
///
/// Immutable variant. For a mutable variant, see [`InputMatrixMut`].
pub trait InputMatrix<const STATES: usize, const INPUTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, INPUTS, T>;
}

/// Input matrix.
///
/// Mutable variant. For an immutable variant, see [`InputMatrix`].
pub trait InputMatrixMut<const STATES: usize, const INPUTS: usize, T = f32>:
    InputMatrix<STATES, INPUTS, T>
{
    fn as_matrix_mut(&self) -> impl MatrixMut<STATES, INPUTS, T>;
}

/// Input covariance matrix.
///
/// Immutable variant. For a mutable variant, see [`InputCovarianceMatrixMut`].
pub trait InputCovarianceMatrix<const INPUTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<INPUTS, INPUTS, T>;
}

/// Input covariance matrix.
///
/// Mutable variant. For an immutable variant, see [`InputCovarianceMatrix`].
pub trait InputCovarianceMatrixMut<const INPUTS: usize, T = f32>:
    InputCovarianceMatrix<INPUTS, T>
{
    fn as_matrix_mut(&self) -> impl MatrixMut<INPUTS, INPUTS, T>;
}

/// x-sized temporary vector.
///
/// Always mutable.
pub trait StatePredictionVector<const STATES: usize, T = f32> {
    fn as_matrix(&mut self) -> impl Matrix<STATES, 1, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, 1, T>;
}

/// P-Sized temporary matrix (number of states × number of states).
///
/// Always mutable.
pub trait TemporaryStateMatrix<const STATES: usize, T = f32> {
    fn as_matrix(&mut self) -> impl Matrix<STATES, STATES, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, STATES, T>;
}

/// B×Q-sized temporary matrix (number of states × number of inputs).
///
/// Always mutable.
pub trait TemporaryBQMatrix<const STATES: usize, const INPUTS: usize, T = f32> {
    fn as_matrix(&mut self) -> impl Matrix<STATES, INPUTS, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, INPUTS, T>;
}

/// Measurement vector.
///
/// Immutable variant. For a mutable variant, see [`MeasurementVectorMut`].
pub trait MeasurementVector<const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, 1, T>;
}

/// Measurement vector.
///
/// Mutable variant. For a immutable variant, see [`MeasurementVector`].
pub trait MeasurementVectorMut<const MEASUREMENTS: usize, T = f32>:
    MeasurementVector<MEASUREMENTS, T>
{
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, 1, T>;
}

/// Measurement vector.
///
/// Immutable variant. For a mutable variant, see [`MeasurementTransformationMatrixMut`].
pub trait MeasurementTransformationMatrix<const MEASUREMENTS: usize, const STATES: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, STATES, T>;
}

/// Measurement vector.
///
/// Mutable variant. For a immutable variant, see [`MeasurementTransformationMatrix`].
pub trait MeasurementTransformationMatrixMut<
    const MEASUREMENTS: usize,
    const STATES: usize,
    T = f32,
>: MeasurementTransformationMatrix<MEASUREMENTS, STATES, T>
{
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, STATES, T>;
}

/// Process noise covariance matrix.
///
/// Always mutable.
pub trait ProcessNoiseCovarianceMatrix<const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, MEASUREMENTS, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;
}

/// Innovation vector.
///
/// Always mutable.
pub trait InnovationVector<const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, 1, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, 1, T>;
}

/// Residual covariance matrix.
///
/// Always mutable.
pub trait ResidualCovarianceMatrix<const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, MEASUREMENTS, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;
}

/// Kalman Gain matrix.
///
/// Always mutable.
pub trait KalmanGainMatrix<const STATES: usize, const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, MEASUREMENTS, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, MEASUREMENTS, T>;
}

/// Temporary residual covariance-inverted matrix.
///
/// Always mutable.
pub trait TemporaryResidualCovarianceInvertedMatrix<const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, MEASUREMENTS, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, MEASUREMENTS, T>;
}

/// Temporary measurement transformation matrix.
///
/// Always mutable.
pub trait TemporaryHPMatrix<const MEASUREMENTS: usize, const STATES: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<MEASUREMENTS, STATES, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<MEASUREMENTS, STATES, T>;
}

/// Temporary system covariance matrix.
///
/// Always mutable.
pub trait TemporaryKHPMatrix<const STATES: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, STATES, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, STATES, T>;
}

/// P×H'-Sized (H'-Sized) temporary matrix  (number of states × number of measurements).
///
/// Always mutable.
pub trait TemporaryPHTMatrix<const STATES: usize, const MEASUREMENTS: usize, T = f32> {
    fn as_matrix(&self) -> impl Matrix<STATES, MEASUREMENTS, T>;
    fn as_matrix_mut(&mut self) -> impl MatrixMut<STATES, MEASUREMENTS, T>;
}
