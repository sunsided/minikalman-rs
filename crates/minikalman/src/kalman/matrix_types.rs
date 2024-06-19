use crate::matrix::{AsMatrix, Matrix, MatrixMut};
use crate::prelude::AsMatrixMut;
use core::ops::{Index, IndexMut};

/// State vector. Represents the internal state of the system.
///
/// Immutable variant. For a mutable variant, see [`StateVectorMut`].
#[doc(alias = "Zustandsvektor")]
pub trait StateVector<const STATES: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T> + AsMatrix<STATES, 1, T>
{
}

/// State vector. Represents the internal state of the system.
///
/// Mutable variant. For an immutable variant, see [`StateVector`].
#[doc(alias = "Zustandsvektor")]
pub trait StateVectorMut<const STATES: usize, T = f32>:
    StateVector<STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T> + AsMatrixMut<STATES, 1, T>
{
}

/// State transition (system) matrix. Describes how the state evolves from one time step to the
/// next in the absence of control inputs.
///
/// Immutable variant. For a mutable variant, see [`StateTransitionMatrixMut`].
#[doc(alias = "Übergangsmatrix")]
pub trait StateTransitionMatrix<const STATES: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<STATES, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// State transition (system) matrix. Describes how the state evolves from one time step to the
/// next in the absence of control inputs.
///
/// Mutable variant. For an immutable variant, see [`StateTransitionMatrix`].
#[doc(alias = "Übergangsmatrix")]
pub trait StateTransitionMatrixMut<const STATES: usize, T = f32>:
    StateTransitionMatrix<STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<STATES, STATES, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Estimate covariance matrix. Represents the uncertainty in the state estimate.
///
/// Always mutable.
#[doc(alias = "SystemCovarianceMatrix")]
#[doc(alias = "Schätzkovarianzmatrix")]
pub trait EstimateCovarianceMatrix<const STATES: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<STATES, STATES, T>;
    type TargetMut: MatrixMut<STATES, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Control vector. Represents external inputs to the system that affect its state.
///
/// Immutable variant. For a mutable variant, see [`ControlVectorMut`].
#[doc(alias = "InputVector")]
#[doc(alias = "Steuervektor")]
pub trait ControlVector<const CONTROLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T> + AsMatrix<CONTROLS, 1, T>
{
}

/// Control vector. Represents external inputs to the system that affect its state.
///
/// Mutable variant. For an immutable variant, see [`ControlVector`].
#[doc(alias = "InputVectorMut")]
#[doc(alias = "Steuervektor")]
pub trait ControlVectorMut<const CONTROLS: usize, T = f32>:
    ControlVector<CONTROLS, T> + AsMut<[T]> + IndexMut<usize, Output = T> + AsMatrixMut<CONTROLS, 1, T>
{
}

/// Control matrix. Maps the control vector to the state space, influencing the state transition.
///
/// Immutable variant. For a mutable variant, see [`ControlMatrixMut`].
#[doc(alias = "InputTransitionMatrix")]
#[doc(alias = "Steuermatrix")]
pub trait ControlMatrix<const STATES: usize, const CONTROLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<STATES, CONTROLS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Control matrix. Maps the control vector to the state space, influencing the state transition.
///
/// Mutable variant. For an immutable variant, see [`ControlMatrix`].
#[doc(alias = "InputTransitionMatrix")]
#[doc(alias = "Steuermatrix")]
pub trait ControlMatrixMut<const STATES: usize, const CONTROLS: usize, T = f32>:
    ControlMatrix<STATES, CONTROLS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<STATES, CONTROLS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Process noise covariance matrix. Represents the uncertainty in the state transition process.
///
/// Immutable variant. For a mutable variant, see [`ProcessNoiseCovarianceMatrixMut`].
#[doc(alias = "ControlCovarianceMatrix")]
#[doc(alias = "Prozessrauschkovarianzmatrix")]
pub trait ProcessNoiseCovarianceMatrix<const CONTROLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<CONTROLS, CONTROLS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Process noise covariance matrix. Represents the uncertainty in the state transition process.
///
/// Mutable variant. For an immutable variant, see [`ProcessNoiseCovarianceMatrix`].
#[doc(alias = "ControlCovarianceMatrixMut")]
#[doc(alias = "Prozessrauschkovarianzmatrix")]
pub trait ProcessNoiseCovarianceMatrixMut<const CONTROLS: usize, T = f32>:
    ProcessNoiseCovarianceMatrix<CONTROLS, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<CONTROLS, CONTROLS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Predicted state estimate. Represents the predicted state before considering the measurement.
///
/// Always mutable.
pub trait PredictedStateEstimateVector<const STATES: usize, T = f32>:
    AsRef<[T]>
    + AsMut<[T]>
    + Index<usize, Output = T>
    + IndexMut<usize, Output = T>
    + AsMatrixMut<STATES, 1, T>
{
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

/// Measurement vector. Represents the observed measurements from the system.
///
/// Immutable variant. For a mutable variant, see [`MeasurementVectorMut`].
#[doc(alias = "ObservationVector")]
#[doc(alias = "Messvektor")]
pub trait MeasurementVector<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T> + AsMatrix<OBSERVATIONS, 1, T>
{
}

/// Measurement vector. Represents the observed measurements from the system.
///
/// Mutable variant. For an immutable variant, see [`MeasurementVector`].
#[doc(alias = "ObservationVectorMut")]
#[doc(alias = "Messvektor")]
pub trait MeasurementVectorMut<const OBSERVATIONS: usize, T = f32>:
    MeasurementVector<OBSERVATIONS, T>
    + AsMut<[T]>
    + IndexMut<usize, Output = T>
    + AsMatrixMut<OBSERVATIONS, 1, T>
{
}

/// Observation matrix. Maps the state vector into the measurement space.
///
/// Immutable variant. For a mutable variant, see [`ObservationMatrixMut`].
#[doc(alias = "Beobachtungsmatrix")]
pub trait ObservationMatrix<const OBSERVATIONS: usize, const STATES: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, STATES, T>;

    fn as_matrix(&self) -> &Self::Target;
}

/// Observation matrix. Maps the state vector into the measurement space.
///
/// Mutable variant. For an immutable variant, see [`ObservationMatrix`].
#[doc(alias = "Beobachtungsmatrix")]
pub trait ObservationMatrixMut<const OBSERVATIONS: usize, const STATES: usize, T = f32>:
    ObservationMatrix<OBSERVATIONS, STATES, T> + AsMut<[T]> + IndexMut<usize, Output = T>
{
    type TargetMut: MatrixMut<OBSERVATIONS, STATES, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Measurement noise covariance matrix. Represents the uncertainty in the measurements.
///
/// Always mutable.
#[doc(alias = "Messrauschkovarianzmatrix")]
pub trait MeasurementNoiseCovarianceMatrix<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Innovation vector. Represents the difference between the actual and predicted measurements.
///
/// Always mutable.
#[doc(alias = "Innovationsvektor")]
#[doc(alias = "Messabweichung")]
pub trait InnovationVector<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]>
    + AsMut<[T]>
    + Index<usize, Output = T>
    + IndexMut<usize, Output = T>
    + AsMatrix<OBSERVATIONS, 1, T>
    + AsMatrixMut<OBSERVATIONS, 1, T>
{
}

/// Residual covariance matrix. Represents the uncertainty in the innovation.
///
/// Always mutable.
#[doc(alias = "ResidualCovarianceMatrix")]
#[doc(alias = "Innovationskovarianz")]
#[doc(alias = "Residualkovarianz")]
pub trait InnovationCovarianceMatrix<const OBSERVATIONS: usize, T = f32>:
    AsRef<[T]> + AsMut<[T]> + Index<usize, Output = T> + IndexMut<usize, Output = T>
{
    type Target: Matrix<OBSERVATIONS, OBSERVATIONS, T>;
    type TargetMut: MatrixMut<OBSERVATIONS, OBSERVATIONS, T>;

    fn as_matrix(&self) -> &Self::Target;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// Kalman Gain matrix. Determines how much the predictions should be corrected based on the measurements.
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
