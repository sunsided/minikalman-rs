use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::*;
use crate::matrix::{AsMatrix, AsMatrixMut, Matrix, MatrixMut};
use crate::{Control, ControlBuilder, Kalman, KalmanBuilder, Observation, ObservationBuilder};

pub fn make_dummy_filter() -> Kalman<
    3,
    f32,
    Dummy<f32, 3, 3>,
    Dummy<f32, 3, 1>,
    Dummy<f32, 3, 3>,
    Dummy<f32, 3, 1>,
    Dummy<f32, 3, 3>,
> {
    KalmanBuilder::new::<3, f32>(
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
    )
}

pub fn make_dummy_control(
) -> Control<3, 2, f32, Dummy<f32, 3, 2>, Dummy<f32, 2, 1>, Dummy<f32, 2, 2>, Dummy<f32, 3, 2>> {
    ControlBuilder::new::<3, 2, f32>(
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
    )
}

pub fn make_dummy_observation() -> Observation<
    3,
    1,
    f32,
    Dummy<f32, 1, 3>,
    Dummy<f32, 1, 1>,
    Dummy<f32, 1, 1>,
    Dummy<f32, 1, 1>,
    Dummy<f32, 1, 1>,
    Dummy<f32, 3, 1>,
    Dummy<f32, 1, 1>,
    Dummy<f32, 1, 3>,
    Dummy<f32, 3, 1>,
    Dummy<f32, 3, 3>,
> {
    ObservationBuilder::new::<3, 1, f32>(
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
    )
}

/// A dummy buffer type that holds a [`DummyMatrix`]
#[derive(Default)]
pub struct Dummy<T, const R: usize, const C: usize>(pub DummyMatrix<T, R, C>);

/// A dummy matrix that is arbitrarily shaped.
#[derive(Default)]
pub struct DummyMatrix<T, const R: usize, const C: usize>([T; 9], PhantomData<T>);

impl<T, const R: usize, const C: usize> Dummy<T, R, C> {
    pub fn as_matrix(&self) -> &impl Matrix<R, C, T> {
        &self.0
    }

    pub fn as_matrix_mut(&mut self) -> &mut impl MatrixMut<R, C, T> {
        &mut self.0
    }
}

impl<T, const R: usize, const C: usize> AsRef<[T]> for Dummy<T, R, C> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T, const R: usize, const C: usize> AsMut<[T]> for Dummy<T, R, C> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for Dummy<T, R, C> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Dummy<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T, const R: usize, const C: usize> AsRef<[T]> for DummyMatrix<T, R, C> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T, const R: usize, const C: usize> AsMut<[T]> for DummyMatrix<T, R, C> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for DummyMatrix<T, R, C> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for DummyMatrix<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix<T, ROWS, COLS> {}

impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T>
    for DummyMatrix<T, ROWS, COLS>
{
}

impl<const STATES: usize, T> StateVector<STATES, T> for Dummy<T, STATES, 1> {}

impl<const STATES: usize, T> StateVectorMut<STATES, T> for Dummy<T, STATES, 1> {}

impl<const STATES: usize, T> StateTransitionMatrix<STATES, T> for Dummy<T, STATES, STATES> {
    type Target = DummyMatrix<T, STATES, STATES>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T> StateTransitionMatrixMut<STATES, T> for Dummy<T, STATES, STATES> {
    type TargetMut = DummyMatrix<T, STATES, STATES>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> EstimateCovarianceMatrix<STATES, T> for Dummy<T, STATES, STATES> {
    type Target = DummyMatrix<T, STATES, STATES>;
    type TargetMut = DummyMatrix<T, STATES, STATES>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> PredictedStateEstimateVector<STATES, T> for Dummy<T, STATES, 1> {}

impl<const STATES: usize, T> TemporaryStateMatrix<STATES, T> for Dummy<T, STATES, STATES> {
    type Target = DummyMatrix<T, STATES, STATES>;
    type TargetMut = DummyMatrix<T, STATES, STATES>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T> ControlVector<CONTROLS, T> for Dummy<T, CONTROLS, 1> {
    type Target = DummyMatrix<T, CONTROLS, 1>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}
impl<const CONTROLS: usize, T> ControlVectorMut<CONTROLS, T> for Dummy<T, CONTROLS, 1> {
    type TargetMut = DummyMatrix<T, CONTROLS, 1>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}
impl<const STATES: usize, const CONTROLS: usize, T> ControlMatrix<STATES, CONTROLS, T>
    for Dummy<T, STATES, CONTROLS>
{
    type Target = DummyMatrix<T, STATES, CONTROLS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T> ControlMatrixMut<STATES, CONTROLS, T>
    for Dummy<T, STATES, CONTROLS>
{
    type TargetMut = DummyMatrix<T, STATES, CONTROLS>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T> ProcessNoiseCovarianceMatrix<CONTROLS, T>
    for Dummy<T, CONTROLS, CONTROLS>
{
    type Target = DummyMatrix<T, CONTROLS, CONTROLS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T> ProcessNoiseCovarianceMatrixMut<CONTROLS, T>
    for Dummy<T, CONTROLS, CONTROLS>
{
    type TargetMut = DummyMatrix<T, CONTROLS, CONTROLS>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T> TemporaryBQMatrix<STATES, CONTROLS, T>
    for Dummy<T, STATES, CONTROLS>
{
    type Target = DummyMatrix<T, STATES, CONTROLS>;
    type TargetMut = DummyMatrix<T, STATES, CONTROLS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> MeasurementVector<STATES, T> for Dummy<T, STATES, 1> {}

impl<const STATES: usize, T> MeasurementVectorMut<STATES, T> for Dummy<T, STATES, 1> {}

impl<const OBSERVATIONS: usize, const STATES: usize, T> ObservationMatrix<OBSERVATIONS, STATES, T>
    for Dummy<T, OBSERVATIONS, STATES>
{
    type Target = DummyMatrix<T, OBSERVATIONS, STATES>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T>
    ObservationMatrixMut<OBSERVATIONS, STATES, T> for Dummy<T, OBSERVATIONS, STATES>
{
    type TargetMut = DummyMatrix<T, OBSERVATIONS, STATES>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T> MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T>
    for Dummy<T, OBSERVATIONS, OBSERVATIONS>
{
    type Target = DummyMatrix<T, OBSERVATIONS, OBSERVATIONS>;
    type TargetMut = DummyMatrix<T, OBSERVATIONS, OBSERVATIONS>;

    fn as_matrix(&self) -> &Self::TargetMut {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const ROWS: usize, const COLS: usize, T> AsMatrix<ROWS, COLS, T> for Dummy<T, ROWS, COLS> {
    type Target = DummyMatrix<T, ROWS, COLS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const ROWS: usize, const COLS: usize, T> AsMatrixMut<ROWS, COLS, T> for Dummy<T, ROWS, COLS> {
    type TargetMut = DummyMatrix<T, ROWS, COLS>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T> InnovationVector<OBSERVATIONS, T> for Dummy<T, OBSERVATIONS, 1> {}

impl<const OBSERVATIONS: usize, T> InnovationCovarianceMatrix<OBSERVATIONS, T>
    for Dummy<T, OBSERVATIONS, OBSERVATIONS>
{
    type Target = DummyMatrix<T, OBSERVATIONS, OBSERVATIONS>;
    type TargetMut = DummyMatrix<T, OBSERVATIONS, OBSERVATIONS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T> KalmanGainMatrix<STATES, OBSERVATIONS, T>
    for Dummy<T, STATES, OBSERVATIONS>
{
    type Target = DummyMatrix<T, STATES, OBSERVATIONS>;
    type TargetMut = DummyMatrix<T, STATES, OBSERVATIONS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T> TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>
    for Dummy<T, OBSERVATIONS, OBSERVATIONS>
{
    type Target = DummyMatrix<T, OBSERVATIONS, OBSERVATIONS>;
    type TargetMut = DummyMatrix<T, OBSERVATIONS, OBSERVATIONS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T> TemporaryHPMatrix<OBSERVATIONS, STATES, T>
    for Dummy<T, OBSERVATIONS, STATES>
{
    type Target = DummyMatrix<T, OBSERVATIONS, STATES>;
    type TargetMut = DummyMatrix<T, OBSERVATIONS, STATES>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> TemporaryKHPMatrix<STATES, T> for Dummy<T, STATES, STATES> {
    type Target = DummyMatrix<T, STATES, STATES>;
    type TargetMut = DummyMatrix<T, STATES, STATES>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T> TemporaryPHTMatrix<STATES, OBSERVATIONS, T>
    for Dummy<T, STATES, OBSERVATIONS>
{
    type Target = DummyMatrix<T, STATES, OBSERVATIONS>;
    type TargetMut = DummyMatrix<T, STATES, OBSERVATIONS>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}
