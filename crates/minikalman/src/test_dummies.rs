use core::marker::PhantomData;
use core::ops::{Index, IndexMut};

use crate::kalman::*;
use crate::matrix::{Matrix, MatrixMut};
use crate::{Control, ControlBuilder, Kalman, KalmanBuilder, Observation, ObservationBuilder};

pub fn make_dummy_filter(
) -> Kalman<3, f32, Dummy<f32>, Dummy<f32>, Dummy<f32>, Dummy<f32>, Dummy<f32>> {
    KalmanBuilder::new::<3, f32>(
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
        Dummy::default(),
    )
}

pub fn make_dummy_control() -> Control<3, 2, f32, Dummy<f32>, Dummy<f32>, Dummy<f32>, Dummy<f32>> {
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
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
    Dummy<f32>,
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
pub struct Dummy<T>(pub DummyMatrix<T>, PhantomData<T>);

/// A dummy matrix that is arbitrarily shaped.
#[derive(Default)]
pub struct DummyMatrix<T>([T; 9], PhantomData<T>);

impl<T> AsRef<[T]> for Dummy<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T> AsMut<[T]> for Dummy<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T> Index<usize> for Dummy<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T> IndexMut<usize> for Dummy<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T> AsRef<[T]> for DummyMatrix<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T> AsMut<[T]> for DummyMatrix<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T> Index<usize> for DummyMatrix<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T> IndexMut<usize> for DummyMatrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix<T> {}

impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T> for DummyMatrix<T> {}

impl<const STATES: usize, T> StateVector<STATES, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T> StateVectorMut<STATES, T> for Dummy<T> {
    type TargetMut = DummyMatrix<T>;
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> StateTransitionMatrix<STATES, T> for Dummy<T> {
    type Target = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, T> StateTransitionMatrixMut<STATES, T> for Dummy<T> {
    type TargetMut = DummyMatrix<T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> EstimateCovarianceMatrix<STATES, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    type TargetMut = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> PredictedStateEstimateVector<STATES, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    type TargetMut = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, T> TemporaryStateMatrix<STATES, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    type TargetMut = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T> ControlVector<CONTROLS, T> for Dummy<T> {
    type Target = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}
impl<const CONTROLS: usize, T> ControlVectorMut<CONTROLS, T> for Dummy<T> {
    type TargetMut = DummyMatrix<T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}
impl<const STATES: usize, const CONTROLS: usize, T> ControlMatrix<STATES, CONTROLS, T>
    for Dummy<T>
{
    type Target = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T> ControlMatrixMut<STATES, CONTROLS, T>
    for Dummy<T>
{
    type TargetMut = DummyMatrix<T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const CONTROLS: usize, T> ProcessNoiseCovarianceMatrix<CONTROLS, T> for Dummy<T> {
    type Target = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const CONTROLS: usize, T> ProcessNoiseCovarianceMatrixMut<CONTROLS, T> for Dummy<T> {
    type TargetMut = DummyMatrix<T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const CONTROLS: usize, T> TemporaryBQMatrix<STATES, CONTROLS, T>
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

impl<const OBSERVATIONS: usize, const STATES: usize, T> ObservationMatrix<OBSERVATIONS, STATES, T>
    for Dummy<T>
{
    type Target = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const OBSERVATIONS: usize, const STATES: usize, T>
    ObservationMatrixMut<OBSERVATIONS, STATES, T> for Dummy<T>
{
    type TargetMut = DummyMatrix<T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T> MeasurementNoiseCovarianceMatrix<OBSERVATIONS, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    type TargetMut = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::TargetMut {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T> InnovationVector<OBSERVATIONS, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    type TargetMut = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const OBSERVATIONS: usize, T> InnovationCovarianceMatrix<OBSERVATIONS, T> for Dummy<T> {
    type Target = DummyMatrix<T>;
    type TargetMut = DummyMatrix<T>;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const STATES: usize, const OBSERVATIONS: usize, T> KalmanGainMatrix<STATES, OBSERVATIONS, T>
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

impl<const OBSERVATIONS: usize, T> TemporaryResidualCovarianceInvertedMatrix<OBSERVATIONS, T>
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

impl<const OBSERVATIONS: usize, const STATES: usize, T> TemporaryHPMatrix<OBSERVATIONS, STATES, T>
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

impl<const STATES: usize, const OBSERVATIONS: usize, T> TemporaryPHTMatrix<STATES, OBSERVATIONS, T>
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
