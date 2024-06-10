use crate::prelude::MeasurementVectorMut;
use crate::type_traits::MeasurementVector;
use crate::{IntoInnerData, MatrixData, MatrixDataMut, MatrixDataOwned};
use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use minikalman_traits::{Matrix, MatrixMut};

pub struct MeasurementVectorBuffer<const MEASUREMENTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<MEASUREMENTS, 1, T>;

// -----------------------------------------------------------

impl<'a, const MEASUREMENTS: usize, T> From<&'a mut [T]>
    for MeasurementVectorBuffer<MEASUREMENTS, T, MatrixDataMut<'a, MEASUREMENTS, 1, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(MEASUREMENTS, value.len());
        }
        Self::new(MatrixData::new_mut::<MEASUREMENTS, 1, T>(value))
    }
}

impl<const MEASUREMENTS: usize, T> From<[T; MEASUREMENTS]>
    for MeasurementVectorBuffer<MEASUREMENTS, T, MatrixDataOwned<MEASUREMENTS, 1, MEASUREMENTS, T>>
{
    fn from(value: [T; MEASUREMENTS]) -> Self {
        Self::new(MatrixData::new_owned::<MEASUREMENTS, 1, MEASUREMENTS, T>(
            value,
        ))
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }
}

impl<const MEASUREMENTS: usize, T, M> AsRef<[T]> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const MEASUREMENTS: usize, T, M> Index<usize> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> IndexMut<usize>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<const MEASUREMENTS: usize, T, M> AsMut<[T]> for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const MEASUREMENTS: usize, T, M> Matrix<MEASUREMENTS, 1, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MatrixMut<MEASUREMENTS, 1, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
}

impl<const MEASUREMENTS: usize, T, M> MeasurementVector<MEASUREMENTS, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MEASUREMENTS: usize, T, M> MeasurementVectorMut<MEASUREMENTS, T>
    for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

// -----------------------------------------------------------

impl<const MEASUREMENTS: usize, T, M> IntoInnerData for MeasurementVectorBuffer<MEASUREMENTS, T, M>
where
    M: MatrixMut<MEASUREMENTS, 1, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
