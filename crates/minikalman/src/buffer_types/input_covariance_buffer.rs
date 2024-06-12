use core::marker::PhantomData;
use core::ops::{Index, IndexMut};
use minikalman_traits::kalman::{InputCovarianceMatrix, InputCovarianceMatrixMut};
use minikalman_traits::matrix::{
    IntoInnerData, MatrixData, MatrixDataArray, MatrixDataMut, MatrixDataRef,
};
use minikalman_traits::matrix::{Matrix, MatrixMut};

pub struct InputCovarianceMatrixBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: Matrix<INPUTS, INPUTS, T>;

pub struct InputCovarianceMatrixMutBuffer<const INPUTS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<INPUTS, INPUTS, T>;

// -----------------------------------------------------------

impl<'a, const INPUTS: usize, T> From<&'a [T]>
    for InputCovarianceMatrixBuffer<INPUTS, T, MatrixDataRef<'a, INPUTS, INPUTS, T>>
{
    fn from(value: &'a [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(INPUTS * INPUTS, value.len());
        }
        Self::new(MatrixData::new_ref::<INPUTS, INPUTS, T>(value))
    }
}

impl<'a, const INPUTS: usize, T> From<&'a mut [T]>
    for InputCovarianceMatrixBuffer<INPUTS, T, MatrixDataRef<'a, INPUTS, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(INPUTS * INPUTS, value.len());
        }
        Self::new(MatrixData::new_ref::<INPUTS, INPUTS, T>(value))
    }
}

impl<'a, const INPUTS: usize, T> From<&'a mut [T]>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, MatrixDataMut<'a, INPUTS, INPUTS, T>>
{
    fn from(value: &'a mut [T]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(INPUTS * INPUTS, value.len());
        }
        Self::new(MatrixData::new_mut::<INPUTS, INPUTS, T>(value))
    }
}

impl<const INPUTS: usize, const TOTAL: usize, T> From<[T; TOTAL]>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, MatrixDataArray<INPUTS, INPUTS, TOTAL, T>>
{
    fn from(value: [T; TOTAL]) -> Self {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(INPUTS * INPUTS, TOTAL);
        }
        Self::new(MatrixData::new_array::<INPUTS, INPUTS, TOTAL, T>(value))
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        INPUTS * INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        INPUTS == 0
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, INPUTS, T>
    for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrix<INPUTS, T>
    for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    pub const fn new(matrix: M) -> Self {
        Self(matrix, PhantomData)
    }

    pub const fn len(&self) -> usize {
        INPUTS * INPUTS
    }

    pub const fn is_empty(&self) -> bool {
        INPUTS == 0
    }
}

impl<const INPUTS: usize, T, M> AsRef<[T]> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<const INPUTS: usize, T, M> AsMut<[T]> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<const INPUTS: usize, T, M> Matrix<INPUTS, INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> MatrixMut<INPUTS, INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrix<INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type Target = M;

    fn as_matrix(&self) -> &Self::Target {
        &self.0
    }
}

impl<const INPUTS: usize, T, M> InputCovarianceMatrixMut<INPUTS, T>
    for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type TargetMut = M;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
        &mut self.0
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: Matrix<INPUTS, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> Index<usize> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<const INPUTS: usize, T, M> IndexMut<usize> for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

// -----------------------------------------------------------

impl<const INPUTS: usize, T, M> IntoInnerData for InputCovarianceMatrixBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}

impl<const INPUTS: usize, T, M> IntoInnerData for InputCovarianceMatrixMutBuffer<INPUTS, T, M>
where
    M: MatrixMut<INPUTS, INPUTS, T> + IntoInnerData,
{
    type Target = M::Target;

    fn into_inner(self) -> Self::Target {
        self.0.into_inner()
    }
}
