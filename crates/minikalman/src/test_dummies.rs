use minikalman_traits::matrix::{Matrix, MatrixMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// A dummy buffer type that holds a [`DummyMatrix`]
#[derive(Default)]
pub struct Dummy<T>(pub DummyMatrix<T>, PhantomData<T>);

/// A dummy matrix that is arbitrarily shaped.
#[derive(Default)]
pub struct DummyMatrix<T>(PhantomData<T>);

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
        unimplemented!("A dummy matrix cannot actually dereference into data")
    }
}

impl<T> AsMut<[T]> for DummyMatrix<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unimplemented!("A dummy matrix cannot actually dereference into data")
    }
}

impl<T> Index<usize> for DummyMatrix<T> {
    type Output = T;

    fn index(&self, _index: usize) -> &Self::Output {
        unimplemented!("A dummy matrix cannot actually index into data")
    }
}

impl<T> IndexMut<usize> for DummyMatrix<T> {
    fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
        unimplemented!("A dummy matrix cannot actually index into data")
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> for DummyMatrix<T> {}

impl<const ROWS: usize, const COLS: usize, T> MatrixMut<ROWS, COLS, T> for DummyMatrix<T> {}
