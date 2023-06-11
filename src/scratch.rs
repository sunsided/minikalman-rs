use crate::{matrix_data_t, Matrix};
use alloc::boxed::Box;
use core::cell::RefCell;
use stdint::{uint_fast16_t, uint_fast8_t};

pub struct ScratchBuffer<'a> {
    len: usize,
    buffer: RefCell<&'a mut [matrix_data_t]>,
}

impl<'a> ScratchBuffer<'a> {
    #[inline(always)]
    pub fn new<const LENGTH: usize>(buffer: &'a mut [matrix_data_t]) -> Self {
        Self {
            len: LENGTH,
            buffer: RefCell::new(buffer),
        }
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub const fn fits(&self, rows: uint_fast8_t, cols: uint_fast8_t) -> bool {
        self.len >= (rows as uint_fast16_t * cols as uint_fast16_t) as _
    }

    #[inline(always)]
    pub fn borrow_predicted_x<F>(&self, num_states: uint_fast8_t, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        let borrow = self.buffer.borrow_mut();
        let mut matrix = Matrix::new(num_states, 1, *borrow);
        f(&mut matrix)
    }

    #[inline(always)]
    #[allow(non_snake_case)]
    pub fn borrow_P<F>(&mut self, num_states: uint_fast8_t, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        let mut matrix = Matrix::new(num_states, num_states, self.buffer.get_mut());
        f(&mut matrix)
    }

    #[inline(always)]
    #[allow(non_snake_case)]
    pub fn borrow_BQ<F>(&mut self, num_states: uint_fast8_t, num_inputs: uint_fast8_t, mut f: F)
    where
        F: FnMut(&mut Matrix<'a>) -> (),
    {
        let mut matrix = Matrix::new(num_states, num_inputs, self.buffer.get_mut());
        f(&mut matrix)
    }
}
