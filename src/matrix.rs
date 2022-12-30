use stdint::{int_fast16_t, uint_fast16_t, uint_fast8_t};

#[allow(non_camel_case_types)]
type matrix_data_t = f32;

pub struct Matrix<'a> {
    pub rows: uint_fast8_t,
    pub cols: uint_fast8_t,
    pub data: &'a mut [matrix_data_t],
}

/// Replaces `x` with `(x) as usize` to simplify index accesses.
macro_rules! idx {
    ( $x:expr ) => {
        ($x) as usize
    };
}

impl<'a> Matrix<'a> {
    /// Initializes a matrix structure.
    ///
    /// ## Arguments
    /// * `mat` - The matrix to initialize
    /// * `rows` - The number of rows
    /// * `cols` - The number of columns
    /// * `buffer` - The data buffer (of size `rows` x `cols`).
    pub fn new(rows: uint_fast8_t, cols: uint_fast8_t, buffer: &'a mut [matrix_data_t]) -> Self {
        debug_assert_eq!(buffer.len(), (rows * cols) as _);
        Self {
            rows,
            cols,
            data: buffer,
        }
    }

    /// Inverts a lower triangular matrix.
    ///
    /// This does not validate that the matrix is indeed of
    /// lower triangular form.
    ///
    /// ## Arguments
    /// * `lower` - The lower triangular matrix to be inverted.
    /// * `inverse` - The calculated inverse of the lower triangular matrix.
    ///
    /// ## Copyright
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_invert_lower")]
    pub fn invert_lower(&self, inverse: &mut Self) {
        debug_assert_eq!(self.rows, self.cols);

        let n = self.rows;
        let t = self.data.as_ref();
        let a = inverse.data.as_mut();

        for i in 0..n {
            let inv_el_ii = 1.0 / t[idx!(i * n + i)];
            for j in 0..i {
                let sum = (j..i).fold(0.0, |sum, k| sum - t[idx!(i * n + k)] * a[idx!(k * n + j)]);
                a[idx!(i * n + j)] = sum * inv_el_ii;
            }
            a[idx!(i * n + i)] = inv_el_ii;
        }
    }

    /// Performs a matrix multiplication such that `C = A * B`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    /// * `aux` -  Auxiliary vector that can hold a column of `b`.
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult")]
    pub fn mult(a: &Self, b: &Self, c: &mut Self, baux: &mut [matrix_data_t]) {
        let bcols = b.cols;
        let ccols = c.cols;
        let brows = b.rows;
        let arows = a.rows;

        let adata = a.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(a.cols, b.rows);

        // test dimension of c
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(b.cols, c.cols);

        // Test aux dimensions.
        debug_assert_eq!(baux.len(), a.cols as _);
        debug_assert_eq!(baux.len(), b.rows as _);

        for j in (0..bcols).rev() {
            // create a copy of the column in B to avoid cache issues
            Self::get_column_copy(&b, j, baux);

            let mut index_a: uint_fast16_t = 0;
            for i in 0..arows {
                let mut total = 0 as matrix_data_t;
                for k in 0..brows {
                    total += adata[idx!(index_a)] * baux[idx!(k)];
                    index_a += 1;
                }
                cdata[idx!(i * ccols + j)] = total;
            }
        }
    }

    /// Performs a matrix multiplication such that `C = A * x`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be overwritten)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult_rowvector")]
    pub fn mult_rowvector(a: &Self, x: &Self, c: &mut Self) {
        let arows = a.rows;
        let acols = a.cols;

        let adata = a.data.as_ref();
        let xdata = x.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(a.cols, x.rows);

        // test dimension of c
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(c.cols, 1);

        let mut index_a: uint_fast16_t = 0;
        let mut index_c: uint_fast16_t = 0;
        let b0 = xdata[0];

        for _ in 0..arows {
            let mut total = adata[idx!(index_a)] * b0;
            index_a += 1;

            for j in 1..acols {
                total += adata[idx!(index_a)] * xdata[idx!(j)];
                index_a += 1;
            }
            cdata[idx!(index_c)] = total;
            index_c += 1;
        }
    }

    /// Performs a matrix-vector multiplication and addition such that `C = C + A * x`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be added to)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library

    #[doc(alias = "matrix_multadd_rowvector")]
    pub fn multadd_rowvector(a: &Self, x: &Self, c: &mut Self) {
        let arows = a.rows;
        let acols = a.cols;

        let adata = a.data.as_ref();
        let xdata = x.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(a.cols, x.rows);

        // test dimension of c
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(c.cols, 1);

        let mut index_a: uint_fast16_t = 0;
        let mut index_c: uint_fast16_t = 0;
        let b0 = xdata[0];

        for _ in 0..arows {
            let mut total = adata[idx!(index_a)] * b0;
            index_a += 1;

            for j in 1..acols {
                total += adata[idx!(index_a)] * xdata[idx!(j)];
                index_a += 1;
            }
            cdata[idx!(index_c)] += total;
            index_c += 1;
        }
    }

    /// Performs a matrix multiplication with transposed `B` such that `C = A * B'`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult_transb")]
    pub fn mult_transb(a: &Self, b: &Self, c: &mut Self) {
        let bcols = b.cols;
        let brows = b.rows;
        let arows = a.rows;
        let acols = a.cols;

        let adata = a.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(a.cols, b.cols);

        // test dimension of c
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(b.rows, c.cols);

        let mut c_index: uint_fast16_t = 0;
        let mut a_index_start: uint_fast16_t = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols as uint_fast16_t;
            let mut index_b: uint_fast16_t = 0;

            for _ in 0..brows {
                let mut index_a = a_index_start;
                let mut total: matrix_data_t = 0.;
                while index_a < end {
                    total += adata[idx!(index_a)] * bdata[idx!(index_b)];
                    index_a += 1;
                    index_b += 1;
                }
                cdata[idx!(c_index)] = total;
                c_index += 1;
            }
            a_index_start += acols as uint_fast16_t;
        }
    }

    /// Performs a matrix multiplication with transposed `B` and adds the result to
    /// `C` such that `C = C + A * B'`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be added to)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_multadd_transb")]
    pub fn multadd_transb(a: &Self, b: &Self, c: &mut Self) {
        let bcols = b.cols;
        let brows = b.rows;
        let arows = a.rows;
        let acols = a.cols;

        let adata = a.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(a.cols, b.cols);

        // test dimension of c
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(b.rows, c.cols);

        let mut c_index: uint_fast16_t = 0;
        let mut a_index_start: uint_fast16_t = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols as uint_fast16_t;
            let mut index_b: uint_fast16_t = 0;

            for _ in 0..brows {
                let mut index_a = a_index_start;
                let mut total: matrix_data_t = 0.;
                while index_a < end {
                    total += adata[idx!(index_a)] * bdata[idx!(index_b)];
                    index_a += 1;
                    index_b += 1;
                }
                cdata[idx!(c_index)] += total;
                c_index += 1;
            }
            a_index_start += acols as uint_fast16_t;
        }
    }

    /// Performs a matrix multiplication with transposed `B` and scales the result such that
    /// `C = A * B' * scale`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `scale` - Scaling factor
    /// * `c` - Resulting matrix C(will be overwritten)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_multscale_transb")]
    pub fn multscale_transb(a: &Self, b: &Self, scale: matrix_data_t, c: &mut Self) {
        let bcols = b.cols;
        let brows = b.rows;
        let arows = a.rows;
        let acols = a.cols;

        let adata = a.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(a.cols, b.cols);

        // test dimension of c
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(b.rows, c.cols);

        let mut c_index: uint_fast16_t = 0;
        let mut a_index_start: uint_fast16_t = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols as uint_fast16_t;
            let mut index_b: uint_fast16_t = 0;

            for _ in 0..brows {
                let mut index_a = a_index_start;
                let mut total: matrix_data_t = 0.;
                while index_a < end {
                    total += adata[idx!(index_a)] * bdata[idx!(index_b)];
                    index_a += 1;
                    index_b += 1;
                }
                cdata[idx!(c_index)] = total * scale;
                c_index += 1;
            }
            a_index_start += acols as uint_fast16_t;
        }
    }

    /// Gets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// * `cols` - The column
    ///
    /// ## Returns
    /// The value at the given cell.
    #[inline(always)]
    #[doc(alias = "matrix_get")]
    pub fn get(&self, row: uint_fast8_t, column: uint_fast8_t) -> matrix_data_t {
        self.data[idx!(row * self.cols + column)]
    }

    /// Sets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to set
    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    #[inline(always)]
    #[doc(alias = "matrix_set")]
    pub fn set(&mut self, row: uint_fast8_t, column: uint_fast8_t, value: matrix_data_t) {
        self.data[idx!(row * self.cols + column)] = value;
    }

    /// Sets matrix elements in a symmetric matrix
    ///
    /// ## Arguments
    /// * `mat` - The matrix to set
    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    #[inline(always)]
    #[doc(alias = "matrix_set_symmetric")]
    pub fn set_symmetric(&mut self, row: uint_fast8_t, column: uint_fast8_t, value: matrix_data_t) {
        self.set(row, column, value);
        self.set(column, row, value);
    }

    /// Gets a pointer to a matrix row
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// *  `row_data` - A pointer to the given matrix row
    #[doc(alias = "matrix_get_row_pointe")]
    pub fn get_row_pointer<'b>(&'a self, row: uint_fast8_t, row_data: &'b mut &'a [matrix_data_t]) {
        *row_data = &self.data[idx!(row * self.cols)..idx!((row + 1) * self.cols)];
    }

    /// Gets a copy of a matrix column
    ///
    /// ## Arguments
    /// * `mat` - The matrix to initialize
    /// * `column` - The column
    /// * `col_data` - Pointer to an array of the correct length to hold a column of matrix `mat`.
    #[doc(alias = "matrix_get_column_copy")]
    pub fn get_column_copy(mat: &Self, column: uint_fast8_t, col_data: &mut [matrix_data_t]) {
        // start from the back, so target index is equal to the index of the last row.
        let mut target_index: int_fast16_t = (mat.rows - 1) as _;

        // also, the source index is the column..th index
        let stride: int_fast16_t = mat.cols as _;
        let mut source_index = (target_index as int_fast16_t) * stride + (column as int_fast16_t);

        let src = mat.data.as_ref();

        // fetch data
        col_data[idx!(target_index)] = src[idx!(source_index)];
        while target_index != 0 {
            target_index -= 1;
            source_index -= stride;

            col_data[idx!(target_index)] = src[idx!(source_index)];
        }
    }

    /// Gets a copy of a matrix row
    ///
    /// ## Arguments
    /// * `mat` - The matrix to initialize
    /// * `rows` - The row
    /// * `row_data` - Pointer to an array of the correct length to hold a row of matrix `mat`.
    #[doc(alias = "matrix_get_row_copy")]
    pub fn get_row_copy(mat: &Self, row: uint_fast8_t, row_data: &mut [matrix_data_t]) {
        let mut target_index: uint_fast16_t = (mat.cols - 1) as _;
        let mut source_index: uint_fast16_t =
            (row as uint_fast16_t + 1) * (mat.cols - 1) as uint_fast16_t;

        row_data[idx!(target_index)] = mat.data[idx!(source_index)];
        while target_index != 0 {
            target_index -= 1;
            source_index -= 1;
            row_data[idx!(target_index)] = mat.data[idx!(source_index)];
        }
    }

    /// Copies the matrix from `mat` to `target`.
    ///
    /// ## Arguments
    /// * `mat` - The matrix to copy
    /// * `target` - The matrix to copy to
    #[inline]
    #[doc(alias = "matrix_copy")]
    pub fn copy(mat: &Self, target: &mut Self) {
        debug_assert_eq!(mat.rows, target.rows);
        debug_assert_eq!(mat.cols, target.cols);

        let count: uint_fast16_t = (mat.cols as uint_fast16_t) * (mat.rows as uint_fast16_t);

        let adata = mat.data.as_ref();
        let bdata = target.data.as_mut();

        for index in (0..=(count - 1)).rev() {
            bdata[idx!(index)] = adata[idx![index]];
        }
    }

    /// Subtracts two matrices, using `C = A - B`.
    ///
    /// ## Arguments
    /// * `a` - The matrix to subtract from
    /// * `b` - The values to subtract
    /// * `c` - The output
    #[inline]
    #[doc(alias = "matrix_sub")]
    pub fn sub(a: &Self, b: &Self, c: &mut Self) {
        debug_assert_eq!(a.rows, b.rows);
        debug_assert_eq!(a.cols, b.cols);
        debug_assert_eq!(a.rows, c.rows);
        debug_assert_eq!(a.cols, c.cols);

        let count: uint_fast16_t = (a.cols as uint_fast16_t) * (a.rows as uint_fast16_t);

        let adata = a.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        for index in (0..=(count - 1)).rev() {
            cdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }

    /// Subtracts two matrices in place, using `B = A - B`.
    ///
    /// ## Arguments
    /// * `a` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    #[inline]
    #[doc(alias = "matrix_sub_inplace_b")]
    pub fn sub_inplace_b(a: &Self, b: &mut Self) {
        debug_assert_eq!(a.rows, b.rows);
        debug_assert_eq!(a.cols, b.cols);

        let count: uint_fast16_t = (a.cols as uint_fast16_t) * (a.rows as uint_fast16_t);

        let adata = a.data.as_ref();
        let bdata = b.data.as_mut();

        for index in (0..=(count - 1)).rev() {
            bdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }

    /// Adds two matrices in place, using `B = A + B`
    ///
    /// ## Arguments
    /// * `a` - The matrix to add to, also the output
    /// * `b` - The values to add
    #[inline]
    #[doc(alias = "matrix_add_inplace_b")]
    pub fn add_inplace_b(a: &Self, b: &mut Self) {
        debug_assert_eq!(a.rows, b.rows);
        debug_assert_eq!(a.cols, b.cols);

        let count: uint_fast16_t = (a.cols as uint_fast16_t) * (a.rows as uint_fast16_t);

        let adata = a.data.as_ref();
        let bdata = b.data.as_mut();

        for index in (0..=(count - 1)).rev() {
            bdata[idx!(index)] = adata[idx!(index)] + bdata[idx![index]];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use assert_float_eq::*;

    #[test]
    #[rustfmt::skip]
    fn invert_lower() {
        let mut a_buf = [
             1.0,  0.0,  0.0,
            -2.0,  1.0,  0.0,
             3.5, -2.5,  1.0];
        let a = Matrix::new(3, 3, &mut a_buf);

        let mut inv_buf = [0f32; 3 * 3];
        let mut inv = Matrix::new(3, 3, &mut inv_buf);

        a.invert_lower(&mut inv);

        assert_f32_near!(inv_buf[0], 1.0);

        assert_f32_near!(inv_buf[3], 2.0);
        assert_f32_near!(inv_buf[4], 1.0);

        assert_f32_near!(inv_buf[6], 1.5);
        assert_f32_near!(inv_buf[7], 2.5);
        assert_f32_near!(inv_buf[8], 1.0);
    }

    #[test]
    #[rustfmt::skip]
    fn mult() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 11.0,
            20.0, 21.0,
            30.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(3, 2, &mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::new(2, 2, &mut c_buf);

        let mut aux = [0f32; 3 * 1];
        Matrix::mult(&a, &b, &mut c, &mut aux);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 1. * 11. + 2. * 21. + 3. * 31.); // 146
        assert_f32_near!(c_buf[2], 4. * 10. + 5. * 20. + 6. * 30.); // 320
        assert_f32_near!(c_buf[3], 4. * 11. + 5. * 21. + 6. * 31.); // 335
    }

    #[test]
    #[rustfmt::skip]
    fn mult_transb() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(2, 3, &mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::new(2, 2, &mut c_buf);

        Matrix::mult_transb(&a, &b, &mut c);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 1. * 11. + 2. * 21. + 3. * 31.); // 146
        assert_f32_near!(c_buf[2], 4. * 10. + 5. * 20. + 6. * 30.); // 320
        assert_f32_near!(c_buf[3], 4. * 11. + 5. * 21. + 6. * 31.); // 335
    }

    #[test]
    #[rustfmt::skip]
    fn multadd_transb() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(2, 3, &mut b_buf);

        let mut c_buf = [
            1000., 2000.,
            3000., 4000.];
        let mut c = Matrix::new(2, 2, &mut c_buf);

        Matrix::multadd_transb(&a, &b, &mut c);
        assert_f32_near!(c.get(0, 0), 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c.get(0, 1), 2000. + 1. * 11. + 2. * 21. + 3. * 31.); // 2146
        assert_f32_near!(c.get(1, 0), 3000. + 4. * 10. + 5. * 20. + 6. * 30.); // 3320
        assert_f32_near!(c.get(1, 1), 4000. + 4. * 11. + 5. * 21. + 6. * 31.); // 4335
    }

    #[test]
    #[rustfmt::skip]
    fn multscale_transb() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(2, 3, &mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::new(2, 2, &mut c_buf);

        Matrix::multscale_transb(&a, &b, 2.0, &mut c);
        assert_f32_near!(c_buf[0], 2.0 * (1. * 10. + 2. * 20. + 3. * 30.)); // 280
        assert_f32_near!(c_buf[1], 2.0 * (1. * 11. + 2. * 21. + 3. * 31.)); // 292
        assert_f32_near!(c_buf[2], 2.0 * (4. * 10. + 5. * 20. + 6. * 30.)); // 640
        assert_f32_near!(c_buf[3], 2.0 * (4. * 11. + 5. * 21. + 6. * 31.)); // 670
    }

    #[test]
    #[rustfmt::skip]
    fn mult_rowvector() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0,
            20.0,
            30.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(3, 1, &mut b_buf);

        let mut c_buf = [0f32; 2 * 1];
        let mut c = Matrix::new(2, 1, &mut c_buf);

        Matrix::mult_rowvector(&a, &b, &mut c);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 4. * 10. + 5. * 20. + 6. * 30.); // 320
    }

    #[test]
    #[rustfmt::skip]
    fn multadd_rowvector() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0,
            20.0,
            30.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(3, 1, &mut b_buf);

        let mut c_buf = [1000., 2000.];
        let mut c = Matrix::new(2, 1, &mut c_buf);

        Matrix::multadd_rowvector(&a, &b, &mut c);
        assert_f32_near!(c.get(0, 0), 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c.get(1, 0), 2000. + 4. * 10. + 5. * 20. + 6. * 30.); // 2320
    }

    #[test]
    #[rustfmt::skip]
    fn get_row_pointer() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let a = Matrix::new(2, 3, &mut a_buf);

        let mut a_out = [0.0; 3].as_slice();
        a.get_row_pointer(0, &mut a_out);
        assert_eq!(a_out, [1.0, 2.0, 3.0]);

        a.get_row_pointer(1, &mut a_out);
        assert_eq!(a_out, [4.0, 5.0, 6.0]);
    }

    #[test]
    #[rustfmt::skip]
    fn sub() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let b = Matrix::new(2, 3, &mut b_buf);

        let mut c_buf = [0f32; 2 * 3];
        let mut c = Matrix::new(2, 3, &mut c_buf);

        Matrix::sub(&a, &b, &mut c);
        assert_eq!(c_buf, [
            1. - 10., 2. - 20., 3. - 30.,
            4. - 11., 5. - 21., 6. - 31.]);
    }

    #[test]
    #[rustfmt::skip]
    fn sub_inplace() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let mut b = Matrix::new(2, 3, &mut b_buf);

        Matrix::sub_inplace_b(&a, &mut b);
        assert_eq!(b_buf, [
            1. - 10., 2. - 20., 3. - 30.,
            4. - 11., 5. - 21., 6. - 31.]);
    }

    #[test]
    #[rustfmt::skip]
    fn add_inplace() {
        let mut a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = Matrix::new(2, 3, &mut a_buf);
        let mut b = Matrix::new(2, 3, &mut b_buf);

        Matrix::add_inplace_b(&a, &mut b);
        assert_eq!(b_buf, [
            1. + 10., 2. + 20., 3. + 30.,
            4. + 11., 5. + 21., 6. + 31.]);
    }
}
