use std::ptr::{null, null_mut};
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

    /**
     * \brief Performs a matrix multiplication such that {\ref c} = {\ref a} * {\ref b}
     * \param[in] a Matrix A
     * \param[in] b Matrix B
     * \param[in] c Resulting matrix C (will be overwritten)
     * \param[in] aux Auxiliary vector that can hold a column of {\ref b}
     *
     * Kudos: https://code.google.com/p/efficient-java-matrix-library
     */
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

    /**
     * \brief Performs a matrix multiplication such that {\ref c} = {\ref x} * {\ref b}
     * \param[in] a Matrix A
     * \param[in] x Vector x
     * \param[in] c Resulting vector C (will be overwritten)
     * \param[in] aux Auxiliary vector that can hold a column of {\ref b}
     *
     * Kudos: https://code.google.com/p/efficient-java-matrix-library
     */
    pub fn mult_rowvector(/*const matrix_t *RESTRICT const a, const matrix_t *RESTRICT const x, matrix_t *RESTRICT const c */
    ) {
        todo!()
    }

    /**
     * \brief Performs a matrix multiplication such that {\ref c} = {\ref c} + {\ref x} * {\ref b}
     * \param[in] a Matrix A
     * \param[in] x Vector x
     * \param[in] c Resulting vector C (will be added to)
     * \param[in] aux Auxiliary vector that can hold a column of {\ref b}
     *
     * Kudos: https://code.google.com/p/efficient-java-matrix-library
     */
    pub fn multadd_rowvector(/* const matrix_t *RESTRICT const a, const matrix_t *RESTRICT const x, matrix_t *RESTRICT const c */
    ) {
        todo!()
    }

    /**
     * \brief Performs a matrix multiplication with transposed B such that {\ref c} = {\ref a} * {\ref b'}
     * \param[in] a Matrix A
     * \param[in] b Matrix B
     * \param[in] c Resulting matrix C (will be overwritten)
     *
     * Kudos: https://code.google.com/p/efficient-java-matrix-library
     */
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

    /**
     * \brief Performs a matrix multiplication with transposed B and adds the result to {\ref c} such that {\ref c} = {\ref c} + {\ref a} * {\ref b'}
     * \param[in] a Matrix A
     * \param[in] b Matrix B
     * \param[in] c Resulting matrix C (will be added to)
     *
     * Kudos: https://code.google.com/p/efficient-java-matrix-library
     */
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

    /**
     * \brief Performs a matrix multiplication with transposed B and scales the result such that {\ref c} = {\ref a} * {\ref b'} * {\ref scale}
     * \param[in] a Matrix A
     * \param[in] b Matrix B
     * \param[in] scale Scaling factor
     * \param[in] c Resulting matrix C(will be overwritten)
     *
     * Kudos: https://code.google.com/p/efficient-java-matrix-library
     */
    pub fn multscale_transb(/*const matrix_t *const a, const matrix_t *const b, register const matrix_data_t scale, const matrix_t *RESTRICT c*/
    ) {
        todo!()
    }

    /**
     * \brief Gets a matrix element
     * \param[in] mat The matrix to get from
     * \param[in] rows The row
     * \param[in] cols The column
     * \return The value at the given cell.
     */
    pub fn get(/*const matrix_t *const mat, const register uint_fast8_t row, const register uint_fast8_t column*/
    ) -> matrix_data_t {
        todo!()
        /*
        register uint_fast16_t address = row * mat->cols + column;
        return mat->data[address];
        */
    }

    /**
     * \brief Sets a matrix element
     * \param[in] mat The matrix to set
     * \param[in] rows The row
     * \param[in] cols The column
     * \param[in] value The value to set
     */
    pub fn set(/*matrix_t *mat, const register uint_fast8_t row, const register uint_fast8_t column, const register matrix_data_t value */
    ) {
        todo!()
        /*
        register uint_fast16_t address = row * mat->cols + column;
        mat->data[address] = value;
         */
    }

    /**
     * \brief Sets matrix elements in a symmetric matrix
     * \param[in] mat The matrix to set
     * \param[in] rows The row
     * \param[in] cols The column
     * \param[in] value The value to set
     */
    pub fn set_symmetric(/*matrix_t *mat, const register uint_fast8_t row, const register uint_fast8_t column, const register matrix_data_t value*/
    ) {
        todo!()
        /*
        matrix_set(mat, row, column, value);
        matrix_set(mat, column, row, value);
         */
    }

    /**
     * \brief Gets a pointer to a matrix row
     * \param[in] mat The matrix to get from
     * \param[in] rows The row
     * \param[out] row_data A pointer to the given matrix row
     */
    pub fn get_row_pointer(/*const matrix_t *const mat, const register uint_fast8_t row, matrix_data_t **row_data*/
    ) {
        /*
        register uint_fast16_t address = row * mat->cols;
        *row_data = &mat->data[address];
         */
    }

    /**
     * \brief Gets a copy of a matrix column
     * \param[in] mat The matrix to initialize
     * \param[in] rows The column
     * \param[in] row_data Pointer to an array of the correct length to hold a column of matrix {\ref mat}.
     */
    pub fn get_column_copy(mat: &Self, column: uint_fast8_t, row_data: &mut [matrix_data_t]) {
        // start from the back, so target index is equal to the index of the last row.
        let mut target_index: int_fast16_t = (mat.rows - 1) as _;

        // also, the source index is the column..th index
        let stride: int_fast16_t = mat.cols as _;
        let mut source_index = (target_index as int_fast16_t) * stride + (column as int_fast16_t);

        let src = mat.data.as_ref();

        // fetch data
        row_data[idx!(target_index)] = src[idx!(source_index)];
        while target_index != 0 {
            target_index -= 1;
            source_index -= stride;

            row_data[idx!(target_index)] = src[idx!(source_index)];
        }
    }

    /**
     * \brief Gets a copy of a matrix row
     * \param[in] mat The matrix to initialize
     * \param[in] rows The row
     * \param[in] row_data Pointer to an array of the correct length to hold a row of matrix {\ref mat}.
     */
    pub fn get_row_copy(/*const matrix_t *const mat, const register uint_fast8_t row, register matrix_data_t *const row_data*/
    ) {
        /*
        register uint_fast8_t target_index = mat->cols - 1;
        register int_fast16_t source_index = (row + 1) * mat->cols - 1;

        // fetch data
        row_data[target_index] = mat->data[source_index];
        while (target_index != 0)
        {
        --target_index;
        --source_index;
        row_data[target_index] = mat->data[source_index];
        }
         */
    }

    /**
     * \brief Copies the matrix from {\ref mat} to {\ref target}
     * \param[in] mat The matrix to copy
     * \param[in] target The matrix to copy to
     */
    pub fn copy(/*const matrix_t *const mat, matrix_t *const target*/) {
        todo!()
        /*
        register const uint_fast16_t count = mat->cols * mat->rows;
        register int_fast16_t index = 0;

        const matrix_data_t *RESTRICT const A = mat->data;
        matrix_data_t *RESTRICT const B = target->data;

        // fetch data
        for (index = count - 1; index >= 0; --index)
        {
        B[index] = A[index];
        }
         */
    }

    /**
     * \brief Subtracts two matrices, using {\ref c} = {\ref a} - {\ref b}
     * \param[in] a The matrix to subtract from
     * \param[in] b The values to subtract
     * \param[in] c The output
     */
    pub fn sub(/*const matrix_t *const a, matrix_t *const b, const matrix_t *c*/) {
        todo!()
        /*
        register const uint_fast16_t count = a->cols * a->rows;
        register int_fast16_t index = 0;

        matrix_data_t *RESTRICT const A = a->data;
        matrix_data_t *const B = b->data;
        matrix_data_t *C = c->data;

        // subtract data
        for (index = count - 1; index >= 0; --index)
        {
        C[index] = A[index] - B[index];
        }
         */
    }

    /**
     * \brief Subtracts two matrices in place, using {\ref b} = {\ref a} - {\ref b}
     * \param[in] a The matrix to subtract from
     * \param[in] b The values to subtract, also the output
     */
    pub fn sub_inplace_b(/*const matrix_t *RESTRICT const a, const matrix_t *RESTRICT b*/) {
        /*
        register const uint_fast16_t count = a->cols * a->rows;
        register int_fast16_t index = 0;

        matrix_data_t *RESTRICT const A = a->data;
        matrix_data_t *RESTRICT B = b->data;

        // subtract data
        for (index = count - 1; index >= 0; --index)
        {
        B[index] = A[index] - B[index];
        }
         */
    }

    /**
     * \brief Adds two matrices in place, using {\ref b} = {\ref a} + {\ref b}
     * \param[in] a The matrix to add to, also the output
     * \param[in] b The values to add
     */
    pub fn add_inplace(/*const matrix_t * a, const matrix_t *const b*/) {
        todo!()
        /*
        register const uint_fast16_t count = a->cols * a->rows;
        register int_fast16_t index = 0;

        matrix_data_t *RESTRICT A = a->data;
        matrix_data_t *RESTRICT const B = b->data;

        // subtract data
        for (index = count - 1; index >= 0; --index)
        {
        A[index] += B[index];
        }
         */
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
        assert_f32_near!(c_buf[0], 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c_buf[1], 2000. + 1. * 11. + 2. * 21. + 3. * 31.); // 2146
        assert_f32_near!(c_buf[2], 3000. + 4. * 10. + 5. * 20. + 6. * 30.); // 3320
        assert_f32_near!(c_buf[3], 4000. + 4. * 11. + 5. * 21. + 6. * 31.); // 4335
    }
}