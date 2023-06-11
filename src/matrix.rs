#![allow(unused_imports)]

use core::ops::{Index, IndexMut};
use core::ptr::null;
use stdint::{int_fast16_t, int_fast32_t, uint_fast16_t, uint_fast8_t};

// Required for sqrt()
#[cfg(feature = "no_std")]
use micromath::F32Ext;

#[allow(non_camel_case_types)]
pub type matrix_data_t = f32;

/// A matrix wrapping a data buffer.
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
        debug_assert!(
            buffer.len() >= (rows * cols) as _,
            "Buffer needs to be large enough to keep at least {} × {} = {} elements",
            rows,
            cols,
            rows * cols
        );
        Self {
            rows,
            cols,
            data: buffer,
        }
    }

    /// Initializes a matrix structure from a pointer to a buffer.
    ///
    /// This method allows aliasing of buffers e.g. for the temporary matrices.
    /// Enabled only on the `unsafe` crate feature.
    ///
    /// ## Arguments
    /// * `mat` - The matrix to initialize
    /// * `rows` - The number of rows
    /// * `cols` - The number of columns
    /// * `buffer` - The data buffer (of size `rows` x `cols`).
    #[cfg_attr(docsrs, doc(cfg(feature = "unsafe")))]
    #[cfg(feature = "unsafe")]
    pub unsafe fn new_unchecked(
        rows: uint_fast8_t,
        cols: uint_fast8_t,
        ptr: *mut [matrix_data_t],
    ) -> Self {
        let buffer = unsafe { &mut *ptr };
        if ptr.is_null() {
            debug_assert_eq!(rows, 0, "For null buffers, the row count must be zero");
            debug_assert_eq!(cols, 0, "For null buffers, the column count must be zero");
            return Self {
                rows,
                cols,
                data: buffer,
            };
        }

        debug_assert!(
            buffer.len() >= (rows * cols) as _,
            "Buffer needs to be large enough to keep at least {} × {} = {} elements",
            rows,
            cols,
            rows * cols
        );
        Self {
            rows,
            cols,
            data: buffer,
        }
    }

    /// Gets the number of elements of this matrix.
    pub const fn len(&self) -> int_fast16_t {
        self.rows as int_fast16_t * self.cols as int_fast16_t
    }

    /// Determines if this matrix has zero elements.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inverts a square lower triangular matrix. Meant to be used with
    /// [`Matrix::cholesky_decompose_lower`].
    ///
    /// This does not validate that the matrix is indeed of
    /// lower triangular form. Note that this does not calculate the inverse
    /// of the lower triangular matrix itself, but the inverse of the matrix
    /// that was triangularized. In other words, this is not `inv(chol(m))`, but `inv(m)`
    /// with `m` being prepared through `chol(m)`.
    ///
    /// ## Arguments
    /// * `self` - The lower triangular matrix to be inverted.
    /// * `inverse` - The calculated inverse of the lower triangular matrix.
    ///
    /// ## Example
    ///
    /// ```
    /// use minikalman::Matrix;
    ///
    /// // data buffer for the original and decomposed matrix
    /// let mut d = [
    ///     1.0, 0.5, 0.0,
    ///     0.5, 1.0, 0.0,
    ///     0.0, 0.0, 1.0];
    /// let mut m = Matrix::new(3, 3, &mut d);
    ///
    /// // data buffer for the inverted matrix
    /// let mut di = [0.0; 3 * 3];
    /// let mut mi = Matrix::new(3, 3, &mut di);
    ///
    /// // Decompose matrix to lower triangular.
    /// m.cholesky_decompose_lower();
    ///
    /// // Invert matrix using lower triangular.
    /// m.invert_l_cholesky(&mut mi);
    ///
    /// let test = mi.get(1, 1);
    /// assert!(test.is_finite());
    /// assert!(test >= 1.3);
    ///
    /// assert!((di[0] - 1.33333325).abs() < 0.1);
    /// assert!((di[1] - -0.666666627).abs() < 0.1);
    /// assert!((di[2] - -0.0).abs() < 0.01);
    ///
    /// assert!((di[3] - -0.666666627).abs() < 0.1);
    /// assert!((di[4] - 1.33333325).abs() < 0.1);
    /// assert!((di[5] - 0.0).abs() < 0.01);
    ///
    /// assert!((di[6] - 0.0).abs() < 0.001);
    /// assert!((di[7] - 0.0).abs() < 0.001);
    /// assert!((di[8] - 1.0).abs() < 0.001);
    /// ```
    ///
    /// ## Copyright
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_invert_lower")]
    pub fn invert_l_cholesky(&self, inverse: &mut Self) {
        debug_assert_eq!(self.rows, self.cols);

        let n = self.rows;
        let mat = self.data.as_ref(); // t
        let inv = inverse.data.as_mut(); // a

        // Inverts the lower triangular system and saves the result
        // in the upper triangle to minimize cache misses.
        for i in 0..n {
            let el_ii = mat[idx!(i * n + i)];
            let inv_el_ii = 1.0 / el_ii;
            for j in 0..=i {
                let mut sum = 0 as matrix_data_t;
                if i == j {
                    sum = 1.0 as matrix_data_t;
                };

                sum += (j..i)
                    .map(|k| -mat[idx!(i * n + k)] * inv[idx!(j * n + k)])
                    .sum::<matrix_data_t>();

                inv[idx!(j * n + i)] = sum * inv_el_ii;
            }
        }

        // Solve the system and handle the previous solution being in the upper triangle
        // takes advantage of symmetry.
        for i in (0..=(n - 1)).rev() {
            let el_ii = mat[idx!(i * n + i)];
            let inv_el_ii = 1.0 / el_ii;
            for j in 0..=i {
                let mut sum = inv[idx!(j * n + i)];
                if i < j {
                    sum = 0 as matrix_data_t;
                };

                sum += ((i + 1)..n)
                    .map(|k| -mat[idx!(k * n + i)] * inv[idx!(j * n + k)])
                    .sum::<matrix_data_t>();

                let value = sum * inv_el_ii;
                inv[idx!(i * n + j)] = value;
                inv[idx!(j * n + i)] = value;
            }
        }
    }

    /// Performs a matrix multiplication such that `C = A * B`. This method
    /// uses an auxiliary buffer for keeping one row of `B` cached. This might
    /// improve performance on very wide matrices but is generally slower than
    /// [`Matrix::mult`].
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    /// * `aux` -  Auxiliary vector that can hold a column of `b`.
    ///
    /// ## Example
    /// ```
    /// use minikalman::Matrix;
    ///
    /// let mut a_buf = [
    ///      1.0, 2.0, 3.0,
    ///      4.0, 5.0, 6.0];
    /// let a = Matrix::new(2, 3, &mut a_buf);
    ///
    /// let mut b_buf = [
    ///     10.0, 11.0,
    ///     20.0, 21.0,
    ///     30.0, 31.0];
    /// let b = Matrix::new(3, 2, &mut b_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = Matrix::new(2, 2, &mut c_buf);
    ///
    /// let mut aux = [0f32; 3 * 1];
    /// a.mult_buffered(&b, &mut c, &mut aux);
    ///
    /// assert!((c_buf[0] - (1. * 10. + 2. * 20. + 3. * 30.)).abs() < 0.01); // 140
    /// assert!((c_buf[1] - (1. * 11. + 2. * 21. + 3. * 31.)).abs() < 0.01); // 146
    /// assert!((c_buf[2] - (4. * 10. + 5. * 20. + 6. * 30.)).abs() < 0.01); // 320
    /// assert!((c_buf[3] - (4. * 11. + 5. * 21. + 6. * 31.)).abs() < 0.01); // 335
    /// ```
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult_buffered")]
    pub fn mult_buffered(&self, b: &Self, c: &mut Self, baux: &mut [matrix_data_t]) {
        let bcols = b.cols;
        let ccols = c.cols;
        let brows = b.rows;
        let arows = self.rows;

        let adata = self.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, b.rows);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
        debug_assert_eq!(b.cols, c.cols);

        // Test aux dimensions.
        debug_assert_eq!(baux.len(), self.cols as _);
        debug_assert_eq!(baux.len(), b.rows as _);

        for j in (0..bcols).rev() {
            // create a copy of the column in B to avoid cache issues
            b.get_column_copy(j, baux);

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

    /// Performs a matrix multiplication such that `C = A * B`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    /// * `aux` -  Auxiliary vector that can hold a column of `b`.
    ///
    /// ## Example
    /// ```
    /// use minikalman::Matrix;
    ///
    /// let mut a_buf = [
    ///      1.0, 2.0, 3.0,
    ///      4.0, 5.0, 6.0];
    /// let a = Matrix::new(2, 3, &mut a_buf);
    ///
    /// let mut b_buf = [
    ///     10.0, 11.0,
    ///     20.0, 21.0,
    ///     30.0, 31.0];
    /// let b = Matrix::new(3, 2, &mut b_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = Matrix::new(2, 2, &mut c_buf);
    ///
    /// a.mult(&b, &mut c);
    ///
    /// assert!((c_buf[0] - (1. * 10. + 2. * 20. + 3. * 30.)).abs() < 0.01); // 140
    /// assert!((c_buf[1] - (1. * 11. + 2. * 21. + 3. * 31.)).abs() < 0.01); // 146
    /// assert!((c_buf[2] - (4. * 10. + 5. * 20. + 6. * 30.)).abs() < 0.01); // 320
    /// assert!((c_buf[3] - (4. * 11. + 5. * 21. + 6. * 31.)).abs() < 0.01); // 335
    /// ```
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult")]
    pub fn mult(&self, b: &Self, c: &mut Self) {
        let bcols = b.cols;
        let ccols = c.cols;
        let brows = b.rows;
        let arows = self.rows;

        let adata = self.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, b.rows);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
        debug_assert_eq!(b.cols, c.cols);

        for j in (0..bcols).rev() {
            let mut index_a: uint_fast16_t = 0;
            for i in 0..arows {
                let mut total = 0 as matrix_data_t;
                for k in 0..brows {
                    total += adata[idx!(index_a)] * bdata[idx!(k * b.cols + j)];
                    index_a += 1;
                }
                cdata[idx!(i * ccols + j)] = total;
            }
        }
    }

    /// Performs a matrix multiplication such that `C = A * x`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be overwritten)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult_rowvector")]
    pub fn mult_rowvector(&self, x: &Self, c: &mut Self) {
        let arows = self.rows;
        let acols = self.cols;

        let adata = self.data.as_ref();
        let xdata = x.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, x.rows);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
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
    /// * `self` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be added to)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library

    #[doc(alias = "matrix_multadd_rowvector")]
    pub fn multadd_rowvector(&self, x: &Self, c: &mut Self) {
        let arows = self.rows;
        let acols = self.cols;

        let adata = self.data.as_ref();
        let xdata = x.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, x.rows);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
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
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_mult_transb")]
    pub fn mult_transb(&self, b: &Self, c: &mut Self) {
        let bcols = b.cols;
        let brows = b.rows;
        let arows = self.rows;
        let acols = self.cols;

        let adata = self.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, b.cols);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
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
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be added to)
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    #[doc(alias = "matrix_multadd_transb")]
    pub fn multadd_transb(&self, b: &Self, c: &mut Self) {
        let bcols = b.cols;
        let brows = b.rows;
        let arows = self.rows;
        let acols = self.cols;

        let adata = self.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, b.cols);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
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
    pub fn multscale_transb(&self, b: &Self, scale: matrix_data_t, c: &mut Self) {
        let bcols = b.cols;
        let brows = b.rows;
        let arows = self.rows;
        let acols = self.cols;

        let adata = self.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        // test dimensions of a and b
        debug_assert_eq!(self.cols, b.cols);

        // test dimension of c
        debug_assert_eq!(self.rows, c.rows);
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
    /// * `mat` - The matrix to set    /// * `rows` - The row
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
    /// * `self` - The matrix to initialize
    /// * `column` - The column
    /// * `col_data` - Pointer to an array of the correct length to hold a column of matrix `mat`.
    #[doc(alias = "matrix_get_column_copy")]
    pub fn get_column_copy(&self, column: uint_fast8_t, col_data: &mut [matrix_data_t]) {
        // start from the back, so target index is equal to the index of the last row.
        let mut target_index: int_fast16_t = (self.rows - 1) as _;

        // also, the source index is the column..th index
        let stride: int_fast16_t = self.cols as _;
        let mut source_index = (target_index as int_fast16_t) * stride + (column as int_fast16_t);

        let src = self.data.as_ref();

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
    /// * `self` - The matrix to initialize
    /// * `rows` - The row
    /// * `row_data` - Pointer to an array of the correct length to hold a row of matrix `mat`.
    #[doc(alias = "matrix_get_row_copy")]
    pub fn get_row_copy(&self, row: uint_fast8_t, row_data: &mut [matrix_data_t]) {
        let mut target_index: uint_fast16_t = (self.cols - 1) as _;
        let mut source_index: uint_fast16_t =
            (row as uint_fast16_t + 1) * (self.cols - 1) as uint_fast16_t;

        row_data[idx!(target_index)] = self.data[idx!(source_index)];
        while target_index != 0 {
            target_index -= 1;
            source_index -= 1;
            row_data[idx!(target_index)] = self.data[idx!(source_index)];
        }
    }

    /// Copies the matrix from `mat` to `target`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to copy
    /// * `target` - The matrix to copy to
    #[inline]
    #[doc(alias = "matrix_copy")]
    pub fn copy(&self, target: &mut Self) {
        debug_assert_eq!(self.rows, target.rows);
        debug_assert_eq!(self.cols, target.cols);

        let count: uint_fast16_t = (self.cols as uint_fast16_t) * (self.rows as uint_fast16_t);

        let adata = self.data.as_ref();
        let bdata = target.data.as_mut();

        for index in (0..=(count - 1)).rev() {
            bdata[idx!(index)] = adata[idx![index]];
        }
    }

    /// Subtracts two matrices, using `C = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract
    /// * `c` - The output
    #[inline]
    #[doc(alias = "matrix_sub")]
    pub fn sub(&self, b: &Self, c: &mut Self) {
        debug_assert_eq!(self.rows, b.rows);
        debug_assert_eq!(self.cols, b.cols);
        debug_assert_eq!(self.rows, c.rows);
        debug_assert_eq!(self.cols, c.cols);

        let count: uint_fast16_t = (self.cols as uint_fast16_t) * (self.rows as uint_fast16_t);

        let adata = self.data.as_ref();
        let bdata = b.data.as_ref();
        let cdata = c.data.as_mut();

        for index in (0..count).rev() {
            cdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }

    /// Subtracts two matrices in place, using `A = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    #[inline]
    #[doc(alias = "matrix_sub_inplace_b")]
    pub fn sub_inplace_a(&mut self, b: &Self) {
        debug_assert_eq!(self.rows, b.rows);
        debug_assert_eq!(self.cols, b.cols);

        let count: uint_fast16_t = (self.cols as uint_fast16_t) * (self.rows as uint_fast16_t);

        let adata = self.data.as_mut();
        let bdata = b.data.as_ref();

        for index in (0..count).rev() {
            adata[idx!(index)] -= bdata[idx![index]];
        }
    }

    /// Subtracts two matrices in place, using `B = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    #[inline]
    #[doc(alias = "matrix_sub_inplace_b")]
    pub fn sub_inplace_b(&self, b: &mut Self) {
        debug_assert_eq!(self.rows, b.rows);
        debug_assert_eq!(self.cols, b.cols);

        let count: uint_fast16_t = (self.cols as uint_fast16_t) * (self.rows as uint_fast16_t);

        let adata = self.data.as_ref();
        let bdata = b.data.as_mut();

        for index in (0..count).rev() {
            bdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }

    /// Adds two matrices in place, using `A = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add to, also the output.
    /// * `b` - The values to add.
    #[inline]
    #[doc(alias = "matrix_add_inplace_b")]
    pub fn add_inplace_a(&mut self, b: &Self) {
        debug_assert_eq!(self.rows, b.rows);
        debug_assert_eq!(self.cols, b.cols);

        let count: uint_fast16_t = (self.cols as uint_fast16_t) * (self.rows as uint_fast16_t);

        let adata = self.data.as_mut();
        let bdata = b.data.as_ref();

        for index in (0..count).rev() {
            adata[idx!(index)] = adata[idx!(index)] + bdata[idx![index]];
        }
    }

    /// Adds two matrices in place, using `B = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add to
    /// * `b` - The values to add, also the output
    #[inline]
    #[doc(alias = "matrix_add_inplace_b")]
    pub fn add_inplace_b(&self, b: &mut Self) {
        debug_assert_eq!(self.rows, b.rows);
        debug_assert_eq!(self.cols, b.cols);

        let count: uint_fast16_t = (self.cols as uint_fast16_t) * (self.rows as uint_fast16_t);

        let adata = self.data.as_ref();
        let bdata = b.data.as_mut();

        for index in (0..count).rev() {
            bdata[idx!(index)] = adata[idx!(index)] + bdata[idx![index]];
        }
    }

    /// Decomposes a matrix into lower triangular form using Cholesky decomposition.
    ///
    /// ## Arguments
    /// * `mat` - The matrix to decompose in place into a lower triangular matrix.
    ///
    /// ## Returns
    /// `true` in case of success, `false` if the matrix is not positive semi-definite.
    ///
    /// ## Example
    /// ```
    /// use minikalman::Matrix;
    ///
    /// // data buffer for the original and decomposed matrix
    /// let mut d = [
    ///     1.0, 0.5, 0.0,
    ///     0.5, 1.0, 0.0,
    ///     0.0, 0.0, 1.0];
    ///
    /// let mut m = Matrix::new(3, 3, &mut d);
    ///
    /// // Decompose matrix to lower triangular.
    /// m.cholesky_decompose_lower();
    ///
    /// assert!((d[0] - 1.0).abs() < 0.01);
    /// assert!((d[1] - 0.0).abs() < 0.01);
    /// assert!((d[2] - 0.0).abs() < 0.01);
    ///
    /// assert!((d[3] - 0.5).abs() < 0.01);
    /// assert!((d[4] - 0.866025388).abs() < 0.01);
    /// assert!((d[5] - 0.0).abs() < 0.01);
    ///
    /// assert!((d[6] - 0.0).abs() < 0.01);
    /// assert!((d[7] - 0.0).abs() < 0.01);
    /// assert!((d[8] - 1.0).abs() < 0.01);
    /// ```
    ///
    /// Kudos: https://code.google.com/p/efficient-java-matrix-library
    pub fn cholesky_decompose_lower(&mut self) -> bool {
        let n = self.rows;
        let t: &mut [matrix_data_t] = self.data;

        let mut div_el_ii = 0 as matrix_data_t;

        debug_assert_eq!(self.rows, self.cols);
        debug_assert!(self.rows > 0);

        for i in 0..n {
            for j in i..n {
                let mut sum = t[idx!(i * n + j)];

                let mut i_el = i * n;
                let mut j_el = j * n;
                let end = i_el + i;
                while i_el < end {
                    sum -= t[idx!(i_el)] * t[idx!(j_el)];

                    i_el += 1;
                    j_el += 1;
                }

                t[idx!(j * n + i)] = sum * div_el_ii;
                if i == j {
                    // is it positive-definite?
                    if sum <= 0.0 {
                        return false;
                    }

                    let el_ii = sum.sqrt() as matrix_data_t;
                    t[idx!(i * n + i)] = el_ii;
                    div_el_ii = (1.0 as matrix_data_t) / el_ii;
                }
            }
        }

        // zero out the top right corner.
        for i in 0..n {
            for j in (i + 1)..n {
                t[idx!(i * n + j)] = 0.0;
            }
        }

        return true;
    }
}

impl<'a> Index<usize> for Matrix<'a> {
    type Output = matrix_data_t;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a> IndexMut<usize> for Matrix<'a> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a> AsRef<[matrix_data_t]> for Matrix<'a> {
    fn as_ref(&self) -> &[matrix_data_t] {
        &self.data
    }
}

impl<'a> AsMut<[matrix_data_t]> for Matrix<'a> {
    fn as_mut(&mut self) -> &mut [matrix_data_t] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use assert_float_eq::*;

    #[test]
    #[rustfmt::skip]
    fn mult_buffered() {
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
        a.mult_buffered(&b, &mut c, &mut aux);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 1. * 11. + 2. * 21. + 3. * 31.); // 146
        assert_f32_near!(c_buf[2], 4. * 10. + 5. * 20. + 6. * 30.); // 320
        assert_f32_near!(c_buf[3], 4. * 11. + 5. * 21. + 6. * 31.); // 335
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

        a.mult(&b, &mut c);
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
    fn mult_abat_reference() {
        let mut a_buf = [
            1.0, 2.0,  3.0,
            4.0, 5.0,  6.0,
            7.0, 8.0, -9.0];
        let a = Matrix::new(3, 3, &mut a_buf);

        let mut b_buf = [
            -4.0, -1.0,  0.0,
             2.0,  3.0,  4.0,
             5.0,  9.0, -10.0];
        let b = Matrix::new(3, 3, &mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::new(3, 3, &mut c_buf);

        let mut d_buf = [0f32; 3 * 3];
        let mut d = Matrix::new(3, 3, &mut d_buf);

        // Example P = A*P*A'
        a.mult(&b, &mut c); // temp = A*P
        c.mult_transb(&a, &mut d); // P = temp*A'

        assert_f32_near!(d_buf[0], 13.0);
        assert_f32_near!(d_buf[1], 88.0);
        assert_f32_near!(d_buf[2], 559.0);
        assert_f32_near!(d_buf[3], 34.0);
        assert_f32_near!(d_buf[4], 181.0);
        assert_f32_near!(d_buf[5], 1048.0);
        assert_f32_near!(d_buf[6], 181.0);
        assert_f32_near!(d_buf[7], 184.0);
        assert_f32_near!(d_buf[8], -2009.0);
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

    /// Tests matrix inversion using Cholesky decomposition
    #[test]
    #[rustfmt::skip]
    fn choleski_decomposition() {
        // data buffer for the original and decomposed matrix
        let mut d = [
            1.0, 0.5, 0.0,
            0.5, 1.0, 0.0,
            0.0, 0.0, 1.0];

        let mut m = Matrix::new(3, 3, &mut d);

        // Decompose matrix to lower triangular.
        m.cholesky_decompose_lower();

        // >> chol(m, 'lower')
        //
        // ans =
        //
        //     1.0000         0         0
        //     0.5000    0.8660         0
        //          0         0    1.0000

        // When cross-checking with e.g. Octave keep in mind that
        // this is `chol(d, 'lower')` since we have a longer triangular.
        assert_f32_near!(d[0], 1.0);
        assert_f32_near!(d[1], 0.0);
        assert_f32_near!(d[2], 0.0);

        assert_f32_near!(d[3], 0.5);
        assert_f32_near!(d[4], 0.866025388);
        assert_f32_near!(d[5], 0.0);

        assert_f32_near!(d[6], 0.0);
        assert_f32_near!(d[7], 0.0);
        assert_f32_near!(d[8], 1.0);
    }

    /// Tests matrix inversion using Cholesky decomposition
    #[test]
    #[rustfmt::skip]
    fn matrix_inverse() {
        // data buffer for the original and decomposed matrix
        let mut d = [
            1.0, 0.5, 0.0,
            0.5, 1.0, 0.0,
            0.0, 0.0, 1.0];
        let mut m = Matrix::new(3, 3, &mut d);

        // data buffer for the inverted matrix
        let mut di = [0.0; 3 * 3];
        let mut mi = Matrix::new(3, 3, &mut di);

        // Decompose matrix to lower triangular.
        m.cholesky_decompose_lower();

        // >> chol(m, 'lower')
        //
        // ans =
        //
        //     1.0000         0         0
        //     0.5000    0.8660         0
        //          0         0    1.0000

        // Invert matrix using lower triangular.
        m.invert_l_cholesky(&mut mi);

        // >> inv(chol(m, 'lower'))
        //
        // ans =
        //
        //     1.0000         0         0
        //    -0.5774    1.1547         0
        //          0         0    1.0000

        // Expected result:
        // >> inv(m)
        //
        // ans =
        //
        //     1.3333   -0.6667         0
        //    -0.6667    1.3333         0
        //          0         0    1.0000

        let test = mi.get(1, 1);
        assert!(test.is_finite());
        assert!(test >= 1.3);

        assert_f32_near!(di[0], 1.33333325);
        assert_f32_near!(di[1], -0.666666627);
        assert_f32_near!(di[2], -0.0);

        assert_f32_near!(di[3], -0.666666627);
        assert_f32_near!(di[4], 1.33333325);
        assert_f32_near!(di[5], 0.0);

        assert_f32_near!(di[6], 0.0);
        assert_f32_near!(di[7], 0.0);
        assert_f32_near!(di[8], 1.0);
    }
}
