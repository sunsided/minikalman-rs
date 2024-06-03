#![allow(unused_imports)]

use crate::types::*;
use core::ops::{Index, IndexMut};
use core::ptr::null;

// Required for sqrt()
use crate::{MatrixBase, MatrixOps};
#[cfg(feature = "no_std")]
use micromath::F32Ext;

#[allow(non_camel_case_types)]
pub type matrix_data_t = f32;

/// A matrix wrapping a data buffer.
pub struct Matrix<'a, const ROWS: usize, const COLS: usize> {
    pub data: &'a mut [matrix_data_t],
}

/// Replaces `x` with `(x) as usize` to simplify index accesses.
macro_rules! idx {
    ( $x:expr ) => {
        ($x) as usize
    };
}

impl<'a, const ROWS: usize, const COLS: usize> Matrix<'a, ROWS, COLS> {
    /// Initializes a matrix structure.
    ///
    /// ## Arguments
    /// * `buffer` - The data buffer (of size `rows` x `cols`).
    pub fn new(buffer: &'a mut [matrix_data_t]) -> Self {
        debug_assert!(
            buffer.len() >= (ROWS * COLS) as _,
            "Buffer needs to be large enough to keep at least {} × {} = {} elements",
            ROWS,
            COLS,
            ROWS * COLS
        );
        Self { data: buffer }
    }

    /// Returns the number of rows of this matrix.
    pub const fn rows(&self) -> FastUInt8 {
        ROWS as _
    }

    /// Returns the number of columns of this matrix.
    pub const fn cols(&self) -> FastUInt8 {
        COLS as _
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
    pub unsafe fn new_unchecked(ptr: *mut [matrix_data_t]) -> Self {
        let buffer = unsafe { &mut *ptr };
        if ptr.is_null() {
            debug_assert_eq!(ROWS, 0, "For null buffers, the row count must be zero");
            debug_assert_eq!(COLS, 0, "For null buffers, the column count must be zero");
            return Self { data: buffer };
        }

        debug_assert!(
            buffer.len() >= (ROWS * COLS) as _,
            "Buffer needs to be large enough to keep at least {} × {} = {} elements",
            ROWS,
            COLS,
            ROWS * COLS
        );
        Self { data: buffer }
    }

    /// Gets the number of elements of this matrix.
    pub const fn len(&self) -> FastUInt16 {
        ROWS as FastUInt16 * COLS as FastUInt16
    }

    /// Determines if this matrix has zero elements.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, const N: usize> Matrix<'a, N, N> {
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
    /// let mut m = Matrix::<3, 3>::new(&mut d);
    ///
    /// // data buffer for the inverted matrix
    /// let mut di = [0.0; 3 * 3];
    /// let mut mi = Matrix::<3, 3>::new(&mut di);
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
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_invert_lower")]
    pub fn invert_l_cholesky(&self, inverse: &mut Self) {
        let n = N;
        let mat = &self.data; // t
        let inv = &mut inverse.data; // a

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
}

impl<'a, const ROWS: usize, const COLS: usize> Matrix<'a, ROWS, COLS> {
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
    /// let a = Matrix::<2, 3>::new(&mut a_buf);
    ///
    /// let mut b_buf = [
    ///     10.0, 11.0,
    ///     20.0, 21.0,
    ///     30.0, 31.0];
    /// let b = Matrix::<3, 2>::new(&mut b_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = Matrix::<2, 2>::new(&mut c_buf);
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
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_mult_buffered")]
    pub fn mult_buffered<const U: usize>(
        &self,
        b: &Matrix<'_, COLS, U>,
        c: &mut Matrix<'_, ROWS, U>,
        baux: &mut [matrix_data_t],
    ) {
        let arows = self.rows();
        let brows = b.rows();
        let bcols = b.cols();
        let ccols = c.cols();
        let crows = c.rows();

        let adata = &self.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, brows as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(bcols, ccols as _);

        // Test aux dimensions.
        debug_assert_eq!(baux.len(), COLS as _);
        debug_assert_eq!(baux.len(), brows as _);

        for j in (0..bcols).rev() {
            // create a copy of the column in B to avoid cache issues
            b.get_column_copy(j as _, baux);

            let mut index_a: FastUInt16 = 0;
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
    /// let a = Matrix::<2, 3>::new(&mut a_buf);
    ///
    /// let mut b_buf = [
    ///     10.0, 11.0,
    ///     20.0, 21.0,
    ///     30.0, 31.0];
    /// let b = Matrix::<3, 2>::new(&mut b_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = Matrix::<2, 2>::new(&mut c_buf);
    ///
    /// a.mult(&b, &mut c);
    ///
    /// assert!((c_buf[0] - (1. * 10. + 2. * 20. + 3. * 30.)).abs() < 0.01); // 140
    /// assert!((c_buf[1] - (1. * 11. + 2. * 21. + 3. * 31.)).abs() < 0.01); // 146
    /// assert!((c_buf[2] - (4. * 10. + 5. * 20. + 6. * 30.)).abs() < 0.01); // 320
    /// assert!((c_buf[3] - (4. * 11. + 5. * 21. + 6. * 31.)).abs() < 0.01); // 335
    /// ```
    ///
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_mult")]
    pub fn mult<const U: usize>(&self, b: &Matrix<'_, COLS, U>, c: &mut Matrix<'_, ROWS, U>) {
        let arows = ROWS;
        let bcols = b.cols() as usize;
        let brows = b.rows() as usize;
        let ccols = c.cols() as usize;
        let crows = c.rows() as usize;

        let adata = &self.data;
        let bdata = &b.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, brows as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(bcols, ccols);

        for j in (0..bcols).rev() {
            let mut index_a: FastUInt16 = 0;
            for i in 0..arows {
                let mut total = 0 as matrix_data_t;
                for k in 0..brows {
                    total += adata[idx!(index_a)] * bdata[idx!(k * bcols + j)];
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
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_mult_rowvector")]
    pub fn mult_rowvector(&self, x: &Matrix<'_, COLS, 1>, c: &mut Matrix<'_, ROWS, 1>) {
        let arows = self.rows();
        let acols = self.cols();

        let xrows = x.rows();

        let crows = c.rows();
        let ccols = c.cols();

        let adata = &self.data;
        let xdata = &x.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, xrows as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(ccols, 1);

        let mut index_a: FastUInt16 = 0;
        let b0 = xdata[0];

        for index_c in 0..arows {
            let mut total = adata[idx!(index_a)] * b0;
            index_a += 1;

            for j in 1..acols {
                total += adata[idx!(index_a)] * xdata[idx!(j)];
                index_a += 1;
            }
            cdata[idx!(index_c)] = total;
        }
    }

    /// Performs a matrix-vector multiplication and addition such that `C = C + A * x`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be added to)
    ///
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_multadd_rowvector")]
    pub fn multadd_rowvector(&self, x: &Matrix<'_, COLS, 1>, c: &mut Matrix<'_, ROWS, 1>) {
        let arows = self.rows();
        let acols = self.cols();

        let xrows = x.rows();

        let crows = c.rows();
        let ccols = c.cols();

        let adata = &self.data;
        let xdata = &x.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, xrows as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(ccols, 1);

        let mut index_a: FastUInt16 = 0;
        let b0 = xdata[0];

        for index_c in 0..arows {
            let mut total = adata[idx!(index_a)] * b0;
            index_a += 1;

            for j in 1..acols {
                total += adata[idx!(index_a)] * xdata[idx!(j)];
                index_a += 1;
            }
            cdata[idx!(index_c)] += total;
        }
    }

    /// Performs a matrix multiplication with transposed `B` such that `C = A * B'`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    ///
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_mult_transb")]
    pub fn mult_transb<const U: usize>(
        &self,
        b: &Matrix<'_, U, COLS>,
        c: &mut Matrix<'_, ROWS, U>,
    ) {
        let arows = self.rows();
        let acols = self.cols();
        let bcols = b.cols();
        let brows = b.rows();
        let ccols = c.cols();
        let crows = c.rows();

        let adata = &self.data;
        let bdata = &b.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, bcols as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(b.rows(), ccols);

        let mut c_index: FastUInt16 = 0;
        let mut a_index_start: FastUInt16 = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols as FastUInt16;
            let mut index_b: FastUInt16 = 0;

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
            a_index_start += acols as FastUInt16;
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
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_multadd_transb")]
    pub fn multadd_transb<const U: usize>(
        &self,
        b: &Matrix<'_, U, COLS>,
        c: &mut Matrix<'_, ROWS, U>,
    ) {
        let arows = self.rows();
        let acols = self.cols();
        let bcols = b.cols();
        let brows = b.rows();
        let ccols = c.cols();
        let crows = c.rows();

        let adata = &self.data;
        let bdata = &b.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, bcols as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(brows, ccols);

        let mut c_index: FastUInt16 = 0;
        let mut a_index_start: FastUInt16 = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols as FastUInt16;
            let mut index_b: FastUInt16 = 0;

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
            a_index_start += acols as FastUInt16;
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
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    #[doc(alias = "matrix_multscale_transb")]
    pub fn multscale_transb<const U: usize>(
        &self,
        b: &Matrix<'_, U, COLS>,
        scale: matrix_data_t,
        c: &mut Matrix<'_, ROWS, U>,
    ) {
        let arows = self.rows();
        let acols = self.cols();
        let bcols = b.cols();
        let brows = b.rows();
        let ccols = c.cols();
        let crows = c.rows();

        let adata = &self.data;
        let bdata = &b.data;
        let cdata = &mut c.data;

        // test dimensions of a and b
        debug_assert_eq!(COLS, bcols as _);

        // test dimension of c
        debug_assert_eq!(ROWS, crows as _);
        debug_assert_eq!(brows, ccols);

        let mut c_index: FastUInt16 = 0;
        let mut a_index_start: FastUInt16 = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols as FastUInt16;
            let mut index_b: FastUInt16 = 0;

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
            a_index_start += acols as FastUInt16;
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
    pub fn get(&self, row: FastUInt8, column: FastUInt8) -> matrix_data_t {
        self.data[idx!(row * self.cols() + column)]
    }

    /// Sets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to set    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    #[inline(always)]
    #[doc(alias = "matrix_set")]
    pub fn set(&mut self, row: FastUInt8, column: FastUInt8, value: matrix_data_t) {
        self.data[idx!(row * self.cols() + column)] = value;
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
    pub fn set_symmetric(&mut self, row: FastUInt8, column: FastUInt8, value: matrix_data_t) {
        self.set(row, column, value);
        self.set(column, row, value);
    }

    /// Gets a pointer to a matrix row
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// *  `row_data` - A pointer to the given matrix row
    #[doc(alias = "matrix_get_row_pointer")]
    pub fn get_row_pointer<'b>(&'a self, row: FastUInt8, row_data: &'b mut &'a [matrix_data_t]) {
        *row_data = &self.data[idx!(row * self.cols())..idx!((row + 1) * self.cols())];
    }

    /// Gets a copy of a matrix column
    ///
    /// ## Arguments
    /// * `self` - The matrix to initialize
    /// * `column` - The column
    /// * `col_data` - Pointer to an array of the correct length to hold a column of matrix `mat`.
    #[doc(alias = "matrix_get_column_copy")]
    pub fn get_column_copy(&self, column: FastUInt8, col_data: &mut [matrix_data_t]) {
        // start from the back, so target index is equal to the index of the last row.
        let mut target_index: FastInt16 = (self.rows() - 1) as _;

        // also, the source index is the column..th index
        let stride: FastInt16 = self.cols() as _;
        let mut source_index = (target_index as FastInt16) * stride + (column as FastInt16);

        let src = &self.data;

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
    pub fn get_row_copy(&self, row: FastUInt8, row_data: &mut [matrix_data_t]) {
        let mut target_index: FastUInt16 = (self.cols() - 1) as _;
        let mut source_index: FastUInt16 =
            (row as FastUInt16 + 1) * (self.cols() - 1) as FastUInt16;

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
        debug_assert_eq!(self.rows(), target.rows());
        debug_assert_eq!(self.cols(), target.cols());

        let count = self.len();

        let adata = &self.data;
        let bdata = &mut target.data;

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
        debug_assert_eq!(self.rows(), b.rows());
        debug_assert_eq!(self.cols(), b.cols());
        debug_assert_eq!(self.rows(), c.rows());
        debug_assert_eq!(self.cols(), c.cols());

        let count = self.len();

        let adata = &self.data;
        let bdata = &b.data;
        let cdata = &mut c.data;

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
        debug_assert_eq!(self.rows(), b.rows());
        debug_assert_eq!(self.cols(), b.cols());

        let count = self.len();

        let adata = &mut self.data;
        let bdata = &b.data;

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
        debug_assert_eq!(self.rows(), b.rows());
        debug_assert_eq!(self.cols(), b.cols());

        let count = self.len();

        let adata = &self.data;
        let bdata = &mut b.data;

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
        debug_assert_eq!(self.rows(), b.rows());
        debug_assert_eq!(self.cols(), b.cols());

        let count = self.len();

        let adata = &mut self.data;
        let bdata = &b.data;

        for index in (0..count).rev() {
            adata[idx!(index)] += bdata[idx![index]];
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
        debug_assert_eq!(self.rows(), b.rows());
        debug_assert_eq!(self.cols(), b.cols());

        let count = self.len();

        let adata = &self.data;
        let bdata = &mut b.data;

        for index in (0..count).rev() {
            bdata[idx!(index)] += adata[idx!(index)];
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
    /// let mut m = Matrix::<3, 3>::new(&mut d);
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
    /// Kudos: <https://code.google.com/p/efficient-java-matrix-library>
    pub fn cholesky_decompose_lower(&mut self) -> bool {
        let n = self.rows();
        let t: &mut [matrix_data_t] = self.data;

        let mut div_el_ii = 0 as matrix_data_t;

        debug_assert_eq!(ROWS, COLS);
        debug_assert!(ROWS > 0);

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

        true
    }
}

impl<'a, const R: usize, const C: usize> Index<usize> for Matrix<'a, R, C> {
    type Output = matrix_data_t;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, const R: usize, const C: usize> IndexMut<usize> for Matrix<'a, R, C> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a, const R: usize, const C: usize> AsRef<[matrix_data_t]> for Matrix<'a, R, C> {
    fn as_ref(&self) -> &[matrix_data_t] {
        self.data
    }
}

impl<'a, const R: usize, const C: usize> AsMut<[matrix_data_t]> for Matrix<'a, R, C> {
    fn as_mut(&mut self) -> &mut [matrix_data_t] {
        self.data
    }
}

impl<'a, const R: usize, const C: usize> MatrixBase for Matrix<'a, R, C> {
    fn rows(&self) -> FastUInt8 {
        self.rows()
    }

    fn columns(&self) -> FastUInt8 {
        self.cols()
    }

    fn len(&self) -> FastUInt16 {
        self.len()
    }

    fn data_ref(&self) -> &[matrix_data_t] {
        self.data
    }

    fn data_mut(&mut self) -> &mut [matrix_data_t] {
        self.data
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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<3, 2>::new(&mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::<2, 2>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<3, 2>::new(&mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::<2, 2>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<2, 3>::new(&mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::<2, 2>::new(&mut c_buf);

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
        let a = Matrix::<3, 3>::new(&mut a_buf);

        let mut b_buf = [
            -4.0, -1.0,  0.0,
             2.0,  3.0,  4.0,
             5.0,  9.0, -10.0];
        let b = Matrix::<3, 3>::new(&mut b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = Matrix::<3, 3>::new(&mut c_buf);

        let mut d_buf = [0f32; 3 * 3];
        let mut d = Matrix::<3, 3>::new(&mut d_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<2, 3>::new(&mut b_buf);

        let mut c_buf = [
            1000., 2000.,
            3000., 4000.];
        let mut c = Matrix::<2, 2>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<2, 3>::new(&mut b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = Matrix::<2, 2>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<3, 1>::new(&mut b_buf);

        let mut c_buf = [0f32; 2 * 1];
        let mut c = Matrix::<2, 1>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<3, 1>::new(&mut b_buf);

        let mut c_buf = [1000., 2000.];
        let mut c = Matrix::<2, 1>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let b = Matrix::<2, 3>::new(&mut b_buf);

        let mut c_buf = [0f32; 2 * 3];
        let mut c = Matrix::<2, 3>::new(&mut c_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let mut b = Matrix::<2, 3>::new(&mut b_buf);

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
        let a = Matrix::<2, 3>::new(&mut a_buf);
        let mut b = Matrix::<2, 3>::new(&mut b_buf);

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

        let mut m = Matrix::<3, 3>::new(&mut d);

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
        let mut m = Matrix::<3, 3>::new(&mut d);

        // data buffer for the inverted matrix
        let mut di = [0.0; 3 * 3];
        let mut mi = Matrix::<3, 3>::new(&mut di);

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

    #[test]
    fn default_matrix_is_empty() {
        let mut d = [];
        let m = Matrix::<0, 0>::new(&mut d);
        assert!(m.is_empty());
    }
}
