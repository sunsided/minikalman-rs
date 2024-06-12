use crate::matrix::MatrixDataType;
use core::ops::{Index, IndexMut};
use num_traits::{One, Zero};

/// Replaces `x` with `(x) as usize` to simplify index accesses.
macro_rules! idx {
    ( $x:expr ) => {
        ($x) as usize
    };
}

/// A matrix wrapping a data buffer.
pub trait Matrix<const ROWS: usize, const COLS: usize, T = f32>:
    AsRef<[T]> + Index<usize, Output = T>
{
    /// Returns the number of rows of this matrix.
    fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns of this matrix.
    fn cols(&self) -> usize {
        COLS
    }

    /// Gets the number of elements of this matrix.
    fn len(&self) -> usize {
        ROWS * COLS
    }

    /// Gets the number of elements in the underlying buffer..
    fn buffer_len(&self) -> usize {
        self.as_ref().len()
    }

    /// Determines if this matrix has zero elements.
    fn is_empty(&self) -> bool {
        ROWS * COLS == 0
    }

    /// Ensures the underlying buffer has enough space for the expected number of values.
    fn is_valid(&self) -> bool {
        self.len() <= self.buffer_len()
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
    fn get(&self, row: usize, column: usize) -> T
    where
        T: Copy,
    {
        self.as_ref()[idx!(row * self.cols() + column)]
    }

    /// Gets a pointer to a matrix row
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// *  `row_data` - A pointer to the given matrix row
    #[doc(alias = "matrix_get_row_pointer")]
    fn get_row_pointer<'a>(&'a self, row: usize, row_data: &mut &'a [T]) {
        let data = self.as_ref();
        *row_data = &data[idx!(row * self.cols())..idx!((row + 1) * self.cols())];
    }

    /// Gets a copy of a matrix column
    ///
    /// ## Arguments
    /// * `self` - The matrix to initialize
    /// * `column` - The column
    /// * `col_data` - Pointer to an array of the correct length to hold a column of matrix `mat`.
    #[doc(alias = "matrix_get_column_copy")]
    fn get_column_copy(&self, column: usize, col_data: &mut [T])
    where
        T: Copy,
    {
        // start from the back, so target index is equal to the index of the last row.
        let mut target_index: isize = (self.rows() - 1) as _;

        // also, the source index is the column..th index
        let stride: isize = self.cols() as _;
        let mut source_index = (target_index) * stride + (column as isize);

        let src = self.as_ref();

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
    fn get_row_copy(&self, row: usize, row_data: &mut [T])
    where
        T: Copy,
    {
        let mut target_index: usize = (self.cols() - 1) as _;
        let mut source_index: usize = (row + 1) * (self.cols() - 1);

        let data = self.as_ref();
        row_data[idx!(target_index)] = data[idx!(source_index)];
        while target_index != 0 {
            target_index -= 1;
            source_index -= 1;
            row_data[idx!(target_index)] = data[idx!(source_index)];
        }
    }

    /// Copies the matrix from `mat` to `target`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to copy
    /// * `target` - The matrix to copy to
    fn copy<Target>(&self, target: &mut Target)
    where
        Target: MatrixMut<ROWS, COLS, T>,
        T: Copy,
    {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(self.rows(), target.rows());
            debug_assert_eq!(self.cols(), target.cols());
        }

        let count = self.len();

        let adata = self.as_ref();
        let bdata = target.as_mut();

        for index in (0..=(count - 1)).rev() {
            bdata[idx!(index)] = adata[idx![index]];
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
    /// use minikalman::matrix::{Matrix, MatrixData};
    ///
    /// let a_buf = [
    ///      1.0, 2.0, 3.0,
    ///      4.0, 5.0, 6.0];
    /// let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
    ///
    /// let b_buf = [
    ///     10.0, 11.0,
    ///     20.0, 21.0,
    ///     30.0, 31.0];
    /// let b = MatrixData::new_ref::<3, 2, f32>(&b_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);
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
    fn mult_buffered<const U: usize, B, C>(&self, b: &B, c: &mut C, baux: &mut [T])
    where
        B: Matrix<COLS, U, T>,
        C: MatrixMut<ROWS, U, T>,
        T: MatrixDataType,
    {
        let arows = self.rows();
        let brows = b.rows();
        let bcols = b.cols();
        let ccols = c.cols();

        #[cfg(not(feature = "no_assert"))]
        {
            let crows = c.rows();

            // test dimensions of a and b
            debug_assert_eq!(COLS, brows);

            // test dimension of c
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(bcols, ccols);

            // Test aux dimensions.
            debug_assert_eq!(baux.len(), { COLS });
            debug_assert_eq!(baux.len(), brows);
        }

        let adata = self.as_ref();
        let cdata = c.as_mut();

        for j in (0..bcols).rev() {
            // create a copy of the column in B to avoid cache issues
            b.get_column_copy(j as _, baux);

            let mut index_a: usize = 0;
            for i in 0..arows {
                let mut total = T::zero();
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
    /// use minikalman::matrix::{MatrixData, Matrix};
    ///
    /// let a_buf = [
    ///      1.0, 2.0, 3.0,
    ///      4.0, 5.0, 6.0];
    /// let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
    ///
    /// let b_buf = [
    ///     10.0, 11.0,
    ///     20.0, 21.0,
    ///     30.0, 31.0];
    /// let b = MatrixData::new_ref::<3, 2, f32>(&b_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);
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
    fn mult<const U: usize, B, C>(&self, b: &B, c: &mut C)
    where
        B: Matrix<COLS, U, T>,
        C: MatrixMut<ROWS, U, T>,
        T: MatrixDataType,
    {
        let arows = ROWS;
        let bcols = b.cols();
        let brows = b.rows();
        let ccols = c.cols();

        #[cfg(not(feature = "no_assert"))]
        {
            let crows = c.rows();

            // test dimensions of a and b
            debug_assert_eq!(COLS, brows);

            // test dimension of c
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(bcols, ccols);
        }

        let adata = self.as_ref();
        let bdata = b.as_ref();
        let cdata = c.as_mut();

        for j in (0..bcols).rev() {
            let mut index_a: usize = 0;
            for i in 0..arows {
                let mut total = T::zero();
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
    fn mult_rowvector<X, C>(&self, x: &X, c: &mut C)
    where
        X: Matrix<COLS, 1, T>,
        C: MatrixMut<ROWS, 1, T>,
        T: MatrixDataType,
    {
        let arows = self.rows();
        let acols = self.cols();

        #[cfg(not(feature = "no_assert"))]
        {
            let xrows = x.rows();

            let crows = c.rows();
            let ccols = c.cols();

            // test dimensions of a and b
            debug_assert_eq!(COLS, xrows);

            // test dimension of c
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(ccols, 1);
        }

        let adata = self.as_ref();
        let xdata = x.as_ref();
        let cdata = c.as_mut();

        let mut index_a: usize = 0;
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
    fn multadd_rowvector<X, C>(&self, x: &X, c: &mut C)
    where
        X: Matrix<COLS, 1, T>,
        C: MatrixMut<ROWS, 1, T>,
        T: MatrixDataType,
    {
        let arows = self.rows();
        let acols = self.cols();

        #[cfg(not(feature = "no_assert"))]
        {
            let xrows = x.rows();

            let crows = c.rows();
            let ccols = c.cols();

            // test dimensions of a and b
            debug_assert_eq!(COLS, xrows);

            // test dimension of c
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(ccols, 1);
        }

        let adata = self.as_ref();
        let xdata = x.as_ref();
        let cdata = c.as_mut();

        let mut index_a: usize = 0;
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
    fn mult_transb<const U: usize, B, C>(&self, b: &B, c: &mut C)
    where
        B: Matrix<U, COLS, T>,
        C: MatrixMut<ROWS, U, T>,
        T: MatrixDataType,
    {
        let arows = self.rows();
        let acols = self.cols();
        let bcols = b.cols();
        let brows = b.rows();

        #[cfg(not(feature = "no_assert"))]
        {
            let ccols = c.cols();
            let crows = c.rows();

            // test dimensions of a and b
            debug_assert_eq!(COLS, bcols);

            // test dimension of c
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(b.rows(), ccols);
        }

        let adata = self.as_ref();
        let bdata = b.as_ref();
        let cdata = c.as_mut();

        let mut c_index: usize = 0;
        let mut a_index_start: usize = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols;
            let mut index_b: usize = 0;

            for _ in 0..brows {
                let mut index_a = a_index_start;
                let mut total: T = T::zero();
                while index_a < end {
                    total += adata[idx!(index_a)] * bdata[idx!(index_b)];
                    index_a += 1;
                    index_b += 1;
                }
                cdata[idx!(c_index)] = total;
                c_index += 1;
            }
            a_index_start += acols;
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
    fn multadd_transb<const U: usize, B, C>(&self, b: &B, c: &mut C)
    where
        B: Matrix<U, COLS, T>,
        C: MatrixMut<ROWS, U, T>,
        T: MatrixDataType,
    {
        let arows = self.rows();
        let acols = self.cols();
        let bcols = b.cols();
        let brows = b.rows();

        #[cfg(not(feature = "no_assert"))]
        {
            let ccols = c.cols();
            let crows = c.rows();

            // test dimensions of a and b
            debug_assert_eq!(COLS, bcols);

            // test dimension of c
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(brows, ccols);
        }

        let adata = self.as_ref();
        let bdata = b.as_ref();
        let cdata = c.as_mut();

        let mut c_index: usize = 0;
        let mut a_index_start: usize = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols;
            let mut index_b: usize = 0;

            for _ in 0..brows {
                let mut index_a = a_index_start;
                let mut total: T = T::zero();
                while index_a < end {
                    total += adata[idx!(index_a)] * bdata[idx!(index_b)];
                    index_a += 1;
                    index_b += 1;
                }
                cdata[idx!(c_index)] += total;
                c_index += 1;
            }
            a_index_start += acols;
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
    fn multscale_transb<const U: usize, B, C>(&self, b: &B, scale: T, c: &mut C)
    where
        B: Matrix<U, COLS, T>,
        C: MatrixMut<ROWS, U, T>,
        T: MatrixDataType,
    {
        let arows = self.rows();
        let acols = self.cols();
        let bcols = b.cols();
        let brows = b.rows();

        // test dimension of c
        #[cfg(not(feature = "no_assert"))]
        {
            let ccols = c.cols();
            let crows = c.rows();
            debug_assert_eq!(ROWS, crows);
            debug_assert_eq!(brows, ccols);
        }

        let adata = &self.as_ref();
        let bdata = &b.as_ref();
        let cdata = &mut c.as_mut();

        // test dimensions of a and b
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(COLS, bcols);
        }

        let mut c_index: usize = 0;
        let mut a_index_start: usize = 0;

        for _ in 0..arows {
            let end = a_index_start + bcols;
            let mut index_b: usize = 0;

            for _ in 0..brows {
                let mut index_a = a_index_start;
                let mut total: T = T::zero();
                while index_a < end {
                    total += adata[idx!(index_a)] * bdata[idx!(index_b)];
                    index_a += 1;
                    index_b += 1;
                }
                cdata[idx!(c_index)] = total * scale;
                c_index += 1;
            }
            a_index_start += acols;
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
    fn sub<B, C>(&self, b: &B, c: &mut C)
    where
        B: Matrix<ROWS, COLS, T>,
        C: MatrixMut<ROWS, COLS, T>,
        T: MatrixDataType,
    {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(self.rows(), b.rows());
            debug_assert_eq!(self.cols(), b.cols());
            debug_assert_eq!(self.rows(), c.rows());
            debug_assert_eq!(self.cols(), c.cols());
        }

        let count = self.len();

        let adata = self.as_ref();
        let bdata = b.as_ref();
        let cdata = c.as_mut();

        for index in (0..count).rev() {
            cdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }

    /// Adds two matrices in place, using `B = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add to
    /// * `b` - The values to add, also the output
    #[inline]
    #[doc(alias = "matrix_add_inplace_b")]
    fn add_inplace_b<B>(&self, b: &mut B)
    where
        B: MatrixMut<ROWS, COLS, T>,
        T: MatrixDataType,
    {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(self.rows(), b.rows());
            debug_assert_eq!(self.cols(), b.cols());
        }

        let count = self.len();

        let adata = self.as_ref();
        let bdata = b.as_mut();

        for index in (0..count).rev() {
            bdata[idx!(index)] += adata[idx!(index)];
        }
    }

    /// Subtracts two matrices in place, using `B = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    #[inline]
    #[doc(alias = "matrix_sub_inplace_b")]
    fn sub_inplace_b<B>(&self, b: &mut B)
    where
        B: MatrixMut<ROWS, COLS, T>,
        T: MatrixDataType,
    {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(self.rows(), b.rows());
            debug_assert_eq!(self.cols(), b.cols());
        }

        let count = self.len();

        let adata = self.as_ref();
        let bdata = b.as_mut();

        for index in (0..count).rev() {
            bdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }
}

/// A square matrix wrapping a data buffer.
pub trait SquareMatrix<const N: usize, T = f32>: AsRef<[T]> {
    /// Inverts a square lower triangular matrix. Meant to be used with
    /// [`MatrixDataMut::cholesky_decompose_lower`](crate::matrix::MatrixDataMut::cholesky_decompose_lower).
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
    /// use minikalman::matrix::{MatrixData, Matrix, MatrixMut, SquareMatrix};
    ///
    /// // data buffer for the original and decomposed matrix
    /// let mut d = [
    ///     1.0, 0.5, 0.0,
    ///     0.5, 1.0, 0.0,
    ///     0.0, 0.0, 1.0];
    /// let mut m = MatrixData::new_mut::<3, 3, f32>(&mut d);
    ///
    /// // data buffer for the inverted matrix
    /// let mut di = [0.0; 3 * 3];
    /// let mut mi = MatrixData::new_mut::<3, 3, f32>(&mut di);
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
    fn invert_l_cholesky<I>(&self, inverse: &mut I)
    where
        I: SquareMatrixMut<N, T>,
        T: MatrixDataType,
    {
        let n = N;
        let mat = self.as_ref(); // t
        let inv = inverse.as_mut(); // a

        // Inverts the lower triangular system and saves the result
        // in the upper triangle to minimize cache misses.
        for i in 0..n {
            let el_ii = mat[idx!(i * n + i)];
            let inv_el_ii = el_ii.recip();
            for j in 0..=i {
                let mut sum = T::zero();
                if i == j {
                    sum = T::one();
                };

                sum += (j..i)
                    .map(|k| -mat[idx!(i * n + k)] * inv[idx!(j * n + k)])
                    .sum::<T>();

                inv[idx!(j * n + i)] = sum * inv_el_ii;
            }
        }

        // Solve the system and handle the previous solution being in the upper triangle
        // takes advantage of symmetry.
        for i in (0..=(n - 1)).rev() {
            let el_ii = mat[idx!(i * n + i)];
            let inv_el_ii = el_ii.recip();
            for j in 0..=i {
                let mut sum = inv[idx!(j * n + i)];
                if i < j {
                    sum = T::zero();
                };

                sum += ((i + 1)..n)
                    .map(|k| -mat[idx!(k * n + i)] * inv[idx!(j * n + k)])
                    .sum::<T>();

                let value = sum * inv_el_ii;
                inv[idx!(i * n + j)] = value;
                inv[idx!(j * n + i)] = value;
            }
        }
    }
}

/// A square matrix wrapping a data buffer.
pub trait SquareMatrixMut<const N: usize, T = f32>: AsMut<[T]> + SquareMatrix<N, T> {
    /// Sets this matrix to the identity matrix, i.e. all off-diagonal entries are set to zero,
    /// diagonal entries are set to 1.0.
    fn make_identity(&mut self)
    where
        T: One + Zero + Copy,
    {
        self.make_comatrix(T::one(), T::zero())
    }

    /// Sets this matrix to a scalar matrix, i.e. all off-diagonal entries are set to zero,
    /// diagonal entries are set to the provided value.
    ///
    /// ## Arguments
    /// * `value` - The value to set the diagonal elements to.
    fn make_scalar(&mut self, diagonal: T)
    where
        T: Zero + Copy,
    {
        self.make_comatrix(diagonal, T::zero())
    }

    /// Sets the diagonal elements of the matrix to the provided value.
    ///
    /// ## Arguments
    /// * `value` - The value to set the diagonal elements to.
    fn set_diagonal_to_scalar(&mut self, value: T)
    where
        T: Copy,
    {
        let data = self.as_mut();
        for i in 0..N {
            data[i * N + i] = value;
        }
    }

    /// Sets this matrix to a comatrix, or constant diagonal matrix, i.e. a matrix where all off-diagonal entries
    /// are identical and all diagonal entries are identical.
    ///
    /// ## Arguments
    /// * `diagonal` - The value to set the diagonal elements to.
    /// * `off_diagonal` - The value to set the off-diagonal elements to.
    fn make_comatrix(&mut self, diagonal: T, off_diagonal: T)
    where
        T: Copy,
    {
        let data = self.as_mut();
        for i in 0..N {
            for j in 0..N {
                data[i * N + j] = if i == j { diagonal } else { off_diagonal };
            }
        }
    }
}

impl<const N: usize, T, M> SquareMatrix<N, T> for M where M: Matrix<N, N, T> {}
impl<const N: usize, T, M> SquareMatrixMut<N, T> for M where M: MatrixMut<N, N, T> {}

/// A mutable matrix wrapping a data buffer.
pub trait MatrixMut<const ROWS: usize, const COLS: usize, T = f32>:
    AsMut<[T]> + Matrix<ROWS, COLS, T> + IndexMut<usize, Output = T>
{
    /// Sets all elements of the matrix to the zero.
    #[doc(alias = "fill")]
    fn clear(&mut self)
    where
        T: Copy + Zero,
    {
        self.set_all(T::zero());
    }

    /// Sets a matrix element
    ///
    /// ## Arguments
    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    #[inline(always)]
    #[doc(alias = "matrix_set")]
    fn set(&mut self, row: usize, column: usize, value: T) {
        let cols = self.cols();
        self.as_mut()[idx!(row * cols + column)] = value;
    }

    /// Sets all elements of the matrix to the provided value.
    ///
    /// ## Arguments
    /// * `value` - The value to set the elements to.
    #[doc(alias = "fill")]
    fn set_all(&mut self, value: T)
    where
        T: Copy,
    {
        self.as_mut().fill(value);
    }

    /// Sets matrix elements in a symmetric matrix
    ///
    /// ## Arguments
    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    #[inline(always)]
    #[doc(alias = "matrix_set_symmetric")]
    fn set_symmetric(&mut self, row: usize, column: usize, value: T)
    where
        T: Copy,
    {
        self.set(row, column, value);
        self.set(column, row, value);
    }

    /// Adds two matrices in place, using `A = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add to, also the output.
    /// * `b` - The values to add.
    #[inline]
    #[doc(alias = "matrix_add_inplace_b")]
    fn add_inplace_a<B>(&mut self, b: &B)
    where
        B: Matrix<ROWS, COLS, T>,
        T: MatrixDataType,
    {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(self.rows(), b.rows());
            debug_assert_eq!(self.cols(), b.cols());
        }

        let count = self.len();

        let adata = self.as_mut();
        let bdata = b.as_ref();

        for index in (0..count).rev() {
            adata[idx!(index)] += bdata[idx![index]];
        }
    }

    /// Subtracts two matrices in place, using `A = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    #[inline]
    #[doc(alias = "matrix_sub_inplace_b")]
    fn sub_inplace_a<B>(&mut self, b: &B)
    where
        B: Matrix<ROWS, COLS, T>,
        T: MatrixDataType,
    {
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(self.rows(), b.rows());
            debug_assert_eq!(self.cols(), b.cols());
        }

        let count = self.len();

        let adata = self.as_mut();
        let bdata = b.as_ref();

        for index in (0..count).rev() {
            adata[idx!(index)] -= bdata[idx![index]];
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
    /// use minikalman::matrix::{MatrixData, MatrixMut};
    ///
    /// // data buffer for the original and decomposed matrix
    /// let mut d = [
    ///     1.0, 0.5, 0.0,
    ///     0.5, 1.0, 0.0,
    ///     0.0, 0.0, 1.0];
    ///
    /// let mut m = MatrixData::new_mut::<3, 3, f32>(&mut d);
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
    fn cholesky_decompose_lower(&mut self) -> bool
    where
        T: MatrixDataType,
    {
        let n = self.rows();
        let t: &mut [T] = self.as_mut();

        let mut div_el_ii = T::zero();

        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(ROWS, COLS);
            debug_assert!(ROWS > 0);
        }

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
                    if sum <= T::zero() {
                        return false;
                    }

                    let el_ii = sum.square_root();
                    t[idx!(i * n + i)] = el_ii;
                    div_el_ii = el_ii.recip();
                }
            }
        }

        // zero out the top right corner.
        for i in 0..n {
            for j in (i + 1)..n {
                t[idx!(i * n + j)] = T::zero();
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;
    use assert_float_eq::*;

    #[test]
    #[rustfmt::skip]
    fn mult() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 11.0,
            20.0, 21.0,
            30.0, 31.0];
        let a = MatrixData::new_array::<2, 3, 6, f32>(a_buf);
        let b = MatrixData::new_ref::<3, 2, f32>(&b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);

        a.mult(&b, &mut c);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 1. * 11. + 2. * 21. + 3. * 31.); // 146
        assert_f32_near!(c_buf[2], 4. * 10. + 5. * 20. + 6. * 30.); // 320
        assert_f32_near!(c_buf[3], 4. * 11. + 5. * 21. + 6. * 31.); // 335
    }

    #[test]
    #[rustfmt::skip]
    fn mult_buffered() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 11.0,
            20.0, 21.0,
            30.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<3, 2, f32>(&b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);

        let mut aux = [0f32; 3];
        a.mult_buffered(&b, &mut c, &mut aux);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 1. * 11. + 2. * 21. + 3. * 31.); // 146
        assert_f32_near!(c_buf[2], 4. * 10. + 5. * 20. + 6. * 30.); // 320
        assert_f32_near!(c_buf[3], 4. * 11. + 5. * 21. + 6. * 31.); // 335
    }

    #[test]
    #[rustfmt::skip]
    fn mult_transb() {
        let  a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<2, 3, f32>(&b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);

        a.mult_transb(&b, &mut c);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 1. * 11. + 2. * 21. + 3. * 31.); // 146
        assert_f32_near!(c_buf[2], 4. * 10. + 5. * 20. + 6. * 30.); // 320
        assert_f32_near!(c_buf[3], 4. * 11. + 5. * 21. + 6. * 31.); // 335
    }

    #[test]
    #[rustfmt::skip]
    fn mult_abat_reference() {
        let a_buf = [
            1.0, 2.0,  3.0,
            4.0, 5.0,  6.0,
            7.0, 8.0, -9.0];
        let a = MatrixData::new_ref::<3, 3, f32>(&a_buf);

        let b_buf = [
            -4.0, -1.0,  0.0,
            2.0,  3.0,  4.0,
            5.0,  9.0, -10.0];
        let b = MatrixData::new_ref::<3, 3, f32>(&b_buf);

        let mut c_buf = [0f32; 3 * 3];
        let mut c = MatrixData::new_mut::<3, 3, f32>(&mut c_buf);

        let mut d_buf = [0f32; 3 * 3];
        let mut d = MatrixData::new_mut::<3, 3, f32>(&mut d_buf);

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
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<2, 3, f32>(&b_buf);

        let mut c_buf = [
            1000., 2000.,
            3000., 4000.];
        let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);

        a.multadd_transb(&b, &mut c);
        assert_f32_near!(c.get(0, 0), 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c.get(0, 1), 2000. + 1. * 11. + 2. * 21. + 3. * 31.); // 2146
        assert_f32_near!(c.get(1, 0), 3000. + 4. * 10. + 5. * 20. + 6. * 30.); // 3320
        assert_f32_near!(c.get(1, 1), 4000. + 4. * 11. + 5. * 21. + 6. * 31.); // 4335
    }

    #[test]
    #[rustfmt::skip]
    fn multscale_transb() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<2, 3, f32>(&b_buf);

        let mut c_buf = [0f32; 2 * 2];
        let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);

        a.multscale_transb(&b, 2.0, &mut c);
        assert_f32_near!(c_buf[0], 2.0 * (1. * 10. + 2. * 20. + 3. * 30.)); // 280
        assert_f32_near!(c_buf[1], 2.0 * (1. * 11. + 2. * 21. + 3. * 31.)); // 292
        assert_f32_near!(c_buf[2], 2.0 * (4. * 10. + 5. * 20. + 6. * 30.)); // 640
        assert_f32_near!(c_buf[3], 2.0 * (4. * 11. + 5. * 21. + 6. * 31.)); // 670
    }

    #[test]
    #[rustfmt::skip]
    fn multscale_transb_owned() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<2, 3, f32>(&b_buf);

        let mut c = MatrixData::new_array::<2, 2, 4, f32>([0f32; 2 * 2]);
        a.multscale_transb(&b, 2.0, &mut c);

        let c_buf = c.as_ref();
        assert_f32_near!(c_buf[0], 2.0 * (1. * 10. + 2. * 20. + 3. * 30.)); // 280
        assert_f32_near!(c_buf[1], 2.0 * (1. * 11. + 2. * 21. + 3. * 31.)); // 292
        assert_f32_near!(c_buf[2], 2.0 * (4. * 10. + 5. * 20. + 6. * 30.)); // 640
        assert_f32_near!(c_buf[3], 2.0 * (4. * 11. + 5. * 21. + 6. * 31.)); // 670
    }

    #[test]
    #[rustfmt::skip]
    fn mult_rowvector() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0,
            20.0,
            30.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<3, 1, f32>(&b_buf);

        let mut c_buf = [0f32; 2];
        let mut c = MatrixData::new_mut::<2, 1, f32>(&mut c_buf);

        a.mult_rowvector(&b, &mut c);
        assert_f32_near!(c_buf[0], 1. * 10. + 2. * 20. + 3. * 30.); // 140
        assert_f32_near!(c_buf[1], 4. * 10. + 5. * 20. + 6. * 30.); // 320
    }

    // #####################################################################################################################

    #[test]
    #[rustfmt::skip]
    fn multadd_rowvector() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0,
            20.0,
            30.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<3, 1, f32>(&b_buf);

        let mut c_buf = [1000., 2000.];
        let mut c = MatrixData::new_mut::<2, 1, f32>(&mut c_buf);

        a.multadd_rowvector(&b, &mut c);
        assert_f32_near!(c.get(0, 0), 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c.get(1, 0), 2000. + 4. * 10. + 5. * 20. + 6. * 30.); // 2320
    }

    #[test]
    #[rustfmt::skip]
    fn get_row_pointer() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);

        let mut a_out = [0.0; 3].as_slice();
        a.get_row_pointer(0, &mut a_out);
        assert_eq!(a_out, [1.0, 2.0, 3.0]);

        a.get_row_pointer(1, &mut a_out);
        assert_eq!(a_out, [4.0, 5.0, 6.0]);
    }

    #[test]
    #[rustfmt::skip]
    fn sub() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let b = MatrixData::new_ref::<2, 3, f32>(&b_buf);

        let mut c_buf = [0f32; 2 * 3];
        let mut c = MatrixData::new_mut::<2, 3, f32>(&mut c_buf);

        a.sub(&b, &mut c);
        assert_eq!(c_buf, [
            1. - 10., 2. - 20., 3. - 30.,
            4. - 11., 5. - 21., 6. - 31.]);
    }

    #[test]
    #[rustfmt::skip]
    fn sub_inplace() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let mut b = MatrixData::new_mut::<2, 3, f32>(&mut b_buf);

        a.sub_inplace_b(&mut b);
        assert_eq!(b_buf, [
            1. - 10., 2. - 20., 3. - 30.,
            4. - 11., 5. - 21., 6. - 31.]);
    }

    #[test]
    #[rustfmt::skip]
    fn add_inplace() {
        let a_buf = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0];
        let mut b_buf = [
            10.0, 20.0, 30.0,
            11.0, 21.0, 31.0];
        let a = MatrixData::new_ref::<2, 3, f32>(&a_buf);
        let mut b = MatrixData::new_mut::<2, 3, f32>(&mut b_buf);

        a.add_inplace_b(&mut b);
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

        let mut m = MatrixData::new_mut::<3, 3, f32>(&mut d);

        // Decompose matrix to lower triangular.
        m.cholesky_decompose_lower();

        // >> chol(m, 'lower')
        //
        // answer =
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
        assert_f32_near!(d[4], 0.866_025_4);
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
        let mut m = MatrixData::new_mut::<3, 3, f32>(&mut d);

        // data buffer for the inverted matrix
        let mut di = [0.0; 3 * 3];
        let mut mi = MatrixData::new_mut::<3, 3, f32>(&mut di);

        // Decompose matrix to lower triangular.
        m.cholesky_decompose_lower();

        // >> chol(m, 'lower')
        //
        // answer =
        //
        //     1.0000         0         0
        //     0.5000    0.8660         0
        //          0         0    1.0000

        // Invert matrix using lower triangular.
        m.invert_l_cholesky(&mut mi);

        // >> inv(chol(m, 'lower'))
        //
        // answer =
        //
        //     1.0000         0         0
        //    -0.5774    1.1547         0
        //          0         0    1.0000

        // Expected result:
        // >> inv(m)
        //
        // answer =
        //
        //     1.3333   -0.6667         0
        //    -0.6667    1.3333         0
        //          0         0    1.0000

        let test = mi.get(1, 1);
        assert!(test.is_finite());
        assert!(test >= 1.3);

        assert_f32_near!(di[0], 1.333_333_3);
        assert_f32_near!(di[1], -0.666_666_6);
        assert_f32_near!(di[2], -0.0);

        assert_f32_near!(di[3], -0.666_666_6);
        assert_f32_near!(di[4], 1.333_333_3);
        assert_f32_near!(di[5], 0.0);

        assert_f32_near!(di[6], 0.0);
        assert_f32_near!(di[7], 0.0);
        assert_f32_near!(di[8], 1.0);
    }

    #[test]
    fn default_matrix_is_empty() {
        let m = MatrixData::empty::<f32>();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    #[rustfmt::skip]
    fn test_set_diagonal_to_scalar() {
        // data buffer for the original and decomposed matrix
        let d = [42.0; 16];
        let mut m = MatrixData::new_array::<4, 4, 16, f32>(d);

        // Set all diagonal elements, keeping the rest.
        m.set_diagonal_to_scalar(0.0);

        assert_eq!(
            m.as_ref(),
            [
                 0.0, 42.0, 42.0, 42.0,
                42.0,  0.0, 42.0, 42.0,
                42.0, 42.0,  0.0, 42.0,
                42.0, 42.0, 42.0,  0.0,
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_make_centrosymmetric() {
        // data buffer for the original and decomposed matrix
        let d = [42.0; 16];
        let mut m = MatrixData::new_array::<4, 4, 16, f32>(d);

        // Make it comatrix.
        m.make_comatrix(42.0, 1.0);

        assert_eq!(
            m.as_ref(),
            [
                42.0,  1.0,  1.0,  1.0,
                 1.0, 42.0,  1.0,  1.0,
                 1.0,  1.0, 42.0,  1.0,
                 1.0,  1.0,  1.0,  42.0,
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_make_scalar() {
        // data buffer for the original and decomposed matrix
        let d = [42.0; 16];
        let mut m = MatrixData::new_array::<4, 4, 16, f32>(d);

        // Make it scalar.
        m.make_scalar(10.0);

        assert_eq!(
            m.as_ref(),
            [
                10.0,  0.0,  0.0,  0.0,
                 0.0, 10.0,  0.0,  0.0,
                 0.0,  0.0, 10.0,  0.0,
                 0.0,  0.0,  0.0, 10.0,
            ]
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_make_identity() {
        // data buffer for the original and decomposed matrix
        let d = [42.0; 16];
        let mut m = MatrixData::new_array::<4, 4, 16, f32>(d);

        // Make it identity.
        m.make_identity();

        assert_eq!(
            m.as_ref(),
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        );
    }
}
