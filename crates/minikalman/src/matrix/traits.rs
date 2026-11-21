use crate::matrix::MatrixDataType;
use crate::matrix::triangular::{self, Transpose, Triangle};
use crate::prelude::{RowMajorSequentialData, RowMajorSequentialDataMut};
use core::ops::{Index, IndexMut};
use num_traits::{One, Zero};

/// Replaces `x` with `(x) as usize` to simplify index accesses.
macro_rules! idx {
    ( $x:expr ) => {
        ($x) as usize
    };
}

pub trait AsMatrix<const ROWS: usize, const COLS: usize, T> {
    type Target: Matrix<ROWS, COLS, T>;

    fn as_matrix(&self) -> &Self::Target;
}

pub trait AsMatrixMut<const ROWS: usize, const COLS: usize, T>: AsMatrix<ROWS, COLS, T> {
    type TargetMut: MatrixMut<ROWS, COLS, T>;

    fn as_matrix_mut(&mut self) -> &mut Self::TargetMut;
}

/// A matrix wrapping a data buffer.
pub trait Matrix<const ROWS: usize, const COLS: usize, T = f32>:
    RowMajorSequentialData<ROWS, COLS, T> + Index<usize, Output = T>
{
    /// Returns the number of rows of this matrix.
    fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns of this matrix.
    fn cols(&self) -> usize {
        COLS
    }

    /// Applies a function to this matrix.
    #[inline(always)]
    fn inspect<F, O>(&self, mut f: F) -> O
    where
        F: FnMut(&Self) -> O,
    {
        f(self)
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
        RowMajorSequentialData::get_at(self, row, column)
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

        let src = self.as_slice();

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

        let data = self.as_slice();
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
    #[deprecated(since = "0.7.0", note = "Use copy_to instead")]
    fn copy<Target>(&self, target: &mut Target)
    where
        Target: MatrixMut<ROWS, COLS, T>,
        T: Copy,
    {
        self.copy_to(target)
    }

    /// Copies the matrix from `mat` to `target`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to copy
    /// * `target` - The matrix to copy to
    fn copy_to<Target>(&self, target: &mut Target)
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &mut target.as_mut_slice()[..ROWS * COLS];

        for index in (0..=(count - 1)).rev() {
            bdata[idx!(index)] = adata[idx![index]];
        }
    }

    /// Copies the matrix from `source` to `self`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to copy to
    /// * `source` - The matrix to copy
    #[inline]
    fn copy_from<Source>(&mut self, source: &Source)
    where
        Self: MatrixMut<ROWS, COLS, T> + Sized,
        Source: MatrixMut<ROWS, COLS, T>,
        T: Copy,
    {
        source.copy_to(self)
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS * U];

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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..COLS * U];
        let cdata = &mut c.as_mut_slice()[..ROWS * U];

        // Rows shorter than ROW_ACCUMULATOR_MIN gain nothing from vectorizing across the row,
        // so they keep one dot product per entry. U is a constant, so the branch folds away.
        if U < ROW_ACCUMULATOR_MIN {
            for i in 0..arows {
                for j in 0..U {
                    let mut total = T::zero();
                    for k in 0..brows {
                        total += adata[i * COLS + k] * bdata[k * U + j];
                    }
                    cdata[i * ccols + j] = total;
                }
            }
            return;
        }

        // i-k-j order with a local row accumulator: every C[i][j] still sums its k terms in
        // ascending order, starting from zero, so the result is bit-identical to a dot product
        // per element; the inner loop runs over contiguous rows that LLVM can vectorize.
        for i in 0..arows {
            let mut row = [T::zero(); U];
            for k in 0..brows {
                let a_ik = adata[i * COLS + k];
                let b_k = &bdata[k * U..(k + 1) * U];
                for (acc, &b_kj) in row.iter_mut().zip(b_k) {
                    *acc += a_ik * b_kj;
                }
            }
            cdata[i * ccols..(i + 1) * ccols].copy_from_slice(&row);
        }
        let _ = bcols;
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let xdata = &x.as_slice()[..COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS];

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

        let adata = &self.as_slice()[..ROWS * COLS];
        let xdata = &x.as_slice()[..COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS];

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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..U * COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS * U];

        products_transb::<ROWS, COLS, U, T>(adata, bdata, cdata, |c, total| *c = total);
        let _ = (arows, acols, bcols, brows);
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..U * COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS * U];

        products_transb::<ROWS, COLS, U, T>(adata, bdata, cdata, |c, total| *c += total);
        let _ = (arows, acols, bcols, brows);
    }

    /// Performs a matrix multiplication with transposed `B` and scales the result such that
    /// `C = A * B' * scale`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `scale` - Scaling factor
    /// * `c` - Resulting matrix C (will be overwritten)
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..U * COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS * U];

        // test dimensions of a and b
        #[cfg(not(feature = "no_assert"))]
        {
            debug_assert_eq!(COLS, bcols);
        }

        products_transb::<ROWS, COLS, U, T>(adata, bdata, cdata, |c, total| *c = total * scale);
        let _ = (arows, acols, bcols, brows);
    }

    /// Computes the symmetric product `C = A * B'` from its lower triangle.
    ///
    /// Use this instead of [`mult_transb`](Matrix::mult_transb) when `A×Bᵀ` is symmetric in
    /// exact arithmetic, e.g. `(A×P)×Aᵀ` or `(K×S)×Kᵀ`. Only the entries on and below the
    /// diagonal are computed; each is mirrored into the upper triangle, so `C` comes out exactly
    /// symmetric for about half the multiplications. The lower triangle is bit-identical to
    /// the one [`mult_transb`](Matrix::mult_transb) computes.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B, with as many rows as A
    /// * `c` - Resulting square matrix C (will be overwritten)
    ///
    /// ## Example
    /// ```
    /// use minikalman::matrix::{MatrixData, Matrix};
    ///
    /// let a_buf = [
    ///     1.0, 2.0,
    ///     3.0, 4.0];
    /// let a = MatrixData::new_ref::<2, 2, f32>(&a_buf);
    ///
    /// let mut c_buf = [0f32; 2 * 2];
    /// let mut c = MatrixData::new_mut::<2, 2, f32>(&mut c_buf);
    ///
    /// // C = A×Aᵀ
    /// a.mult_symmetric_transb(&a, &mut c);
    /// assert_eq!(c_buf, [5.0, 11.0, 11.0, 25.0]);
    /// ```
    #[doc(alias = "matrix_mult_symmetric_transb")]
    fn mult_symmetric_transb<B, C>(&self, b: &B, c: &mut C)
    where
        B: Matrix<ROWS, COLS, T>,
        C: MatrixMut<ROWS, ROWS, T>,
        T: MatrixDataType,
    {
        let (adata, bdata, cdata) = symmetric_transb_operands(self, b, c);
        products_symmetric_transb::<ROWS, COLS, T>(adata, bdata, cdata, |c, total| *c = total);
    }

    /// Adds the symmetric product `A * B'` to `C`, from its lower triangle: `C = C + A * B'`.
    ///
    /// Use this instead of [`multadd_transb`](Matrix::multadd_transb) when `A×Bᵀ` is symmetric
    /// in exact arithmetic. Only the entries of `C` on and below the diagonal are updated; each
    /// result is mirrored into the upper triangle, whose previous values are ignored. The lower
    /// triangle is bit-identical to the one [`multadd_transb`](Matrix::multadd_transb)
    /// computes.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B, with as many rows as A
    /// * `c` - Square matrix C (its lower triangle is added to, its upper one overwritten)
    #[doc(alias = "matrix_multadd_symmetric_transb")]
    fn multadd_symmetric_transb<B, C>(&self, b: &B, c: &mut C)
    where
        B: Matrix<ROWS, COLS, T>,
        C: MatrixMut<ROWS, ROWS, T>,
        T: MatrixDataType,
    {
        let (adata, bdata, cdata) = symmetric_transb_operands(self, b, c);
        products_symmetric_transb::<ROWS, COLS, T>(adata, bdata, cdata, |c, total| *c += total);
    }

    /// Computes the scaled symmetric product `C = A * B' * scale` from its lower triangle.
    ///
    /// Use this instead of [`multscale_transb`](Matrix::multscale_transb) when `A×Bᵀ` is
    /// symmetric in exact arithmetic. Only the entries on and below the diagonal are computed
    /// and mirrored into the upper triangle. The lower triangle is bit-identical to the one
    /// [`multscale_transb`](Matrix::multscale_transb) computes.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B, with as many rows as A
    /// * `scale` - Scaling factor
    /// * `c` - Resulting square matrix C (will be overwritten)
    #[doc(alias = "matrix_multscale_symmetric_transb")]
    fn multscale_symmetric_transb<B, C>(&self, b: &B, scale: T, c: &mut C)
    where
        B: Matrix<ROWS, COLS, T>,
        C: MatrixMut<ROWS, ROWS, T>,
        T: MatrixDataType,
    {
        let (adata, bdata, cdata) = symmetric_transb_operands(self, b, c);
        products_symmetric_transb::<ROWS, COLS, T>(adata, bdata, cdata, |c, total| {
            *c = total * scale;
        });
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..ROWS * COLS];
        let cdata = &mut c.as_mut_slice()[..ROWS * COLS];

        for index in (0..count).rev() {
            cdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }

    /// Adds two matrices in place, using `B = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add (Matrix A)
    /// * `b` - The matrix to add to, i.e. the output (Matrix B)
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &mut b.as_mut_slice()[..ROWS * COLS];

        for index in (0..count).rev() {
            bdata[idx!(index)] += adata[idx!(index)];
        }
    }

    /// Subtracts two matrices in place, using `B = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from (Matrix A)
    /// * `b` - The values to subtract, also the output (Matrix B)
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

        let adata = &self.as_slice()[..ROWS * COLS];
        let bdata = &mut b.as_mut_slice()[..ROWS * COLS];

        for index in (0..count).rev() {
            bdata[idx!(index)] = adata[idx!(index)] - bdata[idx![index]];
        }
    }
}

/// The shortest output row for which `mult` and the `mult_transb` family accumulate the whole
/// row at once. Shorter rows keep one dot product per entry: for them the accumulator only adds
/// overhead (measured with `benches/compare`, e.g. the 2-state, 1-measurement filter). Both
/// paths sum every entry in the same order, so the results are the same bit for bit.
const ROW_ACCUMULATOR_MIN: usize = 4;

/// Computes `C = A×Bᵀ` row by row for `A` (`R`×`K`), `B` (`U`×`K`) and `C` (`R`×`U`),
/// passing each entry to `store`.
///
/// All `U` dot products of a row of `A` run at once (i-k-j order with a local accumulator).
/// Every entry still sums its `K` terms in ascending order from zero, so the result is
/// bit-identical to one dot product per entry, but the inner loop can be vectorized.
#[inline(always)]
fn products_transb<const R: usize, const K: usize, const U: usize, T>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    mut store: impl FnMut(&mut T, T),
) where
    T: MatrixDataType,
{
    // Short rows keep one dot product per entry; see `ROW_ACCUMULATOR_MIN`.
    if U < ROW_ACCUMULATOR_MIN {
        for i in 0..R {
            for j in 0..U {
                let mut total = T::zero();
                for k in 0..K {
                    total += a[i * K + k] * b[j * K + k];
                }
                store(&mut c[i * U + j], total);
            }
        }
        return;
    }

    for i in 0..R {
        let mut row = [T::zero(); U];
        for k in 0..K {
            let a_ik = a[i * K + k];
            for (j, acc) in row.iter_mut().enumerate() {
                *acc += a_ik * b[j * K + k];
            }
        }
        for (c_ij, &total) in c[i * U..(i + 1) * U].iter_mut().zip(&row) {
            store(c_ij, total);
        }
    }
}

/// Checks the shapes of `C = A×Bᵀ` with a square `C` and returns the operands as slices of
/// their static lengths, so that the kernel needs no bounds checks.
#[inline(always)]
fn symmetric_transb_operands<'a, const N: usize, const K: usize, T, A, B, C>(
    a: &'a A,
    b: &'a B,
    c: &'a mut C,
) -> (&'a [T], &'a [T], &'a mut [T])
where
    A: Matrix<N, K, T> + ?Sized,
    B: Matrix<N, K, T>,
    C: MatrixMut<N, N, T>,
    T: MatrixDataType,
{
    #[cfg(not(feature = "no_assert"))]
    {
        debug_assert_eq!(N, b.rows());
        debug_assert_eq!(K, b.cols());
        debug_assert_eq!(N, c.rows());
        debug_assert_eq!(N, c.cols());
    }

    (
        &a.as_slice()[..N * K],
        &b.as_slice()[..N * K],
        &mut c.as_mut_slice()[..N * N],
    )
}

/// Computes the lower triangle of `C = A×Bᵀ` for `A`, `B` (`N`×`K`) and a square `C`, passes
/// each entry to `store` and mirrors the stored value into the upper triangle.
///
/// Every entry sums its `K` terms in ascending order from zero, as in [`products_transb`], so
/// the lower triangle is the same bit for bit. Unlike [`products_transb`], every size keeps one
/// dot product per entry. Measured with `benches/compare` on x86-64: a row accumulator over
/// only the lower part of a row has a varying length and vectorizes poorly (6 states: about
/// 80 ns instead of 62 ns for the prediction). A full-length one wastes the upper half; it was
/// about 1 ns faster for 4 states in `f32`, slower for 6, and added 60 to 80 bytes of `.text`
/// to a `thumbv7em` step.
#[inline(always)]
fn products_symmetric_transb<const N: usize, const K: usize, T>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    mut store: impl FnMut(&mut T, T),
) where
    T: MatrixDataType,
{
    // Row i writes only C[i][..=i] and C[..=i][i]; earlier rows never touch C[i][..i], so
    // `store` always reads the caller's lower-triangle value.
    for i in 0..N {
        for j in 0..=i {
            let mut total = T::zero();
            for k in 0..K {
                total += a[i * K + k] * b[j * K + k];
            }

            store(&mut c[i * N + j], total);
            c[j * N + i] = c[i * N + j];
        }
    }
}

/// A square matrix wrapping a data buffer.
pub trait SquareMatrix<const N: usize, T = f32>: RowMajorSequentialData<N, N, T> {
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
        let mat = &self.as_slice()[..N * N]; // t
        let inv = &mut inverse.as_mut_slice()[..N * N]; // a

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

    /// Solves `L×X = B` in place for the lower triangular `self = L`, i.e. replaces `b` by
    /// `L⁻¹×b` (forward substitution).
    ///
    /// Only the lower triangle of `self` is read, so a Cholesky factor from
    /// [`MatrixMut::cholesky_decompose_lower`] can be used directly.
    ///
    /// ## Returns
    /// `false` if a diagonal entry of `L` is zero, in which case `b` is left untouched.
    ///
    /// ## Example
    /// ```
    /// use minikalman::matrix::{MatrixData, RowMajorSequentialData, SquareMatrix};
    ///
    /// let l = MatrixData::new_array::<2, 2, 4, f32>([
    ///     2.0, 0.0,
    ///     1.0, 4.0]);
    /// let mut b = MatrixData::new_array::<2, 1, 2, f32>([2.0, 9.0]);
    ///
    /// assert!(l.solve_lower_triangular_inplace(&mut b));
    /// assert_eq!(b.as_slice(), &[1.0, 2.0]);
    /// ```
    fn solve_lower_triangular_inplace<const K: usize, B>(&self, b: &mut B) -> bool
    where
        B: MatrixMut<N, K, T>,
        T: MatrixDataType,
    {
        solve(self, Triangle::Lower, Transpose::No, b)
    }

    /// Solves `Lᵀ×X = B` in place for the lower triangular `self = L`, i.e. replaces `b` by
    /// `L⁻ᵀ×b` (back substitution).
    ///
    /// Together with [`solve_lower_triangular_inplace`](Self::solve_lower_triangular_inplace)
    /// this solves `(L×Lᵀ)×X = B` with a Cholesky factor. Only the lower triangle of `self` is
    /// read.
    ///
    /// ## Returns
    /// `false` if a diagonal entry of `L` is zero, in which case `b` is left untouched.
    fn solve_lower_triangular_transposed_inplace<const K: usize, B>(&self, b: &mut B) -> bool
    where
        B: MatrixMut<N, K, T>,
        T: MatrixDataType,
    {
        solve(self, Triangle::Lower, Transpose::Yes, b)
    }

    /// Solves `U×X = B` in place for the upper triangular `self = U`, i.e. replaces `b` by
    /// `U⁻¹×b` (back substitution).
    ///
    /// Only the upper triangle of `self` is read, so the `R` factor from
    /// [`MatrixMut::triangularize_upper`] can be used directly.
    ///
    /// ## Returns
    /// `false` if a diagonal entry of `U` is zero, in which case `b` is left untouched.
    fn solve_upper_triangular_inplace<const K: usize, B>(&self, b: &mut B) -> bool
    where
        B: MatrixMut<N, K, T>,
        T: MatrixDataType,
    {
        solve(self, Triangle::Upper, Transpose::No, b)
    }

    /// Solves `Uᵀ×X = B` in place for the upper triangular `self = U`, i.e. replaces `b` by
    /// `U⁻ᵀ×b` (forward substitution).
    ///
    /// Only the upper triangle of `self` is read.
    ///
    /// ## Returns
    /// `false` if a diagonal entry of `U` is zero, in which case `b` is left untouched.
    fn solve_upper_triangular_transposed_inplace<const K: usize, B>(&self, b: &mut B) -> bool
    where
        B: MatrixMut<N, K, T>,
        T: MatrixDataType,
    {
        solve(self, Triangle::Upper, Transpose::Yes, b)
    }
}

/// Runs [`triangular::solve_triangular`] for the square factor `t` and the right-hand sides `b`.
fn solve<const N: usize, const K: usize, T, F, B>(
    t: &F,
    triangle: Triangle,
    transpose: Transpose,
    b: &mut B,
) -> bool
where
    F: SquareMatrix<N, T> + ?Sized,
    B: MatrixMut<N, K, T>,
    T: MatrixDataType,
{
    // Slices of the static length, so that the substitutions need no bounds checks.
    triangular::solve_triangular(
        &t.as_slice()[..N * N],
        N,
        triangle,
        transpose,
        &mut b.as_mut_slice()[..N * K],
        K,
    )
}

/// A square matrix wrapping a data buffer.
pub trait SquareMatrixMut<const N: usize, T = f32>:
    RowMajorSequentialDataMut<N, N, T> + SquareMatrix<N, T>
{
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
        let data = &mut self.as_mut_slice()[..N * N];
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
        let data = &mut self.as_mut_slice()[..N * N];
        for i in 0..N {
            for j in 0..N {
                data[i * N + j] = if i == j { diagonal } else { off_diagonal };
            }
        }
    }
}

impl<const N: usize, T, M> SquareMatrix<N, T> for M where M: Matrix<N, N, T> {}
impl<const N: usize, T, M> SquareMatrixMut<N, T> for M where M: MatrixMut<N, N, T> {}

pub trait Scalar<T = f32>: Matrix<1, 1, T> {
    /// Gets the value of a matrix of size 1×1.
    #[inline(always)]
    fn get_value(&self) -> T
    where
        T: Copy,
    {
        RowMajorSequentialData::get_at(self, 0, 0)
    }
}

pub trait RowVector<const ROWS: usize, T = f32>: Matrix<ROWS, 1, T> {
    /// Gets the value of the n-th row of a matrix of size N×1.
    #[inline(always)]
    fn get_row(&self, row: usize) -> T
    where
        T: Copy,
    {
        RowMajorSequentialData::get_at(self, row, 0)
    }
}

pub trait ColumnVector<const COLS: usize, T = f32>: Matrix<1, COLS, T> {
    /// Gets the value of the n-th column of a matrix of size 1×N.
    #[inline(always)]
    fn get_col(&self, column: usize) -> T
    where
        T: Copy,
    {
        RowMajorSequentialData::get_at(self, 0, column)
    }
}

pub trait ScalarMut<T = f32>: MatrixMut<1, 1, T> {
    /// Sets the value of a matrix of size 1×1.
    #[inline(always)]
    fn set_value(&mut self, value: T) {
        RowMajorSequentialDataMut::set_at(self, 0, 0, value)
    }
}

pub trait RowVectorMut<const ROWS: usize, T = f32>:
    MatrixMut<ROWS, 1, T> + RowVector<ROWS, T>
{
    /// Sets the value of the n-th row of a matrix of size N×1.
    #[inline(always)]
    fn set_row(&mut self, row: usize, value: T) {
        RowMajorSequentialDataMut::set_at(self, row, 0, value)
    }
}

pub trait ColumnVectorMut<const COLS: usize, T = f32>:
    MatrixMut<1, COLS, T> + ColumnVector<COLS, T>
{
    /// Sets the value of the n-th column of a matrix of size 1×N.
    #[inline(always)]
    fn set_col(&mut self, column: usize, value: T) {
        RowMajorSequentialDataMut::set_at(self, 0, column, value)
    }
}

impl<T, M> Scalar<T> for M where M: Matrix<1, 1, T> {}
impl<T, M> ScalarMut<T> for M where M: MatrixMut<1, 1, T> {}
impl<const N: usize, T, M> RowVector<N, T> for M where M: Matrix<N, 1, T> {}
impl<const N: usize, T, M> RowVectorMut<N, T> for M where M: MatrixMut<N, 1, T> {}
impl<const N: usize, T, M> ColumnVector<N, T> for M where M: Matrix<1, N, T> {}
impl<const N: usize, T, M> ColumnVectorMut<N, T> for M where M: MatrixMut<1, N, T> {}

/// A mutable matrix wrapping a data buffer.
pub trait MatrixMut<const ROWS: usize, const COLS: usize, T = f32>:
    Matrix<ROWS, COLS, T> + RowMajorSequentialDataMut<ROWS, COLS, T> + IndexMut<usize, Output = T>
{
    /// Applies a function to this matrix.
    #[inline(always)]
    fn apply<F, O>(&mut self, mut f: F) -> O
    where
        F: FnMut(&mut Self) -> O,
    {
        f(self)
    }

    /// Sets all elements of the matrix to the zero.
    #[doc(alias = "fill")]
    #[inline(always)]
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
        self.as_mut_slice()[idx!(row * cols + column)] = value;
    }

    /// Sets all elements of the matrix to the provided value.
    ///
    /// ## Arguments
    /// * `value` - The value to set the elements to.
    #[doc(alias = "fill")]
    #[inline(always)]
    fn set_all(&mut self, value: T)
    where
        T: Copy,
    {
        self.as_mut_slice().fill(value);
    }

    /// Sets all elements of the matrix to the provided value.
    ///
    /// ## Arguments
    /// * `value` - The value to set the elements to.
    #[inline(always)]
    fn set_zero(&mut self)
    where
        T: Copy + Zero,
    {
        self.set_all(T::zero())
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
        RowMajorSequentialDataMut::set_at(self, row, column, value);
        RowMajorSequentialDataMut::set_at(self, column, row, value);
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

        let adata = &mut self.as_mut_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..ROWS * COLS];

        for index in (0..count).rev() {
            adata[idx!(index)] += bdata[idx![index]];
        }
    }

    /// Subtracts two matrices in place, using `A = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from, also the output.
    /// * `b` - The values to subtract.
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

        let adata = &mut self.as_mut_slice()[..ROWS * COLS];
        let bdata = &b.as_slice()[..ROWS * COLS];

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
    /// `true` in case of success, `false` if the matrix is not positive definite, or if a pivot
    /// is not finite (`NaN` or infinite, e.g. because the entries overflow). On success, every
    /// entry of the factor is finite.
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
        let t: &mut [T] = &mut self.as_mut_slice()[..ROWS * COLS];

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
                    // Is it positive definite? Written as `!(sum > 0)` so that a NaN pivot,
                    // e.g. from `inf - inf` with overflowing entries, is rejected as well.
                    // `sum - sum` is zero only for finite values, which rejects `+inf`.
                    #[allow(clippy::eq_op, clippy::neg_cmp_op_on_partial_ord)]
                    if !(sum > T::zero()) || sum - sum != T::zero() {
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

    /// Triangularizes the matrix in place with Householder reflections (the `R` of a QR
    /// decomposition), without forming `Mᵀ×M`.
    ///
    /// Afterwards the first `COLS` rows hold the upper triangular `R` with `Mᵀ×M = Rᵀ×R` and a
    /// non-negative diagonal, and the rows below are zero. `Q` is not formed. Unlike
    /// `cholesky(Mᵀ×M)`, this does not square the condition number, which is what square-root
    /// filters need, e.g. `R = qr_r([A×S, Lq]ᵀ)` gives `Rᵀ×R = A×S×Sᵀ×Aᵀ + Lq×Lqᵀ`.
    ///
    /// Each reflector is computed from its column divided by the column's largest magnitude
    /// (as BLAS `nrm2` does), so entries are never squared unscaled. With fixed-point types,
    /// every column norm times `√(2·ROWS + 2·√ROWS)` must fit the type, since that bounds the
    /// dot products with the reflectors. For `I16F16` (maximum 32767), column norms must stay
    /// below about 8000 for 6 rows and 5800 for 12 rows. A column that is zero from the diagonal down is
    /// skipped and leaves a zero diagonal entry, a valid singular factor. The operation count
    /// depends only on `ROWS` and `COLS`, apart from such skipped columns.
    ///
    /// ## Returns
    /// `false` if `ROWS < COLS` (stable const generics cannot rule this out at compile time),
    /// or if an entry or a diagonal of `R` is not finite (`NaN` or infinite, e.g. after an
    /// overflow). The matrix is then partially modified.
    ///
    /// ## Example
    /// ```
    /// use minikalman::matrix::{MatrixData, MatrixMut, RowMajorSequentialData};
    ///
    /// let mut m = MatrixData::new_array::<3, 2, 6, f32>([
    ///     3.0, 1.0,
    ///     0.0, 2.0,
    ///     4.0, 3.0]);
    ///
    /// assert!(m.triangularize_upper());
    ///
    /// // Rᵀ×R = Mᵀ×M = [[25, 15], [15, 14]]. The tolerance allows for the approximate
    /// // square root of the `micromath` feature.
    /// let r = m.as_slice();
    /// assert!((r[0] - 5.0).abs() < 0.2 && (r[1] - 3.0).abs() < 0.2);
    /// assert!(r[2].abs() < 0.2 && (r[3] - 5.0f32.sqrt()).abs() < 0.2);
    /// assert_eq!(&r[4..], &[0.0, 0.0]);
    /// ```
    #[doc(alias = "qr")]
    fn triangularize_upper(&mut self) -> bool
    where
        T: MatrixDataType,
    {
        triangular::triangularize_upper(self.as_mut_slice(), ROWS, COLS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixData;
    use assert_float_eq::*;

    /// Symmetrizes `data` and adds 1 to the diagonal, as the `matrix_cholesky` fuzz target does.
    fn regularized_4x4(data: [f32; 16]) -> [f32; 16] {
        let mut m = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                m[i * 4 + j] = (data[i * 4 + j] + data[j * 4 + i]) * 0.5;
            }
            m[i * 4 + i] += 1.0;
        }
        m
    }

    /// Overflowing entries turn a pivot into `inf - inf = NaN`, which `sum <= 0` let through.
    /// This input was found by the `matrix_cholesky` fuzz target.
    #[test]
    #[rustfmt::skip]
    fn cholesky_rejects_overflowing_entries() {
        let mut data = regularized_4x4([
            5.786e-39, 1.794e-43, 0.0, -1.701e38,
            2.351e-38, 0.0, 0.0, -1.701e38,
            2.351e-38, 0.0, 0.0, 0.0,
            0.0, -2.545e38, 2.351e-38, 0.0,
        ]);
        let mut m = MatrixData::new_mut::<4, 4, f32>(&mut data);
        assert!(!m.cholesky_decompose_lower());
    }

    #[test]
    fn cholesky_rejects_non_finite_pivots() {
        for value in [f32::NAN, f32::INFINITY] {
            let mut data = [1.0, 0.0, 0.0, value];
            let mut m = MatrixData::new_mut::<2, 2, f32>(&mut data);
            assert!(!m.cholesky_decompose_lower(), "pivot {value}");
        }
    }

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
        assert_f32_near!(c.get_at(0, 0), 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c.get_at(0, 1), 2000. + 1. * 11. + 2. * 21. + 3. * 31.); // 2146
        assert_f32_near!(c.get_at(1, 0), 3000. + 4. * 10. + 5. * 20. + 6. * 30.); // 3320
        assert_f32_near!(c.get_at(1, 1), 4000. + 4. * 11. + 5. * 21. + 6. * 31.); // 4335
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

        let c_buf = c.as_slice();
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
        assert_f32_near!(c.get_at(0, 0), 1000. + 1. * 10. + 2. * 20. + 3. * 30.); // 1140
        assert_f32_near!(c.get_at(1, 0), 2000. + 4. * 10. + 5. * 20. + 6. * 30.); // 2320
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
        assert!(d[4] >= 0.866 && d[4] <= 0.875);
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

        let test = mi.get_at(1, 1);
        assert!(test.is_finite());
        assert!(test >= 1.3);

        assert!(di[0] >= 1.32 && di[0] <= 1.34);
        assert!(di[1] >= -0.667 && di[1] <= -0.65);
        assert_f32_near!(di[2], -0.0);

        assert!(di[3] >= -0.667 && di[3] <= -0.65);
        assert!(di[4] >= 1.30 && di[4] <= 1.35);
        assert_f32_near!(di[5], 0.0);

        assert_f32_near!(di[6], 0.0);
        assert_f32_near!(di[7], 0.0);
        assert!(di[8] >= 1.0 && di[8] <= 1.00001);
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
            m.as_slice(),
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
            m.as_slice(),
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
            m.as_slice(),
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
            m.as_slice(),
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        );
    }
}

/// Pins the vectorizable kernels to the loops they replaced: same results, bit for bit.
#[cfg(test)]
mod kernel_bit_identity {
    use super::*;
    use crate::matrix::MatrixData;
    use crate::test_bits::{ToBits, assert_bits_eq, test_value};

    /// Longer than every operand, which `is_valid` allows.
    const CAP: usize = 64;

    /// The former `mult`: j (descending), i, k, one dot product per entry.
    fn old_mult<T: MatrixDataType>(a: &[T], b: &[T], c: &mut [T], r: usize, k: usize, u: usize) {
        for j in (0..u).rev() {
            let mut index_a = 0;
            for i in 0..r {
                let mut total = T::zero();
                for l in 0..k {
                    total += a[index_a] * b[l * u + j];
                    index_a += 1;
                }
                c[i * u + j] = total;
            }
        }
    }

    /// The former `mult_transb` family: one dot product per entry, stored by `store`.
    fn old_transb<T: MatrixDataType>(
        a: &[T],
        b: &[T],
        c: &mut [T],
        (r, k, u): (usize, usize, usize),
        store: impl Fn(&mut T, T),
    ) {
        for i in 0..r {
            for j in 0..u {
                let mut total = T::zero();
                for l in 0..k {
                    total += a[i * k + l] * b[j * k + l];
                }
                store(&mut c[i * u + j], total);
            }
        }
    }

    /// The former `mult_rowvector` family: starts from `A[i][0]×x[0]`.
    fn old_rowvector<T: MatrixDataType>(
        a: &[T],
        x: &[T],
        c: &mut [T],
        (r, k): (usize, usize),
        store: impl Fn(&mut T, T),
    ) {
        for i in 0..r {
            let mut total = a[i * k] * x[0];
            for l in 1..k {
                total += a[i * k + l] * x[l];
            }
            store(&mut c[i], total);
        }
    }

    /// `CAP` deterministic values in `[-scale, scale)`.
    fn values<T: MatrixDataType>(seed: u32, scale: f64) -> [T; CAP] {
        core::array::from_fn(|i| T::from_f64(test_value(seed, i) * scale).unwrap())
    }

    fn check_mult<const R: usize, const K: usize, const U: usize, T>(scale: f64)
    where
        T: MatrixDataType + ToBits,
    {
        let (a, b, c0) = (
            values::<T>(1, scale),
            values::<T>(2, scale),
            values::<T>(3, scale),
        );
        let mut want = c0;
        old_mult(&a, &b, &mut want, R, K, U);
        let mut got = c0;
        MatrixData::new_ref::<R, K, T>(&a).mult(
            &MatrixData::new_ref::<K, U, T>(&b),
            &mut MatrixData::new_mut::<R, U, T>(&mut got),
        );
        // Entries past R×U must stay untouched as well.
        assert_bits_eq(&got, &want, "mult");
    }

    fn check_transb<const R: usize, const K: usize, const U: usize, T>(scale: f64)
    where
        T: MatrixDataType + ToBits,
    {
        let (a, b, c0) = (
            values::<T>(4, scale),
            values::<T>(5, scale),
            values::<T>(6, scale),
        );
        let s = T::from_f64(0.75).unwrap();
        let shape = (R, K, U);
        let am = MatrixData::new_ref::<R, K, T>(&a);
        let bm = MatrixData::new_ref::<U, K, T>(&b);

        let (mut want, mut got) = (c0, c0);
        old_transb(&a, &b, &mut want, shape, |c, t| *c = t);
        am.mult_transb(&bm, &mut MatrixData::new_mut::<R, U, T>(&mut got));
        assert_bits_eq(&got, &want, "mult_transb");

        let (mut want, mut got) = (c0, c0);
        old_transb(&a, &b, &mut want, shape, |c, t| *c += t);
        am.multadd_transb(&bm, &mut MatrixData::new_mut::<R, U, T>(&mut got));
        assert_bits_eq(&got, &want, "multadd_transb");

        let (mut want, mut got) = (c0, c0);
        old_transb(&a, &b, &mut want, shape, |c, t| *c = t * s);
        am.multscale_transb(&bm, s, &mut MatrixData::new_mut::<R, U, T>(&mut got));
        assert_bits_eq(&got, &want, "multscale_transb");
    }

    /// Asserts that the `N`×`N` result `got` holds the lower triangle of `want` bit for bit,
    /// mirrors it exactly into the upper one, and leaves the entries past `N×N` as in `want`.
    fn assert_mirrored_lower<const N: usize, T: ToBits>(got: &[T], want: &[T], what: &str) {
        for i in 0..N {
            for j in 0..=i {
                assert_bits_eq(&[got[i * N + j]], &[want[i * N + j]], what);
                assert_bits_eq(&[got[j * N + i]], &[got[i * N + j]], what);
            }
        }

        assert_bits_eq(&got[N * N..], &want[N * N..], what);
    }

    fn check_symmetric_transb<const N: usize, const K: usize, T>(scale: f64)
    where
        T: MatrixDataType + ToBits,
    {
        let (a, b, c0) = (
            values::<T>(10, scale),
            values::<T>(11, scale),
            values::<T>(12, scale),
        );
        let s = T::from_f64(0.75).unwrap();
        let shape = (N, K, N);
        let am = MatrixData::new_ref::<N, K, T>(&a);
        let bm = MatrixData::new_ref::<N, K, T>(&b);

        // A and B are unrelated, so the full product is not symmetric: the upper triangle
        // must come from the mirror, not from computing it.
        let (mut want, mut got) = (c0, c0);
        old_transb(&a, &b, &mut want, shape, |c, t| *c = t);
        am.mult_symmetric_transb(&bm, &mut MatrixData::new_mut::<N, N, T>(&mut got));
        assert_mirrored_lower::<N, T>(&got, &want, "mult_symmetric_transb");

        let (mut want, mut got) = (c0, c0);
        old_transb(&a, &b, &mut want, shape, |c, t| *c += t);
        am.multadd_symmetric_transb(&bm, &mut MatrixData::new_mut::<N, N, T>(&mut got));
        assert_mirrored_lower::<N, T>(&got, &want, "multadd_symmetric_transb");

        let (mut want, mut got) = (c0, c0);
        old_transb(&a, &b, &mut want, shape, |c, t| *c = t * s);
        am.multscale_symmetric_transb(&bm, s, &mut MatrixData::new_mut::<N, N, T>(&mut got));
        assert_mirrored_lower::<N, T>(&got, &want, "multscale_symmetric_transb");
    }

    fn check_rowvector<const R: usize, const K: usize, T>(scale: f64)
    where
        T: MatrixDataType + ToBits,
    {
        let (a, x, c0) = (
            values::<T>(7, scale),
            values::<T>(8, scale),
            values::<T>(9, scale),
        );
        let am = MatrixData::new_ref::<R, K, T>(&a);
        let xm = MatrixData::new_ref::<K, 1, T>(&x);

        let (mut want, mut got) = (c0, c0);
        old_rowvector(&a, &x, &mut want, (R, K), |c, t| *c = t);
        am.mult_rowvector(&xm, &mut MatrixData::new_mut::<R, 1, T>(&mut got));
        assert_bits_eq(&got, &want, "mult_rowvector");

        let (mut want, mut got) = (c0, c0);
        old_rowvector(&a, &x, &mut want, (R, K), |c, t| *c += t);
        am.multadd_rowvector(&xm, &mut MatrixData::new_mut::<R, 1, T>(&mut got));
        assert_bits_eq(&got, &want, "multadd_rowvector");
    }

    fn check_all<T: MatrixDataType + ToBits>(scale: f64) {
        // U = 1, 2, 3 take the dot-product path, U ≥ ROW_ACCUMULATOR_MIN the row accumulator.
        check_mult::<1, 1, 1, T>(scale);
        check_mult::<2, 2, 2, T>(scale);
        check_mult::<3, 3, 3, T>(scale);
        check_mult::<2, 3, 4, T>(scale);
        check_mult::<3, 2, 5, T>(scale);
        check_mult::<4, 4, 4, T>(scale);
        check_mult::<5, 1, 3, T>(scale);
        check_mult::<6, 6, 6, T>(scale);
        check_mult::<2, 7, 1, T>(scale);

        check_transb::<1, 1, 1, T>(scale);
        check_transb::<2, 2, 2, T>(scale);
        check_transb::<3, 3, 3, T>(scale);
        check_transb::<2, 3, 4, T>(scale);
        check_transb::<4, 4, 2, T>(scale);
        check_transb::<3, 5, 3, T>(scale);
        check_transb::<6, 6, 6, T>(scale);
        check_transb::<5, 1, 2, T>(scale);

        check_symmetric_transb::<1, 1, T>(scale);
        check_symmetric_transb::<2, 1, T>(scale);
        check_symmetric_transb::<2, 2, T>(scale);
        check_symmetric_transb::<3, 5, T>(scale);
        check_symmetric_transb::<4, 2, T>(scale);
        check_symmetric_transb::<4, 4, T>(scale);
        check_symmetric_transb::<5, 3, T>(scale);
        check_symmetric_transb::<6, 6, T>(scale);
        check_symmetric_transb::<7, 1, T>(scale);

        check_rowvector::<1, 1, T>(scale);
        check_rowvector::<4, 2, T>(scale);
        check_rowvector::<2, 5, T>(scale);
        check_rowvector::<6, 6, T>(scale);
        check_rowvector::<7, 3, T>(scale);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    fn kernels_match_the_old_loops_f32() {
        check_all::<f32>(1.0);
        // Large magnitudes, where the summation order matters most.
        check_all::<f32>(1e6);
    }

    #[test]
    #[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
    fn kernels_match_the_old_loops_f64() {
        check_all::<f64>(1.0);
        check_all::<f64>(1e12);
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn kernels_match_the_old_loops_fixed() {
        check_all::<fixed::types::I16F16>(1.0);
        check_all::<fixed::types::I32F32>(100.0);
    }
}
