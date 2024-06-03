use crate::matrix_data_t;
use crate::types::{FastUInt16, FastUInt8};

/// Matrix operations relevant to the Kalman filter calculation.
pub trait MatrixBase {
    /// Gets the number of rows.
    fn rows(&self) -> FastUInt8;

    /// Gets the number of columns.
    fn columns(&self) -> FastUInt8;

    /// Gets the number of elements of this matrix.
    fn len(&self) -> FastUInt16;

    /// Determines if this matrix has zero elements.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets an immutable reference to the data.
    fn data_ref(&self) -> &[matrix_data_t];

    /// Gets a mutable reference to the data.
    fn data_mut(&mut self) -> &mut [matrix_data_t];
}

/// Matrix operations relevant to the Kalman filter calculation.
pub trait MatrixOps: MatrixBase {
    type Target;

    /// Inverts a square lower triangular matrix. Meant to be used with
    /// [`MatrixOps::cholesky_decompose_lower`].
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
    fn invert_l_cholesky(&self, inverse: &mut Self::Target);

    /// Performs a matrix multiplication such that `C = A * B`. This method
    /// uses an auxiliary buffer for keeping one row of `B` cached. This might
    /// improve performance on very wide matrices but is generally slower than
    /// [`Matrix::mult`](crate::Matrix::mult).
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    /// * `aux` -  Auxiliary vector that can hold a column of `b`.
    fn mult_buffered(&self, b: &Self::Target, c: &mut Self::Target, baux: &mut [matrix_data_t]);

    /// Performs a matrix multiplication such that `C = A * B`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    /// * `aux` -  Auxiliary vector that can hold a column of `b`.
    fn mult(&self, b: &Self::Target, c: &mut Self::Target);

    /// Performs a matrix multiplication such that `C = A * x`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be overwritten)
    fn mult_rowvector(&self, x: &Self::Target, c: &mut Self::Target);

    /// Performs a matrix-vector multiplication and addition such that `C = C + A * x`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `x` - Vector x
    /// * `c` - Resulting vector C (will be added to)
    fn multadd_rowvector(&self, x: &Self::Target, c: &mut Self::Target);

    /// Performs a matrix multiplication with transposed `B` such that `C = A * B'`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be overwritten)
    fn mult_transb(&self, b: &Self::Target, c: &mut Self::Target);

    /// Performs a matrix multiplication with transposed `B` and adds the result to
    /// `C` such that `C = C + A * B'`.
    ///
    /// ## Arguments
    /// * `self` - Matrix A
    /// * `b` - Matrix B
    /// * `c` - Resulting matrix C (will be added to)
    fn multadd_transb(&self, b: &Self::Target, c: &mut Self::Target);

    /// Performs a matrix multiplication with transposed `B` and scales the result such that
    /// `C = A * B' * scale`.
    ///
    /// ## Arguments
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `scale` - Scaling factor
    /// * `c` - Resulting matrix C(will be overwritten)
    fn multscale_transb(&self, b: &Self::Target, scale: matrix_data_t, c: &mut Self::Target);

    /// Gets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to get from
    /// * `rows` - The row
    /// * `cols` - The column
    ///
    /// ## Returns
    /// The value at the given cell.
    fn get(&self, row: FastUInt8, column: FastUInt8) -> matrix_data_t;

    /// Sets a matrix element
    ///
    /// ## Arguments
    /// * `mat` - The matrix to set    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    fn set(&mut self, row: FastUInt8, column: FastUInt8, value: matrix_data_t);

    /// Sets matrix elements in a symmetric matrix
    ///
    /// ## Arguments
    /// * `mat` - The matrix to set
    /// * `rows` - The row
    /// * `cols` - The column
    /// * `value` - The value to set
    fn set_symmetric(&mut self, row: FastUInt8, column: FastUInt8, value: matrix_data_t);

    /// Gets a copy of a matrix column
    ///
    /// ## Arguments
    /// * `self` - The matrix to initialize
    /// * `column` - The column
    /// * `col_data` - Pointer to an array of the correct length to hold a column of matrix `mat`.
    fn get_column_copy(&self, column: FastUInt8, col_data: &mut [matrix_data_t]);

    /// Gets a copy of a matrix row
    ///
    /// ## Arguments
    /// * `self` - The matrix to initialize
    /// * `rows` - The row
    /// * `row_data` - Pointer to an array of the correct length to hold a row of matrix `mat`.
    fn get_row_copy(&self, row: FastUInt8, row_data: &mut [matrix_data_t]);

    /// Copies the matrix from `mat` to `target`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to copy
    /// * `target` - The matrix to copy to
    fn copy(&self, target: &mut Self::Target);

    /// Subtracts two matrices, using `C = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract
    /// * `c` - The output
    fn sub(&self, b: &Self::Target, c: &mut Self::Target);

    /// Subtracts two matrices in place, using `A = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    fn sub_inplace_a(&mut self, b: &Self::Target);

    /// Subtracts two matrices in place, using `B = A - B`.
    ///
    /// ## Arguments
    /// * `self` - The matrix to subtract from
    /// * `b` - The values to subtract, also the output
    fn sub_inplace_b(&self, b: &mut Self::Target);

    /// Adds two matrices in place, using `A = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add to, also the output.
    /// * `b` - The values to add.
    fn add_inplace_a(&mut self, b: &Self::Target);

    /// Adds two matrices in place, using `B = A + B`
    ///
    /// ## Arguments
    /// * `self` - The matrix to add to
    /// * `b` - The values to add, also the output
    fn add_inplace_b(&self, b: &mut Self::Target);

    /// Decomposes a matrix into lower triangular form using Cholesky decomposition.
    ///
    /// ## Arguments
    /// * `mat` - The matrix to decompose in place into a lower triangular matrix.
    ///
    /// ## Returns
    /// `true` in case of success, `false` if the matrix is not positive semi-definite.
    fn cholesky_decompose_lower(&mut self) -> bool;
}
