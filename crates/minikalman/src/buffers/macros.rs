#[macro_export]
macro_rules! impl_mutable_vec {
    ($type:ident, $rows:ident) => {
        impl<const $rows: usize, T, M> $crate::matrix::AsMatrix<$rows, 1, T> for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            type Target = M;

            #[inline(always)]
            fn as_matrix(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<const $rows: usize, T, M> $crate::matrix::AsMatrixMut<$rows, 1, T>
            for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            type TargetMut = M;

            #[inline(always)]
            fn as_matrix_mut(&mut self) -> &mut Self::TargetMut {
                &mut self.0
            }
        }
    };
}
