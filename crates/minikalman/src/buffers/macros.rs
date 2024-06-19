#[macro_export]
macro_rules! impl_mutable_vec {
    ($type:ident, $trait:ident, $trait_mut:ident, $rows:ident) => {
        impl<const $rows: usize, T, M> $trait_mut<$rows, T>
            for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
        }

        impl_mutable_vec!($type, $trait, $rows);
    };

    ($type:ident, $trait:ident, $rows:ident) => {
        impl<const $rows: usize, T, M> $trait<$rows, T>
            for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
        }

        /// Constructs the buffer from an exactly-sized array.
        impl<const $rows: usize, T> core::convert::From<[T; $rows]>
            for $type<$rows, T, $crate::matrix::MatrixDataArray<$rows, 1, $rows, T>>
        {
            fn from(value: [T; $rows]) -> Self {
                Self::new($crate::matrix::MatrixData::new_array::<$rows, 1, $rows, T>(
                    value,
                ))
            }
        }

        impl<const $rows: usize, T, M> $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            pub const fn new(matrix: M) -> Self {
                Self(matrix, PhantomData)
            }

            pub const fn len(&self) -> usize {
                $rows
            }

            pub const fn is_empty(&self) -> bool {
                $rows == 0
            }

            /// Ensures the underlying buffer has enough space for the expected number of values.
            pub fn is_valid(&self) -> bool {
                self.0.is_valid()
            }
        }

        impl<'a, const $rows: usize, T> From<&'a mut [T]>
            for $type<$rows, T, $crate::matrix::MatrixDataMut<'a, $rows, 1, T>>
        {
            fn from(value: &'a mut [T]) -> Self {
                #[cfg(not(feature = "no_assert"))]
                {
                    debug_assert!($rows <= value.len());
                }
                Self::new($crate::matrix::MatrixData::new_mut::<$rows, 1, T>(value))
            }
        }

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

        impl<const $rows: usize, T, M> core::convert::AsRef<[T]> for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            fn as_ref(&self) -> &[T] {
                self.0.as_ref()
            }
        }

        impl<const $rows: usize, T, M> core::convert::AsMut<[T]> for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            fn as_mut(&mut self) -> &mut [T] {
                self.0.as_mut()
            }
        }

        impl<const $rows: usize, T, M> $crate::matrix::Matrix<$rows, 1, T> for $type<$rows, T, M> where
            M: $crate::matrix::MatrixMut<$rows, 1, T>
        {
        }

        impl<const $rows: usize, T, M> $crate::matrix::MatrixMut<$rows, 1, T> for $type<$rows, T, M> where
            M: $crate::matrix::MatrixMut<$rows, 1, T>
        {
        }

        impl<const $rows: usize, T, M> core::ops::Index<usize> for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            type Output = T;

            fn index(&self, index: usize) -> &Self::Output {
                self.0.index(index)
            }
        }

        impl<const $rows: usize, T, M> core::ops::IndexMut<usize> for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T>,
        {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.0.index_mut(index)
            }
        }

        impl<const $rows: usize, T, M> $crate::matrix::IntoInnerData for $type<$rows, T, M>
        where
            M: $crate::matrix::MatrixMut<$rows, 1, T> + $crate::matrix::IntoInnerData,
        {
            type Target = M::Target;

            fn into_inner(self) -> Self::Target {
                self.0.into_inner()
            }
        }
    };
}
