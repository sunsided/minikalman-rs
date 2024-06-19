use crate::impl_mutable_vec;
use core::marker::PhantomData;

use crate::kalman::{ControlVector, ControlVectorMut};
use crate::matrix::MatrixMut;

// TODO: Add ControlVectorMutBuffer

/// Mutable buffer for the control (input) vector (`num_controls` × `1`), typically denoted "u".
///
/// Represents external inputs to the system that affect its state.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::ControlVectorBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = ControlVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = ControlVectorBuffer::<2, f32, _>::from(data.as_mut());
/// ```
pub struct ControlVectorBuffer<const CONTROLS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<CONTROLS, 1, T>;

// -----------------------------------------------------------

impl_mutable_vec!(
    ControlVectorBuffer,
    ControlVector,
    ControlVectorMut,
    CONTROLS
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, IntoInnerData, Matrix};

    #[test]
    fn test_from_array() {
        let value: ControlVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: ControlVectorBuffer<5, f32, _> = data.as_mut().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_ref(), &data));
    }

    #[test]
    fn test_access() {
        let mut value: ControlVectorBuffer<5, f32, _> = [0.0; 5].into();

        // Set values.
        {
            let matrix = value.as_matrix_mut();
            for i in 0..matrix.rows() {
                matrix.set(i, 0, i as _);
            }
        }

        // Update values.
        for i in 0..value.len() {
            value[i] += 10.0;
        }

        // Get values.
        {
            let matrix = value.as_matrix();
            for i in 0..matrix.rows() {
                assert_eq!(matrix.get(i, 0), 10.0 + i as f32);
            }
        }

        assert_eq!(value.into_inner(), [10.0, 11.0, 12.0, 13.0, 14.0]);
    }
}
