use crate::impl_mutable_vec;
use core::marker::PhantomData;

use crate::kalman::InnovationVector;
use crate::prelude::MatrixMut;

/// Mutable buffer for the innovation vector (`num_measurements` × `1`), typically denoted "y".
///
/// This vector represents the innovation (or residual). It is the difference between the actual
/// measurement and the predicted measurement based on the current state estimate. The innovation
/// vector \( y \) quantifies the discrepancy between observed data and the filter's predictions,
/// providing a measure of the new information gained from the measurements. This vector is used
/// to update the state estimate, ensuring that the Kalman Filter corrects for any deviations
/// between the predicted and actual observations, thus refining the state estimation.
///
/// Some implementations may choose to use it as a temporary observation buffer, e.g. during
/// Extended Kalman Filter measurement updates.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::InnovationVectorBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = InnovationVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = InnovationVectorBuffer::<2, f32, _>::from(data.as_mut_slice());
/// ```
pub struct InnovationVectorBuffer<const OBSERVATIONS: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<OBSERVATIONS, 1, T>;

// -----------------------------------------------------------

impl_mutable_vec!(InnovationVectorBuffer, InnovationVector, OBSERVATIONS);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, IntoInnerData, Matrix, RowMajorSequentialData};

    #[test]
    fn test_from_array() {
        let value: InnovationVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: InnovationVectorBuffer<5, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_access() {
        let mut value: InnovationVectorBuffer<5, f32, _> = [0.0; 5].into();

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
