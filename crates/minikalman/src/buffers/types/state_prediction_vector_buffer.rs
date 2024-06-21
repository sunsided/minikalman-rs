use crate::impl_mutable_vec;
use core::marker::PhantomData;

use crate::matrix::MatrixMut;
use crate::prelude::PredictedStateEstimateVector;

/// Mutable buffer for the temporary state prediction vector (`num_states` × `1`).
///
/// Represents the predicted state before considering the measurement.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::PredictedStateEstimateVectorBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = PredictedStateEstimateVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = PredictedStateEstimateVectorBuffer::<2, f32, _>::from(data.as_mut_slice());
/// ```
pub struct PredictedStateEstimateVectorBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, 1, T>;

// -----------------------------------------------------------

impl_mutable_vec!(
    PredictedStateEstimateVectorBuffer,
    PredictedStateEstimateVector,
    STATES
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, IntoInnerData, Matrix, RowMajorSequentialData};

    #[test]
    fn test_from_array() {
        let value: PredictedStateEstimateVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: PredictedStateEstimateVectorBuffer<5, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_access() {
        let mut value: PredictedStateEstimateVectorBuffer<5, f32, _> = [0.0; 5].into();

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
