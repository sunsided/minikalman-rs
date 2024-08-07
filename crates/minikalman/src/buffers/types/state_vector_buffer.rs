use crate::impl_mutable_vec;
use core::marker::PhantomData;

use crate::kalman::{StateVector, StateVectorMut};
use crate::prelude::MatrixMut;

/// Mutable buffer for the state vector (`num_states` × `1`), typically denoted "x".
///
/// This vector represents the state estimate. It contains the predicted values of the system's
/// state variables at a given time step. The state vector \( x \) is updated at each time step
/// based on the system dynamics, control inputs, and measurements. It provides the best estimate
/// of the current state of the system, combining prior knowledge with new information from
/// observations to minimize the estimation error.
///
/// ## Example
/// ```
/// use minikalman::buffers::types::StateVectorBuffer;
/// use minikalman::prelude::*;
///
/// // From owned data
/// let buffer = StateVectorBuffer::new(MatrixData::new_array::<4, 1, 4, f32>([0.0; 4]));
///
/// // From a reference
/// let mut data = [0.0; 4];
/// let buffer = StateVectorBuffer::<2, f32, _>::from(data.as_mut_slice());
/// ```
pub struct StateVectorBuffer<const STATES: usize, T, M>(M, PhantomData<T>)
where
    M: MatrixMut<STATES, 1, T>;

// -----------------------------------------------------------

impl_mutable_vec!(StateVectorBuffer, StateVector, StateVectorMut, STATES);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{AsMatrix, AsMatrixMut, IntoInnerData, Matrix, RowMajorSequentialData};

    #[test]
    fn test_from_array() {
        let value: StateVectorBuffer<5, f32, _> = [0.0; 5].into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
    }

    #[test]
    fn test_from_mut() {
        let mut data = [0.0_f32; 5];
        let value: StateVectorBuffer<5, f32, _> = data.as_mut_slice().into();
        assert_eq!(value.len(), 5);
        assert!(!value.is_empty());
        assert!(value.is_valid());
        assert!(core::ptr::eq(value.as_slice(), &data));
    }

    #[test]
    fn test_access() {
        let mut value: StateVectorBuffer<5, f32, _> = [0.0; 5].into();

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
