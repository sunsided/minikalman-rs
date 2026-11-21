//! Support items for the code [`filter!`](crate::filter) generates. Not part of the public API.

use crate::buffers::types::{
    ControlMatrixMutBuffer, ControlProcessNoiseCovarianceMatrixMutBuffer, ControlVectorBuffer,
    DirectProcessNoiseCovarianceMatrixMutBuffer, EstimateCovarianceMatrixBuffer,
    InnovationCovarianceMatrixBuffer, InnovationVectorBuffer, KalmanGainMatrixBuffer,
    MeasurementNoiseCovarianceMatrixBuffer, MeasurementVectorBuffer, ObservationMatrixMutBuffer,
    PredictedStateEstimateVectorBuffer, StateTransitionMatrixMutBuffer, StateVectorBuffer,
    TemporaryBQMatrixBuffer, TemporaryHPMatrixBuffer, TemporaryKHPMatrixBuffer,
    TemporaryPHTMatrixBuffer, TemporaryResidualCovarianceInvertedMatrixBuffer,
    TemporaryStateMatrixBuffer,
};
use crate::matrix::{MatrixDataArray, MatrixDataType};
use crate::regular::{
    Control, ControlBuilder, RegularKalman, RegularKalmanBuilder, RegularObservation,
    RegularObservationBuilder,
};

/// The scalar type of a named vector declared in a [`filter!`](crate::filter) block.
///
/// Implemented by the generated `state`, observation and control value types, so items
/// declared `for` a state find its scalar without the user repeating it.
pub trait Named {
    /// The scalar of every entry, e.g. `f32`.
    type Scalar;
}

/// The largest value of a list, `0` for an empty one: the `MAX_OBSERVATIONS` of a
/// `scratch` item.
#[must_use]
pub const fn max(values: &[usize]) -> usize {
    let mut max = 0;
    let mut i = 0;
    while i < values.len() {
        if values[i] > max {
            max = values[i];
        }

        i += 1;
    }

    max
}

/// A `ROWS × COLS` matrix of `TOTAL` entries held by value, every entry `zero`.
type Array<const ROWS: usize, const COLS: usize, const TOTAL: usize, T> =
    MatrixDataArray<ROWS, COLS, TOTAL, T>;

/// Creates a zeroed [`Array`]; the callers check `TOTAL == ROWS · COLS` at compile time.
fn array<const ROWS: usize, const COLS: usize, const TOTAL: usize, T: Copy>(
    zero: T,
) -> Array<ROWS, COLS, TOTAL, T> {
    MatrixDataArray::new_unchecked([zero; TOTAL])
}

/// A [`RegularKalman`] whose buffers it owns by value: the `Owned` of a `storage = owned`
/// `kalman` item. `SS` is `S²`.
pub type OwnedKalman<const S: usize, const SS: usize, T> = RegularKalman<
    S,
    T,
    StateTransitionMatrixMutBuffer<S, T, Array<S, S, SS, T>>,
    StateVectorBuffer<S, T, Array<S, 1, S, T>>,
    EstimateCovarianceMatrixBuffer<S, T, Array<S, S, SS, T>>,
    DirectProcessNoiseCovarianceMatrixMutBuffer<S, T, Array<S, S, SS, T>>,
    PredictedStateEstimateVectorBuffer<S, T, Array<S, 1, S, T>>,
    TemporaryStateMatrixBuffer<S, T, Array<S, S, SS, T>>,
>;

/// Creates a zeroed [`OwnedKalman`] by value.
#[must_use]
pub fn owned_kalman<const S: usize, const SS: usize, T: MatrixDataType>(
    zero: T,
) -> OwnedKalman<S, SS, T> {
    const { assert!(SS == S * S, "SS must be S²") };
    RegularKalmanBuilder::new::<S, T>(
        StateTransitionMatrixMutBuffer::new(array(zero)),
        StateVectorBuffer::new(array(zero)),
        EstimateCovarianceMatrixBuffer::new(array(zero)),
        DirectProcessNoiseCovarianceMatrixMutBuffer::new(array(zero)),
        PredictedStateEstimateVectorBuffer::new(array(zero)),
        TemporaryStateMatrixBuffer::new(array(zero)),
    )
}

/// A full [`RegularObservation`] whose buffers it owns by value: the `Owned` of a
/// `storage = owned` `observation` item. `OS` is `O·S`, `OO` is `O²` and `SS` is `S²`.
pub type OwnedObservation<
    const S: usize,
    const O: usize,
    const OS: usize,
    const OO: usize,
    const SS: usize,
    T,
> = RegularObservation<
    S,
    O,
    T,
    ObservationMatrixMutBuffer<O, S, T, Array<O, S, OS, T>>,
    MeasurementVectorBuffer<O, T, Array<O, 1, O, T>>,
    MeasurementNoiseCovarianceMatrixBuffer<O, T, Array<O, O, OO, T>>,
    InnovationVectorBuffer<O, T, Array<O, 1, O, T>>,
    InnovationCovarianceMatrixBuffer<O, T, Array<O, O, OO, T>>,
    KalmanGainMatrixBuffer<S, O, T, Array<S, O, OS, T>>,
    TemporaryResidualCovarianceInvertedMatrixBuffer<O, T, Array<O, O, OO, T>>,
    TemporaryHPMatrixBuffer<O, S, T, Array<O, S, OS, T>>,
    TemporaryPHTMatrixBuffer<S, O, T, Array<S, O, OS, T>>,
    TemporaryKHPMatrixBuffer<S, T, Array<S, S, SS, T>>,
>;

/// Creates a zeroed [`OwnedObservation`] by value.
#[must_use]
pub fn owned_observation<
    const S: usize,
    const O: usize,
    const OS: usize,
    const OO: usize,
    const SS: usize,
    T: MatrixDataType,
>(
    zero: T,
) -> OwnedObservation<S, O, OS, OO, SS, T> {
    const {
        assert!(
            OS == O * S && OO == O * O && SS == S * S,
            "wrong derived sizes"
        )
    };
    RegularObservationBuilder::new::<S, O, T>(
        ObservationMatrixMutBuffer::new(array(zero)),
        MeasurementVectorBuffer::new(array(zero)),
        MeasurementNoiseCovarianceMatrixBuffer::new(array(zero)),
        InnovationVectorBuffer::new(array(zero)),
        InnovationCovarianceMatrixBuffer::new(array(zero)),
        KalmanGainMatrixBuffer::new(array(zero)),
        TemporaryResidualCovarianceInvertedMatrixBuffer::new(array(zero)),
        TemporaryHPMatrixBuffer::new(array(zero)),
        TemporaryPHTMatrixBuffer::new(array(zero)),
        TemporaryKHPMatrixBuffer::new(array(zero)),
    )
}

/// A [`Control`] whose buffers it owns by value: the `Owned` of a `storage = owned`
/// `control` item. `SC` is `S·C` and `CC` is `C²`.
pub type OwnedControl<const S: usize, const C: usize, const SC: usize, const CC: usize, T> =
    Control<
        S,
        C,
        T,
        ControlMatrixMutBuffer<S, C, T, Array<S, C, SC, T>>,
        ControlVectorBuffer<C, T, Array<C, 1, C, T>>,
        ControlProcessNoiseCovarianceMatrixMutBuffer<C, T, Array<C, C, CC, T>>,
        TemporaryBQMatrixBuffer<S, C, T, Array<S, C, SC, T>>,
    >;

/// Creates a zeroed [`OwnedControl`] by value.
#[must_use]
pub fn owned_control<
    const S: usize,
    const C: usize,
    const SC: usize,
    const CC: usize,
    T: MatrixDataType,
>(
    zero: T,
) -> OwnedControl<S, C, SC, CC, T> {
    const { assert!(SC == S * C && CC == C * C, "wrong derived sizes") };
    ControlBuilder::new::<S, C, T>(
        ControlMatrixMutBuffer::new(array(zero)),
        ControlVectorBuffer::new(array(zero)),
        ControlProcessNoiseCovarianceMatrixMutBuffer::new(array(zero)),
        TemporaryBQMatrixBuffer::new(array(zero)),
    )
}

#[cfg(test)]
mod tests {
    use super::max;

    #[test]
    fn max_of_a_list() {
        assert_eq!(max(&[]), 0);
        assert_eq!(max(&[3]), 3);
        assert_eq!(max(&[1, 4, 2]), 4);
    }
}
