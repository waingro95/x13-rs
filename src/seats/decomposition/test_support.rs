//! Helpers shared by the decomposition tests.

use crate::Frequency;
use crate::seats::airline::model::ResolvedAirline;

/// The airline model `(1 - ma[0] B)(1 - ma[1] B^s)`.
pub(super) fn airline(frequency: Frequency, ma: [f64; 2]) -> ResolvedAirline {
    ResolvedAirline { frequency, ma }
}

/// Asserts each value is within a relative 6e-15 of its expected one, and is
/// a positive zero where the expected one is zero.
pub(super) fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (lag, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "nonfinite lag {lag}");
        if expected == 0.0 {
            assert_eq!(actual.to_bits(), 0.0_f64.to_bits(), "zero lag {lag}");
        } else {
            let relative = (actual - expected).abs() / expected.abs();
            assert!(
                relative <= 6e-15,
                "lag {lag}: {actual:.17e} != {expected:.17e}, relative {relative:.17e}"
            );
        }
    }
}
