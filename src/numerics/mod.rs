//! Numerical routines shared across the crate, translated from Build 62:
//! packed linear algebra, the Levenberg-Marquardt solver, moving-average
//! filters, lag polynomials, autocorrelations, spectra and probabilities.

pub(crate) mod autocorrelation;
pub(crate) mod blas1;
mod build62;
mod filters;
mod matrix;
pub(crate) mod minpack;
mod order;
pub(crate) mod packed;
pub(crate) mod polynomial;
pub(crate) mod probability;
pub(crate) mod spectral;

pub(crate) use build62::{
    BUILD_62_MACHINE_EPSILON, BUILD_62_MAD_SCALE, BUILD_62_MIN_POSITIVE, BUILD_62_PI,
};
#[cfg(feature = "__conditioning-scales")]
pub(crate) use filters::henderson_filter_operation_scales;
pub(crate) use filters::{henderson_filter, moving_average, s3x5_subseries};
pub(crate) use matrix::RowMajorMatrix;
pub(crate) use order::shell_sort;

const BUILD_62_EQUALITY_DELTA: f64 = 3.834e-20;

/// Build 62 `dpeq.f`: absolute comparison used at numerical branch points.
pub(crate) fn build_62_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < BUILD_62_EQUALITY_DELTA
}

/// Build 62's `Same` flag: the series' maximum and minimum are equal under
/// `dpeq.f`. Compute it from the original series, never from an adjusted
/// component.
pub(crate) fn is_constant_series(values: &[f64]) -> bool {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    build_62_equal(maximum, minimum)
}

/// Build 62's `Same` flag on a series with gaps, which are NaN here. `editor.f`
/// takes the extremes over the observed values, but seeds them from the first
/// value before it replaces the missing code, so a leading gap is never `Same`.
pub(crate) fn is_constant_with_gaps(values: &[f64]) -> bool {
    values.iter().any(|value| value.is_nan())
        && values.first().is_some_and(|value| !value.is_nan())
        && is_constant_series(values)
}

/// The index of the first value at or below zero: the input-domain check of
/// the multiplicative X-11 modes and the log transform. NaN compares false
/// and never counts here.
pub(crate) fn first_nonpositive(values: &[f64]) -> Option<usize> {
    values.iter().position(|value| *value <= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAP: f64 = f64::NAN;

    #[test]
    fn constant_with_gaps_needs_a_gap_and_equal_observed_values() {
        assert!(is_constant_with_gaps(&[5.0, GAP, 5.0]));
        assert!(is_constant_with_gaps(&[5.0, 5.0, GAP]));
        assert!(is_constant_with_gaps(&[0.0, GAP, 0.0]));
        assert!(is_constant_with_gaps(&[1e-200, GAP, 2e-200]));
        assert!(!is_constant_with_gaps(&[5.0, 5.0, 5.0]));
        assert!(!is_constant_with_gaps(&[5.0, GAP, 6.0]));
        assert!(!is_constant_with_gaps(&[5.0, GAP, 5.0 + 1e-14]));
    }

    #[test]
    fn a_leading_gap_is_never_constant() {
        assert!(!is_constant_with_gaps(&[GAP, 5.0, 5.0]));
        assert!(!is_constant_with_gaps(&[GAP, GAP, 5.0]));
        assert!(!is_constant_with_gaps(&[GAP, GAP]));
        assert!(!is_constant_with_gaps(&[]));
    }
}
