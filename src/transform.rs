//! The log transformation and its inverse.

use crate::{Error, Transform};

/// A point value and its interval bounds, on the original scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InverseInterval {
    pub(crate) point: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

/// Transforms `values`, rejecting the first nonpositive value under
/// [`Transform::Log`].
pub(crate) fn forward(values: &[f64], transform: Transform) -> Result<Vec<f64>, Error> {
    let mut transformed = Vec::with_capacity(values.len());
    for (index, value) in values.iter().copied().enumerate() {
        match transform {
            Transform::None => transformed.push(value),
            Transform::Log if value <= 0.0 => {
                return Err(Error::NonPositiveValue { index });
            }
            Transform::Log => transformed.push(value.ln()),
        }
    }
    Ok(transformed)
}

/// Returns a transformed `value` to the original scale.
pub(crate) fn inverse(transform: Transform, value: f64) -> f64 {
    match transform {
        Transform::None => value,
        Transform::Log => value.exp(),
    }
}

/// Builds the interval `point ± critical_value * standard_error` on the
/// transformed scale and returns its three values on the original scale.
pub(crate) fn inverse_interval(
    transformed_point: f64,
    transformed_standard_error: f64,
    critical_value: f64,
    transform: Transform,
) -> InverseInterval {
    let scaled_error = critical_value * transformed_standard_error;
    InverseInterval {
        point: inverse(transform, transformed_point),
        lower: inverse(transform, transformed_point - scaled_error),
        upper: inverse(transform, transformed_point + scaled_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_forward_transform_rejects_the_first_nonpositive_value() {
        assert_eq!(
            forward(&[2.0, 0.0, -1.0], Transform::Log),
            Err(Error::NonPositiveValue { index: 1 })
        );
    }

    #[test]
    fn identity_inverse_interval_stays_on_the_transformed_scale() {
        assert_eq!(inverse(Transform::None, 10.0), 10.0);
        assert_eq!(
            inverse_interval(10.0, 2.0, 1.5, Transform::None),
            InverseInterval {
                point: 10.0,
                lower: 7.0,
                upper: 13.0,
            }
        );
    }
}
