//! AR roots near the unit circle, per factor: Build 62's count of them and
//! its largest real reciprocal modulus among the rest. Differencing
//! identification and the final root check both read these.

use crate::arima::roots::{DetailedRoots, RootComputationError, RootDetails, detailed_roots};

/// What stands for the largest reciprocal modulus of a factor that has no
/// positive, nearly real root outside the unit band.
pub(super) const NO_MAXIMUM: f64 = -999.0;

/// One value per ARIMA factor: regular, then seasonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ByFactor<T> {
    pub(super) regular: T,
    pub(super) seasonal: T,
}

impl<T> ByFactor<T> {
    pub(super) const fn new(regular: T, seasonal: T) -> Self {
        Self { regular, seasonal }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FactorDecision {
    pub(super) unit_count: u8,
    pub(super) maximum: f64,
    pub(super) all_invertible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RootDecision {
    pub(super) unit_counts: ByFactor<u8>,
    /// [`NO_MAXIMUM`] for a factor without a qualifying root.
    pub(super) maxima: ByFactor<f64>,
    pub(super) all_invertible: bool,
}

/// A positive root within `unit_limit` of the origin whose imaginary part is
/// at most 0.05. The imaginary part is compared signed, as Build 62 does.
fn is_unit_root(root: &RootDetails, unit_limit: f64) -> bool {
    root.modulus <= unit_limit && root.imaginary <= 0.05 && root.real > 0.0
}

pub(super) fn factor_decision(details: &DetailedRoots, unit_limit: f64) -> FactorDecision {
    let mut unit_count = 0;
    let mut maximum = NO_MAXIMUM;
    for root in &details.roots {
        if is_unit_root(root, unit_limit) {
            unit_count += 1;
        } else if root.imaginary <= 0.02 && root.real > 0.0 {
            let reciprocal = 1.0 / root.modulus;
            if reciprocal > maximum {
                maximum = reciprocal;
            }
        }
    }
    FactorDecision {
        unit_count,
        maximum,
        all_invertible: details.all_outside_or_on_unit_circle,
    }
}

/// The decisions for the regular and the seasonal AR factor. An empty factor
/// has no roots: no unit count, [`NO_MAXIMUM`] and invertible.
pub(super) fn check_ar_roots(
    regular: &[f64],
    seasonal: &[f64],
    unit_limit: f64,
) -> Result<RootDecision, RootComputationError> {
    let regular = factor_decision(&detailed_roots(regular)?, unit_limit);
    let seasonal = factor_decision(&detailed_roots(seasonal)?, unit_limit);
    Ok(RootDecision {
        unit_counts: ByFactor::new(regular.unit_count, seasonal.unit_count),
        maxima: ByFactor::new(regular.maximum, seasonal.maximum),
        all_invertible: regular.all_invertible && seasonal.all_invertible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arima::roots::{DetailedRoots, RootDetails};

    fn detail(real: f64, imaginary: f64, modulus: f64) -> RootDetails {
        RootDetails {
            real,
            imaginary,
            modulus,
            frequency: 0.0,
        }
    }

    #[test]
    fn signed_imaginary_and_exact_thresholds() {
        let limit: f64 = 1.0 / 0.96;
        let roots = DetailedRoots {
            roots: vec![
                detail(limit, 0.05, limit),
                detail(limit, -0.05, limit),
                detail(limit, f64::from_bits(0.05f64.to_bits() + 1), limit),
                detail(1.2, 0.02, 1.2),
                detail(1.3, -0.4, 1.3),
                detail(-1.0, 0.0, 1.0),
            ],
            all_outside_or_on_unit_circle: true,
        };
        let result = factor_decision(&roots, limit);
        assert_eq!(result.unit_count, 2);
        assert_eq!(result.maximum, 1.0 / 1.2);
        assert!(result.all_invertible);
        let no_maximum = DetailedRoots {
            roots: vec![detail(-1.0, 0.0, 1.0)],
            all_outside_or_on_unit_circle: true,
        };
        assert_eq!(factor_decision(&no_maximum, limit).maximum, -999.0);
    }

    #[test]
    fn linear_boundaries_and_all_factor_invertibility() {
        let limit: f64 = 1.0 / 0.96;
        let below = f64::from_bits(limit.to_bits() - 1);
        let above = f64::from_bits(limit.to_bits() + 1);
        assert_eq!(
            check_ar_roots(&[1.0 / below], &[0.0], limit)
                .unwrap()
                .unit_counts,
            ByFactor::new(1, 0)
        );
        assert_eq!(
            check_ar_roots(&[1.0 / limit], &[], limit)
                .unwrap()
                .unit_counts,
            ByFactor::new(1, 0)
        );
        assert_eq!(
            check_ar_roots(&[1.0 / above], &[], limit)
                .unwrap()
                .unit_counts,
            ByFactor::new(0, 0)
        );
        let result = check_ar_roots(&[0.5], &[2.0], limit).unwrap();
        assert_eq!(result.unit_counts, ByFactor::new(0, 1));
        assert!(!result.all_invertible);
        let maxima = check_ar_roots(&[0.0, 0.0], &[-0.5], limit).unwrap().maxima;
        assert_eq!(
            (maxima.regular.to_bits(), maxima.seasonal.to_bits()),
            ((-999.0_f64).to_bits(), (-999.0_f64).to_bits())
        );
    }

    #[test]
    fn an_empty_factor_has_no_maximum_and_stays_invertible() {
        let limit: f64 = 1.0 / 0.96;
        let result = check_ar_roots(&[], &[], limit).unwrap();
        assert_eq!(result.unit_counts, ByFactor::new(0, 0));
        assert_eq!(result.maxima.regular.to_bits(), (-999.0_f64).to_bits());
        assert_eq!(result.maxima.seasonal.to_bits(), (-999.0_f64).to_bits());
        assert!(result.all_invertible);
        let explosive = check_ar_roots(&[2.0], &[], limit).unwrap();
        assert!(!explosive.all_invertible);
        assert_eq!(explosive.maxima.seasonal.to_bits(), (-999.0_f64).to_bits());
    }
}
