//! Pseudo-additive arithmetic from Build 62 `x11pt1.f` to `x11pt3.f`, where
//! the series is `trend * (seasonal + irregular - 1)`. Results are checked,
//! and a failure names its stage and date.

use crate::Error;
use crate::error::{
    NumericalDetail, X11PseudoAdditiveStage, X11PseudoAdditiveUnavailableReason as Reason,
};
use crate::time::AxisSpan;
use std::ops::Range;

/// The pseudo-additive error for observation `index` of `dates`.
pub(crate) fn failure(
    dates: AxisSpan,
    index: usize,
    stage: X11PseudoAdditiveStage,
    reason: Reason,
) -> Error {
    Error::from(NumericalDetail::X11PseudoAdditive {
        stage,
        index,
        period: dates.period(index),
        reason,
    })
}

/// `value` if it is finite, otherwise a non-finite-value error.
pub(crate) fn checked(
    value: f64,
    dates: AxisSpan,
    index: usize,
    stage: X11PseudoAdditiveStage,
) -> Result<f64, Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(failure(dates, index, stage, Reason::NonFiniteValue))
    }
}

/// `numerator / denominator`. A zero denominator is an error, as is a
/// non-finite operand or quotient.
pub(crate) fn divide(
    numerator: f64,
    denominator: f64,
    dates: AxisSpan,
    index: usize,
    stage: X11PseudoAdditiveStage,
) -> Result<f64, Error> {
    checked(numerator, dates, index, stage)?;
    checked(denominator, dates, index, stage)?;
    if denominator == 0.0 {
        return Err(failure(dates, index, stage, Reason::ZeroDivisor));
    }
    checked(numerator / denominator, dates, index, stage)
}

/// The seasonally adjusted value `original - trend * (seasonal - 1)`.
pub(crate) fn adjusted(original: f64, trend: f64, seasonal: f64) -> f64 {
    // gfortran rounds `seasonal - 1`, then fuses the multiply-subtract, at
    // every such statement in x11pt2.f and x11pt3.f.
    (-trend).mul_add(seasonal - 1.0, original)
}

/// The value rebuilt with its irregular divided by the extreme-value
/// factor: `trend * (seasonal + irregular / extreme - 1)`.
pub(crate) fn modified(trend: f64, seasonal: f64, irregular: f64, extreme: f64) -> f64 {
    trend * (seasonal + ((irregular / extreme) - 1.0))
}

/// The first seasonal adjustment of a pass. Inside `central`, where the
/// centered trend exists, it is [`adjusted`]. At the ends there is no trend
/// yet, so the value is divided by its seasonal factor; a factor within Build
/// 62's zero tolerance is a zero-divisor error.
pub(crate) fn initial_adjusted(
    modified: &[f64],
    trend: &[f64],
    seasonal: &[f64],
    central: Range<usize>,
    dates: AxisSpan,
) -> Result<Vec<f64>, Error> {
    modified
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let stage = X11PseudoAdditiveStage::InitialSeasonalAdjustment;
            if central.contains(&i) {
                checked(adjusted(value, trend[i], seasonal[i]), dates, i, stage)
            } else {
                if crate::numerics::build_62_equal(seasonal[i], 0.0) {
                    return Err(failure(dates, i, stage, Reason::ZeroDivisor));
                }
                divide(value, seasonal[i], dates, i, stage)
            }
        })
        .collect()
}

/// [`adjusted`] for every observation; a non-finite result is an error.
pub(crate) fn adjusted_series(
    original: &[f64],
    trend: &[f64],
    seasonal: &[f64],
    dates: AxisSpan,
    stage: X11PseudoAdditiveStage,
) -> Result<Vec<f64>, Error> {
    original
        .iter()
        .enumerate()
        .map(|(i, &v)| checked(adjusted(v, trend[i], seasonal[i]), dates, i, stage))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NumericalFailure;
    use crate::time::TimeAxis;

    #[test]
    fn adjustment_fuses_the_multiply_subtract() {
        // (1 + 2^-52) * (1 - 2^-52) rounds to one when multiplied on its
        // own; the fused form keeps the exact 2^-104 difference.
        let trend = f64::from_bits(1.0_f64.to_bits() + 1);
        let seasonal = f64::from_bits(2.0_f64.to_bits() - 1);
        assert_eq!(adjusted(1.0, trend, seasonal).to_bits(), 919_u64 << 52);
        assert_eq!((1.0 - trend * (seasonal - 1.0)).to_bits(), 0);
    }

    #[test]
    fn initial_adjustment_uses_trend_only_on_centered_span() {
        let dates = TimeAxis::quarterly(1999, 3).unwrap().span(4).unwrap();
        assert_eq!(
            initial_adjusted(&[12.0; 4], &[0.0, 8.0, 10.0, 0.0], &[1.5; 4], 1..3, dates).unwrap(),
            vec![8.0, 8.0, 7.0, 8.0]
        );
        assert_eq!(adjusted(12.0, 10.0, 1.5), 7.0);
        assert_eq!(modified(10.0, 1.5, 0.8, 2.0), 9.0);
    }

    #[test]
    fn division_rejects_zero_divisors_and_nonfinite_values() {
        let dates = TimeAxis::monthly(1999, 11).unwrap().span(4).unwrap();
        let stage = X11PseudoAdditiveStage::Irregular;
        assert_eq!(divide(-2.0, -4.0, dates, 3, stage).unwrap(), 0.5);
        assert_eq!(divide(0.0, -4.0, dates, 3, stage).unwrap(), -0.0);
        assert_eq!(
            divide(1.0, 0.0, dates, 3, stage),
            Err(Error::from(NumericalDetail::X11PseudoAdditive {
                stage,
                index: 3,
                period: dates.period(3),
                reason: Reason::ZeroDivisor
            }))
        );
        assert!(matches!(
            divide(f64::INFINITY, 2.0, dates, 0, stage),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11PseudoAdditive {
                    reason: Reason::NonFiniteValue,
                    ..
                }
            )))
        ));
        assert!(matches!(
            divide(f64::MAX, 0.5, dates, 0, stage),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11PseudoAdditive {
                    reason: Reason::NonFiniteValue,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn end_divisor_within_zero_tolerance_is_an_error() {
        let dates = TimeAxis::quarterly(1998, 4).unwrap().span(5).unwrap();
        let error = initial_adjusted(
            &[1.0; 5],
            &[1.0; 5],
            &[1.0, 1.0, 1.0, 1.0, 1e-21],
            1..4,
            dates,
        )
        .unwrap_err();
        assert_eq!(
            error,
            Error::from(NumericalDetail::X11PseudoAdditive {
                stage: X11PseudoAdditiveStage::InitialSeasonalAdjustment,
                index: 4,
                period: dates.period(4),
                reason: Reason::ZeroDivisor
            })
        );
        // Plain division rejects only an exact zero.
        assert!(divide(1e-16, 1e-16, dates, 4, X11PseudoAdditiveStage::Irregular).is_ok());
    }
}
