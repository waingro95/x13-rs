//! The centered moving-average trend and `vtc.f`'s variable Henderson trend.

use std::ops::Range;

use super::Pass;
use crate::error::{
    NumericalDetail, X11PseudoAdditiveStage as PseudoStage, X11TrendUnavailableReason,
};
use crate::numerics::build_62_equal;
use crate::time::AxisSpan;
use crate::x11::pseudo_additive as pseudo;
use crate::{Error, numerics, x11};

/// The centered `period`-term moving average of `values`, and the span it
/// covers.
pub(super) fn centered_period_trend(values: &[f64], period: usize) -> (Vec<f64>, Range<usize>) {
    let mut trend = vec![0.0; values.len()];
    numerics::moving_average(values, &mut trend, 0..values.len(), 2, period);
    let half = period / 2;
    (trend, half..values.len() - half)
}

/// A Henderson trend from `variable_trend`.
pub(super) struct TrendEstimate {
    pub(super) values: Vec<f64>,
    /// The I/C ratio before `vtc.f` scales it to a monthly rate.
    pub(super) unscaled_ratio: f64,
    /// The Henderson length used.
    pub(super) terms: usize,
    /// The endpoint I/C ratio the next trend estimate starts from.
    pub(super) trend_ic: f64,
    pub(super) nonpositive_replacements: usize,
    #[cfg(feature = "__conditioning-scales")]
    pub(super) operation_scales: Vec<f64>,
}

/// `editor.f`: the I/C ratio the first Henderson endpoints use.
pub(super) fn initial_trend_ic(period: usize, filter: x11::TrendFilter) -> f64 {
    if let x11::TrendFilter::HendersonWithIcRatio { ratio, .. } = filter
        && !build_62_equal(ratio.value(), 0.0)
    {
        return ratio.value();
    }
    let terms = filter.fixed_length().map_or(0, |length| length.terms());
    let mut trend_ic = 3.5;
    if terms > 0 && terms <= 9 {
        trend_ic = 1.0;
    }
    if terms > 13 {
        trend_ic = 4.5;
    }
    if period == 4 {
        if terms <= 5 {
            trend_ic = 0.001;
        } else if terms >= 7 {
            trend_ic = 4.5;
        }
    }
    trend_ic
}

/// `vtc.f`'s I/C ratio: 999 when the trend does not change.
pub(super) fn unscaled_trend_ratio(irregular_change: f64, trend_change: f64) -> f64 {
    if build_62_equal(trend_change, 0.0) {
        999.0
    } else {
        irregular_change / trend_change
    }
}

/// `vtc.f`: a base Henderson trend, its I/C ratio, then the
/// Henderson length that ratio selects. `trend_ic` is the endpoint ratio the
/// pass inherits; the estimate carries the one the next pass inherits.
pub(super) fn variable_trend(
    values: &[f64],
    period: usize,
    pass: Pass,
    trend_ic: f64,
    mode: x11::Mode,
    observed: Range<usize>,
    options: x11::Options,
) -> Result<TrendEstimate, VariableTrendFailure> {
    let base_terms = period + 1;
    let policy = options.seven_term_endpoints();
    let (initial, trend_ic) = numerics::henderson_filter(values, base_terms, trend_ic, policy);
    let half = base_terms / 2;
    if mode == x11::Mode::PseudoAdditive {
        // `vtc.f` computes `Temp` through `Posffc - half`, although the I/C
        // ratio leaves the forecasts out, so every division there is checked.
        for index in half..values.len() - half {
            pseudo_trend_operand(values[index], index, false)?;
            pseudo_trend_operand(initial[index], index, true)?;
            pseudo_trend_operand(values[index] / initial[index], index, false)?;
        }
    }
    let mut irregular_change = 0.0;
    let mut trend_change = 0.0;
    for index in half..observed.end - half - 1 {
        if mode == x11::Mode::PseudoAdditive {
            for i in [index, index + 1] {
                pseudo_trend_operand(values[i], i, false)?;
                pseudo_trend_operand(initial[i], i, true)?;
            }
        }
        let current_irregular = mode.remove(values[index], initial[index]);
        let next_irregular = mode.remove(values[index + 1], initial[index + 1]);
        if mode == x11::Mode::PseudoAdditive {
            pseudo_trend_operand(current_irregular, index, true)?;
            pseudo_trend_operand(next_irregular, index + 1, false)?;
        }
        trend_change += mode.change(initial[index], initial[index + 1]);
        irregular_change += mode.change(current_irregular, next_irregular);
        if mode == x11::Mode::PseudoAdditive {
            pseudo_trend_operand(trend_change, index, false)?;
            pseudo_trend_operand(irregular_change, index, false)?;
        }
    }
    let unscaled_ratio = unscaled_trend_ratio(irregular_change, trend_change);
    let ratio = if build_62_equal(trend_change, 0.0) {
        999.0
    } else {
        irregular_change / trend_change * 12.0 / period as f64
    };
    if mode == x11::Mode::PseudoAdditive {
        pseudo_trend_operand(ratio, half, false)?;
    }

    // The selected length keeps the inherited I/C unless it sets its own.
    let (terms, trend_ic) = if let Some(length) = options.trend_filter().fixed_length() {
        (usize::from(length.terms()), trend_ic)
    } else if ratio >= 1.0 && (pass == Pass::B || ratio < 3.5) {
        (base_terms, trend_ic)
    } else if ratio < 1.0 {
        if period == 12 {
            (9, 1.0)
        } else {
            (base_terms, trend_ic)
        }
    } else if period == 4 {
        (7, 4.5)
    } else {
        (23, 4.5)
    };
    let (mut trend, trend_ic) = numerics::henderson_filter(values, terms, trend_ic, policy);
    #[cfg(feature = "__conditioning-scales")]
    let mut operation_scales =
        numerics::henderson_filter_operation_scales(values, terms, trend_ic, policy);
    let nonpositive_replacements =
        if matches!(mode, x11::Mode::Multiplicative | x11::Mode::PseudoAdditive) {
            #[cfg(feature = "__conditioning-scales")]
            {
                replace_nonpositive_trend(&mut trend, observed, Some(&mut operation_scales))?
            }
            #[cfg(not(feature = "__conditioning-scales"))]
            {
                replace_nonpositive_trend(&mut trend, observed, None)?
            }
        } else {
            0
        };
    Ok(TrendEstimate {
        unscaled_ratio,
        values: trend,
        terms,
        trend_ic,
        nonpositive_replacements,
        #[cfg(feature = "__conditioning-scales")]
        operation_scales,
    })
}

/// Where and why a trend estimate is unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TrendFailure {
    pub(super) index: usize,
    pub(super) reason: X11TrendUnavailableReason,
}

/// Why `variable_trend` failed: the trend itself, or a pseudo-additive
/// division on the way to it.
#[derive(Debug)]
pub(super) enum VariableTrendFailure {
    Trend(TrendFailure),
    Pseudo {
        index: usize,
        reason: crate::error::X11PseudoAdditiveUnavailableReason,
    },
}

impl From<TrendFailure> for VariableTrendFailure {
    fn from(value: TrendFailure) -> Self {
        Self::Trend(value)
    }
}

/// Check one operand of a pseudo-additive division in `vtc.f`.
fn pseudo_trend_operand(
    value: f64,
    index: usize,
    divisor: bool,
) -> Result<(), VariableTrendFailure> {
    use crate::error::X11PseudoAdditiveUnavailableReason as Reason;
    let reason = if !value.is_finite() {
        Some(Reason::NonFiniteValue)
    } else if divisor && value == 0.0 {
        Some(Reason::ZeroDivisor)
    } else {
        None
    };
    match reason {
        Some(reason) => Err(VariableTrendFailure::Pseudo { index, reason }),
        None => Ok(()),
    }
}

/// The error for a trend failure at `stage`.
pub(super) fn trend_error(
    dates: AxisSpan,
    stage: x11::TrendStage,
    failure: impl Into<VariableTrendFailure>,
) -> Error {
    let failure = match failure.into() {
        VariableTrendFailure::Trend(failure) => failure,
        VariableTrendFailure::Pseudo { index, reason } => {
            return pseudo::failure(dates, index, PseudoStage::TrendSelection, reason);
        }
    };
    Error::from(NumericalDetail::X11Trend {
        stage,
        period: dates.period(failure.index),
        reason: failure.reason,
    })
}

/// `chktrn.f`: replace each nonpositive trend value, in order, with the mean
/// of the nearest positive values on either side within `observed`, or with
/// the one that exists. Returns the number replaced.
pub(super) fn replace_nonpositive_trend(
    trend: &mut [f64],
    observed: Range<usize>,
    mut operation_scales: Option<&mut [f64]>,
) -> Result<usize, TrendFailure> {
    let mut replacements = 0;
    for index in 0..trend.len() {
        if !trend[index].is_finite() {
            return Err(TrendFailure {
                index,
                reason: X11TrendUnavailableReason::NonFiniteEstimate,
            });
        }
        if trend[index] > 0.0 {
            continue;
        }
        replacements += 1;

        let before = (observed.start..index)
            .rev()
            .find(|&candidate| trend[candidate] > 0.0);
        let after = (index + 1..observed.end).find(|&candidate| trend[candidate] > 0.0);
        let replacement = match (before, after) {
            (Some(before), Some(after)) => {
                let replacement = (trend[before] + trend[after]) / 2.0;
                if let Some(scales) = operation_scales.as_deref_mut() {
                    scales[index] = ((scales[before]
                        + scales[after]
                        + trend[before].abs()
                        + trend[after].abs())
                        / 2.0)
                        .max(1.0);
                }
                replacement
            }
            (Some(before), None) => {
                if let Some(scales) = operation_scales.as_deref_mut() {
                    scales[index] = scales[before];
                }
                trend[before]
            }
            (None, Some(after)) => {
                if let Some(scales) = operation_scales.as_deref_mut() {
                    scales[index] = scales[after];
                }
                trend[after]
            }
            (None, None) => {
                return Err(TrendFailure {
                    index,
                    reason: X11TrendUnavailableReason::NoPositiveNeighbor,
                });
            }
        };
        if !replacement.is_finite() {
            return Err(TrendFailure {
                index,
                reason: X11TrendUnavailableReason::NonFiniteReplacement,
            });
        }
        trend[index] = replacement;
    }
    Ok(replacements)
}
