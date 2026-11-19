//! Seasonal factors of one SI: the seasonal filters, their centering and
//! their extension to the ends of the series.

use std::ops::Range;

use super::extreme_scale::{ExtremeScaleState, RobustContext};
use super::extreme_weights::{ExtremeSpans, calendar_extreme_weights, replace_extremes};
use super::mode::remove_on_span;
use crate::error::X11PseudoAdditiveStage as PseudoStage;
use crate::time::{AxisSpan, TimeAxis};
use crate::x11::pseudo_additive as pseudo;
use crate::x11::seasonal::{ResolvedSeasonalFilter, SeasonalFilterSet};
use crate::{Error, numerics, x11};

/// One section's resolved seasonal filters and the short-series rule.
#[derive(Clone, Copy)]
pub(super) struct SeasonalSmoothingRequest<'a> {
    pub(super) filters: &'a SeasonalFilterSet<ResolvedSeasonalFilter>,
    pub(super) short_series_filters: x11::ShortSeriesFilters,
}

/// What `replace_extremes_and_filter` takes besides the SI and its spans.
pub(super) struct SeasonalWeightRequest<'a> {
    pub(super) smoothing: SeasonalSmoothingRequest<'a>,
    pub(super) robust: Option<RobustContext>,
    pub(super) dates: AxisSpan,
}

/// Check that every calendar period has the years of SI its first-section
/// filter needs, unless short series fall back to the stable filter.
pub(super) fn validate_initial_seasonal_history(
    length: usize,
    axis: TimeAxis,
    requested: &SeasonalFilterSet<x11::SeasonalFilter>,
    smoothing: SeasonalSmoothingRequest<'_>,
) -> Result<(), Error> {
    if smoothing.short_series_filters == x11::ShortSeriesFilters::StableFallback {
        return Ok(());
    }
    let frequency = axis.frequency();
    let period = usize::from(frequency.periods_per_year());
    let start_subperiod = usize::from(axis.start().subperiod());
    let span = period / 2..length - period / 2;
    for first in span.start..span.start + period {
        let calendar_index = (start_subperiod - 1 + first) % period;
        let resolved = smoothing.filters[calendar_index];
        let minimum = match resolved {
            ResolvedSeasonalFilter::S3x1 => 2,
            ResolvedSeasonalFilter::S3x3 | ResolvedSeasonalFilter::S3x5 => 3,
            ResolvedSeasonalFilter::S3x9
            | ResolvedSeasonalFilter::S3x15
            | ResolvedSeasonalFilter::Stable => continue,
        };
        let actual = (span.end - first).div_ceil(period);
        if actual < minimum {
            return Err(Error::InsufficientSeasonalHistory {
                frequency,
                calendar_period: (calendar_index + 1) as u8,
                requested: requested[calendar_index],
                resolved: resolved.into(),
                actual,
                minimum,
            });
        }
    }
    Ok(())
}

/// Seasonal factors from `seasonal_filter`.
pub(super) struct SeasonalSmoothing {
    pub(super) values: Vec<f64>,
    /// The filter the last calendar period was smoothed with.
    pub(super) last_filter: ResolvedSeasonalFilter,
}

/// `vsfb.f` and `vsfc.f`: smooth each calendar period's SI over `span` with
/// its seasonal filter, then center the factors.
pub(super) fn seasonal_filter(
    si: &[f64],
    span: Range<usize>,
    mode: x11::Mode,
    request: SeasonalSmoothingRequest<'_>,
    dates: AxisSpan,
) -> Result<SeasonalSmoothing, Error> {
    let period = usize::from(dates.start().frequency().periods_per_year());
    let start_subperiod = usize::from(dates.start().subperiod());
    let mut seasonal = vec![0.0; si.len()];
    let stable = span.len() < 5 * period
        && request.short_series_filters == x11::ShortSeriesFilters::StableFallback;
    let mut last_filter = ResolvedSeasonalFilter::Stable;
    for first in span.start..span.start + period {
        let mut subseries = Vec::new();
        let mut index = first;
        while index < span.end {
            subseries.push(si[index]);
            index += period;
        }
        let effective_filter = if stable {
            ResolvedSeasonalFilter::Stable
        } else {
            // `vsfb.f` indexes `Lter` by the calendar period of this subseries.
            request.filters[(start_subperiod - 1 + first) % period]
        };
        let effective_filter = if request.short_series_filters
            == x11::ShortSeriesFilters::UseRequested
            && effective_filter == ResolvedSeasonalFilter::S3x5
            && subseries.len() == 3
        {
            ResolvedSeasonalFilter::Stable
        } else {
            effective_filter
        };
        let filtered = crate::x11::seasonal::smooth_subseries(&subseries, effective_filter);
        last_filter = effective_filter;
        for (subseries_index, value) in filtered.into_iter().enumerate() {
            seasonal[first + subseries_index * period] = value;
        }
    }

    center_seasonal(
        &mut seasonal,
        span,
        period,
        start_subperiod,
        mode,
        request.filters,
        dates,
    )?;
    Ok(SeasonalSmoothing {
        values: seasonal,
        last_filter,
    })
}

/// `vsfc.f`: center the factors on their centered `period`-term moving
/// average, carried to the ends of `span`.
fn center_seasonal(
    seasonal: &mut [f64],
    span: Range<usize>,
    period: usize,
    start_subperiod: usize,
    mode: x11::Mode,
    filters: &SeasonalFilterSet<ResolvedSeasonalFilter>,
    dates: AxisSpan,
) -> Result<(), Error> {
    let mut moving_center = vec![0.0; seasonal.len()];
    numerics::moving_average(seasonal, &mut moving_center, span.clone(), 2, period);
    let half = period / 2;
    let first_center = span.start + half;
    let last_center = span.end - half - 1;
    for index in span.start..first_center {
        // `vsfc.f` checks each endpoint's declared `Lter`, not `vsfb.f`'s
        // fallback for a short subseries.
        let declared_stable =
            filters[(start_subperiod - 1 + index) % period] == ResolvedSeasonalFilter::Stable;
        moving_center[index] = moving_center[if declared_stable {
            index + period
        } else {
            first_center
        }];
    }
    for index in last_center + 1..span.end {
        let declared_stable =
            filters[(start_subperiod - 1 + index) % period] == ResolvedSeasonalFilter::Stable;
        moving_center[index] = moving_center[if declared_stable {
            index - period
        } else {
            last_center
        }];
    }
    for index in span {
        seasonal[index] = if mode == x11::Mode::PseudoAdditive {
            pseudo::divide(
                seasonal[index],
                moving_center[index],
                dates,
                index,
                PseudoStage::SeasonalSelection,
            )?
        } else {
            mode.remove(seasonal[index], moving_center[index])
        };
    }
    Ok(())
}

/// Repeat the first and last year of factors over the positions outside
/// `span`.
pub(super) fn extend_seasonal(seasonal: &mut [f64], span: Range<usize>, period: usize) {
    for index in span.end..seasonal.len() {
        seasonal[index] = seasonal[index - period];
    }
    for index in (0..span.start).rev() {
        seasonal[index] = seasonal[index + period];
    }
}

/// `si.f`: one section's seasonal in pass B. A first seasonal gives the
/// irregular and its extreme weights; `si` has its extremes replaced in place;
/// the seasonal of that SI is returned.
pub(super) fn replace_extremes_and_filter(
    si: &mut [f64],
    spans: ExtremeSpans,
    mode: x11::Mode,
    state: &mut ExtremeScaleState,
    request: SeasonalWeightRequest<'_>,
) -> Result<Vec<f64>, Error> {
    let SeasonalWeightRequest {
        smoothing,
        robust,
        dates,
    } = request;
    let period = usize::from(dates.start().frequency().periods_per_year());
    let start_subperiod = usize::from(dates.start().subperiod());
    let span = spans.application.clone();
    let options = state.request;
    let trend_only = options.decomposition() == crate::x11::Decomposition::TrendOnly;
    let first_seasonal = if trend_only {
        Vec::new()
    } else {
        seasonal_filter(si, span.clone(), mode, smoothing, dates)?.values
    };
    let irregular = if trend_only {
        si.to_vec()
    } else if mode == x11::Mode::PseudoAdditive {
        let mut values = vec![1.0; si.len()];
        for i in span.clone() {
            values[i] = pseudo::checked(
                si[i] - first_seasonal[i] + 1.0,
                dates,
                i,
                PseudoStage::SeasonalIrregular,
            )?;
        }
        values
    } else {
        remove_on_span(mode, si, &first_seasonal, span.clone())
    };
    if state.initial_si.is_none() {
        // `si.f` tests the first section's irregular, before any weighting.
        // The second section keeps that choice.
        state.resolve(
            &irregular,
            spans.estimation.clone(),
            period,
            mode,
            true,
            dates,
        )?;
    }
    let weights = calendar_extreme_weights(
        &irregular,
        spans,
        period,
        start_subperiod,
        mode,
        options.sigma_limits(),
        state.weight_policy().with_context(robust),
    )?;
    replace_extremes(si, &weights, span.clone(), period);
    if trend_only {
        Ok(vec![mode.center(); si.len()])
    } else {
        seasonal_filter(si, span, mode, smoothing, dates).map(|result| result.values)
    }
}
