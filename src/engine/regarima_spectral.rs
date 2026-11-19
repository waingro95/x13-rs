//! The AR spectrum of the final model's residuals (`spcrsd.f`).
use crate::api::{
    SpectralNotApplicable as NotApplicable, SpectralReason as Reason, SpectralSeries as Series,
    SpectralStage as Stage, SpectralTradingDayPeaks as TradingDay, UnavailableDetail,
};
use crate::diagnostics::{SpectralPeaks as Peaks, SpectralSample as Sample, Unavailable};
use crate::numerics::spectral::{ArSpectrum, SpectralGrid, estimate, peak_counts};
use crate::time::{AxisSpan, TimeAxis};
use crate::{Error, Period};

/// The peaks of the residual spectrum, or why it has none, for the last
/// residuals of a fit whose observations start at `axis`. Fails only if
/// `axis` cannot date `observed_count` observations.
pub(crate) fn compute(
    axis: TimeAxis,
    observed_count: usize,
    residuals: &[f64],
    converged: bool,
    model_variance: f64,
) -> Result<Result<Peaks, Unavailable>, Error> {
    let dates = axis.span(observed_count)?;
    let selected = window(observed_count, residuals);
    let count = selected.len();
    let first = observed_count - count;
    let sample = || {
        residual_sample(
            dates.period(first),
            dates.period(dates.count().saturating_sub(1)),
            count,
        )
    };
    let unavailable = |reason| {
        Ok(Err(Unavailable(UnavailableDetail::Spectrum {
            series: Series::Residuals,
            reason,
            sample: (count > 0).then(sample),
        })))
    };
    let guard = if axis.frequency() == crate::Frequency::Quarterly {
        Some(NotApplicable::QuarterlyFrequency)
    } else if !converged {
        Some(NotApplicable::ModelNotConverged)
    } else if model_variance == 0.0 {
        Some(NotApplicable::ZeroModelVariance)
    } else {
        None
    };
    if let Some(reason) = guard {
        return Ok(Err(Unavailable(UnavailableDetail::SpectrumNotApplicable {
            series: Series::Residuals,
            reason,
        })));
    }
    if !model_variance.is_finite() || model_variance < 0.0 {
        return unavailable(Reason::InvalidModelVariance);
    }
    let (regular, expanded) = match spectra(selected, dates, first) {
        Ok(spectra) => spectra,
        Err(reason) => return unavailable(reason),
    };
    let (seasonal_count, td) = peak_counts(
        &regular.decibels,
        &expanded.decibels,
        &expanded.frequencies,
        count > 60,
    );
    Ok(Ok(Peaks {
        sample: sample(),
        ar_order: regular.order,
        seasonal_count,
        trading_day: if count > 60 {
            TradingDay::Tested { count: td }
        } else {
            TradingDay::NotTestedInsufficientSample { count }
        },
    }))
}

/// The sample of `count` residuals from `first` through `last`. The residuals
/// are neither logged nor differenced, so the input span is the sample.
fn residual_sample(first: Period, last: Period, count: usize) -> Sample {
    Sample {
        input_first: first,
        input_last: last,
        input_count: count,
        first,
        last,
        count,
        difference_order: 0,
        logarithmic_input: false,
    }
}

/// The warning Build 62 prints for visually significant peaks.
pub(super) fn warning(spectrum: &Result<Peaks, Unavailable>) -> Option<crate::Warning> {
    let Ok(peaks) = spectrum else {
        return None;
    };
    let seasonal = peaks.seasonal_count > 0;
    let td = matches!(peaks.trading_day, TradingDay::Tested { count } if count > 0);
    use crate::diagnostics::SpectralPeakKind as K;
    let kind = match (seasonal, td) {
        (true, true) => K::Both,
        (true, false) => K::Seasonal,
        (false, true) => K::TradingDay,
        (false, false) => return None,
    };
    Some(crate::Warning::ResidualSpectralPeaks { kind })
}

/// spcrsd's sample: the last `min(observed, residuals, 96)` residuals.
fn window(observed_count: usize, residuals: &[f64]) -> &[f64] {
    let count = observed_count.min(residuals.len()).min(96);
    &residuals[residuals.len() - count..]
}

/// The regular and expanded AR spectra of `selected`, whose first value is
/// observation `first` of `dates`. spcrsd evaluates the expanded grid before
/// the regular one, with unchanged a(1:na).
fn spectra(
    selected: &[f64],
    dates: AxisSpan,
    first: usize,
) -> Result<(ArSpectrum, ArSpectrum), Reason> {
    let core_reason = |error| core_reason(error, dates, first);
    let expanded = estimate(selected, SpectralGrid::Expanded).map_err(core_reason)?;
    let regular = estimate(selected, SpectralGrid::Regular).map_err(core_reason)?;
    Ok((regular, expanded))
}

fn core_reason(
    error: crate::numerics::spectral::SpectralUnavailable,
    dates: AxisSpan,
    first: usize,
) -> Reason {
    use crate::numerics::spectral::{SpectralArithmeticStage as S, SpectralUnavailable as E};
    match error {
        E::InsufficientInput { length, required } => Reason::InsufficientInput {
            actual: length,
            minimum: required,
        },
        E::NonfiniteInput { index } => Reason::NonFiniteInput {
            period: dates.period(first + index),
        },
        E::ZeroCovariance => Reason::ZeroCovariance,
        E::EarlyVarianceCollapse {
            completed_order,
            required_order,
        } => Reason::EarlyVarianceCollapse {
            completed_order,
            required_order,
        },
        E::NonpositiveInnovationVariance { order } => {
            Reason::NonPositiveInnovationVariance { order }
        }
        E::NonfiniteArithmetic { stage } => Reason::NonFiniteArithmetic {
            stage: match stage {
                S::MeanSum | S::Mean | S::Demeaning => Stage::MeanDeletion,
                S::CovarianceSum | S::Covariance => Stage::Autocovariance,
                S::InitialVariance
                | S::InitialAic
                | S::InitialLagCovariance
                | S::VarianceRatio
                | S::Reflection
                | S::ReflectionSquare
                | S::InnovationVariance
                | S::Aic
                | S::ArCoefficient
                | S::NextLagCovariance
                | S::CovarianceRecurrence => Stage::ArRecursion,
                S::CosineSum | S::SineSum | S::PowerDenominator | S::SpectralPower => {
                    Stage::SpectralPower
                }
                S::Decibels => Stage::DecibelConversion,
            },
        },
    }
}
#[cfg(test)]
mod tests;
