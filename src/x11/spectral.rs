//! X-11 spectral diagnostics (`spcdrv`): the AR spectra of the original
//! series, the modified seasonally adjusted series and the modified
//! irregular, and their seasonal and trading-day peaks.
use crate::diagnostics::{
    SpectralNotApplicable as NotApplicable, SpectralPeaks, SpectralReason as Reason,
    SpectralSample as Sample, SpectralSeries as Series, SpectralStage as Stage,
    SpectralTradingDayPeaks as TradingDay, Unavailable, UnavailableDetail,
};
use crate::numerics::spectral::peak_counts;
use crate::x11;

/// One value of the original series as `spcdrv` reads it: the modified
/// original with its extreme value put back. Pseudo-additive has none here;
/// [`pseudo_original_operand`] builds it from the trend-cycle Part F leaves.
pub(super) fn original_operand(
    mode: crate::x11::Mode,
    modified: f64,
    extreme: f64,
    si: f64,
    unmodified_si: f64,
) -> Option<f64> {
    match mode {
        crate::x11::Mode::Additive => Some(modified + extreme),
        crate::x11::Mode::Multiplicative => Some(modified * extreme),
        // `x11pt3` exponentiates both SI series and makes their quotient the
        // extreme value that `spcdrv` multiplies in. One `exp` of the
        // combined logarithms would round differently.
        crate::x11::Mode::LogAdditive => Some(modified.exp() * (unmodified_si.exp() / si.exp())),
        crate::x11::Mode::PseudoAdditive => None,
    }
}

/// One value of the pseudo-additive original, which `spcdrv` rebuilds from
/// the final components.
pub(super) fn pseudo_original_operand(
    trend: f64,
    seasonal: f64,
    irregular: f64,
    trend_only: bool,
) -> f64 {
    // `spcdrv` subtracts, adds and multiplies; gfortran fused none of it.
    if trend_only {
        trend * irregular
    } else {
        trend * (seasonal + (irregular - 1.0))
    }
}

/// The series the spectra are estimated from, over the observed dates, and
/// what Part F leaves for them. Each is `None` until the step that computes
/// it has run.
#[derive(Debug)]
pub(crate) struct SpectralWorkspace {
    /// The original series (see [`original_operand`]).
    pub(crate) adjusted_original: Option<Vec<f64>>,
    /// The trend-cycle as Part F leaves it, which the pseudo-additive
    /// original is rebuilt from.
    pub(crate) part_f_trend: Option<Vec<f64>>,
    /// The seasonally adjusted series modified for extremes (table E2).
    pub(crate) modified_adjusted: Option<Vec<f64>>,
    /// The irregular modified for extremes (table E3).
    pub(crate) modified_irregular: Option<Vec<f64>>,
    /// Whether Part F reached its level-shift round trip.
    pub(crate) level_shift_round_trip: LevelShiftRoundTrip,
}

#[cfg(test)]
impl SpectralWorkspace {
    /// A workspace with no retained operands, for tests that build
    /// components by hand.
    pub(crate) fn empty() -> Self {
        Self {
            adjusted_original: None,
            part_f_trend: None,
            modified_adjusted: None,
            modified_irregular: None,
            level_shift_round_trip: LevelShiftRoundTrip::Unknown,
        }
    }
}

/// Whether Part F got as far as taking the level shift out of the modified
/// seasonally adjusted series and putting it back. Build 62 does so once its
/// variances are normalized, and the round trip can change the last bit of
/// the series the spectrum then reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LevelShiftRoundTrip {
    /// Part F got that far, whatever became of its measures afterwards.
    Reached,
    /// Build 62 returns from Part F first, on a constant series.
    NotReached,
    /// Part F stopped on an arithmetic failure first, so there is no telling.
    Unknown,
}

/// What the spectra take from outside the X-11 run.
pub(crate) struct SpectralCompletion {
    /// How many times the original and the seasonally adjusted series are
    /// differenced first.
    pub(crate) difference_order: usize,
    /// Whether the original series is constant.
    pub(crate) same_original: bool,
    /// The level-shift factors over the observed dates, if the model has any.
    pub(crate) level_shift: Option<Vec<f64>>,
    /// Whether the final seasonally adjusted series has its level shift
    /// removed (Build 62's `Finls`).
    pub(crate) remove_final_level_shift: bool,
}

/// Keeps the modified seasonally adjusted series for its spectrum.
pub(crate) fn retain_modified(c: &mut super::Components, values: &[f64]) {
    c.spectral_workspace.modified_adjusted = Some(values.to_vec());
}

fn reconstruct_pseudo_original(c: &mut super::Components) {
    if c.mode == crate::x11::Mode::PseudoAdditive {
        let work = &mut c.spectral_workspace;
        // `spcdrv` reads the trend-cycle as Part F leaves it, not the
        // published D12. If Part F failed before that point there is no
        // original to test; a later failure of the measures does not matter.
        work.adjusted_original = work.part_f_trend.as_ref().map(|trend| {
            trend
                .iter()
                .zip(&c.seasonal)
                .zip(&c.irregular)
                .map(|((&t, &s), &i)| {
                    pseudo_original_operand(
                        t,
                        s,
                        i,
                        c.decomposition == crate::x11::Decomposition::TrendOnly,
                    )
                })
                .collect()
        });
    }
}

/// Keeps the irregular modified for extremes for its spectrum: `values` with
/// the neutral value wherever the extreme weight is zero.
pub(crate) fn retain_irregular(c: &mut super::Components, values: &[f64]) {
    let center = if c.mode == crate::x11::Mode::Additive {
        0.0
    } else {
        1.0
    };
    c.spectral_workspace.modified_irregular = Some(
        values
            .iter()
            .zip(&c.intermediate_extreme_weights)
            .map(|(&value, &weight)| if weight > 0.0 { value } else { center })
            .collect(),
    );
}

/// The spectral diagnostics of a finished run, with their peak warnings.
pub(super) fn complete(
    c: &mut super::Components,
    dates: crate::time::AxisSpan,
    completion: SpectralCompletion,
) -> x11::SpectralDiagnostics {
    reconstruct_pseudo_original(c);
    if c.spectral_workspace.modified_irregular.is_none() {
        let irregular = c.irregular.clone();
        retain_irregular(c, &irregular);
    }
    let work = &mut c.spectral_workspace;
    if let (Some(values), Some(factors)) = (&mut work.modified_adjusted, &completion.level_shift) {
        let multiplicative = c.mode != crate::x11::Mode::Additive;
        // Part F takes the level shift out and puts it back, which can round;
        // the spectrum then reads the series without it.
        let round_trip = work.level_shift_round_trip == LevelShiftRoundTrip::Reached
            && !completion.remove_final_level_shift;
        for (value, &factor) in values.iter_mut().zip(factors) {
            if round_trip {
                *value = with_level_shift(
                    without_level_shift(*value, factor, multiplicative),
                    factor,
                    multiplicative,
                );
            }
            *value = without_level_shift(*value, factor, multiplicative);
        }
    }
    let guard = if dates.start().frequency() == crate::Frequency::Quarterly {
        Some(NotApplicable::QuarterlyFrequency)
    } else if completion.same_original {
        Some(NotApplicable::ConstantOriginalSeries)
    } else {
        None
    };
    let trend_only = c.decomposition == crate::x11::Decomposition::TrendOnly;
    let difference_order = completion.difference_order;
    // Build 62 decides whether to test trading-day peaks from this count for
    // every series, the undifferenced irregular included.
    let differenced_count = c.irregular.len().min(96).saturating_sub(difference_order);
    let evaluate = |series, values: Option<&[f64]>| {
        if let Some(reason) = guard {
            return Err(not_examined(series, reason));
        }
        if trend_only && series != Series::AdjustedOriginal {
            return Err(not_examined(series, NotApplicable::TrendOnlyDecomposition));
        }
        let Some(values) = values else {
            return Err(undefined(
                series,
                Reason::PartFPrerequisiteUnavailable,
                None,
            ));
        };
        // The irregular is neither differenced nor logged.
        let order = if series == Series::ModifiedIrregular {
            0
        } else {
            difference_order
        };
        let logarithmic =
            series != Series::ModifiedIrregular && c.mode != crate::x11::Mode::Additive;
        let (sample, transformed) = match prepare(
            values,
            dates,
            order,
            logarithmic,
            series == Series::ModifiedIrregular && c.mode != crate::x11::Mode::Additive,
        ) {
            Ok(value) => value,
            Err((reason, sample)) => return Err(undefined(series, reason, sample)),
        };
        match estimate_peaks(&transformed, differenced_count) {
            Ok(PeakEstimate {
                ar_order,
                seasonal_count,
                trading_day,
            }) => Ok(SpectralPeaks {
                sample,
                ar_order,
                seasonal_count,
                trading_day,
            }),
            Err(reason) => Err(undefined(series, reason, Some(sample))),
        }
    };
    let original = evaluate(Series::AdjustedOriginal, work.adjusted_original.as_deref());
    let adjusted = evaluate(
        Series::ModifiedSeasonallyAdjusted,
        if completion.level_shift.is_some()
            && !completion.remove_final_level_shift
            && work.level_shift_round_trip == LevelShiftRoundTrip::Unknown
        {
            None
        } else {
            work.modified_adjusted.as_deref()
        },
    );
    let irregular = evaluate(
        Series::ModifiedIrregular,
        work.modified_irregular.as_deref(),
    );
    let diagnostics = x11::SpectralDiagnostics {
        adjusted_original: original,
        modified_seasonally_adjusted: adjusted,
        modified_irregular: irregular,
    };
    c.warnings.extend(warnings(&diagnostics, trend_only));
    diagnostics
}

/// A spectrum X-13ARIMA-SEATS does not examine.
fn not_examined(series: Series, reason: NotApplicable) -> Unavailable {
    Unavailable(UnavailableDetail::SpectrumNotApplicable { series, reason })
}

/// A spectrum that would be examined but is undefined.
fn undefined(series: Series, reason: Reason, sample: Option<Sample>) -> Unavailable {
    Unavailable(UnavailableDetail::Spectrum {
        series,
        reason,
        sample,
    })
}

fn without_level_shift(value: f64, factor: f64, multiplicative: bool) -> f64 {
    if multiplicative {
        value / factor
    } else {
        value - factor
    }
}

fn with_level_shift(value: f64, factor: f64, multiplicative: bool) -> f64 {
    if multiplicative {
        value * factor
    } else {
        value + factor
    }
}

/// The sample and its transformed values, or why there are none.
type Prepared = Result<(Sample, Vec<f64>), (Reason, Option<Sample>)>;

/// The last 96 values as `spcdrv` hands them to the spectrum: logged, or
/// centered when `center_one`, then differenced `order` times.
fn prepare(
    values: &[f64],
    dates: crate::time::AxisSpan,
    order: usize,
    logarithmic: bool,
    center_one: bool,
) -> Prepared {
    let start = values.len().saturating_sub(96);
    let count = values.len() - start;
    if count <= order {
        return Err((
            Reason::InsufficientInput {
                actual: count,
                minimum: order.saturating_add(1),
            },
            None,
        ));
    }
    let sample = Sample {
        input_first: dates.period(start),
        input_last: dates.period(values.len() - 1),
        input_count: count,
        first: dates.period(start + order),
        last: dates.period(values.len() - 1),
        count: count - order,
        difference_order: order,
        logarithmic_input: logarithmic,
    };
    let mut values = values[start..].to_vec();
    for (i, value) in values.iter_mut().enumerate() {
        let period = dates.period(start + i);
        if !value.is_finite() {
            return Err((Reason::NonFiniteInput { period }, Some(sample)));
        }
        if logarithmic {
            if *value <= 0.0 {
                return Err((Reason::NonPositiveLogInput { period }, Some(sample)));
            }
            *value = value.ln();
            if !value.is_finite() {
                return Err((
                    Reason::NonFiniteArithmetic {
                        stage: Stage::LogTransform,
                    },
                    Some(sample),
                ));
            }
        } else if center_one {
            *value -= 1.0;
        }
    }
    // `gendff` differences the logged values in place, each pass from the
    // end back to one value later than the pass before.
    for pass in 1..=order {
        for i in (pass..values.len()).rev() {
            values[i] -= values[i - 1];
            if !values[i].is_finite() {
                return Err((
                    Reason::NonFiniteArithmetic {
                        stage: Stage::Differencing,
                    },
                    Some(sample),
                ));
            }
        }
    }
    Ok((sample, values[order..].to_vec()))
}

/// The peaks of one series' AR spectrum.
#[derive(Debug)]
struct PeakEstimate {
    ar_order: usize,
    seasonal_count: usize,
    trading_day: TradingDay,
}

/// Estimates the AR spectrum of `values` and counts its visually significant
/// peaks. Trading-day peaks are tested only when `differenced_count` exceeds
/// 60.
fn estimate_peaks(values: &[f64], differenced_count: usize) -> Result<PeakEstimate, Reason> {
    use crate::numerics::spectral::{SpectralGrid, estimate};
    let regular = estimate(values, SpectralGrid::Regular).map_err(core_reason)?;
    let expanded = estimate(values, SpectralGrid::Expanded).map_err(core_reason)?;
    let (seasonal_count, trading_day_count) = peak_counts(
        &regular.decibels,
        &expanded.decibels,
        &expanded.frequencies,
        differenced_count > 60,
    );
    let trading_day = if differenced_count > 60 {
        TradingDay::Tested {
            count: trading_day_count,
        }
    } else {
        TradingDay::NotTestedInsufficientSample {
            count: differenced_count,
        }
    };
    Ok(PeakEstimate {
        ar_order: regular.order,
        seasonal_count,
        trading_day,
    })
}

/// The reason for a spectrum that could not be estimated.
fn core_reason(error: crate::numerics::spectral::SpectralUnavailable) -> Reason {
    use crate::numerics::spectral::{SpectralArithmeticStage as S, SpectralUnavailable as E};
    match error {
        E::InsufficientInput { length, required } => Reason::InsufficientInput {
            actual: length,
            minimum: required,
        },
        E::NonfiniteInput { .. } => Reason::NonFiniteArithmetic {
            stage: Stage::OperandConstruction,
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

/// The warnings Build 62 prints for visually significant peaks, and for an
/// original series with no seasonal peak.
fn warnings(diagnostics: &x11::SpectralDiagnostics, trend_only: bool) -> Vec<crate::Warning> {
    use crate::diagnostics::SpectralPeakKind as Kind;
    let records = [
        &diagnostics.adjusted_original,
        &diagnostics.modified_seasonally_adjusted,
        &diagnostics.modified_irregular,
    ];
    // A seasonal peak in the original is expected unless the decomposition is
    // trend-only, in which case only the original is tested.
    let seasonal_records = if trend_only {
        &records[..1]
    } else {
        &records[1..]
    };
    let seasonal = seasonal_records
        .iter()
        .any(|record| matches!(record, Ok(peaks) if peaks.seasonal_count > 0));
    let trading_day = records.iter().any(|record| {
        matches!(
            record,
            Ok(SpectralPeaks {
                trading_day: TradingDay::Tested { count },
                ..
            }) if *count > 0
        )
    });
    let mut result = Vec::new();
    if let Some(kind) = match (seasonal, trading_day) {
        (true, true) => Some(Kind::Both),
        (true, false) => Some(Kind::Seasonal),
        (false, true) => Some(Kind::TradingDay),
        (false, false) => None,
    } {
        result.push(crate::Warning::X11SpectralPeaks { kind });
    }
    if !trend_only
        && matches!(diagnostics.adjusted_original, Ok(ref peaks) if peaks.seasonal_count == 0)
    {
        result.push(crate::Warning::X11NoOriginalSeasonalSpectralPeaks);
    }
    result
}

#[cfg(test)]
mod tests;
