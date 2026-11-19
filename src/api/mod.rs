//! The public entry point, [`adjust`], and the request, result and error types.

// Implements `Display` for a fieldless enum from a table of phrases. There is
// no wildcard arm, so a new variant fails to compile until it has a phrase.
// Defined before the module declarations so every submodule can use it.
macro_rules! display_phrases {
    ($type:ty { $($variant:ident => $phrase:literal),+ $(,)? }) => {
        impl std::fmt::Display for $type {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(match self {
                    $(Self::$variant => $phrase),+
                })
            }
        }
    };
}

pub(crate) mod diagnostics;
pub(crate) mod error;
pub(crate) mod regarima;
pub(crate) mod result;
pub(crate) mod seats;
mod span;
pub(crate) use seats::{
    ArmaCoordinates, SeatsFamily, SeatsForecastField, SeatsStrategy, SeatsUnsupportedOption,
    selected_family, unsupported_calendar, validate_request as validate_seats_request,
};
pub use span::PeriodSpan;
pub(crate) mod spec;
pub(crate) mod x11;

pub(crate) use diagnostics::{
    SpectralNotApplicable, SpectralReason, SpectralSeries, SpectralStage, SpectralTradingDayPeaks,
    UnavailableDetail,
};
pub(crate) use regarima::X11ModeSelection;
pub(crate) use spec::{FlowLengthContext, MINIMUM_TRADING_DAY_YEAR, ResolvedRegArimaSpec};

pub(crate) use x11::{X11EasterConflict, X11EasterStage, X11EasterUnavailableReason};

pub(crate) use error::{
    AdjustmentOutput, ArimaOrderTerm, Component, ExtensionDirection, SpecProblem,
    UnsupportedConfiguration, UnsupportedDetail, X11DecompositionConflict,
};
pub use error::{Error, ErrorKind, InternalError, InvalidSpec, NumericalFailure, Unsupported};
pub use result::{Adjustment, Backcasts, Diagnostics, Forecasts, ModelResult, Warning};
pub use spec::{ArimaModel, ArimaOrder, Method, RegArimaSpec, Spec, Transform};

use crate::regarima::{
    ConstantSelection, ForecastSpec, ModelSelection, OutlierType, TransformSelection,
};
// The request types `seats` names through this module.
use crate::regarima::{CalendarSpec, ResolvedArimaModel, ResolvedArimaOrder};
use crate::time::TimeAxis;
use crate::{Series, engine, numerics::build_62_equal};
use spec::SpecKind;

/// Seasonally adjusts one monthly or quarterly series according to `spec`.
///
/// `series` holds one observation per period, starting at its first
/// period, and is never modified. The [`Spec`] picks the pipeline
/// (direct X-11, X-11 with a regARIMA model, or SEATS with a regARIMA model)
/// and its options; the returned [`Adjustment`] holds the components,
/// forecasts, fitted model, diagnostics and warnings. Calls share no state:
/// the same inputs always give the same result, and separate series can be
/// adjusted in parallel.
///
/// # Input values
///
/// - **NaN marks a missing observation.** A regARIMA request (X-11 or SEATS)
///   estimates each gap with a dated regression column, as X-13ARIMA-SEATS
///   does. Direct X-11 cannot estimate gaps and returns
///   [`Error::MissingValue`]. A calendar month or quarter with no observation
///   at all is
///   [`Error::MissingSeason`]. If the observed values are all equal and the
///   first value is observed, each gap takes that value instead.
/// - **Every finite value is an observation**, including `-99999.0`, which
///   X-13ARIMA-SEATS reads as its missing-value code.
/// - **Infinity is invalid** ([`Error::NonFiniteValue`]).
///
/// # Errors
///
/// Every failure is an [`Error`]. Route on [`Error::kind`] rather than on
/// individual variants, which are many and may grow. A numerical overflow
/// that leaves a component, forecast or backcast, fitted-model estimate or
/// statistic, information criterion, outlier estimate or calendar effect
/// non-finite is reported as [`Error::Numerical`]; some numerical failures
/// are caught earlier and reported with their context instead. Diagnostics,
/// spectral estimates and the automatic search's selection scores are not
/// checked.
///
/// # Examples
///
/// A batch job that skips series this pipeline cannot adjust, and stops on
/// anything that points to a bug in the caller or the spec:
///
/// ```
/// use x13::{Adjustment, Error, ErrorKind, Period, Series, Spec, adjust, x11};
///
/// /// `Ok(None)` skips a series that cannot be adjusted this way.
/// fn adjust_or_skip(series: Series<'_>, spec: &Spec) -> Result<Option<Adjustment>, Error> {
///     match adjust(series, spec) {
///         Ok(adjustment) => Ok(Some(adjustment)),
///         Err(error) => match error.kind() {
///             // Too short, a gap this pipeline cannot fill, and so on.
///             ErrorKind::TooShort | ErrorKind::UnsuitableData => Ok(None),
///             // Invalid input or spec, an unsupported request, or a failed
///             // fit (which another model may avoid).
///             _ => Err(error),
///         },
///     }
/// }
///
/// let start = Period::monthly(2015, 1)?;
/// let spec = Spec::x11(x11::Options::new());
/// let values: Vec<f64> = (0..72)
///     .map(|i| {
///         let season = 8.0 * (i as f64 * std::f64::consts::PI / 6.0).sin();
///         let noise = ((i * 37) % 11) as f64 / 11.0 - 0.5;
///         100.0 + 0.5 * i as f64 + season + noise
///     })
///     .collect();
///
/// // Six years adjust.
/// assert!(adjust_or_skip(Series::new(start, &values), &spec)?.is_some());
///
/// // X-11 needs three full years, so two are skipped ...
/// let two_years = Series::new(start, &values[..24]);
/// let error = adjust(two_years, &spec).unwrap_err();
/// assert!(matches!(error, Error::TooShort { minimum: 36, .. }));
/// assert!(adjust_or_skip(two_years, &spec)?.is_none());
///
/// // ... and so is a gap, which X-11 alone cannot estimate.
/// let mut gapped = values.clone();
/// gapped[30] = f64::NAN;
/// assert!(adjust_or_skip(Series::new(start, &gapped), &spec)?.is_none());
///
/// // An infinite value is a caller bug: it stops the batch.
/// let mut broken = values.clone();
/// broken[30] = f64::INFINITY;
/// let error = adjust_or_skip(Series::new(start, &broken), &spec).unwrap_err();
/// assert_eq!(error.kind(), ErrorKind::InvalidInput);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn adjust(series: Series<'_>, spec: &Spec) -> Result<Adjustment, Error> {
    let adjustment = adjust_unvalidated(series, spec)?;
    adjustment.validate_finite()?;
    Ok(adjustment)
}

/// Rejects an empty input or an infinite observation. NaN is a gap, and
/// `-99999.0` is an ordinary value, not Build 62's missing-value code.
fn validate_values(values: &[f64]) -> Result<(), Error> {
    if values.is_empty() {
        return Err(Error::EmptyInput);
    }
    if let Some(index) = values.iter().position(|value| value.is_infinite()) {
        return Err(Error::NonFiniteValue { index });
    }
    Ok(())
}

fn adjust_unvalidated(series: Series<'_>, spec: &Spec) -> Result<Adjustment, Error> {
    let values = series.values();
    let axis = TimeAxis::from_start(series.start());
    validate_values(values)?;
    // A gap is never zero (`build_62_equal` is false for NaN).
    if values.iter().all(|value| build_62_equal(*value, 0.0)) {
        return Err(Error::AllZeroInput);
    }
    axis.periods(values.len())?;
    // The print check runs first, for every pipeline. The regARIMA-X-11 length
    // checks run inside the engines once the X-11 options are prepared, so
    // option errors keep precedence over length errors.
    validate_print_format(values)?;

    // editor.f requires a regARIMA model for missing-value estimation.
    let supports_missing = !matches!(spec.kind(), SpecKind::X11 { .. });
    if !supports_missing && let Some(index) = values.iter().position(|value| value.is_nan()) {
        return Err(Error::MissingValue { index });
    }

    let prepared_input = if supports_missing {
        Some(engine::missing::PreparedInput::new(values, axis)?)
    } else {
        None
    };
    let values = prepared_input
        .as_ref()
        .map_or(values, |input| input.values.as_ref());
    let missing = prepared_input
        .as_ref()
        .map_or(&[][..], |input| input.missing.as_slice());
    // Gaps in a series of zeros are filled with zero.
    if values.iter().all(|value| build_62_equal(*value, 0.0)) {
        return Err(Error::AllZeroInput);
    }

    match spec.kind() {
        SpecKind::Seats { regarima } => {
            // editor.f cancels backcasts for SEATS before calendar validation
            // and estimation; they do not extend the decomposition input.
            let ignored_backcasts = regarima.forecast().backcast_periods();
            let horizon = regarima.forecast().horizon();
            let regarima = regarima.with_forecast(ForecastSpec::new(horizon, 0)?);
            let mut adjustment = match seats::validate(values, axis, regarima.clone())? {
                seats::SeatsRequest::Explicit(family) => {
                    engine::seats::adjust(values, missing, axis, regarima, family)
                }
                seats::SeatsRequest::Automatic => {
                    let prepared =
                        engine::seats::prepare_automatic(values, missing, axis, regarima)?;
                    engine::seats::finish_prepared(values, axis, prepared)
                }
            }?;
            if ignored_backcasts > 0 {
                adjustment.warnings.insert(
                    0,
                    Warning::SeatsBackcastsIgnored {
                        periods: ignored_backcasts,
                    },
                );
            }
            Ok(adjustment)
        }
        SpecKind::X11 { options } => {
            let mode = options.mode().unwrap_or(x11::Mode::Multiplicative);
            let holiday = options.holiday_adjustment();
            // Without a regARIMA model Build 62 generates no extensions. The
            // three-year minimum is the X-11 core's own check.
            validate_augmented_length(axis, values.len(), 0, 0)?;
            validate_direct_x11(values, axis, mode, options, holiday)?;
            engine::direct_x11::adjust(values, axis, mode, options, holiday)
        }
        SpecKind::X11RegArima { regarima, options } => {
            let mode = options.mode_selection();
            let holiday = options.holiday_adjustment();
            let final_outliers = options.final_outliers();
            let allocation = options.temporary_change_allocation();
            let easter_transform = match (options.easter_estimation(), regarima.transform()) {
                (false, _) => None,
                (true, TransformSelection::Explicit(transform)) => Some(transform),
                (true, TransformSelection::Automatic(_)) => {
                    return Err(Error::from(SpecProblem::IncompatibleX11Easter {
                        reason: X11EasterConflict::AutomaticTransformation,
                    }));
                }
            };
            if matches!(regarima.transform(), TransformSelection::Automatic(_))
                && matches!(mode, X11ModeSelection::Explicit(_))
            {
                return Err(Error::from(SpecProblem::AutomaticTransformWithExplicitMode));
            }
            // The positive-domain modes validate observations before an explicit
            // preprocessing LOG can report its own domain error.
            if mode == X11ModeSelection::Explicit(x11::Mode::LogAdditive)
                && let Some(index) = crate::numerics::first_nonpositive(values)
            {
                return Err(Error::NonPositiveValue { index });
            }
            if mode == X11ModeSelection::Explicit(x11::Mode::PseudoAdditive) {
                validate_pseudo_values(values)?;
            }
            regarima.validate(values, axis)?;
            // Multiplicative's input-domain error follows preprocessing
            // validation and precedes the decomposition conflicts.
            if mode == X11ModeSelection::Explicit(x11::Mode::Multiplicative)
                && let Some(index) = crate::numerics::first_nonpositive(values)
            {
                return Err(Error::NonPositiveValue { index });
            }
            validate_decomposition(
                options,
                mode == X11ModeSelection::Explicit(x11::Mode::LogAdditive),
            )?;
            if let Some(transform) = easter_transform {
                engine::validate_legacy_easter(
                    axis.frequency(),
                    engine::resolve_mode(mode, transform),
                    regarima
                        .calendar()
                        .and_then(|calendar| calendar.easter())
                        .is_some(),
                    holiday,
                )?;
            }
            if mode == X11ModeSelection::Explicit(x11::Mode::PseudoAdditive) {
                if regarima
                    .calendar()
                    .is_some_and(|calendar| !calendar.is_empty())
                    || regarima.outliers().is_some()
                    || !regarima.manual_outliers().is_empty()
                {
                    return Err(Error::from(
                        SpecProblem::IncompatiblePseudoAdditivePreadjustment,
                    ));
                }
                // editor.f counts a missing-value regressor as an additive
                // outlier. A regression block keeps Adjao set, and final
                // removal keeps Finao set, so either refuses a gapped series;
                // the other final flags clear because Nb is positive.
                if !missing.is_empty() {
                    if regarima.constant() == ConstantSelection::Included
                        || final_outliers.removes(OutlierType::Additive)
                    {
                        return Err(Error::from(
                            SpecProblem::IncompatiblePseudoAdditivePreadjustment,
                        ));
                    }
                } else if matches!(final_outliers, x11::FinalOutliers::Remove(_))
                    && (regarima.model() == ModelSelection::Automatic
                        || regarima.constant() == ConstantSelection::Excluded)
                {
                    return Err(Error::from(
                        SpecProblem::IncompatiblePseudoAdditiveFinalOutliers {
                            model: regarima.model(),
                            constant: regarima.constant(),
                        },
                    ));
                }
            }
            let final_options = engine::FinalX11Options::new(final_outliers, allocation, holiday);
            if regarima.model() == ModelSelection::Automatic {
                engine::automatic_model_x11::adjust(
                    values,
                    missing,
                    axis,
                    regarima,
                    mode,
                    options,
                    final_options,
                )
            } else {
                engine::regarima_x11::adjust(
                    values,
                    missing,
                    axis,
                    regarima,
                    mode,
                    options,
                    final_options,
                )
            }
        }
    }
}

/// Validates direct X-11's input and options before the engine runs it.
fn validate_direct_x11(
    values: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    options: x11::Options,
    holiday: x11::HolidayAdjustment,
) -> Result<(), Error> {
    if mode == x11::Mode::PseudoAdditive {
        validate_pseudo_values(values)?;
    }
    if matches!(mode, x11::Mode::Multiplicative | x11::Mode::LogAdditive)
        && let Some(index) = crate::numerics::first_nonpositive(values)
    {
        return Err(Error::NonPositiveValue { index });
    }
    validate_decomposition(options, mode == x11::Mode::LogAdditive)?;
    if options.easter_estimation() {
        engine::validate_legacy_easter(axis.frequency(), mode, false, holiday)?;
    }
    Ok(())
}

fn validate_decomposition(options: x11::Options, log_additive: bool) -> Result<(), Error> {
    if options.decomposition() == x11::Decomposition::TrendOnly {
        let reason = if options.has_explicit_seasonal_filter() {
            Some(X11DecompositionConflict::ExplicitSeasonalFilter)
        } else if log_additive {
            Some(X11DecompositionConflict::UnsafeLogAdditive)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(Error::from(SpecProblem::IncompatibleX11Decomposition {
                reason,
            }));
        }
    }
    Ok(())
}

/// srslen.prm `PLEN`: positions in a forecast/backcast-augmented series.
const MAX_AUGMENTED_POSITIONS: usize = 1020;
/// srslen.prm `PYRS`: calendar years spanned by the augmented series.
const MAX_AUGMENTED_YEARS: usize = 85;
/// tfmts.f: the widest table column; wider observations fail to print.
const MAX_PRINT_WIDTH: f64 = 21.0;

/// tfmts.f, called from editor.f with the observed maximum and minimum
/// (missing values excluded). Build 62 accepts negative values (`Muladd = 1`)
/// and this crate has no decimals option (`Kdec = 0`).
fn validate_print_format(values: &[f64]) -> Result<(), Error> {
    // `int(log10(X)) + Kdec + 3` grows with |value|, so the first observation
    // past the limit identifies a failing maximum. A zero magnitude gives
    // `-inf` here where tfmts.f's `dpeq(Maxy,0)` branch prints 3 columns;
    // both pass.
    let too_wide = |value: f64| value.abs().log10().trunc() + 3.0 > MAX_PRINT_WIDTH;
    match values
        .iter()
        .position(|&value| !value.is_nan() && too_wide(value))
    {
        Some(index) => Err(Error::ObservationTooLarge { index }),
        None => Ok(()),
    }
}

/// editor.f's series-length checks for a regARIMA-X-11 run. `extensions`
/// are the ones Build 62 generates: backcasts after the model-span
/// cancellation, forecasts as requested.
pub(crate) fn validate_regarima_x11_length(
    axis: TimeAxis,
    count: usize,
    extensions: ForecastSpec,
) -> Result<(), Error> {
    // It precedes estimation, so rgarma.f never sees a series this short.
    crate::x11::require_three_years(axis.frequency(), count)?;
    validate_augmented_length(
        axis,
        count,
        extensions.backcast_periods(),
        extensions.forecast_periods(axis.frequency()),
    )
}

/// editor.f `Posffc.gt.PLEN` and `nyr.gt.PYRS`.
///
/// getsrs.f keeps only the first `PLEN` values, so a longer monthly series
/// starting in January with no extensions runs on silently truncated input.
/// This function rejects it instead. SEATS applies its own, tighter window in
/// `seats::validate_request`.
fn validate_augmented_length(
    axis: TimeAxis,
    count: usize,
    backcasts: u16,
    forecasts: u16,
) -> Result<(), Error> {
    let period = axis.frequency().periods_per_year();
    let periods_per_year = usize::from(period);
    // setxpt.f `Posffc = Nbcst2 + Nspobs + Nfcst`: editor.f's `Nbcst2` adds
    // the months before the backcast start in its calendar year.
    let leading = (i64::from(axis.start().subperiod()) - 1 - i64::from(backcasts))
        .rem_euclid(i64::from(period)) as usize;
    let positions = leading + usize::from(backcasts) + count + usize::from(forecasts);
    let years = positions.div_ceil(periods_per_year);
    if positions > MAX_AUGMENTED_POSITIONS || years > MAX_AUGMENTED_YEARS {
        return Err(Error::SeriesTooLong {
            positions,
            years,
            maximum_positions: MAX_AUGMENTED_POSITIONS,
            maximum_years: MAX_AUGMENTED_YEARS,
        });
    }
    Ok(())
}

fn validate_pseudo_values(values: &[f64]) -> Result<(), Error> {
    if let Some(index) = values.iter().position(|value| *value < 0.0) {
        return Err(Error::NegativeValue { index });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
