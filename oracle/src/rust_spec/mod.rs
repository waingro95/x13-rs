//! Builds the `x13::Spec` a configuration asks for and runs `x13::adjust`
//! on it.

use thiserror::Error;

use crate::*;

mod regarima_request;

/// Why the Rust side of a case produced no adjustment. The Rust bit records
/// hold the `Display` text of every variant but `Adjustment`.
#[derive(Debug, Error)]
pub(crate) enum RustError {
    #[error("corpus comparison supports monthly and quarterly series only, not {0:?}")]
    UnsupportedFrequency(Frequency),
    #[error("unsupported corpus comparison configuration")]
    UnsupportedConfiguration,
    #[error("invalid Rust period: {0}")]
    Period(#[from] x13::PeriodError),
    #[error("Rust seasonal adjustment failed: {0}")]
    Adjustment(#[from] x13::Error),
    #[error(transparent)]
    Oracle(#[from] OracleError),
}

fn rust_final_outliers(configuration: &OracleConfiguration) -> Option<x13::x11::FinalOutliers> {
    let types = match configuration.final_outliers()? {
        OracleX11FinalOutliers::Retain => return Some(x13::x11::FinalOutliers::Retain),
        OracleX11FinalOutliers::Remove { types } => types,
    };
    let mut selected = None;
    for (oracle, rust) in [
        (
            crate::OracleOutlierType::Additive,
            x13::regarima::OutlierTypes::ADDITIVE,
        ),
        (
            crate::OracleOutlierType::LevelShift,
            x13::regarima::OutlierTypes::LEVEL_SHIFT,
        ),
        (
            crate::OracleOutlierType::TemporaryChange,
            x13::regarima::OutlierTypes::TEMPORARY_CHANGE,
        ),
    ] {
        if types.contains(oracle) {
            selected = Some(
                selected.map_or(rust, |current: x13::regarima::OutlierTypes| {
                    current.union(rust)
                }),
            );
        }
    }
    Some(x13::x11::FinalOutliers::Remove(
        selected.expect("validated oracle mask is nonempty"),
    ))
}

fn rust_temporary_change_allocation(
    configuration: &OracleConfiguration,
) -> Option<x13::x11::TemporaryChangeAllocation> {
    Some(match configuration.temporary_change_allocation()? {
        OracleX11TemporaryChangeAllocation::Irregular => {
            x13::x11::TemporaryChangeAllocation::Irregular
        }
        OracleX11TemporaryChangeAllocation::Trend => x13::x11::TemporaryChangeAllocation::Trend,
    })
}

fn rust_holiday_adjustment(
    configuration: &OracleConfiguration,
) -> Option<x13::x11::HolidayAdjustment> {
    Some(match configuration.holiday_adjustment()? {
        crate::OracleX11HolidayAdjustment::Remove => x13::x11::HolidayAdjustment::Remove,
        crate::OracleX11HolidayAdjustment::Retain => x13::x11::HolidayAdjustment::Retain,
    })
}

/// Sets the final X-11 requests the configuration states.
fn with_final_x11_requests(
    mut options: x13::x11::Options,
    configuration: &OracleConfiguration,
) -> x13::x11::Options {
    if let Some(final_outliers) = rust_final_outliers(configuration) {
        options = options.with_final_outliers(final_outliers);
    }
    if let Some(allocation) = rust_temporary_change_allocation(configuration) {
        options = options.with_temporary_change_allocation(allocation);
    }
    if let Some(holiday) = rust_holiday_adjustment(configuration) {
        options = options.with_holiday_adjustment(holiday);
    }
    options
}

/// Sets the X-11 Easter request the configuration states.
fn with_legacy_easter(
    options: x13::x11::Options,
    configuration: &OracleConfiguration,
) -> x13::x11::Options {
    match configuration.legacy_easter() {
        Some(crate::OracleX11Easter::None) => options.with_easter_estimation(false),
        Some(crate::OracleX11Easter::Estimate) => options.with_easter_estimation(true),
        None => options,
    }
}

/// A regARIMA X-11 request before it becomes a `Spec`.
struct RegArimaX11Request {
    regarima: x13::RegArimaSpec,
    /// Every X-11 option but the mode.
    options: x13::x11::Options,
    /// `None` takes the mode from the transform.
    mode: Option<x13::x11::Mode>,
}

impl RegArimaX11Request {
    fn into_spec(self) -> x13::Spec {
        let options = match self.mode {
            Some(mode) => self.options.with_mode(mode),
            None => self.options,
        };
        x13::Spec::x11_regarima(self.regarima, options)
    }
}

const fn rust_x11_mode(mode: crate::X11Mode) -> x13::x11::Mode {
    match mode {
        crate::X11Mode::Additive => x13::x11::Mode::Additive,
        crate::X11Mode::Multiplicative => x13::x11::Mode::Multiplicative,
        crate::X11Mode::LogAdditive => x13::x11::Mode::LogAdditive,
        crate::X11Mode::PseudoAdditive => x13::x11::Mode::PseudoAdditive,
    }
}

fn rust_short_series_filters(configuration: &OracleConfiguration) -> x13::x11::ShortSeriesFilters {
    match configuration.short_series_filters() {
        None | Some(crate::OracleX11ShortSeriesFilters::StableFallback) => {
            x13::x11::ShortSeriesFilters::StableFallback
        }
        Some(crate::OracleX11ShortSeriesFilters::UseRequested) => {
            x13::x11::ShortSeriesFilters::UseRequested
        }
    }
}

/// Runs `x13::adjust` on `series` with the spec of `configuration`.
pub(crate) fn adjust(
    series: &Series,
    configuration: &OracleConfiguration,
) -> Result<x13::Adjustment, RustError> {
    let (spec, start) = build(series, configuration)?;
    Ok(x13::adjust(
        x13::Series::new(start, &series.rust_values()),
        &spec,
    )?)
}

/// The `Spec` of `configuration` and the first period of `series`.
pub(crate) fn build(
    series: &Series,
    configuration: &OracleConfiguration,
) -> Result<(x13::Spec, x13::Period), RustError> {
    let start = start_period(series)?;
    let spec = match configuration {
        OracleConfiguration::RegArima { .. } => regarima_request::spec(configuration, series)?,
        OracleConfiguration::DirectX11 {
            mode,
            seasonal_filter,
            seasonal_filters,
            trend_filter,
            sigma_limits,
            ..
        } => {
            let options = rust_x11_options(
                RustSeasonalRequest {
                    decomposition: configuration.decomposition(),
                    filter: *seasonal_filter,
                    filters: *seasonal_filters,
                    short_series_filters: rust_short_series_filters(configuration),
                    shrinkage: configuration.seasonal_shrinkage(),
                },
                *trend_filter,
                *sigma_limits,
                configuration.extreme_estimation_span(),
                configuration.extreme_scale(),
                configuration.extreme_estimator(),
            )?
            .with_mode(rust_x11_mode(*mode));
            x13::Spec::x11(with_legacy_easter(
                with_final_x11_requests(options, configuration),
                configuration,
            ))
        }
        OracleConfiguration::ExplicitRegArimaX11 { .. }
        | OracleConfiguration::AutomaticTransformRegArimaX11 { .. }
        | OracleConfiguration::CalendarOutlierRegArimaX11 { .. } => {
            explicit_model_request(configuration)?.into_spec()
        }
        OracleConfiguration::AutomaticModelX11 { .. } => {
            automatic_model_request(configuration)?.into_spec()
        }
    };
    Ok((spec, start))
}

pub(crate) fn start_period(series: &Series) -> Result<x13::Period, RustError> {
    match series.frequency {
        Frequency::Monthly => Ok(x13::Period::monthly(
            series.start.year,
            series.start.subperiod,
        )?),
        Frequency::Quarterly => Ok(x13::Period::quarterly(
            series.start.year,
            series.start.subperiod,
        )?),
        frequency => Err(RustError::UnsupportedFrequency(frequency)),
    }
}

/// The period of observation `index` in a series that begins at `start`. A
/// year past the supported range is always `PeriodOverflow { index }`, as
/// `Series::periods` reports it.
pub(crate) fn period_at(start: x13::Period, index: usize) -> Result<x13::Period, x13::PeriodError> {
    let overflow = x13::PeriodError::PeriodOverflow { index };
    let offset = i64::try_from(index).map_err(|_| overflow)?;
    start.shift(offset).map_err(|_| overflow)
}

fn automatic_model_request(
    configuration: &OracleConfiguration,
) -> Result<RegArimaX11Request, RustError> {
    let OracleConfiguration::AutomaticModelX11 {
        transform,
        mode,
        forecast_horizon,
        backcast_horizon,
        calendar,
        outliers,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
        ..
    } = *configuration
    else {
        return Err(RustError::UnsupportedConfiguration);
    };
    let selection = match transform {
        crate::OracleAutomaticModelTransform::Explicit { transform } => {
            x13::regarima::TransformSelection::Explicit(match transform {
                crate::OracleTransform::None => x13::Transform::None,
                crate::OracleTransform::Log => x13::Transform::Log,
            })
        }
        crate::OracleAutomaticModelTransform::Automatic { options } => {
            x13::regarima::TransformSelection::Automatic(
                x13::regarima::AutoTransformOptions::new()
                    .with_aic_difference(options.aic_difference())?,
            )
        }
    };
    let forecast = match forecast_horizon {
        crate::OracleForecastHorizon::OneYear => x13::regarima::ForecastHorizon::OneYear,
        crate::OracleForecastHorizon::Periods(periods) => {
            x13::regarima::ForecastHorizon::Periods(periods)
        }
    };
    let mut requested = x13::RegArimaSpec::automatic_model()
        .with_transform(selection)
        .with_forecast(x13::regarima::ForecastSpec::new(
            forecast,
            backcast_horizon,
        )?);
    if let Some(calendar) = calendar {
        requested = requested.with_calendar(rust_calendar_spec(calendar)?);
    }
    if let Some(outliers) = outliers {
        requested = with_rust_outliers(requested, outliers)?;
    }
    let options = rust_x11_options(
        RustSeasonalRequest {
            decomposition: configuration.decomposition(),
            filter: seasonal_filter,
            filters: seasonal_filters,
            short_series_filters: rust_short_series_filters(configuration),
            shrinkage: configuration.seasonal_shrinkage(),
        },
        trend_filter,
        sigma_limits,
        configuration.extreme_estimation_span(),
        configuration.extreme_scale(),
        configuration.extreme_estimator(),
    )?;
    Ok(RegArimaX11Request {
        regarima: requested,
        options: with_legacy_easter(
            with_final_x11_requests(options, configuration),
            configuration,
        ),
        mode: mode.map(rust_x11_mode),
    })
}

fn explicit_model_request(
    configuration: &OracleConfiguration,
) -> Result<RegArimaX11Request, RustError> {
    let original_configuration = configuration;
    let short_series_filters = rust_short_series_filters(configuration);
    let Some(configuration) = configuration.regarima() else {
        return Err(RustError::UnsupportedConfiguration);
    };
    let transform = configuration.transform;
    let automatic_transform = configuration.automatic_transform;
    let model = configuration.model;
    let include_constant = configuration.include_constant;
    let forecast_horizon = configuration.forecast_horizon;
    let backcast_horizon = configuration.backcast_horizon;
    let mode = configuration.mode;
    let seasonal_filter = configuration.seasonal_filter;
    let trend_filter = configuration.trend_filter;
    let sigma_limits = configuration.sigma_limits;
    let regular = model.regular();
    let seasonal = model.seasonal();
    let model = x13::ArimaModel::new(
        x13::ArimaOrder::new(regular.p(), regular.d(), regular.q())?,
        x13::ArimaOrder::new(seasonal.p(), seasonal.d(), seasonal.q())?,
    )?;
    let transform = match transform {
        crate::OracleTransform::None => x13::Transform::None,
        crate::OracleTransform::Log => x13::Transform::Log,
    };
    let forecast_horizon = match forecast_horizon {
        crate::OracleForecastHorizon::OneYear => x13::regarima::ForecastHorizon::OneYear,
        crate::OracleForecastHorizon::Periods(periods) => {
            x13::regarima::ForecastHorizon::Periods(periods)
        }
    };
    let extensions = x13::regarima::ForecastSpec::new(forecast_horizon, backcast_horizon)?;
    let transform_selection = match automatic_transform {
        Some(options) => x13::regarima::TransformSelection::Automatic(
            x13::regarima::AutoTransformOptions::new()
                .with_aic_difference(options.aic_difference())?,
        ),
        None => x13::regarima::TransformSelection::Explicit(transform),
    };
    let mut regarima = x13::RegArimaSpec::new(model)
        .with_transform(transform_selection)
        .with_constant(include_constant)
        .with_forecast(extensions);
    if let Some(calendar) = configuration.calendar {
        regarima = regarima.with_calendar(rust_calendar_spec(calendar)?);
    }
    if let Some(outliers) = configuration.outliers {
        regarima = with_rust_outliers(regarima, outliers)?;
    }
    let options = rust_x11_options(
        RustSeasonalRequest {
            decomposition: configuration.decomposition,
            filter: seasonal_filter,
            filters: configuration.seasonal_filters,
            short_series_filters,
            shrinkage: configuration.seasonal_shrinkage,
        },
        trend_filter,
        sigma_limits,
        configuration.extreme_estimation_span,
        configuration.extreme_scale,
        configuration.extreme_estimator,
    )?;
    Ok(RegArimaX11Request {
        regarima,
        options: with_legacy_easter(
            with_final_x11_requests(options, original_configuration),
            original_configuration,
        ),
        // An automatic transform chooses the mode with the transform.
        mode: automatic_transform.is_none().then(|| rust_x11_mode(mode)),
    })
}

struct RustSeasonalRequest {
    decomposition: Option<crate::OracleX11Decomposition>,
    shrinkage: Option<crate::OracleX11SeasonalShrinkage>,
    filter: Option<crate::X11SeasonalFilter>,
    filters: Option<crate::OracleX11SeasonalFilters>,
    short_series_filters: x13::x11::ShortSeriesFilters,
}

fn rust_x11_options(
    seasonal: RustSeasonalRequest,
    trend_filter: Option<crate::OracleX11TrendFilter>,
    sigma_limits: Option<crate::OracleSigmaLimits>,
    extreme_estimation_span: Option<crate::OracleX11ExtremeEstimationSpan>,
    extreme_scale: Option<crate::OracleX11ExtremeScale>,
    extreme_estimator: Option<crate::OracleX11ExtremeEstimator>,
) -> Result<x13::x11::Options, RustError> {
    let RustSeasonalRequest {
        decomposition,
        shrinkage,
        filter: seasonal_filter,
        filters: seasonal_filters,
        short_series_filters,
    } = seasonal;
    let convert_filter = |filter| match filter {
        crate::X11SeasonalFilter::X11Default => x13::x11::SeasonalFilter::X11Default,
        crate::X11SeasonalFilter::S3x1 => x13::x11::SeasonalFilter::S3x1,
        crate::X11SeasonalFilter::S3x3 => x13::x11::SeasonalFilter::S3x3,
        crate::X11SeasonalFilter::S3x5 => x13::x11::SeasonalFilter::S3x5,
        crate::X11SeasonalFilter::S3x9 => x13::x11::SeasonalFilter::S3x9,
        crate::X11SeasonalFilter::S3x15 => x13::x11::SeasonalFilter::S3x15,
        crate::X11SeasonalFilter::Stable => x13::x11::SeasonalFilter::Stable,
        crate::X11SeasonalFilter::Msr => x13::x11::SeasonalFilter::Msr,
    };
    let mut options = x13::x11::Options::new()
        .with_decomposition(match decomposition {
            Some(crate::OracleX11Decomposition::TrendOnly) => x13::x11::Decomposition::TrendOnly,
            None | Some(crate::OracleX11Decomposition::SeasonalAdjustment) => {
                x13::x11::Decomposition::SeasonalAdjustment
            }
        })
        .with_short_series_filters(short_series_filters)
        .with_seasonal_shrinkage(match shrinkage {
            None | Some(crate::OracleX11SeasonalShrinkage::None) => {
                x13::x11::SeasonalShrinkage::None
            }
            Some(crate::OracleX11SeasonalShrinkage::Global) => x13::x11::SeasonalShrinkage::Global,
            Some(crate::OracleX11SeasonalShrinkage::Local) => x13::x11::SeasonalShrinkage::Local,
        });
    if let Some(estimator) = extreme_estimator {
        options = options.with_extreme_estimator(match estimator {
            crate::OracleX11ExtremeEstimator::LogMedianAbsoluteDeviation => {
                x13::x11::ExtremeEstimator::LogMedianAbsoluteDeviation
            }
            crate::OracleX11ExtremeEstimator::TauAdjustedLogMedianAbsoluteDeviation => {
                x13::x11::ExtremeEstimator::TauAdjustedLogMedianAbsoluteDeviation
            }
            crate::OracleX11ExtremeEstimator::StandardDeviation => {
                x13::x11::ExtremeEstimator::StandardDeviation
            }
            crate::OracleX11ExtremeEstimator::MedianAbsoluteDeviation => {
                x13::x11::ExtremeEstimator::MedianAbsoluteDeviation
            }
            crate::OracleX11ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation => {
                x13::x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation
            }
        });
    }
    if let Some(scale) = extreme_scale {
        let convert_group = |group| match group {
            crate::OracleX11SigmaGroup::Selected => x13::x11::SigmaGroup::Selected,
            crate::OracleX11SigmaGroup::Complement => x13::x11::SigmaGroup::Complement,
        };
        options = options.with_extreme_scale(match scale {
            crate::OracleX11ExtremeScale::Automatic => x13::x11::ExtremeScale::Automatic,
            crate::OracleX11ExtremeScale::RollingFiveYear => {
                x13::x11::ExtremeScale::RollingFiveYear
            }
            crate::OracleX11ExtremeScale::ByCalendarPeriod => {
                x13::x11::ExtremeScale::ByCalendarPeriod
            }
            crate::OracleX11ExtremeScale::Grouped { groups } => {
                x13::x11::ExtremeScale::Grouped(match groups {
                    crate::OracleX11SigmaGroups::Monthly(groups) => {
                        x13::x11::SigmaGroups::Monthly(groups.map(convert_group))
                    }
                    crate::OracleX11SigmaGroups::Quarterly(groups) => {
                        x13::x11::SigmaGroups::Quarterly(groups.map(convert_group))
                    }
                })
            }
        });
    }
    if let Some(span) = extreme_estimation_span {
        options = options.with_extreme_estimation_span(match span {
            crate::OracleX11ExtremeEstimationSpan::WithExtensions => {
                x13::x11::ExtremeEstimationSpan::WithExtensions
            }
            crate::OracleX11ExtremeEstimationSpan::ObservedOnly => {
                x13::x11::ExtremeEstimationSpan::ObservedOnly
            }
        });
    }
    if let Some(filters) = seasonal_filters {
        let filters = match filters {
            crate::OracleX11SeasonalFilters::Monthly(filters) => {
                let mut converted = [x13::x11::SeasonalFilter::S3x5; 12];
                for (output, input) in converted.iter_mut().zip(filters) {
                    *output = convert_filter(input);
                }
                x13::x11::SeasonalFilters::Monthly(converted)
            }
            crate::OracleX11SeasonalFilters::Quarterly(filters) => {
                let mut converted = [x13::x11::SeasonalFilter::S3x5; 4];
                for (output, input) in converted.iter_mut().zip(filters) {
                    *output = convert_filter(input);
                }
                x13::x11::SeasonalFilters::Quarterly(converted)
            }
        };
        options = options.with_seasonal_filters(filters);
    } else if let Some(filter) = seasonal_filter {
        options = options.with_seasonal_filter(convert_filter(filter));
    } else if decomposition != Some(crate::OracleX11Decomposition::TrendOnly) {
        return Err(RustError::UnsupportedConfiguration);
    }
    // A trend-only request leaves the seasonal filter unset: its pass ignores
    // the filter, and `adjust` rejects an explicit one.
    if let Some(trend_filter) = trend_filter {
        let trend_filter = match trend_filter {
            crate::OracleX11TrendFilter::Automatic => x13::x11::TrendFilter::Automatic,
            crate::OracleX11TrendFilter::Henderson(length) => {
                x13::x11::TrendFilter::Henderson(x13::x11::HendersonLength::new(length.terms())?)
            }
            crate::OracleX11TrendFilter::AutomaticWithTrueSevenTerm => {
                options = options
                    .with_seven_term_endpoints(x13::x11::SevenTermEndpoints::HendersonSevenTerm);
                x13::x11::TrendFilter::Automatic
            }
            crate::OracleX11TrendFilter::HendersonWithOptions {
                length,
                ic_ratio,
                true_seven_term,
            } => {
                options = options.with_seven_term_endpoints(if true_seven_term {
                    x13::x11::SevenTermEndpoints::HendersonSevenTerm
                } else {
                    x13::x11::SevenTermEndpoints::LegacyFiveTerm
                });
                let length = x13::x11::HendersonLength::new(length.terms())?;
                if let Some(ratio) = ic_ratio {
                    x13::x11::TrendFilter::HendersonWithIcRatio {
                        length,
                        ratio: x13::x11::HendersonIcRatio::new(ratio.value())?,
                    }
                } else {
                    x13::x11::TrendFilter::Henderson(length)
                }
            }
        };
        options = options.with_trend_filter(trend_filter);
    }
    if let Some(limits) = sigma_limits {
        options =
            options.with_sigma_limits(x13::x11::SigmaLimits::new(limits.lower(), limits.upper())?);
    }
    Ok(options)
}

fn rust_outlier_spec(
    outliers: crate::OracleOutlierSpec,
) -> Result<x13::regarima::OutlierSpec, RustError> {
    let mut types = None;
    for outlier_type in [
        crate::OracleOutlierType::Additive,
        crate::OracleOutlierType::LevelShift,
        crate::OracleOutlierType::TemporaryChange,
    ] {
        if outliers.types().contains(outlier_type) {
            let next = match outlier_type {
                crate::OracleOutlierType::Additive => x13::regarima::OutlierTypes::ADDITIVE,
                crate::OracleOutlierType::LevelShift => x13::regarima::OutlierTypes::LEVEL_SHIFT,
                crate::OracleOutlierType::TemporaryChange => {
                    x13::regarima::OutlierTypes::TEMPORARY_CHANGE
                }
            };
            types = Some(types.map_or(next, |current: x13::regarima::OutlierTypes| {
                current.union(next)
            }));
        }
    }
    let types = types.ok_or(RustError::UnsupportedConfiguration)?;
    let method = match outliers.method() {
        crate::OracleOutlierMethod::AddOne => x13::regarima::OutlierMethod::AddOne,
        crate::OracleOutlierMethod::AddAll => x13::regarima::OutlierMethod::AddAll,
    };
    let critical_values = match outliers.critical_values() {
        crate::OracleOutlierCriticalValues::Automatic => {
            x13::regarima::OutlierCriticalValues::Automatic
        }
        crate::OracleOutlierCriticalValues::Shared { value } => {
            x13::regarima::OutlierCriticalValues::shared(x13::regarima::OutlierCriticalValue::new(
                value.value(),
            )?)
        }
        crate::OracleOutlierCriticalValues::PerType {
            additive,
            level_shift,
            temporary_change,
        } => x13::regarima::OutlierCriticalValues::per_type(
            additive
                .map(|value| x13::regarima::OutlierCriticalValue::new(value.value()))
                .transpose()?,
            level_shift
                .map(|value| x13::regarima::OutlierCriticalValue::new(value.value()))
                .transpose()?,
            temporary_change
                .map(|value| x13::regarima::OutlierCriticalValue::new(value.value()))
                .transpose()?,
        ),
    };
    Ok(x13::regarima::OutlierSpec::default()
        .with_types(types)
        .with_method(method)
        .with_critical_values(critical_values))
}

/// Adds the outlier search and its `outlier{tcrate}`, which lands on the
/// request's one temporary-change rate: Build 62 keeps a single `Tcalfa` for
/// both `outlier{tcrate}` and `regression{tcrate}`.
fn with_rust_outliers(
    regarima: x13::RegArimaSpec,
    outliers: crate::OracleOutlierSpec,
) -> Result<x13::RegArimaSpec, RustError> {
    let mut regarima = regarima.with_outliers(rust_outlier_spec(outliers)?);
    if let Some(rate) = outliers.temporary_change_rate() {
        regarima = regarima
            .with_temporary_change_rate(x13::regarima::TemporaryChangeRate::new(rate.value())?);
    }
    Ok(regarima)
}

fn rust_calendar_spec(
    calendar: crate::OracleCalendarSpec,
) -> Result<x13::regarima::CalendarSpec, RustError> {
    let mut result = x13::regarima::CalendarSpec::new();
    if let Some(basis) = calendar.basis() {
        let basis = match basis {
            crate::OracleCalendarBasis::FlowTradingDay(flow) => {
                let length_treatment = match flow.length_treatment() {
                    crate::OracleFlowLengthTreatment::LeapYear => {
                        x13::regarima::FlowLengthTreatment::LeapYear
                    }
                    crate::OracleFlowLengthTreatment::NoLengthEffect => {
                        x13::regarima::FlowLengthTreatment::NoLengthEffect
                    }
                    crate::OracleFlowLengthTreatment::LengthOfPeriod => {
                        x13::regarima::FlowLengthTreatment::LengthOfPeriod
                    }
                };
                x13::regarima::CalendarBasis::FlowTradingDay(
                    x13::regarima::FlowTradingDay::new(rust_trading_day_contrast(flow.contrast()))
                        .with_length_treatment(length_treatment),
                )
            }
            crate::OracleCalendarBasis::StockTradingDay(stock) => {
                x13::regarima::CalendarBasis::StockTradingDay(x13::regarima::StockTradingDay::new(
                    rust_trading_day_contrast(stock.contrast()),
                    stock.sample_day(),
                )?)
            }
            crate::OracleCalendarBasis::Length(effect) => {
                x13::regarima::CalendarBasis::Length(match effect {
                    crate::OracleLengthEffect::LeapYear => x13::regarima::LengthEffect::LeapYear,
                    crate::OracleLengthEffect::LengthOfPeriod => {
                        x13::regarima::LengthEffect::LengthOfPeriod
                    }
                })
            }
        };
        result = result.with_basis(basis);
    }
    if let Some(easter) = calendar.easter() {
        result = result.with_easter(x13::regarima::Easter::new(easter.window())?);
    }
    if let Some(labor_day) = calendar.labor_day() {
        result = result.with_labor_day(x13::regarima::LaborDay::new(labor_day.window())?);
    }
    if let Some(thanksgiving) = calendar.thanksgiving() {
        result = result.with_thanksgiving(x13::regarima::Thanksgiving::new(thanksgiving.offset())?);
    }
    Ok(result)
}

const fn rust_trading_day_contrast(
    contrast: crate::OracleTradingDayContrast,
) -> x13::regarima::TradingDayContrast {
    match contrast {
        crate::OracleTradingDayContrast::SixCoefficient => {
            x13::regarima::TradingDayContrast::SixCoefficient
        }
        crate::OracleTradingDayContrast::WeekdayWeekend => {
            x13::regarima::TradingDayContrast::WeekdayWeekend
        }
    }
}
