//! Renders the Build 62 spec file and input file of a case.

use std::fmt::Write as _;

use crate::configuration::OracleRegArimaConfiguration;
use crate::*;

mod automatic_model;
mod regarima_request;

pub(crate) use regarima_request::uses_summary;

const X13_MAX_INPUT_LINE_LENGTH: usize = 132;

/// The input file's name, which the spec file gives as `series { file }`.
pub(crate) const INPUT_FILE: &str = "input.dat";

const INPUT_DATA_FORMAT: &str = "free";

/// The spec file that asks Build 62 for `configuration` on `series`.
pub(crate) fn render(
    series: &Series,
    configuration: &OracleConfiguration,
) -> Result<String, OracleError> {
    if let OracleConfiguration::RegArima { method, request } = *configuration {
        return regarima_request::render(series, method, request);
    }
    if series.values.is_empty() {
        return Err(OracleError::EmptySeries);
    }
    validate_series_values(series)?;
    if series.frequency == Frequency::Semiannual {
        return Err(OracleError::UnsupportedX11Frequency(series.frequency));
    }
    configuration.validate_for_frequency(series.frequency)?;
    let title = series
        .series_id
        .replace(['\n', '\r'], " ")
        .replace('"', "'");
    let mut spec = String::new();
    writeln!(spec, "series {{").unwrap();
    let regarima = configuration.regarima();
    if regarima.is_some() || configuration.is_automatic_model() {
        writeln!(spec, "  saveprecision = 15").unwrap();
    }
    if configuration.automatic_model_calendar().is_some() {
        writeln!(spec, "  save = (calendaradjorig outlieradjorig)").unwrap();
    } else if configuration.automatic_model_outliers().is_some() {
        writeln!(spec, "  save = outlieradjorig").unwrap();
    }
    if let Some(regarima) = regarima
        && configuration.is_calendar_outlier_regarima()
    {
        let mut series_tables = Vec::with_capacity(2);
        if regarima
            .calendar
            .is_some_and(|calendar| !calendar.is_empty())
        {
            series_tables.push("calendaradjorig");
        }
        if regarima.outliers.is_some() {
            series_tables.push("outlieradjorig");
        }
        if !series_tables.is_empty() {
            writeln!(spec, "  save = ({})", series_tables.join(" ")).unwrap();
        }
    }
    writeln!(spec, "  title = \"{title}\"").unwrap();
    writeln!(
        spec,
        "  start = {}.{}",
        series.start.year, series.start.subperiod
    )
    .unwrap();
    writeln!(spec, "  period = {}", series.frequency.periods_per_year()).unwrap();
    writeln!(spec, "  file = \"{INPUT_FILE}\"").unwrap();
    writeln!(spec, "  format = {INPUT_DATA_FORMAT}").unwrap();
    writeln!(spec, "}}\n").unwrap();
    if configuration.is_automatic_model() {
        automatic_model::render_blocks(&mut spec, series.frequency, configuration)?;
        return Ok(spec);
    }
    if let Some(regarima) = regarima {
        render_explicit_regarima_blocks(
            &mut spec,
            series.frequency,
            regarima,
            configuration.legacy_easter(),
        )?;
        return Ok(spec);
    }
    let OracleConfiguration::DirectX11 {
        mode,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
        ..
    } = *configuration
    else {
        unreachable!("every other configuration was rendered above")
    };
    writeln!(spec, "x11 {{").unwrap();
    render_x11_decomposition(&mut spec, configuration.decomposition());
    render_legacy_easter(&mut spec, configuration.legacy_easter());
    writeln!(spec, "  mode = {}", mode.spec_value()).unwrap();
    render_x11_options(
        &mut spec,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
    );
    render_x11_final_outliers(&mut spec, configuration.final_outliers());
    render_x11_holiday_adjustment(&mut spec, configuration.holiday_adjustment());
    render_x11_short_series_filters(&mut spec, configuration.short_series_filters());
    render_x11_print(
        &mut spec,
        configuration.requests_msr(),
        configuration.requests_automatic_calendar_sigma(),
    );
    render_x11_extreme_settings(
        &mut spec,
        configuration.extreme_estimation_span(),
        configuration.extreme_scale(),
        configuration.extreme_estimator(),
        X11SavedTables {
            trend_only: configuration.is_trend_only(),
            extreme_weights: configuration.requests_x11_extreme_weights(),
            log_additive: mode == X11Mode::LogAdditive,
            shrinkage: configuration.seasonal_shrinkage(),
        },
    );
    writeln!(spec, "}}").unwrap();
    Ok(spec)
}

/// Renders the free-format input file Build 62 reads the series from.
pub(crate) fn render_input(series: &Series) -> Result<String, OracleError> {
    validate_series_values(series)?;
    let mut data = String::new();
    for value in &series.values {
        writeln!(data, "{value:.17e}").unwrap();
    }
    Ok(data)
}

fn validate_series_values(series: &Series) -> Result<(), OracleError> {
    for &value in &series.values {
        if !value.is_finite() {
            return Err(OracleError::NonFiniteValue {
                series_id: series.series_id.clone(),
                value,
            });
        }
    }
    Ok(())
}

fn render_explicit_regarima_blocks(
    spec: &mut String,
    frequency: Frequency,
    configuration: OracleRegArimaConfiguration,
    legacy_easter: Option<crate::OracleX11Easter>,
) -> Result<(), OracleError> {
    let OracleRegArimaConfiguration {
        decomposition,
        seasonal_shrinkage,
        final_outliers,
        temporary_change_allocation,
        holiday_adjustment,
        short_series_filters,
        transform,
        automatic_transform,
        model,
        include_constant,
        calendar,
        outliers,
        forecast_horizon,
        backcast_horizon,
        mode,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
        extreme_estimation_span,
        extreme_scale,
        extreme_estimator,
    } = configuration;
    writeln!(spec, "transform {{").unwrap();
    if let Some(automatic_transform) = automatic_transform {
        writeln!(spec, "  function = auto").unwrap();
        writeln!(spec, "  aicdiff = {}", automatic_transform.aic_difference()).unwrap();
        writeln!(spec, "  print = aictransform").unwrap();
    } else {
        writeln!(spec, "  function = {}", transform.spec_value()).unwrap();
    }
    writeln!(spec, "  save = transformed").unwrap();
    writeln!(spec, "}}\n").unwrap();

    let mut regression_variables = Vec::new();
    if include_constant {
        regression_variables.push("const".to_owned());
    }
    if let Some(calendar) = calendar {
        regression_variables.extend(calendar_variables(calendar, frequency));
    }
    if !regression_variables.is_empty() || outliers.is_some() {
        writeln!(spec, "regression {{").unwrap();
        if regression_variables.len() == 1 {
            writeln!(spec, "  variables = {}", regression_variables[0]).unwrap();
        } else if !regression_variables.is_empty() {
            writeln!(spec, "  variables = ({})", regression_variables.join(" ")).unwrap();
        }
        render_temporary_change_allocation(spec, temporary_change_allocation);
        writeln!(spec, "  save = regressionmatrix").unwrap();
        writeln!(spec, "}}\n").unwrap();
    }

    let has_regression = !regression_variables.is_empty() || outliers.is_some();

    let regular = model.regular();
    let seasonal = model.seasonal();
    writeln!(spec, "arima {{").unwrap();
    writeln!(
        spec,
        "  model = ({} {} {})({} {} {})",
        regular.p(),
        regular.d(),
        regular.q(),
        seasonal.p(),
        seasonal.d(),
        seasonal.q()
    )
    .unwrap();
    writeln!(spec, "}}\n").unwrap();

    writeln!(spec, "estimate {{").unwrap();
    writeln!(spec, "  exact = arma").unwrap();
    if has_regression {
        if automatic_transform.is_some() {
            writeln!(spec, "  save = (estimates armacmatrix lkstats regressioneffects regressionresiduals residuals)").unwrap();
        } else {
            writeln!(spec, "  save = (estimates iterations armacmatrix lkstats regressioneffects regressionresiduals residuals)").unwrap();
        }
    } else if automatic_transform.is_some() {
        writeln!(spec, "  save = (estimates armacmatrix lkstats residuals)").unwrap();
    } else {
        writeln!(
            spec,
            "  save = (estimates iterations armacmatrix lkstats residuals)"
        )
        .unwrap();
    }
    writeln!(spec, "}}\n").unwrap();

    if let Some(outliers) = outliers {
        render_outlier_block(spec, outliers);
    }

    let forecast_horizon = forecast_horizon.resolve(frequency)?;
    writeln!(spec, "forecast {{").unwrap();
    writeln!(spec, "  maxlead = {forecast_horizon}").unwrap();
    writeln!(spec, "  maxback = {backcast_horizon}").unwrap();
    writeln!(spec, "  lognormal = no").unwrap();
    let mut forecast_tables = Vec::with_capacity(5);
    if forecast_horizon > 0 {
        forecast_tables.push("transformed");
        if has_regression {
            forecast_tables.push("variances");
        }
        forecast_tables.push("forecasts");
    }
    if backcast_horizon > 0 {
        forecast_tables.extend(["transformedbcst", "backcasts"]);
    }
    if !forecast_tables.is_empty() {
        writeln!(spec, "  save = ({})", forecast_tables.join(" ")).unwrap();
    }
    writeln!(spec, "}}\n").unwrap();

    writeln!(spec, "x11 {{").unwrap();
    render_x11_decomposition(spec, decomposition);
    render_legacy_easter(spec, legacy_easter);
    if automatic_transform.is_none() {
        writeln!(spec, "  mode = {}", mode.spec_value()).unwrap();
    }
    render_x11_options(
        spec,
        seasonal_filter,
        seasonal_filters,
        trend_filter,
        sigma_limits,
    );
    render_x11_final_outliers(spec, final_outliers);
    render_x11_holiday_adjustment(spec, holiday_adjustment);
    render_x11_short_series_filters(spec, short_series_filters);
    render_x11_append_policy(spec);
    render_x11_print(
        spec,
        requests_msr(seasonal_filter, seasonal_filters),
        matches!(extreme_scale, Some(crate::OracleX11ExtremeScale::Automatic)),
    );
    render_x11_extreme_settings(
        spec,
        extreme_estimation_span,
        extreme_scale,
        extreme_estimator,
        X11SavedTables {
            trend_only: decomposition == Some(crate::OracleX11Decomposition::TrendOnly),
            extreme_weights: final_outliers.is_some()
                || mode == crate::X11Mode::PseudoAdditive
                || temporary_change_allocation.is_some()
                || holiday_adjustment.is_some()
                || short_series_filters.is_some(),
            log_additive: mode == crate::X11Mode::LogAdditive,
            shrinkage: seasonal_shrinkage,
        },
    );
    writeln!(spec, "}}").unwrap();
    Ok(())
}

fn render_x11_options(
    spec: &mut String,
    seasonal_filter: Option<crate::X11SeasonalFilter>,
    seasonal_filters: Option<crate::OracleX11SeasonalFilters>,
    trend_filter: Option<crate::OracleX11TrendFilter>,
    sigma_limits: Option<crate::OracleSigmaLimits>,
) {
    if let Some(filters) = seasonal_filters {
        let values = filters
            .filters()
            .iter()
            .map(|filter| filter.spec_value())
            .collect::<Vec<_>>();
        writeln!(spec, "  seasonalma = ({})", values.join(" ")).unwrap();
    } else if let Some(seasonal_filter) = seasonal_filter {
        writeln!(spec, "  seasonalma = {}", seasonal_filter.spec_value()).unwrap();
    }
    match trend_filter {
        Some(crate::OracleX11TrendFilter::Henderson(length)) => {
            writeln!(spec, "  trendma = {}", length.terms()).unwrap();
        }
        Some(crate::OracleX11TrendFilter::AutomaticWithTrueSevenTerm) => {
            writeln!(spec, "  true7term = yes").unwrap();
        }
        Some(crate::OracleX11TrendFilter::HendersonWithOptions {
            length,
            ic_ratio,
            true_seven_term,
        }) => {
            writeln!(spec, "  trendma = {}", length.terms()).unwrap();
            if let Some(ratio) = ic_ratio {
                let value = ratio.value();
                let decimal = value.to_string();
                if "  trendic = ".len() + decimal.len() > X13_MAX_INPUT_LINE_LENGTH {
                    writeln!(spec, "  trendic = {value:e}").unwrap();
                } else {
                    writeln!(spec, "  trendic = {decimal}").unwrap();
                }
            }
            writeln!(
                spec,
                "  true7term = {}",
                if true_seven_term { "yes" } else { "no" }
            )
            .unwrap();
        }
        None | Some(crate::OracleX11TrendFilter::Automatic) => {}
    }
    if let Some(limits) = sigma_limits {
        writeln!(spec, "  sigmalim = ({} {})", limits.lower(), limits.upper()).unwrap();
    }
}

/// The saved X-11 tables stop at the observed span, as Rust's components do.
fn render_x11_append_policy(spec: &mut String) {
    writeln!(spec, "  appendfcst = no").unwrap();
    writeln!(spec, "  appendbcst = no").unwrap();
}

fn render_legacy_easter(spec: &mut String, legacy_easter: Option<crate::OracleX11Easter>) {
    if let Some(easter) = legacy_easter {
        let keyword = match easter {
            crate::OracleX11Easter::None => "no",
            crate::OracleX11Easter::Estimate => "yes",
        };
        writeln!(spec, "  x11easter = {keyword}").unwrap();
    }
}

fn render_x11_decomposition(
    spec: &mut String,
    decomposition: Option<crate::OracleX11Decomposition>,
) {
    if decomposition == Some(crate::OracleX11Decomposition::TrendOnly) {
        writeln!(spec, "  type = trend").unwrap();
    }
}

fn render_x11_final_outliers(
    spec: &mut String,
    final_outliers: Option<crate::OracleX11FinalOutliers>,
) {
    if let Some(crate::OracleX11FinalOutliers::Remove { types }) = final_outliers {
        let tokens = types
            .iter()
            .map(|kind| kind.spec_value())
            .collect::<Vec<_>>();
        writeln!(spec, "  final = ({})", tokens.join(" ")).unwrap();
    }
}

fn render_x11_holiday_adjustment(
    spec: &mut String,
    policy: Option<crate::OracleX11HolidayAdjustment>,
) {
    if let Some(policy) = policy {
        let value = match policy {
            crate::OracleX11HolidayAdjustment::Remove => "no",
            crate::OracleX11HolidayAdjustment::Retain => "yes",
        };
        writeln!(spec, "  keepholiday = {value}").unwrap();
    }
}

fn render_x11_short_series_filters(
    spec: &mut String,
    policy: Option<crate::OracleX11ShortSeriesFilters>,
) {
    if let Some(policy) = policy {
        let value = match policy {
            crate::OracleX11ShortSeriesFilters::StableFallback => "no",
            crate::OracleX11ShortSeriesFilters::UseRequested => "yes",
        };
        writeln!(spec, "  sfshort = {value}").unwrap();
    }
}

fn render_temporary_change_allocation(
    spec: &mut String,
    allocation: Option<crate::OracleX11TemporaryChangeAllocation>,
) {
    if let Some(allocation) = allocation {
        let keyword = match allocation {
            crate::OracleX11TemporaryChangeAllocation::Irregular => "no",
            crate::OracleX11TemporaryChangeAllocation::Trend => "yes",
        };
        writeln!(spec, "  trendtc = {keyword}").unwrap();
    }
}

struct X11SavedTables {
    trend_only: bool,
    extreme_weights: bool,
    log_additive: bool,
    shrinkage: Option<crate::OracleX11SeasonalShrinkage>,
}

fn render_x11_extreme_settings(
    spec: &mut String,
    span: Option<crate::OracleX11ExtremeEstimationSpan>,
    scale: Option<crate::OracleX11ExtremeScale>,
    estimator: Option<crate::OracleX11ExtremeEstimator>,
    saved: X11SavedTables,
) {
    if let Some(estimator) = estimator {
        writeln!(spec, "  extremeadj = {}", estimator.spec_value()).unwrap();
    }
    if let Some(span) = span {
        let exclude = match span {
            crate::OracleX11ExtremeEstimationSpan::WithExtensions => "no",
            crate::OracleX11ExtremeEstimationSpan::ObservedOnly => "yes",
        };
        writeln!(spec, "  excludefcst = {exclude}").unwrap();
    }
    if let Some(scale) = scale {
        let value = match scale {
            crate::OracleX11ExtremeScale::Automatic => "signif",
            crate::OracleX11ExtremeScale::RollingFiveYear => "none",
            crate::OracleX11ExtremeScale::ByCalendarPeriod => "all",
            crate::OracleX11ExtremeScale::Grouped { .. } => "select",
        };
        writeln!(spec, "  calendarsigma = {value}").unwrap();
        if let crate::OracleX11ExtremeScale::Grouped { groups } = scale {
            let selected = groups.selected_names();
            if !selected.is_empty() {
                writeln!(spec, "  sigmavec = ({})", selected.join(" ")).unwrap();
            }
        }
    }
    let tables = if saved.log_additive {
        "d10 d11 d12 d13 b17 c17 c13 bcf"
    } else if estimator.is_some()
        || crate::configuration::requests_x11_extreme_weights(span, scale)
        || saved.extreme_weights
    {
        "d10 d11 d12 d13 b17 c17"
    } else {
        "d10 d11 d12 d13"
    };
    let shrink = match saved.shrinkage {
        Some(crate::OracleX11SeasonalShrinkage::Global) => Some("global"),
        Some(crate::OracleX11SeasonalShrinkage::Local) => Some("local"),
        _ => None,
    };
    if let Some(shrink) = shrink {
        writeln!(spec, "  shrink = {shrink}").unwrap();
    }
    let sns = if shrink.is_some() && !saved.trend_only {
        " sns"
    } else {
        ""
    };
    writeln!(spec, "  save = ({tables}{sns})").unwrap();
}

fn requests_msr(
    filter: Option<crate::X11SeasonalFilter>,
    filters: Option<crate::OracleX11SeasonalFilters>,
) -> bool {
    filters.map_or(filter == Some(crate::X11SeasonalFilter::Msr), |filters| {
        filters.filters().contains(&crate::X11SeasonalFilter::Msr)
    })
}

fn render_x11_print(spec: &mut String, msr: bool, automatic_calendar_sigma: bool) {
    writeln!(
        spec,
        "  print = {}",
        match (msr, automatic_calendar_sigma) {
            (false, false) => "d12",
            (true, false) => "(d12 autosf)",
            (false, true) => "(d12 b4 b17)",
            (true, true) => "(d12 autosf b4 b17)",
        }
    )
    .unwrap();
}

pub(super) fn calendar_variables(
    calendar: OracleCalendarSpec,
    frequency: Frequency,
) -> Vec<String> {
    let mut variables = Vec::new();
    if let Some(basis) = calendar.basis() {
        match basis {
            OracleCalendarBasis::FlowTradingDay(flow) => {
                let no_length_alias = match flow.contrast() {
                    OracleTradingDayContrast::SixCoefficient => "tdnolpyear",
                    OracleTradingDayContrast::WeekdayWeekend => "td1nolpyear",
                };
                match flow.length_treatment() {
                    OracleFlowLengthTreatment::LeapYear => {
                        variables.push(
                            match flow.contrast() {
                                OracleTradingDayContrast::SixCoefficient => "td",
                                OracleTradingDayContrast::WeekdayWeekend => "td1coef",
                            }
                            .to_owned(),
                        );
                    }
                    OracleFlowLengthTreatment::NoLengthEffect => {
                        variables.push(no_length_alias.to_owned());
                    }
                    OracleFlowLengthTreatment::LengthOfPeriod => {
                        variables.push(no_length_alias.to_owned());
                        variables.push(length_of_period_alias(frequency).to_owned());
                    }
                }
            }
            OracleCalendarBasis::StockTradingDay(stock) => {
                variables.push(stock_trading_day_alias(stock));
            }
            OracleCalendarBasis::Length(length) => variables.push(
                match length {
                    OracleLengthEffect::LeapYear => "lpyear",
                    OracleLengthEffect::LengthOfPeriod => length_of_period_alias(frequency),
                }
                .to_owned(),
            ),
        }
    }
    if let Some(easter) = calendar.easter() {
        variables.push(format!("easter[{}]", easter.window()));
    }
    if let Some(labor_day) = calendar.labor_day() {
        variables.push(format!("labor[{}]", labor_day.window()));
    }
    if let Some(thanksgiving) = calendar.thanksgiving() {
        variables.push(format!("thank[{}]", thanksgiving.offset()));
    }
    variables
}

fn length_of_period_alias(frequency: Frequency) -> &'static str {
    match frequency {
        Frequency::Monthly => "lom",
        Frequency::Quarterly => "loq",
        Frequency::Semiannual => {
            unreachable!("semiannual X-11 configurations are rejected before rendering")
        }
    }
}

fn stock_trading_day_alias(stock: OracleStockTradingDay) -> String {
    let alias = match stock.contrast() {
        OracleTradingDayContrast::SixCoefficient => "tdstock",
        OracleTradingDayContrast::WeekdayWeekend => "tdstock1coef",
    };
    format!("{alias}[{}]", stock.sample_day())
}

pub(crate) fn render_outlier_block(spec: &mut String, outliers: crate::OracleOutlierSpec) {
    writeln!(spec, "outlier {{").unwrap();
    render_outlier_settings(spec, outliers);
    writeln!(spec, "  save = (iterations finaltests)").unwrap();
    writeln!(spec, "  print = tests").unwrap();
    writeln!(spec, "}}\n").unwrap();
}

fn render_outlier_settings(spec: &mut String, outliers: crate::OracleOutlierSpec) {
    let types = outliers
        .types()
        .iter()
        .map(OracleOutlierType::spec_value)
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(spec, "  types = ({types})").unwrap();
    writeln!(spec, "  method = {}", outliers.method().spec_value()).unwrap();
    match outliers.critical_values() {
        OracleOutlierCriticalValues::Automatic => {}
        OracleOutlierCriticalValues::Shared { value } => {
            writeln!(spec, "  critical = {}", value.value()).unwrap();
        }
        OracleOutlierCriticalValues::PerType {
            additive,
            level_shift,
            temporary_change,
        } => {
            let values = [additive, level_shift, temporary_change]
                .map(|value| value.map_or_else(String::new, |value| value.value().to_string()));
            writeln!(spec, "  critical = ({})", values.join(",")).unwrap();
        }
    }
    if let Some(rate) = outliers.temporary_change_rate() {
        writeln!(spec, "  tcrate = {}", rate.value()).unwrap();
    }
}
