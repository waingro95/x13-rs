//! Parses Build 62's saved tables into an [`Output`] and builds the same
//! [`Output`] from Rust's `Adjustment`.
//!
//! # Unrecorded columns
//!
//! Backcasts and the final SEATS model are compared without entering the
//! recorded Rust output: `rust_output` leaves both fields empty, serde skips
//! them when empty, and `compare_unrecorded` reads Rust's values straight
//! from the live `Adjustment`. Every `PARITY_COMPARE_RUST` record therefore
//! stays byte-identical.
//!
//! The SEATS model comes from `savelog = seatsmodel` (seatdg.f), which logs
//! once per run the orders and ARMA coefficients SEATS decomposed with, after
//! any model change or bound fix. Only each coefficient's value is read; a
//! second model block is a parse error rather than a silent choice. The
//! requested and final mean, refit flag, temporary mean removal and
//! bound-fixed parameters are printed only in the `.out` text, so they are not
//! compared. Both columns reuse existing tolerance kinds: backcasts use the
//! forecast print-floor tolerance, SEATS coefficients the ARMA-estimate one.
//!
//! # Exact comparison of the whole `Adjustment`
//!
//! A Rust record also carries `adjustment_debug`, the whole `Adjustment`'s
//! `Debug` text. Exact compare then sees every public output (diagnostics,
//! warnings and their order, backcasts, the SEATS model), not only the call
//! table's columns, so a refactor cannot change an uncompared value. Float
//! `Debug` is the shortest round-trip text, which is bit-exact apart from NaN
//! payloads. Tolerant compare ignores it: it is not one of the table's
//! columns.

use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::{
    Frequency, OracleConfiguration, OracleOutlierType, OracleOutlierVariable, Period, Series,
    SeriesPoint,
};

/// The four components every row compares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Components {
    pub(crate) seasonal: Vec<SeriesPoint>,
    pub(crate) seasonally_adjusted: Vec<SeriesPoint>,
    pub(crate) trend_cycle: Vec<SeriesPoint>,
    pub(crate) irregular: Vec<SeriesPoint>,
}

// The `Debug` names of these two are part of a coefficient's name.
#[derive(Debug, Clone, Copy)]
enum ParameterKind {
    Ar,
    Ma,
}

#[derive(Debug, Clone, Copy)]
enum Factor {
    Nonseasonal,
    Seasonal,
}

/// Only the outputs in the call table; no diagnostics or uncertainty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Output {
    pub(crate) components: Components,
    pub(crate) transitory: Option<Vec<SeriesPoint>>,
    pub(crate) model: Option<Model>,
    pub(crate) forecasts: Vec<SeriesPoint>,
    /// Unrecorded columns. `rust_output` leaves them
    /// empty, so Rust bit records stay byte-identical; `compare_unrecorded`
    /// reads Rust's values from the live `Adjustment`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) backcasts: Vec<SeriesPoint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) seats_model: Option<SeatsModel>,
    /// Rust only: the whole `Adjustment`'s `Debug` text, so exact compare
    /// also covers diagnostics, warnings, backcasts and the SEATS model.
    /// Build 62's output leaves it empty; tolerant compare ignores it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) adjustment_debug: String,
}

/// The model SEATS decomposed with, after any SEATS change or bound fix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SeatsModel {
    pub(crate) orders: [[u8; 3]; 2],
    /// Nonseasonal AR, seasonal AR, nonseasonal MA, then seasonal MA, each in
    /// `(1 - c B)` sign convention and named as seatdg.f logs them.
    pub(crate) coefficients: Vec<Coefficient>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Model {
    pub(crate) transform: String,
    pub(crate) orders: [[u8; 3]; 2],
    pub(crate) arma: Vec<Coefficient>,
    pub(crate) regression: Vec<Coefficient>,
    pub(crate) outliers: Vec<OracleOutlierVariable>,
    /// Whether `arma` holds 15 significant digits and so takes the tighter
    /// of the two saved-scalar tolerances. Build 62 only; a Rust record
    /// stores `false`.
    pub(crate) strict_arma_gate: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Coefficient {
    pub(crate) name: String,
    pub(crate) value: f64,
}

pub(crate) fn rust_output(series: &Series, adjustment: &x13::Adjustment) -> Result<Output> {
    let start = start_period(series)?;
    // A component Build 62 omits (SEATS S10 or S12) records as an empty table.
    let components = Components {
        seasonal: points(
            start,
            adjustment
                .seasonal()
                .map(x13::Series::values)
                .unwrap_or_default(),
        )?,
        seasonally_adjusted: points(start, adjustment.seasonally_adjusted().values())?,
        trend_cycle: points(
            start,
            adjustment
                .trend_cycle()
                .map(x13::Series::values)
                .unwrap_or_default(),
        )?,
        irregular: points(start, adjustment.irregular().values())?,
    };
    // A transform or outlier type this harness does not know is an error.
    let model = adjustment.model().map(|model| -> Result<Model> {
        let order = |o: x13::regarima::ResolvedArimaOrder| [o.p(), o.d(), o.q()];
        Ok(Model {
            transform: match model.transform() {
                x13::Transform::None => "none",
                x13::Transform::Log => "log",
                other => bail!("unknown Rust transform {other:?}"),
            }
            .into(),
            orders: [
                order(model.model().regular()),
                order(model.model().seasonal()),
            ],
            arma: model
                .parameters()
                .iter()
                .map(|p| Coefficient {
                    name: parameter_name(
                        match p.kind() {
                            x13::regarima::ParameterKind::Ar => ParameterKind::Ar,
                            x13::regarima::ParameterKind::Ma => ParameterKind::Ma,
                        },
                        match p.factor() {
                            x13::regarima::ArimaFactor::Regular => Factor::Nonseasonal,
                            x13::regarima::ArimaFactor::Seasonal => Factor::Seasonal,
                        },
                        p.period(),
                        p.lag(),
                    ),
                    value: p.estimate(),
                })
                .collect(),
            regression: model
                .regression_coefficients()
                .iter()
                .map(|p| Coefficient {
                    name: p.name().into(),
                    value: p.estimate(),
                })
                .collect(),
            outliers: adjustment
                .outliers()
                .iter()
                .map(|o| {
                    Ok(OracleOutlierVariable {
                        outlier_type: match o.kind() {
                            x13::regarima::OutlierType::Additive => OracleOutlierType::Additive,
                            x13::regarima::OutlierType::LevelShift => OracleOutlierType::LevelShift,
                            x13::regarima::OutlierType::TemporaryChange => {
                                OracleOutlierType::TemporaryChange
                            }
                            other => bail!("unknown Rust outlier type {other:?}"),
                        },
                        period: Period::new(o.period().year(), o.period().subperiod()),
                    })
                })
                .collect::<Result<_>>()?,
            strict_arma_gate: false,
        })
    });
    let model = model.transpose()?;
    if let Some(model) = &model {
        ensure!(
            model
                .arma
                .iter()
                .chain(&model.regression)
                .all(|p| p.value.is_finite()),
            "nonfinite Rust coefficient"
        );
    }
    let forecasts = adjustment
        .forecasts()
        .map(|forecast| {
            let mut input = series.clone();
            input.start = Period::new(forecast.start().year(), forecast.start().subperiod());
            points(
                period_for(input.frequency, input.start)?,
                forecast.series().values(),
            )
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Output {
        components,
        transitory: adjustment
            .seats_transitory()
            .map(|component| points(start, component.series().values()))
            .transpose()?,
        model,
        forecasts,
        backcasts: vec![],
        seats_model: None,
        adjustment_debug: format!("{adjustment:?}"),
    })
}

/// Rust's point backcasts, for the unrecorded backcast column.
pub(crate) fn rust_backcasts(
    series: &Series,
    adjustment: &x13::Adjustment,
) -> Result<Vec<SeriesPoint>> {
    adjustment.backcasts().map_or(Ok(vec![]), |backcasts| {
        let start = Period::new(backcasts.start().year(), backcasts.start().subperiod());
        points(
            period_for(series.frequency, start)?,
            backcasts.series().values(),
        )
    })
}

/// Rust's final SEATS model in the order and names seatdg.f logs.
pub(crate) fn rust_seats_model(adjustment: &x13::Adjustment) -> Option<SeatsModel> {
    let model = adjustment.diagnostics().seats()?.model();
    let order = |o: x13::regarima::ResolvedArimaOrder| [o.p(), o.d(), o.q()];
    let orders = [
        order(model.orders().regular()),
        order(model.orders().seasonal()),
    ];
    // seatdg.f's labels follow the order of `coefficients()`.
    let names = seats_coefficient_names(orders);
    Some(SeatsModel {
        orders,
        coefficients: names
            .into_iter()
            .zip(model.coefficients().iter().copied())
            .map(|(name, value)| Coefficient { name, value })
            .collect(),
    })
}

/// seatdg.f's log labels: AR before MA, nonseasonal before seasonal, each
/// numbered from one (seasonal lags are not multiplied by the period).
fn seats_coefficient_names([regular, seasonal]: [[u8; 3]; 2]) -> Vec<String> {
    [
        ("Nonseasonal AR", regular[0]),
        ("Seasonal AR", seasonal[0]),
        ("Nonseasonal MA", regular[2]),
        ("Seasonal MA", seasonal[2]),
    ]
    .into_iter()
    .flat_map(|(label, count)| (1..=count).map(move |i| format!("{label}({i:02})")))
    .collect()
}

pub(crate) fn parse_output(
    directory: &Path,
    series: &Series,
    configuration: &OracleConfiguration,
) -> Result<Output> {
    let read = |extension: &str| {
        let bytes = fs::read(directory.join(format!("output.{extension}")))
            .with_context(|| format!("missing or unreadable output.{extension}"))?;
        if matches!(extension, "out" | "err" | "udg") {
            // Build 62's human-readable reports can contain non-UTF-8 bytes in
            // unrelated SEATS reporting, and on a constant series savtpk.f
            // writes uninitialized Tukey peak arrays into the UDG
            // (`peaks.tukey.*`). Compared labels and keys must still match
            // exactly; all saved numeric files remain strict UTF-8.
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        } else {
            String::from_utf8(bytes).with_context(|| format!("invalid UTF-8 in output.{extension}"))
        }
    };
    let component = |extension| -> Result<_> {
        let parsed = crate::saved_tables::parse_series_table(&read(extension)?, series.frequency)?;
        let expected = x13::Series::new(start_period(series)?, &series.values)
            .periods()?
            .map(|p| Period::new(p.year(), p.subperiod()))
            .collect::<Vec<_>>();
        let observed = parsed
            .into_iter()
            .filter(|point| expected.contains(&point.period))
            .collect::<Vec<_>>();
        ensure!(
            observed.iter().map(|p| p.period).eq(expected),
            "{extension}: missing, duplicate or unordered observed periods"
        );
        Ok(observed)
    };
    let seats = configuration.method() == crate::Method::Seats;
    let summary = crate::spec_file::uses_summary(configuration);
    // seatdg.f marks Havesa && !Havesf as nosf, and seatpr.f then saves no
    // S10. The seasonal component is read as absent, not as zeros or ones.
    let no_seasonal = seats
        && if summary {
            read("udg")?
                .lines()
                .any(|line| line.trim() == "seatsadj: nosf")
        } else {
            // seatpr.f saves requested S10 iff Havesf; no-summary calls have
            // no UDG nosf marker. Still require complete S11/S13 below, and
            // compare the absent seasonal component rather than invent values.
            !directory.join("output.s10").exists()
        };
    let components = Components {
        seasonal: if no_seasonal {
            ensure!(
                !directory.join("output.s10").exists(),
                "S10 present with seatsadj: nosf"
            );
            vec![]
        } else {
            component(if seats { "s10" } else { "d10" })?
        },
        seasonally_adjusted: component(if seats { "s11" } else { "d11" })?,
        trend_cycle: if seats && !directory.join("output.s12").try_exists()? {
            // ansub9 USRENTRY1310 marks an all-zero trend absent; seatpr
            // saves S12 only when Havetr. An absent trend must match an
            // empty Rust trend exactly.
            vec![]
        } else {
            component(if seats { "s12" } else { "d12" })?
        },
        irregular: component(if seats { "s13" } else { "d13" })?,
    };
    let transitory = if seats && directory.join("output.s14").exists() {
        Some(component("s14")?)
    } else {
        None
    };
    let seats_model = if seats {
        seats_model(&read("log")?)?
    } else {
        None
    };
    let uses_aic_test =
        matches!(configuration, OracleConfiguration::RegArima { request, .. } if request.aic_test);
    let equivalent = match *configuration {
        OracleConfiguration::RegArima { request, .. } => request.x11_equivalent()?,
        configuration => configuration,
    };
    let configuration = &equivalent;
    if matches!(configuration, OracleConfiguration::DirectX11 { .. }) {
        return Ok(Output {
            components,
            transitory,
            model: None,
            forecasts: vec![],
            backcasts: vec![],
            seats_model,
            adjustment_debug: String::new(),
        });
    }
    // The automatic path reports the constant-series bypass in the .out
    // report; the explicit x11ari.f path reports it only in the .err file.
    let constant_bypass = "The program will not estimate a regARIMA model for a constant series.";
    if !directory.join("output.est").exists()
        && (read("out")?.contains(constant_bypass) || read("err")?.contains(constant_bypass))
    {
        return Ok(Output {
            components,
            transitory,
            model: None,
            forecasts: vec![],
            backcasts: vec![],
            seats_model,
            adjustment_debug: String::new(),
        });
    }
    let (transform, orders) = if summary {
        model_choices(&read("udg")?)?
    } else {
        model_choices_without_summary(&read("mdl")?, &read("out")?)?
    };
    let (mut arma, regression) = estimate_coefficients(&read("est")?, orders)?;
    // The last row of the .itr file gives the ARMA values to 15 digits.
    // Automatic-model and outlier-search runs take their final coefficients
    // from the .est file, which prints 14 and has its own tolerance.
    let iteration_values = if !uses_aic_test
        && matches!(
            configuration,
            OracleConfiguration::ExplicitRegArimaX11 { .. }
                | OracleConfiguration::CalendarOutlierRegArimaX11 { outliers: None, .. }
        ) {
        let values = final_iteration_parameters(&read("itr")?)?;
        ensure!(
            values.len() == arma.len(),
            "itr: final ARMA count differs from est"
        );
        Some(values)
    } else {
        None
    };
    if let Some(values) = &iteration_values {
        for (parameter, &value) in arma.iter_mut().zip(values) {
            parameter.value = value;
        }
    }
    let outliers = regression
        .iter()
        .filter(|p| {
            ["AO", "LS", "TC"]
                .iter()
                .any(|prefix| p.name.starts_with(prefix))
        })
        .map(|p| outlier_label(&p.name, series.frequency))
        .collect::<Result<Vec<_>, _>>()?;
    let horizon = if let Some(reg) = configuration.regarima() {
        reg.forecast_horizon.resolve(series.frequency)?
    } else if let OracleConfiguration::AutomaticModelX11 {
        forecast_horizon, ..
    } = configuration
    {
        forecast_horizon.resolve(series.frequency)?
    } else {
        bail!("unsupported harness configuration")
    };
    let forecasts = if horizon > 0 {
        let mut table = point_table(&read("fct")?, "forecast", series.frequency)?;
        // SEATS internally extends to three years even when the public
        // request is shorter; only the original-fit requested prefix is public.
        if seats {
            ensure!(
                table.len() >= usize::from(horizon),
                "fct: missing requested forecast prefix"
            );
            table.truncate(usize::from(horizon));
        }
        ensure!(
            table.len() == usize::from(horizon),
            "fct: wrong forecast horizon"
        );
        table
    } else {
        vec![]
    };
    // editor.f zeroes Nbcst when the model span starts late, and then no
    // backcast table is saved; Rust likewise returns no backcasts.
    let backcasts =
        if requested_backcasts(configuration) > 0 && directory.join("output.bct").try_exists()? {
            point_table(&read("bct")?, "backcast", series.frequency)?
        } else {
            vec![]
        };
    Ok(Output {
        components,
        transitory,
        forecasts,
        backcasts,
        seats_model,
        adjustment_debug: String::new(),
        model: Some(Model {
            transform,
            orders,
            arma,
            regression,
            outliers,
            strict_arma_gate: iteration_values.is_some()
                || matches!(
                    configuration,
                    OracleConfiguration::ExplicitRegArimaX11 { .. }
                ),
        }),
    })
}

fn parameter_name(kind: ParameterKind, factor: Factor, period: u8, lag: u8) -> String {
    format!("{kind:?}/{factor:?}/{period}/{lag}")
}

pub(super) fn outlier_label(
    label: &str,
    frequency: Frequency,
) -> Result<OracleOutlierVariable, crate::OracleError> {
    // The .est file labels a requested outlier with its month number and an
    // automatically selected one with the month's abbreviation.
    if frequency == Frequency::Monthly
        && let Some((year, month)) = label.get(2..).and_then(|s| s.split_once('.'))
        && let Ok(month) = month.parse::<usize>()
    {
        let names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        let name = month
            .checked_sub(1)
            .and_then(|m| names.get(m))
            .ok_or_else(|| crate::OracleError::InvalidTable("invalid outlier month".into()))?;
        return crate::saved_tables::parse_outlier_label(
            &format!("{}{year}.{name}", &label[..2]),
            frequency,
        );
    }
    crate::saved_tables::parse_outlier_label(label, frequency)
}

// Reads only the compared fields of the .est file: a missing standard
// error, variance or likelihood does not fail the parse.
fn estimate_coefficients(
    table: &str,
    orders: [[u8; 3]; 2],
) -> Result<(Vec<Coefficient>, Vec<Coefficient>)> {
    let lines = table.lines().map(str::trim).collect::<Vec<_>>();
    let section = |marker: &str, columns: &[&str]| -> Result<Vec<Vec<&str>>> {
        let positions = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| **line == marker)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        ensure!(positions.len() <= 1, "est: duplicate {marker}");
        let Some(&start) = positions.first() else {
            return Ok(vec![]);
        };
        let header = lines
            .get(start + 1)
            .context("est: missing section header")?
            .split('\t')
            .map(str::trim)
            .collect::<Vec<_>>();
        ensure!(header.starts_with(columns), "est: invalid {marker} header");
        ensure!(
            lines
                .get(start + 2)
                .is_some_and(|line| line.starts_with('-')),
            "est: missing separator"
        );
        lines
            .iter()
            .skip(start + 3)
            .take_while(|line| !line.starts_with('$'))
            .filter(|line| !line.is_empty())
            .map(|line| {
                let fields = line.split('\t').map(str::trim).collect::<Vec<_>>();
                ensure!(
                    fields.len() >= columns.len(),
                    "est: missing compared column"
                );
                Ok(fields)
            })
            .collect()
    };
    let arma = section(
        "$arima$estimates:",
        &["operator", "factor", "period", "lag", "estimate"],
    )?
    .into_iter()
    .map(|fields| {
        let kind = match fields[0] {
            "AR" => ParameterKind::Ar,
            "MA" => ParameterKind::Ma,
            other => bail!("est: invalid operator {other}"),
        };
        let factor = match fields[1] {
            "Nonseasonal" => Factor::Nonseasonal,
            "Seasonal" => Factor::Seasonal,
            other => bail!("est: invalid factor {other}"),
        };
        Ok(Coefficient {
            name: parameter_name(kind, factor, fields[2].parse()?, fields[3].parse()?),
            value: finite(fields[4])?,
        })
    })
    .collect::<Result<Vec<_>>>()?;
    let count = orders
        .iter()
        .map(|o| usize::from(o[0]) + usize::from(o[2]))
        .sum::<usize>();
    ensure!(
        arma.len() == count,
        "est: ARMA coefficient count differs from selected model"
    );
    let regression = section("$regression$estimates:", &["group", "variable", "estimate"])?
        .into_iter()
        .map(|fields| {
            Ok(Coefficient {
                name: if fields[1] == "Constant" {
                    "const"
                } else {
                    fields[1]
                }
                .into(),
                value: finite(fields[2])?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((arma, regression))
}

fn final_iteration_parameters(table: &str) -> Result<Vec<f64>> {
    let mut lines = table.lines();
    let header = lines
        .next()
        .context("itr: missing header")?
        .split_whitespace()
        .collect::<Vec<_>>();
    let start = if header.starts_with(&["overall", "nonlinear", "loglikelihood"]) {
        3
    } else if header.starts_with(&["nonlinear", "loglikelihood"]) {
        2
    } else {
        bail!("itr: invalid header")
    };
    let count = header[start..]
        .iter()
        .take_while(|name| name.starts_with("arma"))
        .count();
    ensure!(
        header[start + count..]
            .iter()
            .all(|name| name.starts_with("reg")),
        "itr: invalid parameter columns"
    );
    ensure!(
        lines
            .next()
            .is_some_and(|line| line.trim().starts_with('-')),
        "itr: missing separator"
    );
    let fields = lines
        .rfind(|line| !line.trim().is_empty())
        .context("itr: missing final iteration")?
        .split_whitespace()
        .collect::<Vec<_>>();
    ensure!(
        fields.len() >= start + count,
        "itr: missing final ARMA parameters"
    );
    fields[start..start + count]
        .iter()
        .map(|value| finite(value))
        .collect()
}

fn finite(text: &str) -> Result<f64> {
    let value: f64 = text.parse()?;
    ensure!(value.is_finite(), "nonfinite compared value {text}");
    Ok(value)
}

/// A saved `date <point> lowerci upperci` table (`fct` or `bct`); only the
/// point column is compared.
fn point_table(text: &str, point: &str, frequency: Frequency) -> Result<Vec<SeriesPoint>> {
    let header = text
        .lines()
        .next()
        .with_context(|| format!("{point}s: missing header"))?
        .split_whitespace()
        .collect::<Vec<_>>();
    ensure!(
        header.starts_with(&["date", point]),
        "{point}s: missing point {point} column"
    );
    Ok(crate::saved_tables::parse_series_table(text, frequency)?)
}

/// Whether the Build 62 run saves a backcast table, which older cache entries
/// do not record.
pub(crate) fn requests_backcasts(configuration: &OracleConfiguration) -> Result<bool> {
    let equivalent = match *configuration {
        OracleConfiguration::RegArima { request, .. } => request.x11_equivalent()?,
        configuration => configuration,
    };
    Ok(requested_backcasts(&equivalent) > 0)
}

fn requested_backcasts(configuration: &OracleConfiguration) -> u16 {
    match configuration {
        OracleConfiguration::AutomaticModelX11 {
            backcast_horizon, ..
        } => *backcast_horizon,
        configuration => configuration
            .regarima()
            .map_or(0, |regarima| regarima.backcast_horizon),
    }
}

/// `seats { savelog = seatsmodel }` (LSLSMD): seatdg.f logs
/// `SEATS model: (p d q)(P D Q)`, optionally followed by "(SEATS routines
/// changed ARIMA model)", then one `<label>(nn) - value se t` line per
/// coefficient (FORMAT 1100, e21.14: 14 significant digits). Only the value is
/// read; the logged MA standard errors index Sebp, a seatdg.f quirk.
fn seats_model(log: &str) -> Result<Option<SeatsModel>> {
    let mut lines = log.lines().map(str::trim);
    let Some(model) = lines.find_map(|line| line.strip_prefix("SEATS model:")) else {
        return Ok(None);
    };
    let model = model.trim();
    let model = model
        .strip_suffix("(SEATS routines changed ARIMA model)")
        .unwrap_or(model);
    let model = crate::saved_tables::parse_selected_model(model.trim())?;
    let mut coefficients = vec![];
    for line in lines.by_ref() {
        let Some((name, values)) = line.split_once(" - ") else {
            break;
        };
        let value = values
            .split_whitespace()
            .next()
            .with_context(|| format!("log: SEATS coefficient {name} has no value"))?;
        coefficients.push(Coefficient {
            name: name.trim().into(),
            value: finite(value)?,
        });
    }
    ensure!(
        !lines.any(|line| line.starts_with("SEATS model:")),
        "log: duplicate SEATS model"
    );
    Ok(Some(SeatsModel {
        orders: [model.regular, model.seasonal],
        coefficients,
    }))
}

fn model_choices(udg: &str) -> Result<(String, [[u8; 3]; 2])> {
    let unique = |key: &str| -> Result<&str> {
        let values = udg
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter(|(k, _)| k.trim() == key)
            .map(|(_, v)| v.trim())
            .collect::<Vec<_>>();
        ensure!(values.len() == 1, "missing or duplicate {key} in UDG");
        Ok(values[0])
    };
    let mut transform = unique("transform")?;
    if transform == "Automatic selection" {
        transform = unique("aictrans")?;
    }
    let transform = match transform {
        "No transformation" => "none",
        "Log(y)" => "log",
        other => bail!("unsupported resolved transformation {other}"),
    };
    let model = crate::saved_tables::parse_selected_model(unique("arimamdl")?)?;
    Ok((transform.into(), [model.regular, model.seasonal]))
}

fn model_choices_without_summary(mdl: &str, report: &str) -> Result<(String, [[u8; 3]; 2])> {
    // savmdl.f writes the original model, not SEATS' possibly changed model.
    let mut models = mdl.split("arima{model=").skip(1);
    let orders = models
        .next()
        .and_then(|text| text.lines().find(|line| !line.trim().is_empty()))
        .context("mdl: missing original ARIMA model")?;
    ensure!(models.next().is_none(), "mdl: duplicate original model");
    let model = crate::saved_tables::parse_selected_model(orders.trim())?;
    // arima.f -> prtnfn.f (Pcode=0): estimate print=model reports the final
    // resolved transformation even when transform=function=auto was requested.
    let mut lines = report.lines().map(str::trim);
    let mut transforms = Vec::new();
    while let Some(line) = lines.next() {
        let value = if line == "Transformation" {
            lines.next()
        } else {
            line.strip_prefix("Transformation:").map(str::trim)
        };
        if let Some(value) = value {
            transforms.push(match value {
                "No transformation" => "none",
                "Log(y)" => "log",
                other => bail!("unsupported resolved transformation {other}"),
            });
        }
    }
    ensure!(
        transforms.len() == 1,
        "out: missing or duplicate resolved transformation"
    );
    Ok((transforms[0].into(), [model.regular, model.seasonal]))
}

pub(crate) fn start_period(series: &Series) -> Result<x13::Period> {
    period_for(series.frequency, series.start)
}

fn period_for(frequency: Frequency, period: Period) -> Result<x13::Period> {
    Ok(match frequency {
        Frequency::Monthly => x13::Period::monthly(period.year, period.subperiod)?,
        Frequency::Quarterly => x13::Period::quarterly(period.year, period.subperiod)?,
        Frequency::Semiannual => bail!("unsupported frequency"),
    })
}

fn points(start: x13::Period, values: &[f64]) -> Result<Vec<SeriesPoint>> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite Rust output"
    );
    Ok(x13::Series::new(start, values)
        .periods()?
        .zip(values)
        .map(|(period, &value)| SeriesPoint {
            period: Period::new(period.year(), period.subperiod()),
            value,
        })
        .collect())
}

pub(crate) fn bit_record(value: &impl Serialize) -> Result<Vec<u8>> {
    fn replace(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Number(n) if n.is_f64() => {
                *value = serde_json::Value::String(format!(
                    "f64:{:016x}",
                    n.as_f64().unwrap().to_bits()
                ));
            }
            serde_json::Value::Array(values) => values.iter_mut().for_each(replace),
            serde_json::Value::Object(values) => values.values_mut().for_each(replace),
            _ => (),
        }
    }
    let mut json = serde_json::to_value(value)?;
    replace(&mut json);
    Ok(serde_json::to_vec(&json)?)
}

pub(crate) fn decode_bit_record<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    fn restore(value: &mut serde_json::Value) -> Result<()> {
        match value {
            serde_json::Value::String(s) if s.starts_with("f64:") => {
                let hex = &s[4..];
                anyhow::ensure!(hex.len() == 16, "invalid f64 bit record");
                let bits = u64::from_str_radix(hex, 16)?;
                let number = serde_json::Number::from_f64(f64::from_bits(bits))
                    .ok_or_else(|| anyhow::anyhow!("nonfinite f64 bit record"))?;
                *value = serde_json::Value::Number(number);
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    restore(item)?;
                }
            }
            serde_json::Value::Object(items) => {
                for item in items.values_mut() {
                    restore(item)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    let mut value = serde_json::from_slice(bytes)?;
    restore(&mut value)?;
    Ok(serde_json::from_value(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlier_labels_take_a_month_number_or_a_quarter() {
        let outlier = outlier_label("AO2017.10", Frequency::Monthly).unwrap();
        assert_eq!(outlier.outlier_type, OracleOutlierType::Additive);
        assert_eq!(outlier.period, Period::new(2017, 10));
        assert!(outlier_label("TC2017.13", Frequency::Monthly).is_err());
        assert_eq!(
            outlier_label("LS2017.4", Frequency::Quarterly)
                .unwrap()
                .period,
            Period::new(2017, 4)
        );
    }

    #[test]
    fn coefficients_ignore_standard_errors_but_need_every_estimate() {
        let table = "$regression$estimates:\ngroup\tvariable\testimate\tstandard error\tfixed\n---\t---\t---\t---\t---\nConstant\tConstant\t1.25\tNaN\t\n$arima:\n$arima$estimates:\noperator\tfactor\tperiod\tlag\testimate\tstandard error\tfixed\n---\t---\t---\t---\t---\t---\t---\nMA\tSeasonal\t12\t1\t0.4\tNaN\t\n$variance:\nmle\tNaN\nse\tNaN\n";
        let (arma, regression) = estimate_coefficients(table, [[0, 1, 0], [0, 1, 1]]).unwrap();
        assert_eq!(
            arma,
            vec![Coefficient {
                name: "Ma/Seasonal/12/1".into(),
                value: 0.4
            }]
        );
        assert_eq!(
            regression,
            vec![Coefficient {
                name: "const".into(),
                value: 1.25
            }]
        );
        assert!(
            estimate_coefficients(&table.replace("0.4", "NaN"), [[0, 1, 0], [0, 1, 1]]).is_err()
        );
        assert!(estimate_coefficients(table, [[0, 1, 1], [0, 1, 1]]).is_err());
        assert!(
            estimate_coefficients(&table.replace("1.25", "NaN"), [[0, 1, 0], [0, 1, 1]]).is_err()
        );
        assert!(
            estimate_coefficients(
                &table.replace("estimate\tstandard", "wrong\tstandard"),
                [[0, 1, 0], [0, 1, 1]]
            )
            .is_err()
        );
    }

    #[test]
    fn the_iteration_table_gives_the_final_arma_values() {
        let table = "overall\tnonlinear\tloglikelihood\tarma1\tarma2\treg1\n---\t---\t---\t---\t---\t---\n0\t1\tNaN\tNaN\tNaN\tNaN\n1\t2\tNaN\t0.25\t0.5\tNaN\n";
        assert_eq!(final_iteration_parameters(table).unwrap(), vec![0.25, 0.5]);
        assert!(final_iteration_parameters(&table.replace("0.25", "NaN")).is_err());
        assert!(
            final_iteration_parameters("nonlinear\twrong\tarma1\n---\t---\t---\n1\t0\t1").is_err()
        );
        assert!(
            final_iteration_parameters("nonlinear\tloglikelihood\tarma1\n---\t---\t---\n").is_err()
        );
    }

    #[test]
    fn forecasts_ignore_missing_intervals_but_need_point_values() {
        let table =
            "date\tforecast\tlowerci\tupperci\n----\t----\t----\t----\n200101\t12.5\tNaN\tNaN\n";
        let points = point_table(table, "forecast", Frequency::Monthly).unwrap();
        assert_eq!(
            points,
            vec![SeriesPoint {
                period: Period::new(2001, 1),
                value: 12.5
            }]
        );
        assert!(
            point_table(
                &table.replace("12.5", "NaN"),
                "forecast",
                Frequency::Monthly
            )
            .is_err()
        );
        assert!(
            point_table(
                &table.replace("forecast", "lowerci"),
                "forecast",
                Frequency::Monthly
            )
            .is_err()
        );
        let backcasts = table.replace("forecast", "backcast");
        assert_eq!(
            point_table(&backcasts, "backcast", Frequency::Monthly).unwrap(),
            points
        );
        assert!(point_table(table, "backcast", Frequency::Monthly).is_err());
    }

    #[test]
    fn unrecorded_columns_never_enter_rust_bit_records() {
        let output = Output {
            components: Components {
                seasonal: vec![],
                seasonally_adjusted: vec![],
                trend_cycle: vec![],
                irregular: vec![],
            },
            transitory: None,
            model: None,
            forecasts: vec![],
            backcasts: vec![],
            seats_model: None,
            adjustment_debug: String::new(),
        };
        let record = String::from_utf8(bit_record(&output).unwrap()).unwrap();
        assert!(!record.contains("backcasts") && !record.contains("seats_model"));
        assert_eq!(
            decode_bit_record::<Output>(record.as_bytes()).unwrap(),
            output
        );
    }

    #[test]
    fn seats_model_log_reads_final_orders_and_values_only() {
        let log = "  M-NONE   output -------- AMTMNO\n\
            \x20   SEATS model: (1 1 0)(0 1 1) (SEATS routines changed ARIMA model)\n\
            \x20   Nonseasonal AR(01) - -0.25000000000000E+00  0.00000000000000E+00  0.0E+00\n\
            \x20   Seasonal MA(01) -     0.99000000000000E+00  0.10000000000000E+00  9.9E+00\n\
            \x20\n\
            \x20    Error messages generated from processing the spec file\n";
        let model = seats_model(log).unwrap().unwrap();
        assert_eq!(model.orders, [[1, 1, 0], [0, 1, 1]]);
        let names = model.coefficients.iter().map(|c| c.name.as_str());
        assert!(names.eq(["Nonseasonal AR(01)", "Seasonal MA(01)"]));
        assert_eq!(model.coefficients[0].value, -0.25);
        assert_eq!(model.coefficients[1].value, 0.99);
        assert_eq!(
            seats_coefficient_names(model.orders),
            ["Nonseasonal AR(01)", "Seasonal MA(01)"]
        );
        let nonseasonal = seats_model("SEATS model: (0 1 1)\nNonseasonal MA(01) - 0.5 0 0\n")
            .unwrap()
            .unwrap();
        assert_eq!(nonseasonal.orders, [[0, 1, 1], [0; 3]]);
        assert_eq!(seats_model("no seats section\n").unwrap(), None);
        assert!(seats_model("SEATS model: (0 1 1)\nNonseasonal MA(01) - NaN 0 0\n").is_err());
        assert!(seats_model("SEATS model: (0 1 1)\n\nSEATS model: (0 1 1)\n").is_err());
    }

    #[test]
    fn transform_and_model_are_read_from_the_summary() {
        let (transform, orders) = model_choices(
            "transform: Automatic selection\naictrans: Log(y)\narimamdl: (1 1 0)(0 1 1)\n",
        )
        .unwrap();
        assert_eq!(transform, "log");
        assert_eq!(orders, [[1, 1, 0], [0, 1, 1]]);
        assert!(
            model_choices(
                "transform: Log(y)\ntransform: No transformation\narimamdl: (0 1 1)(0 1 1)\n"
            )
            .is_err()
        );
        assert!(
            model_choices("transform: Automatic selection\narimamdl: (0 1 1)(0 1 1)\n").is_err()
        );
    }

    #[test]
    fn bit_records_decode_without_rounding_and_reject_corruption() {
        let values = [
            0.0,
            -0.0,
            f64::from_bits(1_f64.to_bits() + 1),
            f64::MIN_POSITIVE,
        ];
        let decoded: Vec<f64> = decode_bit_record(&bit_record(&values).unwrap()).unwrap();
        assert_eq!(
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.map(f64::to_bits)
        );
        for bad in [
            br#"["f64:0"]"#.as_slice(),
            br#"["f64:zzzzzzzzzzzzzzzz"]"#,
            br#"["f64:7ff0000000000000"]"#,
            b"{",
        ] {
            assert!(decode_bit_record::<Vec<f64>>(bad).is_err());
        }
    }
}
