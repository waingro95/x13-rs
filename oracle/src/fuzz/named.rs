//! The named-spec grammar for hand-built repros (`SpecSource::Named`).
//!
//! `kind:model:transform[:option...]`, for example
//! `x11reg:0,1,1,0,1,1:log:span=60-79` or `direct:pseudo:sig=0.05,0.05`.
//! One parser feeds both sides of a replay: [`Named::to_spec`] builds the
//! Rust `Spec`, and `build62::render_named` writes the equivalent Build 62
//! spec, so the two runs cannot drift apart.
//!
//! | kind | meaning |
//! |---|---|
//! | `x11reg` / `x11reg_easter` | regARIMA-X-11, mode from the transform (`_easter` adds the legacy X-11 Easter estimate) |
//! | `seatsreg` | regARIMA-SEATS |
//! | `direct:<add\|mult\|pseudo\|logadd>` | direct X-11 (`direct_add`, `direct_mult` are shorthand) |
//!
//! The model field is `auto` or `p,d,q,P,D,Q`; the transform field is `log` or
//! anything else for none. Options (regARIMA kinds):
//!
//! | option | meaning |
//! |---|---|
//! | `span=a-b` | model span, zero-based observation offsets |
//! | `mspan=Y.P-Y.P` | model span, literal periods |
//! | `fcst=h` | forecast horizon in periods |
//! | `mao=Y.P,...` | manual additive outliers at literal periods |
//! | `out=auto`, `out=cv`, `out=a,l,t` | outlier search: AO+LS+TC at the automatic or shared critical value, or per type (`t` = `-` drops TC) |
//! | `ospan=Y.P-Y.P` | outlier search span (needs `out=`) |
//! | `ureg=k` | k Gaussian user regressors covering the series plus ten years |
//! | `udup=seed,eps` | two user regressors; the second is the first plus `eps` noise |
//!
//! Direct X-11 options: `sig=a,b` (sigma limits), `sd` (standard-deviation
//! estimator, the default), `h=L,r` (Henderson length and I/C ratio), `h7`
//! (seven-term endpoints).
//!
//! A literal `Y.P` period is built in the series' frequency, as Build 62 reads
//! it.
//!
//! Direct X-11 leaves `seasonalma` unset on both sides, so both select by the
//! moving-seasonality ratio; regARIMA X-11 states `s3x5` on both.

use super::generate::Rng;
use std::str::FromStr;

/// Extra years of regressor rows beyond the series, so forecasts are covered.
const REGRESSOR_MARGIN: usize = 120;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Date {
    pub(crate) year: i32,
    pub(crate) period: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Add,
    Mult,
    Pseudo,
    LogAdd,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind {
    X11Reg { easter: bool },
    SeatsReg,
    Direct(Mode),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Model {
    Auto,
    Explicit([u8; 6]),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Outliers {
    Auto,
    Shared(f64),
    PerType { ao: f64, ls: f64, tc: Option<f64> },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Named {
    pub(crate) kind: Kind,
    pub(crate) model: Model,
    pub(crate) log: bool,
    pub(crate) span_offsets: Option<(usize, usize)>,
    pub(crate) model_span: Option<(Date, Date)>,
    pub(crate) forecast: Option<u16>,
    pub(crate) manual_ao: Vec<Date>,
    pub(crate) outliers: Option<Outliers>,
    pub(crate) outlier_span: Option<(Date, Date)>,
    pub(crate) user_regressors: Option<usize>,
    pub(crate) duplicate: Option<(u64, f64)>,
    pub(crate) sigma: Option<(f64, f64)>,
    pub(crate) standard_deviation: bool,
    pub(crate) henderson: Option<(u8, f64)>,
    pub(crate) seven_term: bool,
}

fn debug(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

fn num<T: FromStr>(text: &str, what: &str) -> Result<T, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("bad {what} {text:?}"))
}

fn pair<'a>(text: &'a str, separator: char, what: &str) -> Result<(&'a str, &'a str), String> {
    text.split_once(separator)
        .ok_or_else(|| format!("bad {what} {text:?}"))
}

fn date(text: &str) -> Result<Date, String> {
    let (year, period) = pair(text, '.', "period")?;
    Ok(Date {
        year: num(year, "year")?,
        period: num(period, "period")?,
    })
}

fn date_range(text: &str) -> Result<(Date, Date), String> {
    let (a, b) = pair(text, '-', "span")?;
    Ok((date(a)?, date(b)?))
}

pub(crate) fn parse(name: &str) -> Result<Named, String> {
    let parts: Vec<&str> = name.split(':').collect();
    let mut named = Named {
        kind: Kind::SeatsReg,
        model: Model::Auto,
        log: false,
        span_offsets: None,
        model_span: None,
        forecast: None,
        manual_ao: vec![],
        outliers: None,
        outlier_span: None,
        user_regressors: None,
        duplicate: None,
        sigma: None,
        standard_deviation: false,
        henderson: None,
        seven_term: false,
    };
    let options = match parts[0] {
        "direct_add" => {
            named.kind = Kind::Direct(Mode::Add);
            &parts[1..]
        }
        "direct_mult" => {
            named.kind = Kind::Direct(Mode::Mult);
            &parts[1..]
        }
        "direct" => {
            named.kind = Kind::Direct(match parts.get(1).copied() {
                Some("add") => Mode::Add,
                Some("mult") => Mode::Mult,
                Some("pseudo") => Mode::Pseudo,
                _ => Mode::LogAdd,
            });
            parts.get(2..).unwrap_or(&[])
        }
        kind @ ("x11reg" | "x11reg_easter" | "seatsreg") => {
            named.kind = match kind {
                "seatsreg" => Kind::SeatsReg,
                _ => Kind::X11Reg {
                    easter: kind == "x11reg_easter",
                },
            };
            let model = parts.get(1).ok_or("missing model field")?;
            named.model = if *model == "auto" {
                Model::Auto
            } else {
                let orders: Vec<u8> = model
                    .split(',')
                    .map(|order| num(order, "order"))
                    .collect::<Result<_, _>>()?;
                Model::Explicit(
                    orders
                        .try_into()
                        .map_err(|_| format!("model {model:?} needs six orders"))?,
                )
            };
            named.log = parts.get(2) == Some(&"log");
            parts.get(3..).unwrap_or(&[])
        }
        other => return Err(format!("unknown named spec {other}")),
    };
    for option in options {
        if let Some(span) = option.strip_prefix("span=") {
            let (a, b) = pair(span, '-', "span")?;
            named.span_offsets = Some((num(a, "offset")?, num(b, "offset")?));
        } else if let Some(span) = option.strip_prefix("mspan=") {
            named.model_span = Some(date_range(span)?);
        } else if let Some(horizon) = option.strip_prefix("fcst=") {
            named.forecast = Some(num(horizon, "horizon")?);
        } else if let Some(list) = option.strip_prefix("mao=") {
            named.manual_ao = list.split(',').map(date).collect::<Result<_, _>>()?;
        } else if let Some(value) = option.strip_prefix("out=") {
            named.outliers = Some(parse_outliers(value)?);
        } else if let Some(span) = option.strip_prefix("ospan=") {
            named.outlier_span = Some(date_range(span)?);
        } else if let Some(count) = option.strip_prefix("ureg=") {
            named.user_regressors = Some(num(count, "count")?);
        } else if let Some(spec) = option.strip_prefix("udup=") {
            let (seed, eps) = spec.split_once(',').unwrap_or((spec, "0"));
            named.duplicate = Some((num(seed, "seed")?, num(eps, "epsilon")?));
        } else if let Some(limits) = option.strip_prefix("sig=") {
            let (lower, upper) = pair(limits, ',', "sigma limits")?;
            named.sigma = Some((num(lower, "limit")?, num(upper, "limit")?));
        } else if let Some(filter) = option.strip_prefix("h=") {
            let (length, ratio) = pair(filter, ',', "Henderson filter")?;
            named.henderson = Some((num(length, "length")?, num(ratio, "ratio")?));
        } else if *option == "h7" {
            named.seven_term = true;
        } else if *option == "sd" {
            named.standard_deviation = true;
        } else {
            return Err(format!("unknown option {option:?}"));
        }
    }
    Ok(named)
}

fn parse_outliers(value: &str) -> Result<Outliers, String> {
    if value == "auto" {
        return Ok(Outliers::Auto);
    }
    let Some((ao, rest)) = value.split_once(',') else {
        return Ok(Outliers::Shared(num(value, "critical value")?));
    };
    let (ls, tc) = pair(rest, ',', "critical values")?;
    Ok(Outliers::PerType {
        ao: num(ao, "critical value")?,
        ls: num(ls, "critical value")?,
        tc: if tc == "-" {
            None
        } else {
            Some(num(tc, "critical value")?)
        },
    })
}

/// The `ureg=k` regressor columns: `rows` standard-normal draws per column.
pub(crate) fn ureg_columns(k: usize, rows: usize) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(0x5EED ^ k as u64);
    (0..k)
        .map(|_| (0..rows).map(|_| rng.normal()).collect())
        .collect()
}

/// The `udup=seed,eps` columns: a standard-normal column and a noisy copy.
pub(crate) fn udup_columns(seed: u64, eps: f64, rows: usize) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(0xD0B1E ^ seed);
    let first: Vec<f64> = (0..rows).map(|_| rng.normal()).collect();
    let second = first.iter().map(|v| v + eps * rng.normal()).collect();
    vec![first, second]
}

impl Named {
    /// The user regressor columns this spec asks for (empty if none).
    pub(crate) fn regressor_columns(&self, n: usize) -> Vec<Vec<f64>> {
        let rows = n + REGRESSOR_MARGIN;
        let mut columns = vec![];
        if let Some(k) = self.user_regressors {
            columns.extend(ureg_columns(k, rows));
        }
        if let Some((seed, eps)) = self.duplicate {
            columns.extend(udup_columns(seed, eps, rows));
        }
        columns
    }

    pub(crate) fn to_spec(&self, start: x13::Period, n: usize) -> Result<x13::Spec, String> {
        if let Kind::Direct(mode) = self.kind {
            return self.direct_spec(mode);
        }
        let frequency = start.frequency();
        let transform = if self.log {
            x13::Transform::Log
        } else {
            x13::Transform::None
        };
        let mut r = match self.model {
            Model::Auto => x13::RegArimaSpec::automatic_model().with_transform(transform),
            Model::Explicit([p, d, q, sp, sd, sq]) => {
                let model = x13::ArimaModel::new(
                    x13::ArimaOrder::new(p, d, q).map_err(debug)?,
                    x13::ArimaOrder::new(sp, sd, sq).map_err(debug)?,
                )
                .map_err(debug)?;
                x13::RegArimaSpec::new(model).with_transform(transform)
            }
        };
        if let Some((a, b)) = self.span_offsets {
            let a = crate::rust_spec::period_at(start, a).map_err(debug)?;
            let b = crate::rust_spec::period_at(start, b).map_err(debug)?;
            r = r.with_model_span(x13::PeriodSpan::new(a, b));
        }
        if let Some(horizon) = self.forecast {
            r = r.with_forecast(
                x13::regarima::ForecastSpec::new(
                    x13::regarima::ForecastHorizon::Periods(horizon),
                    0,
                )
                .map_err(debug)?,
            );
        }
        if let Some((a, b)) = self.model_span {
            r = r.with_model_span(x13::PeriodSpan::new(
                literal(a, frequency)?,
                literal(b, frequency)?,
            ));
        }
        if !self.manual_ao.is_empty() {
            let outliers: Vec<x13::regarima::ManualOutlier> = self
                .manual_ao
                .iter()
                .map(|d| {
                    literal(*d, frequency).map(|p| {
                        x13::regarima::ManualOutlier::new(x13::regarima::OutlierType::Additive, p)
                    })
                })
                .collect::<Result<_, _>>()?;
            r = r.with_manual_outliers(&outliers);
        }
        if let Some(search) = self.outliers {
            let mut o =
                x13::regarima::OutlierSpec::default().with_types(x13::regarima::OutlierTypes::ALL);
            match search {
                Outliers::Auto => {}
                Outliers::Shared(value) => {
                    let value = x13::regarima::OutlierCriticalValue::new(value).map_err(debug)?;
                    o = o.with_critical_values(x13::regarima::OutlierCriticalValues::shared(value));
                }
                Outliers::PerType { ao, ls, tc } => {
                    let value = |v: f64| x13::regarima::OutlierCriticalValue::new(v).ok();
                    if tc.is_none() {
                        o = o.with_types(x13::regarima::OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT);
                    }
                    o = o.with_critical_values(x13::regarima::OutlierCriticalValues::per_type(
                        value(ao),
                        value(ls),
                        tc.and_then(value),
                    ));
                }
            }
            if let Some((a, b)) = self.outlier_span {
                o = o.with_span(x13::PeriodSpan::new(
                    literal(a, frequency)?,
                    literal(b, frequency)?,
                ));
            }
            r = r.with_outliers(o);
        }
        let columns = self.regressor_columns(n);
        if !columns.is_empty() {
            let regressors = columns
                .into_iter()
                .map(|values| {
                    let series = x13::Series::new(start, &values);
                    x13::regarima::UserRegressor::new(
                        x13::regarima::UserRegressorType::User,
                        series,
                    )
                    .map_err(debug)
                })
                .collect::<Result<Vec<_>, _>>()?;
            r = r.with_user_regressors(&regressors);
        }
        // The Build 62 spec asks for `seasonalma = s3x5`.
        let s3x5 = x13::x11::Options::new().with_seasonal_filter(x13::x11::SeasonalFilter::S3x5);
        Ok(match self.kind {
            Kind::X11Reg { easter: false } => x13::Spec::x11_regarima(r, s3x5),
            Kind::X11Reg { easter: true } => {
                x13::Spec::x11_regarima(r, s3x5.with_easter_estimation(true))
            }
            Kind::SeatsReg => x13::Spec::seats(r),
            Kind::Direct(_) => unreachable!("direct specs return early"),
        })
    }

    fn direct_spec(&self, mode: Mode) -> Result<x13::Spec, String> {
        let mut o = x13::x11::Options::new();
        if let Some((lower, upper)) = self.sigma {
            o = o.with_sigma_limits(x13::x11::SigmaLimits::new(lower, upper).map_err(debug)?);
        }
        if let Some((length, ratio)) = self.henderson {
            o = o.with_trend_filter(x13::x11::TrendFilter::HendersonWithIcRatio {
                length: x13::x11::HendersonLength::new(length).map_err(debug)?,
                ratio: x13::x11::HendersonIcRatio::new(ratio).map_err(debug)?,
            });
        }
        if self.standard_deviation {
            o = o.with_extreme_estimator(x13::x11::ExtremeEstimator::StandardDeviation);
        }
        if self.seven_term {
            o = o.with_seven_term_endpoints(x13::x11::SevenTermEndpoints::HendersonSevenTerm);
        }
        let mode = match mode {
            Mode::Add => x13::x11::Mode::Additive,
            Mode::Mult => x13::x11::Mode::Multiplicative,
            Mode::Pseudo => x13::x11::Mode::PseudoAdditive,
            Mode::LogAdd => x13::x11::Mode::LogAdditive,
        };
        Ok(x13::Spec::x11(o.with_mode(mode)))
    }
}

/// Builds `Y.P` with the validating constructors, so a subperiod outside the
/// year (`2000.13`) is an error, as Build 62 reports an improper date, rather
/// than rolling into the next year.
fn literal(date: Date, frequency: x13::Frequency) -> Result<x13::Period, String> {
    let subperiod = u8::try_from(date.period)
        .map_err(|_| format!("period {} is outside the year", date.period))?;
    match frequency {
        x13::Frequency::Monthly => x13::Period::monthly(date.year, subperiod),
        x13::Frequency::Quarterly => x13::Period::quarterly(date.year, subperiod),
        other => return Err(format!("no period literal for {other:?} series")),
    }
    .map_err(debug)
}
