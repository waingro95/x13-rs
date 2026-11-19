//! Spec construction for a [`Case`]: random public-builder specs, hand-built
//! named specs (grammar in `named.rs`) and call-table rows.

use super::generate::{Case, Rng, SpecSource, context};
use super::named;
use crate::{Frequency, Period, Series};

// ---------------------------------------------------------------- random specs
fn opt<T>(r: Result<T, x13::Error>) -> Option<T> {
    r.ok()
}

fn random_period(rng: &mut Rng, start: x13::Period, n: usize) -> x13::Period {
    let s = i64::from(start.frequency().periods_per_year());
    let offset: i64 = match rng.below(6) {
        0 => -rng.range(1, 3 * s),
        1 => n as i64 + rng.range(0, 3 * s),
        2 => n as i64 - 1,
        3 => 0,
        _ => rng.range(0, (n as i64 - 1).max(0)),
    };
    // Occasionally a period from the other frequency.
    let other = if rng.chance(0.07) {
        match start.frequency() {
            x13::Frequency::Monthly => x13::Period::quarterly(start.year(), 1).ok(),
            x13::Frequency::Quarterly => x13::Period::monthly(start.year(), 12).ok(),
            other => panic!("no other-frequency start for {other:?} series"),
        }
    } else {
        None
    };
    other.unwrap_or(start).shift(offset).unwrap_or(start)
}

fn random_filter(rng: &mut Rng) -> x13::x11::SeasonalFilter {
    use x13::x11::SeasonalFilter::*;
    rng.pick(&[S3x1, S3x3, S3x5, S3x9, S3x15, Stable, X11Default, Msr])
}

fn random_x11_options(rng: &mut Rng, monthly: bool) -> x13::x11::Options {
    let mut o = x13::x11::Options::new();
    if rng.chance(0.5) {
        o = o.with_seasonal_filter(random_filter(rng));
    }
    if rng.chance(0.15) {
        let m = if rng.chance(0.9) { monthly } else { !monthly };
        o = o.with_seasonal_filters(if m {
            x13::x11::SeasonalFilters::Monthly(std::array::from_fn(|_| random_filter(rng)))
        } else {
            x13::x11::SeasonalFilters::Quarterly(std::array::from_fn(|_| random_filter(rng)))
        });
    }
    if rng.chance(0.1) {
        o = o.with_decomposition(x13::x11::Decomposition::TrendOnly);
    }
    if rng.chance(0.15) {
        o = o.with_seasonal_shrinkage(rng.pick(&[
            x13::x11::SeasonalShrinkage::None,
            x13::x11::SeasonalShrinkage::Global,
            x13::x11::SeasonalShrinkage::Local,
        ]));
    }
    if rng.chance(0.2) {
        o = o.with_short_series_filters(x13::x11::ShortSeriesFilters::UseRequested);
    }
    if rng.chance(0.3) {
        let terms = if rng.chance(0.8) {
            rng.pick(&[3u8, 5, 7, 9, 13, 23, 33, 51, 79, 99, 101])
        } else {
            rng.range(0, 255) as u8
        };
        if let Some(length) = opt(x13::x11::HendersonLength::new(terms)) {
            o = o.with_trend_filter(if rng.chance(0.4) {
                match opt(x13::x11::HendersonIcRatio::new(rng.pick(&[
                    1e-300,
                    3.833e-20,
                    3.834e-20,
                    0.001,
                    1.0,
                    3.5,
                    1e6,
                    1e300,
                    f64::MAX,
                ]))) {
                    Some(ratio) => x13::x11::TrendFilter::HendersonWithIcRatio { length, ratio },
                    None => x13::x11::TrendFilter::Henderson(length),
                }
            } else {
                x13::x11::TrendFilter::Henderson(length)
            });
        }
    }
    if rng.chance(0.2) {
        o = o.with_seven_term_endpoints(x13::x11::SevenTermEndpoints::HendersonSevenTerm);
    }
    if rng.chance(0.3) {
        let lower = rng.pick(&[1e-300, 0.01, 0.5, 1.0, 1.5, 2.5, 3.0, 10.0, 1e300]);
        let upper = if rng.chance(0.2) {
            lower
        } else if rng.chance(0.1) {
            lower * 0.5
        } else {
            lower * rng.pick(&[1.0000001, 1.5, 2.0, 1e10])
        };
        if let Some(l) = opt(x13::x11::SigmaLimits::new(lower, upper)) {
            o = o.with_sigma_limits(l);
        }
    }
    if rng.chance(0.2) {
        o = o.with_extreme_estimation_span(x13::x11::ExtremeEstimationSpan::ObservedOnly);
    }
    if rng.chance(0.3) {
        let scale = match rng.below(4) {
            0 => x13::x11::ExtremeScale::RollingFiveYear,
            1 => x13::x11::ExtremeScale::Automatic,
            2 => x13::x11::ExtremeScale::ByCalendarPeriod,
            _ => {
                let g = |rng: &mut Rng| {
                    if rng.chance(0.5) {
                        x13::x11::SigmaGroup::Selected
                    } else {
                        x13::x11::SigmaGroup::Complement
                    }
                };
                let m = if rng.chance(0.9) { monthly } else { !monthly };
                x13::x11::ExtremeScale::Grouped(if m {
                    x13::x11::SigmaGroups::Monthly(std::array::from_fn(|_| g(rng)))
                } else {
                    x13::x11::SigmaGroups::Quarterly(std::array::from_fn(|_| g(rng)))
                })
            }
        };
        o = o.with_extreme_scale(scale);
    }
    if rng.chance(0.3) {
        use x13::x11::ExtremeEstimator::*;
        o = o.with_extreme_estimator(rng.pick(&[
            StandardDeviation,
            MedianAbsoluteDeviation,
            TauAdjustedMedianAbsoluteDeviation,
            LogMedianAbsoluteDeviation,
            TauAdjustedLogMedianAbsoluteDeviation,
        ]));
    }
    if rng.chance(0.12) {
        o = o.with_easter_estimation(true);
    }
    o
}

fn random_types(rng: &mut Rng) -> x13::regarima::OutlierTypes {
    use x13::regarima::OutlierTypes as T;
    rng.pick(&[
        T::ADDITIVE,
        T::LEVEL_SHIFT,
        T::TEMPORARY_CHANGE,
        T::ADDITIVE_AND_LEVEL_SHIFT,
        T::ADDITIVE_AND_TEMPORARY_CHANGE,
        T::LEVEL_SHIFT_AND_TEMPORARY_CHANGE,
        T::ALL,
    ])
}

fn random_cv(rng: &mut Rng) -> Option<x13::regarima::OutlierCriticalValue> {
    opt(x13::regarima::OutlierCriticalValue::new(rng.pick(&[
        1e-300, 0.01, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 6.0, 100.0, 1e300,
    ])))
}

fn random_regarima(rng: &mut Rng, start: x13::Period, n: usize) -> x13::RegArimaSpec {
    let monthly = start.frequency() == x13::Frequency::Monthly;
    let mut r = if rng.chance(0.35) {
        x13::RegArimaSpec::automatic_model()
    } else {
        let model = (|| {
            let reg = x13::ArimaOrder::new(
                rng.range(0, 3) as u8,
                rng.range(0, 2) as u8,
                rng.range(0, 3) as u8,
            )
            .ok()?;
            let sea = x13::ArimaOrder::new(
                rng.range(0, 2) as u8,
                rng.range(0, 1) as u8,
                rng.range(0, 2) as u8,
            )
            .ok()?;
            x13::ArimaModel::new(reg, sea).ok()
        })();
        let airline = x13::ArimaModel::new(
            x13::ArimaOrder::new(0, 1, 1).unwrap(),
            x13::ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap();
        x13::RegArimaSpec::new(if rng.chance(0.5) {
            airline
        } else {
            model.unwrap_or(airline)
        })
    };
    r = match rng.below(4) {
        0 => r.with_transform(x13::Transform::None),
        1 => r.with_transform(x13::Transform::Log),
        2 => {
            let opts = x13::regarima::AutoTransformOptions::new();
            let opts = if rng.chance(0.3) {
                opts.with_aic_difference(rng.pick(&[-1e300, -2.0, 0.0, 1e-300, 5.0, 1e300]))
                    .unwrap_or(opts)
            } else {
                opts
            };
            r.with_transform(opts)
        }
        _ => r,
    };
    if rng.chance(0.3) {
        r = r.with_constant(rng.chance(0.5));
    }
    if rng.chance(0.4) {
        let horizon = if rng.chance(0.3) {
            x13::regarima::ForecastHorizon::OneYear
        } else {
            x13::regarima::ForecastHorizon::Periods(rng.pick(&[0u16, 1, 3, 12, 36, 60, 120, 121]))
        };
        let back = rng.pick(&[0u16, 0, 1, 12, 60, 120, 121]);
        if let Some(e) = opt(x13::regarima::ForecastSpec::new(horizon, back)) {
            r = r.with_forecast(e);
        }
    }
    if rng.chance(0.4) {
        let mut c = x13::regarima::CalendarSpec::new();
        match rng.below(5) {
            0 => {
                let contrast = if rng.chance(0.5) {
                    x13::regarima::TradingDayContrast::SixCoefficient
                } else {
                    x13::regarima::TradingDayContrast::WeekdayWeekend
                };
                let lt = rng.pick(&[
                    x13::regarima::FlowLengthTreatment::LeapYear,
                    x13::regarima::FlowLengthTreatment::NoLengthEffect,
                    x13::regarima::FlowLengthTreatment::LengthOfPeriod,
                ]);
                c = c.with_basis(x13::regarima::CalendarBasis::FlowTradingDay(
                    x13::regarima::FlowTradingDay::new(contrast).with_length_treatment(lt),
                ));
            }
            1 => {
                let contrast = if rng.chance(0.5) {
                    x13::regarima::TradingDayContrast::SixCoefficient
                } else {
                    x13::regarima::TradingDayContrast::WeekdayWeekend
                };
                if let Some(st) = opt(x13::regarima::StockTradingDay::new(
                    contrast,
                    rng.range(0, 32) as u8,
                )) {
                    c = c.with_basis(x13::regarima::CalendarBasis::StockTradingDay(st));
                }
            }
            2 => {
                c = c.with_basis(x13::regarima::CalendarBasis::Length(if rng.chance(0.5) {
                    x13::regarima::LengthEffect::LeapYear
                } else {
                    x13::regarima::LengthEffect::LengthOfPeriod
                }));
            }
            _ => {}
        }
        if rng.chance(0.5)
            && let Some(e) = opt(x13::regarima::Easter::new(rng.range(0, 26) as u8))
        {
            c = c.with_easter(e);
        }
        if rng.chance(0.2)
            && let Some(e) = opt(x13::regarima::LaborDay::new(rng.range(0, 26) as u8))
        {
            c = c.with_labor_day(e);
        }
        if rng.chance(0.2)
            && let Some(e) = opt(x13::regarima::Thanksgiving::new(rng.range(-9, 18) as i8))
        {
            c = c.with_thanksgiving(e);
        }
        if rng.chance(0.2) {
            c = c.with_selection(x13::regarima::CalendarSelection::AicTest);
        }
        r = r.with_calendar(c);
    }
    if rng.chance(0.4) {
        let mut o = x13::regarima::OutlierSpec::default();
        if rng.chance(0.6) {
            o = o.with_types(random_types(rng));
        }
        if rng.chance(0.3) {
            o = o.with_method(if rng.chance(0.5) {
                x13::regarima::OutlierMethod::AddOne
            } else {
                x13::regarima::OutlierMethod::AddAll
            });
        }
        if rng.chance(0.4) {
            o = o.with_critical_values(match rng.below(2) {
                0 => match random_cv(rng) {
                    Some(v) => x13::regarima::OutlierCriticalValues::shared(v),
                    None => x13::regarima::OutlierCriticalValues::Automatic,
                },
                _ => x13::regarima::OutlierCriticalValues::per_type(
                    random_cv(rng),
                    random_cv(rng),
                    random_cv(rng),
                ),
            });
        }
        if rng.chance(0.3)
            && let Some(rate) = opt(x13::regarima::TemporaryChangeRate::new(rng.pick(&[
                1e-300,
                0.001,
                0.3,
                0.7,
                0.95,
                0.999_999_999,
                1.0,
            ])))
        {
            // A request has one TC rate. Drawing it here keeps each
            // `(seed, index)` case reproducible.
            r = r.with_temporary_change_rate(rate);
        }
        if rng.chance(0.25) {
            let a = random_period(rng, start, n);
            let b = random_period(rng, start, n);
            o = o.with_span(x13::PeriodSpan::new(a, b));
        }
        r = r.with_outliers(o);
    }
    if rng.chance(0.2) {
        let k = rng.range(1, 5) as usize;
        let mut outliers = Vec::new();
        for _ in 0..k {
            let kind = rng.pick(&[
                x13::regarima::OutlierType::Additive,
                x13::regarima::OutlierType::LevelShift,
                x13::regarima::OutlierType::TemporaryChange,
            ]);
            outliers.push(x13::regarima::ManualOutlier::new(
                kind,
                random_period(rng, start, n),
            ));
        }
        if rng.chance(0.2) && !outliers.is_empty() {
            outliers.push(outliers[0]);
        }
        r = r.with_manual_outliers(&outliers);
    }
    if rng.chance(0.15) {
        let k = rng.range(1, 3) as usize;
        let mut regs = Vec::new();
        let mut first: Option<Vec<f64>> = None;
        for _ in 0..k {
            let kind = rng.pick(&[
                x13::regarima::UserRegressorType::TradingDay,
                x13::regarima::UserRegressorType::Holiday,
                x13::regarima::UserRegressorType::User,
            ]);
            let shift = rng.pick(&[0i64, 0, 0, -1, -12, -24, 1, 12]);
            let regressor_start = start.shift(shift).unwrap_or(start);
            let extra = rng.pick(&[0i64, 12, 24, 36, 120, -1, -12, 1, 360]);
            let len = (n as i64 - shift + extra).max(0) as usize;
            let len = len.min(5000);
            let values: Vec<f64> = match rng.below(7) {
                0 => vec![1.0; len],
                1 => vec![0.0; len],
                2 => (0..len).map(|t| t as f64).collect(),
                3 => first.clone().unwrap_or_else(|| vec![1.0; len]),
                4 => (0..len).map(|_| rng.normal() * 1e300).collect(),
                5 => (0..len)
                    .map(|t| if t % 12 == 3 { 1.0 } else { 0.0 })
                    .collect(),
                _ => (0..len).map(|_| rng.normal()).collect(),
            };
            if first.is_none() {
                first = Some(values.clone());
            }
            let values = if values.len() != len {
                let mut v = values;
                v.resize(len, 0.5);
                v
            } else {
                values
            };
            if let Some(u) = opt(x13::regarima::UserRegressor::new(
                kind,
                x13::Series::new(regressor_start, &values),
            )) {
                regs.push(u);
            }
        }
        r = r.with_user_regressors(&regs);
    }
    if rng.chance(0.15) {
        let a = random_period(rng, start, n);
        let b = random_period(rng, start, n);
        r = r.with_model_span(x13::PeriodSpan::new(a, b));
    }
    let _ = monthly;
    r
}

pub(crate) fn random_spec(seed: u64, start: x13::Period, n: usize) -> x13::Spec {
    let mut rng = Rng::new(seed);
    let monthly = start.frequency() == x13::Frequency::Monthly;
    let modes = [
        x13::x11::Mode::Additive,
        x13::x11::Mode::Multiplicative,
        x13::x11::Mode::LogAdditive,
        x13::x11::Mode::PseudoAdditive,
    ];
    let kind = rng.below(10);
    if kind >= 7 {
        return x13::Spec::seats(random_regarima(&mut rng, start, n));
    }
    let mode = (kind < 5).then(|| rng.pick(&modes));
    let regarima = (kind >= 3).then(|| random_regarima(&mut rng, start, n));
    let mut options = if rng.chance(0.7) {
        random_x11_options(&mut rng, monthly)
    } else {
        x13::x11::Options::new()
    };
    if let Some(mode) = mode {
        options = options.with_mode(mode);
    }
    if rng.chance(0.2) {
        options =
            options.with_final_outliers(x13::x11::FinalOutliers::Remove(random_types(&mut rng)));
    }
    if rng.chance(0.2) {
        options =
            options.with_temporary_change_allocation(x13::x11::TemporaryChangeAllocation::Trend);
    }
    if rng.chance(0.2) {
        options = options.with_holiday_adjustment(x13::x11::HolidayAdjustment::Retain);
    }
    match regarima {
        Some(regarima) => x13::Spec::x11_regarima(regarima, options),
        None => x13::Spec::x11(options),
    }
}

/// The `Spec` and first period a case runs, or `kind:detail` naming where building
/// failed.
pub(crate) fn build(case: &Case) -> Result<(x13::Spec, x13::Period), String> {
    let start = match case.frequency {
        Frequency::Monthly => x13::Period::monthly(case.start_year, case.start_subperiod),
        Frequency::Quarterly => x13::Period::quarterly(case.start_year, case.start_subperiod),
        Frequency::Semiannual => return Err("period_error:semiannual".into()),
    }
    .map_err(|e| format!("period_error:{e:?}"))?;
    let values = case.values();
    match &case.spec {
        SpecSource::Random { seed } => Ok((random_spec(*seed, start, values.len()), start)),
        SpecSource::Named { name } => {
            let named = named::parse(name).map_err(|e| format!("spec_error:{e}"))?;
            let spec = named
                .to_spec(start, values.len())
                .map_err(|e| format!("spec_error:{e}"))?;
            Ok((spec, start))
        }
        SpecSource::Row { id } => {
            let (series, config) = row_configuration(case, id)?;
            crate::rust_spec::build(&series, config).map_err(|e| format!("spec_error:{e:?}"))
        }
    }
}

/// The call-table row's series and configuration for this case's frequency.
pub(crate) fn row_configuration(
    case: &Case,
    id: &str,
) -> Result<(Series, &'static crate::OracleConfiguration), String> {
    let row = context()
        .rows
        .iter()
        .find(|row| row.id == id)
        .ok_or_else(|| format!("spec_error:no call-table row {id}"))?;
    let config = if case.frequency == Frequency::Quarterly {
        row.quarterly_configuration
            .as_ref()
            .unwrap_or(&row.configuration)
    } else {
        &row.configuration
    };
    let series = Series {
        series_id: "fuzz".into(),
        frequency: case.frequency,
        start: Period::new(case.start_year, case.start_subperiod),
        values: crate::series::build_62_values(&case.values()),
    };
    Ok((series, config))
}
