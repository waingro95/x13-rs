//! The dates the public outputs carry: components, calendar effects and the
//! SEATS transitory component start at the input's first period, forecasts
//! one period after its last, and backcasts end one period before its first.

use crate::api::*;
use crate::regarima::{CalendarBasis, FlowTradingDay, ForecastHorizon, TradingDayContrast};
use crate::{Frequency, Period, Series};

const MONTHS: usize = 120;
const QUARTERS: usize = 60;

fn airline() -> RegArimaSpec {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
}

/// A positive seasonal series with deterministic noise.
fn values(count: usize, periods_per_year: usize) -> Vec<f64> {
    let mut state = 12_345_u64;
    (0..count)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let angle = 2.0 * std::f64::consts::PI * (index % periods_per_year) as f64
                / periods_per_year as f64;
            100.0 + 0.2 * index as f64 + 8.0 * angle.sin() + 3.0 * noise
        })
        .collect()
}

/// An AR(2) regular factor gives SEATS a transitory component.
fn with_transitory() -> RegArimaSpec {
    let regular = ArimaOrder::new(2, 1, 1).unwrap();
    let seasonal = ArimaOrder::new(0, 1, 1).unwrap();
    RegArimaSpec::new(ArimaModel::new(regular, seasonal).unwrap()).with_transform(Transform::Log)
}

fn with_trading_day(regarima: RegArimaSpec) -> RegArimaSpec {
    regarima.with_calendar(
        CalendarSpec::new().with_basis(CalendarBasis::FlowTradingDay(FlowTradingDay::new(
            TradingDayContrast::WeekdayWeekend,
        ))),
    )
}

fn extended(regarima: RegArimaSpec, forecasts: u16, backcasts: u16) -> RegArimaSpec {
    regarima
        .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(forecasts), backcasts).unwrap())
}

/// Every column on the series' own scale starts at `start`.
fn assert_dated_from(adjustment: &Adjustment, start: Period, count: usize, label: &str) {
    assert_eq!(adjustment.start(), start, "{label}");
    let mut components = vec![adjustment.seasonally_adjusted(), adjustment.irregular()];
    components.extend(adjustment.seasonal());
    components.extend(adjustment.trend_cycle());
    assert!(components.len() >= 3, "{label}");
    if let Some(transitory) = adjustment.seats_transitory() {
        components.push(transitory.series());
    }
    if let Some(calendar) = adjustment.calendar_effects() {
        components.extend([
            calendar.estimated(),
            calendar.fixed_length(),
            calendar.combined(),
        ]);
    }
    for component in components {
        assert_eq!(component.start(), start, "{label}");
        assert_eq!(component.len(), count, "{label}");
    }
}

#[test]
fn output_series_start_at_the_input_start() {
    let start = Period::monthly(2001, 3).unwrap();
    let input = values(MONTHS, 12);
    let period = |index: i64| start.shift(index).unwrap();
    let late_span = PeriodSpan::new(period(24), period(MONTHS as i64 - 1));
    let early_end = PeriodSpan::new(period(0), period(95));
    let cases = [
        ("direct X-11", Spec::x11(x11::Options::new())),
        (
            "regARIMA X-11 with extensions",
            Spec::x11_regarima(
                with_trading_day(extended(airline(), 12, 12)),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "regARIMA X-11, model span starting late",
            Spec::x11_regarima(
                with_trading_day(extended(airline(), 12, 12)).with_model_span(late_span),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "regARIMA X-11, model span ending early",
            Spec::x11_regarima(
                with_trading_day(extended(airline(), 12, 12)).with_model_span(early_end),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "SEATS with forecasts",
            Spec::seats(with_trading_day(extended(airline(), 12, 0))),
        ),
        (
            "SEATS, model span starting late",
            Spec::seats(with_trading_day(extended(airline(), 12, 0)).with_model_span(late_span)),
        ),
        (
            "SEATS, model span ending early",
            Spec::seats(with_trading_day(extended(airline(), 12, 0)).with_model_span(early_end)),
        ),
        (
            "SEATS with a transitory component",
            Spec::seats(with_trading_day(with_transitory())),
        ),
        (
            "SEATS with a transitory component, model span starting late",
            Spec::seats(with_trading_day(with_transitory()).with_model_span(late_span)),
        ),
        (
            "log SEATS",
            Spec::seats(with_trading_day(airline().with_transform(Transform::Log))),
        ),
    ];
    let mut dated_calendars = 0;
    let mut dated_transitories = 0;
    for (label, spec) in cases {
        let adjustment = adjust(Series::new(start, &input), &spec).unwrap();
        assert_dated_from(&adjustment, start, MONTHS, label);
        dated_calendars += usize::from(adjustment.calendar_effects().is_some());
        dated_transitories += usize::from(adjustment.seats_transitory().is_some());
    }
    // The loop must have reached both optional outputs.
    assert!(
        dated_calendars >= 9,
        "calendar effects reached {dated_calendars} times"
    );
    assert!(
        dated_transitories >= 2,
        "transitory reached {dated_transitories} times"
    );
}

#[test]
fn constant_easter_and_automatic_outputs_start_at_the_input_start() {
    let start = Period::monthly(2001, 3).unwrap();
    let noisy = values(MONTHS, 12);
    let constant = [100.0; MONTHS];
    let easter = x11::Options::new().with_easter_estimation(true);
    let log_airline = || airline().with_transform(Transform::Log);
    let cases: [(&str, &[f64], Spec); 6] = [
        (
            "constant series skips the fit",
            &constant,
            Spec::x11_regarima(
                airline(),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "constant series with legacy Easter",
            &constant,
            Spec::x11_regarima(log_airline(), easter.with_mode(x11::Mode::Multiplicative)),
        ),
        (
            "direct X-11 with legacy Easter",
            &noisy,
            Spec::x11(easter.with_mode(x11::Mode::Multiplicative)),
        ),
        (
            "regARIMA X-11 with legacy Easter",
            &noisy,
            Spec::x11_regarima(
                extended(log_airline(), 12, 12),
                easter.with_mode(x11::Mode::Multiplicative),
            ),
        ),
        (
            "automatic model under X-11",
            &noisy,
            Spec::x11_regarima(
                RegArimaSpec::automatic_model(),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "automatic model under SEATS",
            &noisy,
            Spec::seats(RegArimaSpec::automatic_model()),
        ),
    ];
    let mut easter_results = 0;
    for (label, input, spec) in cases {
        let adjustment = adjust(Series::new(start, input), &spec).unwrap();
        assert_dated_from(&adjustment, start, MONTHS, label);
        if let Some(easter) = adjustment.x11_easter() {
            assert_eq!(easter.factors().start(), start, "{label}");
            assert_eq!(easter.factors().len(), MONTHS, "{label}");
            easter_results += 1;
        }
    }
    // The constant series and both fitted pipelines produced Easter factors.
    assert_eq!(easter_results, 3);
}

#[test]
fn quarterly_output_series_start_at_the_input_start() {
    let start = Period::quarterly(2003, 2).unwrap();
    let input = values(QUARTERS, 4);
    for (label, spec) in [
        ("direct X-11", Spec::x11(x11::Options::new())),
        (
            "regARIMA X-11",
            Spec::x11_regarima(
                extended(airline(), 8, 8),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
    ] {
        let adjustment = adjust(Series::new(start, &input), &spec).unwrap();
        assert_dated_from(&adjustment, start, QUARTERS, label);
    }
}

#[test]
fn gap_filled_output_series_start_at_the_input_start() {
    let start = Period::monthly(1999, 11).unwrap();
    let mut input = values(MONTHS, 12);
    input[17] = f64::NAN;
    for (label, spec) in [
        (
            "regARIMA X-11",
            Spec::x11_regarima(
                extended(airline(), 12, 12),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        ("SEATS", Spec::seats(extended(airline(), 12, 0))),
    ] {
        let adjustment = adjust(Series::new(start, &input), &spec).unwrap();
        assert_dated_from(&adjustment, start, MONTHS, label);
    }
}

fn assert_all_start_at(series: [Series<'_>; 4], start: Period, count: usize, label: &str) {
    for column in series {
        assert_eq!(column.start(), start, "{label}");
        assert_eq!(column.len(), count, "{label}");
    }
}

#[test]
fn forecasts_start_after_the_last_observation() {
    let start = Period::monthly(2001, 3).unwrap();
    let after_last = start.shift(MONTHS as i64).unwrap();
    assert_eq!(after_last, Period::monthly(2011, 3).unwrap());
    let input = values(MONTHS, 12);
    let period = |index: i64| start.shift(index).unwrap();
    let early_end = PeriodSpan::new(period(0), period(95));
    let late_span = PeriodSpan::new(period(24), period(MONTHS as i64 - 1));
    let regarima = || extended(airline(), 18, 0);
    let cases = [
        (
            "regARIMA X-11",
            Spec::x11_regarima(
                regarima(),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "regARIMA X-11, model span ending early",
            Spec::x11_regarima(
                regarima().with_model_span(early_end),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        (
            "regARIMA X-11, model span starting late",
            Spec::x11_regarima(
                regarima().with_model_span(late_span),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ),
        ("SEATS", Spec::seats(regarima())),
        (
            "SEATS, model span ending early",
            Spec::seats(regarima().with_model_span(early_end)),
        ),
        (
            "log SEATS",
            Spec::seats(regarima().with_transform(Transform::Log)),
        ),
    ];
    for (label, spec) in cases {
        let adjustment = adjust(Series::new(start, &input), &spec).unwrap();
        let forecasts = adjustment.forecasts().unwrap();
        assert_eq!(forecasts.start(), after_last, "{label}");
        assert_all_start_at(
            [
                forecasts.series(),
                forecasts.lower_95(),
                forecasts.upper_95(),
                forecasts.transformed(),
            ],
            after_last,
            18,
            label,
        );
        // The variance-scale columns stay plain slices of the same length.
        assert_eq!(forecasts.transformed_standard_errors().len(), 18, "{label}");
    }
}

#[test]
fn backcasts_end_before_the_first_observation() {
    let start = Period::monthly(2001, 3).unwrap();
    let first_backcast = start.shift(-12).unwrap();
    let input = values(MONTHS, 12);
    let spec = Spec::x11_regarima(
        extended(airline(), 12, 12),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let adjustment = adjust(Series::new(start, &input), &spec).unwrap();
    let backcasts = adjustment.backcasts().unwrap();
    assert_eq!(backcasts.start(), first_backcast);
    assert_all_start_at(
        [
            backcasts.series(),
            backcasts.lower_95(),
            backcasts.upper_95(),
            backcasts.transformed(),
        ],
        first_backcast,
        12,
        "backcasts",
    );
    // Twelve backcasts from one year before the start end the month before it.
    let last_backcast = backcasts.series().periods().unwrap().last();
    assert_eq!(last_backcast, Some(start.shift(-1).unwrap()));
    assert_eq!(backcasts.series().frequency(), Frequency::Monthly);
}
