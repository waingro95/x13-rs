use super::{SeatsUnsupportedCalendar, SeatsUnsupportedOption};
use crate::error::{UnsupportedConfiguration, UnsupportedDetail};
use crate::regarima::{
    AutoTransformOptions, CalendarBasis, CalendarFeature, CalendarSpec, ConstantSelection, Easter,
    ForecastHorizon, ForecastSpec, LaborDay, StockTradingDay, Thanksgiving, TradingDayContrast,
    TransformSelection, UserRegressor, UserRegressorType,
};
use crate::time::TimeAxis;
use crate::{
    ArimaModel, ArimaOrder, Error, Frequency, Method, PeriodError, PeriodSpan, RegArimaSpec,
    Series, Spec, Transform, adjust,
};

/// Positive monthly seasonal values with deterministic noise.
fn values(count: usize) -> Vec<f64> {
    let mut state = 12_345_u64;
    (0..count)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let angle = 2.0 * std::f64::consts::PI * (index % 12) as f64 / 12.0;
            100.0 + 0.2 * index as f64 + 8.0 * angle.sin() + 3.0 * noise
        })
        .collect()
}

fn easter() -> CalendarSpec {
    CalendarSpec::new().with_easter(Easter::new(8).unwrap())
}

fn airline() -> RegArimaSpec {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
}

fn beyond(feature: CalendarFeature, year: i32) -> Result<(), Error> {
    Err(Error::CalendarYearOutOfRange {
        feature,
        year,
        minimum: 1901,
        maximum: Some(2100),
    })
}

fn unsupported(option: SeatsUnsupportedOption) -> Error {
    Error::from(UnsupportedDetail::SeatsOption(option))
}

fn reject(values: &[f64], axis: TimeAxis, request: RegArimaSpec) -> Error {
    adjust(Series::new(axis.start(), values), &Spec::seats(request)).unwrap_err()
}

#[test]
fn supplied_regressors_must_cover_three_forecast_years() {
    for axis in [
        TimeAxis::monthly(2000, 1).unwrap(),
        TimeAxis::quarterly(2000, 1).unwrap(),
    ] {
        let s = usize::from(axis.frequency().periods_per_year());
        let values = vec![100.0; 10 * s];
        let request =
            airline().with_forecast(ForecastSpec::new(ForecastHorizon::OneYear, 0).unwrap());
        let supplied = |years: usize| {
            let column = vec![1.0; (10 + years) * s];
            UserRegressor::new(UserRegressorType::User, Series::new(axis.start(), &column)).unwrap()
        };
        let short = request.clone().with_user_regressors(&[supplied(1)]);
        assert!(matches!(
            super::validate_request(&values, axis, short),
            Err(Error::UserRegressorCoverage { .. })
        ));
        let covered = request.with_user_regressors(&[supplied(3)]);
        super::validate_request(&values, axis, covered).unwrap();
    }
}

#[test]
fn automatic_model_rejects_holidays_past_2100_in_its_forecasts() {
    // 2088-01..2099-12 plus 36 internal forecasts reaches 2102.
    let axis = TimeAxis::monthly(2088, 1).unwrap();
    let series = values(144);
    let automatic = RegArimaSpec::automatic_model().with_transform(Transform::Log);
    for (calendar, feature) in [
        (easter(), CalendarFeature::Easter),
        (
            CalendarSpec::new().with_labor_day(LaborDay::new(8).unwrap()),
            CalendarFeature::LaborDay,
        ),
    ] {
        let request = automatic.clone().with_calendar(calendar);
        assert_eq!(
            super::validate_request(&series, axis, request.clone()),
            beyond(feature, 2102)
        );
        assert_eq!(
            adjust(Series::new(axis.start(), &series), &Spec::seats(request)).map(|_| ()),
            beyond(feature, 2102)
        );
    }
    // automd evaluates through the series end, whatever the model span.
    let span = PeriodSpan::new(axis.period_at(0).unwrap(), axis.period_at(107).unwrap());
    assert_eq!(
        super::validate_request(
            &series,
            axis,
            automatic
                .clone()
                .with_calendar(easter())
                .with_model_span(span)
        ),
        beyond(CalendarFeature::Easter, 2102)
    );
    // Two years earlier the internal forecasts end in 2100-12.
    let earlier = TimeAxis::monthly(2086, 1).unwrap();
    super::validate_request(&series, earlier, automatic.with_calendar(easter())).unwrap();
}

#[test]
fn automatic_transform_checks_easter_over_the_log_candidate_span() {
    let axis = TimeAxis::monthly(2088, 1).unwrap();
    let series = values(144);
    let automatic = airline()
        .with_transform(TransformSelection::Automatic(AutoTransformOptions::new()))
        .with_calendar(easter());
    assert_eq!(
        super::validate_request(&series, axis, automatic.clone()),
        beyond(CalendarFeature::Easter, 2102)
    );
    // The candidate means cover the model span plus 36 forecasts:
    // 2097-12 + 36 is still 2100.
    let span = PeriodSpan::new(axis.period_at(0).unwrap(), axis.period_at(119).unwrap());
    super::validate_request(&series, axis, automatic.clone().with_model_span(span)).unwrap();
    // A nonpositive observation bypasses the comparison (editor.f).
    let mut nonpositive = series.clone();
    nonpositive[5] = -1.0;
    super::validate_request(&nonpositive, axis, automatic).unwrap();
}

#[test]
fn explicit_model_and_transform_run_with_forecasts_past_2100() {
    // Only the requested year of forecasts carries holidays here, so the
    // 2088 case stays Ok (its SA matches Build 62's s11).
    let axis = TimeAxis::monthly(2088, 1).unwrap();
    let request = airline()
        .with_transform(Transform::Log)
        .with_calendar(easter());
    adjust(
        Series::new(axis.start(), &values(144)),
        &Spec::seats(request),
    )
    .unwrap();
}

#[test]
fn model_parts_split_the_coefficients_by_order() {
    let coefficients = [0.4, -0.1, 0.7, 0.2, 0.3];
    let model = super::Model::new(Frequency::Monthly, [2, 1, 1], [1, 0, 1], &coefficients).unwrap();
    assert_eq!(model.regular_ar(), [0.4, -0.1]);
    assert_eq!(model.seasonal_ar(), [0.7]);
    assert_eq!(model.regular_ma(), [0.2]);
    assert_eq!(model.seasonal_ma(), [0.3]);
    assert_eq!(model.coefficients(), coefficients);

    let airline =
        super::Model::new(Frequency::Quarterly, [0, 1, 1], [0, 1, 1], &[0.6, 0.5]).unwrap();
    assert!(airline.regular_ar().is_empty() && airline.seasonal_ar().is_empty());
    assert_eq!(airline.regular_ma(), [0.6]);
    assert_eq!(airline.seasonal_ma(), [0.5]);
}

#[test]
fn automatic_model_checks_options_before_the_series() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let request = RegArimaSpec::automatic_model();
    assert_eq!(
        reject(
            &[1.0],
            axis,
            request.clone().with_calendar(CalendarSpec::new())
        ),
        unsupported(SeatsUnsupportedOption::Calendar(
            SeatsUnsupportedCalendar::Empty
        ))
    );
    for calendar in [
        CalendarSpec::new().with_basis(CalendarBasis::StockTradingDay(
            StockTradingDay::new(TradingDayContrast::WeekdayWeekend, 15).unwrap(),
        )),
        CalendarSpec::new().with_labor_day(LaborDay::new(8).unwrap()),
        CalendarSpec::new().with_thanksgiving(Thanksgiving::new(1).unwrap()),
    ] {
        assert_eq!(
            reject(&[1.0], axis, request.clone().with_calendar(calendar)),
            Error::TooShort {
                method: Method::Seats,
                actual: 1,
                minimum: 36,
            }
        );
    }
    for extensions in [
        ForecastSpec::new(ForecastHorizon::Periods(37), 0).unwrap(),
        ForecastSpec::new(ForecastHorizon::Periods(120), 1).unwrap(),
    ] {
        assert_eq!(
            reject(&[1.0], axis, request.clone().with_forecast(extensions)),
            unsupported(SeatsUnsupportedOption::RequestedExtensions)
        );
    }
}

#[test]
fn automatic_model_rejects_bad_series_before_selection() {
    let request = RegArimaSpec::automatic_model();
    let monthly = TimeAxis::monthly(2000, 1).unwrap();
    for (values, expected) in [
        (vec![], Error::EmptyInput),
        (vec![f64::NEG_INFINITY], Error::NonFiniteValue { index: 0 }),
        (vec![f64::INFINITY], Error::NonFiniteValue { index: 0 }),
        (vec![0.0; 36], Error::AllZeroInput),
    ] {
        assert_eq!(reject(&values, monthly, request.clone()), expected);
    }
    assert_eq!(
        reject(
            &[1.0; 36],
            TimeAxis::monthly(-1, 1).unwrap(),
            request.clone()
        ),
        Error::SeatsNegativeYear { year: -1 }
    );
    assert_eq!(
        reject(
            &[1.0; 36],
            TimeAxis::monthly(i32::MAX, 1).unwrap(),
            request.clone()
        ),
        Error::InvalidPeriod(PeriodError::PeriodOverflow { index: 35 })
    );
    for (axis, minimum, maximum) in [
        (monthly, 36, 780),
        (TimeAxis::quarterly(2000, 1).unwrap(), 16, 328),
        (TimeAxis::quarterly(2000, 2).unwrap(), 16, 327),
        (TimeAxis::quarterly(2000, 3).unwrap(), 16, 326),
        (TimeAxis::quarterly(2000, 4).unwrap(), 16, 325),
    ] {
        assert_eq!(
            reject(&vec![1.0; minimum - 1], axis, request.clone()),
            Error::TooShort {
                method: Method::Seats,
                actual: minimum - 1,
                minimum
            }
        );
        assert_eq!(
            reject(&vec![1.0; maximum + 1], axis, request.clone()),
            Error::SeatsTooLong {
                actual: maximum + 1,
                maximum
            }
        );
    }
    let mut values = vec![1.0; 36];
    for index in [3, 15, 27] {
        values[index] = f64::NAN;
    }
    assert_eq!(
        reject(&values, monthly, request.clone()),
        Error::MissingSeason {
            frequency: Frequency::Monthly,
            subperiod: 4,
        }
    );
    values[15] = 1.0;
    values[27] = 1.0;
    for value in [0.0, -1.0] {
        values[3] = value;
        assert_eq!(
            reject(
                &values,
                monthly,
                request.clone().with_transform(Transform::Log)
            ),
            Error::NonPositiveValue { index: 3 }
        );
    }
    for constant in [false, true] {
        assert_eq!(
            reject(&[1.0; 36], monthly, request.clone().with_constant(constant)),
            Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::AutomaticModelConstant
            ))
        );
    }
}

#[test]
fn default_automatic_model_rejects_a_constant_series_as_degenerate() {
    let request = RegArimaSpec::automatic_model();
    assert_eq!(request.constant(), ConstantSelection::Automatic);
    assert_eq!(
        request.transform(),
        TransformSelection::Explicit(Transform::None)
    );
    assert_eq!(request.forecast(), ForecastSpec::default());
    assert_eq!(request.calendar(), None);
    assert_eq!(request.outliers(), None);
    // A constant series passes the request checks and fails only because no
    // model can be estimated from it.
    assert_eq!(
        reject(&[1.0; 36], TimeAxis::monthly(2000, 1).unwrap(), request),
        Error::DegenerateSeries
    );
}

#[test]
fn explicit_model_rejects_a_constant_series_as_degenerate() {
    assert_eq!(
        reject(
            &[100.0; 120],
            TimeAxis::monthly(2000, 1).unwrap(),
            airline()
        ),
        Error::DegenerateSeries
    );
}

#[test]
fn second_difference_validation_checks_length_and_dates_not_values() {
    let request = RegArimaSpec::new(
        ArimaModel::new(
            ArimaOrder::new(0, 2, 1).unwrap(),
            ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap(),
    )
    .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(0), 0).unwrap());
    for (axis, maximum, minimum) in [
        (TimeAxis::monthly(2000, 1).unwrap(), 780, 36),
        (TimeAxis::monthly(2000, 2).unwrap(), 780, 36),
        (TimeAxis::monthly(2000, 12).unwrap(), 780, 36),
        (TimeAxis::quarterly(2000, 1).unwrap(), 328, 16),
        (TimeAxis::quarterly(2000, 2).unwrap(), 327, 16),
        (TimeAxis::quarterly(2000, 3).unwrap(), 326, 16),
        (TimeAxis::quarterly(2000, 4).unwrap(), 325, 16),
    ] {
        for n in [minimum, maximum] {
            let values: Vec<_> = (0..n).map(|j| -(j as f64) - 1.0).collect();
            assert_eq!(
                super::validate(&values, axis, request.clone()).unwrap(),
                super::SeatsRequest::Explicit(super::SeatsFamily::SecondDifference)
            );
        }
        assert!(matches!(
            super::validate(&vec![1.0; minimum - 1], axis, request.clone()),
            Err(Error::TooShort {
                method: Method::Seats,
                ..
            })
        ));
        assert!(matches!(
            super::validate(&vec![1.0; maximum + 1], axis, request.clone()),
            Err(Error::SeatsTooLong { .. })
        ));
    }
    assert!(matches!(
        super::validate(
            &[1.0; 36],
            TimeAxis::monthly(-1, 1).unwrap(),
            request.clone()
        ),
        Err(Error::SeatsNegativeYear { .. })
    ));
    assert!(matches!(
        super::validate(&[1.0; 36], TimeAxis::monthly(i32::MAX, 1).unwrap(), request),
        Err(Error::InvalidPeriod(_))
    ));
}
