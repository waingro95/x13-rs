use crate::api::*;
use crate::regarima::{CalendarFeature, Easter, ForecastHorizon};
use crate::{Period, PeriodError};

fn too_long(positions: usize, years: usize) -> Result<(), Error> {
    Err(Error::SeriesTooLong {
        positions,
        years,
        maximum_positions: 1020,
        maximum_years: 85,
    })
}

fn airline() -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(0, 1, 1).unwrap(),
        ArimaOrder::new(0, 1, 1).unwrap(),
    )
    .unwrap()
}

fn seasonal_values(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| 100.0 + (index % 12) as f64 + 0.1 * index as f64)
        .collect()
}

#[test]
fn print_format_rejects_magnitudes_wider_than_21_columns() {
    assert_eq!(validate_print_format(&[1.0, 9.0e18, -9.0e18]), Ok(()));
    assert_eq!(
        validate_print_format(&[1.0, -1.0e19]),
        Err(Error::ObservationTooLarge { index: 1 })
    );
    assert_eq!(validate_print_format(&[-99999.0, 0.0]), Ok(()));
    let mut values = vec![100.0; 120];
    values[7] = 2.0e19;
    assert_eq!(
        adjust(
            Series::new(Period::monthly(2000, 1).unwrap(), &values),
            &Spec::x11(x11::Options::new())
        )
        .unwrap_err(),
        Error::ObservationTooLarge { index: 7 }
    );
}

#[test]
fn augmented_length_counts_leading_positions_and_extensions() {
    let january = TimeAxis::monthly(1900, 1).unwrap();
    assert_eq!(validate_augmented_length(january, 1020, 0, 0), Ok(()));
    assert_eq!(validate_augmented_length(january, 1008, 0, 12), Ok(()));
    assert_eq!(
        validate_augmented_length(january, 1021, 0, 0),
        too_long(1021, 86)
    );
    let february = TimeAxis::monthly(1900, 2).unwrap();
    assert_eq!(validate_augmented_length(february, 1019, 0, 0), Ok(()));
    assert_eq!(
        validate_augmented_length(february, 1020, 0, 0),
        too_long(1021, 86)
    );
    // Five backcasts from March start in October: nine leading months.
    let march = TimeAxis::monthly(1900, 3).unwrap();
    assert_eq!(validate_augmented_length(march, 1000, 5, 6), Ok(()));
    assert_eq!(
        validate_augmented_length(march, 1001, 5, 6),
        too_long(1021, 86)
    );
    // Quarterly input reaches PYRS (340 quarters) first.
    let third_quarter = TimeAxis::quarterly(1950, 3).unwrap();
    assert_eq!(validate_augmented_length(third_quarter, 334, 0, 4), Ok(()));
    assert_eq!(
        validate_augmented_length(third_quarter, 335, 0, 4),
        too_long(341, 86)
    );
    let extensions = ForecastSpec::new(ForecastHorizon::OneYear, 0).unwrap();
    assert_eq!(
        validate_regarima_x11_length(third_quarter, 11, extensions),
        Err(Error::TooShort {
            method: crate::Method::X11,
            actual: 11,
            minimum: 12,
        })
    );
    assert_eq!(
        validate_regarima_x11_length(third_quarter, 335, extensions),
        too_long(341, 86)
    );
    let long = vec![100.0; 1021];
    assert_eq!(
        Err(adjust(
            Series::new(january.start(), &long),
            &Spec::x11(x11::Options::new())
        )
        .unwrap_err()),
        too_long(1021, 86)
    );
}

#[test]
fn series_past_the_last_year_is_rejected() {
    let start = Period::monthly(i32::MAX, 12).unwrap();
    for spec in [
        Spec::x11(x11::Options::new()),
        Spec::seats(RegArimaSpec::new(airline())),
    ] {
        assert_eq!(
            adjust(Series::new(start, &[1.0, 2.0]), &spec).unwrap_err(),
            Error::InvalidPeriod(PeriodError::PeriodOverflow { index: 1 }),
            "{spec:?}"
        );
    }
}

#[test]
fn empty_series_is_rejected() {
    let series = Series::new(Period::monthly(2000, 1).unwrap(), &[]);
    assert_eq!(
        adjust(series, &Spec::x11(x11::Options::new())).unwrap_err(),
        Error::EmptyInput
    );
}

#[test]
fn regarima_x11_rejects_a_series_or_model_span_too_short_to_fit() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let spec = Spec::x11_regarima(
        RegArimaSpec::new(airline()),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    assert_eq!(
        adjust(Series::new(axis.start(), &seasonal_values(24)), &spec).unwrap_err(),
        Error::TooShort {
            method: crate::Method::X11,
            actual: 24,
            minimum: 36,
        }
    );
    // rgarma.f: 20 - 13 differenced observations cannot cover 1 + 13 lags.
    let model = ArimaModel::new(
        ArimaOrder::new(1, 1, 1).unwrap(),
        ArimaOrder::new(0, 1, 1).unwrap(),
    )
    .unwrap();
    let span = PeriodSpan::new(axis.period_at(100).unwrap(), axis.period_at(119).unwrap());
    let spec = Spec::x11_regarima(
        RegArimaSpec::new(model).with_model_span(span),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    assert_eq!(
        adjust(Series::new(axis.start(), &seasonal_values(120)), &spec).unwrap_err(),
        Error::InsufficientEffectiveObservations {
            actual: 7,
            minimum: 14,
        }
    );
}

#[test]
fn late_model_span_backcasts_do_not_widen_calendar_years() {
    let axis = TimeAxis::monthly(1901, 1).unwrap();
    let span = PeriodSpan::new(axis.period_at(40).unwrap(), axis.period_at(119).unwrap());
    let regarima = RegArimaSpec::new(airline())
        .with_forecast(ForecastSpec::new(ForecastHorizon::OneYear, 12).unwrap())
        .with_calendar(CalendarSpec::new().with_easter(Easter::new(8).unwrap()));
    assert!(matches!(
        regarima.validate(&seasonal_values(120), axis),
        Err(Error::CalendarYearOutOfRange {
            feature: CalendarFeature::Easter,
            year: 1900,
            ..
        })
    ));
    assert_eq!(
        regarima
            .with_model_span(span)
            .validate(&seasonal_values(120), axis),
        Ok(())
    );
}
