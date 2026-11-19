use x13::{Error, Frequency, Period, Series, Spec, x11, x11::SeasonalFilter as Filter};

fn values(length: usize, period: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 100.0 + 0.2 * i as f64 + (i % period) as f64)
        .collect()
}

fn requested(filters: x11::SeasonalFilters) -> Spec {
    Spec::x11(
        x11::Options::new()
            .with_seasonal_filters(filters)
            .with_short_series_filters(x11::ShortSeriesFilters::UseRequested),
    )
}

/// The fields of an `InsufficientSeasonalHistory` error, in declaration
/// order. The variant is `#[non_exhaustive]`, so a test cannot build one.
fn history(error: Error) -> (Frequency, u8, Filter, Filter, usize, usize) {
    match error {
        Error::InsufficientSeasonalHistory {
            frequency,
            calendar_period,
            requested,
            resolved,
            actual,
            minimum,
            ..
        } => (
            frequency,
            calendar_period,
            requested,
            resolved,
            actual,
            minimum,
        ),
        other => panic!("unexpected error {other:?}"),
    }
}

#[test]
fn short_history_error_names_the_period_and_resolved_filter() {
    for (start, filters, expected) in [
        (
            Period::monthly(2001, 7).unwrap(),
            x11::SeasonalFilters::Monthly([Filter::X11Default; 12]),
            (Frequency::Monthly, 1, Filter::X11Default, Filter::S3x3),
        ),
        (
            Period::quarterly(2001, 3).unwrap(),
            x11::SeasonalFilters::Quarterly([Filter::Msr; 4]),
            (Frequency::Quarterly, 1, Filter::Msr, Filter::S3x3),
        ),
    ] {
        let period = usize::from(start.frequency().periods_per_year());
        let error = x13::adjust(
            Series::new(start, &values(3 * period, period)),
            &requested(filters),
        )
        .unwrap_err();
        assert_eq!(
            history(error),
            (expected.0, expected.1, expected.2, expected.3, 2, 3)
        );
    }
}

#[test]
fn mixed_filters_report_the_first_period_without_enough_history() {
    let mut months = [Filter::S3x1; 12];
    months[2] = Filter::S3x3; // March; July start and centered January start.
    let start = Period::monthly(2001, 7).unwrap();
    assert_eq!(
        history(
            x13::adjust(
                Series::new(start, &values(37, 12)),
                &requested(x11::SeasonalFilters::Monthly(months))
            )
            .unwrap_err()
        ),
        (Frequency::Monthly, 3, Filter::S3x3, Filter::S3x3, 2, 3)
    );

    let mut quarters = [Filter::Stable; 4];
    quarters[1] = Filter::S3x5; // Q2; Q3 start and centered Q1 start.
    let start = Period::quarterly(2001, 3).unwrap();
    assert_eq!(
        history(
            x13::adjust(
                Series::new(start, &values(13, 4)),
                &requested(x11::SeasonalFilters::Quarterly(quarters))
            )
            .unwrap_err()
        ),
        (Frequency::Quarterly, 2, Filter::S3x5, Filter::S3x5, 2, 3)
    );
}

#[test]
fn constant_series_history_excludes_requested_extensions() {
    let start = Period::quarterly(2000, 1).unwrap();
    let options = x11::Options::new()
        .with_seasonal_filter(Filter::S3x3)
        .with_short_series_filters(x11::ShortSeriesFilters::UseRequested);
    let spec = Spec::x11_regarima(x13::RegArimaSpec::automatic_model(), options);
    assert_eq!(
        history(x13::adjust(Series::new(start, &[100.0; 12]), &spec).unwrap_err()),
        (Frequency::Quarterly, 3, Filter::S3x3, Filter::S3x3, 2, 3)
    );
}
