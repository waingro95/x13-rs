use x13::{Error, ErrorKind, Period, RegArimaSpec, Series, Spec, x11};

const TREND: x11::Options = x11::Options::new().with_decomposition(x11::Decomposition::TrendOnly);

fn values(count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| 100.0 + i as f64 * 0.2 + [5.0, -3.0, 2.0, -4.0][i % 4])
        .collect()
}

const EXPLICIT_SEASONAL_FILTER: &str =
    "incompatible X-11 decomposition: trend-only X-11 does not accept an explicit seasonal filter";
const UNSAFE_LOG_ADDITIVE: &str =
    "incompatible X-11 decomposition: trend-only log-additive X-11 is undefined in Build 62";

/// The message of an error that refuses the spec.
fn conflict(error: Error) -> String {
    assert_eq!(error.kind(), ErrorKind::InvalidSpec, "{error:?}");
    error.to_string()
}

#[test]
fn trend_only_leaves_the_series_unadjusted() {
    assert_eq!(
        x11::Options::new().decomposition(),
        x11::Decomposition::SeasonalAdjustment
    );
    // Build 62's MSR default is stored but unused: trend-only fixes S3x5.
    assert_eq!(TREND.seasonal_filter(), x11::SeasonalFilter::Msr);
    let start = Period::quarterly(2000, 3).unwrap();
    let input = values(80);
    // Additive, so trend-only's neutral seasonal factors are zero.
    let additive = x11::Options::new().with_mode(x11::Mode::Additive);
    let ordinary = x13::adjust(Series::new(start, &input), &Spec::x11(additive)).unwrap();
    let trend = x13::adjust(
        Series::new(start, &input),
        &Spec::x11(additive.with_decomposition(x11::Decomposition::TrendOnly)),
    )
    .unwrap();
    assert_eq!(
        trend.diagnostics().x11().unwrap().decomposition(),
        x11::Decomposition::TrendOnly
    );
    assert!(
        trend
            .seasonal()
            .unwrap()
            .values()
            .iter()
            .all(|v| v.to_bits() == 0.0f64.to_bits())
    );
    assert_eq!(trend.seasonally_adjusted().values(), input);
    assert_ne!(
        ordinary.seasonally_adjusted().values(),
        trend.seasonally_adjusted().values()
    );
}

#[test]
fn trend_only_rejects_an_explicit_seasonal_filter() {
    let start = Period::monthly(2000, 1).unwrap();
    for filter in [
        x11::SeasonalFilter::S3x1,
        x11::SeasonalFilter::S3x3,
        x11::SeasonalFilter::S3x5,
        x11::SeasonalFilter::S3x9,
        x11::SeasonalFilter::S3x15,
        x11::SeasonalFilter::Stable,
        x11::SeasonalFilter::X11Default,
        x11::SeasonalFilter::Msr,
    ] {
        for options in [
            TREND.with_seasonal_filter(filter),
            x11::Options::new()
                .with_seasonal_filter(filter)
                .with_decomposition(x11::Decomposition::TrendOnly),
            TREND.with_seasonal_filters(x11::SeasonalFilters::Monthly([filter; 12])),
            x11::Options::new()
                .with_seasonal_filters(x11::SeasonalFilters::Monthly([filter; 12]))
                .with_decomposition(x11::Decomposition::TrendOnly),
        ] {
            for spec in [
                Spec::x11(options),
                Spec::x11_regarima(RegArimaSpec::automatic_model(), options),
            ] {
                assert_eq!(
                    conflict(x13::adjust(Series::new(start, &[100.0; 120]), &spec).unwrap_err()),
                    EXPLICIT_SEASONAL_FILTER
                );
            }
        }
    }
}

#[test]
fn trend_only_rejects_log_additive() {
    let start = Period::quarterly(2000, 2).unwrap();
    let spec = Spec::x11(TREND.with_mode(x11::Mode::LogAdditive));
    assert_eq!(
        conflict(x13::adjust(Series::new(start, &values(80)), &spec).unwrap_err()),
        UNSAFE_LOG_ADDITIVE
    );
    let mut invalid = values(80);
    invalid[3] = f64::NAN;
    assert!(
        !x13::adjust(Series::new(start, &invalid), &spec)
            .is_err_and(|error| error.kind() == ErrorKind::InvalidSpec)
    );
    let explicit = Spec::x11(
        TREND
            .with_mode(x11::Mode::LogAdditive)
            .with_seasonal_filter(x11::SeasonalFilter::S3x5),
    );
    assert_eq!(
        conflict(x13::adjust(Series::new(start, &values(80)), &explicit).unwrap_err()),
        EXPLICIT_SEASONAL_FILTER
    );
}

#[test]
fn input_domain_errors_precede_trend_only_conflicts() {
    let start = Period::quarterly(2000, 2).unwrap();
    let options = TREND.with_seasonal_filter(x11::SeasonalFilter::S3x5);
    let mut input = values(80);
    input[4] = -1.0;
    for mode in [
        x11::Mode::Multiplicative,
        x11::Mode::LogAdditive,
        x11::Mode::PseudoAdditive,
    ] {
        let error = x13::adjust(
            Series::new(start, &input),
            &Spec::x11(options.with_mode(mode)),
        )
        .unwrap_err();
        if mode == x11::Mode::PseudoAdditive {
            assert!(
                matches!(error, Error::NegativeValue { index: 4, .. }),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(error, Error::NonPositiveValue { index: 4, .. }),
                "{error:?}"
            );
        }
    }
}

#[test]
fn nonpositive_value_precedes_filter_conflict_with_explicit_model() {
    use x13::{ArimaModel, ArimaOrder};
    let order = ArimaOrder::new(0, 0, 0).unwrap();
    assert_nonpositive_value_precedes_filter_conflict(RegArimaSpec::new(
        ArimaModel::new(order, order).unwrap(),
    ));
}

#[test]
fn nonpositive_value_precedes_filter_conflict_with_automatic_model() {
    assert_nonpositive_value_precedes_filter_conflict(RegArimaSpec::automatic_model());
}

fn assert_nonpositive_value_precedes_filter_conflict(request: RegArimaSpec) {
    let start = Period::quarterly(2000, 2).unwrap();
    let spec = Spec::x11_regarima(
        request,
        TREND
            .with_seasonal_filter(x11::SeasonalFilter::S3x5)
            .with_mode(x11::Mode::Multiplicative),
    );
    for nonpositive in [0.0, -1.0] {
        let mut input = values(80);
        input[4] = nonpositive;
        let error = x13::adjust(Series::new(start, &input), &spec).unwrap_err();
        assert!(
            matches!(error, Error::NonPositiveValue { index: 4, .. }),
            "identity preprocessing with observation {nonpositive}: {error:?}"
        );
    }
    assert_eq!(
        conflict(x13::adjust(Series::new(start, &values(80)), &spec).unwrap_err()),
        EXPLICIT_SEASONAL_FILTER
    );
}

#[test]
fn log_domain_error_precedes_filter_conflict() {
    use x13::{ArimaModel, ArimaOrder, Transform};
    let order = ArimaOrder::new(0, 0, 0).unwrap();
    let explicit = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
    let start = Period::quarterly(2000, 2).unwrap();
    for request in [explicit, RegArimaSpec::automatic_model()] {
        let spec = Spec::x11_regarima(
            request.with_transform(Transform::Log),
            TREND
                .with_seasonal_filter(x11::SeasonalFilter::S3x5)
                .with_mode(x11::Mode::Multiplicative),
        );
        for nonpositive in [0.0, -1.0] {
            let mut input = values(80);
            input[4] = nonpositive;
            let error = x13::adjust(Series::new(start, &input), &spec).unwrap_err();
            assert!(
                matches!(error, Error::NonPositiveValue { index: 4, .. }),
                "{error:?}"
            );
        }
    }
}

#[test]
fn trend_only_keeps_regarima_and_pseudo_additive_restrictions() {
    use x13::regarima::{CalendarBasis, CalendarSpec, LengthEffect, OutlierSpec, OutlierTypes};
    use x13::{ArimaModel, ArimaOrder, Transform, x11};
    let start = Period::quarterly(2000, 2).unwrap();
    let order = ArimaOrder::new(0, 0, 0).unwrap();
    let model = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
    let mut input = values(80);
    input[4] = 0.0;
    let spec = Spec::x11_regarima(
        model.clone().with_transform(Transform::Log),
        TREND
            .with_seasonal_filter(x11::SeasonalFilter::S3x5)
            .with_mode(x11::Mode::PseudoAdditive),
    );
    let error = x13::adjust(Series::new(start, &input), &spec).unwrap_err();
    assert!(
        matches!(error, Error::NonPositiveValue { index: 4, .. }),
        "{error:?}"
    );
    for request in [model, RegArimaSpec::automatic_model()] {
        let unsafe_mode =
            Spec::x11_regarima(request.clone(), TREND.with_mode(x11::Mode::LogAdditive));
        assert_eq!(
            conflict(x13::adjust(Series::new(start, &[100.0; 80]), &unsafe_mode).unwrap_err()),
            UNSAFE_LOG_ADDITIVE
        );
        for forbidden in [
            request.clone().with_outliers(OutlierSpec::default()),
            request.clone().with_calendar(
                CalendarSpec::new().with_basis(CalendarBasis::Length(LengthEffect::LeapYear)),
            ),
        ] {
            let spec = Spec::x11_regarima(forbidden, TREND.with_mode(x11::Mode::PseudoAdditive));
            assert!(
                conflict(x13::adjust(Series::new(start, &[100.0; 80]), &spec).unwrap_err())
                    .starts_with("pseudo-additive X-11 cannot take regression preadjustment")
            );
        }
        let spec = Spec::x11_regarima(
            request,
            TREND
                .with_mode(x11::Mode::PseudoAdditive)
                .with_final_outliers(x11::FinalOutliers::Remove(OutlierTypes::ALL)),
        );
        assert!(
            conflict(x13::adjust(Series::new(start, &[100.0; 80]), &spec).unwrap_err())
                .starts_with("pseudo-additive X-11 final-outlier removal on a series without gaps")
        );
    }
    for options in [
        TREND.with_seasonal_filters(x11::SeasonalFilters::Quarterly(
            [x11::SeasonalFilter::S3x5; 4],
        )),
        x11::Options::new()
            .with_seasonal_filters(x11::SeasonalFilters::Quarterly([
                x11::SeasonalFilter::Stable,
                x11::SeasonalFilter::S3x3,
                x11::SeasonalFilter::Msr,
                x11::SeasonalFilter::X11Default,
            ]))
            .with_decomposition(x11::Decomposition::TrendOnly),
    ] {
        assert_eq!(
            conflict(
                x13::adjust(Series::new(start, &values(80)), &Spec::x11(options)).unwrap_err()
            ),
            EXPLICIT_SEASONAL_FILTER
        );
    }
}

#[test]
fn trend_only_needs_three_years_and_returns_neutral_factors() {
    for (start, count) in [
        (Period::monthly(2000, 4).unwrap(), 36),
        (Period::quarterly(2000, 3).unwrap(), 12),
    ] {
        for mode in [
            x11::Mode::Additive,
            x11::Mode::Multiplicative,
            x11::Mode::PseudoAdditive,
        ] {
            let spec = Spec::x11(TREND.with_mode(mode));
            assert!(x13::adjust(Series::new(start, &values(count - 1)), &spec).is_err());
            let actual = x13::adjust(Series::new(start, &values(count)), &spec).unwrap();
            let neutral: f64 = if mode == x11::Mode::Additive {
                0.0
            } else {
                1.0
            };
            assert!(
                actual
                    .seasonal()
                    .unwrap()
                    .values()
                    .iter()
                    .all(|v| v.to_bits() == neutral.to_bits())
            );
            assert_eq!(actual.seasonally_adjusted().values(), values(count));
        }
    }
}

#[test]
fn shrinkage_and_short_series_options_are_inert_under_trend_only() {
    let start = Period::monthly(2001, 4).unwrap();
    let input = values(120);
    let baseline = x13::adjust(Series::new(start, &input), &Spec::x11(TREND)).unwrap();
    for options in [
        TREND.with_short_series_filters(x11::ShortSeriesFilters::UseRequested),
        TREND.with_seasonal_shrinkage(x11::SeasonalShrinkage::Global),
        TREND.with_seasonal_shrinkage(x11::SeasonalShrinkage::Local),
        TREND
            .with_seasonal_shrinkage(x11::SeasonalShrinkage::Global)
            .with_short_series_filters(x11::ShortSeriesFilters::UseRequested),
        TREND
            .with_seasonal_shrinkage(x11::SeasonalShrinkage::Local)
            .with_short_series_filters(x11::ShortSeriesFilters::UseRequested),
    ] {
        let actual = x13::adjust(Series::new(start, &input), &Spec::x11(options)).unwrap();
        assert!(
            format!("{actual:?}") == format!("{baseline:?}"),
            "{options:?}"
        );
    }
}
