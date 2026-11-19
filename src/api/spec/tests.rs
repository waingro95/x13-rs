use super::*;
use crate::api::regarima::{Easter, ForecastHorizon, LaborDay, OutlierType, OutlierTypes};
use crate::error::{InvalidSpec, NumericalDetail, SpecProblem};
use crate::regarima::{UserRegressor, UserRegressorType};
use crate::{Series, x11};

mod model_selection;

#[test]
fn airline_is_the_0_1_1_seasonal_model() {
    const AIRLINE: ArimaModel = ArimaModel::airline();
    let unit_ma = ArimaOrder::new(0, 1, 1).unwrap();
    assert_eq!(AIRLINE.regular(), unit_ma);
    assert_eq!(AIRLINE.seasonal(), unit_ma);
    assert_eq!(AIRLINE, ArimaModel::new(unit_ma, unit_ma).unwrap());
    assert_eq!(AIRLINE.to_string(), "(0 1 1)(0 1 1)");
}

#[test]
fn as_many_regressors_as_observations_is_rejected() {
    // 36 generic user regressors fit 36 observations exactly, leaving no
    // residual degrees of freedom. The 1e6 level keeps the rounding-error
    // variance above Build 62's reporting threshold, so the covariance is
    // reached.
    let axis = TimeAxis::monthly(2023, 9).unwrap();
    let values = (0..36)
        .map(|i| 1.0e6 + 1.0e3 * ((i * 37 % 101) as f64) + 1.0e4 * (i % 12) as f64)
        .collect::<Vec<_>>();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut draw = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    };
    let regressors = (0..36)
        .map(|_| {
            let column = (0..156).map(|_| draw()).collect::<Vec<_>>();
            UserRegressor::new(UserRegressorType::User, Series::new(axis.start(), &column)).unwrap()
        })
        .collect::<Vec<_>>();
    let order = ArimaOrder::new(0, 0, 0).unwrap();
    let request =
        RegArimaSpec::new(ArimaModel::new(order, order).unwrap()).with_user_regressors(&regressors);
    let result = crate::adjust(Series::new(axis.start(), &values), &Spec::seats(request));
    assert!(matches!(
        result,
        Err(Error::RegressionColumnsExceedObservations {
            columns: 36,
            effective_observations: 36,
        })
    ));
}

#[test]
fn seats_accepts_manual_outliers_with_an_ar2_model() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let request = RegArimaSpec::new(
        ArimaModel::new(
            ArimaOrder::new(2, 1, 0).unwrap(),
            ArimaOrder::new(0, 1, 1).unwrap(),
        )
        .unwrap(),
    )
    .with_manual_outliers(&[
        ManualOutlier::new(OutlierType::Additive, axis.period_at(100).unwrap()),
        ManualOutlier::new(OutlierType::LevelShift, axis.period_at(200).unwrap()),
        ManualOutlier::new(OutlierType::TemporaryChange, axis.period_at(350).unwrap()),
    ]);
    let values = (0..360)
        .map(|i| {
            100.0
                + i as f64 * 0.2
                + 12.0 * (i as f64 * std::f64::consts::TAU / 12.0).sin()
                + ((i * 37 % 101) as f64 - 50.0) * 0.1
        })
        .collect::<Vec<_>>();
    let adjusted =
        crate::adjust(Series::new(axis.start(), &values), &Spec::seats(request)).unwrap();
    assert_eq!(adjusted.outliers().len(), 3);
    assert_eq!(adjusted.seasonally_adjusted().len(), values.len());
}

#[test]
fn one_temporary_change_rate_serves_every_tc_column() {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let base = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
    let rate = TemporaryChangeRate::new(0.8).unwrap();
    let search = OutlierSpec::default().with_types(OutlierTypes::ALL);
    // gtinpt.f's default, 0.7**(12/Sp), when no spec sets `tcrate`.
    let unset = base.resolve_explicit().unwrap();
    assert_eq!(
        unset.resolved_temporary_change_rate(Frequency::Monthly),
        0.7
    );
    assert_eq!(
        unset.resolved_temporary_change_rate(Frequency::Quarterly),
        0.342_999_999_999_999_9
    );
    // One rate drives the manual TCs and the automatic TC candidates,
    // and the outlier spec passes through unchanged.
    let request = base.with_temporary_change_rate(rate);
    assert_eq!(request.temporary_change_rate(), Some(rate));
    let resolved = request.with_outliers(search).resolve_explicit().unwrap();
    assert_eq!(resolved.outliers(), Some(search));
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        assert_eq!(resolved.resolved_temporary_change_rate(frequency), 0.8);
    }
}

#[test]
fn manual_temporary_change_uses_the_requested_rate_in_x11_and_seats() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let tc = ManualOutlier::new(OutlierType::TemporaryChange, axis.period_at(300).unwrap());
    let base = RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
        .with_transform(Transform::Log)
        .with_manual_outliers(&[tc]);
    let configured = base
        .clone()
        .with_temporary_change_rate(TemporaryChangeRate::new(0.8).unwrap());
    let values = (0..360)
        .map(|i| {
            let effect = if i >= 300 { 0.8_f64.powi(i - 300) } else { 0.0 };
            100.0
                + f64::from(i) * 0.2
                + 12.0 * (f64::from(i) * std::f64::consts::TAU / 12.0).sin()
                + f64::from(i * 37 % 101 - 50) * 0.1
                + 15.0 * effect
        })
        .collect::<Vec<_>>();
    let estimate = |spec: Spec| {
        let adjusted = crate::adjust(Series::new(axis.start(), &values), &spec).unwrap();
        assert_eq!(adjusted.outliers().len(), 1);
        adjusted.outliers()[0].estimate()
    };
    // X-11 and SEATS fit the same regARIMA, so their estimates at the same
    // rate agree exactly.
    let x11 = estimate(Spec::x11_regarima(configured.clone(), x11::Options::new()));
    assert_eq!(estimate(Spec::seats(configured)), x11);
    assert_ne!(
        estimate(Spec::x11_regarima(base.clone(), x11::Options::new())),
        x11
    );
    assert_ne!(estimate(Spec::seats(base)), x11);
}

#[test]
fn manual_outliers_are_validated_before_a_constant_series_skips_the_model() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let base = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
    let options = crate::x11::Options::new().with_easter_estimation(true);
    let ao = ManualOutlier::new(OutlierType::Additive, axis.period_at(24).unwrap());
    let outside = ManualOutlier::new(OutlierType::TemporaryChange, axis.period_at(120).unwrap());
    let first_ls = ManualOutlier::new(OutlierType::LevelShift, axis.start());
    let last_ls = ManualOutlier::new(OutlierType::LevelShift, axis.period_at(119).unwrap());
    for (outliers, invalid) in [
        (vec![outside], outside),
        (vec![ao, ao], ao),
        (vec![first_ls], first_ls),
        (vec![last_ls], last_ls),
    ] {
        let spec = Spec::x11_regarima(
            base.clone()
                .with_transform(Transform::Log)
                .with_manual_outliers(&outliers),
            options.with_mode(x11::Mode::Multiplicative),
        );
        assert_eq!(
            crate::adjust(Series::new(axis.start(), &[100.0; 120]), &spec).unwrap_err(),
            Error::InvalidManualOutlier {
                kind: invalid.kind(),
                period: invalid.period()
            }
        );
    }
}

#[test]
fn spans_of_another_frequency_are_rejected() {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let base = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
    let quarterly = TimeAxis::quarterly(2000, 1).unwrap();
    let march = crate::Period::monthly(2001, 3).unwrap();
    let span = crate::PeriodSpan::new(quarterly.start(), march);
    for request in [
        base.clone().with_model_span(span),
        base.with_outliers(OutlierSpec::default().with_span(span)),
    ] {
        assert_eq!(
            crate::adjust(
                Series::new(quarterly.start(), &[100.0; 40]),
                &Spec::x11_regarima(request, x11::Options::new().with_mode(x11::Mode::Additive))
            )
            .unwrap_err(),
            Error::PeriodFrequencyMismatch {
                period: march,
                expected: Frequency::Quarterly,
            }
        );
    }
}

/// A positive monthly series with a trend, a seasonal pattern and noise.
fn seasonal_values() -> Vec<f64> {
    (0..120_i32)
        .map(|i| {
            100.0
                + f64::from(i) * 0.2
                + 12.0 * (f64::from(i) * std::f64::consts::TAU / 12.0).sin()
                + f64::from(i * 37 % 101 - 50) * 0.1
        })
        .collect()
}

/// The `Debug` text of the adjustment `Spec::x11(options)` gives.
fn x11_adjustment(options: x11::Options) -> String {
    let start = crate::Period::monthly(2000, 1).unwrap();
    let adjustment =
        crate::adjust(Series::new(start, &seasonal_values()), &Spec::x11(options)).unwrap();
    format!("{adjustment:?}")
}

#[test]
fn a_spec_returns_what_it_was_built_from() {
    use crate::{Method, x11};
    let options = x11::Options::new()
        .with_mode(x11::Mode::LogAdditive)
        .with_holiday_adjustment(x11::HolidayAdjustment::Retain);
    let regarima = RegArimaSpec::automatic_model();

    let x11 = Spec::x11(options);
    assert_eq!(x11.method(), Method::X11);
    assert_eq!(x11.x11_options(), Some(options));
    assert_eq!(x11.regarima(), None);

    let x11_regarima = Spec::x11_regarima(regarima.clone(), options);
    assert_eq!(x11_regarima.method(), Method::X11);
    assert_eq!(x11_regarima.x11_options(), Some(options));
    assert_eq!(x11_regarima.regarima(), Some(&regarima));

    let seats = Spec::seats(regarima.clone());
    assert_eq!(seats.method(), Method::Seats);
    assert_eq!(seats.x11_options(), None);
    assert_eq!(seats.regarima(), Some(&regarima));
}

#[test]
fn x11_option_defaults_follow_build_62() {
    use crate::regarima::X11ModeSelection;
    use crate::x11;
    let options = x11::Options::new();
    assert_eq!(options, x11::Options::default());
    // editor.f: no seasonalma gives Lterm=6 (MSR selection).
    assert_eq!(options.seasonal_filter(), x11::SeasonalFilter::Msr);
    assert!(!options.has_explicit_seasonal_filter());
    for explicit in [
        options.with_seasonal_filter(x11::SeasonalFilter::Msr),
        options.with_seasonal_filters(x11::SeasonalFilters::Monthly(
            [x11::SeasonalFilter::S3x5; 12],
        )),
    ] {
        assert!(explicit.has_explicit_seasonal_filter());
    }
    assert_eq!(options.mode(), None);
    assert_eq!(options.mode_selection(), X11ModeSelection::FromTransform);
    let additive = options.with_mode(x11::Mode::Additive);
    assert_eq!(additive.mode(), Some(x11::Mode::Additive));
    assert_eq!(
        additive.mode_selection(),
        X11ModeSelection::Explicit(x11::Mode::Additive)
    );
    assert_eq!(options.holiday_adjustment(), x11::HolidayAdjustment::Remove);
    assert_eq!(options.final_outliers(), x11::FinalOutliers::Retain);
    assert_eq!(
        options.temporary_change_allocation(),
        x11::TemporaryChangeAllocation::Irregular
    );
}

#[test]
fn x11_without_a_mode_is_multiplicative() {
    // gtinpt.f: no transform spec gives Muladd=0 (multiplicative).
    let start = crate::Period::monthly(2000, 1).unwrap();
    let spec = Spec::x11(x11::Options::new());
    let adjustment = crate::adjust(Series::new(start, &seasonal_values()), &spec).unwrap();
    assert_eq!(
        adjustment.diagnostics().x11().unwrap().mode(),
        x11::Mode::Multiplicative
    );
    assert_eq!(
        x11_adjustment(x11::Options::new()),
        x11_adjustment(x11::Options::new().with_mode(x11::Mode::Multiplicative))
    );
    assert_ne!(
        x11_adjustment(x11::Options::new()),
        x11_adjustment(x11::Options::new().with_mode(x11::Mode::Additive))
    );
}

#[test]
fn x11_without_a_model_ignores_the_outlier_settings() {
    use crate::regarima::OutlierTypes;
    use crate::x11;
    let default = x11_adjustment(x11::Options::new());
    for options in [
        x11::Options::new().with_final_outliers(x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE)),
        x11::Options::new().with_temporary_change_allocation(x11::TemporaryChangeAllocation::Trend),
    ] {
        assert_eq!(x11_adjustment(options), default, "{options:?}");
    }
}

#[test]
fn outlier_mode_conflict_is_reported_for_a_constant_series_with_legacy_easter() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let request =
        RegArimaSpec::new(ArimaModel::new(order, order).unwrap()).with_manual_outliers(&[
            ManualOutlier::new(OutlierType::Additive, axis.period_at(24).unwrap()),
        ]);
    let spec = Spec::x11_regarima(
        request,
        crate::x11::Options::new()
            .with_mode(x11::Mode::Multiplicative)
            .with_easter_estimation(true),
    );
    assert_eq!(
        crate::adjust(Series::new(axis.start(), &[100.0; 120]), &spec).unwrap_err(),
        Error::InvalidSpec(InvalidSpec(
            SpecProblem::IncompatibleRegressionAdjustmentMode {
                transform: Transform::None,
                mode: x11::Mode::Multiplicative
            }
        ))
    );
}

/// editor.f adds a missing-value regressor per gap while it reads
/// the input, and adrgef stops at the 81st, before X-11 reaches its
/// legacy-Easter prepass. The pre-1901 start would fail that prepass's
/// date-table lookup, so only the capacity check firing first passes.
#[test]
fn missing_value_capacity_precedes_the_legacy_easter_prepass() {
    let axis = TimeAxis::monthly(1890, 1).unwrap();
    let values: Vec<_> = (0..420)
        .map(|i| {
            if i % 5 == 2 {
                f64::NAN
            } else {
                100.0 + f64::from(i % 13)
            }
        })
        .collect();
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let options = crate::x11::Options::new().with_easter_estimation(true);
    let explicit =
        RegArimaSpec::new(ArimaModel::new(order, order).unwrap()).with_transform(Transform::Log);
    let automatic = RegArimaSpec::automatic_model().with_transform(Transform::Log);
    for regarima in [explicit, automatic] {
        let spec = Spec::x11_regarima(regarima, options.with_mode(x11::Mode::Multiplicative));
        assert_eq!(
            crate::adjust(Series::new(axis.start(), &values), &spec).unwrap_err(),
            Error::TooManyRegressionVariables {
                actual: 84,
                maximum: 80
            }
        );
    }
}

/// `CalendarSpec::validate` rejects holiday years outside 1901–2100 before
/// the legacy-Easter prepass runs and before a constant series skips the
/// model. Inside that range a constant series gets the prepass's error or
/// succeeds.
#[test]
fn holiday_years_are_checked_before_a_constant_series_skips_the_model() {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    let easter = CalendarSpec::new().with_easter(Easter::new(8).unwrap());
    let labor_day = CalendarSpec::new().with_labor_day(LaborDay::new(8).unwrap());
    let year_error = |feature, year| Error::CalendarYearOutOfRange {
        feature,
        year,
        minimum: 1901,
        maximum: Some(2100),
    };
    let prepass_error = Error::from(NumericalDetail::X11Easter {
        stage: crate::x11::X11EasterStage::CalendarLookup,
        period: Some(TimeAxis::monthly(2101, 1).unwrap().start()),
        reason: crate::x11::X11EasterUnavailableReason::OutsideDateTable,
    });
    let none = false;
    let estimate = true;
    for (calendar, start, forecasts, x11_easter, expected) in [
        (
            easter,
            1890,
            0,
            none,
            Err(year_error(CalendarFeature::Easter, 1890)),
        ),
        (
            easter,
            2081,
            12,
            none,
            Err(year_error(CalendarFeature::Easter, 2101)),
        ),
        (easter, 2081, 0, none, Ok(())),
        (
            labor_day,
            1890,
            0,
            estimate,
            Err(year_error(CalendarFeature::LaborDay, 1890)),
        ),
        (
            labor_day,
            2081,
            12,
            estimate,
            Err(year_error(CalendarFeature::LaborDay, 2101)),
        ),
        (labor_day, 2081, 0, estimate, Err(prepass_error.clone())),
        (labor_day, 2081, 0, none, Ok(())),
    ] {
        let explicit = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
        for regarima in [explicit, RegArimaSpec::automatic_model()] {
            let regarima = regarima
                .with_transform(Transform::Log)
                .with_calendar(calendar)
                .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(forecasts), 0).unwrap());
            let spec = Spec::x11_regarima(
                regarima,
                crate::x11::Options::new()
                    .with_mode(x11::Mode::Multiplicative)
                    .with_easter_estimation(x11_easter),
            );
            let axis = TimeAxis::monthly(start, 1).unwrap();
            assert_eq!(
                crate::adjust(Series::new(axis.start(), &[100.0; 240]), &spec).map(drop),
                expected
            );
        }
    }
}
