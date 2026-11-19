//! Branches of the X-11 pipelines: a constant series, legacy Easter,
//! pseudo-additive mode, and the order of the checks that refuse a request.

use crate::api::*;
use crate::error::{InvalidSpec, SpecProblem};
use crate::regarima::{
    CalendarBasis, CalendarSelection, Easter, FlowTradingDay, ForecastHorizon, ManualOutlier,
    ModelSelectionResult, TradingDayContrast,
};

fn monthly() -> TimeAxis {
    TimeAxis::monthly(2000, 1).unwrap()
}

fn airline() -> RegArimaSpec {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
}

fn log_airline() -> RegArimaSpec {
    airline().with_transform(Transform::Log)
}

/// A positive seasonal series with a trend and deterministic noise.
fn noisy(count: usize) -> Vec<f64> {
    let mut state = 98_765_u64;
    (0..count)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let angle = 2.0 * std::f64::consts::PI * (index % 12) as f64 / 12.0;
            200.0 + 0.3 * index as f64 + 15.0 * angle.sin() + 4.0 * noise
        })
        .collect()
}

/// Legacy Easter with a 3x15 seasonal filter, which ten years cannot support.
fn easter_options() -> x11::Options {
    x11::Options::default()
        .with_easter_estimation(true)
        .with_seasonal_filter(x11::SeasonalFilter::S3x15)
}

const FILTER_REPLACED: Warning = Warning::SeasonalFilterReplaced {
    requested: x11::SeasonalFilter::S3x15,
    replacement: x11::SeasonalFilter::Stable,
};

fn no_forecasts() -> ForecastSpec {
    ForecastSpec::new(ForecastHorizon::Periods(0), 0).unwrap()
}

fn assert_close(series: Series<'_>, expected: &[f64], label: &str) {
    assert_eq!(series.len(), expected.len(), "{label}");
    for (index, (value, expected)) in series.values().iter().zip(expected).enumerate() {
        assert!(
            (value - expected).abs() <= 1e-12 * expected.abs(),
            "{label} at {index}: {value} is not {expected}"
        );
    }
}

/// `x11ari.f` skips estimation when the series is constant (`Same`).
#[test]
fn constant_series_skips_the_model() {
    let constant = [100.0; 120];
    let spec = Spec::x11_regarima(
        airline(),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let adjustment = adjust(Series::new(monthly().start(), &constant), &spec).unwrap();
    assert_eq!(adjustment.warnings()[0], Warning::ConstantSeries);
    assert!(adjustment.model().is_none());
    assert!(adjustment.forecasts().is_none());
    assert_eq!(adjustment.seasonally_adjusted().values(), constant);
}

#[test]
fn constant_series_with_legacy_easter_skips_the_model() {
    let constant = [100.0; 120];
    for regarima in [
        log_airline(),
        RegArimaSpec::automatic_model().with_transform(Transform::Log),
    ] {
        let automatic = regarima.model() == ModelSelection::Automatic;
        let spec = Spec::x11_regarima(
            regarima,
            easter_options().with_mode(x11::Mode::Multiplicative),
        );
        let adjustment = adjust(Series::new(monthly().start(), &constant), &spec).unwrap();
        assert_eq!(
            adjustment.warnings()[..2],
            [Warning::ConstantSeries, FILTER_REPLACED],
            "{spec:?}"
        );
        assert!(adjustment.model().is_none(), "{spec:?}");
        assert_eq!(
            adjustment.model_selection(),
            automatic.then_some(&ModelSelectionResult::ConstantSeries {
                transform: Transform::Log
            }),
            "{spec:?}"
        );
        // The Easter regression still runs, and finds no effect.
        let easter = adjustment.x11_easter().unwrap();
        assert_eq!(easter.status(), &x11::EasterStatus::Estimated, "{spec:?}");
        assert_close(easter.factors(), &[1.0; 120], "Easter factors");
        assert_close(adjustment.seasonally_adjusted(), &constant, "adjusted");
    }
}

/// `editor.f` warns in this order: constant series, replaced filter,
/// pseudo-additive without forecasts.
#[test]
fn constant_pseudo_additive_series_warns_in_editor_order() {
    let constant = [100.0; 120];
    let filter = x11::Options::default().with_seasonal_filter(x11::SeasonalFilter::S3x15);
    let pseudo = |regarima: RegArimaSpec, options: x11::Options| {
        Spec::x11_regarima(
            regarima
                .with_transform(Transform::Log)
                .with_forecast(no_forecasts()),
            options.with_mode(x11::Mode::PseudoAdditive),
        )
    };
    for spec in [
        pseudo(airline(), filter),
        pseudo(RegArimaSpec::automatic_model(), filter),
        pseudo(
            airline(),
            easter_options().with_holiday_adjustment(x11::HolidayAdjustment::Retain),
        ),
    ] {
        let adjustment = adjust(Series::new(monthly().start(), &constant), &spec).unwrap();
        assert_eq!(
            adjustment.warnings()[..3],
            [
                Warning::ConstantSeries,
                FILTER_REPLACED,
                Warning::PseudoAdditiveWithoutForecasts
            ],
            "{spec:?}"
        );
        assert!(adjustment.model().is_none(), "{spec:?}");
        assert_close(adjustment.seasonally_adjusted(), &constant, "adjusted");
    }

    // The warning follows the request: a constant series gets no forecasts
    // either way.
    let requested = Spec::x11_regarima(
        log_airline(),
        x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
    );
    let adjustment = adjust(Series::new(monthly().start(), &constant), &requested).unwrap();
    assert!(adjustment.forecasts().is_none());
    assert!(
        !adjustment
            .warnings()
            .contains(&Warning::PseudoAdditiveWithoutForecasts)
    );
}

/// `x11pt3.f` divides D11 by the calendar factors only when the model has
/// calendar regressors, so a legacy Easter effect on its own stays in the
/// seasonally adjusted series whatever the holiday adjustment.
#[test]
fn legacy_easter_alone_stays_in_the_adjusted_series() {
    let values = noisy(144);
    let extended =
        log_airline().with_forecast(ForecastSpec::new(ForecastHorizon::OneYear, 12).unwrap());
    for holiday in [
        x11::HolidayAdjustment::Remove,
        x11::HolidayAdjustment::Retain,
    ] {
        let options = easter_options().with_holiday_adjustment(holiday);
        for (spec, has_model) in [
            (Spec::x11(options), false),
            (
                Spec::x11_regarima(
                    extended.clone(),
                    options.with_mode(x11::Mode::Multiplicative),
                ),
                true,
            ),
        ] {
            let adjustment = adjust(Series::new(monthly().start(), &values), &spec).unwrap();
            assert_eq!(adjustment.warnings()[0], FILTER_REPLACED, "{spec:?}");

            let easter = adjustment.x11_easter().unwrap().factors();
            let affected = |index: usize| matches!(index % 12, 2 | 3);
            for (index, &factor) in easter.values().iter().enumerate() {
                assert!(affected(index) || factor == 1.0, "{index}: {factor}");
            }
            assert!(easter.values().iter().any(|&factor| factor != 1.0));

            let seasonal = adjustment.seasonal().unwrap();
            let restored: Vec<f64> = adjustment
                .seasonally_adjusted()
                .values()
                .iter()
                .zip(seasonal.values())
                .map(|(adjusted, seasonal)| adjusted * seasonal)
                .collect();
            assert_close(
                Series::new(monthly().start(), &restored),
                &values,
                "D11 x D10",
            );

            assert_eq!(adjustment.model().is_some(), has_model);
            if has_model {
                assert_eq!(adjustment.forecasts().unwrap().series().len(), 12);
                assert_eq!(adjustment.backcasts().unwrap().series().len(), 12);
            }
        }
    }
}

#[test]
fn direct_pseudo_additive_warns_that_it_has_no_forecasts() {
    let values = noisy(120);
    let spec = Spec::x11(x11::Options::new().with_mode(x11::Mode::PseudoAdditive));
    let adjustment = adjust(Series::new(monthly().start(), &values), &spec).unwrap();
    assert_eq!(
        adjustment.warnings(),
        [Warning::PseudoAdditiveWithoutForecasts]
    );
    // Pseudo-additive D11 is O - C x (S - 1).
    let trend = adjustment.trend_cycle().unwrap();
    let seasonal = adjustment.seasonal().unwrap();
    let expected: Vec<f64> = (0..values.len())
        .map(|index| values[index] - trend.values()[index] * (seasonal.values()[index] - 1.0))
        .collect();
    assert_close(adjustment.seasonally_adjusted(), &expected, "adjusted");
}

/// The legacy Easter checks run in the order frequency, mode, fitted Easter
/// regressor, pseudo-additive holiday retention; the first to fail is reported.
#[test]
fn legacy_easter_reports_its_first_conflict() {
    let quarterly = TimeAxis::quarterly(2000, 1).unwrap();
    let conflict =
        |reason| Error::InvalidSpec(InvalidSpec(SpecProblem::IncompatibleX11Easter { reason }));
    let fitted_easter = CalendarSpec::new().with_easter(Easter::new(8).unwrap());
    let direct = |mode| Spec::x11(easter_options().with_mode(mode));
    let regarima = |mode, regarima| Spec::x11_regarima(regarima, easter_options().with_mode(mode));
    for (values, axis, spec, expected) in [
        (
            noisy(40),
            quarterly,
            direct(x11::Mode::Multiplicative),
            conflict(X11EasterConflict::Frequency {
                actual: crate::Frequency::Quarterly,
            }),
        ),
        (
            noisy(40),
            quarterly,
            direct(x11::Mode::PseudoAdditive),
            conflict(X11EasterConflict::Frequency {
                actual: crate::Frequency::Quarterly,
            }),
        ),
        (
            noisy(120),
            monthly(),
            direct(x11::Mode::Additive),
            conflict(X11EasterConflict::Mode {
                actual: x11::Mode::Additive,
            }),
        ),
        (
            noisy(120),
            monthly(),
            direct(x11::Mode::LogAdditive),
            conflict(X11EasterConflict::Mode {
                actual: x11::Mode::LogAdditive,
            }),
        ),
        (
            noisy(120),
            monthly(),
            direct(x11::Mode::PseudoAdditive),
            conflict(X11EasterConflict::PseudoRequiresHolidayRetention),
        ),
        (
            noisy(40),
            quarterly,
            regarima(x11::Mode::Multiplicative, log_airline()),
            conflict(X11EasterConflict::Frequency {
                actual: crate::Frequency::Quarterly,
            }),
        ),
        (
            noisy(120),
            monthly(),
            Spec::x11_regarima(airline(), easter_options()),
            conflict(X11EasterConflict::Mode {
                actual: x11::Mode::Additive,
            }),
        ),
        (
            noisy(120),
            monthly(),
            regarima(x11::Mode::Additive, airline().with_calendar(fitted_easter)),
            conflict(X11EasterConflict::Mode {
                actual: x11::Mode::Additive,
            }),
        ),
        (
            noisy(120),
            monthly(),
            regarima(
                x11::Mode::Multiplicative,
                log_airline().with_calendar(fitted_easter),
            ),
            conflict(X11EasterConflict::FittedEaster),
        ),
        (
            noisy(120),
            monthly(),
            regarima(
                x11::Mode::PseudoAdditive,
                log_airline().with_calendar(fitted_easter),
            ),
            conflict(X11EasterConflict::FittedEaster),
        ),
        (
            noisy(120),
            monthly(),
            regarima(x11::Mode::PseudoAdditive, log_airline()),
            conflict(X11EasterConflict::PseudoRequiresHolidayRetention),
        ),
    ] {
        assert_eq!(
            adjust(Series::new(axis.start(), &values), &spec).err(),
            Some(expected),
            "{spec:?}"
        );
    }
}

/// With legacy Easter, a calendar AIC test is refused before a manual outlier
/// is found incompatible with the X-11 mode.
#[test]
fn calendar_test_conflict_precedes_the_outlier_mode_conflict() {
    let outlier = [ManualOutlier::new(
        OutlierType::Additive,
        monthly().period_at(30).unwrap(),
    )];
    let aic_test = CalendarSpec::new()
        .with_basis(CalendarBasis::FlowTradingDay(FlowTradingDay::new(
            TradingDayContrast::SixCoefficient,
        )))
        .with_selection(CalendarSelection::AicTest);
    let mode_conflict = Error::InvalidSpec(InvalidSpec(
        SpecProblem::IncompatibleRegressionAdjustmentMode {
            transform: Transform::None,
            mode: x11::Mode::Multiplicative,
        },
    ));
    let calendar_test = Error::from(UnsupportedDetail::Configuration(
        UnsupportedConfiguration::CalendarSelection,
    ));
    for (regarima, expected) in [
        (airline().with_manual_outliers(&outlier), mode_conflict),
        (airline().with_calendar(aic_test), calendar_test.clone()),
        (
            airline()
                .with_calendar(aic_test)
                .with_manual_outliers(&outlier),
            calendar_test,
        ),
    ] {
        let spec = Spec::x11_regarima(
            regarima,
            easter_options().with_mode(x11::Mode::Multiplicative),
        );
        assert_eq!(
            adjust(Series::new(monthly().start(), &noisy(120)), &spec).err(),
            Some(expected),
            "{spec:?}"
        );
    }
}

/// `editor.f` sets `Adjmod` to 2 for additive X-11, the one mode whose
/// `prlkhd.f` Jacobian keeps a value `dpeq` calls zero. SEATS takes the
/// same loop as multiplicative X-11. The targets are Build 62's AICCs.
#[test]
fn only_additive_x11_counts_a_tiny_observation_in_the_likelihood() {
    let mut values: Vec<f64> = (0..96)
        .map(|index| {
            let month = f64::from(index);
            let angle = month * std::f64::consts::PI / 6.0;
            100.0 + 0.3 * month + 6.0 * angle.sin() + f64::from(index * 37 % 11) / 5.5
        })
        .collect();
    values[50] = 1e-25;
    for (spec, expected) in [
        (
            Spec::x11_regarima(
                log_airline(),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
            1246.6857,
        ),
        (
            Spec::x11_regarima(
                log_airline(),
                x11::Options::new().with_mode(x11::Mode::Multiplicative),
            ),
            1361.8149,
        ),
        (Spec::seats(log_airline()), 1361.8149),
    ] {
        let aicc = adjust(Series::new(monthly().start(), &values), &spec)
            .unwrap()
            .model()
            .and_then(|model| model.information_criteria())
            .and_then(|criteria| criteria.aicc())
            .unwrap();
        assert!((aicc - expected).abs() < 5e-5, "{spec:?}: {aicc}");
    }
}
