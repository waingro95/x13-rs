use super::{insert_extensions, legacy_calendar, restore_final_components, restore_interval};
use crate::engine::FinalX11Options;
use crate::regarima::{OutlierType, OutlierTypes, RegressionVariable, TradingDayVariable};
use crate::regression::effects::{EvaluatedEffects, evaluate_regression_effects};
use crate::regression::model::{RegressionColumn, RegressionModel};
use crate::time::TimeAxis;
use crate::{Transform, Warning, x11};

mod final_outliers;
mod temporary_change;

const TRADING_DAY: RegressionVariable =
    RegressionVariable::TradingDay(TradingDayVariable::FlowWeekdayWeekend);

const FINAL_OUTLIER_CHOICES: [x11::FinalOutliers; 8] = [
    x11::FinalOutliers::Retain,
    x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE),
    x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT),
    x11::FinalOutliers::Remove(OutlierTypes::TEMPORARY_CHANGE),
    x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT),
    x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_TEMPORARY_CHANGE),
    x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT_AND_TEMPORARY_CHANGE),
    x11::FinalOutliers::Remove(OutlierTypes::ALL),
];

/// An additive outlier, a level shift and a temporary change.
fn outliers() -> [RegressionVariable; 3] {
    let period = TimeAxis::monthly(2000, 1).unwrap().start();
    [
        OutlierType::Additive,
        OutlierType::LevelShift,
        OutlierType::TemporaryChange,
    ]
    .map(|kind| RegressionVariable::Outlier { kind, period })
}

/// The effects of regressors that are 1 at the one observed period, the
/// second of three, and 0 at the others.
fn effects<const N: usize>(
    regressors: [RegressionVariable; N],
    coefficients: [f64; N],
    transform: Transform,
) -> EvaluatedEffects {
    let mut model = RegressionModel::new();
    for variable in regressors {
        model
            .push(RegressionColumn::with_values(variable, vec![0.0, 1.0, 0.0]))
            .unwrap();
    }
    evaluate_regression_effects(&model, &coefficients, transform, 1..2).unwrap()
}

/// Additive-outlier, level-shift and temporary-change effects, in that order.
fn outlier_effects(transform: Transform, coefficients: [f64; 3]) -> EvaluatedEffects {
    effects(outliers(), coefficients, transform)
}

/// Final restoration does not read the seasonality diagnostics.
fn unavailable_seasonality() -> crate::diagnostics::SeasonalityTests {
    use crate::diagnostics::SeasonalityTest as Stage;
    use crate::diagnostics::{
        Unavailable, UnavailableDetail, X11DiagnosticUnavailableReason as Reason,
    };
    let missing = |test| {
        Unavailable(UnavailableDetail::SeasonalityTest {
            test,
            reason: Reason::InsufficientSample,
            sample: None,
            differencing_lag: None,
        })
    };
    crate::diagnostics::SeasonalityTests {
        preliminary_stable: Err(missing(Stage::PreliminaryStable)),
        final_stable: Err(missing(Stage::FinalStable)),
        moving: Err(missing(Stage::Moving)),
        kruskal_wallis: Err(missing(Stage::KruskalWallis)),
        combined: Err(missing(Stage::Combined)),
        residual_full: Err(missing(Stage::ResidualFull)),
        residual_recent_three_years: Err(missing(Stage::ResidualRecentThreeYears)),
        modified_residual_full: Err(missing(Stage::ModifiedResidualFull)),
        modified_residual_recent_three_years: Err(missing(Stage::ModifiedResidualRecentThreeYears)),
    }
}

/// X-11 output for one observed period. Final restoration reads the mode,
/// seasonal and trend, and replaces the adjusted series and irregular.
fn components(mode: x11::Mode, seasonal: f64, trend: f64) -> crate::x11::Components {
    crate::x11::Components {
        spectral_workspace: crate::x11::spectral::SpectralWorkspace::empty(),
        decomposition: crate::x11::Decomposition::SeasonalAdjustment,
        quality_workspace: crate::x11::quality::QualityWorkspace::empty(),
        seasonality: unavailable_seasonality(),
        mode,
        seasonal: vec![seasonal],
        seasonally_adjusted: vec![0.0],
        trend_cycle: vec![trend],
        irregular: vec![0.0],
        final_henderson_terms: 0,
        msr_selection: None,
        calendar_sigma_selection: None,
        warnings: Vec::new(),
        intermediate_extreme_weights: vec![1.0],
        conditioning_scales: Default::default(),
    }
}

/// Final options that remove the holiday effect, the default.
fn final_options(
    outliers: x11::FinalOutliers,
    allocation: x11::TemporaryChangeAllocation,
) -> FinalX11Options {
    FinalX11Options::new(outliers, allocation, x11::HolidayAdjustment::Remove)
}

/// Final restoration of the one observed period, without legacy Easter.
fn restore(
    components: &mut crate::x11::Components,
    original: f64,
    effects: &EvaluatedEffects,
    transform: Transform,
    final_options: FinalX11Options,
) {
    restore_final_components(
        components,
        &[original],
        effects,
        1..2,
        transform,
        final_options,
        None,
    )
    .unwrap();
}

/// `Faccal` is the Easter factor when the holiday is removed. `keepholiday`
/// divides it back out, leaving one, or NaN where the factor is zero or not
/// finite.
#[test]
fn retaining_the_holiday_cancels_the_legacy_easter_factor() {
    let factors = [0.5, 2.0, 1.25, f64::MIN_POSITIVE];
    assert_eq!(
        legacy_calendar(&factors, x11::HolidayAdjustment::Remove),
        factors
    );
    assert_eq!(
        legacy_calendar(&factors, x11::HolidayAdjustment::Retain),
        [1.0; 4]
    );
    let undefined = [0.0, -0.0, f64::NAN, f64::INFINITY];
    let retained = legacy_calendar(&undefined, x11::HolidayAdjustment::Retain);
    assert!(retained.iter().all(|value| value.is_nan()), "{retained:?}");
}

#[test]
fn legacy_easter_factor_scales_the_forecast_interval() {
    let mut interval = [0.01, 0.07, 0.19];
    let [point, lower, upper] = &mut interval;
    restore_interval(point, lower, upper, 0.3).unwrap();
    assert_eq!(interval, [0.01 * 0.3, 0.07 * 0.3, 0.19 * 0.3]);

    let mut interval = [f64::MAX; 3];
    let [point, lower, upper] = &mut interval;
    assert!(matches!(
        restore_interval(point, lower, upper, 2.0),
        Err(error) if error == crate::Error::numerical(crate::error::NumericalOperation::Extension)
    ));
}

#[test]
fn additive_and_log_multiplicative_extensions_keep_both_sides() {
    let observed = [10.0, 11.0, 12.0];
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    for (transform, mode) in [
        (Transform::None, x11::Mode::Additive),
        (Transform::Log, x11::Mode::Multiplicative),
    ] {
        let inserted =
            insert_extensions(&observed, axis, transform, mode, &[-2.0], &[-1.0]).unwrap();
        assert_eq!(inserted.values, [-1.0, 10.0, 11.0, 12.0, -2.0]);
        assert_eq!(inserted.axis, axis.shift(-1).unwrap());
        assert_eq!(inserted.observed, 1..4);
        assert!(inserted.warning.is_none());
    }
}

/// `extend.f` checks the forecasts before the backcasts, and one bad value
/// drops both.
#[test]
fn multiplicative_rejects_nonpositive_extensions_forecasts_first() {
    let observed = [10.0, 11.0, 12.0];
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    for (forecasts, backcasts, warning) in [
        (
            &[13.0, 0.0][..],
            &[-1.0, 9.0][..],
            Warning::ForecastsNotUsed,
        ),
        (&[13.0], &[-1.0], Warning::BackcastsNotUsed),
    ] {
        let inserted = insert_extensions(
            &observed,
            axis,
            Transform::None,
            x11::Mode::Multiplicative,
            forecasts,
            backcasts,
        )
        .unwrap();
        assert_eq!(inserted.values, observed);
        assert_eq!(inserted.observed, 0..observed.len());
        assert_eq!(inserted.warning, Some(warning));
    }
}

#[test]
fn log_additive_rejects_nonpositive_extensions_forecasts_first() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    for (forecast, backcast, warning) in [
        (-1.0, -2.0, Some(Warning::ForecastsNotUsed)),
        (1.0, 0.0, Some(Warning::BackcastsNotUsed)),
        (1.0, 2.0, None),
    ] {
        let inserted = insert_extensions(
            &[10.0; 12],
            axis,
            Transform::None,
            x11::Mode::LogAdditive,
            &[forecast],
            &[backcast],
        )
        .unwrap();
        if warning.is_some() {
            assert_eq!(inserted.axis, axis);
            assert_eq!(inserted.observed, 0..12);
            assert_eq!(inserted.values, [10.0; 12]);
            assert_eq!(inserted.warning, warning);
        } else {
            assert!(inserted.warning.is_none());
            assert_eq!(inserted.axis, axis.shift(-1).unwrap());
            assert_eq!(inserted.observed, 1..13);
            assert_eq!(inserted.values.len(), 14);
        }
    }
}

/// Zero and negative values are both unusable, and the forecasts are checked
/// before the backcasts.
#[test]
fn pseudo_additive_rejects_zero_and_negative_extensions_forecasts_first() {
    use Warning::{BackcastsNotUsed, ForecastsNotUsed};
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let insert = |transform, forecasts: &[f64], backcasts: &[f64]| {
        insert_extensions(
            &[100.0; 12],
            axis,
            transform,
            x11::Mode::PseudoAdditive,
            forecasts,
            backcasts,
        )
        .unwrap()
    };
    for (forecasts, backcasts, warning) in [
        ([0.0, -1.0], [-2.0, 1.0], ForecastsNotUsed),
        ([-1.0, 0.0], [0.0, 1.0], ForecastsNotUsed),
        ([1.0, 2.0], [0.0, -2.0], BackcastsNotUsed),
        ([1.0, 2.0], [-2.0, 0.0], BackcastsNotUsed),
        ([1.0, 2.0], [-2.0, -0.0], BackcastsNotUsed),
    ] {
        let rejected = insert(Transform::None, &forecasts, &backcasts);
        assert_eq!(rejected.values, [100.0; 12]);
        assert_eq!(rejected.axis, axis);
        assert_eq!(rejected.observed, 0..12);
        assert_eq!(rejected.warning, Some(warning));
    }

    let positive = insert(Transform::None, &[110.0, 120.0], &[80.0, 90.0]);
    assert!(positive.warning.is_none());
    assert_eq!(positive.values[..2], [80.0, 90.0]);
    assert_eq!(positive.values[14..], [110.0, 120.0]);
    assert_eq!(positive.observed, 2..14);
    assert_eq!(positive.axis, axis.shift(-2).unwrap());

    // Extensions of a log model are exponentials, so they are not checked.
    let log = insert(Transform::Log, &[0.0], &[-0.0]);
    assert!(log.warning.is_none());
    assert_eq!(log.observed, 1..13);
}

/// The outlier explains the series exactly, which leaves no residual variance
/// to compute a standard error from.
#[test]
fn outlier_without_a_standard_error_keeps_its_estimate() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let mut values = vec![100.0; 13];
    values[6] = 150.0;
    let mut ao = vec![0.0; 13];
    ao[6] = 1.0;
    let mut regression = RegressionModel::new();
    regression
        .insert_outlier(
            OutlierType::Additive,
            axis.period_at(6).unwrap(),
            crate::Frequency::Quarterly,
            ao,
            0.7,
        )
        .unwrap();
    let fit = crate::arima::estimate(
        &values,
        (&values).into(),
        axis,
        crate::RegArimaSpec::new(crate::ArimaModel::airline())
            .resolve_explicit()
            .unwrap(),
        regression,
        crate::arima::ArmaStart::Default,
    )
    .unwrap();
    let outliers = crate::engine::outlier_metadata::detected_outliers(&fit);
    assert_eq!(outliers.len(), 1);
    assert_eq!(outliers[0].estimate(), 50.0);
    assert_eq!(outliers[0].standard_error(), None);
    assert_eq!(outliers[0].t_value(), None);
}

/// `adjreg.f` subtracts the trading-day effect, then the level shift:
/// `(1e16 - 1e16) - -1` is 1. Summing the effects first would lose the level
/// shift.
#[test]
fn regression_effects_leave_the_response_one_group_at_a_time() {
    let [_, level_shift, _] = outliers();
    let effects = effects([TRADING_DAY, level_shift], [1.0e16, -1.0], Transform::None);
    let mut response = [0.0, 1.0e16, 0.0];
    effects
        .remove_from_transformed_response(&mut response)
        .unwrap();
    assert_eq!(response, [0.0, 1.0, 0.0]);
}

/// `x11pt3.f` forms D11 from the original series less the seasonal and the
/// calendar: 100 - 10 - 1. The level shift goes to the trend, and the
/// irregular, D11 less the trend and every outlier, gets the additive outlier
/// and temporary change back.
#[test]
fn final_restoration_removes_the_calendar_and_allocates_the_outliers() {
    let [ao, ls, tc] = outliers();
    let effects = effects(
        [TRADING_DAY, ao, ls, tc],
        [1.0, 3.0, 2.0, 4.0],
        Transform::None,
    );
    let mut result = components(x11::Mode::Additive, 10.0, 70.0);
    restore(
        &mut result,
        100.0,
        &effects,
        Transform::None,
        final_options(
            x11::FinalOutliers::Retain,
            x11::TemporaryChangeAllocation::Irregular,
        ),
    );
    assert_eq!(result.seasonally_adjusted, [89.0]);
    assert_eq!(result.trend_cycle, [72.0]);
    assert_eq!(result.irregular, [17.0]);
}

/// Retaining the holiday effect puts it back in the adjusted series and the
/// irregular; the seasonal and the trend do not change.
#[test]
fn retaining_the_holiday_changes_only_the_adjusted_series_and_irregular() {
    let easter = crate::regarima::HolidayVariable::Easter(crate::regarima::Easter::new(1).unwrap());
    let [ao, ls, tc] = outliers();
    let regressors = [TRADING_DAY, RegressionVariable::Holiday(easter), ao, ls, tc];
    for (transform, mode, seasonal, trend, coefficients) in [
        (
            Transform::None,
            x11::Mode::Additive,
            10.0,
            70.0,
            [2.0, 3.0, 1.0, 2.0, 3.0],
        ),
        (
            Transform::Log,
            x11::Mode::Multiplicative,
            2.0,
            40.0,
            [0.02, 0.03, 0.01, 0.02, 0.03],
        ),
    ] {
        let effects = effects(regressors, coefficients, transform);
        for outliers in FINAL_OUTLIER_CHOICES {
            for allocation in [
                x11::TemporaryChangeAllocation::Irregular,
                x11::TemporaryChangeAllocation::Trend,
            ] {
                let run = |holiday| {
                    let mut result = components(mode, seasonal, trend);
                    let options = FinalX11Options::new(outliers, allocation, holiday);
                    restore(&mut result, 100.0, &effects, transform, options);
                    result
                };
                let remove = run(x11::HolidayAdjustment::Remove);
                let retain = run(x11::HolidayAdjustment::Retain);
                let case = format!("{transform:?}/{outliers:?}/{allocation:?}");
                assert_eq!(remove.seasonal, retain.seasonal, "{case}");
                assert_eq!(remove.trend_cycle, retain.trend_cycle, "{case}");
                assert_ne!(
                    remove.seasonally_adjusted, retain.seasonally_adjusted,
                    "{case}"
                );
                assert_ne!(remove.irregular, retain.irregular, "{case}");
                if transform == Transform::None {
                    let holiday = 3.0;
                    assert_eq!(
                        retain.seasonally_adjusted[0] - remove.seasonally_adjusted[0],
                        holiday,
                        "{case}"
                    );
                    assert_eq!(retain.irregular[0] - remove.irregular[0], holiday, "{case}");
                }
            }
        }
    }
}
