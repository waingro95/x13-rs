//! The fitted regression effects a SEATS run restores, from a regARIMA fit.
use crate::arima::EstimatedRegArima;
use crate::regression::model::{RegressionModel, RegressorKind};
use crate::seats::Stage;
use crate::seats::airline::SeatsTransform;
use crate::seats::effects::{FittedEffectsInput, SeatsEffectSeries as Series, checked};
use crate::time::TimeAxis;
use crate::{Error, Transform};

/// The fitted effects a SEATS run restores, with the calendar effects the
/// adjustment reports.
pub(super) struct PreparedSeatsEffects {
    pub(super) input: FittedEffectsInput,
    pub(super) calendar_effects: Option<crate::regarima::CalendarEffects>,
}

/// The fitted effects of `fit` over the series `original`.
///
/// Fails when SEATS does not support the fit's calendar, or when an effect
/// or a value made from it is unusable.
pub(super) fn prepare_with_transformed_input(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    transformed_input: &super::pipeline::SeatsTransformedInput,
) -> Result<PreparedSeatsEffects, Error> {
    check_calendar(fit)?;
    debug_assert!(fit_matches_spec(original, axis, fit, transformed_input));
    let transform = SeatsTransform::from(fit.spec.transform());
    if fit.regressor_kinds.is_empty() && fit.regression_model.fixed_length().is_empty() {
        return neutral_effects(original, axis, fit, transform);
    }
    fitted_effects(original, axis, fit, transform, transformed_input)
}

/// Refuses a fit whose calendar SEATS does not support.
fn check_calendar(fit: &EstimatedRegArima) -> Result<(), Error> {
    let reason = fit
        .spec
        .calendar()
        .and_then(crate::api::unsupported_calendar);
    if let Some(reason) = reason {
        return Err(Error::from(crate::error::UnsupportedDetail::SeatsOption(
            crate::api::SeatsUnsupportedOption::Calendar(reason),
        )));
    }
    Ok(())
}

/// Whether `fit` is the fit its spec generates over `original`.
///
/// The fit must be on `axis` and carry the expected regressors, each once and
/// in the fit's own order, with finite estimates of the right shapes. Its
/// columns, fixed-length prior, transformed series and temporary-change rates
/// must carry the bits the spec gives them.
fn fit_matches_spec(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    transformed_input: &super::pipeline::SeatsTransformedInput,
) -> bool {
    use super::pipeline::SeatsTransformedInput;

    let n = original.len();
    if axis != fit.axis {
        return false;
    }
    let Some(expected) = expected_model(original, axis, fit) else {
        return false;
    };

    // automdl can add the constant after the calendar columns, so the fit
    // keeps its own order rather than `from_spec`'s.
    let kinds = &fit.regressor_kinds;
    if !fit.regression_model.kinds().eq(kinds.iter().copied())
        || kinds.len() != expected.column_count()
        || expected
            .kinds()
            .any(|kind| kinds.iter().filter(|&&actual| actual == kind).count() != 1)
    {
        return false;
    }

    let k = kinds.len();
    let parameters = fit.parameters.len();
    let estimates = [
        (Some(fit.regression_coefficients.as_slice()), k),
        (fit.regression_standard_errors.as_deref(), k),
        (fit.regression_covariance.as_deref(), k * (k + 1) / 2),
        (fit.parameter_standard_errors.as_deref(), parameters),
        (fit.arma_covariance.matrix(), parameters * parameters),
    ];
    if !estimates.iter().all(|&(values, count)| {
        values.is_none_or(|values| values.len() == count && values.iter().all(|v| v.is_finite()))
    }) {
        return false;
    }

    if fit.transformed.len() != n {
        return false;
    }
    if let SeatsTransformedInput::SelectedTerminal { values, .. } = transformed_input
        && values.len() != n
    {
        return false;
    }

    for (_, column) in fit.regression_model.columns() {
        if column.len() != n || !column.iter().all(|value| value.is_finite()) {
            return false;
        }
    }
    for (kind, column) in fit.regression_model.columns() {
        let Some((_, expected)) = expected
            .columns()
            .find(|&(expected_kind, _)| expected_kind == kind)
        else {
            return false;
        };
        if !same_bits(column, expected) {
            return false;
        }
    }

    let prior = fit.regression_model.fixed_length();
    if prior.len() != expected.fixed_length().len() {
        return false;
    }
    if !prior.is_empty()
        && (prior.len() != n
            || !prior
                .iter()
                .zip(expected.fixed_length())
                .all(|(&actual, &expected)| {
                    actual.is_finite() && actual > 0.0 && actual.to_bits() == expected.to_bits()
                }))
    {
        return false;
    }

    let log = fit.spec.transform() == Transform::Log;
    for (index, (&original, &transformed)) in original.iter().zip(&fit.transformed).enumerate() {
        if !original.is_finite() || (log && original <= 0.0) || !transformed.is_finite() {
            return false;
        }
        let expected = match transformed_input {
            SeatsTransformedInput::ReconstructOriginal => match fit.spec.transform() {
                Transform::None => original,
                Transform::Log => {
                    let missing = fit.regressor_kinds.iter().any(|kind| {
                        matches!(kind, RegressorKind::MissingValue { period, .. }
                            if axis.period_at(index).is_ok_and(|date| date == *period))
                    });
                    if prior.is_empty() || missing {
                        original.ln()
                    } else {
                        let Some(&prior) = prior.get(index) else {
                            return false;
                        };
                        (original / prior).ln()
                    }
                }
            },
            // automdl's round trip through the fixed-length prior restores a
            // response that need not carry the raw transform's bits.
            SeatsTransformedInput::SelectedTerminal { values, .. } => match values.get(index) {
                Some(&value) if value.is_finite() => value,
                _ => return false,
            },
        };
        if transformed.to_bits() != expected.to_bits() {
            return false;
        }
    }

    let Ok(evaluated) = fit.regression_model.evaluate(axis, n) else {
        return false;
    };
    for (&kind, actual) in fit.regressor_kinds.iter().zip(&evaluated) {
        let Some((_, expected)) = expected
            .columns()
            .find(|&(expected_kind, _)| expected_kind == kind)
        else {
            return false;
        };
        if !same_bits(actual, expected) {
            return false;
        }
    }

    // Specified and automatic temporary changes share Build 62's one Tcalfa.
    fit.regressor_kinds
        .iter()
        .enumerate()
        .all(|(column, kind)| match *kind {
            RegressorKind::Outlier {
                kind: crate::regarima::OutlierType::TemporaryChange,
                frequency,
                ..
            } => fit
                .regression_model
                .temporary_change_rate(column)
                .is_some_and(|actual| {
                    actual.to_bits() == fit.spec.resolved_temporary_change_rate(frequency).to_bits()
                }),
            _ => true,
        })
}

/// Whether each value of `actual` carries the bits of the value of
/// `expected` at its index.
fn same_bits(actual: &[f64], expected: &[f64]) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
}

/// The regression model the fit's spec, outliers, user regressors and missing
/// values generate over the observed span, or `None` when the fit holds an
/// outlier its spec does not declare or one off the series' dates.
fn expected_model(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
) -> Option<RegressionModel> {
    let n = original.len();
    let mut expected = RegressionModel::from_spec(fit.spec, axis, n).ok()?;
    expected
        .add_manual_outliers(
            fit.regression_model.manual_outliers(),
            axis,
            n,
            fit.spec.resolved_temporary_change_rate(axis.frequency()),
        )
        .ok()?;
    expected
        .add_user_regressors(fit.regression_model.user_regressors(), axis, n)
        .ok()?;
    let missing = fit
        .regressor_kinds
        .iter()
        .filter_map(|kind| {
            if let RegressorKind::MissingValue { period, frequency } = *kind {
                Some((period, frequency))
            } else {
                None
            }
        })
        .map(|(period, frequency)| {
            axis.periods(n)
                .ok()?
                .position(|date| date == period)
                .filter(|&index| {
                    frequency == axis.frequency()
                        && original[index] == crate::engine::missing::PLACEHOLDER
                })
        })
        .collect::<Option<Vec<_>>>()?;
    expected.add_missing(&missing, axis, n).ok()?;
    for kind in &fit.regressor_kinds {
        if let RegressorKind::Outlier {
            kind,
            period,
            frequency,
        } = *kind
        {
            if fit
                .regression_model
                .manual_outliers()
                .contains(&crate::regarima::ManualOutlier::new(kind, period))
            {
                continue;
            }
            let declared = fit
                .spec
                .outliers()
                .is_some_and(|request| request.types().contains(kind));
            if !declared
                || frequency != axis.frequency()
                || !axis.periods(n).ok()?.any(|date| date == period)
            {
                return None;
            }
            let rate = fit.spec.resolved_temporary_change_rate(frequency);
            let column =
                crate::regression::model::evaluate_outlier(kind, period, rate, axis, n).ok()?;
            expected
                .insert_outlier(kind, period, frequency, column, rate)
                .ok()?;
        }
    }
    Some(expected)
}

/// The effects of a fit with no regressors and no fixed-length prior: neutral
/// values over the observed span.
fn neutral_effects(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    transform: SeatsTransform,
) -> Result<PreparedSeatsEffects, Error> {
    let n = original.len();
    let zero = vec![0.0; n];
    let fixed = vec![
        if transform == SeatsTransform::Log {
            1.0
        } else {
            0.0
        };
        n
    ];
    Ok(PreparedSeatsEffects {
        input: FittedEffectsInput::new(
            axis,
            transform,
            original,
            &fit.transformed,
            &zero,
            &zero,
            &fixed,
        )?,
        calendar_effects: None,
    })
}

/// The fitted effects over the observed span, with the trading-day and
/// combined calendar factors checked as inputs.
fn fitted_effects(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    transform: SeatsTransform,
    transformed_input: &super::pipeline::SeatsTransformedInput,
) -> Result<PreparedSeatsEffects, Error> {
    if fit.regression_model.value_count() != original.len() {
        return Err(Error::internal(
            "the fitted regression does not span the series",
        ));
    }
    let effects = crate::regression::effects::evaluate_regression_effects(
        &fit.regression_model,
        &fit.regression_coefficients,
        fit.spec.transform(),
        0..original.len(),
    )?;
    let input = FittedEffectsInput::with_outliers(
        axis,
        transform,
        original,
        &fit.transformed,
        crate::seats::effects::RegressionEffects {
            trading_day: effects.trading_day_linear(),
            holiday: effects.holiday_linear(),
            fixed_length: &effects.full_fixed_length,
            missing: effects.missing_linear(),
            user: effects.user_linear(),
        },
        match transformed_input {
            super::pipeline::SeatsTransformedInput::ReconstructOriginal => {
                crate::seats::effects::SeriesPriorRestoration::Restore
            }
            super::pipeline::SeatsTransformedInput::SelectedTerminal {
                prior_restoration, ..
            } => *prior_restoration,
        },
        crate::seats::effects::OutlierLinearEffects {
            // CHKADJ:146 counts missing-value columns in NAO as well.
            // ANSUB9 sets Nouir, activating DETCOMP's final adjustment.
            requires_final_adjustment: effects.has_user()
                || fit.regressor_kinds.iter().any(|kind| {
                    matches!(
                        kind,
                        RegressorKind::Outlier { .. } | RegressorKind::MissingValue { .. }
                    )
                }),
            level_shift: effects.level_shift_linear(),
            additive: effects.additive_outlier_linear(),
            temporary_change: effects.temporary_change_linear(),
        },
    )?;
    let observed = effects.calendar_effects(axis.start());
    for (values, series) in [
        (observed.estimated().values(), Series::TradingDayLinear),
        (observed.combined().values(), Series::Patd),
    ] {
        for (index, &value) in values.iter().enumerate() {
            checked(
                Stage::FittedEffectsPreparation,
                series,
                index,
                value,
                transform == SeatsTransform::Log,
            )?;
        }
    }
    Ok(PreparedSeatsEffects {
        input,
        calendar_effects: (fit.spec.calendar().is_some() || effects.has_final_calendar())
            .then_some(observed),
    })
}
