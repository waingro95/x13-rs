//! Final X-11 decomposition after the regARIMA fit, and the X-11 runs without
//! one (the Same bypass and direct X-11).
//!
//! Every pipeline completes its X-11 run through
//! [`x11::Components::complete`]: residual seasonality of D11, then of the
//! modified series, then quality, then spectra, each appending its warnings
//! in that order. It returns [`x11::CompletedComponents`], the only form
//! [`Adjustment`] accepts, so a skipped or repeated step cannot reach a
//! public result. The pipelines differ only in the [`x11::Completion`] they
//! pass. The Same bypass, with or without legacy Easter, and direct X-11
//! decompose the raw series with no regARIMA effects and share
//! [`complete_unmodeled`]: the modified series from the run's own D11 and
//! D12, quality with the x11pt3 legacy calendar when Easter ran, and spectra
//! of the first difference. [`finish`] builds its own in [`complete_fitted`],
//! because the fitted effects change the modified series, the level shift and
//! the difference order.

use crate::error::SpecProblem;
use std::ops::Range;

use super::FinalX11Options;
use super::x11_configuration::regression_mode_compatible;
use crate::api::{Adjustment, Backcasts, Forecasts, ModelResult, ResolvedRegArimaSpec};
use crate::arima::{EstimatedRegArima, ExtensionsResult};
use crate::regarima::{CalendarEffects, TransformSelectionResult};
use crate::regression::effects::{EvaluatedEffects, evaluate_regression_effects};
use crate::regression::model::RegressorKind;
use crate::time::{AxisSpan, TimeAxis};
use crate::{Error, Transform, Warning, x11};

pub(super) struct PostFitContext {
    pub(super) series_axis: TimeAxis,
    pub(super) easter: Option<x11::PreparedEaster>,
    pub(super) mode: x11::Mode,
    pub(super) final_options: FinalX11Options,
    pub(super) prepared_options: x11::PreparedOptions,
    pub(super) transform_selection: Option<TransformSelectionResult>,
    pub(super) restore_missing_prior: bool,
    pub(super) warnings: PostFitWarnings,
    pub(super) residual_override: Option<Vec<f64>>,
}

/// x11ari.f `IF(Same)`: a constant series skips regARIMA estimation for every
/// model, explicit or automatic, with or without legacy Easter. Raw
/// observations survive; the editor options keep the requested augmented span.
pub(super) fn finish_bypass(
    values: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    prepared: x11::PreparedOptions,
    easter: Option<x11::PreparedEaster>,
    holiday: x11::HolidayAdjustment,
    warnings: Vec<Warning>,
) -> Result<Adjustment, Error> {
    let observed = 0..values.len();
    let (components, easter_result) = match easter {
        Some(easter) => {
            let working = easter.remove_observed_prior(values)?;
            let (components, result) =
                x11::finish_easter(&working, values, axis, observed, mode, prepared, easter)?;
            (components, Some(result))
        }
        None => (
            x11::adjust_extended(values, axis, observed, mode, prepared)?,
            None,
        ),
    };
    let calendar = easter_result
        .as_ref()
        .map(|easter| legacy_calendar(easter.factors().values(), holiday));
    let completed = complete_unmodeled(components, values, axis, calendar)?;
    let mut adjustment = Adjustment::from_x11_components(axis.start(), completed, warnings);
    if let Some(easter) = easter_result {
        adjustment.set_x11_easter(easter);
    }
    Ok(adjustment)
}

/// The diagnostic completions of an X-11 run without regARIMA effects:
/// the modified series from the run's own D11 and D12, quality with the
/// legacy-Easter `calendar` when Easter ran, and spectra of the first
/// difference.
pub(super) fn complete_unmodeled(
    components: x11::Components,
    values: &[f64],
    axis: TimeAxis,
    calendar: Option<Vec<f64>>,
) -> Result<x11::CompletedComponents, Error> {
    let dates = axis.span(values.len())?;
    let modified = x11::seasonality::modified_adjusted(
        &components.seasonally_adjusted,
        &components.trend_cycle,
        &components.intermediate_extreme_weights,
    )?;
    Ok(components.complete(
        values,
        dates,
        x11::Completion {
            modified,
            quality: x11::quality::QualityAdjustments {
                calendar,
                ..Default::default()
            },
            spectral: x11::spectral::SpectralCompletion {
                difference_order: 1,
                same_original: crate::numerics::is_constant_series(values),
                level_shift: None,
                remove_final_level_shift: false,
            },
        },
    ))
}

/// x11pt3's final calendar factors (Faccal) from the legacy-Easter factors
/// H1. They are not table CHL, the combined holiday factors computed earlier.
pub(super) fn legacy_calendar(factors: &[f64], holiday: x11::HolidayAdjustment) -> Vec<f64> {
    factors
        .iter()
        .map(|&factor| {
            // Faccal = 1.0 * H1; the unit factor is neutral and dropped.
            let calendar = factor;
            match holiday {
                // keepholiday divides H1 back out: 1.0, or NaN where H1 is
                // zero or not finite.
                x11::HolidayAdjustment::Retain => calendar / factor,
                x11::HolidayAdjustment::Remove => calendar,
            }
        })
        .collect()
}

/// The warnings raised before the final X-11 run.
pub(super) enum PostFitWarnings {
    /// The fit's own warnings follow these.
    BeforeFit(Vec<Warning>),
    /// The model search has already merged the fit's warnings into these.
    Complete(Vec<Warning>),
}

/// The X-11 finish after a regARIMA fit, following arima.f, adjreg.f and
/// x11pt3.f in order: extensions, the missing-adjusted original
/// ([`missing_adjusted_original`]), the two compatibility checks, the X-11
/// input with or without fitted effects ([`input_with_effects`],
/// [`input_without_effects`]), the run ([`run_x11`]), the fitted completion
/// ([`complete_fitted`]), then warnings and the result.
///
/// The order of the steps, of the compatibility checks and of the warning
/// pushes decides which error a request gets and how its warnings are ordered.
pub(super) fn finish(
    values: &[f64],
    mut estimated: EstimatedRegArima,
    context: PostFitContext,
) -> Result<Adjustment, Error> {
    let axis = context.series_axis;
    let spec = estimated.spec;
    let mode = context.mode;
    let residual_spectral = super::regarima_spectral::compute(
        estimated.axis,
        estimated.transformed.len(),
        context
            .residual_override
            .as_deref()
            .unwrap_or(&estimated.residuals),
        estimated.report().status.converged,
        estimated.innovation_variance,
    )?;
    let mut extensions = super::model_span::apply_to_series(
        &mut estimated,
        values,
        axis,
        context
            .easter
            .as_ref()
            .map(x11::PreparedEaster::observed_factors),
    )?;
    // arima/adjreg replace the original series by Orixmv before final X-11
    // reconstruction. MV effects are never restored as AO irregular effects.
    let missing_adjusted =
        missing_adjusted_original(&estimated, values.len(), context.restore_missing_prior)?;
    let presence = EffectPresence::of(&estimated, missing_adjusted.is_some());
    let values = missing_adjusted.as_deref().unwrap_or(values);
    if mode == x11::Mode::PseudoAdditive && presence.preadjustment {
        return Err(Error::from(
            SpecProblem::IncompatiblePseudoAdditivePreadjustment,
        ));
    }
    if presence.any && !regression_mode_compatible(spec.transform(), mode) {
        return Err(Error::from(
            SpecProblem::IncompatibleRegressionAdjustmentMode {
                transform: spec.transform(),
                mode,
            },
        ));
    }
    let run = FittedRun {
        values,
        axis,
        dates: axis.span(values.len())?,
        spec,
        mode,
        final_options: context.final_options,
    };
    let input = if presence.any {
        input_with_effects(run, &estimated, &extensions, presence.calendar)?
    } else {
        input_without_effects(run, &estimated, &extensions, context.easter.is_some())?
    };
    let (components, easter_result) = run_x11(
        run,
        &input.inserted,
        &mut extensions,
        context.easter,
        context.prepared_options,
    )?;
    let components = complete_fitted(
        components,
        run,
        input.restoration,
        &input.inserted,
        easter_result
            .as_ref()
            .map(|easter| easter.factors().values()),
    )?;
    let mut warnings = match context.warnings {
        PostFitWarnings::BeforeFit(mut warnings) => {
            warnings.append(&mut estimated.warnings);
            warnings
        }
        PostFitWarnings::Complete(warnings) => warnings,
    };
    if let Some(warning) = super::regarima_spectral::warning(&residual_spectral) {
        warnings.push(warning);
    }
    if let Some(warning) = input.inserted.warning {
        warnings.push(warning);
    }
    // arima calls prtfct/mkback before extend: saved forecasts and backcasts
    // remain available even when extend rejects their insertion into X-11.
    let forecasts = Forecasts::from_points(extensions.forecasts);
    let backcasts = Backcasts::from_points(extensions.backcasts);
    let outliers = super::outlier_metadata::detected_outliers(&estimated);
    let mut model =
        ModelResult::from_estimated(estimated, context.residual_override, residual_spectral);
    model.set_transform_selection(context.transform_selection);
    let mut adjustment = Adjustment::from_x11_components(axis.start(), components, warnings);
    adjustment.set_regarima_outputs(
        model,
        forecasts,
        backcasts,
        input.calendar_effects,
        outliers,
    );
    if let Some(easter) = easter_result {
        adjustment.set_x11_easter(easter);
    }
    Ok(adjustment)
}

/// The fitted series and settings the steps after estimation share.
#[derive(Clone, Copy)]
struct FittedRun<'a> {
    /// The original series, with missing-value effects removed when the fit
    /// estimated any.
    values: &'a [f64],
    axis: TimeAxis,
    /// `axis` checked over `values`.
    dates: AxisSpan,
    spec: ResolvedRegArimaSpec,
    mode: x11::Mode,
    final_options: FinalX11Options,
}

/// arima/adjreg's Orixmv: the original series with the fitted missing-value
/// effects removed, when the fit estimated any.
fn missing_adjusted_original(
    estimated: &EstimatedRegArima,
    count: usize,
    restore_prior: bool,
) -> Result<Option<Vec<f64>>, Error> {
    let has_missing = estimated
        .regressor_kinds
        .iter()
        .any(|kind| matches!(kind, RegressorKind::MissingValue { .. }));
    if !has_missing {
        return Ok(None);
    }
    let transform = estimated.spec.transform();
    let effects = evaluate_regression_effects(
        &estimated.regression_model,
        &estimated.regression_coefficients,
        transform,
        0..count,
    )?;
    Ok(Some(effects.missing_adjusted_original(
        &estimated.transformed,
        transform,
        restore_prior,
    )))
}

/// Which fitted effects adjreg removes before the final X-11 run.
#[derive(Clone, Copy)]
struct EffectPresence {
    /// A configured calendar, or supplied trading-day or holiday regressors.
    calendar: bool,
    /// Calendar, outliers or any supplied regressor: the preadjustment
    /// factors editor.f refuses under pseudo-additive X-11. Missing values
    /// produce none.
    preadjustment: bool,
    /// Missing values or any preadjustment.
    any: bool,
}

impl EffectPresence {
    fn of(estimated: &EstimatedRegArima, missing: bool) -> Self {
        let kinds = &estimated.regressor_kinds;
        let calendar = estimated
            .spec
            .calendar()
            .is_some_and(|calendar| !calendar.is_empty())
            || kinds.iter().any(|kind| {
                matches!(
                    kind,
                    RegressorKind::User {
                        kind: crate::regarima::UserRegressorType::TradingDay
                            | crate::regarima::UserRegressorType::Holiday,
                        ..
                    }
                )
            });
        let preadjustment = calendar
            || kinds.iter().any(|kind| {
                matches!(
                    kind,
                    RegressorKind::Outlier { .. } | RegressorKind::User { .. }
                )
            });
        Self {
            calendar,
            preadjustment,
            any: missing || preadjustment,
        }
    }
}

/// What the final X-11 run decomposes, and what the final restoration and
/// the public result need from the fitted effects.
struct X11Input {
    inserted: InsertedSeries,
    calendar_effects: Option<CalendarEffects>,
    restoration: Option<EffectRestoration>,
}

/// The fitted effects over the extended span, restored to the final
/// components over `observed`, their span within it.
///
/// It exists only when effects were removed; otherwise [`rebuild_unmodeled`]
/// rebuilds D11 and D13 from the run.
struct EffectRestoration {
    effects: EvaluatedEffects,
    observed: Range<usize>,
}

/// adjreg with effects: the transformed response and its extensions, less
/// the fitted effects, back on the original scale, then extend's insertion.
fn input_with_effects(
    run: FittedRun<'_>,
    estimated: &EstimatedRegArima,
    extensions: &ExtensionsResult,
    calendar: bool,
) -> Result<X11Input, Error> {
    let FittedRun {
        values,
        axis,
        spec,
        mode,
        ..
    } = run;
    let backcast_count = extensions.backcasts.len();
    let full_count = backcast_count + values.len() + extensions.forecasts.len();
    let extended_axis = axis.shift(-signed(backcast_count))?;
    let full_model = estimated
        .regression_model
        .evaluated_over(extended_axis, full_count)?;
    let effects = evaluate_regression_effects(
        &full_model,
        &estimated.regression_coefficients,
        spec.transform(),
        backcast_count..backcast_count + values.len(),
    )?;

    let mut adjusted = Vec::with_capacity(full_count);
    adjusted.extend(extensions.backcasts.iter().map(|point| point.transformed));
    adjusted.extend_from_slice(&estimated.transformed);
    adjusted.extend(extensions.forecasts.iter().map(|point| point.transformed));
    effects.remove_from_transformed_response(&mut adjusted)?;
    let adjusted = adjusted
        .into_iter()
        .map(|value| crate::transform::inverse(spec.transform(), value))
        .collect::<Vec<_>>();
    let forecast_start = backcast_count + values.len();
    let inserted = insert_extensions(
        &adjusted[backcast_count..forecast_start],
        axis,
        spec.transform(),
        mode,
        &adjusted[forecast_start..],
        &adjusted[..backcast_count],
    )?;
    let calendar_effects = calendar.then(|| effects.calendar_effects(axis.start()));
    // Pseudo-additive allows only missing-value effects, which leave nothing
    // to restore; its D11 comes from x11pt3's own reconstruction instead.
    let restoration = (mode != x11::Mode::PseudoAdditive).then_some(EffectRestoration {
        effects,
        observed: backcast_count..forecast_start,
    });
    Ok(X11Input {
        inserted,
        calendar_effects,
        restoration,
    })
}

/// adjreg without effects, then extend's insertion. `easter` says whether
/// legacy Easter runs, which decides the extension values X-11 receives.
fn input_without_effects(
    run: FittedRun<'_>,
    estimated: &EstimatedRegArima,
    extensions: &ExtensionsResult,
    easter: bool,
) -> Result<X11Input, Error> {
    let FittedRun {
        axis, spec, mode, ..
    } = run;
    // `arima` calls `adjreg` even with zero effects: X-11 receives the
    // inverse of the fitted transformed observations, not the raw bits.
    let observed_inverse = estimated
        .transformed
        .iter()
        .map(|&value| crate::transform::inverse(spec.transform(), value))
        .collect::<Vec<_>>();
    // With legacy Easter, X-11 extends with the prior-adjusted working
    // tails; without it, with the original-scale extensions.
    let extension_value = |transformed: f64, original: f64| {
        if easter {
            crate::transform::inverse(spec.transform(), transformed)
        } else {
            original
        }
    };
    let inserted = insert_extensions(
        &observed_inverse,
        axis,
        spec.transform(),
        mode,
        &extensions
            .forecasts
            .iter()
            .map(|point| extension_value(point.transformed, point.original))
            .collect::<Vec<_>>(),
        &extensions
            .backcasts
            .iter()
            .map(|point| extension_value(point.transformed, point.original))
            .collect::<Vec<_>>(),
    )?;
    Ok(X11Input {
        inserted,
        calendar_effects: None,
        restoration: None,
    })
}

/// The final X-11 run over the inserted series. Legacy Easter first restores
/// the public forecast and backcast intervals, then runs on the working
/// series beside the raw one, whose observed span is the original series.
fn run_x11(
    run: FittedRun<'_>,
    inserted: &InsertedSeries,
    extensions: &mut ExtensionsResult,
    easter: Option<x11::PreparedEaster>,
    prepared_options: x11::PreparedOptions,
) -> Result<(x11::Components, Option<crate::x11::EasterEffect>), Error> {
    let FittedRun {
        values, axis, mode, ..
    } = run;
    if let Some(easter) = easter {
        restore_extension_intervals(extensions, &easter, axis, values.len())?;
        let mut raw = inserted.values.clone();
        raw[inserted.observed.clone()].copy_from_slice(values);
        let (components, result) = x11::finish_easter(
            &inserted.values,
            &raw,
            inserted.axis,
            inserted.observed.clone(),
            mode,
            prepared_options,
            easter,
        )?;
        Ok((components, Some(result)))
    } else {
        let components = x11::adjust_extended(
            &inserted.values,
            inserted.axis,
            inserted.observed.clone(),
            mode,
            prepared_options,
        )?;
        Ok((components, None))
    }
}

/// x11pt3 after a fitted run: the quality factors of the fitted effects, the
/// legacy-Easter calendar from `easter_factors`, the final D11, D12 and D13
/// with the effects restored (or rebuilt without them), the pseudo-additive
/// final check, then the shared completions.
fn complete_fitted(
    mut components: x11::Components,
    run: FittedRun<'_>,
    restoration: Option<EffectRestoration>,
    inserted: &InsertedSeries,
    easter_factors: Option<&[f64]>,
) -> Result<x11::CompletedComponents, Error> {
    let FittedRun {
        values,
        dates,
        spec,
        mode,
        final_options,
        ..
    } = run;
    let holiday = final_options.holiday_adjustment;
    let mut quality_adjustments = restoration
        .as_ref()
        .map(|restoration| {
            restoration.effects.quality_adjustments(
                spec.transform(),
                restoration.observed.clone(),
                holiday,
            )
        })
        .unwrap_or_default();
    let legacy_calendar = easter_factors
        .map(|factors| match &restoration {
            // The combined-holiday factors (the second vector) are not read here.
            Some(restoration) => restoration
                .effects
                .legacy_calendar_factors(restoration.observed.clone(), factors, holiday)
                .map(|(calendar, _)| calendar),
            None => Ok(legacy_calendar(factors, holiday)),
        })
        .transpose()?;
    if let Some(calendar) = &legacy_calendar {
        quality_adjustments.calendar = Some(calendar.clone());
    }
    let modified = if let Some(EffectRestoration { effects, observed }) = restoration {
        restore_final_components(
            &mut components,
            values,
            &effects,
            observed,
            spec.transform(),
            final_options,
            legacy_calendar.as_deref(),
        )?
    } else {
        rebuild_unmodeled(&mut components, values, mode)?
    };
    if mode == x11::Mode::PseudoAdditive {
        check_pseudo_additive_final(&components, inserted)?;
    }
    let spectral_completion = x11::spectral::SpectralCompletion {
        difference_order: usize::from(spec.model().regular().d() + spec.model().seasonal().d())
            .saturating_sub(1)
            .max(1),
        same_original: crate::numerics::is_constant_series(values),
        level_shift: quality_adjustments.level_shift.clone(),
        remove_final_level_shift: matches!(final_options.final_outliers, x11::FinalOutliers::Remove(types) if types.contains(crate::regarima::OutlierType::LevelShift)),
    };
    Ok(components.complete(
        values,
        dates,
        x11::Completion {
            modified,
            quality: quality_adjustments,
            spectral: spectral_completion,
        },
    ))
}

/// `x11pt3` without fitted effects reconstructs D11 from the original series
/// and then D13 from D11 and D12. The X-11 mode, not the transform, decides
/// between division and subtraction here (divsub). Returns the modified
/// seasonally adjusted series.
fn rebuild_unmodeled(
    components: &mut x11::Components,
    values: &[f64],
    mode: x11::Mode,
) -> Result<Vec<f64>, Error> {
    components.seasonally_adjusted = values
        .iter()
        .zip(&components.seasonal)
        .zip(&components.trend_cycle)
        .map(|((&value, &seasonal), &trend)| match mode {
            x11::Mode::Additive => value - seasonal,
            x11::Mode::Multiplicative | x11::Mode::LogAdditive => value / seasonal,
            x11::Mode::PseudoAdditive => x11::pseudo_additive::adjusted(value, trend, seasonal),
        })
        .collect();
    components.irregular = components
        .seasonally_adjusted
        .iter()
        .zip(&components.trend_cycle)
        .map(|(&value, &trend)| match mode {
            x11::Mode::Additive => value - trend,
            x11::Mode::Multiplicative | x11::Mode::LogAdditive | x11::Mode::PseudoAdditive => {
                value / trend
            }
        })
        .collect();
    components.quality_workspace.base_irregular = components.irregular.clone();
    x11::seasonality::modified_adjusted(
        &components.seasonally_adjusted,
        &components.trend_cycle,
        &components.intermediate_extreme_weights,
    )
}

/// The pseudo-additive final adjustment divides D11 by D12 at every observed
/// period, dated within the series X-11 ran on.
fn check_pseudo_additive_final(
    components: &x11::Components,
    inserted: &InsertedSeries,
) -> Result<(), Error> {
    let dates = inserted.axis.span(inserted.values.len())?;
    for (index, (&adjusted, &trend)) in components
        .seasonally_adjusted
        .iter()
        .zip(&components.trend_cycle)
        .enumerate()
    {
        x11::pseudo_additive::divide(
            adjusted,
            trend,
            dates,
            inserted.observed.start + index,
            crate::error::X11PseudoAdditiveStage::FinalAdjustment,
        )?;
    }
    Ok(())
}

/// x11pt3's final D11, irregular and D13 restoration of the fitted effects.
/// `legacy_calendar` is the legacy-Easter Faccal, which replaces the fitted
/// calendar and prior removal from D11 when Easter ran.
fn restore_final_components(
    components: &mut x11::Components,
    original: &[f64],
    effects: &EvaluatedEffects,
    observed: Range<usize>,
    transform: Transform,
    final_options: FinalX11Options,
    legacy_calendar: Option<&[f64]>,
) -> Result<Vec<f64>, Error> {
    debug_assert_eq!(observed.len(), original.len());
    debug_assert_eq!(components.seasonal.len(), original.len());
    debug_assert_eq!(components.trend_cycle.len(), original.len());
    let mut seasonally_adjusted = original.to_vec();
    remove_component(&mut seasonally_adjusted, &components.seasonal, transform);
    if let Some(calendar) = legacy_calendar {
        // x11pt3 applies the calendar to D11 only when the model has calendar
        // effects; legacy Easter (Khol) alone does not.
        if effects.has_final_calendar() {
            for (value, factor) in seasonally_adjusted.iter_mut().zip(calendar) {
                *value /= factor;
            }
        }
    } else {
        effects.remove_final_calendar_and_prior(
            &mut seasonally_adjusted,
            transform,
            observed.clone(),
            final_options.holiday_adjustment,
        );
    }

    let mut irregular = seasonally_adjusted.clone();
    remove_component(&mut irregular, &components.trend_cycle, transform);
    effects.remove_outliers_from_irregular(&mut irregular, transform, observed.clone());
    // The spectrum reads x11pt3's Part E irregular (Sti) as it stands here,
    // before the separate sti2/D13 restoration.
    x11::spectral::retain_irregular(components, &irregular);
    components.quality_workspace.base_irregular = irregular.clone();
    effects.restore_additive_outliers_to_irregular(&mut irregular, transform, observed.clone());
    if final_options.temporary_change_allocation == x11::TemporaryChangeAllocation::Irregular {
        effects.restore_temporary_changes(&mut irregular, transform, observed.clone());
    }
    if let x11::FinalOutliers::Remove(types) = final_options.final_outliers {
        effects.remove_selected_outliers_from_final(
            &mut seasonally_adjusted,
            transform,
            observed.clone(),
            types,
        );
    }
    let mut replacement = components.trend_cycle.clone();
    effects.restore_modified_replacement(
        &mut replacement,
        transform,
        observed.clone(),
        final_options.final_outliers,
        final_options.temporary_change_allocation,
    );
    let mut modified = x11::seasonality::modified_adjusted(
        &seasonally_adjusted,
        &replacement,
        &components.intermediate_extreme_weights,
    )?;
    effects.remove_modified_outliers(
        &mut modified,
        transform,
        observed.clone(),
        final_options.final_outliers,
    );
    if !final_options
        .final_outliers
        .removes(crate::regarima::OutlierType::LevelShift)
    {
        effects.restore_level_shift_to_trend(
            &mut components.trend_cycle,
            transform,
            observed.clone(),
        );
    }
    if final_options.temporary_change_allocation == x11::TemporaryChangeAllocation::Trend
        && !final_options
            .final_outliers
            .removes(crate::regarima::OutlierType::TemporaryChange)
    {
        effects.restore_temporary_changes(&mut components.trend_cycle, transform, observed);
    }
    components.seasonally_adjusted = seasonally_adjusted;
    components.irregular = irregular;
    Ok(modified)
}

/// `count` as a signed number of periods. It saturates, and a shift that
/// large is past the end of any axis, which [`TimeAxis::shift`] reports.
fn signed(count: usize) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}

/// prtfct/mkback restore only the public original-scale interval after
/// ordinary priors, multiplying each forecast, then each backcast, by the
/// legacy Easter factor of its period. extend consumes the unchanged working
/// tails.
fn restore_extension_intervals(
    extensions: &mut ExtensionsResult,
    easter: &x11::PreparedEaster,
    axis: TimeAxis,
    observed_count: usize,
) -> Result<(), Error> {
    let forecast_axis = axis.shift(signed(observed_count))?;
    let forecast_factors = easter.factors_at(forecast_axis, extensions.forecasts.len())?;
    for (point, &factor) in extensions.forecasts.iter_mut().zip(forecast_factors) {
        restore_interval(
            &mut point.original,
            &mut point.lower_95,
            &mut point.upper_95,
            factor,
        )?;
    }
    let backcast_axis = axis.shift(-signed(extensions.backcasts.len()))?;
    let backcast_factors = easter.factors_at(backcast_axis, extensions.backcasts.len())?;
    for (point, &factor) in extensions.backcasts.iter_mut().zip(backcast_factors) {
        restore_interval(
            &mut point.original,
            &mut point.lower_95,
            &mut point.upper_95,
            factor,
        )?;
    }
    Ok(())
}

fn restore_interval(
    point: &mut f64,
    lower: &mut f64,
    upper: &mut f64,
    factor: f64,
) -> Result<(), Error> {
    for value in [point, lower, upper] {
        *value *= factor;
        if !value.is_finite() {
            return Err(Error::numerical(
                crate::error::NumericalOperation::Extension,
            ));
        }
    }
    Ok(())
}

fn remove_component(values: &mut [f64], component: &[f64], transform: Transform) {
    debug_assert_eq!(values.len(), component.len());
    for (value, component) in values.iter_mut().zip(component) {
        match transform {
            Transform::None => *value -= component,
            Transform::Log => *value /= component,
        }
    }
}

struct InsertedSeries {
    values: Vec<f64>,
    axis: TimeAxis,
    observed: Range<usize>,
    warning: Option<Warning>,
}

fn insert_extensions(
    observed_values: &[f64],
    axis: TimeAxis,
    transform: Transform,
    mode: x11::Mode,
    forecasts: &[f64],
    backcasts: &[f64],
) -> Result<InsertedSeries, Error> {
    // extend.f checks the forecasts before the backcasts, and one zero or
    // negative value drops both. Extensions of a log model are exponentials,
    // so they are not checked.
    let checked = mode != x11::Mode::Additive && transform != Transform::Log;
    let unusable = |values: &[f64]| checked && values.iter().any(|value| *value <= 0.0);
    let warning = if unusable(forecasts) {
        Some(Warning::ForecastsNotUsed)
    } else if unusable(backcasts) {
        Some(Warning::BackcastsNotUsed)
    } else {
        None
    };
    if warning.is_some() {
        return Ok(InsertedSeries {
            values: observed_values.to_vec(),
            axis,
            observed: 0..observed_values.len(),
            warning,
        });
    }

    let backcast_count = backcasts.len();
    let mut values = Vec::with_capacity(backcast_count + observed_values.len() + forecasts.len());
    values.extend_from_slice(backcasts);
    values.extend_from_slice(observed_values);
    values.extend_from_slice(forecasts);
    Ok(InsertedSeries {
        values,
        axis: axis.shift(-signed(backcast_count))?,
        observed: backcast_count..backcast_count + observed_values.len(),
        warning: None,
    })
}

#[cfg(test)]
mod tests;
