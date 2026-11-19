//! Explicit-model AICC tests from arima.f, tdaic.f and easaic.f.
use crate::Error;
use crate::api::ResolvedRegArimaSpec;
use crate::arima::{ArmaStart, EstimatedRegArima, FitContext, estimate, fit_transform_candidate};
use crate::error::NumericalDetail;
use crate::regarima::{CalendarFeature, CalendarSelection, CalendarSpec};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;

/// The calendar `spec` asks the AICC tests to select from, if any.
pub(super) fn requested(spec: ResolvedRegArimaSpec) -> Option<CalendarSpec> {
    spec.calendar()
        .filter(|calendar| calendar.selection() == CalendarSelection::AicTest)
}

pub(super) fn estimate_selected(
    original: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    mut spec: ResolvedRegArimaSpec,
    tested: CalendarSpec,
    mut regression: RegressionModel,
) -> Result<EstimatedRegArima, Error> {
    let mut calendar = tested.with_selection(CalendarSelection::Specified);
    if let Some(basis) = calendar.basis() {
        let without = calendar.without_basis();
        regression.replace_aic_trading_day(None, spec, axis, original.len())?;
        let no_td = score(
            original,
            missing,
            axis,
            selected_spec(spec, without),
            &regression,
            CalendarFeature::TradingDay,
            false,
        )?;
        regression.replace_aic_trading_day(Some(basis), spec, axis, original.len())?;
        let with_td = score(
            original,
            missing,
            axis,
            selected_spec(spec, calendar),
            &regression,
            CalendarFeature::TradingDay,
            true,
        )?;
        // tdaic's strict difference test chooses no TD on an exact tie.
        if no_td - with_td <= 0.0 {
            calendar = without;
            regression.replace_aic_trading_day(None, spec, axis, original.len())?;
        }
        spec = selected_spec(spec, calendar);
    }
    if let Some(easter) = calendar.easter() {
        let without = calendar.without_easter();
        regression.replace_aic_easter(None, axis, original.len())?;
        let no_easter = score(
            original,
            missing,
            axis,
            selected_spec(spec, without),
            &regression,
            CalendarFeature::Easter,
            false,
        )?;
        regression.replace_aic_easter(Some(easter), axis, original.len())?;
        let with_easter = score(
            original,
            missing,
            axis,
            selected_spec(spec, calendar),
            &regression,
            CalendarFeature::Easter,
            true,
        )?;
        if no_easter - with_easter <= 0.0 {
            calendar = without;
            regression.replace_aic_easter(None, axis, original.len())?;
        }
        spec = selected_spec(spec, calendar);
    }
    // editor backs up the original starts in Ap1; both test routines restore
    // them before every fit (strtvl resolves unspecified values to 0.1).
    // The selected model is refitted from those same starts, which also gives
    // the covariance the report and the forecasts use.
    let mut fitting =
        super::automatic_transform::remove_fixed_length(original, regression.fixed_length());
    super::missing::reset(&mut fitting, missing);
    estimate(
        &fitting,
        FitContext::unadjusted(original),
        axis,
        spec,
        regression,
        ArmaStart::Default,
    )
}

fn selected_spec(spec: ResolvedRegArimaSpec, calendar: CalendarSpec) -> ResolvedRegArimaSpec {
    spec.with_calendar((!calendar.is_empty()).then_some(calendar))
}

fn score(
    original: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression: &RegressionModel,
    feature: CalendarFeature,
    included: bool,
) -> Result<f64, Error> {
    let mut fitting =
        super::automatic_transform::remove_fixed_length(original, regression.fixed_length());
    super::missing::reset(&mut fitting, missing);
    let fit = fit_transform_candidate(
        &fitting,
        FitContext::unadjusted(original),
        axis,
        spec,
        regression.clone(),
        ArmaStart::Default,
    )?;
    fit.likelihood
        .and_then(|likelihood| likelihood.aicc)
        .filter(|score| score.is_finite() && fit.innovation_variance > 0.0)
        .ok_or(Error::from(NumericalDetail::CalendarSelectionUnavailable {
            feature,
            included,
        }))
}
