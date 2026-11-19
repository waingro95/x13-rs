//! The SEATS pipeline: estimate the regARIMA model (or take automd's
//! terminal fit), save the original forecasts, then hand the fit to its SEATS
//! family.
//!
//! # Layers
//!
//! SEATS composition has two layers, each in its own modules:
//!
//! - **Pipeline** (this module): Build 62's sequence around SEATS.
//!   [`adjust`] runs editor.f, trnaic.f, regvar and the explicit estimator.
//!   [`prepare_automatic`] and [`finish_prepared`] split automd's selection
//!   from the SEATS call, so the API can check the selected model in between.
//!   Both paths save the original regARIMA forecasts and the summary Q
//!   ([`summary_q`]) before SEATS runs, as arima.f does. Neither knows any
//!   family.
//! - **Family strategy** (`strategy`, plus one module per family:
//!   `airline`, `seasonal_ma`, `second_difference`, `ar1_seasonal_ma`,
//!   `arma`, `ar2`): `strategy::finish` maps the selected family to its
//!   `FamilyStrategy`, prepares the fitted effects once and assembles the
//!   public result. Each family module checks the fit, runs the family in
//!   `crate::seats` and converts its passes into a `seats::Diagnostics`.
//!
//! The families raise [`crate::seats::Error`], which converts into
//! [`Error`](crate::Error) through `From`.
//!
//! # One regression build
//!
//! [`regression_response`] is regvar on the modelspan for both entry points:
//! the spec's columns, the manual outliers, the user regressors and one
//! missing-value column per gap, then the response without the fixed length
//! and with x11pt1.f's placeholder reset at each gap.
//!
//! The automatic path also runs the manual-outlier step.
//! `validate_model_policy` leaves it no manual outliers, and the spec's own
//! columns stay far below adrgef's limit, so the step is a no-op there, as in
//! `x11_configuration::X11Preparation`.
use crate::api::ResolvedRegArimaSpec;
use crate::arima::model::ExpandedModel;
use crate::arima::{EstimatedRegArima, FitContext};
use crate::engine::automatic_transform::{
    CandidateModelContext, OriginalSeries, ResolvedTransform, TransformRequest, remove_fixed_length,
};
use crate::engine::model_span::EstimationInput;
use crate::error::{NumericalDetail, UnsupportedDetail};
use crate::regarima::{ForecastSpec, UserRegressor};
use crate::regression::model::RegressionModel;
use crate::seats::pipeline::SummaryQ;
use crate::seats::{Failure, Stage};
use crate::time::TimeAxis;
use crate::{Adjustment, Error, Frequency, RegArimaSpec, x11};
use std::ops::RangeInclusive;

/// The series lengths SEATS accepts from a start in `start_subperiod` of
/// `frequency`: at least 36 monthly or 16 quarterly observations, and at most
/// POBS=780 or editor's 85 calendar years less the leading positions and the
/// 3s forecasts SEATS adds, whichever is tighter. From the first subperiod
/// this is CheckLen's range: 36..=780 monthly, 16..=328 quarterly.
pub(crate) fn seats_length_range(
    frequency: Frequency,
    start_subperiod: u8,
) -> RangeInclusive<usize> {
    let period = usize::from(frequency.periods_per_year());
    let minimum = match frequency {
        Frequency::Monthly => 36,
        Frequency::Quarterly => 16,
    };
    let maximum = 780.min(85 * period - (usize::from(start_subperiod) - 1) - 3 * period);
    minimum..=maximum
}

/// automd's selected fit, with what [`finish_prepared`] needs to run SEATS
/// on it.
pub(crate) struct PreparedSeatsFit {
    pub(super) terminal: crate::automdl::TerminalFit,
    pub(super) transformed_input: SeatsTransformedInput,
    pub(super) requested_extensions: ForecastSpec,
    pub(super) transform_selection: Option<crate::regarima::TransformSelectionResult>,
    pub(super) model_selection: Option<crate::regarima::ModelSelectionResult>,
    pub(super) warnings: SeatsFitWarnings,
}

/// Checks an automatic-model SEATS request and selects its model as automd
/// does, up to the SEATS call.
pub(crate) fn prepare_automatic(
    original: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    request: RegArimaSpec,
) -> Result<PreparedSeatsFit, Error> {
    request.validate_model_policy()?;
    if request.model() != crate::regarima::ModelSelection::Automatic {
        return Err(Error::internal(
            "the automatic SEATS path received an explicit model",
        ));
    }
    if original.is_empty() {
        return Err(Error::EmptyInput);
    }
    if let Some(index) = original.iter().position(|x| !x.is_finite()) {
        return Err(Error::NonFiniteValue { index });
    }
    if original
        .iter()
        .all(|x| crate::numerics::build_62_equal(*x, 0.0))
    {
        return Err(Error::AllZeroInput);
    }
    crate::api::validate_seats_request(original, axis, request.clone())?;
    let estimation = EstimationInput::new(original, missing, axis, &request)?;
    let internal = request
        .clone()
        .with_forecast(seats_forecast_extensions(estimation.axis.frequency())?);
    let resolved = TransformRequest {
        context: CandidateModelContext::AutomaticModel,
        decomposition: x11::Decomposition::SeasonalAdjustment,
        original: OriginalSeries {
            values: original,
            axis,
        },
    }
    .resolve(
        estimation.values,
        &estimation.missing,
        estimation.axis,
        internal,
    )?;
    if crate::engine::missing::skips_model(original, missing) {
        return Err(Error::DegenerateSeries);
    }
    // automd builds its own regression; this one runs regvar's checks and
    // gives the response without the fixed length.
    let (_, fitting) = regression_response(&estimation, resolved.spec, request.user_regressors())?;
    let selected = crate::automdl::select(
        &fitting,
        FitContext::unadjusted(estimation.values)
            .with_missing(&estimation.missing)
            .with_user_regressors(request.user_regressors()),
        estimation.axis,
        resolved.spec,
    )?;
    let model_selection = Some(crate::regarima::ModelSelectionResult::Selected {
        first_model: selected.first_selected_model,
        mean_significant: selected.procedural_mean,
    });
    let mut warnings = resolved.warnings;
    warnings.extend(selected.warnings);
    // trnaic saves the prior-format state before enabling its log prior;
    // ssprep also clears the saved flag when Lpradj is false. automd's
    // same-baseline restor restores that flag, not the calendar factors.
    let prior_restoration = if resolved.selection.is_some() && selected.restored_baseline {
        crate::seats::effects::SeriesPriorRestoration::Omit
    } else {
        crate::seats::effects::SeriesPriorRestoration::Restore
    };
    let transformed_input = SeatsTransformedInput::SelectedTerminal {
        values: selected.terminal.fit.transformed.clone(),
        prior_restoration,
    };
    Ok(PreparedSeatsFit {
        terminal: selected.terminal,
        transformed_input,
        requested_extensions: request.forecast(),
        transform_selection: resolved.selection,
        model_selection,
        warnings: SeatsFitWarnings::Complete(warnings),
    })
}

/// Runs SEATS on automd's selected fit and assembles the adjustment, with
/// the original forecasts and the model selection.
pub(crate) fn finish_prepared(
    original: &[f64],
    axis: TimeAxis,
    prepared: PreparedSeatsFit,
) -> Result<Adjustment, Error> {
    let selected = prepared.terminal.fit.spec.model();
    let family = crate::api::selected_family(selected).ok_or({
        if let Some(crate::regarima::ModelSelectionResult::Selected {
            first_model,
            mean_significant,
        }) = prepared.model_selection
        {
            Error::from(UnsupportedDetail::SeatsSelectedModel {
                selected,
                first_model,
                mean_significant,
            })
        } else {
            Error::internal("a fit without an automatic model selection has no SEATS family")
        }
    })?;
    let summary_q = automatic_summary_q(&prepared.terminal)?;
    let mut fit = prepared.terminal.fit;
    let estimation_span = (fit.axis != axis || fit.transformed.len() != original.len())
        .then_some((fit.axis, fit.transformed.len()));
    let count = usize::from(
        prepared
            .requested_extensions
            .forecast_periods(axis.frequency()),
    );
    // arima.f restores the ending dates before forecasting, then the beginning
    // before SEATS. The selected fit's summary Q above remains model-span based.
    let mut points =
        crate::engine::model_span::apply_to_series(&mut fit, original, axis, None)?.forecasts;
    let forecasts = if count == 0 {
        None
    } else {
        if points.len() < count {
            return Err(Error::internal(
                "the selected fit has fewer forecasts than requested",
            ));
        }
        points.truncate(count);
        validate_forecasts(&points)?;
        crate::api::Forecasts::from_points(points)
    };
    let outliers = crate::engine::outlier_metadata::detected_outliers(&fit);
    fit.spec = fit.spec.with_extensions(prepared.requested_extensions);
    let mut transformed_input = prepared.transformed_input;
    if estimation_span.is_some()
        && let SeatsTransformedInput::SelectedTerminal { values, .. } = &mut transformed_input
    {
        // setspn/regvar replaces the selected terminal response with the
        // transformed full series, without refitting its coefficients.
        values.clone_from(&fit.transformed);
    }
    let context = SeatsFinishContext {
        summary_q: SummaryQSource::Saved(summary_q),
        transformed_input,
        estimation_span,
        residual_override: prepared.terminal.residual_override,
        transform_selection: prepared.transform_selection,
        warnings: prepared.warnings,
    };
    let mut adjustment = super::strategy::finish(family, original, axis, fit, context)?;
    adjustment.set_forecasts(forecasts);
    adjustment.set_outliers(outliers);
    if let Some(selection) = prepared.model_selection {
        adjustment.set_model_selection(selection);
    }
    Ok(adjustment)
}

/// The warnings gathered before SEATS runs.
pub(super) enum SeatsFitWarnings {
    /// The fit's own warnings follow these.
    BeforeFit(Vec<crate::Warning>),
    /// The list already has the fit's warnings.
    Complete(Vec<crate::Warning>),
}

/// Where a SEATS run takes the regARIMA summary Q from.
#[derive(Clone, Copy)]
pub(super) enum SummaryQSource {
    /// From the fit the family runs on.
    FromFit,
    /// Saved before SEATS ran: from automd's terminal fit, or from the fit
    /// over the model span.
    Saved(SummaryQ),
}

impl SummaryQSource {
    /// The summary Q for a run on `fit`.
    pub(super) fn resolve(self, fit: &EstimatedRegArima) -> Result<SummaryQ, crate::seats::Error> {
        match self {
            Self::FromFit => summary_q(fit),
            Self::Saved(summary_q) => Ok(summary_q),
        }
    }
}

/// Where the fit's transformed series comes from.
#[derive(Clone)]
pub(super) enum SeatsTransformedInput {
    /// The transform of the original series.
    ReconstructOriginal,
    /// automd's selected response, with whether SEATS restores the prior.
    SelectedTerminal {
        values: Vec<f64>,
        prior_restoration: crate::seats::effects::SeriesPriorRestoration,
    },
}

/// What the family strategy needs besides the fit and the series.
pub(super) struct SeatsFinishContext {
    pub summary_q: SummaryQSource,
    pub transformed_input: SeatsTransformedInput,
    pub estimation_span: Option<(TimeAxis, usize)>,
    pub residual_override: Option<Vec<f64>>,
    pub transform_selection: Option<crate::regarima::TransformSelectionResult>,
    pub warnings: SeatsFitWarnings,
}

impl SeatsFinishContext {
    /// The context of an explicit model estimated over the whole series.
    fn explicit(
        transform_selection: Option<crate::regarima::TransformSelectionResult>,
        warnings: Vec<crate::Warning>,
    ) -> Self {
        Self {
            summary_q: SummaryQSource::FromFit,
            transformed_input: SeatsTransformedInput::ReconstructOriginal,
            estimation_span: None,
            residual_override: None,
            transform_selection,
            warnings: SeatsFitWarnings::BeforeFit(warnings),
        }
    }
}

impl SeatsFitWarnings {
    /// Every warning of the run, in order, taking the fit's own from `fit`.
    pub(super) fn into_warnings(self, fit: &mut EstimatedRegArima) -> Vec<crate::Warning> {
        match self {
            Self::BeforeFit(mut warnings) => {
                warnings.append(&mut fit.warnings);
                warnings
            }
            Self::Complete(warnings) => warnings,
        }
    }
}

/// Runs SEATS with an explicit model: estimates the regARIMA model, saves
/// its forecasts, and assembles the adjustment from the family's run.
pub(crate) fn adjust(
    original: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    request: RegArimaSpec,
    family: crate::api::SeatsFamily,
) -> Result<Adjustment, Error> {
    if let Some(calendar) = request.calendar()
        && let Some(reason) = crate::api::unsupported_calendar(calendar)
    {
        return Err(Error::from(UnsupportedDetail::SeatsOption(
            crate::api::SeatsUnsupportedOption::Calendar(reason),
        )));
    }
    let estimation = EstimationInput::new(original, missing, axis, &request)?;
    let resolved = resolve_transform(
        &estimation,
        request.clone(),
        OriginalSeries {
            values: original,
            axis,
        },
    )?;
    // Build 62 skips regARIMA estimation for a constant series whatever the
    // model, so SEATS has no fitted model here either (AUTO then fails on
    // a series of zeros). The automatic path reports the same error.
    if crate::engine::missing::skips_model(original, missing) {
        return Err(Error::DegenerateSeries);
    }
    let (regression, fitting) =
        regression_response(&estimation, resolved.spec, request.user_regressors())?;
    let mut fit =
        if let Some(calendar) = crate::engine::calendar_selection::requested(resolved.spec) {
            crate::engine::calendar_selection::estimate_selected(
                estimation.values,
                &estimation.missing,
                estimation.axis,
                resolved.spec,
                calendar,
                regression,
            )?
        } else if resolved.spec.outliers().is_some() {
            let search = crate::outliers::search::search(
                &fitting,
                FitContext::unadjusted(estimation.values),
                estimation.axis,
                resolved.spec,
                regression,
                resolved.start,
            )?;
            search.fit
        } else {
            crate::arima::estimate(
                &fitting,
                FitContext::unadjusted(estimation.values),
                estimation.axis,
                resolved.spec,
                regression,
                resolved.start,
            )?
        };
    let outliers = crate::engine::outlier_metadata::detected_outliers(&fit);
    // editor.f's model-span warnings precede the transform selection's.
    let mut warnings: Vec<_> = estimation.dropped_outlier_warnings().collect();
    warnings.extend(resolved.warnings);
    let mut context = SeatsFinishContext::explicit(resolved.selection, warnings);
    if estimation.range != (0..original.len()) {
        // arima.f saves the model-span summary Q before restoring the response.
        context.summary_q = SummaryQSource::Saved(summary_q(&fit)?);
        context.estimation_span = Some((fit.axis, fit.transformed.len()));
    }
    // Build 62 saves the original regARIMA forecasts before calling SEATS, so
    // they come from the regARIMA fit, not from SEATS's tails or refits.
    let forecasts = if fit.spec.extensions().forecast_periods(axis.frequency()) == 0
        && context.estimation_span.is_none()
    {
        None
    } else {
        let points =
            crate::engine::model_span::apply_to_series(&mut fit, original, axis, None)?.forecasts;
        validate_forecasts(&points)?;
        crate::api::Forecasts::from_points(points)
    };
    let mut adjustment = super::strategy::finish(family, original, axis, fit, context)?;
    adjustment.set_forecasts(forecasts);
    adjustment.set_outliers(outliers);
    Ok(adjustment)
}

/// editor.f:384–396 raises the internal SEATS forecast span to three
/// seasonal years before trnaic.
fn seats_forecast_extensions(frequency: Frequency) -> Result<ForecastSpec, Error> {
    ForecastSpec::new(
        crate::regarima::ForecastHorizon::Periods(3 * u16::from(frequency.periods_per_year())),
        0,
    )
}

fn resolve_transform(
    estimation: &EstimationInput<'_>,
    request: RegArimaSpec,
    original: OriginalSeries<'_>,
) -> Result<ResolvedTransform, Error> {
    // The raised forecast span means trnaic's log-candidate Easter means
    // include three seasonal years, independently of the requested public
    // original-forecast prefix.
    let candidate_request = if matches!(
        request.transform(),
        crate::regarima::TransformSelection::Automatic(_)
    ) {
        request
            .clone()
            .with_forecast(seats_forecast_extensions(estimation.axis.frequency())?)
    } else {
        request.clone()
    };
    let mut resolved = TransformRequest {
        context: CandidateModelContext::ExplicitModel,
        decomposition: x11::Decomposition::SeasonalAdjustment,
        original,
    }
    .resolve(
        estimation.values,
        &estimation.missing,
        estimation.axis,
        candidate_request,
    )?;
    resolved.spec = request
        .resolve(resolved.spec.transform())?
        .with_flow_length_context(resolved.spec.flow_length_context());
    Ok(resolved)
}

/// regvar on the modelspan, and the estimation response; see the module
/// docs.
fn regression_response(
    estimation: &EstimationInput<'_>,
    spec: ResolvedRegArimaSpec,
    user_regressors: &[UserRegressor],
) -> Result<(RegressionModel, Vec<f64>), Error> {
    let (axis, count) = (estimation.axis, estimation.values.len());
    let mut regression = RegressionModel::from_spec(spec, axis, count)?;
    estimation.add_manual_outliers(&mut regression, spec)?;
    regression.add_user_regressors(user_regressors, axis, count)?;
    regression.add_missing(&estimation.missing, axis, count)?;
    let mut response = remove_fixed_length(estimation.values, regression.fixed_length());
    crate::engine::missing::reset(&mut response, &estimation.missing);
    Ok((regression, response))
}

/// Refuses a nonfinite value in a field `Forecasts::from_points` exposes. The
/// first lead with one is reported, and within it the first field in this
/// order.
fn validate_forecasts(points: &[crate::arima::ForecastPoint]) -> Result<(), Error> {
    use crate::api::SeatsForecastField as Field;
    for (index, point) in points.iter().enumerate() {
        let check = |field, value: f64| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(Error::from(NumericalDetail::SeatsForecast {
                    field,
                    index,
                    period: point.period,
                    value,
                }))
            }
        };
        for (field, value) in [
            (Field::Original, point.original),
            (Field::Transformed, point.transformed),
            (
                Field::TransformedStandardError,
                point.transformed_standard_error,
            ),
            (Field::Lower95, point.lower_95),
            (Field::Upper95, point.upper_95),
        ] {
            check(field, value)?;
        }
        if let Some(variance) = point.regression_variance {
            check(Field::RegressionVariance, variance)?;
            check(Field::StochasticVariance, point.stochastic_variance)?;
        }
    }
    Ok(())
}

/// The summary Q SEATS reads from an explicitly estimated fit: present only
/// when outlier search is configured.
fn summary_q(fit: &EstimatedRegArima) -> Result<SummaryQ, crate::seats::Error> {
    if fit.spec.outliers().is_none() {
        return Ok(SummaryQ::Absent);
    }
    box_pierce_q(fit, &fit.residuals)
}

/// The summary Q of automd's terminal fit, from the residuals it reports.
fn automatic_summary_q(
    terminal: &crate::automdl::TerminalFit,
) -> Result<SummaryQ, crate::seats::Error> {
    box_pierce_q(&terminal.fit, terminal.reported_residuals())
}

/// The regARIMA summary's Q statistic for SEATS: the Box-Pierce Q of the
/// residuals past the model's differencing loss.
fn box_pierce_q(
    fit: &EstimatedRegArima,
    residuals: &[f64],
) -> Result<SummaryQ, crate::seats::Error> {
    use crate::numerics::autocorrelation::{AutocorrelationError, sample_acf};
    let count = fit.transformed.len();
    // arima.f:1046: a nonconverged fit writes no summary Q.
    if !fit.report().status.converged {
        return Err(Failure::EstimationNotConverged.at(Stage::Import));
    }
    // The exact MA families retain AR initial conditions: the span of the
    // expanded differencing operator is the estimator's conditioning loss.
    let model = ExpandedModel::from_spec(fit.arma_workspace.model(), fit.axis.frequency());
    let loss = model.differencing_loss();
    let effective = count
        .checked_sub(loss)
        .filter(|n| *n > 1)
        .ok_or(Failure::AutocorrelationSample { count }.at(Stage::Import))?;
    let start = residuals.len().checked_sub(effective).ok_or(
        Failure::AutocorrelationSample {
            count: residuals.len(),
        }
        .at(Stage::Import),
    )?;
    let suffix = &residuals[start..];
    if let Some(index) = suffix.iter().position(|value| !value.is_finite()) {
        return Err(Failure::Unusable {
            quantity: "regARIMA residual",
            index: Some(start + index),
            value: suffix[index],
        }
        .at(Stage::Import));
    }
    // gtinpt sets 3s lags, prtacf clips them to n_eff-1, ANSUB9 imports min(clipped, 2s).
    let lag = (2 * usize::from(fit.axis.frequency().periods_per_year())).min(effective - 1);
    let acf = sample_acf(suffix, lag).map_err(|source| {
        match source {
            AutocorrelationError::ZeroVariance => Failure::Unusable {
                quantity: "regARIMA residual variance",
                index: None,
                value: 0.0,
            },
            AutocorrelationError::InvalidInput => {
                Failure::AutocorrelationSample { count: effective }
            }
            // `sample_acf` does not report the nonfinite value.
            AutocorrelationError::NonFinite => Failure::Unusable {
                quantity: "regARIMA residual autocorrelation",
                index: None,
                value: f64::NAN,
            },
        }
        .at(Stage::Import)
    })?;
    // acfdgn.f overwrites Ljung–Box with centered residual Box–Pierce.
    // pracf2's independent print/save/graphics guard is false on this path.
    // The fold starts from +0.0: `Iterator::sum` would start from -0.0.
    let mut sum = 0.0;
    for correlation in acf.values.correlations {
        sum += correlation * correlation;
    }
    let blqt = sum * effective as f64;
    if !blqt.is_finite() || blqt < 0.0 {
        return Err(Failure::Unusable {
            quantity: "Box-Pierce Q",
            index: None,
            value: blqt,
        }
        .at(Stage::Import));
    }
    Ok(SummaryQ::Present { blqt })
}

#[cfg(test)]
mod tests {
    use super::box_pierce_q;
    use crate::arima::{ArmaStart, estimate};
    use crate::regression::model::RegressionModel;
    use crate::seats::pipeline::SummaryQ;
    use crate::seats::{Failure, Stage};
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, RegArimaSpec};

    #[test]
    fn a_nonconverged_fit_has_no_summary_q() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let original = (0..120)
            .map(|t| {
                let angle = std::f64::consts::TAU * (t % 12) as f64 / 12.0;
                100.0 + 0.3 * t as f64 + 10.0 * angle.sin() + ((t * 37 % 101) as f64 - 50.0) * 0.05
            })
            .collect::<Vec<_>>();
        let order = ArimaOrder::new(0, 1, 1).unwrap();
        let spec = RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
            .with_constant(false)
            .resolve_explicit()
            .unwrap();
        let mut fit = estimate(
            &original,
            (&original).into(),
            axis,
            spec,
            RegressionModel::new(),
            ArmaStart::Default,
        )
        .unwrap();
        assert!(matches!(
            box_pierce_q(&fit, &fit.residuals),
            Ok(SummaryQ::Present { .. })
        ));
        fit.pre_covariance_status.converged = false;
        assert_eq!(
            box_pierce_q(&fit, &fit.residuals),
            Err(Failure::EstimationNotConverged.at(Stage::Import))
        );
    }
}
