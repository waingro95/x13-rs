//! Fits for `iddiff.f`'s differencing identification: the working ARMA
//! models, with a conditional-AR retry when the exact fit does not converge,
//! and the constant-only model behind the mean test.

use super::super::model::ExpandedModel;
use super::attempt::validate_effective_length;
use super::policy::EstimationPolicy;
use super::status::FitDiagnostic;
use super::{
    ArmaStart, EstimatedRegArima, FitAttempt, FitControl, FitInitialization, FitReport,
    FitReportDiagnostic, FitRequest, finish_estimate, fit_attempt, starting_parameters,
};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{ArimaModel, Error, RegArimaSpec, Transform, Warning};

/// The overall iteration limit of the mean-test fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeanFitBudget {
    Exact200,
    Exact1500,
}

/// A converged fit of one differencing working model.
#[derive(Debug, Clone)]
pub(crate) struct IdentificationFit {
    pub(crate) parameters: Vec<f64>,
    /// The exact fit did not converge and the conditional-AR retry did.
    pub(crate) retried_conditional: bool,
    /// The fit carries an estimation or covariance diagnostic.
    pub(crate) has_diagnostic: bool,
    pub(crate) report: FitReport,
    pub(crate) warnings: Vec<Warning>,
}

impl IdentificationFit {
    /// `exact_warnings` are those of an exact attempt the conditional retry
    /// replaced; they are reported first.
    fn new(result: EstimatedRegArima, exact_warnings: Option<Vec<Warning>>) -> Self {
        let report = result.report();
        let retried_conditional = exact_warnings.is_some();
        let mut warnings = exact_warnings.unwrap_or_default();
        warnings.extend(result.warnings);
        Self {
            parameters: result.parameters,
            retried_conditional,
            has_diagnostic: report.diagnostic()
                != FitReportDiagnostic::Estimation(FitDiagnostic::None),
            report,
            warnings,
        }
    }
}

/// The constant-only fit `iddiff.f` tests the mean with.
#[derive(Debug, Clone)]
pub(crate) struct MeanFit {
    pub(crate) coefficient: f64,
    pub(crate) packed_factor: Vec<f64>,
    pub(crate) innovation_variance: f64,
    pub(crate) report: FitReport,
    pub(crate) warnings: Vec<Warning>,
}

/// Fits a differencing working model to `series` from `starts`.
pub(crate) fn fit_prepared_identification(
    series: &[f64],
    axis: TimeAxis,
    model: ArimaModel,
    starts: Vec<f64>,
) -> Result<IdentificationFit, Error> {
    let spec = RegArimaSpec::new(model)
        .with_transform(Transform::None)
        .with_constant(false)
        .resolve_explicit()?;
    let regression = RegressionModel::from_spec(spec, axis, series.len())?;
    let request = FitRequest {
        original: crate::arima::FitContext::unadjusted(series),
        axis,
        spec,
        regression_model: regression,
        start: ArmaStart::Previous(starts),
    };
    fit_identification(series, request)
}

/// Fits `model` with a constant to `series` for the mean test.
pub(crate) fn fit_local_mean(
    series: &[f64],
    axis: TimeAxis,
    model: ArimaModel,
    budget: MeanFitBudget,
) -> Result<MeanFit, Error> {
    let spec = RegArimaSpec::new(model)
        .with_transform(Transform::None)
        .with_constant(true)
        .resolve_explicit()?;
    let regression = RegressionModel::from_spec(spec, axis, series.len())?;
    let expanded = ExpandedModel::from_spec(model, axis.frequency());
    let parameter_count = expanded.parameter_slots().len();
    let policy = match budget {
        MeanFitBudget::Exact200 => EstimationPolicy::AUTOMATIC_IDENTIFICATION,
        MeanFitBudget::Exact1500 => EstimationPolicy::AUTOMATIC_SEARCH,
    };
    let request = FitRequest {
        original: crate::arima::FitContext::unadjusted(series),
        axis,
        spec,
        regression_model: regression,
        start: ArmaStart::Previous(vec![0.1; parameter_count]),
    };
    // The exact filter indexes out of bounds when the differenced sample is
    // shorter than the AR and MA spans.
    validate_effective_length(series.len(), &expanded)?;
    let attempt = fit_attempt(series, request, FitControl::fresh(policy))?;
    if !attempt.fitted.status.converged {
        return Err(Error::EstimationDidNotConverge {
            iterations: attempt.fitted.nonlinear_iterations,
        });
    }
    let result = finish_estimate(*attempt.fitted)?;
    let report = result.report();
    Ok(MeanFit {
        coefficient: result.regression_coefficients[0],
        packed_factor: result.regression_factor,
        innovation_variance: result.innovation_variance,
        report,
        warnings: result.warnings,
    })
}

fn fit_identification(series: &[f64], request: FitRequest<'_>) -> Result<IdentificationFit, Error> {
    let model = ExpandedModel::from_spec(request.spec.model(), request.axis.frequency());
    // An invalid previous start is reported before a short series.
    starting_parameters(request.start.clone(), &model)?;
    validate_effective_length(series.len(), &model)?;
    let exact = fit_attempt(
        series,
        request.clone(),
        FitControl::fresh(EstimationPolicy::AUTOMATIC_IDENTIFICATION),
    )?;
    if exact.fitted.status.converged {
        return Ok(IdentificationFit::new(
            finish_estimate(*exact.fitted)?,
            None,
        ));
    }
    let conditional = retry_conditional(series, request, &exact)?;
    if !conditional.fitted.status.converged {
        return Err(Error::EstimationDidNotConverge {
            iterations: conditional.fitted.nonlinear_iterations,
        });
    }
    Ok(IdentificationFit::new(
        finish_estimate(*conditional.fitted)?,
        Some(exact.fitted.state.warnings),
    ))
}

/// The conditional-AR retry after an exact fit that did not converge: it
/// starts from the exact fit's parameters and workspace, and carries its
/// determinant when the model has no MA part.
fn retry_conditional(
    series: &[f64],
    request: FitRequest<'_>,
    exact: &FitAttempt,
) -> Result<FitAttempt, Error> {
    let request = FitRequest {
        start: ArmaStart::PreviousWithWorkspace {
            parameters: exact.fitted.parameters.clone(),
            workspace: exact.fitted.arma_workspace.clone(),
        },
        ..request
    };
    let control = FitControl {
        initialization: FitInitialization::AfterExact {
            raw_log_determinant_correction: exact.fitted.final_stage.raw_log_determinant_correction,
        },
        ..FitControl::fresh(EstimationPolicy::CONDITIONAL_IDENTIFICATION)
    };
    Ok(fit_attempt(series, request, control)?)
}

#[cfg(test)]
mod tests;
