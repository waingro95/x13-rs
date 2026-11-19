//! Fits for the automatic-model procedure, which carries `stpitr.f`'s
//! convergence flag from one fit to the next and keeps a fit that did not
//! converge instead of failing.

use super::attempt::validate_effective_length;
use super::policy::EstimationPolicy;
use super::{
    ArmaStart, EstimatedRegArima, FitAttempt, FitControl, FitReport, FitRequest, FitStatus,
    FittedNumericalView, FittedRegArima, finish_estimate, fit_transformed,
};
use crate::api::ResolvedRegArimaSpec;
use crate::arima::model::ExpandedModel;
use crate::regression::model::{RegressionModel, RegressorKind};
use crate::time::TimeAxis;
use crate::{Error, Transform, Warning};

/// The response of an automatic-model fit.
#[derive(Debug, Clone, Copy)]
pub(crate) enum AutomaticInput<'a> {
    /// The series before its transformation, with any length-of-period prior
    /// divided out.
    Untransformed(&'a [f64]),
    /// A response already on the transformed scale, fitted as given.
    Transformed(&'a [f64]),
}

impl<'a> AutomaticInput<'a> {
    pub(crate) fn values(self) -> &'a [f64] {
        match self {
            Self::Untransformed(values) | Self::Transformed(values) => values,
        }
    }
}

/// Fits `spec` to `input`. `incoming_convergence` is the previous fit's
/// convergence flag, which `stpitr.f` reads before it sets its own.
pub(crate) fn fit_automatic_input(
    input: AutomaticInput<'_>,
    original_series: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
    incoming_convergence: bool,
) -> Result<AutomaticFit, Error> {
    if let AutomaticInput::Transformed(response) = input {
        debug_assert_eq!(response.len(), original_series.raw().len());
        debug_assert!(
            regression_model
                .columns()
                .all(|(_, column)| column.len() == response.len())
        );
        validate_transformed_input(response, original_series, axis, spec)?;
    }
    let expanded = ExpandedModel::from_spec(spec.model(), axis.frequency());
    validate_effective_length(input.values().len(), &expanded)?;
    let transformed = match input {
        AutomaticInput::Untransformed(series) => {
            crate::transform::forward(series, spec.transform())?
        }
        AutomaticInput::Transformed(response) => response.to_vec(),
    };
    let request = FitRequest {
        original: original_series,
        axis,
        spec,
        regression_model,
        start,
    };
    let control = FitControl::carried(EstimationPolicy::EXPLICIT, incoming_convergence);
    let attempt = fit_transformed(transformed, request, control)?;
    finish_automatic_attempt(attempt)
}

/// The input checks `transform::forward` makes for an untransformed response.
fn validate_transformed_input(
    response: &[f64],
    original_series: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
) -> Result<(), Error> {
    if let Some(index) = response.iter().position(|value| !value.is_finite()) {
        return Err(Error::NonFiniteValue { index });
    }
    if let Some(index) = original_series
        .raw()
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(Error::NonFiniteValue { index });
    }
    axis.periods(response.len())?;
    if spec.transform() == Transform::Log
        && let Some(index) = crate::numerics::first_nonpositive(original_series.raw())
    {
        return Err(Error::NonPositiveValue { index });
    }
    Ok(())
}

/// An automatic-model fit, converged or not.
pub(crate) struct AutomaticFit {
    pre_covariance_status: FitStatus,
    effective_observations: usize,
    state: AutomaticFitState,
}

enum AutomaticFitState {
    Converged(Box<EstimatedRegArima>),
    /// The fit as `rgarma.f` left it: Build 62 skips the covariance step.
    Unconverged {
        fitted: Box<FittedRegArima>,
        regressor_kinds: Vec<RegressorKind>,
    },
}

/// [`fit_automatic_input`] for an untransformed series.
pub(crate) fn fit_automatic(
    series: &[f64],
    original_series: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
    incoming_convergence: bool,
) -> Result<AutomaticFit, Error> {
    fit_automatic_input(
        AutomaticInput::Untransformed(series),
        original_series,
        axis,
        spec,
        regression_model,
        start,
        incoming_convergence,
    )
}

fn finish_automatic_attempt(attempt: FitAttempt) -> Result<AutomaticFit, Error> {
    let pre_covariance_status = attempt.fitted.status;
    let effective_observations = attempt.fitted.effective_observations;
    let state = if pre_covariance_status.converged {
        AutomaticFitState::Converged(Box::new(finish_estimate(*attempt.fitted)?))
    } else {
        let regressor_kinds = attempt.fitted.regression_model.kinds().collect();
        AutomaticFitState::Unconverged {
            fitted: attempt.fitted,
            regressor_kinds,
        }
    };
    Ok(AutomaticFit {
        pre_covariance_status,
        effective_observations,
        state,
    })
}

impl AutomaticFit {
    pub(crate) fn report(&self) -> FitReport {
        match &self.state {
            AutomaticFitState::Converged(fit) => fit.report(),
            AutomaticFitState::Unconverged { fitted, .. } => FitReport {
                status: fitted.status,
                covariance_diagnostic: super::ArmaCovarianceDiagnostic::None,
                nonlinear_iterations: fitted.nonlinear_iterations,
                arma_parameter_count: fitted.parameters.len(),
            },
        }
    }

    pub(crate) fn spec(&self) -> ResolvedRegArimaSpec {
        match &self.state {
            AutomaticFitState::Converged(fit) => fit.spec,
            AutomaticFitState::Unconverged { fitted, .. } => fitted.spec,
        }
    }

    pub(crate) fn numerical_view(&self) -> FittedNumericalView<'_> {
        match &self.state {
            AutomaticFitState::Converged(fit) => fit.numerical_view(),
            AutomaticFitState::Unconverged {
                fitted,
                regressor_kinds,
            } => FittedNumericalView {
                axis: fitted.axis,
                transformed: &fitted.transformed,
                parameters: &fitted.parameters,
                arma_workspace: &fitted.arma_workspace,
                arma_covariance_diagnostic: super::ArmaCovarianceDiagnostic::None,
                residuals: &fitted.final_stage.output_residuals,
                regression_coefficients: fitted.final_stage.regression.coefficients(),
                regressor_kinds,
                outlier_test_factor: fitted.final_stage.regression.packed_factor(),
                filter_workspace: &fitted.filter_workspace,
            },
        }
    }

    pub(crate) fn pre_covariance_status(&self) -> FitStatus {
        self.pre_covariance_status
    }

    pub(crate) fn effective_observations(&self) -> usize {
        self.effective_observations
    }

    pub(crate) fn innovation_variance(&self) -> f64 {
        match &self.state {
            AutomaticFitState::Converged(fit) => fit.innovation_variance,
            AutomaticFitState::Unconverged { fitted, .. } => fitted.innovation_variance,
        }
    }

    pub(crate) fn regression_model(&self) -> &RegressionModel {
        match &self.state {
            AutomaticFitState::Converged(fit) => &fit.regression_model,
            AutomaticFitState::Unconverged { fitted, .. } => &fitted.regression_model,
        }
    }

    pub(crate) fn nonlinear_iterations(&self) -> usize {
        match &self.state {
            AutomaticFitState::Converged(fit) => fit.nonlinear_iterations,
            AutomaticFitState::Unconverged { fitted, .. } => fitted.nonlinear_iterations,
        }
    }

    pub(crate) fn warnings_mut(&mut self) -> &mut Vec<Warning> {
        match &mut self.state {
            AutomaticFitState::Converged(fit) => &mut fit.warnings,
            AutomaticFitState::Unconverged { fitted, .. } => &mut fitted.state.warnings,
        }
    }

    pub(crate) fn warnings(&self) -> &[Warning] {
        match &self.state {
            AutomaticFitState::Converged(fit) => &fit.warnings,
            AutomaticFitState::Unconverged { fitted, .. } => &fitted.state.warnings,
        }
    }

    /// The finished fit, if it converged.
    pub(crate) fn as_converged(&self) -> Option<&EstimatedRegArima> {
        match &self.state {
            AutomaticFitState::Converged(fit) => Some(fit),
            AutomaticFitState::Unconverged { .. } => None,
        }
    }

    /// The finished fit, or `self` back if it did not converge.
    pub(crate) fn into_converged(self) -> Result<EstimatedRegArima, Self> {
        match self.state {
            AutomaticFitState::Converged(fit) => Ok(*fit),
            AutomaticFitState::Unconverged {
                fitted,
                regressor_kinds,
            } => Err(Self {
                state: AutomaticFitState::Unconverged {
                    fitted,
                    regressor_kinds,
                },
                ..self
            }),
        }
    }
}

#[cfg(test)]
mod tests;
