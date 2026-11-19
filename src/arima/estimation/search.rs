//! Fits for `amdid.f`'s ARMA order search: each candidate model on the
//! differenced series, then the chosen model again under the explicit-fit
//! tolerances.

use super::attempt::validate_effective_length;
use super::policy::EstimationPolicy;
use super::status::FitDiagnostic;
use super::{
    ArmaStart, AttemptFailure, EstimatedRegArima, FitAttempt, FitControl, FitRequest,
    append_dynamic_fit_warnings, finish_estimate, fit_attempt,
};
use crate::api::result::ArmaCovariance;
use crate::arima::exact::ExactFilterError;
use crate::arima::likelihood::{FitLikelihood, likelihood_statistics};
use crate::arima::model::ExpandedModel;
use crate::error::{NumericalDetail, NumericalOperation};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{ArimaModel, Error, RegArimaSpec, Transform, Warning};

/// The series the search fits, with what the reported BIC's transformation
/// adjustment needs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedSearchInput<'a> {
    pub(crate) prepared: &'a [f64],
    original: crate::arima::FitContext<'a>,
    pub(crate) axis: TimeAxis,
    transform: Transform,
    prior_factors: &'a [f64],
}

impl<'a> PreparedSearchInput<'a> {
    /// Fails if the series outruns the axis, or if removing the regression
    /// effects left a nonfinite value in `prepared`.
    pub(crate) fn new(
        prepared: &'a [f64],
        original: crate::arima::FitContext<'a>,
        axis: TimeAxis,
        transform: Transform,
        prior_factors: &'a [f64],
    ) -> Result<Self, Error> {
        debug_assert_eq!(prepared.len(), original.raw().len());
        debug_assert!(prior_factors.is_empty() || prior_factors.len() == prepared.len());
        axis.periods(prepared.len())?;
        if prepared.iter().any(|value| !value.is_finite()) {
            return Err(Error::numerical(NumericalOperation::Estimation));
        }
        Ok(Self {
            prepared,
            original,
            axis,
            transform,
            prior_factors,
        })
    }
}

/// One candidate of the order search.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CandidateFit {
    /// `amdid.f`'s BIC2: the unadjusted BIC over the effective observations.
    /// `None` leaves the candidate out of the ranking: its fit did not
    /// converge, carries a diagnostic or a singular covariance, or its exact
    /// filter found a singular initial covariance.
    pub(crate) bic2: Option<f64>,
    pub(crate) warnings: Vec<Warning>,
}

impl CandidateFit {
    fn unranked(fit_warnings: &[Warning]) -> Self {
        let mut warnings = Vec::new();
        append_dynamic_fit_warnings(&mut warnings, fit_warnings);
        Self {
            bic2: None,
            warnings,
        }
    }
}

/// One search fit. `carried_convergence` is the previous search fit's flag.
fn search_attempt(
    input: &PreparedSearchInput<'_>,
    model: ArimaModel,
    policy: EstimationPolicy,
    carried_convergence: bool,
) -> Result<FitAttempt, AttemptFailure> {
    let spec = RegArimaSpec::new(model)
        .with_transform(Transform::None)
        .with_constant(false)
        .resolve_explicit()?;
    let expanded = ExpandedModel::from_spec(model, input.axis.frequency());
    validate_effective_length(input.prepared.len(), &expanded)?;
    let request = FitRequest {
        original: crate::arima::FitContext::unadjusted(input.prepared),
        axis: input.axis,
        spec,
        regression_model: RegressionModel::new(),
        start: ArmaStart::Default,
    };
    fit_attempt(
        input.prepared,
        request,
        FitControl::carried(policy, carried_convergence),
    )
}

/// A candidate whose exact filter found a singular or invalid initial
/// covariance is left unranked; any other failure ends the search.
fn unranked_for_filter_failure(
    error: AttemptFailure,
    carried_convergence: &mut bool,
) -> Result<CandidateFit, Error> {
    if let AttemptFailure::Filter {
        convergence_after_check: Some(converged),
        ..
    } = &error
    {
        *carried_convergence = *converged;
    }
    match error {
        AttemptFailure::Filter {
            cause:
                ExactFilterError::SingularMaInitialCovariance { .. }
                | ExactFilterError::SingularArInitialCovariance { .. }
                | ExactFilterError::InvalidArCovariance,
            warnings,
            ..
        } => Ok(CandidateFit::unranked(&warnings)),
        other => Err(other.into_public()),
    }
}

/// Fits one candidate of the order search and scores it.
///
/// `carried_convergence` starts `true` and passes from each search fit to
/// the next.
pub(crate) fn fit_prepared_candidate(
    input: &PreparedSearchInput<'_>,
    model: ArimaModel,
    carried_convergence: &mut bool,
) -> Result<CandidateFit, Error> {
    let attempt = match search_attempt(
        input,
        model,
        EstimationPolicy::AUTOMATIC_SEARCH,
        *carried_convergence,
    ) {
        Ok(attempt) => attempt,
        Err(error) => return unranked_for_filter_failure(error, carried_convergence),
    };
    *carried_convergence = attempt.fitted.status.converged;
    score_candidate(input, attempt)
}

/// Scores a search fit, or leaves it unranked.
fn score_candidate(
    input: &PreparedSearchInput<'_>,
    attempt: FitAttempt,
) -> Result<CandidateFit, Error> {
    let status = attempt.fitted.status;
    let candidate = attempt.fitted.spec.model();
    if !status.converged {
        return Ok(CandidateFit::unranked(&attempt.fitted.state.warnings));
    }
    let effective_observations = attempt.fitted.effective_observations;
    let parameter_count = attempt.fitted.parameters.len() + 1;
    let innovation_variance = attempt.fitted.innovation_variance;
    let determinant = attempt
        .fitted
        .final_stage
        .likelihood_log_determinant_correction;
    let unadjusted = attempt
        .fitted
        .likelihood
        .and_then(FitLikelihood::exact_statistics);
    let completed = finish_estimate(*attempt.fitted)?;
    if status.diagnostic != FitDiagnostic::None
        || completed.arma_covariance == ArmaCovariance::Singular
    {
        return Ok(CandidateFit::unranked(&completed.warnings));
    }
    // `rgarma.f` computes the covariance before `amdid2.f` ranks candidates,
    // so a converged fit without a score (zero variance, say) is left out for
    // a diagnostic or a singular covariance before its score is missed.
    let unadjusted = unadjusted.ok_or(Error::from(NumericalDetail::ModelSelectionUnavailable {
        candidate,
    }))?;
    let adjustment = input.original.transformation_adjustment(
        input.prior_factors,
        input.prepared.len() - effective_observations,
        input.transform,
    )?;
    let report_bic = likelihood_statistics(
        effective_observations,
        parameter_count,
        determinant,
        innovation_variance,
        adjustment,
    )
    .bic;
    let bic2 = unadjusted.bic / effective_observations as f64;
    if !report_bic.is_finite() || !bic2.is_finite() {
        return Err(Error::from(NumericalDetail::ModelSelectionUnavailable {
            candidate,
        }));
    }
    let mut warnings = Vec::new();
    append_dynamic_fit_warnings(&mut warnings, &completed.warnings);
    Ok(CandidateFit {
        bic2: Some(bic2),
        warnings,
    })
}

/// Refits the model the search chose, under the explicit-fit tolerances.
pub(crate) fn refit_prepared_selected(
    input: &PreparedSearchInput<'_>,
    model: ArimaModel,
    carried_convergence: &mut bool,
) -> Result<EstimatedRegArima, Error> {
    let attempt = search_attempt(
        input,
        model,
        EstimationPolicy::EXPLICIT,
        *carried_convergence,
    )
    .map_err(AttemptFailure::into_public)?;
    *carried_convergence = attempt.fitted.status.converged;
    finish_selected(attempt)
}

fn finish_selected(attempt: FitAttempt) -> Result<EstimatedRegArima, Error> {
    if !attempt.fitted.status.converged {
        // Build 62 automdl: "Estimation failed to converge for automatically
        // identified model".
        return Err(Error::EstimationDidNotConverge {
            iterations: attempt.fitted.nonlinear_iterations,
        });
    }
    finish_estimate(*attempt.fitted)
}

#[cfg(test)]
mod tests;
