//! `automd.f`'s final coefficient check: insignificant ARMA terms are
//! dropped one at a time, and dropping one of several can send the selector
//! back to the outlier search or the identification.

use super::coefficients::{ArmaVariances, check_coefficients};
use super::fallback::{FallbackFit, FallbackState, FinalChecks, LoopStage, RetryRoute};
use super::outlier_policy::reduced_outlier_cutoffs;
use super::refit::Refit;
use crate::arima::{ArmaCovarianceDiagnostic, ArmaStart};
use crate::regarima::OutlierSpec;
use crate::regression::model::RegressorKind;
use crate::{Error, Warning};

pub(crate) enum CoefficientOutcome {
    /// No insignificant coefficient is left: the selection is complete.
    Checked(FallbackState),
    /// `automd.f` goes back to the identification or the outlier search.
    Retry(CoefficientRetry),
}

/// A restart after the reduced model's refit.
pub(crate) struct CoefficientRetry {
    pub(crate) state: FallbackState,
    pub(crate) route: RetryRoute,
    /// The pass the restart enters: the second or the final.
    pub(crate) stage: LoopStage,
    pub(crate) policy: OutlierSpec,
}

pub(crate) fn check_final_coefficients(
    checks: FinalChecks,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<CoefficientOutcome, Error> {
    let FinalChecks { mut state, stage } = checks;
    debug_assert!(state.is_consistent());
    let spec = state.fit.spec();
    loop {
        let view = state.fit.numerical_view();
        let model = state.current.model;
        debug_assert!(view.arma_workspace.model() == model);
        let variances = (view.arma_covariance_diagnostic != ArmaCovarianceDiagnostic::Singular)
            .then(|| ArmaVariances {
                innovation_variance: state.fit.innovation_variance(),
                diagonal: view.arma_workspace.diagonal(),
            });
        let check = check_coefficients(model, original.raw().len(), view.parameters, variances)?;
        let Some(reduced_model) = check.reduced_model else {
            return Ok(CoefficientOutcome::Checked(state));
        };
        // `automd.f` restarts when more than one coefficient is
        // insignificant, outliers are searched, `nloop.le.2` and the
        // critical values can still be reduced.
        let restart = match spec.outliers() {
            Some(policy) if check.insignificant_count > 1 => stage.next().and_then(|next| {
                reduced_outlier_cutoffs(state.active_cutoffs).map(|cutoffs| (next, cutoffs, policy))
            }),
            _ => None,
        };
        if let Some((_, cutoffs, _)) = restart {
            state.active_cutoffs = cutoffs;
        }
        let response = view.transformed.to_vec();
        let regression = state.fit.regression_model().clone();
        let outlier_regressors = regression
            .kinds()
            .filter_map(|kind| match kind {
                RegressorKind::Outlier { kind, period, .. } => Some((kind, period)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let fit = Refit {
            response: &response,
            original,
            axis: view.axis,
            spec: spec.with_model(reduced_model),
            regression,
            start: ArmaStart::Default,
            carried_convergence: state.fit.converged(),
        }
        .fit_converged(warnings)?;
        debug_assert!(outlier_regressors.iter().all(|regressor| {
            state
                .selected_outliers
                .iter()
                .any(|selected| *regressor == (selected.kind, selected.period))
        }));
        // After `clrotl` the selection can still list outliers whose
        // regressors are gone.
        state
            .selected_outliers
            .retain(|selected| outlier_regressors.contains(&(selected.kind, selected.period)));
        state.fit = FallbackFit::Existing(Box::new(fit));
        state.residual_override = None;
        state.current.model = reduced_model;
        if let Some((stage, _, policy)) = restart {
            let route = if state.current.automatic_outlier_count > 0 {
                RetryRoute::Identification
            } else {
                RetryRoute::OutlierSearch
            };
            return Ok(CoefficientOutcome::Retry(CoefficientRetry {
                state,
                route,
                stage,
                policy,
            }));
        }
    }
}
