//! `automd.f`'s return to an earlier step after a rejection: a new outlier
//! search with the current critical values, after a new identification when
//! the rejected model has automatic outliers.

use super::differencing::DifferencingState;
use super::fallback::{
    ComparedModel, FallbackFit, FallbackState, FinalChecks, LoopStage, OutlierRetry, RetryRoute,
};
use super::final_coefficients::CoefficientRetry;
use super::first_pass::FirstPass;
use super::outlier_policy::active_outlier_policy;
use super::preparation::prepare_identification;
use super::restoration::{OutlierRestoration, restore_identified};
use super::search::search_prepared;
use crate::arima::{AutomaticInput, EstimatedRegArima, PreparedSearchInput};
use crate::outliers::search::{
    SelectedOutlier, search_automatic_finalized_input, search_automatic_residuals,
};
use crate::regarima::OutlierSpec;
use crate::{Error, Warning};

/// A fit `bkdfmd` saved, with its automatic outliers.
pub(crate) struct SavedFit {
    pub(crate) fit: EstimatedRegArima,
    pub(crate) selected_outliers: Vec<SelectedOutlier>,
}

/// A reentered model awaiting `pass2.f`'s second comparison.
pub(crate) struct ReenteredComparison {
    pub(crate) state: FallbackState,
    /// The fit `bkdfmd` saved before the reentry.
    pub(crate) saved: Box<SavedFit>,
    pub(crate) policy: OutlierSpec,
}

/// Where a coefficient restart continues after its reentry.
pub(crate) enum AfterCoefficientReentry {
    /// A restart into pass two compares the reentered model.
    Compare(ReenteredComparison),
    /// A restart into the final pass goes straight to the final checks.
    FinalChecks(FinalChecks),
}

/// Reenters after the first comparison's residual check rejected the model.
pub(crate) fn reenter_first_outlier_retry(
    first: &FirstPass,
    retry: OutlierRetry,
    original: crate::arima::FitContext<'_>,
    differencing_state: &mut DifferencingState,
    warnings: &mut Vec<Warning>,
) -> Result<ReenteredComparison, Error> {
    reenter(first, retry, false, original, differencing_state, warnings)
}

/// Reenters after the final coefficient check dropped insignificant terms.
pub(crate) fn reenter_coefficient_retry(
    first: &FirstPass,
    retry: CoefficientRetry,
    original: crate::arima::FitContext<'_>,
    differencing_state: &mut DifferencingState,
    warnings: &mut Vec<Warning>,
) -> Result<AfterCoefficientReentry, Error> {
    let CoefficientRetry {
        state,
        route,
        stage,
        policy,
    } = retry;
    let reentered = reenter(
        first,
        OutlierRetry {
            state,
            route,
            policy,
        },
        true,
        original,
        differencing_state,
        warnings,
    )?;
    // `stage` is the pass being entered: the second or the final.
    Ok(if stage == LoopStage::Second {
        AfterCoefficientReentry::Compare(reentered)
    } else {
        AfterCoefficientReentry::FinalChecks(FinalChecks {
            state: reentered.state,
            stage,
        })
    })
}

fn reenter(
    first: &FirstPass,
    retry: OutlierRetry,
    after_coefficients: bool,
    original: crate::arima::FitContext<'_>,
    differencing_state: &mut DifferencingState,
    warnings: &mut Vec<Warning>,
) -> Result<ReenteredComparison, Error> {
    let OutlierRetry {
        state: pending,
        route,
        policy,
    } = retry;
    debug_assert!(pending.is_consistent());
    let n = original.raw().len();

    // `automd.f` calls `bkdfmd` on its way back to the identification,
    // before it skips the identification for a model without automatic
    // outliers. Going back to the outlier search alone leaves the baseline
    // saved.
    let saved = Box::new(
        if after_coefficients || route == RetryRoute::Identification {
            SavedFit {
                fit: pending
                    .fit
                    .as_converged()
                    .ok_or(Error::internal(
                        "model selection saved a fit that did not converge",
                    ))?
                    .clone(),
                selected_outliers: pending.selected_outliers.clone(),
            }
        } else {
            SavedFit {
                fit: first.baseline.fit.clone(),
                selected_outliers: first.baseline.selected_outliers.clone(),
            }
        },
    );
    let active_policy = active_outlier_policy(policy, pending.active_cutoffs)?;
    let FallbackState {
        fit: previous_fit,
        residual_override,
        selected_outliers,
        current,
        reference,
        residual_mean_t,
        original_cutoffs,
        active_cutoffs,
    } = pending;
    let model;
    let procedural_mean;
    let fit;
    let selected;
    let residual_override =
        if route == RetryRoute::Identification && current.automatic_outlier_count > 0 {
            let source = &saved.fit;
            let prepared = prepare_identification(source)?;
            let identification =
                differencing_state.identify_prepared(&prepared.values, source.axis, warnings)?;
            let input = PreparedSearchInput::new(
                &prepared.values,
                original,
                source.axis,
                source.spec.transform(),
                source.regression_model.fixed_length(),
            )?;
            let search = search_prepared(&input, &identification, warnings)?;
            model = search.model.into();
            procedural_mean = search.mean_significant;
            let restored = restore_identified(
                source,
                &prepared,
                &search,
                original,
                OutlierRestoration {
                    previous: &selected_outliers,
                    // `automd.f` clears the automatic outliers when the
                    // reference model has some and `pass2.f` left `igo` at zero,
                    // which it has after the final checks but not after a
                    // residual rejection.
                    clear_previous: after_coefficients && reference.automatic_outlier_count > 0,
                    spec: active_policy,
                },
                warnings,
            )?;
            fit = restored.fit;
            selected = restored.selected_outliers;
            None
        } else {
            model = current.model;
            procedural_mean = current.procedural_mean;
            let response = previous_fit.numerical_view().transformed.to_vec();
            let (searched, residual_override) = match previous_fit {
                FallbackFit::Existing(fit) => {
                    debug_assert!(residual_override.is_none());
                    let searched = search_automatic_finalized_input(
                        AutomaticInput::Transformed(&response),
                        original,
                        *fit,
                        active_policy,
                        selected_outliers,
                    )?;
                    (searched, None)
                }
                FallbackFit::Refit(fit) => {
                    let searched = search_automatic_residuals(
                        AutomaticInput::Transformed(&response),
                        original,
                        fit,
                        residual_override.as_deref(),
                        active_policy,
                        selected_outliers,
                    )?;
                    (searched.result, searched.residual_override)
                }
                // `clrotl` runs only where no retry can follow.
                FallbackFit::ClearedRegression(_) => {
                    return Err(Error::internal(
                        "model selection retried a fit whose regression was cleared",
                    ));
                }
            };
            warnings.extend_from_slice(&searched.fit.warnings);
            fit = searched.fit;
            selected = searched.selected;
            residual_override
        };
    let mut state = FallbackState {
        fit: FallbackFit::Existing(Box::new(fit)),
        residual_override,
        current: ComparedModel {
            model,
            procedural_mean,
            automatic_outlier_count: selected.len(),
            metrics: current.metrics,
        },
        selected_outliers: selected,
        reference,
        residual_mean_t,
        original_cutoffs,
        active_cutoffs,
    };
    state.recheck(n)?;
    Ok(ReenteredComparison {
        state,
        saved,
        policy,
    })
}
