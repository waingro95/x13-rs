//! `pass2.f`'s comparison of a reentered model.

use super::{FallbackFit, FinalChecks, LoopStage, SavedReference, compare, fit_last_resort};
use crate::arima::AutomaticInput;
use crate::automdl::first_pass::FirstPass;
use crate::automdl::reentry::ReenteredComparison;
use crate::outliers::search::{
    ClearedRegressionFit, search_automatic_cleared_input, search_automatic_finalized_input,
};
use crate::{Error, Warning};

/// `pass2.f`'s second comparison, of a reentered model with the reference:
/// the survivor is accepted or replaced by the last resort.
pub(crate) fn second_fallback(
    first: &FirstPass,
    reentered: ReenteredComparison,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<FinalChecks, Error> {
    let ReenteredComparison {
        mut state,
        saved,
        policy,
    } = reentered;
    debug_assert!(state.is_consistent());
    let n = original.raw().len();
    let spec = state.fit.spec();
    let comparison = compare(
        &mut state,
        SavedReference {
            fit: &saved.fit,
            selected_outliers: &saved.selected_outliers,
            residuals: &first.baseline.fit.residuals,
        },
        LoopStage::Second,
        original,
        warnings,
    )?;
    if comparison.accepted {
        return Ok(FinalChecks {
            state,
            stage: LoopStage::Second,
        });
    }
    let response = state.fit.numerical_view().transformed.to_vec();
    let fitted = fit_last_resort(&mut state, original, spec, warnings)?;
    state.active_cutoffs = state.original_cutoffs;
    state.selected_outliers = if state.current.automatic_outlier_count > 0 {
        let cleared = ClearedRegressionFit::clear_automatic_outliers(fitted)?;
        let search = search_automatic_cleared_input(
            AutomaticInput::Transformed(&response),
            original,
            cleared,
            policy,
        )?;
        warnings.extend_from_slice(&search.fit.last_fitted().warnings);
        state.fit = FallbackFit::ClearedRegression(search.fit);
        search.selected
    } else {
        let search = search_automatic_finalized_input(
            AutomaticInput::Transformed(&response),
            original,
            fitted,
            policy,
            state.selected_outliers.clone(),
        )?;
        warnings.extend_from_slice(&search.fit.warnings);
        state.fit = FallbackFit::Existing(Box::new(search.fit));
        search.selected
    };
    state.current.automatic_outlier_count = state.selected_outliers.len();
    state.residual_override = None;
    state.recheck(n)?;
    // `pass2.f` sets `Nloop=3` after its last-resort model: no retry follows.
    Ok(FinalChecks {
        state,
        stage: LoopStage::Final,
    })
}
