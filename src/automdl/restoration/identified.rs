//! `automd.f`'s `addfix(...,2)`: after an order search, the chosen model
//! gets the saved regression back, is refitted if that changes anything, and
//! goes through an outlier search.
//!
//! The first pass's same-baseline path does not come here: there `restor`
//! recovers the state `ssprep` saved, baseline column order included. Reentry
//! has no `restor`, so it reorders the columns even when the model it chose
//! equals the reference. `addusr` also restarts the user regressors'
//! coefficients from a backup, which has no counterpart here because a fit
//! takes no regression-coefficient starts.

use super::{recompute_likelihood_statistics, restored_response};
use crate::arima::{AutomaticInput, EstimatedRegArima};
use crate::automdl::preparation::PreparedIdentification;
use crate::automdl::refit::Refit;
use crate::automdl::search::PreparedSearchResult;
use crate::outliers::search::{SelectedOutlier, search_automatic_finalized_input};
use crate::regarima::OutlierSpec;
use crate::regression::model::RegressorKind;
use crate::{Error, Warning};

/// What a restoration does about automatic outliers.
pub(crate) struct OutlierRestoration<'a> {
    /// The automatic outliers the previous pass selected.
    pub(crate) previous: &'a [SelectedOutlier],
    /// Whether `automd.f` removes them from the regression before the new
    /// search.
    pub(crate) clear_previous: bool,
    /// The settings of the new outlier search.
    pub(crate) spec: OutlierSpec,
}

/// A model fitted with its regression and its automatic outliers.
pub(crate) struct RestoredFit {
    pub(crate) fit: EstimatedRegArima,
    pub(crate) selected_outliers: Vec<SelectedOutlier>,
}

/// Fits the model `search` chose with the regression of `saved`, then
/// searches it for outliers.
pub(crate) fn restore_identified(
    saved: &EstimatedRegArima,
    prepared: &PreparedIdentification,
    search: &PreparedSearchResult,
    original: crate::arima::FitContext<'_>,
    outliers: OutlierRestoration<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<RestoredFit, Error> {
    let n = original.raw().len();
    let axis = saved.axis;
    let mean = search.mean_significant;
    let had_constant = saved.regressor_kinds.contains(&RegressorKind::Constant);
    let response = restored_response(prepared);
    let mut regression = saved.regression_model.clone();
    // `addfix(...,2)` re-appends the user groups (`addusr`) after the
    // missing-value and constant columns, and column order changes the GLS
    // arithmetic.
    regression.move_user_regressors_last();
    if !mean {
        regression.set_constant(false, search.model, axis.frequency(), n)?;
    }
    // `lester` is decided after an insignificant constant is removed and
    // before the automatic outliers are cleared.
    let mut refit = regression.column_count() > 0;
    if outliers.clear_previous {
        for outlier in outliers.previous {
            let removed = regression.remove_outlier(outlier.kind, outlier.period);
            debug_assert!(removed, "a selected outlier is missing from the regression");
        }
    }
    regression.set_constant(mean, search.model, axis.frequency(), n)?;
    if mean && !had_constant {
        refit = true;
    }
    regression = regression.evaluated_over(axis, n)?;
    let spec = saved.spec.with_model(search.model).with_constant(mean);
    let fit = if refit {
        Refit {
            response: &response,
            original,
            axis,
            spec,
            regression,
            start: search.fit.numerical_view().refit_start(),
            carried_convergence: true,
        }
        .fit_converged(warnings)?
    } else {
        let mut fit = search.fit.clone();
        fit.spec = spec;
        fit.transformed = response.clone();
        fit.regression_model = regression;
        recompute_likelihood_statistics(&mut fit, original)?;
        fit
    };
    let selected = if outliers.clear_previous {
        Vec::new()
    } else {
        outliers.previous.to_vec()
    };
    let searched = search_automatic_finalized_input(
        AutomaticInput::Transformed(&response),
        original,
        fit,
        outliers.spec,
        selected,
    )?;
    warnings.extend_from_slice(&searched.fit.warnings);
    Ok(RestoredFit {
        fit: searched.fit,
        selected_outliers: searched.selected,
    })
}
