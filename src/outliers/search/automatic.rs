//! The entry points automatic modelling uses to run the `idotlr.f` search
//! from a fit it already has.

use super::fit::ResidualFit;
use super::{
    OutlierSearchResult, SearchInitial, SearchProblem, SelectedOutlier, period_index,
    search_with_initial_fit,
};
use crate::Error;
use crate::arima::{
    AutomaticFit, AutomaticInput, EstimatedRegArima, FitContext, FittedNumericalView,
    fit_automatic_input,
};
use crate::regarima::OutlierSpec;
use crate::regression::model::RegressorKind;

mod cleared;
pub(crate) use cleared::{ClearedRegressionFit, search_automatic_cleared_input};

/// Searches the untransformed `series` from `initial_fit`, which must have
/// converged.
pub(crate) fn search_automatic(
    series: &[f64],
    original_series: FitContext<'_>,
    initial_fit: AutomaticFit,
    outlier_spec: OutlierSpec,
    previous_selected: Vec<SelectedOutlier>,
) -> Result<OutlierSearchResult, Error> {
    search_automatic_finalized_input(
        AutomaticInput::Untransformed(series),
        original_series,
        accept_automatic_fit(initial_fit)?,
        outlier_spec,
        previous_selected,
    )
}

/// An automatic search result plus the residuals it ran on when automdl's
/// saved residual prefix replaced the fit's own.
pub(crate) struct OutlierSearch {
    pub(crate) result: OutlierSearchResult,
    pub(crate) residual_override: Option<Vec<f64>>,
}

/// Searches from a refit that may not have converged yet, with its first
/// residuals optionally replaced by `saved_residual_prefix`. Every later refit
/// must converge and uses its own residuals.
pub(crate) fn search_automatic_residuals(
    input: AutomaticInput<'_>,
    original_series: FitContext<'_>,
    initial_fit: AutomaticFit,
    saved_residual_prefix: Option<&[f64]>,
    outlier_spec: OutlierSpec,
    previous_selected: Vec<SelectedOutlier>,
) -> Result<OutlierSearch, Error> {
    let view = initial_fit.numerical_view();
    let observation_count = view.transformed.len();
    debug_assert_eq!(input.values().len(), observation_count);
    debug_assert_eq!(original_series.raw().len(), observation_count);
    debug_assert!(selection_matches_fit(view, &previous_selected));
    let axis = view.axis;
    let spec = initial_fit.spec();
    let regression_model = initial_fit.regression_model().clone();
    let mut incoming_convergence = initial_fit.pre_covariance_status().converged;
    let initial_fit = ResidualFit::new(initial_fit, saved_residual_prefix);
    let OutlierSearchResult {
        fit: ResidualFit {
            fit,
            residual_override,
        },
        regression_model,
        selected,
    } = search_with_initial_fit(
        SearchProblem {
            observation_count,
            axis,
            spec,
            outlier_spec,
        },
        regression_model,
        SearchInitial::Automatic {
            fit: Box::new(initial_fit),
            selected: previous_selected,
        },
        |regression, start| {
            let refitted = fit_automatic_input(
                input,
                original_series,
                axis,
                spec,
                regression,
                start,
                incoming_convergence,
            )?;
            if refitted.as_converged().is_none() {
                return Err(Error::EstimationDidNotConverge {
                    iterations: refitted.nonlinear_iterations(),
                });
            }
            incoming_convergence = true;
            Ok(ResidualFit::new(refitted, None))
        },
    )?;
    Ok(OutlierSearch {
        result: OutlierSearchResult {
            fit: accept_automatic_fit(fit)?,
            regression_model,
            selected,
        },
        residual_override,
    })
}

/// Searches from the converged `fit` of `input`. `previous_selected` are the
/// outliers an earlier search left in its regression. Every refit must
/// converge.
pub(crate) fn search_automatic_finalized_input(
    input: AutomaticInput<'_>,
    original_series: FitContext<'_>,
    fit: EstimatedRegArima,
    outlier_spec: OutlierSpec,
    previous_selected: Vec<SelectedOutlier>,
) -> Result<OutlierSearchResult, Error> {
    let observation_count = fit.transformed.len();
    debug_assert_eq!(input.values().len(), observation_count);
    debug_assert_eq!(original_series.raw().len(), observation_count);
    debug_assert!(selection_matches_fit(
        fit.numerical_view(),
        &previous_selected
    ));
    let axis = fit.axis;
    let spec = fit.spec;
    let regression_model = fit.regression_model.clone();
    search_with_initial_fit(
        SearchProblem {
            observation_count,
            axis,
            spec,
            outlier_spec,
        },
        regression_model,
        SearchInitial::Automatic {
            fit: Box::new(fit),
            selected: previous_selected,
        },
        |regression, start| {
            accept_automatic_fit(fit_automatic_input(
                input,
                original_series,
                axis,
                spec,
                regression,
                start,
                true,
            )?)
        },
    )
}

/// The converged fit, or the error for one that stopped at the iteration
/// limit.
fn accept_automatic_fit(fit: AutomaticFit) -> Result<EstimatedRegArima, Error> {
    let iterations = fit.nonlinear_iterations();
    fit.into_converged()
        .map_err(|_| Error::EstimationDidNotConverge { iterations })
}

/// Whether `selected` is exactly `fit`'s outlier regressors, in Build 62's
/// date-then-type order.
fn selection_matches_fit(fit: FittedNumericalView<'_>, selected: &[SelectedOutlier]) -> bool {
    let fitted = fit.regressor_kinds.iter().filter_map(|kind| match kind {
        RegressorKind::Outlier { kind, period, .. } => Some((*kind, *period)),
        _ => None,
    });
    let order = |outlier: &SelectedOutlier| (outlier.period, outlier.kind.build_62_index());
    fitted.eq(selected
        .iter()
        .map(|outlier| (outlier.kind, outlier.period)))
        && selected.iter().all(|outlier| {
            period_index(fit.axis, fit.transformed.len(), outlier.period) == Some(outlier.index)
        })
        && selected
            .windows(2)
            .all(|pair| order(&pair[0]) < order(&pair[1]))
}

#[cfg(test)]
mod tests;
