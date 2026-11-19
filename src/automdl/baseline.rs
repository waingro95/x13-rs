//! The airline-model baseline `automd.f` fits first: two constant tests
//! around an outlier search, then the first `mdlchk`.

use super::constant::constant_is_insignificant;
use super::model_check::{ModelCheck, model_check};
use super::outlier_policy::no_request_outlier_policy;
use super::warnings::{push_constant_test_fit, push_reported_fit};
use crate::api::ResolvedRegArimaSpec;
use crate::arima::{ArmaStart, EstimatedRegArima, fit_automatic};
use crate::error::ModelSelectionConstantTest;
use crate::outliers::search::{SelectedOutlier, search_automatic};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{ArimaModel, Error, Warning};

/// The fitted baseline and its model check.
#[derive(Debug, PartialEq)]
pub(crate) struct BaselineFit {
    pub(crate) fit: EstimatedRegArima,
    pub(crate) selected_outliers: Vec<SelectedOutlier>,
    pub(crate) check: ModelCheck,
}

/// Fits the airline model in `spec` to `series`, appending each fit's
/// warnings to `warnings`.
pub(crate) fn initial_baseline(
    series: &[f64],
    original: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    warnings: &mut Vec<Warning>,
) -> Result<BaselineFit, Error> {
    debug_assert!(spec.model() == ArimaModel::airline().into() && !spec.includes_constant());
    debug_assert_eq!(series.len(), original.raw().len());
    axis.periods(series.len())?;

    let mut regression = RegressionModel::from_spec(spec, axis, series.len())?;
    regression.add_user_regressors(original.user_regressors(), axis, series.len())?;
    regression.add_missing(original.missing(), axis, series.len())?;
    regression.set_constant(true, spec.model(), axis.frequency(), series.len())?;
    let mut fit_spec = spec.with_constant(true);
    let initial = fit_automatic(
        series,
        original,
        axis,
        fit_spec,
        regression.clone(),
        ArmaStart::Default,
        true,
    )?;
    push_constant_test_fit(warnings, &initial);
    let drop_constant = match initial.as_converged() {
        Some(fit) => constant_is_insignificant(fit, ModelSelectionConstantTest::Initial, 1.96)?,
        None => true,
    };
    if drop_constant {
        regression.set_constant(false, spec.model(), axis.frequency(), series.len())?;
        fit_spec = fit_spec.with_constant(false);
    }
    let before = fit_automatic(
        series,
        original,
        axis,
        fit_spec,
        regression,
        initial.numerical_view().refit_start(),
        initial.pre_covariance_status().converged,
    )?;
    push_reported_fit(warnings, &before);
    if !before.pre_covariance_status().converged {
        return Err(Error::EstimationDidNotConverge {
            iterations: before.nonlinear_iterations(),
        });
    }
    let policy = spec.outliers().unwrap_or_else(no_request_outlier_policy);
    let search = search_automatic(series, original, before, policy, Vec::new())?;
    warnings.extend_from_slice(&search.fit.warnings);
    let mut regression = search.regression_model;
    if constant_is_insignificant(
        &search.fit,
        ModelSelectionConstantTest::AfterOutlierSearch,
        1.96,
    )? {
        regression.set_constant(false, spec.model(), axis.frequency(), series.len())?;
        fit_spec = fit_spec.with_constant(false);
    }
    let final_fit = fit_automatic(
        series,
        original,
        axis,
        fit_spec,
        regression,
        search.fit.numerical_view().refit_start(),
        true,
    )?;
    push_reported_fit(warnings, &final_fit);
    let iterations = final_fit.nonlinear_iterations();
    let effective_observations = final_fit.effective_observations();
    let fit = final_fit
        .into_converged()
        .map_err(|_| Error::EstimationDidNotConverge { iterations })?;
    let check = model_check(
        fit_spec.model(),
        fit.numerical_view().residuals,
        effective_observations,
        axis.frequency(),
        2,
        fit.innovation_variance,
    )?;
    Ok(BaselineFit {
        fit,
        selected_outliers: search.selected,
        check,
    })
}
