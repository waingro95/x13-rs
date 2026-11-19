//! The fit `automd.f` continues with after its first order search.
//!
//! When the search chose the baseline's model and mean, `restor` recovers the
//! baseline. Otherwise the chosen model gets the baseline's regression back
//! and a new outlier search ([`restore_identified`]).

mod identified;
pub(crate) use identified::{OutlierRestoration, RestoredFit, restore_identified};

use super::errors::estimation_failure;
use super::first_pass::FirstPass;
use super::model_check::{ModelCheck, model_check};
use super::outlier_policy::no_request_outlier_policy;
use super::preparation::PreparedIdentification;
use crate::arima::model::ExpandedModel;
use crate::arima::{EstimatedRegArima, statistics_from_log_likelihood};
use crate::outliers::search::SelectedOutlier;
use crate::regarima::ResolvedArimaModel;
use crate::regression::model::RegressorKind;
use crate::{Error, Warning};

/// The first pass's model fitted with its regression, and its model check.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FirstRestoration {
    pub(crate) fit: EstimatedRegArima,
    pub(crate) check: ModelCheck,
    pub(crate) selected_outliers: Vec<SelectedOutlier>,
    /// The search chose the baseline's model and mean.
    pub(crate) same_baseline: bool,
    /// `restor` recovered the baseline's estimates.
    pub(crate) restored_baseline: bool,
}

/// The transformed series: the identification input with the removed
/// regression effects added back, which need not reproduce the bits the
/// subtraction started from.
fn restored_response(prepared: &PreparedIdentification) -> Vec<f64> {
    let mut response = prepared.values.clone();
    if let Some(effects) = &prepared.removed_effects {
        debug_assert_eq!(effects.len(), response.len());
        for (value, effect) in response.iter_mut().zip(effects) {
            *value += effect;
        }
    }
    response
}

pub(crate) fn restore_first_pass(
    first: &FirstPass,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<FirstRestoration, Error> {
    let baseline = &first.baseline.fit;
    let n = original.raw().len();
    let baseline_constant = baseline.regressor_kinds.contains(&RegressorKind::Constant);
    let same_baseline = ResolvedArimaModel::from(first.search.model) == baseline.spec.model()
        && first.search.mean_significant == baseline_constant;
    let restored_baseline = same_baseline && first.prepared.removed_effects.is_some();
    let restored = if same_baseline {
        let mut fit = first.search.fit.clone();
        let mut selected_outliers = Vec::new();
        fit.spec = baseline.spec.with_model(first.search.model);
        fit.transformed = restored_response(&first.prepared);
        fit.residuals = baseline.residuals.clone();
        if restored_baseline {
            fit.parameters = baseline.parameters.clone();
            fit.arma_workspace = baseline.arma_workspace.clone();
            fit.parameter_standard_errors = baseline.parameter_standard_errors.clone();
            fit.arma_covariance = baseline.arma_covariance.clone();
            fit.regression_coefficients = baseline.regression_coefficients.clone();
            fit.regression_standard_errors = baseline.regression_standard_errors.clone();
            fit.regressor_kinds = baseline.regressor_kinds.clone();
            fit.regression_covariance = baseline.regression_covariance.clone();
            fit.regression_factor = baseline.regression_factor.clone();
            fit.outlier_test_factor = baseline.outlier_test_factor.clone();
            fit.forecast_filter_workspace
                .restore_ma_initial_conditions(&baseline.forecast_filter_workspace);
            fit.innovation_variance = baseline.innovation_variance;
            fit.log_determinant_correction = baseline.log_determinant_correction;
            fit.regression_model = baseline.regression_model.evaluated_over(baseline.axis, n)?;
            selected_outliers = first.baseline.selected_outliers.clone();
        } else {
            // No effects were removed, so the baseline has no regressors.
            debug_assert!(
                baseline.regression_model.column_count() == 0
                    && first.baseline.selected_outliers.is_empty()
            );
            fit.regression_model = baseline.regression_model.clone();
        }
        recompute_likelihood_statistics(&mut fit, original)?;
        RestoredFit {
            fit,
            selected_outliers,
        }
    } else {
        restore_identified(
            baseline,
            &first.prepared,
            &first.search,
            original,
            OutlierRestoration {
                previous: &first.baseline.selected_outliers,
                clear_previous: true,
                spec: baseline
                    .spec
                    .outliers()
                    .unwrap_or_else(no_request_outlier_policy),
            },
            warnings,
        )?
    };
    let RestoredFit {
        fit,
        selected_outliers,
    } = restored;
    let expanded = ExpandedModel::from_spec(first.search.model, fit.axis.frequency());
    let check = model_check(
        first.search.model.into(),
        &fit.residuals,
        n - expanded.differencing_loss(),
        fit.axis.frequency(),
        expanded.parameter_slots().len(),
        fit.innovation_variance,
    )?;
    Ok(FirstRestoration {
        fit,
        check,
        selected_outliers,
        same_baseline,
        restored_baseline,
    })
}

/// Recomputes the likelihood statistics of `fit` for its current model and
/// regressors from the log likelihood it already has.
fn recompute_likelihood_statistics(
    fit: &mut EstimatedRegArima,
    original: crate::arima::FitContext<'_>,
) -> Result<(), Error> {
    let expanded = ExpandedModel::from_spec(fit.spec.model(), fit.axis.frequency());
    let loss = expanded.differencing_loss();
    // A fit with zero innovation variance, or one that fell back to the
    // conditional likelihood, has no statistics to recompute.
    let likelihood = fit.likelihood.ok_or_else(estimation_failure)?;
    let adjustment = original.transformation_adjustment(
        fit.regression_model.fixed_length(),
        loss,
        fit.spec.transform(),
    )?;
    fit.likelihood = Some(statistics_from_log_likelihood(
        original.raw().len() - loss,
        expanded.parameter_slots().len() + fit.regressor_kinds.len() + 1,
        likelihood.log_likelihood,
        adjustment,
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_the_effects_back_can_change_the_response_bits() {
        let x = 1.0e-10_f64;
        let effect = 0.1_f64;
        let prepared = PreparedIdentification {
            values: vec![x - effect],
            removed_effects: Some(vec![effect]),
        };
        let rebuilt = restored_response(&prepared);
        assert_eq!(rebuilt[0].to_bits(), ((x - effect) + effect).to_bits());
        assert_ne!(rebuilt[0].to_bits(), x.to_bits());
    }
}
