//! The fit the automatic model selection returns.

use super::errors::estimation_failure;
use super::fallback::FallbackFit;
use crate::Error;
use crate::arima::EstimatedRegArima;
use crate::arima::model::ExpandedModel;

pub(crate) struct TerminalFit {
    pub(crate) fit: EstimatedRegArima,
    /// The residuals to report when they are not the fit's own.
    pub(crate) residual_override: Option<Vec<f64>>,
}

impl TerminalFit {
    pub(crate) fn reported_residuals(&self) -> &[f64] {
        self.residual_override
            .as_deref()
            .unwrap_or(&self.fit.residuals)
    }
}

/// Converts the selector's last fit into the fit it returns. Fails if that
/// fit did not converge or holds a value that is not finite.
pub(crate) fn finish_fit(
    fit: FallbackFit,
    residual_override: Option<Vec<f64>>,
    original: crate::arima::FitContext<'_>,
) -> Result<TerminalFit, Error> {
    let fit = match fit {
        FallbackFit::Existing(fit) => *fit,
        FallbackFit::Refit(fit) => {
            let iterations = fit.nonlinear_iterations();
            fit.into_converged()
                .map_err(|_| Error::EstimationDidNotConverge { iterations })?
        }
        FallbackFit::ClearedRegression(fit) => match fit.into_synchronized() {
            Ok(fit) => fit,
            Err(cleared) => cleared.into_reported_fit(original)?,
        },
    };
    let n = original.raw().len();
    debug_assert_eq!(fit.transformed.len(), n);
    debug_assert_eq!(
        fit.parameters.len(),
        ExpandedModel::from_spec(fit.spec.model(), fit.axis.frequency())
            .parameter_slots()
            .len()
    );
    debug_assert_eq!(fit.regression_coefficients.len(), fit.regressor_kinds.len());
    debug_assert!(
        fit.regression_model
            .kinds()
            .eq(fit.regressor_kinds.iter().copied())
    );
    debug_assert!(
        fit.regression_model
            .columns()
            .all(|(_, column)| column.len() == n)
    );
    if !has_finite_values(&fit, original.raw())
        || residual_override
            .as_deref()
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err(estimation_failure());
    }
    Ok(TerminalFit {
        fit,
        residual_override,
    })
}

/// Whether every number in `fit` and `raw` is finite, the innovation
/// variance is not negative and the length-of-period factors are positive.
fn has_finite_values(fit: &EstimatedRegArima, raw: &[f64]) -> bool {
    let all_finite = |values: &[f64]| values.iter().all(|value| value.is_finite());
    fit.innovation_variance.is_finite()
        && fit.innovation_variance >= 0.0
        && fit.log_determinant_correction.is_finite()
        && fit.likelihood.is_none_or(|value| {
            [
                value.log_likelihood,
                value.aic,
                value.hannan_quinn,
                value.bic,
            ]
            .into_iter()
            .chain(value.aicc)
            .all(f64::is_finite)
        })
        && fit
            .regression_model
            .columns()
            .all(|(_, column)| all_finite(column))
        && fit
            .regression_model
            .fixed_length()
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && [
            raw,
            &fit.transformed,
            &fit.parameters,
            &fit.residuals,
            &fit.regression_coefficients,
            &fit.regression_factor,
            &fit.outlier_test_factor,
        ]
        .into_iter()
        .all(all_finite)
        && [
            fit.parameter_standard_errors.as_deref(),
            fit.regression_standard_errors.as_deref(),
            fit.regression_covariance.as_deref(),
        ]
        .into_iter()
        .flatten()
        .all(all_finite)
}
