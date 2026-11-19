//! `mdlchk`: the residual statistics `automd.f` compares models by.

use crate::Error;
use crate::error::NumericalDetail;
use crate::numerics::autocorrelation::AutocorrelationError;
use crate::regarima::ResolvedArimaModel;
use crate::time::Frequency;

use super::diagnostics::{DiagnosticError, model_check_lag, residual_acf};
use super::errors::estimation_failure;

/// The residual statistics of one fitted model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ModelCheck {
    /// One minus the Ljung-Box p-value at the checked lag.
    pub(crate) confidence: f64,
    /// The square root of the innovation variance.
    pub(crate) innovation_scale: f64,
    /// The t statistic of the residual mean. Residuals with no spread give
    /// an infinity or NaN.
    pub(crate) residual_mean_t: f64,
}

/// Checks the last `effective_observations` residuals of a fit of `model`.
/// The Ljung-Box degrees of freedom exclude `free_arma_count`.
pub(crate) fn model_check(
    model: ResolvedArimaModel,
    residuals: &[f64],
    effective_observations: usize,
    frequency: Frequency,
    free_arma_count: usize,
    innovation_variance: f64,
) -> Result<ModelCheck, Error> {
    if effective_observations < 2 || effective_observations > residuals.len() {
        return Err(Error::internal(
            "residual check sample does not fit the residuals",
        ));
    }
    if !innovation_variance.is_finite() || innovation_variance < 0.0 {
        return Err(estimation_failure());
    }
    let effective_residuals = &residuals[residuals.len() - effective_observations..];
    let lag = model_check_lag(frequency, effective_observations);
    let rows =
        residual_acf(effective_residuals, lag, free_arma_count).map_err(|error| match error {
            DiagnosticError::Autocorrelation(AutocorrelationError::ZeroVariance) => {
                Error::from(NumericalDetail::ModelSelectionResidualCheckUnavailable { model })
            }
            _ => estimation_failure(),
        })?;
    let last = rows
        .last()
        .ok_or(Error::internal("residual autocorrelation has no lags"))?;

    let mut sum = 0.0;
    let mut squares = 0.0;
    for &value in effective_residuals {
        sum += value;
        squares = value.mul_add(value, squares);
    }
    let n = effective_residuals.len() as f64;
    let mean = sum / n;
    let raw_variance = (-mean).mul_add(mean, squares / n);

    Ok(ModelCheck {
        confidence: 1.0 - last.p_value,
        innovation_scale: innovation_variance.sqrt(),
        residual_mean_t: mean / (raw_variance / n).sqrt(),
    })
}

#[cfg(test)]
mod tests;
