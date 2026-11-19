//! The t-test `automd.f` applies to the constant of a converged fit.

use crate::Error;
use crate::arima::EstimatedRegArima;
use crate::error::{ModelSelectionConstantTest, NumericalDetail};
use crate::regression::model::RegressorKind;

/// Whether `fit` has a constant whose |t| is below `threshold`, which
/// `chkmu.f` sets to 1.96 in its first round and 1.6 in its second. A
/// constant with a zero standard error counts as t = 0.
pub(super) fn constant_is_insignificant(
    fit: &EstimatedRegArima,
    test: ModelSelectionConstantTest,
    threshold: f64,
) -> Result<bool, Error> {
    let Some(index) = fit
        .regressor_kinds
        .iter()
        .position(|kind| *kind == RegressorKind::Constant)
    else {
        return Ok(false);
    };
    let standard_errors = fit.regression_standard_errors.as_ref().ok_or(Error::from(
        NumericalDetail::ModelSelectionConstantTestUnavailable {
            model: fit.spec.model(),
            test,
        },
    ))?;
    let standard_error = standard_errors[index];
    let t = if standard_error > 0.0 {
        fit.regression_coefficients[index] / standard_error
    } else {
        0.0
    };
    Ok(t.abs() < threshold)
}
