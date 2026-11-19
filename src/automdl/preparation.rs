//! `rmfix.f`: the series identification runs on, the transformed series less
//! the fitted regression effects.

use crate::Error;
use crate::arima::EstimatedRegArima;
use crate::numerics::blas1::axpy;
use crate::regression::model::RegressorKind;

/// The transformed series with the regression effects removed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedIdentification {
    pub(crate) values: Vec<f64>,
    /// The effects subtracted from `values`; `None` for a fit without
    /// regressors.
    pub(crate) removed_effects: Option<Vec<f64>>,
}

/// Removes every fitted regression effect but the constant from the
/// transformed series of `fitted`.
pub(crate) fn prepare_identification(
    fitted: &EstimatedRegArima,
) -> Result<PreparedIdentification, Error> {
    let model = &fitted.regression_model;
    let count = model.column_count();
    debug_assert_eq!(count, fitted.regression_coefficients.len());
    debug_assert!(model.kinds().eq(fitted.regressor_kinds.iter().copied()));
    if count == 0 {
        return Ok(PreparedIdentification {
            values: fitted.transformed.clone(),
            removed_effects: None,
        });
    }

    let n = fitted.transformed.len();
    let extensions = fitted.spec.extensions();
    let f = usize::from(extensions.forecast_periods(fitted.axis.frequency()));
    let b = usize::from(extensions.backcast_periods());
    let forward_count = n + f;
    // Build 62 quirk (`automd.f`, `rmfix.f`): automd builds the regression
    // matrix from the span start, without backcast rows, yet passes the
    // backcast count to rmfix, which then subtracts the effect `b` rows past
    // each observation and reads zero padding when `b > f`. Evaluating from
    // the axis start and taking `effects[b..b + n]` reproduces it.
    fitted.axis.periods(forward_count)?;
    let columns = model.evaluate(fitted.axis, forward_count)?;
    let mut effects = vec![0.0; n + f.max(b)];
    for index in (0..count).rev() {
        if fitted.regressor_kinds[index] != RegressorKind::Constant {
            axpy(
                fitted.regression_coefficients[index],
                &columns[index],
                &mut effects[..forward_count],
            );
        }
    }
    let removed_effects = effects[b..b + n].to_vec();
    let mut values = fitted.transformed.clone();
    for (value, effect) in values.iter_mut().zip(&removed_effects) {
        *value -= effect;
    }
    Ok(PreparedIdentification {
        values,
        removed_effects: Some(removed_effects),
    })
}
