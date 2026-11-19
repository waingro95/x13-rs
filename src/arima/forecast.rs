use crate::arima::estimation::EstimatedRegArima;
use crate::arima::exact::{FilterOptions, exact_filter};
use crate::arima::model::{ExpandedModel, operator_span};
use crate::arima::operators::multiply_operators;
use crate::error::NumericalOperation;
use crate::numerics::RowMajorMatrix;
use crate::numerics::blas1::dot;
use crate::numerics::packed::packed_forward_solve;
use crate::numerics::polynomial::{Polynomial, rational_expand};
use crate::regression::design::RegressionMatrix;
use crate::regression::gls::regression_residuals;
use crate::{Error, Period};

// The value Build 62's `dinvnr` returns for the default 95% coverage.
const NORMAL_95_CRITICAL_VALUE: f64 = f64::from_bits(0x3fff_5c03_31ee_ff83);

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ForecastPoint {
    pub(crate) period: Period,
    pub(crate) transformed: f64,
    pub(crate) transformed_standard_error: f64,
    pub(crate) regression_variance: Option<f64>,
    pub(crate) stochastic_variance: f64,
    pub(crate) original: f64,
    pub(crate) lower_95: f64,
    pub(crate) upper_95: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BackcastPoint {
    pub(crate) period: Period,
    pub(crate) transformed: f64,
    pub(crate) transformed_standard_error: f64,
    pub(crate) original: f64,
    pub(crate) lower_95: f64,
    pub(crate) upper_95: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExtensionsResult {
    pub(crate) forecasts: Vec<ForecastPoint>,
    pub(crate) backcasts: Vec<BackcastPoint>,
}

struct RawExtensionPoint {
    transformed: f64,
    transformed_standard_error: f64,
    regression_variance: Option<f64>,
    stochastic_variance: f64,
}

pub(crate) fn generate_extensions(
    estimated: &EstimatedRegArima,
) -> Result<ExtensionsResult, Error> {
    let horizon = usize::from(
        estimated
            .spec
            .extensions()
            .forecast_periods(estimated.axis.frequency()),
    );
    let forecasts = generate_forecasts(estimated, horizon)?;
    let backcast_horizon = usize::from(estimated.spec.extensions().backcast_periods());
    let backcasts = generate_backcasts(estimated, backcast_horizon)?;
    Ok(ExtensionsResult {
        forecasts,
        backcasts,
    })
}

fn generate_forecasts(
    estimated: &EstimatedRegArima,
    horizon: usize,
) -> Result<Vec<ForecastPoint>, Error> {
    if horizon == 0 {
        return Ok(Vec::new());
    }

    let observed_count = estimated.transformed.len();
    let mut extended_response = estimated.transformed.clone();
    extended_response.resize(observed_count + horizon, 0.0);
    let model = ExpandedModel::from_spec(estimated.spec.model(), estimated.axis.frequency());
    let extended_matrix = regression_matrix(&extended_response, estimated.axis, estimated)?;
    let raw = forecast_rows(&extended_matrix, observed_count, horizon, &model, estimated)?;
    let forecast_axis = estimated.axis.shift(
        i64::try_from(observed_count)
            .map_err(|_| Error::internal("observation count exceeds i64"))?,
    )?;
    let fixed_length = estimated
        .regression_model
        .evaluate_fixed_length(forecast_axis, horizon)?;

    raw.into_iter()
        .zip(fixed_length)
        .enumerate()
        .map(|(lead, (raw, fixed_length))| {
            let interval = original_interval(&raw, fixed_length, estimated);
            Ok(ForecastPoint {
                period: estimated.axis.period_at(observed_count + lead)?,
                transformed: raw.transformed,
                transformed_standard_error: raw.transformed_standard_error,
                regression_variance: raw.regression_variance,
                stochastic_variance: raw.stochastic_variance,
                original: interval.point,
                lower_95: interval.lower,
                upper_95: interval.upper,
            })
        })
        .collect()
}

fn generate_backcasts(
    estimated: &EstimatedRegArima,
    horizon: usize,
) -> Result<Vec<BackcastPoint>, Error> {
    if horizon == 0 {
        return Ok(Vec::new());
    }

    let observed_count = estimated.transformed.len();
    let mut extended_response = vec![0.0; horizon];
    extended_response.extend_from_slice(&estimated.transformed);
    let model = ExpandedModel::from_spec(estimated.spec.model(), estimated.axis.frequency());
    let axis = estimated.axis.shift(
        -i64::try_from(horizon).map_err(|_| Error::internal("backcast horizon exceeds i64"))?,
    )?;
    let forward_matrix = regression_matrix(&extended_response, axis, estimated)?;
    let reversed_matrix = RegressionMatrix::from(reverse_rows(forward_matrix.as_matrix()));
    let mut raw = forecast_rows(&reversed_matrix, observed_count, horizon, &model, estimated)?;
    raw.reverse();
    let fixed_length = estimated
        .regression_model
        .evaluate_fixed_length(axis, horizon)?;
    raw.into_iter()
        .zip(fixed_length)
        .enumerate()
        .map(|(index, (raw, fixed_length))| {
            let interval = original_interval(&raw, fixed_length, estimated);
            Ok(BackcastPoint {
                period: axis.period_at(index)?,
                transformed: raw.transformed,
                transformed_standard_error: raw.transformed_standard_error,
                original: interval.point,
                lower_95: interval.lower,
                upper_95: interval.upper,
            })
        })
        .collect()
}

/// The point and its 95% interval on the original scale, with the
/// length-of-period prior multiplied back in.
fn original_interval(
    raw: &RawExtensionPoint,
    fixed_length: f64,
    estimated: &EstimatedRegArima,
) -> crate::transform::InverseInterval {
    let interval = crate::transform::inverse_interval(
        raw.transformed,
        raw.transformed_standard_error,
        NORMAL_95_CRITICAL_VALUE,
        estimated.spec.transform(),
    );
    crate::transform::InverseInterval {
        point: interval.point * fixed_length,
        lower: interval.lower * fixed_length,
        upper: interval.upper * fixed_length,
    }
}

fn forecast_rows(
    extended_matrix: &RegressionMatrix,
    observed_count: usize,
    horizon: usize,
    model: &ExpandedModel,
    estimated: &EstimatedRegArima,
) -> Result<Vec<RawExtensionPoint>, Error> {
    let column_count = extended_matrix.column_count();
    let extended = extended_matrix.values();
    let observed = RowMajorMatrix::new(
        extended[..observed_count * column_count].to_vec(),
        column_count,
    );

    let mut filter_workspace = estimated.forecast_filter_workspace.clone();
    let filtered = exact_filter(
        observed,
        model,
        &estimated.parameters,
        &mut filter_workspace,
        FilterOptions::FORECAST_TAIL,
    )
    .map_err(|error| Error::numerical_from(NumericalOperation::Extension, error.into()))?;
    let residuals = filtered.matrix().as_slice();

    let mut ar_and_differencing = model.differencing_operators().to_vec();
    ar_and_differencing.extend_from_slice(model.ar_operators());
    let full_ar = multiply_operators(&ar_and_differencing, &estimated.parameters);
    let full_ma = multiply_operators(model.ma_operators(), &estimated.parameters);
    let full_ar_span = operator_span(&ar_and_differencing);
    let full_ma_span = model.ma_span();
    let psi_weights = psi_weights(
        full_ar.as_polynomial(),
        full_ma.as_polynomial(),
        full_ma_span,
        horizon,
    );

    let first_history_row = observed_count
        .checked_sub(full_ar_span)
        .ok_or(Error::internal(
            "forecast history is shorter than the AR operator",
        ))?;
    let history_elements = full_ar_span * column_count;
    let future_elements = horizon * column_count;
    // `fcstxy`'s recursion over the last `full_ar_span` observed rows and the
    // future rows, in place. Each future element's prediction replaces it,
    // for later leads to read, and the prediction less the original value
    // goes to the element's own lead row at the front of the window.
    let mut window = extended
        [first_history_row * column_count..(observed_count + horizon) * column_count]
        .to_vec();
    debug_assert_eq!(window.len(), history_elements + future_elements);

    for element in 0..future_elements {
        let shifted = element + history_elements;
        let mut value = 0.0;
        for (coefficient, lag) in full_ar.terms() {
            value = coefficient.mul_add(window[shifted - lag * column_count], value);
        }
        for (coefficient, lag) in full_ma.terms() {
            // The residual `lag` rows before this element; none before the
            // first filtered row or past the last.
            let residual = (element + residuals.len())
                .checked_sub(lag * column_count)
                .and_then(|index| residuals.get(index));
            if let Some(residual) = residual {
                value = (-coefficient).mul_add(*residual, value);
            }
        }
        window[element] = value - window[shifted];
        if element < shifted {
            window[shifted] = value;
        }
    }

    window.truncate(future_elements);
    let forecast_matrix = RegressionMatrix::from(RowMajorMatrix::new(window, column_count));
    let transformed = regression_residuals(&forecast_matrix, &estimated.regression_coefficients)?;
    let regression_count = estimated.regression_coefficients.len();
    let mut future_regressors = vec![0.0; regression_count];
    let mut cumulative_psi_squares = 0.0;
    let mut points = Vec::with_capacity(horizon);
    for ((row, psi_weight), transformed) in forecast_matrix
        .as_matrix()
        .rows()
        .zip(psi_weights)
        .zip(transformed)
    {
        let regression_sum = if regression_count == 0 {
            0.0
        } else {
            future_regressors.copy_from_slice(&row[..regression_count]);
            packed_forward_solve(
                &estimated.regression_factor,
                regression_count,
                1,
                &mut future_regressors,
            );
            dot(&future_regressors, &future_regressors)
        };
        let raw_regression_variance = regression_sum * estimated.innovation_variance;
        cumulative_psi_squares = psi_weight.mul_add(psi_weight, cumulative_psi_squares);
        let transformed_standard_error =
            ((regression_sum + cumulative_psi_squares) * estimated.innovation_variance).sqrt();
        // `prtfct.f` reports a regression variance of zero unless its
        // `log10` exceeds -100, which a zero or NaN variance fails.
        let regression_variance = if raw_regression_variance.log10() > -100.0 {
            raw_regression_variance
        } else {
            0.0
        };
        let stochastic_variance =
            transformed_standard_error * transformed_standard_error - regression_variance;
        points.push(RawExtensionPoint {
            transformed,
            transformed_standard_error,
            regression_variance: (regression_count > 0).then_some(regression_variance),
            stochastic_variance,
        });
    }
    Ok(points)
}

/// The matrix's rows in reverse order: backcasts are forecasts of the
/// time-reversed `[X:y]` (`revrse.f`).
fn reverse_rows(matrix: &RowMajorMatrix) -> RowMajorMatrix {
    let reversed = matrix.rows().rev().flatten().copied().collect();
    RowMajorMatrix::new(reversed, matrix.column_count())
}

fn regression_matrix(
    response: &[f64],
    axis: crate::time::TimeAxis,
    estimated: &EstimatedRegArima,
) -> Result<RegressionMatrix, Error> {
    let columns = estimated.regression_model.evaluate(axis, response.len())?;
    RegressionMatrix::from_columns(&columns, response)
}

/// `fcstxy.f`'s psi weights, `MA(B) / AR(B)` to `horizon` terms: the MA
/// weights `1, -theta` divided by the full AR with `ratpos.f`, which reads
/// only the first `full_ma_span + 1` of them.
fn psi_weights(
    full_ar: Polynomial<'_>,
    full_ma: Polynomial<'_>,
    full_ma_span: usize,
    horizon: usize,
) -> Vec<f64> {
    let mut ma_weights = vec![0.0; (full_ma_span + 1).min(horizon)];
    ma_weights[0] = 1.0;
    for (coefficient, lag) in full_ma.terms() {
        if let Some(weight) = ma_weights.get_mut(lag) {
            *weight = -coefficient;
        }
    }
    rational_expand(&ma_weights, horizon, &[full_ar])
}

#[cfg(test)]
mod tests {
    use crate::Transform;
    use crate::numerics::RowMajorMatrix;

    use super::{NORMAL_95_CRITICAL_VALUE, reverse_rows};

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn reverse_rows_keeps_each_row_intact() {
        let odd = RowMajorMatrix::new(vec![1.0, 2.0, 3.0, 4.0, -0.0, f64::NAN], 2);
        assert_eq!(
            bits(reverse_rows(&odd).as_slice()),
            bits(&[-0.0, f64::NAN, 3.0, 4.0, 1.0, 2.0])
        );
        let even = RowMajorMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 4);
        assert_eq!(
            reverse_rows(&even).as_slice(),
            [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]
        );
        let empty = reverse_rows(&RowMajorMatrix::new(Vec::new(), 3));
        assert_eq!((empty.row_count(), empty.column_count()), (0, 3));
    }

    #[test]
    fn the_95_percent_interval_uses_build_62s_critical_value() {
        for (point, lower_bits, upper_bits) in [
            (10.0, 0x4018_51fe_6708_803e, 0x402b_d700_cc7b_bfe1),
            (
                3.919_927_969_080_108,
                0x3cc0_0000_0000_0000,
                0x401f_5c03_31ee_ff84,
            ),
            (
                -3.919_927_969_080_108,
                0xc01f_5c03_31ee_ff84,
                0xbcc0_0000_0000_0000,
            ),
        ] {
            let interval = crate::transform::inverse_interval(
                point,
                2.0,
                NORMAL_95_CRITICAL_VALUE,
                Transform::None,
            );
            assert_eq!(interval.point.to_bits(), point.to_bits());
            assert_eq!(interval.lower.to_bits(), lower_bits);
            assert_eq!(interval.upper.to_bits(), upper_bits);
        }
    }
}
