//! Hannan–Rissanen estimation for differencing identification: Build 62's
//! `hrest.f` and the MA refit `amdest.f` runs after it.

use crate::Frequency;
use crate::numerics::autocorrelation::{
    AutocorrelationError, Autocorrelations, ar_filtered_acf, sample_acf,
};
use crate::numerics::blas1::axpy;
use crate::numerics::packed::{packed_cholesky, packed_rank_product};

/// The ARMA orders of a working model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HrOrders {
    regular_ar: u8,
    seasonal_ar: u8,
    regular_ma: u8,
    seasonal_ma: u8,
}

impl HrOrders {
    pub(crate) fn new(regular_ar: u8, seasonal_ar: u8, regular_ma: u8, seasonal_ma: u8) -> Self {
        Self {
            regular_ar,
            seasonal_ar,
            regular_ma,
            seasonal_ma,
        }
    }

    /// The largest AR lag: regular plus seasonal span.
    fn ar_lag(self, period: usize) -> usize {
        usize::from(self.regular_ar) + period * usize::from(self.seasonal_ar)
    }

    /// The largest MA lag: regular plus seasonal span.
    fn ma_lag(self, period: usize) -> usize {
        usize::from(self.regular_ma) + period * usize::from(self.seasonal_ma)
    }
}

/// The estimated coefficients, in the order regular AR, seasonal AR, regular
/// MA, seasonal MA, and the number of regression rows behind them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HrEstimate {
    pub(crate) parameters: Vec<f64>,
    pub(crate) effective_observations: usize,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(crate) enum HrFailureReason {
    #[error("invalid Hannan–Rissanen input")]
    InvalidInput,
    #[error("insufficient observations for Hannan–Rissanen regression")]
    InsufficientObservations,
    #[error("autocorrelation failed: {0}")]
    Autocorrelation(#[from] AutocorrelationError),
    #[error("singular Hannan–Rissanen regression at pivot {pivot}")]
    SingularRegression { pivot: usize },
    #[error("nonfinite Hannan–Rissanen calculation")]
    NonFinite,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(crate) enum HrFailure {
    #[error("initial Hannan–Rissanen stage failed: {0}")]
    Initial(#[source] HrFailureReason),
    #[error("Hannan–Rissanen MA refit failed: {reason}")]
    MaRefit {
        #[source]
        reason: HrFailureReason,
        /// The estimate the refit started from.
        first_stage: HrEstimate,
    },
}

/// Estimates the working model's coefficients from `values`. A model with
/// both AR and MA terms has its MA coefficients estimated again from the
/// AR-filtered series, as `amdest.f` does.
pub(crate) fn estimate_prepared(
    values: &[f64],
    frequency: Frequency,
    orders: HrOrders,
) -> Result<HrEstimate, HrFailure> {
    estimate_validated(values, frequency, orders)
        .map_err(HrFailure::Initial)
        .and_then(|initial| {
            let period = usize::from(frequency.periods_per_year());
            let ar_lag = orders.ar_lag(period);
            let ma_lag = orders.ma_lag(period);
            if ar_lag == 0 || ma_lag == 0 {
                return Ok(initial);
            }
            refit_ma(values, period, orders, ar_lag, ma_lag, initial)
        })
}

fn estimate_validated(
    values: &[f64],
    frequency: Frequency,
    orders: HrOrders,
) -> Result<HrEstimate, HrFailureReason> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(HrFailureReason::InvalidInput);
    }
    let period = usize::from(frequency.periods_per_year());
    let ar_lag = orders.ar_lag(period);
    let ma_lag = orders.ma_lag(period);
    if ar_lag == 0 && ma_lag == 0 {
        return Ok(HrEstimate {
            parameters: vec![],
            effective_observations: values.len(),
        });
    }
    let lag_count = initial_lags(values.len(), ar_lag, ma_lag);
    let acf = if ma_lag > 0 {
        if lag_count == 0 || lag_count > values.len() {
            return Err(HrFailureReason::InsufficientObservations);
        }
        Some(sample_acf(values, lag_count)?.values)
    } else {
        // `amdest` calls `acf`, but `hrest` never consumes it for pure AR.
        None
    };
    estimate_stage(values, period, orders, acf.as_ref())
}

fn initial_lags(length: usize, ar_lag: usize, ma_lag: usize) -> usize {
    let log_squared = (length as f64).ln().powi(2) as usize;
    let minimum = log_squared.max(ar_lag).max(2 * ma_lag);
    let reset = if minimum >= length {
        minimum.min(length - length / 4)
    } else {
        minimum
    };
    reset.min(50)
}

fn refit_ma(
    values: &[f64],
    period: usize,
    orders: HrOrders,
    ar_lag: usize,
    ma_lag: usize,
    initial: HrEstimate,
) -> Result<HrEstimate, HrFailure> {
    let refit = (|| {
        let mut filtered = values.to_vec();
        let mut offset = 0;
        if orders.regular_ar > 0 {
            let lag = usize::from(orders.regular_ar);
            filtered = filter_factor(&filtered, lag, lag, |k| (k, initial.parameters[k - 1]))?;
            offset += lag;
        }
        if orders.seasonal_ar > 0 {
            filtered = filter_factor(&filtered, period, usize::from(orders.seasonal_ar), |k| {
                (
                    k * period,
                    initial.parameters[usize::from(orders.regular_ar) + k - 1],
                )
            })?;
            offset += period;
        }
        debug_assert_eq!(offset, ar_lag);
        let requested = initial_lags(filtered.len(), 0, ma_lag);
        let acf = ar_filtered_acf(&filtered, requested, ar_lag, ma_lag)?;
        let ma_orders = HrOrders::new(0, 0, orders.regular_ma, orders.seasonal_ma);
        let ma = estimate_stage(&filtered, period, ma_orders, Some(&acf))?;
        let mut estimate = initial.clone();
        let ma_start = usize::from(orders.regular_ar + orders.seasonal_ar);
        estimate.parameters[ma_start..].copy_from_slice(&ma.parameters);
        estimate.effective_observations = ma.effective_observations;
        Ok(estimate)
    })();
    refit.map_err(|reason| HrFailure::MaRefit {
        reason,
        first_stage: initial,
    })
}

fn filter_factor(
    input: &[f64],
    largest_lag: usize,
    factor_order: usize,
    factor: impl Fn(usize) -> (usize, f64),
) -> Result<Vec<f64>, HrFailureReason> {
    if largest_lag >= input.len() {
        return Err(HrFailureReason::InsufficientObservations);
    }
    let mut filtered = input.to_vec();
    for output_index in 0..input.len() - largest_lag {
        let current = output_index + largest_lag;
        let mut value = filtered[current];
        for k in 1..=factor_order {
            let (lag, coefficient) = factor(k);
            value = finite((-coefficient).mul_add(filtered[current - lag], value))?;
        }
        filtered[output_index] = value;
    }
    filtered.truncate(input.len() - largest_lag);
    Ok(filtered)
}

/// One regression of `hrest.f`. `acf` holds the autocorrelations the
/// innovations are estimated from, and is given exactly when `orders` has MA
/// terms.
fn estimate_stage(
    values: &[f64],
    period: usize,
    orders: HrOrders,
    acf: Option<&Autocorrelations>,
) -> Result<HrEstimate, HrFailureReason> {
    let regular_ar = usize::from(orders.regular_ar);
    let seasonal_ar = usize::from(orders.seasonal_ar);
    let regular_ma = usize::from(orders.regular_ma);
    let seasonal_ma = usize::from(orders.seasonal_ma);
    debug_assert_eq!(acf.is_some(), regular_ma + seasonal_ma > 0);
    let max_lag = orders.ar_lag(period).max(orders.ma_lag(period));
    if max_lag >= values.len() {
        return Err(HrFailureReason::InsufficientObservations);
    }
    let innovations = acf.map(|acf| innovations(values, acf)).transpose()?;
    // Each seasonal lag brings its own column and one per regular lag.
    let ar_columns = regular_ar + seasonal_ar * (regular_ar + 1);
    let column_count = ar_columns + regular_ma + seasonal_ma * (regular_ma + 1) + 1;
    let row_count = values.len() - max_lag;
    let matrix = augmented_matrix(values, period, orders, innovations.as_deref());
    let coefficients = solve_augmented(&matrix, row_count, column_count)?;
    let mut parameters = Vec::with_capacity(regular_ar + seasonal_ar + regular_ma + seasonal_ma);
    parameters.extend((0..regular_ar).map(|i| -coefficients[i]));
    parameters.extend((1..=seasonal_ar).map(|i| -coefficients[(regular_ar + 1) * i - 1]));
    parameters.extend((0..regular_ma).map(|i| -coefficients[ar_columns + i]));
    parameters
        .extend((1..=seasonal_ma).map(|i| -coefficients[ar_columns + (regular_ma + 1) * i - 1]));
    Ok(HrEstimate {
        parameters,
        effective_observations: row_count,
    })
}

fn augmented_matrix(
    values: &[f64],
    period: usize,
    orders: HrOrders,
    innovations: Option<&[f64]>,
) -> Vec<f64> {
    let regular_ar = usize::from(orders.regular_ar);
    let seasonal_ar = usize::from(orders.seasonal_ar);
    let regular_ma = usize::from(orders.regular_ma);
    let seasonal_ma = usize::from(orders.seasonal_ma);
    let max_lag = orders.ar_lag(period).max(orders.ma_lag(period));
    let column_count = regular_ar
        + seasonal_ar * (regular_ar + 1)
        + regular_ma
        + seasonal_ma * (regular_ma + 1)
        + 1;
    let row_count = values.len() - max_lag;
    let mut matrix = Vec::with_capacity(row_count * column_count);
    for index in max_lag..values.len() {
        for lag in 1..=regular_ar {
            matrix.push(-values[index - lag]);
        }
        for seasonal in 1..=seasonal_ar {
            matrix.push(-values[index - seasonal * period]);
            for regular in 1..=regular_ar {
                matrix.push(-values[index - seasonal * period - regular]);
            }
        }
        if let Some(ahat) = &innovations {
            for lag in 1..=regular_ma {
                matrix.push(ahat[index - lag]);
            }
            for seasonal in 1..=seasonal_ma {
                matrix.push(ahat[index - seasonal * period]);
                for regular in 1..=regular_ma {
                    matrix.push(ahat[index - seasonal * period - regular]);
                }
            }
        }
        matrix.push(values[index]);
    }
    matrix
}

fn solve_augmented(
    matrix: &[f64],
    row_count: usize,
    column_count: usize,
) -> Result<Vec<f64>, HrFailureReason> {
    if row_count == 0 {
        return Err(HrFailureReason::InsufficientObservations);
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(HrFailureReason::NonFinite);
    }
    let coefficient_count = column_count - 1;
    let mut packed = vec![0.0; column_count * (column_count + 1) / 2];
    packed_rank_product(matrix, row_count, column_count, &mut packed);
    if packed.iter().any(|value| !value.is_finite()) {
        return Err(HrFailureReason::NonFinite);
    }
    // hrest.f calls olsreg.f, which accepts `Info.eq.Ncxy`: a stop at the
    // augmented response column, the last, is a perfect fit, not a singular X.
    if let Err(failure) = packed_cholesky(&mut packed, column_count)
        && failure.column + 1 != column_count
    {
        return Err(HrFailureReason::SingularRegression {
            pivot: failure.column,
        });
    }
    if packed.iter().any(|value| !value.is_finite()) {
        return Err(HrFailureReason::NonFinite);
    }
    let mut coefficients =
        packed[coefficient_count * column_count / 2..][..coefficient_count].to_vec();
    let mut diagonal_end = coefficient_count * column_count / 2;
    for column in (1..=coefficient_count).rev() {
        let diagonal = packed[diagonal_end - 1];
        if diagonal == 0.0 {
            return Err(HrFailureReason::SingularRegression { pivot: column });
        }
        coefficients[column - 1] = finite(coefficients[column - 1] / diagonal)?;
        diagonal_end -= column;
        let alpha = -coefficients[column - 1];
        axpy(
            alpha,
            &packed[diagonal_end..][..column - 1],
            &mut coefficients[..column - 1],
        );
    }
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err(HrFailureReason::NonFinite);
    }
    Ok(coefficients)
}

/// Durbin's recursion for the long AR fit, then its residuals: `hrest.f`'s
/// innovations.
///
/// Lag `j` pairs with correlation `index - j - 1`, which the loops express as a
/// reversed zip. Every accumulation keeps `hrest.f`'s order and its `mul_add`
/// fusing.
fn innovations(values: &[f64], acf: &Autocorrelations) -> Result<Vec<f64>, HrFailureReason> {
    let lag_count = acf.correlations.len();
    if lag_count == 0 || lag_count > values.len() {
        return Err(HrFailureReason::InsufficientObservations);
    }
    let mut phat = vec![0.0; lag_count];
    let mut previous = vec![0.0; lag_count];
    phat[0] = acf.correlations[0];
    let mut pv = finite(acf.variance * (-phat[0]).mul_add(phat[0], 1.0))?;
    for index in 1..lag_count {
        let mut sum = 0.0;
        for (&coefficient, &correlation) in phat[..index]
            .iter()
            .zip(acf.correlations[..index].iter().rev())
        {
            sum = finite(coefficient.mul_add(correlation * acf.variance, sum))?;
        }
        let numerator = acf.correlations[index] * acf.variance - sum;
        let partial = finite(numerator / pv)?;
        pv = finite(pv * (-partial).mul_add(partial, 1.0))?;
        previous[..index].copy_from_slice(&phat[..index]);
        phat[index] = partial;
        for (coefficient, &reflected) in
            phat[..index].iter_mut().zip(previous[..index].iter().rev())
        {
            *coefficient = finite((-partial).mul_add(reflected, *coefficient))?;
        }
    }
    let mut ahat = Vec::with_capacity(values.len());
    for (index, &observation) in values.iter().enumerate() {
        let mut value = observation;
        for (&coefficient, &lagged) in phat[..lag_count.min(index)]
            .iter()
            .zip(values[..index].iter().rev())
        {
            value = finite((-coefficient).mul_add(lagged, value))?;
        }
        ahat.push(value);
    }
    Ok(ahat)
}

fn finite(value: f64) -> Result<f64, HrFailureReason> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HrFailureReason::NonFinite)
    }
}

#[cfg(test)]
mod tests;
