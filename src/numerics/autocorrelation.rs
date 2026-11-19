//! Sample autocorrelations (`acf.f`, `acfar.f`).

/// A lag-zero variance and the autocorrelations at lags one onward.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Autocorrelations {
    pub(crate) variance: f64,
    pub(crate) correlations: Vec<f64>,
}

/// Autocorrelations with their standard errors.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SampleAcf {
    pub(crate) values: Autocorrelations,
    pub(crate) standard_errors: Vec<f64>,
}

/// Why an autocorrelation could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum AutocorrelationError {
    #[error("invalid autocorrelation input or lag count")]
    InvalidInput,
    #[error("autocorrelation input has zero variance")]
    ZeroVariance,
    #[error("autocorrelation calculation is nonfinite")]
    NonFinite,
}

/// Build 62 `acf.f`: the autocorrelations of `values` about their mean at
/// lags 1 through `lag_count`, with Bartlett standard errors.
///
/// The covariance and the variance are each divided by `n` before their
/// ratio is taken.
pub(crate) fn sample_acf(
    values: &[f64],
    lag_count: usize,
) -> Result<SampleAcf, AutocorrelationError> {
    if values.is_empty() || lag_count == 0 || lag_count > values.len() {
        return Err(AutocorrelationError::InvalidInput);
    }
    validate_finite(values)?;
    let n = values.len() as f64;
    let mut mean = 0.0;
    for &value in values {
        mean = finite(mean + value)?;
    }
    mean = finite(mean / n)?;

    let mut c0 = 0.0;
    for &value in values {
        let centered = finite(value - mean)?;
        c0 = finite(centered.mul_add(centered, c0))?;
    }
    if c0 <= 0.0 {
        return Err(AutocorrelationError::ZeroVariance);
    }
    c0 = finite(c0 / n)?;
    let mut correlations = Vec::with_capacity(lag_count);
    for lag in 1..=lag_count {
        let mut covariance = 0.0;
        for (&current, &previous) in values[lag..].iter().zip(values) {
            let current = finite(current - mean)?;
            let previous = finite(previous - mean)?;
            covariance = finite(current.mul_add(previous, covariance))?;
        }
        covariance = finite(covariance / n)?;
        correlations.push(finite(covariance / c0)?);
    }

    let mut standard_errors = Vec::with_capacity(lag_count);
    standard_errors.push(finite(1.0 / n.sqrt())?);
    let mut squared_sum = 0.0;
    for &correlation in correlations.iter().take(lag_count - 1) {
        squared_sum = finite(correlation.mul_add(correlation, squared_sum))?;
        let numerator = finite(2.0_f64.mul_add(squared_sum, 1.0))?;
        standard_errors.push(finite((numerator / n).sqrt())?);
    }
    Ok(SampleAcf {
        values: Autocorrelations {
            variance: c0,
            correlations,
        },
        standard_errors,
    })
}

/// Build 62 `acfar.f`: the autocorrelations of `values` about zero, at
/// enough lags to fit a long autoregression.
///
/// The lag count is the largest of `requested_lags`, `ar_lag + ma_lag + 1`
/// and `ln(n)^2`. When that reaches `n`, it is the larger of the first two,
/// capped at `n - n / 4`. Each covariance is divided once, by `n * c0`.
pub(crate) fn ar_filtered_acf(
    values: &[f64],
    requested_lags: usize,
    ar_lag: usize,
    ma_lag: usize,
) -> Result<Autocorrelations, AutocorrelationError> {
    if values.is_empty() {
        return Err(AutocorrelationError::InvalidInput);
    }
    validate_finite(values)?;
    let model_lags = ar_lag
        .checked_add(ma_lag)
        .and_then(|sum| sum.checked_add(1))
        .ok_or(AutocorrelationError::InvalidInput)?;
    let minimum_lags = requested_lags.max(model_lags);
    let n = values.len();
    let log_squared_lags = (n as f64).ln().powi(2) as usize;
    let mut lag_count = minimum_lags.max(log_squared_lags);
    if lag_count >= n {
        lag_count = minimum_lags.min(n - n / 4);
    }

    let n_float = n as f64;
    let mut c0 = 0.0;
    for &value in values {
        c0 = finite(value.mul_add(value, c0))?;
    }
    if c0 <= 0.0 {
        return Err(AutocorrelationError::ZeroVariance);
    }
    c0 = finite(c0 / n_float)?;
    let denominator = finite(n_float * c0)?;
    let mut correlations = Vec::with_capacity(lag_count);
    for lag in 1..=lag_count {
        let mut covariance = 0.0;
        // `lag_count` never exceeds `n`, so the slice is in bounds.
        for (&current, &previous) in values[lag..].iter().zip(values) {
            covariance = finite(current.mul_add(previous, covariance))?;
        }
        correlations.push(finite(covariance / denominator)?);
    }
    Ok(Autocorrelations {
        variance: c0,
        correlations,
    })
}

fn validate_finite(values: &[f64]) -> Result<(), AutocorrelationError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(AutocorrelationError::NonFinite)
    } else {
        Ok(())
    }
}

fn finite(value: f64) -> Result<f64, AutocorrelationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AutocorrelationError::NonFinite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_acf_matches_hand_computed_variance_correlations_and_bartlett_errors() {
        let result = sample_acf(&[1.0, 2.0, 3.0, 4.0], 3).unwrap();
        assert_eq!(result.values.variance, 1.25);
        assert_eq!(result.values.correlations, [0.25, -0.3, -0.45]);
        assert_eq!(result.standard_errors[0], 0.5);
        assert_eq!(result.standard_errors[1], (1.125_f64 / 4.0).sqrt());
        assert_eq!(result.standard_errors[2], (1.305_f64 / 4.0).sqrt());
    }

    #[test]
    fn the_two_routines_normalize_in_different_orders() {
        // The input has mean zero, so both routines see the same products.
        // `sample_acf` divides the covariance and the variance by `n` before
        // taking their ratio; `ar_filtered_acf` divides once by `n * c0`.
        let input = [75.0, -53.0, -22.0];
        let sample = sample_acf(&input, 1).unwrap();
        let filtered = ar_filtered_acf(&input, 1, 0, 0).unwrap();
        assert_eq!(sample.values.correlations[0].to_bits(), 0xbfd428a5ce47fea0);
        assert_eq!(filtered.correlations[0].to_bits(), 0xbfd428a5ce47fe9f);
    }

    #[test]
    fn invalid_lag_counts_and_zero_variance_are_errors() {
        assert_eq!(sample_acf(&[], 1), Err(AutocorrelationError::InvalidInput));
        assert_eq!(
            sample_acf(&[1.0], 0),
            Err(AutocorrelationError::InvalidInput)
        );
        assert_eq!(
            sample_acf(&[1.0], 2),
            Err(AutocorrelationError::InvalidInput)
        );
        // A constant series has zero variance about its mean, but not about
        // zero, which is what `ar_filtered_acf` measures.
        assert_eq!(
            sample_acf(&[2.0, 2.0], 1),
            Err(AutocorrelationError::ZeroVariance)
        );
        assert_eq!(
            ar_filtered_acf(&[], 1, 0, 0),
            Err(AutocorrelationError::InvalidInput)
        );
        assert_eq!(
            ar_filtered_acf(&[0.0], 0, 0, 0),
            Err(AutocorrelationError::ZeroVariance)
        );
        assert_eq!(
            ar_filtered_acf(&[2.0], 0, 0, 0).unwrap().correlations,
            [0.0]
        );
    }

    #[test]
    fn nonfinite_inputs_and_intermediate_overflow_are_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                sample_acf(&[1.0, bad], 1),
                Err(AutocorrelationError::NonFinite)
            );
            assert_eq!(
                ar_filtered_acf(&[1.0, bad], 1, 0, 0),
                Err(AutocorrelationError::NonFinite)
            );
        }
        assert_eq!(
            sample_acf(&[1e308, 1e308], 1),
            Err(AutocorrelationError::NonFinite)
        );
        assert_eq!(
            ar_filtered_acf(&[1e308], 1, 0, 0),
            Err(AutocorrelationError::NonFinite)
        );
    }

    #[test]
    fn ar_filtered_lag_count_is_log_squared_with_a_short_series_reset() {
        let values = (0..80)
            .map(|i| f64::from((i * 17 % 31) + 1))
            .collect::<Vec<_>>();
        assert_eq!(
            ar_filtered_acf(&values, 0, 0, 0)
                .unwrap()
                .correlations
                .len(),
            19
        );
        assert_eq!(
            ar_filtered_acf(&values, 2, 5, 3)
                .unwrap()
                .correlations
                .len(),
            19
        );
        let small = &values[..8];
        assert_eq!(
            ar_filtered_acf(small, 50, 0, 0).unwrap().correlations.len(),
            6
        );
        assert_eq!(
            ar_filtered_acf(small, 1, 0, 0).unwrap().correlations.len(),
            4
        );
        assert_eq!(
            ar_filtered_acf(small, 0, usize::MAX, 1),
            Err(AutocorrelationError::InvalidInput)
        );
        assert!(ar_filtered_acf(&values, 0, 0, usize::MAX).is_err());
        let long = (0..2000)
            .map(|i| f64::from((i * 17 % 31) + 1))
            .collect::<Vec<_>>();
        assert!(ar_filtered_acf(&long, 0, 0, 0).unwrap().correlations.len() > 50);
    }
}
