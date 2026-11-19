//! Residual autocorrelations with their Ljung-Box statistics, and the lag
//! `mdlchk` reads them at.

use std::num::NonZeroUsize;

use crate::numerics::autocorrelation::{AutocorrelationError, sample_acf};
use crate::numerics::probability::{ProbabilityError, chi_square_upper_tail};
use crate::time::Frequency;

/// One lag of the residual autocorrelation table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResidualAcfRow {
    pub(crate) lag: usize,
    pub(crate) correlation: f64,
    pub(crate) standard_error: f64,
    /// The Ljung-Box Q statistic through this lag.
    pub(crate) ljung_box_q: f64,
    /// The lag less the number of estimated ARMA coefficients, floored at
    /// zero.
    pub(crate) degrees_of_freedom: usize,
    /// The chi-square upper tail of Q, or zero without degrees of freedom.
    pub(crate) p_value: f64,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum DiagnosticError {
    #[error("invalid residual ACF lag or sample size")]
    InvalidLagOrSampleSize,
    #[error("residual ACF calculation is nonfinite")]
    NonFiniteQ,
    #[error(transparent)]
    Autocorrelation(#[from] AutocorrelationError),
    #[error(transparent)]
    Probability(#[from] ProbabilityError),
}

/// The autocorrelations of `values` at lags 1 to `max_lag`, each with its
/// Ljung-Box statistic.
pub(crate) fn residual_acf(
    values: &[f64],
    max_lag: usize,
    estimated_arma_count: usize,
) -> Result<Vec<ResidualAcfRow>, DiagnosticError> {
    if values.is_empty() || max_lag == 0 || max_lag >= values.len() {
        return Err(DiagnosticError::InvalidLagOrSampleSize);
    }
    let acf = sample_acf(values, max_lag)?;
    let n = values.len() as f64;
    let mut squared_sum = 0.0;
    let mut rows = Vec::with_capacity(max_lag);
    for (index, (&correlation, &standard_error)) in acf
        .values
        .correlations
        .iter()
        .zip(&acf.standard_errors)
        .enumerate()
    {
        let lag = index + 1;
        squared_sum += correlation * correlation / (n - lag as f64);
        let ljung_box_q = squared_sum * n * (n + 2.0);
        if !squared_sum.is_finite() || !ljung_box_q.is_finite() {
            return Err(DiagnosticError::NonFiniteQ);
        }
        let degrees_of_freedom = lag.saturating_sub(estimated_arma_count);
        let p_value = match NonZeroUsize::new(degrees_of_freedom) {
            Some(df) => chi_square_upper_tail(ljung_box_q, df)?,
            None => 0.0,
        };
        rows.push(ResidualAcfRow {
            lag,
            correlation,
            standard_error,
            ljung_box_q,
            degrees_of_freedom,
            p_value,
        });
    }
    Ok(rows)
}

/// The lag `mdlchk` reads the Ljung-Box statistic at: 24 for a monthly series
/// and 16 for a quarterly one, or half the residuals when there are no more
/// than that. A quarterly series of 18 to 22 residuals uses lag 6.
pub(crate) fn model_check_lag(frequency: Frequency, effective_observations: usize) -> usize {
    let lag = match frequency {
        Frequency::Monthly => 24,
        Frequency::Quarterly if (18..=22).contains(&effective_observations) => 6,
        Frequency::Quarterly => 16,
    };
    if lag >= effective_observations {
        effective_observations / 2
    } else {
        lag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lag_without_degrees_of_freedom_has_a_zero_p_value() {
        let values = [-1.0, 1.0, -1.0, 1.0];
        let rows = residual_acf(&values, 3, 1).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].ljung_box_q, 4.5);
        assert_eq!((rows[0].degrees_of_freedom, rows[0].p_value), (0, 0.0));
        assert_eq!(rows[1].degrees_of_freedom, 1);
        assert_eq!(
            residual_acf(&values, 1, usize::MAX).unwrap()[0].p_value,
            0.0
        );
        assert!(matches!(
            residual_acf(&values, 0, 0),
            Err(DiagnosticError::InvalidLagOrSampleSize)
        ));
        assert!(matches!(
            residual_acf(&values, 4, 0),
            Err(DiagnosticError::InvalidLagOrSampleSize)
        ));
        assert!(matches!(
            residual_acf(&values, 5, 0),
            Err(DiagnosticError::InvalidLagOrSampleSize)
        ));
        assert!(matches!(
            residual_acf(&[], 1, 0),
            Err(DiagnosticError::InvalidLagOrSampleSize)
        ));
        let alternating = (0..200)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let row = residual_acf(&alternating, 1, 0).unwrap();
        assert!(row[0].ljung_box_q >= 90.0);
        assert_eq!(row[0].p_value, 0.0);
    }

    #[test]
    fn constant_and_nonfinite_values_are_autocorrelation_errors() {
        assert!(matches!(
            residual_acf(&[2.0, 2.0], 1, 0),
            Err(DiagnosticError::Autocorrelation(
                AutocorrelationError::ZeroVariance
            ))
        ));
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                residual_acf(&[1.0, bad], 1, 0),
                Err(DiagnosticError::Autocorrelation(
                    AutocorrelationError::NonFinite
                ))
            ));
        }
    }

    #[test]
    fn the_checked_lag_is_half_of_a_short_sample() {
        for (frequency, n, expected) in [
            (Frequency::Monthly, 23, 11),
            (Frequency::Monthly, 24, 12),
            (Frequency::Monthly, 25, 24),
            (Frequency::Quarterly, 15, 7),
            (Frequency::Quarterly, 16, 8),
            (Frequency::Quarterly, 17, 16),
            (Frequency::Quarterly, 18, 6),
            (Frequency::Quarterly, 22, 6),
            (Frequency::Quarterly, 23, 16),
            (Frequency::Monthly, 2, 1),
            (Frequency::Quarterly, 2, 1),
        ] {
            assert_eq!(
                model_check_lag(frequency, n),
                expected,
                "{frequency:?}, n={n}"
            );
        }
        assert_eq!(model_check_lag(Frequency::Quarterly, 80), 16);
    }
}
