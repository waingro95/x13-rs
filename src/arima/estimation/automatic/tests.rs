use super::*;
use crate::arima::estimation::fit_attempt;
use crate::{ArimaModel, ArimaOrder, RegArimaSpec};

impl AutomaticFit {
    /// A fit of `series` with its convergence flag cleared, for tests of what
    /// the automatic-model procedure does with a fit that did not converge.
    pub(crate) fn test_with_source_nonconvergence(
        series: &[f64],
        original_series: crate::arima::FitContext<'_>,
        axis: TimeAxis,
        spec: ResolvedRegArimaSpec,
        regression_model: RegressionModel,
        start: ArmaStart,
        incoming_convergence: bool,
    ) -> Result<Self, crate::Error> {
        let request = FitRequest {
            original: original_series,
            axis,
            spec,
            regression_model,
            start,
        };
        let control = FitControl::carried(EstimationPolicy::EXPLICIT, incoming_convergence);
        let mut attempt = fit_attempt(series, request, control)?;
        attempt.fitted.status.converged = false;
        finish_automatic_attempt(attempt)
    }
}

fn no_arma_terms() -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(0, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap()
}

fn alternating(low: f64, high: f64) -> Vec<f64> {
    (0..40)
        .map(|index| if index % 2 == 0 { low } else { high })
        .collect()
}

/// Fits `response`, already on the transformed scale, with no regressors.
fn fit_transformed_response(
    response: &[f64],
    original: &[f64],
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    incoming_convergence: bool,
) -> Result<AutomaticFit, Error> {
    fit_automatic_input(
        AutomaticInput::Transformed(response),
        original.into(),
        axis,
        spec,
        RegressionModel::new(),
        ArmaStart::Default,
        incoming_convergence,
    )
}

#[test]
fn a_fit_without_arma_terms_reports_the_incoming_convergence() {
    let values = alternating(-1.0, 1.0);
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(no_arma_terms())
        .resolve_explicit()
        .unwrap();
    let fit = |incoming_convergence| {
        fit_automatic(
            &values,
            (&values).into(),
            axis,
            spec,
            RegressionModel::new(),
            ArmaStart::Default,
            incoming_convergence,
        )
        .unwrap()
    };
    let after_failure = fit(false);
    let after_success = fit(true);
    assert!(!after_failure.pre_covariance_status().converged);
    assert!(after_failure.as_converged().is_none());
    assert!(after_success.pre_covariance_status().converged);
    assert!(after_success.as_converged().is_some());
    assert_eq!(
        after_failure.numerical_view().residuals,
        after_success.numerical_view().residuals
    );
}

#[test]
fn a_series_shorter_than_the_model_needs_is_rejected() {
    let fit = |values: &[f64], axis, model| {
        fit_automatic(
            values,
            values.into(),
            axis,
            RegArimaSpec::new(model).resolve_explicit().unwrap(),
            RegressionModel::new(),
            ArmaStart::Default,
            true,
        )
        .err()
    };
    assert_eq!(
        fit(&[], TimeAxis::monthly(2000, 1).unwrap(), no_arma_terms()),
        Some(Error::InsufficientEffectiveObservations {
            actual: 0,
            minimum: 1
        })
    );
    let model = ArimaModel::new(
        ArimaOrder::new(3, 1, 0).unwrap(),
        ArimaOrder::new(1, 1, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(
        fit(&[1.0; 11], TimeAxis::quarterly(2000, 1).unwrap(), model),
        Some(Error::InsufficientEffectiveObservations {
            actual: 6,
            minimum: 7
        })
    );
}

#[test]
fn a_transformed_response_is_fitted_as_given() {
    // Taking the log of the exponential would not give these bits back.
    assert_ne!(0.1_f64.exp().ln().to_bits(), 0.1_f64.to_bits());
    let response = alternating(-0.6, 0.1);
    let original = vec![2.0; response.len()];
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let log_spec = RegArimaSpec::new(no_arma_terms())
        .with_transform(Transform::Log)
        .resolve_explicit()
        .unwrap();
    let identity_spec = RegArimaSpec::new(no_arma_terms())
        .resolve_explicit()
        .unwrap();
    let actual = fit_transformed_response(&response, &original, axis, log_spec, true).unwrap();
    let identity = fit_automatic(
        &response,
        (&response).into(),
        axis,
        identity_spec,
        RegressionModel::new(),
        ArmaStart::Default,
        true,
    )
    .unwrap();
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(actual.numerical_view().transformed), bits(&response));
    assert_eq!(
        actual.numerical_view().residuals,
        identity.numerical_view().residuals
    );
    assert_eq!(actual.innovation_variance(), identity.innovation_variance());
    assert_eq!(
        actual.as_converged().unwrap().spec.transform(),
        Transform::Log
    );
    assert_ne!(
        actual.as_converged().unwrap().likelihood,
        identity.as_converged().unwrap().likelihood
    );
}

#[test]
fn the_original_series_changes_only_the_likelihood_adjustment() {
    let response = alternating(-0.6, 0.1);
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(no_arma_terms())
        .with_transform(Transform::Log)
        .resolve_explicit()
        .unwrap();
    let fit = |value| {
        fit_transformed_response(&response, &vec![value; response.len()], axis, spec, true).unwrap()
    };
    let low = fit(2.0);
    let high = fit(3.0);
    assert_eq!(
        low.numerical_view().residuals,
        high.numerical_view().residuals
    );
    assert_eq!(low.innovation_variance(), high.innovation_variance());
    let low_likelihood = low.as_converged().unwrap().likelihood.unwrap();
    let high_likelihood = high.as_converged().unwrap().likelihood.unwrap();
    assert_eq!(
        low_likelihood.log_likelihood,
        high_likelihood.log_likelihood
    );
    let expected_aic_delta = 2.0 * response.len() as f64 * (3.0_f64 / 2.0).ln();
    assert!(((high_likelihood.aic - low_likelihood.aic) - expected_aic_delta).abs() < 1.0e-12);
}

#[test]
fn an_unconverged_fit_is_the_same_for_either_kind_of_input() {
    let response = alternating(-1.0, 1.0);
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(no_arma_terms())
        .resolve_explicit()
        .unwrap();
    let untransformed = fit_automatic(
        &response,
        (&response).into(),
        axis,
        spec,
        RegressionModel::new(),
        ArmaStart::Default,
        false,
    )
    .unwrap();
    let transformed = fit_transformed_response(&response, &response, axis, spec, false).unwrap();
    assert!(transformed.as_converged().is_none());
    assert_eq!(
        transformed.pre_covariance_status(),
        untransformed.pre_covariance_status()
    );
    assert_eq!(
        transformed.numerical_view().residuals,
        untransformed.numerical_view().residuals
    );
    assert_eq!(
        transformed.innovation_variance(),
        untransformed.innovation_variance()
    );
    assert_eq!(transformed.warnings(), untransformed.warnings());
}

#[test]
fn a_transformed_response_gets_the_checks_of_an_untransformed_one() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(no_arma_terms())
        .with_transform(Transform::Log)
        .resolve_explicit()
        .unwrap();
    let response = vec![0.1; 40];
    let original = vec![2.0; 40];
    let fit = |values: &[f64], originals: &[f64], axis| {
        fit_transformed_response(values, originals, axis, spec, true).err()
    };

    let mut nonfinite_response = response.clone();
    nonfinite_response[7] = f64::NAN;
    assert_eq!(
        fit(&nonfinite_response, &original, axis),
        Some(Error::NonFiniteValue { index: 7 })
    );
    let mut nonfinite_original = original.clone();
    nonfinite_original[9] = f64::INFINITY;
    assert_eq!(
        fit(&response, &nonfinite_original, axis),
        Some(Error::NonFiniteValue { index: 9 })
    );
    let mut nonpositive_original = original.clone();
    nonpositive_original[11] = 0.0;
    assert_eq!(
        fit(&response, &nonpositive_original, axis),
        Some(Error::NonPositiveValue { index: 11 })
    );
    let past_the_last_period = TimeAxis::monthly(i32::MAX, 12).unwrap();
    assert!(matches!(
        fit(&response, &original, past_the_last_period),
        Some(Error::InvalidPeriod(_))
    ));
}
