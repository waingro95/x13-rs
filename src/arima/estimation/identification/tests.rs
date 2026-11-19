use super::*;
use crate::arima::estimation::ArmaStart;
use crate::error::{NumericalDetail, NumericalFailure};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, RegArimaSpec, Transform};

#[test]
fn the_conditional_retry_starts_from_the_exact_fits_workspace() {
    let series = vec![0.0; 13];
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 1).unwrap(),
        )
        .unwrap(),
    )
    .resolve_explicit()
    .unwrap();
    let regression = RegressionModel::from_spec(spec, axis, series.len()).unwrap();
    let request = FitRequest {
        original: (&series).into(),
        axis,
        spec,
        regression_model: regression,
        start: ArmaStart::Default,
    };
    let mut exact = fit_attempt(
        &series,
        request.clone(),
        FitControl::fresh(EstimationPolicy::AUTOMATIC_IDENTIFICATION),
    )
    .unwrap();
    let bits = [0x8000_0000_0000_0000, 0x7ff8_0000_0000_0051];
    exact
        .fitted
        .arma_workspace
        .replace_diagonal(bits.map(f64::from_bits).to_vec());
    let conditional = retry_conditional(&series, request, &exact).unwrap();
    assert_eq!(conditional.fitted.nonlinear_iterations, 0);
    assert_eq!(
        conditional
            .fitted
            .arma_workspace
            .diagonal()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        bits,
    );
    assert_eq!(conditional.fitted.initial_log_determinant_correction, 0.0);
}

#[test]
fn an_invalid_previous_ma_start_is_an_error() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let model = ArimaModel::new(
        ArimaOrder::new(0, 0, 1).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = RegArimaSpec::new(model)
        .with_transform(Transform::None)
        .resolve_explicit()
        .unwrap();
    let series = (0..48)
        .map(|index| 100.0 + (index % 7) as f64)
        .collect::<Vec<_>>();
    let regression = RegressionModel::from_spec(spec, axis, series.len()).unwrap();
    assert!(matches!(
        fit_identification(
            &series,
            FitRequest {
                original: (&series).into(),
                axis,
                spec,
                regression_model: regression,
                start: ArmaStart::Previous(vec![1.1]),
            }
        ),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::InvalidMaStartingRoots { .. }
        )))
    ));
}

#[test]
fn a_singular_regression_is_an_error_and_is_not_retried() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let duplicate = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut regression = RegressionModel::new();
    regression
        .insert_outlier(
            crate::regarima::OutlierType::TemporaryChange,
            axis.period_at(6).unwrap(),
            crate::Frequency::Quarterly,
            duplicate.clone(),
            0.7,
        )
        .unwrap();
    regression
        .insert_outlier(
            crate::regarima::OutlierType::Additive,
            axis.period_at(2).unwrap(),
            crate::Frequency::Quarterly,
            duplicate,
            0.7,
        )
        .unwrap();
    let response = [1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 5.0, 6.0];
    let model = ArimaModel::new(
        ArimaOrder::new(0, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = RegArimaSpec::new(model).resolve_explicit().unwrap();
    assert_eq!(
        fit_identification(
            &response,
            FitRequest {
                original: (&response).into(),
                axis,
                spec,
                regression_model: regression,
                start: ArmaStart::Default,
            }
        )
        .err(),
        Some(Error::SingularRegression {
            pivot: 1,
            variable: crate::regarima::RegressionVariable::Outlier {
                kind: crate::regarima::OutlierType::TemporaryChange,
                period: axis.period_at(6).unwrap(),
            },
        })
    );
}

#[test]
fn a_mean_fit_rejects_a_series_shorter_than_its_lags() {
    // (1 1 1)(1 1 1)4 loses 5 observations to differencing and its AR and MA
    // lags need 10 more. At 14 observations the exact filter would index past
    // its rows; the length check rejects the fit first.
    let model = ArimaModel::new(
        ArimaOrder::new(1, 1, 1).unwrap(),
        ArimaOrder::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let series = (0..14)
        .map(|t| 100.0 + f64::from(t) + 1.5 * f64::from(t % 5))
        .collect::<Vec<_>>();
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    assert_eq!(
        fit_local_mean(&series, axis, model, MeanFitBudget::Exact200).err(),
        Some(Error::InsufficientEffectiveObservations {
            actual: 9,
            minimum: 10
        })
    );
}
