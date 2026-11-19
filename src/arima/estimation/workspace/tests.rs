use crate::arima::estimation::*;
use crate::regarima::{OutlierType, ResolvedArimaModel};
use crate::{ArimaModel, ArimaOrder, Frequency, RegArimaSpec};

/// A fit with the explicit-model tolerances, without `fit`'s length check.
fn fit_explicit(
    values: &[f64],
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
) -> FittedRegArima {
    let request = FitRequest {
        original: values.into(),
        axis,
        spec,
        regression_model,
        start,
    };
    fit_attempt(
        values,
        request,
        FitControl::fresh(EstimationPolicy::EXPLICIT),
    )
    .unwrap()
    .into_completed()
    .unwrap()
}

fn model() -> ResolvedArimaModel {
    ArimaModel::new(
        ArimaOrder::new(1, 0, 2).unwrap(),
        ArimaOrder::new(1, 0, 1).unwrap(),
    )
    .unwrap()
    .into()
}

/// A (0 0 1)(0 0 1) or (0 1 1)(0 1 1) model without regressors.
fn ma_spec(differences: u8) -> ResolvedRegArimaSpec {
    RegArimaSpec::new(
        ArimaModel::new(
            ArimaOrder::new(0, differences, 1).unwrap(),
            ArimaOrder::new(0, differences, 1).unwrap(),
        )
        .unwrap(),
    )
    .resolve_explicit()
    .unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn a_new_workspace_has_one_positive_zero_per_arma_parameter() {
    let model = model();
    let workspace = ArmaWorkspace::reset(model);
    assert_eq!(workspace.model(), model);
    assert_eq!(bits(workspace.diagonal()), [0; 5]);
}

#[test]
fn a_previous_start_carries_its_workspace_to_a_fit_of_the_same_model() {
    let receiving = model();
    let expanded = ExpandedModel::from_spec(receiving, Frequency::Monthly);
    let mut workspace = ArmaWorkspace::reset(receiving);
    let diagonal = [
        0x3ff1_9999_9999_999a,
        0x8000_0000_0000_0000,
        0x7ff8_0000_0000_0042,
        0x4002_0000_0000_0000,
        0xbff8_0000_0000_0000,
    ];
    workspace.replace_diagonal(diagonal.map(f64::from_bits).to_vec());
    let (parameters, carried) = starting_parameters_with_workspace(
        ArmaStart::PreviousWithWorkspace {
            parameters: vec![0.1; 5],
            workspace: workspace.clone(),
        },
        &expanded,
        receiving,
    )
    .unwrap();
    assert_eq!(parameters, vec![0.1; 5]);
    assert_eq!(bits(carried.diagonal()), diagonal);

    // Another model with as many parameters does not take the workspace.
    let other = ArimaModel::new(
        ArimaOrder::new(2, 0, 1).unwrap(),
        ArimaOrder::new(1, 0, 1).unwrap(),
    )
    .unwrap()
    .into();
    assert!(matches!(
        starting_parameters_with_workspace(
            ArmaStart::PreviousWithWorkspace {
                parameters: vec![0.1; 5],
                workspace,
            },
            &ExpandedModel::from_spec(other, Frequency::Monthly),
            other,
        ),
        Err(Error::Internal(_))
    ));
}

#[test]
fn a_fit_that_skips_the_optimizer_keeps_the_incoming_workspace() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let values = vec![0.0; 13];
    let spec = ma_spec(0);
    let mut workspace = ArmaWorkspace::reset(spec.model());
    let diagonal = [0x8000_0000_0000_0000, 0x7ff8_0000_0000_0042];
    workspace.replace_diagonal(diagonal.map(f64::from_bits).to_vec());
    let regression = RegressionModel::from_spec(spec, axis, values.len()).unwrap();
    let fitted = fit_explicit(
        &values,
        axis,
        spec,
        regression,
        ArmaStart::PreviousWithWorkspace {
            parameters: vec![0.1, 0.1],
            workspace,
        },
    );
    assert_eq!(fitted.nonlinear_iterations, 0);
    assert_eq!(bits(fitted.arma_workspace.diagonal()), diagonal);
}

#[test]
fn default_and_previous_starts_begin_with_a_zero_workspace() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let values = vec![0.0; 13];
    let spec = ma_spec(0);
    for start in [ArmaStart::Default, ArmaStart::Previous(vec![0.1, 0.1])] {
        let regression = RegressionModel::from_spec(spec, axis, values.len()).unwrap();
        let fitted = fit_explicit(&values, axis, spec, regression, start);
        assert_eq!(fitted.nonlinear_iterations, 0);
        assert_eq!(bits(fitted.arma_workspace.diagonal()), [0, 0]);
    }
}

#[test]
fn a_fit_without_arma_parameters_has_an_empty_workspace() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let series = (0..48)
        .map(|index| 100.0 + (index % 7) as f64)
        .collect::<Vec<_>>();
    let spec = RegArimaSpec::new(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
    )
    .resolve_explicit()
    .unwrap();
    let regression = RegressionModel::from_spec(spec, axis, series.len()).unwrap();
    let estimated = estimate(
        &series,
        (&series).into(),
        axis,
        spec,
        regression,
        ArmaStart::Default,
    )
    .unwrap();
    assert!(estimated.arma_workspace.diagonal().is_empty());
    assert_eq!(
        estimated.arma_covariance_diagnostic,
        ArmaCovarianceDiagnostic::None
    );
    assert_eq!(estimated.arma_covariance, ArmaCovariance::NotRequired);
}

#[test]
fn the_covariance_step_rewrites_the_workspace_of_a_zero_variance_fit() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let values = (0..13)
        .map(|index| (100.0 + ((17 * index * index + 37 * index + 11) % 71) as f64) * 1e-10)
        .collect::<Vec<_>>();
    let spec = ma_spec(1);
    let regression = RegressionModel::from_spec(spec, axis, values.len()).unwrap();
    let fitted = fit_explicit(&values, axis, spec, regression, ArmaStart::Default);
    let before_covariance = fitted.arma_workspace.diagonal().to_vec();
    let estimated = finish_estimate(fitted).unwrap();
    assert_eq!(estimated.innovation_variance, 0.0);
    assert_eq!(estimated.arma_covariance.matrix().unwrap(), [0.0; 4]);
    assert_eq!(estimated.arma_workspace.diagonal().len(), 2);
    assert!(
        estimated
            .arma_workspace
            .diagonal()
            .iter()
            .any(|value| *value != 0.0)
    );
    assert_ne!(estimated.arma_workspace.diagonal(), before_covariance);
    assert_eq!(
        estimated.arma_covariance_diagnostic,
        ArmaCovarianceDiagnostic::None
    );
}

#[test]
fn a_singular_covariance_is_reported_and_the_workspace_kept() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let mut values = vec![100.0; 13];
    values[6] = 150.0;
    let mut ao = vec![0.0; 13];
    ao[6] = 1.0;
    let mut regression = RegressionModel::new();
    regression
        .insert_outlier(
            OutlierType::Additive,
            axis.period_at(6).unwrap(),
            Frequency::Quarterly,
            ao,
            0.7,
        )
        .unwrap();
    let estimated = estimate(
        &values,
        (&values).into(),
        axis,
        ma_spec(1),
        regression,
        ArmaStart::Default,
    )
    .unwrap();
    assert_eq!(estimated.arma_covariance, ArmaCovariance::Singular);
    assert_eq!(
        estimated.arma_covariance_diagnostic,
        ArmaCovarianceDiagnostic::Singular
    );
    assert_eq!(estimated.arma_workspace.diagonal().len(), 2);
}
