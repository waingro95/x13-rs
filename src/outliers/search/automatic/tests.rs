use super::*;
use crate::arima::{ArmaStart, fit_automatic};
use crate::error::{NumericalDetail, NumericalFailure};
use crate::regression::model::RegressionModel;
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, RegArimaSpec};

fn white_noise_model() -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(0, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap()
}

#[test]
fn a_nonconverged_fit_is_rejected_with_its_iteration_count() {
    let series = (0..40)
        .map(|index| if index % 2 == 0 { -1.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(white_noise_model())
        .resolve_explicit()
        .unwrap();
    let fit = |incoming_convergence| {
        fit_automatic(
            &series,
            (&series).into(),
            axis,
            spec,
            RegressionModel::new(),
            ArmaStart::Default,
            incoming_convergence,
        )
        .unwrap()
    };
    let nonconverged = fit(false);
    let iterations = nonconverged.nonlinear_iterations();
    assert_eq!(
        accept_automatic_fit(nonconverged),
        Err(Error::EstimationDidNotConverge { iterations })
    );
    assert!(accept_automatic_fit(fit(true)).is_ok());
}

#[test]
fn singular_arma_covariance_warning_does_not_reject_a_converged_fit() {
    let series = vec![0.0; 40];
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let model = ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = RegArimaSpec::new(model).resolve_explicit().unwrap();
    let fit = fit_automatic(
        &series,
        (&series).into(),
        axis,
        spec,
        RegressionModel::new(),
        ArmaStart::Default,
        true,
    )
    .unwrap();
    let accepted = accept_automatic_fit(fit).unwrap();
    assert_eq!(
        accepted.arma_covariance,
        crate::api::result::ArmaCovariance::Singular
    );
    assert!(
        accepted
            .warnings
            .contains(&crate::Warning::SingularArmaCovariance)
    );
}

#[test]
fn only_the_residual_search_starts_from_a_nonconverged_fit() {
    let series = vec![0.0; 48];
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let spec = RegArimaSpec::new(white_noise_model())
        .resolve_explicit()
        .unwrap();
    let nonconverged = || {
        AutomaticFit::test_with_source_nonconvergence(
            &series,
            (&series).into(),
            axis,
            spec,
            RegressionModel::from_spec(spec, axis, series.len()).unwrap(),
            ArmaStart::Default,
            true,
        )
        .unwrap()
    };
    let fit = nonconverged();
    assert!(fit.as_converged().is_none());
    let iterations = fit.nonlinear_iterations();
    // The residual search tests candidates first, and a zero series has no
    // robust scale to test them against.
    assert_eq!(
        search_automatic_residuals(
            AutomaticInput::Untransformed(&series),
            (&series).into(),
            fit,
            None,
            OutlierSpec::default(),
            vec![],
        )
        .map(|search| search.result),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::ZeroOutlierRobustScale
        ))),
    );
    assert_eq!(
        search_automatic(
            &series,
            (&series).into(),
            nonconverged(),
            OutlierSpec::default(),
            vec![],
        ),
        Err(Error::EstimationDidNotConverge { iterations }),
    );
}
