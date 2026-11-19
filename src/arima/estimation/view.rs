use super::{ArmaCovarianceDiagnostic, ArmaStart, ArmaWorkspace, EstimatedRegArima};
use crate::arima::exact::ExactFilterWorkspace;
use crate::regression::model::RegressorKind;
use crate::time::TimeAxis;

#[derive(Debug, Clone, Copy)]
pub(crate) struct FittedNumericalView<'a> {
    pub(crate) axis: TimeAxis,
    pub(crate) transformed: &'a [f64],
    pub(crate) parameters: &'a [f64],
    pub(crate) arma_workspace: &'a ArmaWorkspace,
    pub(crate) arma_covariance_diagnostic: ArmaCovarianceDiagnostic,
    pub(crate) residuals: &'a [f64],
    pub(crate) regression_coefficients: &'a [f64],
    pub(crate) regressor_kinds: &'a [RegressorKind],
    pub(crate) outlier_test_factor: &'a [f64],
    pub(crate) filter_workspace: &'a ExactFilterWorkspace,
}

impl FittedNumericalView<'_> {
    pub(crate) fn refit_start(&self) -> ArmaStart {
        ArmaStart::PreviousWithWorkspace {
            parameters: self.parameters.to_vec(),
            workspace: self.arma_workspace.clone(),
        }
    }
}

impl EstimatedRegArima {
    pub(crate) fn numerical_view(&self) -> FittedNumericalView<'_> {
        FittedNumericalView {
            axis: self.axis,
            transformed: &self.transformed,
            parameters: &self.parameters,
            arma_workspace: &self.arma_workspace,
            arma_covariance_diagnostic: self.arma_covariance_diagnostic,
            residuals: &self.residuals,
            regression_coefficients: &self.regression_coefficients,
            regressor_kinds: &self.regressor_kinds,
            outlier_test_factor: &self.outlier_test_factor,
            filter_workspace: &self.forecast_filter_workspace,
        }
    }
}
