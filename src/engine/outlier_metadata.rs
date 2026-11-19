//! The outliers of a fitted model, as the pipelines report them.
use crate::arima::EstimatedRegArima;
use crate::regarima::DetectedOutlier;
use crate::regression::model::RegressorKind;

pub(super) fn detected_outliers(estimated: &EstimatedRegArima) -> Vec<DetectedOutlier> {
    estimated
        .regressor_kinds
        .iter()
        .zip(&estimated.regression_coefficients)
        .enumerate()
        .filter_map(|(index, (kind, estimate))| {
            let RegressorKind::Outlier { kind, period, .. } = *kind else {
                return None;
            };
            let standard_error = estimated
                .regression_standard_errors
                .as_ref()
                .map(|errors| errors[index]);
            Some(DetectedOutlier {
                kind,
                period,
                estimate: *estimate,
                standard_error,
                t_value: standard_error.map(|error| *estimate / error),
            })
        })
        .collect()
}
