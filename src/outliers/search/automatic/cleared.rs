//! A fit whose outlier regressors `clrotl.f` dropped without refitting, and
//! the search that restarts from it.

use super::accept_automatic_fit;
use crate::Error;
use crate::arima::model::ExpandedModel;
use crate::arima::{
    AutomaticInput, EstimatedRegArima, FitContext, FittedNumericalView, fit_automatic_input,
    statistics_from_log_likelihood, uncertainty_is_available,
};
use crate::numerics::packed::{packed_inverse, packed_len};
use crate::outliers::search::{
    OutlierSearchResult, SearchFit, SearchInitial, SearchProblem, outlier_failure_from,
    search_with_initial_fit,
};
use crate::regarima::OutlierSpec;
use crate::regression::gls::{regression_covariance, regression_standard_errors};
use crate::regression::model::{RegressionModel, RegressorKind};

/// A fit, and the regression left once its outliers are dropped.
///
/// Build 62's `clrotl.f` removes the outlier regressors without refitting, so
/// the next search tests the surviving regressors against the last fit's
/// numbers.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClearedRegressionFit {
    last_fitted: Box<EstimatedRegArima>,
    /// `None` while the last fit's regression is still the current one.
    cleared: Option<Box<ClearedRegression>>,
}

/// The regressors that survive, with their coefficients from the last fit.
#[derive(Debug, Clone, PartialEq)]
struct ClearedRegression {
    model: RegressionModel,
    coefficients: Vec<f64>,
    kinds: Vec<RegressorKind>,
}

impl ClearedRegressionFit {
    /// Build 62 `clrotl.f`: drops every outlier regressor of `fit`.
    pub(crate) fn clear_automatic_outliers(fit: EstimatedRegArima) -> Result<Self, Error> {
        let view = fit.numerical_view();
        debug_assert!(
            fit.regression_model
                .kinds()
                .eq(view.regressor_kinds.iter().copied())
        );
        let (kinds, coefficients): (Vec<_>, Vec<_>) = view
            .regressor_kinds
            .iter()
            .copied()
            .zip(view.regression_coefficients.iter().copied())
            .filter(|(kind, _)| !matches!(kind, RegressorKind::Outlier { .. }))
            .unzip();
        if kinds.len() == view.regressor_kinds.len() {
            return Ok(Self::synchronized(fit));
        }
        let mut model = fit.regression_model.clone();
        for kind in view.regressor_kinds.iter().rev() {
            if let RegressorKind::Outlier { kind, period, .. } = kind {
                let removed = model.remove_outlier(*kind, *period);
                debug_assert!(removed, "the fit's outliers are its model's columns");
            }
        }
        let model = model
            .evaluated_over(view.axis, view.transformed.len())
            .map_err(outlier_failure_from)?;
        debug_assert!(model.kinds().eq(kinds.iter().copied()));
        Ok(Self {
            last_fitted: Box::new(fit),
            cleared: Some(Box::new(ClearedRegression {
                model,
                coefficients,
                kinds,
            })),
        })
    }

    /// A fit whose regression is the current one.
    fn synchronized(fit: EstimatedRegArima) -> Self {
        Self {
            last_fitted: Box::new(fit),
            cleared: None,
        }
    }

    pub(crate) fn last_fitted(&self) -> &EstimatedRegArima {
        &self.last_fitted
    }

    /// The current regression: the cleared one, or the last fit's.
    pub(crate) fn regression_model(&self) -> &RegressionModel {
        self.cleared
            .as_ref()
            .map_or(&self.last_fitted.regression_model, |cleared| &cleared.model)
    }

    /// The last fit's numbers, limited to the surviving regressors and their
    /// leading part of the packed factor.
    pub(crate) fn numerical_view(&self) -> FittedNumericalView<'_> {
        let mut view = self.last_fitted.numerical_view();
        if let Some(cleared) = &self.cleared {
            view.regressor_kinds = &cleared.kinds;
            view.regression_coefficients = &cleared.coefficients;
            view.outlier_test_factor =
                &view.outlier_test_factor[..packed_len(cleared.kinds.len() + 1)];
        }
        view
    }

    /// The last fit, if its regression is still the current one.
    pub(crate) fn as_synchronized(&self) -> Option<&EstimatedRegArima> {
        self.cleared.is_none().then_some(&self.last_fitted)
    }

    /// The last fit, or `self` back if its outliers have been cleared since.
    pub(crate) fn into_synchronized(self) -> Result<EstimatedRegArima, Self> {
        if self.cleared.is_some() {
            Err(self)
        } else {
            Ok(*self.last_fitted)
        }
    }

    /// The fit to report for the surviving regressors: their standard
    /// errors, covariance and likelihood statistics come from the last fit's
    /// factor, because Build 62's `autoer` does not refit after `clrotl`.
    pub(crate) fn into_reported_fit(
        self,
        original: FitContext<'_>,
    ) -> Result<EstimatedRegArima, Error> {
        let Some(cleared) = self.cleared else {
            return Ok(*self.last_fitted);
        };
        let mut fit = *self.last_fitted;
        debug_assert_eq!(original.raw().len(), fit.transformed.len());
        // Build 62 lowers the regressor count and keeps reading the same
        // packed `[X:y]` factor, so the surviving regressors take its leading
        // columns: their triangle first, then one more column as the response.
        let count = cleared.kinds.len();
        let normal_length = packed_len(count);
        let expanded = ExpandedModel::from_spec(fit.spec.model(), fit.axis.frequency());
        let loss = expanded.differencing_loss();
        let effective = fit.transformed.len() - loss;
        let fixed = cleared.model.fixed_length();
        let (standard_errors, covariance) = if uncertainty_is_available(fit.innovation_variance) {
            let mut inverse = fit.outlier_test_factor[..normal_length].to_vec();
            packed_inverse(&mut inverse, count);
            (
                Some(regression_standard_errors(
                    &inverse,
                    fit.innovation_variance,
                    count,
                )),
                Some(regression_covariance(
                    &inverse,
                    fit.innovation_variance,
                    effective,
                    count,
                )?),
            )
        } else {
            (None, None)
        };
        if let Some(last) = fit.likelihood {
            let adjustment =
                original.transformation_adjustment(fixed, loss, fit.spec.transform())?;
            fit.likelihood = Some(statistics_from_log_likelihood(
                effective,
                count + fit.parameters.len() + 1,
                last.log_likelihood,
                adjustment,
            ));
        }
        fit.regression_model = cleared.model;
        fit.regressor_kinds = cleared.kinds;
        fit.regression_coefficients = cleared.coefficients;
        fit.regression_standard_errors = standard_errors;
        fit.regression_covariance = covariance;
        fit.regression_factor = fit.outlier_test_factor[..normal_length].to_vec();
        fit.outlier_test_factor.truncate(packed_len(count + 1));
        Ok(fit)
    }
}

/// Searches from a fit whose outliers were just cleared. Every refit must
/// converge.
pub(crate) fn search_automatic_cleared_input(
    input: AutomaticInput<'_>,
    original: FitContext<'_>,
    initial: ClearedRegressionFit,
    outlier_spec: OutlierSpec,
) -> Result<OutlierSearchResult<ClearedRegressionFit>, Error> {
    let view = initial.numerical_view();
    let observation_count = view.transformed.len();
    debug_assert_eq!(input.values().len(), observation_count);
    debug_assert_eq!(original.raw().len(), observation_count);
    debug_assert!(
        !view
            .regressor_kinds
            .iter()
            .any(|kind| matches!(kind, RegressorKind::Outlier { .. }))
    );
    let axis = view.axis;
    let spec = initial.last_fitted.spec;
    let regression = initial.regression_model().clone();
    search_with_initial_fit(
        SearchProblem {
            observation_count,
            axis,
            spec,
            outlier_spec,
        },
        regression,
        SearchInitial::Automatic {
            fit: Box::new(initial),
            selected: vec![],
        },
        |regression, start| {
            let fitted = fit_automatic_input(input, original, axis, spec, regression, start, true)?;
            Ok(ClearedRegressionFit::synchronized(accept_automatic_fit(
                fitted,
            )?))
        },
    )
}

impl SearchFit for ClearedRegressionFit {
    fn numerical_view(&self) -> FittedNumericalView<'_> {
        self.numerical_view()
    }

    fn warnings_mut(&mut self) -> &mut Vec<crate::Warning> {
        &mut self.last_fitted.warnings
    }

    fn report(&self) -> crate::arima::FitReport {
        self.last_fitted.report()
    }
}

#[cfg(test)]
mod tests {
    use super::ClearedRegressionFit;
    use crate::arima::{ArmaStart, estimate};
    use crate::regarima::{LengthEffect, OutlierType, RegressionVariable};
    use crate::regression::model::{RegressionColumn, RegressionModel};
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, Frequency, RegArimaSpec};

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        ignore = "pins exact results of the Apple silicon math library"
    )]
    fn cleared_report_reads_the_surviving_regressors_from_the_last_factor() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let mut values = (0..48)
            .map(|index| 100.0 + (index as f64 * 0.7).sin() * 3.0 + index as f64 * 0.25)
            .collect::<Vec<_>>();
        values[17] += 9.0;
        for value in &mut values[30..] {
            *value += 4.0;
        }
        let spec = RegArimaSpec::new(
            ArimaModel::new(
                ArimaOrder::new(1, 0, 0).unwrap(),
                ArimaOrder::new(0, 0, 0).unwrap(),
            )
            .unwrap(),
        )
        .resolve_explicit()
        .unwrap();
        let mut regression = RegressionModel::new();
        for phase in [0.0, 1.5] {
            let column = (0..48)
                .map(|index| (index as f64 * 0.3 + phase).cos())
                .collect::<Vec<_>>();
            regression
                .push(RegressionColumn::with_values(
                    RegressionVariable::Length(LengthEffect::LeapYear),
                    column,
                ))
                .unwrap();
        }
        let mut additive = vec![0.0; 48];
        additive[17] = 1.0;
        regression
            .insert_outlier(
                OutlierType::Additive,
                axis.period_at(17).unwrap(),
                Frequency::Monthly,
                additive,
                0.7,
            )
            .unwrap();
        let level_shift = (0..48)
            .map(|index| if index < 30 { -1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        regression
            .insert_outlier(
                OutlierType::LevelShift,
                axis.period_at(30).unwrap(),
                Frequency::Monthly,
                level_shift,
                0.7,
            )
            .unwrap();
        let fit = estimate(
            &values,
            (&values).into(),
            axis,
            spec,
            regression,
            ArmaStart::Default,
        )
        .unwrap();

        let cleared = ClearedRegressionFit::clear_automatic_outliers(fit).unwrap();
        let view = cleared.numerical_view();
        assert_eq!(view.regressor_kinds.len(), 2);
        let view_factor = bits(view.outlier_test_factor);

        let reported = cleared.into_reported_fit((&values).into()).unwrap();
        let likelihood = reported.likelihood.unwrap();
        let actual = [
            view_factor,
            bits(reported.regression_standard_errors.as_deref().unwrap()),
            bits(reported.regression_covariance.as_deref().unwrap()),
            bits(&reported.regression_factor),
            bits(&reported.outlier_test_factor),
            bits(&[likelihood.aic, likelihood.bic, likelihood.hannan_quinn]),
        ];
        let expected: [Vec<u64>; 6] = [
            vec![
                0x3ff7_2670_f43e_ccce,
                0x3fca_6ae4_fbe3_a6eb,
                0x3ff7_1814_e9fe_d3f8,
                0x3f97_e4b0_e382_105d,
                0x3fac_6604_907a_5fe5,
                0x3ff6_9b1d_b170_775c,
            ],
            vec![0x3ff0_9100_0b7a_34a8, 0x3ff0_707e_dbcc_63ef],
            vec![
                0x3ff1_e60d_6126_05e6,
                0xbfc4_1cea_9075_1249,
                0x3ff1_a015_919c_fc1e,
            ],
            vec![
                0x3ff7_2670_f43e_ccce,
                0x3fca_6ae4_fbe3_a6eb,
                0x3ff7_1814_e9fe_d3f8,
            ],
            vec![
                0x3ff7_2670_f43e_ccce,
                0x3fca_6ae4_fbe3_a6eb,
                0x3ff7_1814_e9fe_d3f8,
                0x3f97_e4b0_e382_105d,
                0x3fac_6604_907a_5fe5,
                0x3ff6_9b1d_b170_775c,
            ],
            vec![
                0x4067_d656_8b20_fbc5,
                0x4068_c5da_0ee6_077c,
                0x4068_30d9_c41b_d4f2,
            ],
        ];
        assert_eq!(actual, expected);
    }
}
