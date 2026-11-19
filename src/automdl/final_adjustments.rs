//! `automd.f`'s last two model changes: a regular difference that the MA
//! polynomial cancels is removed, and a constant is added when the residual
//! mean is significant.

use super::constant::constant_is_insignificant;
use super::errors::estimation_failure;
use super::fallback::{FallbackFit, FallbackState};
use super::outlier_policy::active_outlier_policy;
use super::refit::Refit;
use super::warnings::push_constant_test_fit;
use crate::arima::{ArmaStart, AutomaticInput};
use crate::error::ModelSelectionConstantTest;
use crate::outliers::search::search_automatic_finalized_input;
use crate::regarima::ResolvedArimaModel;
use crate::regression::model::RegressorKind;
use crate::{Error, Warning};

pub(crate) fn apply_final_adjustments(
    mut state: FallbackState,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<FallbackState, Error> {
    debug_assert!(state.is_consistent());
    let n = original.raw().len();
    let model = state.current.model;
    let ma_sum = regular_ma_sum(model, state.fit.numerical_view().parameters)?;
    if ma_sum.is_some_and(cancels_a_difference) {
        let regular = model.regular();
        let seasonal = model.seasonal();
        let repaired = ResolvedArimaModel::new(
            [regular.p(), regular.d() - 1, regular.q() - 1],
            [seasonal.p(), seasonal.d(), seasonal.q()],
        )?;
        let view = state.fit.numerical_view();
        let axis = view.axis;
        let response = view.transformed.to_vec();
        let spec = state.fit.spec().with_model(repaired).with_constant(true);
        let mut regression = state.fit.regression_model().clone();
        regression.set_constant(true, repaired, axis.frequency(), n)?;
        state.remove_selected_outliers(&mut regression);
        let mut fit = Refit {
            response: &response,
            original,
            axis,
            spec,
            regression: regression.evaluated_over(axis, n)?,
            start: ArmaStart::Default,
            carried_convergence: state.fit.converged(),
        }
        .fit_converged(warnings)?;
        if let Some(policy) = spec.outliers() {
            let active = active_outlier_policy(policy, state.active_cutoffs)?;
            let searched = search_automatic_finalized_input(
                AutomaticInput::Transformed(&response),
                original,
                fit,
                active,
                std::mem::take(&mut state.selected_outliers),
            )?;
            warnings.extend_from_slice(&searched.fit.warnings);
            state.selected_outliers = searched.selected;
            fit = searched.fit;
        }

        // The constant the repair added is tested in a fit of its own, and
        // dropped if it is insignificant or that fit does not converge.
        let test = ModelSelectionConstantTest::AfterOverdifferenceRepair;
        let constant_regression = fit.regression_model.clone();
        let attempt = Refit {
            response: &response,
            original,
            axis,
            spec,
            regression: constant_regression.clone(),
            start: fit.numerical_view().refit_start(),
            carried_convergence: true,
        }
        .fit()?;
        let drop_constant = match attempt.as_converged() {
            Some(fit) => constant_is_insignificant(fit, test, 1.6)?,
            None => true,
        };
        push_constant_test_fit(warnings, &attempt);
        let start = attempt.numerical_view().refit_start();
        let carried_convergence = attempt.pre_covariance_status().converged;
        let fit = match attempt.into_converged() {
            Ok(fit) if !drop_constant => fit,
            _ => {
                let mut regression = constant_regression;
                regression.set_constant(false, repaired, axis.frequency(), n)?;
                Refit {
                    response: &response,
                    original,
                    axis,
                    spec: spec.with_constant(false),
                    regression: regression.evaluated_over(axis, n)?,
                    start,
                    carried_convergence,
                }
                .fit_converged(warnings)?
            }
        };
        state.fit = FallbackFit::Existing(Box::new(fit));
        state.residual_override = None;
        state.current.model = repaired;
        state.current.automatic_outlier_count = state.selected_outliers.len();
        debug_assert!(state.is_consistent());
        state.recheck(n)?;
    }

    if !state.current.procedural_mean && residual_mean_is_significant(state.residual_mean_t) {
        let spec = state.fit.spec();
        let model = spec.model();
        if spec.includes_constant()
            || state
                .fit
                .regression_model()
                .kinds()
                .any(|kind| kind == RegressorKind::Constant)
        {
            return Err(Error::internal(
                "residual mean is significant in a model that has a constant",
            ));
        }
        let view = state.fit.numerical_view();
        let axis = view.axis;
        let response = view.transformed.to_vec();
        let mut regression = state.fit.regression_model().clone();
        regression.set_constant(true, model, axis.frequency(), n)?;
        let fit = Refit {
            response: &response,
            original,
            axis,
            spec: spec.with_constant(true),
            regression: regression.evaluated_over(axis, n)?,
            start: view.refit_start(),
            carried_convergence: state.fit.converged(),
        }
        .fit_converged(warnings)?;
        state.fit = FallbackFit::Existing(Box::new(fit));
        state.residual_override = None;
        debug_assert!(state.is_consistent());
    }
    Ok(state)
}

/// Whether a residual-mean t statistic calls for a constant. The test is
/// one-sided, and a NaN statistic does not pass it.
fn residual_mean_is_significant(t: f64) -> bool {
    t > 2.5
}

/// Whether a regular MA polynomial whose coefficients sum to `ma_sum` has a
/// root close enough to one to cancel a regular difference.
fn cancels_a_difference(ma_sum: f64) -> bool {
    (ma_sum - 1.0).abs() < 0.001
}

/// The sum of the regular MA coefficients, or `None` for a model without a
/// regular difference or a regular MA term. `parameters` holds the regular
/// AR, seasonal AR, regular MA and seasonal MA coefficients in that order.
fn regular_ma_sum(model: ResolvedArimaModel, parameters: &[f64]) -> Result<Option<f64>, Error> {
    let regular = model.regular();
    let seasonal = model.seasonal();
    let p = usize::from(regular.p());
    let seasonal_p = usize::from(seasonal.p());
    let q = usize::from(regular.q());
    debug_assert_eq!(
        parameters.len(),
        p + seasonal_p + q + usize::from(seasonal.q())
    );
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err(estimation_failure());
    }
    if regular.d() == 0 || q == 0 {
        return Ok(None);
    }
    // A running sum in coefficient order, as `automd.f` accumulates it.
    let mut sum = 0.0;
    for &value in &parameters[p + seasonal_p..p + seasonal_p + q] {
        sum += value;
    }
    if !sum.is_finite() {
        return Err(estimation_failure());
    }
    Ok(Some(sum))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(regular: [u8; 3], seasonal: [u8; 3]) -> ResolvedArimaModel {
        ResolvedArimaModel::new(regular, seasonal).unwrap()
    }

    #[test]
    fn regular_ma_sum_reads_the_coefficients_after_both_ar_blocks() {
        assert_eq!(
            regular_ma_sum(model([1, 1, 2], [1, 1, 1]), &[0.4, 0.6, 0.75, 0.25, -0.8]).unwrap(),
            Some(1.0),
        );
    }

    #[test]
    fn regular_ma_sum_accumulates_in_coefficient_order() {
        let parameters = [0.4, 0.6, 1.0e16, -1.0e16, 1.0, -0.8];
        assert_eq!(
            regular_ma_sum(model([1, 1, 3], [1, 1, 1]), &parameters).unwrap(),
            Some(1.0),
        );
        // The reverse order loses the one.
        assert_eq!((1.0 + -1.0e16) + 1.0e16, 0.0);
    }

    #[test]
    fn regular_ma_sum_needs_a_regular_difference_and_ma_term() {
        assert_eq!(
            regular_ma_sum(model([1, 0, 2], [1, 1, 1]), &[0.4, 0.6, 0.75, 0.25, -0.8]).unwrap(),
            None,
        );
        assert_eq!(
            regular_ma_sum(model([1, 1, 0], [1, 1, 1]), &[0.4, 0.6, -0.8]).unwrap(),
            None,
        );
    }

    #[test]
    fn regular_ma_sum_rejects_nonfinite_values() {
        assert!(
            regular_ma_sum(
                model([1, 1, 2], [1, 1, 1]),
                &[0.4, 0.6, f64::NAN, 0.25, -0.8]
            )
            .is_err()
        );
        assert!(
            regular_ma_sum(
                model([0, 1, 3], [0, 0, 0]),
                &[f64::MAX, f64::MAX, -f64::MAX]
            )
            .is_err()
        );
    }

    #[test]
    fn the_residual_mean_test_is_strict_and_one_sided() {
        assert!(!residual_mean_is_significant(2.5));
        assert!(residual_mean_is_significant(2.5_f64.next_up()));
        assert!(!residual_mean_is_significant(-100.0));
        assert!(residual_mean_is_significant(f64::INFINITY));
        assert!(!residual_mean_is_significant(f64::NEG_INFINITY));
        assert!(!residual_mean_is_significant(f64::NAN));
    }

    #[test]
    fn a_difference_is_cancelled_within_a_thousandth_of_one() {
        assert!(cancels_a_difference(1.0));
        assert!(cancels_a_difference(1.001));
        assert!(!cancels_a_difference(1.001_f64.next_up()));
        assert!(!cancels_a_difference(0.999));
        assert!(cancels_a_difference(0.999_f64.next_up()));
    }
}
