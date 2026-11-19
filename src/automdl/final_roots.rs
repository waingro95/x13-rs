//! `automd.f`'s final AR-root check: an AR factor with a root near the unit
//! circle trades one order for a difference, and the model is refitted.

use super::constant::constant_is_insignificant;
use super::errors::estimation_failure;
use super::fallback::{FallbackFit, FallbackState};
use super::outlier_policy::active_outlier_policy;
use super::refit::Refit;
use super::roots::{ByFactor, check_ar_roots};
use super::warnings::push_constant_test_fit;
use crate::arima::{ArmaStart, AutomaticInput};
use crate::error::ModelSelectionConstantTest;
use crate::outliers::search::search_automatic_finalized_input;
use crate::regarima::ResolvedArimaModel;
use crate::{Error, Warning};

/// An AR root counts as a unit root within this modulus.
const UNIT_LIMIT: f64 = 1.05;

pub(crate) fn check_final_roots(
    mut state: FallbackState,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<FallbackState, Error> {
    debug_assert!(state.is_consistent());
    let view = state.fit.numerical_view();
    let spec = state.fit.spec();
    let model = spec.model();
    let axis = view.axis;
    let n = original.raw().len();
    let p = usize::from(model.regular().p());
    let seasonal_p = usize::from(model.seasonal().p());
    let roots = check_ar_roots(
        &view.parameters[..p],
        &view.parameters[p..p + seasonal_p],
        UNIT_LIMIT,
    )
    .map_err(|_| estimation_failure())?;
    let Some(final_model) = repaired_model(model, roots.unit_counts)? else {
        return Ok(state);
    };

    let response = view.transformed.to_vec();
    let mut regression = state.fit.regression_model().clone();
    let has_constant = spec.includes_constant();
    let mut current_spec = spec.with_model(final_model);
    regression.set_constant(has_constant, final_model, axis.frequency(), n)?;
    regression = regression.evaluated_over(axis, n)?;
    let mut carried_convergence = state.fit.converged();
    let mut start = ArmaStart::Default;
    if has_constant {
        // The constant is tested in the new model and dropped if it is
        // insignificant or the test fit does not converge.
        let test = ModelSelectionConstantTest::AfterFinalRootRepair;
        let attempt = Refit {
            response: &response,
            original,
            axis,
            spec: current_spec,
            regression: regression.clone(),
            start: ArmaStart::Default,
            carried_convergence,
        }
        .fit()?;
        let drop_constant = match attempt.as_converged() {
            Some(fit) => constant_is_insignificant(fit, test, 1.6)?,
            None => true,
        };
        push_constant_test_fit(warnings, &attempt);
        start = attempt.numerical_view().refit_start();
        carried_convergence = attempt.pre_covariance_status().converged;
        if drop_constant {
            regression.set_constant(false, final_model, axis.frequency(), n)?;
            current_spec = current_spec.with_constant(false);
        }
    }

    state.remove_selected_outliers(&mut regression);
    regression = regression.evaluated_over(axis, n)?;
    let mut fit = Refit {
        response: &response,
        original,
        axis,
        spec: current_spec,
        regression,
        start,
        carried_convergence,
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
    state.fit = FallbackFit::Existing(Box::new(fit));
    state.residual_override = None;
    state.current.model = final_model;
    state.current.automatic_outlier_count = state.selected_outliers.len();
    debug_assert!(state.is_consistent());
    state.recheck(n)?;
    Ok(state)
}

/// `model` with one AR order of each factor that has a unit root turned
/// into a difference, unless the factor already has its maximum differencing
/// order. `None` when nothing changes.
fn repaired_model(
    model: ResolvedArimaModel,
    unit_counts: ByFactor<u8>,
) -> Result<Option<ResolvedArimaModel>, Error> {
    let mut repaired = model;
    if unit_counts.regular > 0 && repaired.regular().d() <= 2 {
        let regular = repaired.regular();
        let seasonal = repaired.seasonal();
        repaired = ResolvedArimaModel::new(
            [regular.p() - 1, regular.d() + 1, regular.q()],
            [seasonal.p(), seasonal.d(), seasonal.q()],
        )?;
    }
    if unit_counts.seasonal > 0 && repaired.seasonal().d() <= 1 {
        let regular = repaired.regular();
        let seasonal = repaired.seasonal();
        repaired = ResolvedArimaModel::new(
            [regular.p(), regular.d(), regular.q()],
            [seasonal.p() - 1, seasonal.d() + 1, seasonal.q()],
        )?;
    }
    Ok((repaired != model).then_some(repaired))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arima::roots::{DetailedRoots, RootDetails};
    use crate::automdl::roots::factor_decision;

    #[test]
    fn a_unit_root_trades_one_ar_order_for_a_difference() {
        let model = ResolvedArimaModel::new([2, 2, 1], [1, 1, 1]).unwrap();
        assert_eq!(
            repaired_model(model, ByFactor::new(2, 0)).unwrap(),
            Some(ResolvedArimaModel::new([1, 3, 1], [1, 1, 1]).unwrap())
        );
        assert_eq!(
            repaired_model(model, ByFactor::new(2, 1)).unwrap(),
            Some(ResolvedArimaModel::new([1, 3, 1], [0, 2, 1]).unwrap())
        );
    }

    #[test]
    fn a_factor_at_its_maximum_difference_is_not_repaired() {
        let model = ResolvedArimaModel::new([1, 3, 1], [1, 2, 1]).unwrap();
        assert_eq!(repaired_model(model, ByFactor::new(1, 1)).unwrap(), None);
        assert_eq!(repaired_model(model, ByFactor::new(0, 0)).unwrap(), None);
    }

    #[test]
    fn unit_roots_need_a_positive_real_part_within_the_limit() {
        let detail = |real, imaginary, modulus| RootDetails {
            real,
            imaginary,
            modulus,
            frequency: 0.0,
        };
        let at = UNIT_LIMIT;
        let below = f64::from_bits(at.to_bits() - 1);
        let above = f64::from_bits(at.to_bits() + 1);
        let roots = DetailedRoots {
            roots: vec![
                detail(at, 0.05, at),
                detail(below, -0.5, below),
                detail(above, 0.0, above),
                detail(-1.0, 0.0, 1.0),
            ],
            all_outside_or_on_unit_circle: true,
        };
        assert_eq!(factor_decision(&roots, at).unit_count, 2);
        assert_eq!(factor_decision(&roots, at).maximum, 1.0 / above);
        let threshold = 1.0 / at;
        assert_eq!(
            check_ar_roots(&[], &[threshold], at).unwrap().unit_counts,
            ByFactor::new(0, 1)
        );
        assert_eq!(
            check_ar_roots(&[-1.0], &[], at).unwrap().unit_counts,
            ByFactor::new(0, 0)
        );
        // Two unit roots still cost the factor only one order.
        assert_eq!(
            check_ar_roots(&[2.0, -1.0], &[], at).unwrap().unit_counts,
            ByFactor::new(2, 0)
        );
    }
}
