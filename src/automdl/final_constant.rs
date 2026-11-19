//! `chkmu.f` before the final corrections: a constant whose |t| is below one
//! is dropped and the model refitted.

use super::errors::estimation_failure;
use super::fallback::{FallbackFit, FinalChecks, LoopStage};
use super::refit::Refit;
use crate::error::{ModelSelectionConstantTest, NumericalDetail};
use crate::numerics::BUILD_62_MACHINE_EPSILON;
use crate::numerics::packed::{packed_diagonal, packed_inverse, packed_len};
use crate::regarima::ResolvedArimaModel;
use crate::regression::model::RegressorKind;
use crate::{Error, Warning};

pub(crate) fn check_final_constant(
    checks: FinalChecks,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<FinalChecks, Error> {
    let FinalChecks { mut state, stage } = checks;
    debug_assert!(state.is_consistent());
    let view = state.fit.numerical_view();
    let spec = state.fit.spec();
    let model = spec.model();
    let t_value = constant_t_value(
        view.regressor_kinds,
        view.regression_coefficients,
        view.outlier_test_factor,
        state.fit.innovation_variance(),
        model,
    )?;
    if !t_value.is_some_and(is_insignificant) {
        return Ok(FinalChecks { state, stage });
    }

    let response = view.transformed.to_vec();
    let mut regression = state.fit.regression_model().clone();
    regression.set_constant(false, model, view.axis.frequency(), original.raw().len())?;
    let fit = Refit {
        response: &response,
        original,
        axis: view.axis,
        spec: spec.with_constant(false),
        regression,
        start: view.refit_start(),
        carried_convergence: state.fit.converged(),
    }
    .fit_converged(warnings)?;
    state.fit = FallbackFit::Existing(Box::new(fit));
    state.residual_override = None;
    state.current.procedural_mean = false;
    // automd.f: `IF(nloop.lt.3)nloop=3` once chkmu drops the constant.
    Ok(FinalChecks {
        state,
        stage: LoopStage::Final,
    })
}

/// `chkmu.f` drops a constant whose |t| is below one.
fn is_insignificant(t_value: f64) -> bool {
    t_value.abs() < 1.0
}

/// The constant's t-value from the regression's packed factor, or `None`
/// when there is no constant. `factor` packs the normal equations plus the
/// response column.
fn constant_t_value(
    kinds: &[RegressorKind],
    coefficients: &[f64],
    factor: &[f64],
    variance: f64,
    model: ResolvedArimaModel,
) -> Result<Option<f64>, Error> {
    let count = kinds.len();
    debug_assert_eq!(coefficients.len(), count);
    debug_assert_eq!(factor.len(), packed_len(count + 1));
    if !variance.is_finite()
        || coefficients
            .iter()
            .chain(factor)
            .any(|value| !value.is_finite())
    {
        return Err(estimation_failure());
    }
    let Some(index) = kinds
        .iter()
        .position(|kind| *kind == RegressorKind::Constant)
    else {
        return Ok(None);
    };
    // Typed error, not a copy of Build 62. `genrtt.f` runs the t-test only
    // when `Var.gt.2D0*dpmpar(1)`, where `dpmpar.f` has 2.220446d-16 for
    // `dpmpar(1)`, not `f64::EPSILON`. Otherwise it sets `rmse` to zero and
    // returns without writing `Tval`. `chkmu.f` then reads `tval(icol)` from
    // its own never-initialised local array to decide whether to drop the
    // constant, so Build 62's result in this branch is whatever the stack
    // held.
    if variance <= 2.0 * BUILD_62_MACHINE_EPSILON {
        return Err(Error::from(
            NumericalDetail::ModelSelectionConstantTestUnavailable {
                model,
                test: ModelSelectionConstantTest::BeforeFinalCorrections,
            },
        ));
    }
    let mut inverse = factor[..packed_len(count)].to_vec();
    if (0..count).any(|column| inverse[packed_diagonal(column)] == 0.0) {
        return Err(estimation_failure());
    }
    packed_inverse(&mut inverse, count);
    let diagonal = inverse[packed_diagonal(index)];
    if inverse.iter().any(|value| !value.is_finite()) || diagonal < 0.0 {
        return Err(estimation_failure());
    }
    let standard_error = diagonal.sqrt() * variance.sqrt();
    let t_value = if standard_error > 0.0 {
        coefficients[index] / standard_error
    } else {
        0.0
    };
    if !standard_error.is_finite() || !t_value.is_finite() {
        return Err(estimation_failure());
    }
    Ok(Some(t_value))
}

#[cfg(test)]
mod tests;
