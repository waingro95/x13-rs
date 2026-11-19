//! `automd.f` after the first order search: `pass2.f`'s comparison with the
//! reference model, then the final constant, root, adjustment and
//! coefficient checks, which can send the selector back for another pass.

use super::fallback::{
    FallbackState, FinalChecks, FirstComparison, first_fallback, second::second_fallback,
};
use super::final_adjustments::apply_final_adjustments;
use super::final_coefficients::{CoefficientOutcome, check_final_coefficients};
use super::final_constant::check_final_constant;
use super::final_roots::check_final_roots;
use super::first_pass::FirstPass;
use super::reentry::{
    AfterCoefficientReentry, reenter_coefficient_retry, reenter_first_outlier_retry,
};
use super::restoration::restore_first_pass;
use crate::{Error, Warning};

pub(crate) struct CorrectionRun {
    pub(crate) state: FallbackState,
    /// `restor` recovered the baseline's estimates for the first pass's
    /// model.
    pub(crate) restored_baseline: bool,
}

pub(crate) fn complete_corrections(
    first: &FirstPass,
    original: crate::arima::FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<CorrectionRun, Error> {
    let mut differencing_state = first.differencing_state;
    let restored = restore_first_pass(first, original, warnings)?;
    let restored_baseline = restored.restored_baseline;
    let mut checks = match first_fallback(first, restored, original, warnings)? {
        FirstComparison::FinalChecks(checks) => checks,
        FirstComparison::OutlierRetry(retry) => {
            let reentered = reenter_first_outlier_retry(
                first,
                retry,
                original,
                &mut differencing_state,
                warnings,
            )?;
            second_fallback(first, reentered, original, warnings)?
        }
    };

    // Each coefficient restart moves to a later pass, so the loop runs at
    // most three times.
    loop {
        let FinalChecks { state, stage } = check_final_constant(checks, original, warnings)?;
        let state = check_final_roots(state, original, warnings)?;
        let state = apply_final_adjustments(state, original, warnings)?;
        let retry =
            match check_final_coefficients(FinalChecks { state, stage }, original, warnings)? {
                CoefficientOutcome::Checked(state) => {
                    return Ok(CorrectionRun {
                        state,
                        restored_baseline,
                    });
                }
                CoefficientOutcome::Retry(retry) => retry,
            };
        let reentered =
            reenter_coefficient_retry(first, retry, original, &mut differencing_state, warnings)?;
        checks = match reentered {
            AfterCoefficientReentry::Compare(reentered) => {
                second_fallback(first, reentered, original, warnings)?
            }
            AfterCoefficientReentry::FinalChecks(checks) => checks,
        };
    }
}
