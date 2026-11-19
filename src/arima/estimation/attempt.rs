use super::FittedRegArima;
use crate::Error;
use crate::arima::model::ExpandedModel;
use crate::error::NumericalOperation;
use crate::numerics::minpack::LmExit;

/// A fit as `rgarma.f` left it, with the reason its loop ended.
pub(super) struct FitAttempt {
    pub(super) fitted: Box<FittedRegArima>,
    pub(super) stop: FitStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitStop {
    Completed,
    OverallIterations,
    NonlinearEvaluations,
    ImproperInput,
    ToleranceBelowPrecision,
}

impl FitAttempt {
    pub(super) fn into_completed(self) -> Result<FittedRegArima, Error> {
        match self.stop {
            // The loop can end normally with `rgarma.f`'s `Convrg` false:
            // MINPACK reaches its evaluation limit after accepting a step, and
            // the next GLS pass needs no further optimizer call.
            FitStop::Completed if self.fitted.status.converged => Ok(*self.fitted),
            FitStop::Completed | FitStop::OverallIterations | FitStop::NonlinearEvaluations => {
                Err(Error::EstimationDidNotConverge {
                    iterations: self.fitted.nonlinear_iterations,
                })
            }
            FitStop::ImproperInput | FitStop::ToleranceBelowPrecision => {
                Err(Error::numerical(NumericalOperation::Estimation))
            }
        }
    }
}

/// Checks that the observations left after differencing cover the AR and MA
/// lags: `Nefobs` against `Nextvl` in `rgarma.f`.
pub(super) fn validate_effective_length(
    series_len: usize,
    model: &ExpandedModel,
) -> Result<(), Error> {
    let difference_loss = model.differencing_loss();
    let ar_loss = model.ar_span();
    let ma_loss = model.ma_span();
    let actual = series_len.saturating_sub(difference_loss);
    let minimum = ar_loss + ma_loss;
    // `actual == 0` is stricter than `rgarma.f`: with no ARMA lags and a span
    // equal to its differencing loss, `Nefobs = Nextvl = 0` passes `.lt.` and
    // Build 62 fits an empty sample.
    if series_len < difference_loss || actual < minimum || actual == 0 {
        return Err(Error::InsufficientEffectiveObservations {
            actual,
            minimum: minimum.max(1),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PostOptimizerDecision {
    /// Run another outer GLS pass.
    Refit,
    /// End the fit without a final residual evaluation.
    Stop(FitStop),
    Complete,
}

pub(super) fn post_optimizer_decision(
    exit: LmExit,
    starting_iterations: usize,
    nonlinear_iterations: usize,
    maximum_overall_iterations: usize,
) -> PostOptimizerDecision {
    if nonlinear_iterations > starting_iterations {
        return PostOptimizerDecision::Refit;
    }
    if exit == LmExit::EvaluationOrIterationLimit {
        let stop = if nonlinear_iterations >= maximum_overall_iterations {
            FitStop::OverallIterations
        } else {
            FitStop::NonlinearEvaluations
        };
        return PostOptimizerDecision::Stop(stop);
    }
    if exit == LmExit::ImproperInput {
        return PostOptimizerDecision::Stop(FitStop::ImproperInput);
    }
    // MINPACK infos 6–8 complete the fit. Only the outer `stpitr` precision
    // check uses ToleranceBelowPrecision.
    PostOptimizerDecision::Complete
}

#[cfg(test)]
mod tests;
