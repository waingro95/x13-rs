use crate::numerics::minpack::LmExit;

/// How the nonlinear estimation of a fit ended, when not normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitDiagnostic {
    None,
    ImproperInput,
    OverallIterationLimit,
    FunctionEvaluationLimit,
    FunctionToleranceTooSmall,
    ParameterToleranceTooSmall,
    GradientToleranceTooSmall,
}

/// `rgarma.f`'s `Convrg` flag and the diagnostic a fit ended with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FitStatus {
    pub(crate) converged: bool,
    pub(crate) diagnostic: FitDiagnostic,
}

impl FitStatus {
    /// The status `rgarma.f` derives from the outer loop's convergence and
    /// the last MINPACK exit, which count differently with regressors.
    pub(super) fn classify(
        outer_converged: bool,
        has_regression: bool,
        last_exit: Option<LmExit>,
        nonlinear_iterations: usize,
        maximum_iterations: usize,
    ) -> Self {
        // `rgarma.f` initializes info=1 even when there are no nonlinear
        // parameters, but `None` still records that MINPACK was not called.
        let exit = last_exit.unwrap_or(LmExit::RelativeReduction);
        let converged = if has_regression {
            outer_converged && exit != LmExit::ImproperInput
        } else {
            (outer_converged
                && matches!(
                    exit,
                    LmExit::RelativeReduction
                        | LmExit::RelativeStep
                        | LmExit::RelativeReductionAndStep
                        | LmExit::Orthogonal
                ))
                || matches!(
                    exit,
                    LmExit::FunctionToleranceTooSmall
                        | LmExit::ParameterToleranceTooSmall
                        | LmExit::GradientToleranceTooSmall
                )
        };
        let diagnostic = match exit {
            LmExit::ImproperInput => FitDiagnostic::ImproperInput,
            LmExit::EvaluationOrIterationLimit => {
                if nonlinear_iterations >= maximum_iterations {
                    FitDiagnostic::OverallIterationLimit
                } else {
                    FitDiagnostic::FunctionEvaluationLimit
                }
            }
            _ if nonlinear_iterations >= maximum_iterations && !converged => {
                FitDiagnostic::OverallIterationLimit
            }
            LmExit::RelativeReduction
            | LmExit::RelativeStep
            | LmExit::RelativeReductionAndStep
            | LmExit::Orthogonal => FitDiagnostic::None,
            LmExit::FunctionToleranceTooSmall => FitDiagnostic::FunctionToleranceTooSmall,
            LmExit::ParameterToleranceTooSmall => FitDiagnostic::ParameterToleranceTooSmall,
            LmExit::GradientToleranceTooSmall => FitDiagnostic::GradientToleranceTooSmall,
        };
        Self {
            converged,
            diagnostic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FitDiagnostic as D, FitStatus};
    use crate::numerics::minpack::LmExit as E;

    const fn expected(converged: bool, diagnostic: D) -> FitStatus {
        FitStatus {
            converged,
            diagnostic,
        }
    }

    #[test]
    fn each_minpack_exit_has_the_convergence_and_diagnostic_of_rgarma() {
        // For each exit: without and with regressors, each with the outer
        // loop converged and not, below the overall iteration limit.
        let cases = [
            (
                E::ImproperInput,
                [
                    expected(false, D::ImproperInput),
                    expected(false, D::ImproperInput),
                    expected(false, D::ImproperInput),
                    expected(false, D::ImproperInput),
                ],
            ),
            (
                E::RelativeReduction,
                [
                    expected(true, D::None),
                    expected(false, D::None),
                    expected(true, D::None),
                    expected(false, D::None),
                ],
            ),
            (
                E::RelativeStep,
                [
                    expected(true, D::None),
                    expected(false, D::None),
                    expected(true, D::None),
                    expected(false, D::None),
                ],
            ),
            (
                E::RelativeReductionAndStep,
                [
                    expected(true, D::None),
                    expected(false, D::None),
                    expected(true, D::None),
                    expected(false, D::None),
                ],
            ),
            (
                E::Orthogonal,
                [
                    expected(true, D::None),
                    expected(false, D::None),
                    expected(true, D::None),
                    expected(false, D::None),
                ],
            ),
            (
                E::EvaluationOrIterationLimit,
                [
                    expected(false, D::FunctionEvaluationLimit),
                    expected(false, D::FunctionEvaluationLimit),
                    expected(true, D::FunctionEvaluationLimit),
                    expected(false, D::FunctionEvaluationLimit),
                ],
            ),
            (
                E::FunctionToleranceTooSmall,
                [
                    expected(true, D::FunctionToleranceTooSmall),
                    expected(true, D::FunctionToleranceTooSmall),
                    expected(true, D::FunctionToleranceTooSmall),
                    expected(false, D::FunctionToleranceTooSmall),
                ],
            ),
            (
                E::ParameterToleranceTooSmall,
                [
                    expected(true, D::ParameterToleranceTooSmall),
                    expected(true, D::ParameterToleranceTooSmall),
                    expected(true, D::ParameterToleranceTooSmall),
                    expected(false, D::ParameterToleranceTooSmall),
                ],
            ),
            (
                E::GradientToleranceTooSmall,
                [
                    expected(true, D::GradientToleranceTooSmall),
                    expected(true, D::GradientToleranceTooSmall),
                    expected(true, D::GradientToleranceTooSmall),
                    expected(false, D::GradientToleranceTooSmall),
                ],
            ),
        ];
        for (exit, expected_statuses) in cases {
            for ((has_regression, outer_converged), wanted) in [
                ((false, true), expected_statuses[0]),
                ((false, false), expected_statuses[1]),
                ((true, true), expected_statuses[2]),
                ((true, false), expected_statuses[3]),
            ] {
                assert_eq!(
                    FitStatus::classify(outer_converged, has_regression, Some(exit), 199, 200),
                    wanted,
                    "{exit:?}, regression={has_regression}, outer={outer_converged}"
                );
            }
        }
    }

    #[test]
    fn an_unconverged_fit_at_the_iteration_limit_reports_the_limit() {
        for iterations in [200, 201] {
            assert_eq!(
                FitStatus::classify(false, true, Some(E::RelativeReduction), iterations, 200),
                expected(false, D::OverallIterationLimit)
            );
            assert_eq!(
                FitStatus::classify(
                    false,
                    true,
                    Some(E::FunctionToleranceTooSmall),
                    iterations,
                    200
                ),
                expected(false, D::OverallIterationLimit)
            );
            assert_eq!(
                FitStatus::classify(
                    false,
                    false,
                    Some(E::FunctionToleranceTooSmall),
                    iterations,
                    200
                ),
                expected(true, D::FunctionToleranceTooSmall)
            );
            assert_eq!(
                FitStatus::classify(
                    true,
                    true,
                    Some(E::EvaluationOrIterationLimit),
                    iterations,
                    200
                ),
                expected(true, D::OverallIterationLimit)
            );
            assert_eq!(
                FitStatus::classify(false, true, Some(E::ImproperInput), iterations, 200),
                expected(false, D::ImproperInput)
            );
        }
    }

    #[test]
    fn a_fit_that_never_calls_the_optimizer_has_no_diagnostic() {
        assert_eq!(
            FitStatus::classify(true, false, None, 0, 200),
            expected(true, D::None)
        );
        assert_eq!(
            FitStatus::classify(true, true, None, 0, 200),
            expected(true, D::None)
        );
        assert_eq!(
            FitStatus::classify(false, true, None, 200, 200),
            expected(false, D::OverallIterationLimit)
        );
    }
}
