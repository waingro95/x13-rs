use super::{ArmaCovarianceDiagnostic, FitDiagnostic, FitStatus};
use crate::{Warning, regarima};

/// What a fit's final warnings are reported from. A covariance diagnostic
/// of `None` means the covariance was not found singular, not that it was
/// calculated: a fit stopped before its covariance step never runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FitReport {
    pub(crate) status: FitStatus,
    pub(crate) covariance_diagnostic: ArmaCovarianceDiagnostic,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) arma_parameter_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FitReportDiagnostic {
    Estimation(FitDiagnostic),
    SingularArmaCovariance,
}

impl FitReport {
    pub(crate) fn diagnostic(self) -> FitReportDiagnostic {
        match self.covariance_diagnostic {
            ArmaCovarianceDiagnostic::Singular => FitReportDiagnostic::SingularArmaCovariance,
            ArmaCovarianceDiagnostic::None => {
                FitReportDiagnostic::Estimation(self.status.diagnostic)
            }
        }
    }
}

/// Appends a fit's estimation warnings to `history`, then the warning for
/// the diagnostic it ended with.
pub(crate) fn append_reported_fit_warnings(
    history: &mut Vec<Warning>,
    raw_warnings: &[Warning],
    report: FitReport,
) {
    append_dynamic_fit_warnings(history, raw_warnings);
    let reported = match report.diagnostic() {
        FitReportDiagnostic::SingularArmaCovariance => Some(Warning::SingularArmaCovariance),
        FitReportDiagnostic::Estimation(FitDiagnostic::ImproperInput) => {
            Some(Warning::ImproperLikelihoodInput)
        }
        FitReportDiagnostic::Estimation(FitDiagnostic::FunctionToleranceTooSmall) => {
            Some(Warning::NoFurtherLikelihoodImprovement {
                criterion: regarima::PrecisionCriterion::Likelihood,
            })
        }
        FitReportDiagnostic::Estimation(FitDiagnostic::ParameterToleranceTooSmall) => {
            Some(Warning::NoFurtherLikelihoodImprovement {
                criterion: regarima::PrecisionCriterion::Parameters,
            })
        }
        FitReportDiagnostic::Estimation(FitDiagnostic::GradientToleranceTooSmall) => {
            Some(Warning::NoFurtherLikelihoodImprovement {
                criterion: regarima::PrecisionCriterion::Gradient,
            })
        }
        FitReportDiagnostic::Estimation(
            FitDiagnostic::None
            | FitDiagnostic::OverallIterationLimit
            | FitDiagnostic::FunctionEvaluationLimit,
        ) => None,
    };
    history.extend(reported);
}

/// Appends a fit's warnings except the singular-covariance and
/// iteration-count ones, which are reported from the fit's final status.
pub(crate) fn append_dynamic_fit_warnings(history: &mut Vec<Warning>, raw_warnings: &[Warning]) {
    history.extend(
        raw_warnings
            .iter()
            .filter(|warning| {
                !matches!(
                    warning,
                    Warning::SingularArmaCovariance
                        | Warning::EstimationRequiredManyIterations { .. }
                )
            })
            .cloned(),
    );
}

#[cfg(test)]
mod tests {
    use super::{FitReport, FitReportDiagnostic, append_reported_fit_warnings};
    use crate::arima::estimation::{ArmaCovarianceDiagnostic, FitDiagnostic, FitStatus};
    use crate::{Warning, regarima};

    #[test]
    fn reported_warnings_follow_the_final_diagnostic() {
        let cases = [
            (FitDiagnostic::None, None),
            (
                FitDiagnostic::ImproperInput,
                Some(Warning::ImproperLikelihoodInput),
            ),
            (FitDiagnostic::OverallIterationLimit, None),
            (FitDiagnostic::FunctionEvaluationLimit, None),
            (
                FitDiagnostic::FunctionToleranceTooSmall,
                Some(Warning::NoFurtherLikelihoodImprovement {
                    criterion: regarima::PrecisionCriterion::Likelihood,
                }),
            ),
            (
                FitDiagnostic::ParameterToleranceTooSmall,
                Some(Warning::NoFurtherLikelihoodImprovement {
                    criterion: regarima::PrecisionCriterion::Parameters,
                }),
            ),
            (
                FitDiagnostic::GradientToleranceTooSmall,
                Some(Warning::NoFurtherLikelihoodImprovement {
                    criterion: regarima::PrecisionCriterion::Gradient,
                }),
            ),
        ];
        let raw = vec![
            Warning::DevianceIncreased,
            Warning::DevianceIncreased,
            Warning::SingularArmaCovariance,
            Warning::EstimationRequiredManyIterations { iterations: 237 },
        ];
        for (diagnostic, expected) in cases {
            for covariance_diagnostic in [
                ArmaCovarianceDiagnostic::None,
                ArmaCovarianceDiagnostic::Singular,
            ] {
                let report = FitReport {
                    status: FitStatus {
                        converged: true,
                        diagnostic,
                    },
                    covariance_diagnostic,
                    nonlinear_iterations: 237,
                    arma_parameter_count: 3,
                };
                let sentinel = Warning::AutomaticTransformNonPositiveInput { index: 7 };
                let mut actual = vec![sentinel.clone()];
                append_reported_fit_warnings(&mut actual, &raw, report);
                let final_warning = if covariance_diagnostic == ArmaCovarianceDiagnostic::Singular {
                    Some(Warning::SingularArmaCovariance)
                } else {
                    expected.clone()
                };
                let mut wanted = vec![
                    sentinel,
                    Warning::DevianceIncreased,
                    Warning::DevianceIncreased,
                ];
                wanted.extend(final_warning);
                assert_eq!(actual, wanted, "{diagnostic:?} {covariance_diagnostic:?}");
            }
        }
    }

    #[test]
    fn singular_covariance_takes_precedence_over_the_estimation_diagnostic() {
        let diagnostics = [
            FitDiagnostic::None,
            FitDiagnostic::ImproperInput,
            FitDiagnostic::OverallIterationLimit,
            FitDiagnostic::FunctionEvaluationLimit,
            FitDiagnostic::FunctionToleranceTooSmall,
            FitDiagnostic::ParameterToleranceTooSmall,
            FitDiagnostic::GradientToleranceTooSmall,
        ];
        for diagnostic in diagnostics {
            for (covariance_diagnostic, expected) in [
                (
                    ArmaCovarianceDiagnostic::None,
                    FitReportDiagnostic::Estimation(diagnostic),
                ),
                (
                    ArmaCovarianceDiagnostic::Singular,
                    FitReportDiagnostic::SingularArmaCovariance,
                ),
            ] {
                let report = FitReport {
                    status: FitStatus {
                        converged: true,
                        diagnostic,
                    },
                    covariance_diagnostic,
                    nonlinear_iterations: 237,
                    arma_parameter_count: 3,
                };
                assert_eq!(report.diagnostic(), expected);
            }
        }
    }
}
