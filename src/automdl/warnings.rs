//! The warnings the selector reports for its fits.

use crate::Warning;
use crate::arima::{
    AutomaticFit, FitReport, append_dynamic_fit_warnings, append_reported_fit_warnings,
};

/// Appends the warnings of a fit whose estimation Build 62 reports.
pub(super) fn push_reported_fit(warnings: &mut Vec<Warning>, fit: &AutomaticFit) {
    append_reported_fit_warnings(warnings, fit.warnings(), fit.report());
}

/// Appends the warnings of a fit run only to test the constant. If it did
/// not converge, the constant is dropped untested, which gets its own
/// warning.
pub(super) fn push_constant_test_fit(warnings: &mut Vec<Warning>, fit: &AutomaticFit) {
    append_dynamic_fit_warnings(warnings, fit.warnings());
    if !fit.pre_covariance_status().converged {
        warnings.push(Warning::ModelSelectionConstantOmitted {
            iterations: fit.nonlinear_iterations(),
        });
    }
}

/// Appends the note for a final model that converged after more than 200
/// iterations.
pub(crate) fn push_many_iterations(warnings: &mut Vec<Warning>, report: FitReport) {
    if report.status.converged
        && report.arma_parameter_count > 0
        && report.nonlinear_iterations > 200
    {
        warnings.push(Warning::EstimationRequiredManyIterations {
            iterations: report.nonlinear_iterations,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::push_many_iterations;
    use crate::Warning;
    use crate::arima::{ArmaCovarianceDiagnostic, FitDiagnostic, FitReport, FitStatus};

    #[test]
    fn many_iterations_note_needs_a_converged_fit_with_arma_parameters() {
        for (converged, parameters, iterations, expected) in [
            (true, 1, 200, false),
            (true, 1, 201, true),
            (true, 1, 500, true),
            (true, 1, 501, true),
            (true, 0, 501, false),
            (false, 1, 501, false),
        ] {
            let mut warnings = vec![Warning::DevianceIncreased];
            push_many_iterations(
                &mut warnings,
                FitReport {
                    status: FitStatus {
                        converged,
                        diagnostic: FitDiagnostic::None,
                    },
                    covariance_diagnostic: ArmaCovarianceDiagnostic::None,
                    nonlinear_iterations: iterations,
                    arma_parameter_count: parameters,
                },
            );
            let mut wanted = vec![Warning::DevianceIncreased];
            if expected {
                wanted.push(Warning::EstimationRequiredManyIterations { iterations });
            }
            assert_eq!(warnings, wanted, "{converged} {parameters} {iterations}");
        }
    }
}
