//! Which fits' warnings an outlier search reports.

use super::SearchFit;
use crate::Warning;
use crate::arima::append_reported_fit_warnings;

/// Tracks whose warnings the search is collecting.
pub(super) enum SearchWarningState {
    /// The explicit search, which reports the warnings of every fit it runs.
    Explicit,
    /// An automatic search still on the fit it was given. Automatic
    /// modelling has already reported that fit's warnings.
    AutomaticIncoming,
    /// An automatic search on one of its own refits.
    AutomaticRefit,
}

impl SearchWarningState {
    /// Collects the warnings of `fit`, which a refit is about to replace.
    pub(super) fn before_refit(&mut self, history: &mut Vec<Warning>, fit: &mut impl SearchFit) {
        match self {
            Self::Explicit => {
                // `stpitr.f` emits during estimation. The explicit path
                // retains only the terminal `arima.f` many-iterations note.
                history.extend(fit.warnings_mut().drain(..).filter(|warning| {
                    !matches!(warning, Warning::EstimationRequiredManyIterations { .. })
                }));
            }
            Self::AutomaticIncoming => *self = Self::AutomaticRefit,
            Self::AutomaticRefit => {
                let report = fit.report();
                append_reported_fit_warnings(history, fit.warnings_mut(), report);
            }
        }
    }

    /// Collects the warnings of the search's last fit.
    pub(super) fn finish(self, history: &mut Vec<Warning>, fit: &mut impl SearchFit) {
        match self {
            Self::Explicit => history.append(fit.warnings_mut()),
            Self::AutomaticIncoming => {}
            Self::AutomaticRefit => {
                let report = fit.report();
                append_reported_fit_warnings(history, fit.warnings_mut(), report);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SearchWarningState;
    use crate::arima::{
        ArmaCovarianceDiagnostic, ArmaStart, EstimatedRegArima, FitDiagnostic, estimate,
    };
    use crate::regression::model::RegressionModel;
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, RegArimaSpec, Warning, regarima};

    fn fitted() -> EstimatedRegArima {
        let values = (0..48)
            .map(|index| (index % 11) as f64 - 5.0)
            .collect::<Vec<_>>();
        let order = ArimaOrder::new(0, 0, 0).unwrap();
        let model = ArimaModel::new(order, order).unwrap();
        let spec = RegArimaSpec::new(model)
            .with_constant(false)
            .resolve_explicit()
            .unwrap();
        estimate(
            &values,
            (&values).into(),
            TimeAxis::monthly(2000, 1).unwrap(),
            spec,
            RegressionModel::new(),
            ArmaStart::Default,
        )
        .unwrap()
    }

    #[test]
    fn an_automatic_search_skips_the_incoming_fit_and_reports_its_first_refit() {
        let baseline = fitted();
        let mut initial = baseline.clone();
        initial.warnings = vec![Warning::AutomaticTransformNonPositiveInput { index: 7 }];
        let mut history = Vec::new();
        SearchWarningState::AutomaticIncoming.finish(&mut history, &mut initial);
        assert!(history.is_empty());

        let mut state = SearchWarningState::AutomaticIncoming;
        state.before_refit(&mut history, &mut initial);
        assert!(history.is_empty());
        let mut refit = baseline;
        refit.warnings = vec![
            Warning::DevianceIncreased,
            Warning::SingularArmaCovariance,
            Warning::EstimationRequiredManyIterations { iterations: 237 },
        ];
        refit.pre_covariance_status.diagnostic = FitDiagnostic::FunctionToleranceTooSmall;
        state.finish(&mut history, &mut refit);
        assert_eq!(
            history,
            [
                Warning::DevianceIncreased,
                Warning::NoFurtherLikelihoodImprovement {
                    criterion: regarima::PrecisionCriterion::Likelihood
                },
            ]
        );
    }

    #[test]
    fn an_automatic_search_reports_each_refit_once() {
        let baseline = fitted();
        let mut state = SearchWarningState::AutomaticIncoming;
        let mut history = Vec::new();
        let mut incoming = baseline.clone();
        incoming.warnings = vec![Warning::AutomaticTransformNonPositiveInput { index: 7 }];
        state.before_refit(&mut history, &mut incoming);

        let mut first = baseline.clone();
        first.warnings = vec![Warning::DevianceIncreased, Warning::SingularArmaCovariance];
        first.arma_covariance_diagnostic = ArmaCovarianceDiagnostic::Singular;
        state.before_refit(&mut history, &mut first);

        let mut second = baseline.clone();
        second.warnings = vec![
            Warning::DevianceIncreased,
            Warning::DevianceIncreased,
            Warning::EstimationRequiredManyIterations { iterations: 220 },
        ];
        second.pre_covariance_status.diagnostic = FitDiagnostic::ParameterToleranceTooSmall;
        state.before_refit(&mut history, &mut second);

        let mut final_fit = baseline;
        final_fit.warnings = vec![
            Warning::DevianceIncreased,
            Warning::EstimationRequiredManyIterations { iterations: 221 },
        ];
        final_fit.pre_covariance_status.diagnostic = FitDiagnostic::GradientToleranceTooSmall;
        state.finish(&mut history, &mut final_fit);
        assert_eq!(
            history,
            [
                Warning::DevianceIncreased,
                Warning::SingularArmaCovariance,
                Warning::DevianceIncreased,
                Warning::DevianceIncreased,
                Warning::NoFurtherLikelihoodImprovement {
                    criterion: regarima::PrecisionCriterion::Parameters
                },
                Warning::DevianceIncreased,
                Warning::NoFurtherLikelihoodImprovement {
                    criterion: regarima::PrecisionCriterion::Gradient
                },
            ]
        );
    }

    #[test]
    fn the_explicit_search_keeps_only_the_last_many_iterations_note() {
        let baseline = fitted();
        let mut state = SearchWarningState::Explicit;
        let mut history = Vec::new();
        let mut first = baseline.clone();
        first.warnings = vec![
            Warning::DevianceIncreased,
            Warning::SingularArmaCovariance,
            Warning::EstimationRequiredManyIterations { iterations: 220 },
        ];
        state.before_refit(&mut history, &mut first);
        let mut final_fit = baseline;
        final_fit.warnings = vec![
            Warning::DevianceIncreased,
            Warning::EstimationRequiredManyIterations { iterations: 221 },
        ];
        state.finish(&mut history, &mut final_fit);
        assert_eq!(
            history,
            [
                Warning::DevianceIncreased,
                Warning::SingularArmaCovariance,
                Warning::DevianceIncreased,
                Warning::EstimationRequiredManyIterations { iterations: 221 },
            ]
        );
    }
}
