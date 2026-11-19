use super::*;
use crate::ArimaOrder;
use crate::arima::FitContext;
use crate::arima::exact::ExactFilterError;
use crate::error::{ExactLikelihoodFailure, NumericalDetail, NumericalFailure};
use crate::regarima::ParameterKind;

fn model(regular: (u8, u8, u8), seasonal: (u8, u8, u8)) -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(regular.0, regular.1, regular.2).unwrap(),
        ArimaOrder::new(seasonal.0, seasonal.1, seasonal.2).unwrap(),
    )
    .unwrap()
}

fn no_arma_terms() -> ArimaModel {
    model((0, 0, 0), (0, 0, 0))
}

fn ar1() -> ArimaModel {
    model((1, 0, 0), (0, 0, 0))
}

fn alternating(amplitude: f64) -> Vec<f64> {
    (0..40)
        .map(|index| {
            if index % 2 == 0 {
                -amplitude
            } else {
                amplitude
            }
        })
        .collect()
}

fn monthly(values: &[f64]) -> PreparedSearchInput<'_> {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    PreparedSearchInput::new(values, values.into(), axis, Transform::None, &[]).unwrap()
}

/// A search fit that follows a converged one.
fn attempt(input: &PreparedSearchInput<'_>, model: ArimaModel) -> FitAttempt {
    search_attempt(input, model, EstimationPolicy::AUTOMATIC_SEARCH, true)
        .ok()
        .unwrap()
}

fn filter_failure(
    cause: ExactFilterError,
    convergence_after_check: Option<bool>,
) -> AttemptFailure {
    AttemptFailure::Filter {
        cause,
        convergence_after_check,
        warnings: vec![
            Warning::DevianceIncreased,
            Warning::DevianceIncreased,
            Warning::SingularArmaCovariance,
            Warning::EstimationRequiredManyIterations { iterations: 237 },
        ],
    }
}

#[test]
fn a_nonfinite_series_or_one_past_the_last_period_is_rejected() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    assert!(
        PreparedSearchInput::new(&[f64::NAN], (&[1.0]).into(), axis, Transform::None, &[]).is_err()
    );
    let end = TimeAxis::monthly(i32::MAX, 12).unwrap();
    assert!(
        PreparedSearchInput::new(&[1.0, 2.0], (&[1.0, 2.0]).into(), end, Transform::None, &[])
            .is_err()
    );
    assert!(
        PreparedSearchInput::new(&[1.0], (&[1.0]).into(), axis, Transform::Log, &[1.0]).is_ok()
    );
}

#[test]
fn a_fit_needs_as_many_differenced_observations_as_its_longest_ar_lag() {
    let expanded =
        ExpandedModel::from_spec(model((3, 1, 0), (1, 1, 0)), crate::Frequency::Quarterly);
    // Differencing leaves n - 1 - 4 observations; the AR lags reach 3 + 4.
    assert_eq!(
        validate_effective_length(11, &expanded),
        Err(Error::InsufficientEffectiveObservations {
            actual: 6,
            minimum: 7
        })
    );
    assert_eq!(validate_effective_length(12, &expanded), Ok(()));
}

#[test]
fn a_short_or_constant_series_fails_the_search() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let candidate = model((3, 1, 0), (1, 1, 0));
    let short = [1.0; 11];
    let input =
        PreparedSearchInput::new(&short, (&short).into(), axis, Transform::None, &[]).unwrap();
    assert_eq!(
        fit_prepared_candidate(&input, candidate, &mut true),
        Err(Error::InsufficientEffectiveObservations {
            actual: 6,
            minimum: 7
        })
    );
    let constant = [1.0; 20];
    let input = PreparedSearchInput::new(&constant, (&constant).into(), axis, Transform::None, &[])
        .unwrap();
    assert_eq!(
        fit_prepared_candidate(&input, candidate, &mut true),
        Err(Error::DegenerateSeries)
    );
}

#[test]
fn a_filter_failure_becomes_the_matching_public_error() {
    let cases = [
        (
            ExactFilterError::SingularMaInitialCovariance { pivot: 2 },
            Error::Numerical(NumericalFailure(
                NumericalDetail::ExactLikelihoodUnavailable {
                    failure: ExactLikelihoodFailure::SingularMaInitialCovariance { pivot: 2 },
                },
            )),
        ),
        (
            ExactFilterError::InvalidRoots {
                kind: ParameterKind::Ma,
            },
            Error::Numerical(NumericalFailure(NumericalDetail::InvalidRoots {
                kind: ParameterKind::Ma,
            })),
        ),
    ];
    for (cause, expected) in cases {
        assert_eq!(filter_failure(cause, None).into_public(), expected);
    }
    assert!(matches!(
        filter_failure(ExactFilterError::InvalidDimensions, None).into_public(),
        Error::Internal(_)
    ));
}

#[test]
fn a_singular_initial_covariance_leaves_the_candidate_unranked() {
    let singular = ExactFilterError::SingularMaInitialCovariance { pivot: 1 };
    // Only the warnings raised while iterating are kept.
    assert_eq!(
        unranked_for_filter_failure(filter_failure(singular, None), &mut true),
        Ok(CandidateFit {
            bic2: None,
            warnings: vec![Warning::DevianceIncreased, Warning::DevianceIncreased],
        })
    );
    assert!(
        unranked_for_filter_failure(
            filter_failure(ExactFilterError::InvalidDimensions, None),
            &mut true
        )
        .is_err()
    );
}

#[test]
fn a_filter_failure_carries_the_convergence_it_was_checked_with() {
    let singular = ExactFilterError::SingularMaInitialCovariance { pivot: 1 };
    let mut converged = true;
    // A failure before the convergence check leaves the flag as it was.
    for (convergence_after_check, expected) in
        [(Some(false), false), (None, false), (Some(true), true)]
    {
        unranked_for_filter_failure(
            filter_failure(singular, convergence_after_check),
            &mut converged,
        )
        .unwrap();
        assert_eq!(converged, expected);
    }
}

#[test]
fn a_singular_covariance_leaves_a_converged_candidate_unranked() {
    let values = [0.0; 40];
    let input = monthly(&values);
    let mut fit = attempt(&input, ar1());
    fit.fitted.state.warnings = vec![
        Warning::DevianceIncreased,
        Warning::DevianceIncreased,
        Warning::EstimationRequiredManyIterations { iterations: 237 },
    ];
    // The fit has no likelihood either; the covariance is looked at first.
    assert_eq!(
        score_candidate(&input, fit),
        Ok(CandidateFit {
            bic2: None,
            warnings: vec![Warning::DevianceIncreased, Warning::DevianceIncreased],
        })
    );

    let mut converged = true;
    fit_prepared_candidate(&input, ar1(), &mut converged).unwrap();
    assert!(converged);
}

#[test]
fn a_fit_with_a_diagnostic_is_unranked_but_can_be_the_selected_fit() {
    let values = alternating(1.0);
    let input = monthly(&values);
    let with_diagnostic = || {
        let mut fit = attempt(&input, no_arma_terms());
        fit.fitted.status.diagnostic = FitDiagnostic::FunctionToleranceTooSmall;
        fit
    };
    assert_eq!(
        score_candidate(&input, with_diagnostic()).unwrap().bic2,
        None
    );
    assert!(finish_selected(with_diagnostic()).is_ok());
    assert!(
        score_candidate(&input, attempt(&input, no_arma_terms()))
            .unwrap()
            .bic2
            .is_some()
    );
}

#[test]
fn a_converged_fit_without_a_likelihood_fails_the_search() {
    let values = alternating(1e-9);
    let input = monthly(&values);
    let fit = attempt(&input, no_arma_terms());
    assert!(fit.fitted.status.converged);
    assert_eq!(fit.fitted.innovation_variance, 0.0);
    assert!(fit.fitted.likelihood.is_none());
    assert_eq!(
        fit_prepared_candidate(&input, no_arma_terms(), &mut true),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::ModelSelectionUnavailable {
                candidate: no_arma_terms().into(),
            }
        )))
    );
}

#[test]
fn an_infinite_transformation_adjustment_fails_the_search() {
    let prepared = alternating(1.0);
    let original = vec![f64::from_bits(1); prepared.len()];
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let context = FitContext::unadjusted(&original).with_additive_adjustment(true);
    let input = PreparedSearchInput::new(&prepared, context, axis, Transform::Log, &[]).unwrap();
    let fit = attempt(&input, no_arma_terms());
    assert!(matches!(
        fit.fitted.likelihood,
        Some(FitLikelihood::Exact(_))
    ));
    assert_eq!(
        score_candidate(&input, fit),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::ModelSelectionUnavailable {
                candidate: no_arma_terms().into(),
            }
        )))
    );
}

#[test]
fn the_selected_fit_may_have_a_singular_covariance() {
    let values = [0.0; 40];
    let selected = refit_prepared_selected(&monthly(&values), ar1(), &mut true).unwrap();
    assert_eq!(selected.arma_covariance, ArmaCovariance::Singular);
    assert!(selected.warnings.contains(&Warning::SingularArmaCovariance));
}

#[test]
fn a_fit_without_arma_terms_reports_the_previous_fits_convergence() {
    let values = alternating(1.0);
    let input = monthly(&values);
    let after_failure = search_attempt(
        &input,
        no_arma_terms(),
        EstimationPolicy::AUTOMATIC_SEARCH,
        false,
    )
    .ok()
    .unwrap();
    assert!(!after_failure.fitted.status.converged);
    let iterations = after_failure.fitted.nonlinear_iterations;

    let mut converged = false;
    assert_eq!(
        fit_prepared_candidate(&input, no_arma_terms(), &mut converged)
            .unwrap()
            .bic2,
        None
    );
    assert!(!converged);
    // As the selected model, the same fit is automdl's "Estimation failed to
    // converge" stop.
    assert_eq!(
        finish_selected(after_failure),
        Err(Error::EstimationDidNotConverge { iterations })
    );
}
