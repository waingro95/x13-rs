use crate::arima::estimation::attempt::{PostOptimizerDecision, post_optimizer_decision};
use crate::arima::estimation::status::{FitDiagnostic, FitStatus};
use crate::arima::estimation::*;
use crate::numerics::minpack::LmExit;
use crate::{ArimaModel, ArimaOrder, RegArimaSpec};

#[test]
fn a_minpack_precision_exit_completes_the_fit() {
    for exit in [
        LmExit::FunctionToleranceTooSmall,
        LmExit::ParameterToleranceTooSmall,
        LmExit::GradientToleranceTooSmall,
    ] {
        assert_eq!(
            post_optimizer_decision(exit, 7, 7, 1500),
            PostOptimizerDecision::Complete
        );
    }
}

#[test]
fn an_evaluation_limit_without_a_step_stops_the_fit() {
    let mut parameters = [0.0];
    let mut residuals = [0.0];
    let mut counters = LmCounters {
        iterations: 1,
        function_evaluations: 0,
    };
    let outcome = lmdif(
        &mut parameters,
        &mut residuals,
        LmOptions {
            function_tolerance: 0.0,
            parameter_tolerance: 0.0,
            gradient_tolerance: 0.0,
            maximum_iterations: 1,
            finite_difference_epsilon: 0.0,
            factor: 100.0,
        },
        &mut counters,
        |candidate, output, evaluation| {
            output[0] = if evaluation == LmEvaluation::ConstrainedTrial {
                2.0
            } else {
                1.0 + candidate[0]
            };
            Ok(())
        },
    )
    .unwrap();
    let exit = outcome.exit;
    // `fdjac2.f` divides `(1 + h) - 1` by the unrounded h, which with
    // Build 62's machine epsilon is not exactly -1.
    assert_eq!(
        outcome.work_diagonal.unwrap()[0].to_bits(),
        0xbfeffffffdf4392f
    );
    assert_eq!(exit, LmExit::EvaluationOrIterationLimit);
    assert_eq!(parameters, [0.0]);
    assert_eq!(
        counters,
        LmCounters {
            iterations: 1,
            function_evaluations: 3
        }
    );
    assert_eq!(
        post_optimizer_decision(exit, 1, counters.iterations, 200),
        PostOptimizerDecision::Stop(FitStop::NonlinearEvaluations)
    );
    assert_eq!(
        post_optimizer_decision(exit, 1, counters.iterations, 1),
        PostOptimizerDecision::Stop(FitStop::OverallIterations)
    );
    assert_eq!(
        FitStatus::classify(true, false, Some(exit), counters.iterations, 200),
        FitStatus {
            converged: false,
            diagnostic: FitDiagnostic::FunctionEvaluationLimit
        }
    );
    assert_eq!(
        FitStatus::classify(true, true, Some(exit), counters.iterations, 200),
        FitStatus {
            converged: true,
            diagnostic: FitDiagnostic::FunctionEvaluationLimit
        }
    );
}

#[test]
fn a_fit_without_arma_parameters_completes_as_converged() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let model = ArimaModel::new(
        ArimaOrder::new(0, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let varying = (0..48)
        .map(|index| 100.0 + (index % 7) as f64)
        .collect::<Vec<_>>();
    let constant = vec![100.0; 48];
    for (series, has_constant, expected_zero_variance) in
        [(&varying, false, false), (&constant, true, true)]
    {
        let spec = RegArimaSpec::new(model)
            .with_constant(has_constant)
            .resolve_explicit()
            .unwrap();
        let regression = RegressionModel::from_spec(spec, axis, series.len()).unwrap();
        let request = FitRequest {
            original: series.into(),
            axis,
            spec,
            regression_model: regression,
            start: ArmaStart::Default,
        };
        let attempt = fit_attempt(
            series,
            request,
            FitControl::fresh(EstimationPolicy::EXPLICIT),
        )
        .unwrap();
        assert_eq!(attempt.stop, FitStop::Completed);
        let fitted = attempt.fitted;
        assert_eq!(
            fitted.status,
            FitStatus {
                converged: true,
                diagnostic: FitDiagnostic::None
            }
        );
        assert_eq!(fitted.nonlinear_iterations, 0);
        assert!(fitted.parameters.is_empty());
        assert_eq!(fitted.innovation_variance == 0.0, expected_zero_variance);
    }
}
