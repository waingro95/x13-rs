use crate::arima::estimation::*;

#[test]
fn each_policy_has_its_tolerances_and_iteration_limits() {
    let effective = 137.0;
    for (policy, overall, initial, final_tolerance, maximum_iterations) in [
        (EstimationPolicy::EXPLICIT, 1e-5, 1e-3, 1e-5, 1500),
        (
            EstimationPolicy::AUTOMATIC_IDENTIFICATION,
            1e-3,
            0.1,
            1e-3,
            200,
        ),
        (EstimationPolicy::AUTOMATIC_SEARCH, 1e-3, 0.1, 1e-3, 1500),
    ] {
        assert_eq!(
            policy.overall_deviance_tolerance(effective),
            2.0 / effective * overall
        );
        assert_eq!(policy.maximum_overall_iterations(), maximum_iterations);
        // With regressors the first two overall iterations use the looser
        // tolerance, and each nonlinear step may add 40 iterations.
        for iteration in [1, 2] {
            assert_eq!(
                policy.nonlinear_deviance_tolerance(effective, true, iteration),
                2.0 / effective * initial
            );
        }
        for iteration in [3, 17] {
            assert_eq!(
                policy.nonlinear_deviance_tolerance(effective, true, iteration),
                2.0 / effective * final_tolerance
            );
        }
        assert_eq!(policy.nonlinear_iteration_limit(true, 199), 239);
        // Without regressors there is one nonlinear step, with the overall
        // tolerance and limit.
        assert_eq!(
            policy.nonlinear_deviance_tolerance(effective, false, 1),
            policy.overall_deviance_tolerance(effective)
        );
        assert_eq!(
            policy.nonlinear_iteration_limit(false, 199),
            maximum_iterations
        );
    }
}

#[test]
fn the_iteration_limit_stops_a_fit_whose_objective_has_converged() {
    for (nonlinear_iterations, expected) in [
        (199, OuterDecision::Complete),
        (200, OuterDecision::Stop(FitStop::OverallIterations)),
        (201, OuterDecision::Stop(FitStop::OverallIterations)),
    ] {
        let mut state = EstimationState {
            overall_iterations: 2,
            previous_objective: Some(10.0),
            ..EstimationState::default()
        };
        assert_eq!(
            state.check_outer_convergence(10.0, 1e-3, nonlinear_iterations, 200),
            expected
        );
    }
}
