use crate::error::{NumericalDetail, NumericalFailure};
use crate::numerics::RowMajorMatrix;
use crate::regarima::{ArimaFactor, RegressionVariable};
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, Frequency, RegArimaSpec, Transform};

use super::*;

#[test]
fn a_series_whose_differences_are_all_zero_is_degenerate() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let model = ArimaModel::new(
        ArimaOrder::new(0, 1, 1).unwrap(),
        ArimaOrder::new(0, 1, 1).unwrap(),
    )
    .unwrap();
    let values = (0..13)
        .map(|index| 100.0 + index as f64)
        .collect::<Vec<_>>();
    assert_eq!(
        estimate_default(&values, axis, RegArimaSpec::new(model)).err(),
        Some(Error::DegenerateSeries)
    );
}

#[test]
fn exact_filter_failures_keep_their_condition() {
    assert_eq!(
        map_filter_error(ExactFilterError::InsufficientObservations {
            actual: 1,
            minimum: 2
        }),
        Error::InsufficientEffectiveObservations {
            actual: 1,
            minimum: 2
        }
    );
    assert_eq!(
        map_filter_error(ExactFilterError::SingularMaInitialCovariance { pivot: 5 }),
        Error::Numerical(NumericalFailure(
            NumericalDetail::ExactLikelihoodUnavailable {
                failure: ExactLikelihoodFailure::SingularMaInitialCovariance { pivot: 5 }
            }
        ))
    );
    assert_eq!(
        map_filter_error(ExactFilterError::SingularArInitialCovariance { pivot: 2 }),
        Error::Numerical(NumericalFailure(
            NumericalDetail::ExactLikelihoodUnavailable {
                failure: ExactLikelihoodFailure::SingularArInitialCovariance { pivot: 2 }
            }
        ))
    );
    assert_eq!(
        map_filter_error(ExactFilterError::InvalidArCovariance),
        Error::Numerical(NumericalFailure(
            NumericalDetail::ExactLikelihoodUnavailable {
                failure: ExactLikelihoodFailure::ArmaAutocovariance
            }
        ))
    );
}

#[test]
fn a_variance_at_the_machine_limit_has_a_likelihood_but_no_standard_errors() {
    let boundary = 2.0 * 2.220_446e-16_f64;
    assert_eq!(reported_innovation_variance(boundary.next_down(), 1), 0.0);
    assert_eq!(reported_innovation_variance(boundary, 1), boundary);
    assert!(!uncertainty_is_available(0.0));
    assert!(!uncertainty_is_available(boundary.next_down()));
    assert!(!uncertainty_is_available(boundary));
    assert!(uncertainty_is_available(boundary.next_up()));
    assert!(
        likelihood_statistics(8, 3, 0.0, boundary, 0.0)
            .log_likelihood
            .is_finite()
    );
}

#[test]
fn a_converged_small_objective_completes_the_fit() {
    let mut state = EstimationState {
        overall_iterations: 2,
        previous_objective: Some(1.152_815_670_019_706_2e-16),
        ..EstimationState::default()
    };
    let objective = 1.149_569_735_695_782_2e-16;
    assert_eq!(
        state.check_outer_convergence(objective, 2.5e-6, 7, 1500),
        OuterDecision::Complete
    );
    assert_eq!(state.previous_objective, Some(objective));
    assert!(state.warnings.is_empty());
}

#[test]
fn outer_convergence_at_the_objective_build_62_treats_as_zero() {
    let boundary = 3.834e-20_f64;
    for (objective, should_continue) in [
        (boundary.next_down(), true),
        (boundary, false),
        (boundary.next_up(), false),
    ] {
        let mut state = EstimationState {
            overall_iterations: 2,
            previous_objective: Some(1e-16),
            ..EstimationState::default()
        };
        assert_eq!(
            state.check_outer_convergence(objective, 2.5e-6, 7, 1500),
            if should_continue {
                OuterDecision::Continue
            } else {
                OuterDecision::Complete
            }
        );
        assert_eq!(state.previous_objective, Some(objective));
    }
}

#[test]
fn the_filter_takes_a_noninvertible_ma_when_roots_are_not_checked() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let matrix = RegressionMatrix::without_regressors(&[1.0, 2.0, 1.5, 3.0, 2.5, 4.0]);
    let parameters = [1.1];

    assert_eq!(
        filter_matrix(
            &matrix,
            &model,
            &parameters,
            &mut ExactFilterWorkspace::default(),
            LikelihoodMode::Exact,
        ),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::InvalidRoots {
                kind: crate::regarima::ParameterKind::Ma,
            }
        )))
    );

    let filtered = filter_matrix_without_root_validation(
        &matrix,
        &model,
        &parameters,
        &mut ExactFilterWorkspace::default(),
        LikelihoodMode::Exact,
    )
    .unwrap();
    assert_eq!(filtered.matrix().row_count(), 7);
    assert_eq!(filtered.matrix().column_count(), 1);
    assert!(
        filtered
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(matches!(
        filter_matrix_without_root_validation(
            &matrix,
            &model,
            &[],
            &mut ExactFilterWorkspace::default(),
            LikelihoodMode::Exact,
        ),
        Err(Error::Internal(_))
    ));
}

#[test]
fn a_fit_with_zero_innovation_variance_still_forecasts() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    for (count, constant) in [(13, false), (81, false), (13, true)] {
        let values = (0..count)
            .map(|i| (100.0 + ((17 * i * i + 37 * i + 11) % 71) as f64) * 1e-10)
            .collect::<Vec<_>>();
        let spec = RegArimaSpec::new(
            ArimaModel::new(
                ArimaOrder::new(0, 1, 1).unwrap(),
                ArimaOrder::new(0, 1, 1).unwrap(),
            )
            .unwrap(),
        )
        .with_constant(constant);
        let fit = estimate_default(&values, axis, spec)
            .unwrap_or_else(|error| panic!("count={count}, constant={constant}: {error:?}"));
        assert_eq!(fit.innovation_variance, 0.0);
        assert!(fit.likelihood.is_none());
        assert!(fit.parameter_standard_errors.is_none());
        assert!(fit.regression_standard_errors.is_none());
        assert_eq!(fit.arma_covariance.matrix().unwrap(), [0.0; 4]);
        assert_eq!(fit.parameters.len(), 2);
        assert_eq!(fit.regression_coefficients.len(), usize::from(constant));
        let extensions = crate::arima::forecast::generate_extensions(&fit).unwrap();
        assert_eq!(extensions.forecasts.len(), 4);
        for point in extensions.forecasts {
            assert_eq!(point.transformed_standard_error, 0.0);
            assert_eq!(point.lower_95, point.upper_95);
        }
    }
}

#[test]
fn a_singular_arma_covariance_keeps_the_fit_and_its_filter_workspace() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let mut values = vec![100.0; 13];
    values[6] = 150.0;
    let mut ao = vec![0.0; 13];
    ao[6] = 1.0;
    let mut regression = RegressionModel::new();
    regression
        .insert_outlier(
            crate::regarima::OutlierType::Additive,
            axis.period_at(6).unwrap(),
            Frequency::Quarterly,
            ao,
            0.7,
        )
        .unwrap();
    let fit = estimate(
        &values,
        (&values).into(),
        axis,
        RegArimaSpec::new(
            ArimaModel::new(
                ArimaOrder::new(0, 1, 1).unwrap(),
                ArimaOrder::new(0, 1, 1).unwrap(),
            )
            .unwrap(),
        )
        .resolve_explicit()
        .unwrap(),
        regression,
        ArmaStart::Default,
    )
    .unwrap();
    assert_eq!(fit.parameters, [0.1, 0.1]);
    assert_eq!(fit.regression_coefficients, [50.0]);
    assert_eq!(fit.innovation_variance, 0.0);
    assert_eq!(fit.arma_covariance, ArmaCovariance::Singular);
    assert_eq!(fit.warnings, [Warning::SingularArmaCovariance]);
    assert!(fit.likelihood.is_none());
    assert!(fit.residuals.iter().all(|value| *value == 0.0));
    assert_print_value(
        fit.log_determinant_correction,
        0.050_266_696_521_949_6,
        "singular AO determinant",
    );
    let extensions = crate::arima::forecast::generate_extensions(&fit).unwrap();
    assert_eq!(extensions.forecasts.len(), 4);
    for point in extensions.forecasts {
        assert!((point.original - 100.0).abs() <= 6e-15 * 100.0);
        assert_eq!(point.transformed_standard_error, 0.0);
    }

    // The fit keeps the initial conditions of its last perturbed
    // evaluation, which later filters reuse.
    let model = ExpandedModel::from_spec(fit.spec.model(), Frequency::Quarterly);
    let probe = (0..13)
        .map(|index| (index * index) as f64)
        .collect::<Vec<_>>();
    let mut saved = fit.forecast_filter_workspace.clone();
    let saved_result = exact_filter(
        RowMajorMatrix::new(probe.clone(), 1),
        &model,
        &fit.parameters,
        &mut saved,
        FilterOptions::OUTLIER_TEST,
    )
    .unwrap();
    let mut recomputed = ExactFilterWorkspace::default();
    let recomputed_result = exact_filter(
        RowMajorMatrix::new(probe, 1),
        &model,
        &fit.parameters,
        &mut recomputed,
        FilterOptions::computed(ArInitialization::Exact, RootValidation::Validate),
    )
    .unwrap();
    assert_ne!(saved_result.matrix(), recomputed_result.matrix());
}

#[test]
fn a_singular_regression_names_the_regressor_at_its_pivot() {
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let duplicate = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut regression = RegressionModel::new();
    regression
        .insert_outlier(
            crate::regarima::OutlierType::TemporaryChange,
            axis.period_at(6).unwrap(),
            Frequency::Quarterly,
            duplicate.clone(),
            0.7,
        )
        .unwrap();
    regression
        .insert_outlier(
            crate::regarima::OutlierType::Additive,
            axis.period_at(2).unwrap(),
            Frequency::Quarterly,
            duplicate,
            0.7,
        )
        .unwrap();
    let response = [1.0, 4.0, 2.0, 8.0, 3.0, 7.0, 5.0, 6.0];
    let spec = RegArimaSpec::new(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
    );

    let error = estimate(
        &response,
        (&response).into(),
        axis,
        spec.resolve_explicit().unwrap(),
        regression,
        ArmaStart::Default,
    )
    .unwrap_err();

    assert_eq!(
        error,
        Error::SingularRegression {
            pivot: 1,
            variable: RegressionVariable::Outlier {
                kind: crate::regarima::OutlierType::TemporaryChange,
                period: axis.period_at(6).unwrap(),
            },
        }
    );
}

#[test]
fn a_previous_ma_start_near_the_unit_circle_is_shrunk() {
    let expanded = ExpandedModel::from_spec(airline_model(), Frequency::Monthly);
    let previous = vec![0.964_811_434_783_407_7, 0.999_965_652_571_960_8];

    let prepared = starting_parameters(ArmaStart::Previous(previous.clone()), &expanded).unwrap();

    assert_eq!(prepared[0], previous[0] * 0.9);
    assert_eq!(prepared[1], previous[1] * ma_shrink(12));
}

#[test]
fn an_invalid_previous_ma_start_names_the_first_failing_factor() {
    let model = ArimaModel::new(
        ArimaOrder::new(1, 0, 1).unwrap(),
        ArimaOrder::new(1, 0, 1).unwrap(),
    )
    .unwrap();
    let expanded = ExpandedModel::from_spec(model, Frequency::Quarterly);
    for (previous, factor) in [
        (vec![0.2, 0.3, 1.1, 1.2], ArimaFactor::Regular),
        (vec![0.2, 0.3, 0.1, 1.2], ArimaFactor::Seasonal),
    ] {
        assert_eq!(
            starting_parameters(ArmaStart::Previous(previous), &expanded),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::InvalidMaStartingRoots { factor }
            )))
        );
    }
}

#[test]
fn a_previous_ma_start_with_an_inside_root_is_rejected_not_shrunk() {
    let expanded = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );

    assert_eq!(
        starting_parameters(ArmaStart::Previous(vec![1.1]), &expanded),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::InvalidMaStartingRoots {
                factor: ArimaFactor::Regular,
            }
        )))
    );
}

#[test]
fn a_seasonal_ma2_start_is_checked_and_shrunk_at_its_seasonal_lags() {
    let expanded = expanded_ma_model(0, 2);

    assert_eq!(
        starting_parameters(ArmaStart::Previous(vec![0.2, 1.0]), &expanded),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::InvalidMaStartingRoots {
                factor: ArimaFactor::Seasonal
            }
        )))
    );
    assert_eq!(
        starting_parameters(ArmaStart::Previous(vec![0.0, 1.0]), &expanded).unwrap(),
        [0.0, ma_shrink(24)]
    );
}

#[test]
fn a_previous_ma3_start_on_the_unit_circle_is_shrunk_lag_by_lag() {
    let expanded = expanded_ma_model(3, 0);
    assert_eq!(
        starting_parameters(ArmaStart::Previous(vec![0.2, -0.2, 1.0]), &expanded).unwrap(),
        [
            0.18000000000000002,
            -0.16200000000000003,
            0.7290000000000001
        ]
    );
}

#[test]
fn a_previous_ma1_start_of_one_is_shrunk_to_nine_tenths() {
    let expanded = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );

    assert_eq!(
        starting_parameters(ArmaStart::Previous(vec![1.0]), &expanded).unwrap(),
        [0.9]
    );
}

#[test]
fn a_model_without_arma_terms_has_no_arma_covariance() {
    let values = (0..80)
        .map(|index| 100.0 + index as f64 * 0.25 + (index % 7) as f64)
        .collect::<Vec<_>>();
    let no_arima_parameters = ArimaModel::new(
        ArimaOrder::new(0, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();

    let estimated = estimate_default(
        &values,
        TimeAxis::monthly(2000, 1).unwrap(),
        RegArimaSpec::new(no_arima_parameters).with_constant(true),
    )
    .unwrap();

    assert!(estimated.parameters.is_empty());
    assert!(estimated.parameter_standard_errors.is_none());
    assert_eq!(estimated.arma_covariance, ArmaCovariance::NotRequired);
    assert_eq!(estimated.regression_coefficients.len(), 1);
}

#[test]
fn a_last_optimizer_call_without_iterations_keeps_its_residuals() {
    let mut state = 28_u64;
    let values = (0..144)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let value = if state % 100 < 8 {
                ((state >> 16) % 37) as f64 * 0.001
            } else {
                0.0
            };
            build_62_ctod(&format!("{value:.17}"))
        })
        .collect::<Vec<_>>();
    let model = ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();

    let estimated = estimate_default(
        &values,
        TimeAxis::monthly(1973, 1).unwrap(),
        RegArimaSpec::new(model).with_constant(true),
    )
    .unwrap();

    assert_eq!(estimated.overall_iterations, 2);
    assert_eq!(estimated.nonlinear_iterations, 2);
    assert_print_value(
        estimated.parameters[0],
        2.168_795_202_192_31e-4,
        "AR parameter",
    );
    assert_print_value(
        estimated.innovation_variance,
        1.805_859_291_623_02e-5,
        "innovation variance",
    );
    assert_print_value(
        estimated.likelihood.unwrap().log_likelihood,
        582.048_853_767_46,
        "log likelihood",
    );
    assert_print_values(
        &estimated.residuals[..3],
        &[
            -9.374_971_535_485_48e-4,
            -9.372_938_516_592_97e-4,
            -9.372_938_516_592_97e-4,
        ],
        "callback residuals",
    );
}

#[test]
fn the_covariance_step_may_evaluate_a_noninvertible_ma() {
    let mut state = 149_u64;
    let mut level = 7.0;
    let values = (0..108)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let innovation = ((state >> 32) as f64 / u32::MAX as f64 - 0.5) * 0.02;
            level += 0.002 + innovation;
            build_62_ctod(&format!("{:.17}", level.exp()))
        })
        .collect::<Vec<_>>();

    let estimated = estimate_default(
        &values,
        TimeAxis::monthly(2015, 4).unwrap(),
        RegArimaSpec::new(airline_model())
            .with_transform(Transform::Log)
            .with_constant(true),
    )
    .unwrap();

    assert_print_values(
        &estimated.parameters,
        &[0.014_521_302_761_996_4, 0.999_999_999_625_875],
        "boundary ARMA parameters",
    );
    assert_arma_standard_error_print_values(
        estimated.parameter_standard_errors.as_deref().unwrap(),
        &[0.090_432_274_627_861, 0.111_213_033_464_84],
        "boundary ARMA standard errors",
    );
    assert_print_values(
        estimated.arma_covariance.matrix().unwrap(),
        &[
            0.008_177_996_294_368_91,
            0.001_836_772_442_338_88,
            0.001_836_772_442_338_88,
            0.012_368_338_812_452_6,
        ],
        "boundary ARMA covariance",
    );
}

#[test]
fn many_iteration_warning_starts_after_the_build_62_threshold() {
    let mut warnings = Vec::new();
    append_iteration_warning(200, &mut warnings);
    assert!(warnings.is_empty());

    append_iteration_warning(201, &mut warnings);
    assert_eq!(
        warnings,
        [Warning::EstimationRequiredManyIterations { iterations: 201 }]
    );
}

fn estimate_default(
    series: &[f64],
    axis: TimeAxis,
    spec: RegArimaSpec,
) -> Result<EstimatedRegArima, Error> {
    let spec = spec.resolve_explicit()?;
    let regression = RegressionModel::from_spec(spec, axis, series.len())?;
    estimate(
        series,
        series.into(),
        axis,
        spec,
        regression,
        ArmaStart::Default,
    )
}

/// The factor `prepare_previous_parameters` applies at `lag`. Its exponent
/// is a runtime value, so this one must be too: release builds fold `powi`
/// of two literals to a differently rounded constant.
fn ma_shrink(lag: i32) -> f64 {
    0.9_f64.powi(std::hint::black_box(lag))
}

fn expanded_ma_model(regular_order: u8, seasonal_order: u8) -> ExpandedModel {
    ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, regular_order).unwrap(),
            ArimaOrder::new(0, 0, seasonal_order).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    )
}

fn airline_model() -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(0, 1, 1).unwrap(),
        ArimaOrder::new(0, 1, 1).unwrap(),
    )
    .unwrap()
}

/// Parses a decimal the way Build 62 reads its input, digit by digit, so
/// the two generated series above have the values Build 62 would read.
fn build_62_ctod(token: &str) -> f64 {
    let bytes = token.as_bytes();
    let mut position = 0;
    let sign = match bytes.first() {
        Some(b'-') => {
            position += 1;
            -1.0
        }
        Some(b'+') => {
            position += 1;
            1.0
        }
        _ => 1.0,
    };
    let mut value = 0.0;
    while position < bytes.len() && bytes[position].is_ascii_digit() {
        value = 10.0 * value + f64::from(bytes[position] - b'0');
        position += 1;
    }
    if bytes.get(position) == Some(&b'.') {
        position += 1;
        let mut scale = 1.0;
        while position < bytes.len() && bytes[position].is_ascii_digit() {
            scale *= 10.0;
            value += f64::from(bytes[position] - b'0') / scale;
            position += 1;
        }
    }
    sign * value
}

fn assert_print_values(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_print_value(*actual, *expected, &format!("{context}[{index}]"));
    }
}

fn assert_arma_standard_error_print_values(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_print_value_with_tolerance(
            *actual,
            *expected,
            6.0e-14,
            &format!("{context}[{index}]"),
        );
    }
}

fn assert_print_value(actual: f64, expected: f64, context: &str) {
    assert_print_value_with_tolerance(actual, expected, 6.0e-15, context);
}

fn assert_print_value_with_tolerance(
    actual: f64,
    expected: f64,
    relative_tolerance: f64,
    context: &str,
) {
    if expected == 0.0 {
        assert!(
            actual == 0.0,
            "{context}: expected exact zero, got {actual:.17e}"
        );
        return;
    }
    let tolerance = relative_tolerance * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.3e}"
    );
}

fn filter_matrix(
    matrix: &RegressionMatrix,
    model: &ExpandedModel,
    parameters: &[f64],
    workspace: &mut ExactFilterWorkspace,
    mode: LikelihoodMode,
) -> Result<ExactFilterResult, Error> {
    filter_for(
        matrix,
        model,
        parameters,
        workspace,
        mode,
        RootValidation::Validate,
    )
    .map_err(map_filter_error)
}

fn filter_matrix_without_root_validation(
    matrix: &RegressionMatrix,
    model: &ExpandedModel,
    parameters: &[f64],
    workspace: &mut ExactFilterWorkspace,
    mode: LikelihoodMode,
) -> Result<ExactFilterResult, Error> {
    filter_for(
        matrix,
        model,
        parameters,
        workspace,
        mode,
        RootValidation::Skip,
    )
    .map_err(map_filter_error)
}

#[test]
fn conditional_callbacks_reject_invalid_ma_but_accept_unstable_ar() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        crate::Frequency::Monthly,
    );
    let matrix = RegressionMatrix::without_regressors(&[1.0, 2.0, 1.5, 3.0, 2.5, 4.0]);
    let mut workspace = ExactFilterWorkspace::default();
    let filtered = filter_for(
        &matrix,
        &model,
        &[1.125, 0.1],
        &mut workspace,
        LikelihoodMode::ConditionalAr,
        RootValidation::Validate,
    )
    .unwrap();
    assert_eq!(filtered.effective_observations(), 5);
    let mut output = vec![0.0; filtered.matrix().row_count()];
    let residuals = ScaledResiduals {
        matrix: &matrix,
        model: &model,
        mode: LikelihoodMode::ConditionalAr,
        effective_observations: 5.0,
        invalid_residual: 99.0,
        initial_log_determinant_correction: 0.0,
    };
    let accepted = residuals.evaluate(
        &[1.125, 0.1],
        RootValidation::Validate,
        &mut output,
        &mut workspace,
    );
    assert!(accepted.is_some());
    assert!(output.iter().all(|value| value.is_finite()));
    let rejected = residuals.evaluate(
        &[1.125, 1.1],
        RootValidation::Validate,
        &mut output,
        &mut workspace,
    );
    assert!(rejected.is_none());
    assert!(output.iter().all(|value| *value == 99.0));
    let unchecked = residuals.evaluate(
        &[1.125, 1.1],
        RootValidation::Skip,
        &mut output,
        &mut workspace,
    );
    assert!(unchecked.is_some());
}

#[test]
fn zero_variance_has_no_likelihood_and_raw_differenced_zero_still_fails() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let series = vec![100.0; 48];
    let model = ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = RegArimaSpec::new(model)
        .with_constant(true)
        .resolve_explicit()
        .unwrap();
    let regression = RegressionModel::from_spec(spec, axis, series.len()).unwrap();
    let fitted = fit_conditional(&series, axis, spec, regression).unwrap();
    assert_eq!(fitted.innovation_variance, 0.0);
    assert!(fitted.likelihood.is_none());
    assert!(finish_estimate(fitted).unwrap().likelihood.is_none());

    let differenced_model = ArimaModel::new(
        ArimaOrder::new(1, 1, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = RegArimaSpec::new(differenced_model)
        .resolve_explicit()
        .unwrap();
    let regression = RegressionModel::from_spec(spec, axis, series.len()).unwrap();
    assert!(matches!(
        fit_conditional(&series, axis, spec, regression),
        Err(Error::DegenerateSeries)
    ));
}

fn fit_conditional(
    series: &[f64],
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
) -> Result<FittedRegArima, Error> {
    let request = FitRequest {
        original: series.into(),
        axis,
        spec,
        regression_model,
        start: ArmaStart::Default,
    };
    let control = FitControl::fresh(EstimationPolicy::CONDITIONAL_IDENTIFICATION);
    fit_attempt(series, request, control)?.into_completed()
}
