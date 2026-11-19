use super::*;

#[test]
fn an_ar_one_estimate_is_not_limited_to_stationary_values() {
    let fit = estimate_prepared(
        &[1.0, 2.0, 4.0, 8.0],
        Frequency::Quarterly,
        HrOrders::new(1, 0, 0, 0),
    )
    .unwrap();
    assert_eq!(fit.parameters, [2.0]);
    assert_eq!(fit.effective_observations, 3);
}

#[test]
fn a_pure_ar_model_does_not_need_autocorrelations() {
    // A constant series has no centred autocorrelations.
    let fit = estimate_prepared(
        &[2.0, 2.0, 2.0, 2.0],
        Frequency::Quarterly,
        HrOrders::new(1, 0, 0, 0),
    )
    .unwrap();
    assert!((fit.parameters[0] - 1.0).abs() <= f64::EPSILON);
}

#[test]
fn negating_a_zero_coefficient_gives_negative_zero() {
    let fit = estimate_prepared(
        &[0.0, 1.0, 0.0, 1.0],
        Frequency::Quarterly,
        HrOrders::new(1, 0, 0, 0),
    )
    .unwrap();
    assert_eq!(format!("{:.4}", fit.parameters[0]), "-0.0000");
}

#[test]
fn a_model_without_arma_terms_keeps_every_observation() {
    let fit = estimate_prepared(
        &[0.0, 0.0, 0.0],
        Frequency::Monthly,
        HrOrders::new(0, 0, 0, 0),
    )
    .unwrap();
    assert!(fit.parameters.is_empty());
    assert_eq!(fit.effective_observations, 3);
}

#[test]
fn the_long_ar_fit_has_at_most_fifty_lags_and_fewer_on_a_short_series() {
    assert_eq!(initial_lags(2_000, 0, 0), 50);
    assert_eq!(initial_lags(8, 0, 3), 6);
    assert_eq!(initial_lags(144, 12, 13), 26);
}

#[test]
fn innovations_are_the_residuals_of_the_long_ar_fit() {
    let acf = Autocorrelations {
        variance: 2.0,
        correlations: vec![0.25, -0.125],
    };
    let actual = innovations(&[2.0, 4.0, 8.0, 16.0], &acf).unwrap();
    assert_eq!(actual, [2.0, 3.4, 7.2, 14.4]);
}

#[test]
fn each_seasonal_lag_adds_its_own_column_and_one_per_regular_lag() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let ahat = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
    let rows = augmented_matrix(&values, 4, HrOrders::new(1, 1, 1, 1), Some(&ahat));
    assert_eq!(
        rows,
        [
            -5.0, -2.0, -1.0, 15.0, 12.0, 11.0, 6.0, -6.0, -3.0, -2.0, 16.0, 13.0, 12.0, 7.0
        ]
    );
}

#[test]
fn a_perfect_fit_is_solved_but_duplicate_regressors_are_singular() {
    assert_eq!(
        solve_augmented(&[1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 3, 2).unwrap(),
        [2.0]
    );
    assert_eq!(
        solve_augmented(&[1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 3.0, 3.0, 6.0], 3, 3),
        Err(HrFailureReason::SingularRegression { pivot: 1 })
    );
}

#[test]
fn each_first_stage_failure_has_its_own_reason() {
    let ar = HrOrders::new(1, 0, 0, 0);
    let fit = |values: &[f64]| estimate_prepared(values, Frequency::Quarterly, ar);
    assert_eq!(
        fit(&[]),
        Err(HrFailure::Initial(HrFailureReason::InvalidInput))
    );
    assert_eq!(
        fit(&[1.0]),
        Err(HrFailure::Initial(
            HrFailureReason::InsufficientObservations
        ))
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            fit(&[1.0, bad]),
            Err(HrFailure::Initial(HrFailureReason::InvalidInput))
        );
    }
    assert_eq!(
        fit(&[1e308, 1e308]),
        Err(HrFailure::Initial(HrFailureReason::NonFinite))
    );
    assert_eq!(
        fit(&[0.0, 0.0, 0.0]),
        Err(HrFailure::Initial(HrFailureReason::SingularRegression {
            pivot: 0
        }))
    );
    assert_eq!(
        estimate_prepared(
            &[2.0, 2.0, 2.0, 2.0],
            Frequency::Quarterly,
            HrOrders::new(0, 0, 1, 0)
        ),
        Err(HrFailure::Initial(HrFailureReason::Autocorrelation(
            AutocorrelationError::ZeroVariance
        )))
    );
}

#[test]
fn a_failed_ma_refit_returns_the_first_stage_estimate() {
    let first_stage = HrEstimate {
        parameters: vec![f64::MAX, 0.25],
        effective_observations: 5,
    };
    let failure = refit_ma(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        4,
        HrOrders::new(1, 0, 1, 0),
        1,
        1,
        first_stage.clone(),
    );
    assert_eq!(
        failure,
        Err(HrFailure::MaRefit {
            reason: HrFailureReason::NonFinite,
            first_stage
        })
    );
}

#[test]
fn innovations_keep_the_fused_operations_and_signed_zeros_of_hrest() {
    let acf = Autocorrelations {
        variance: 1.7,
        correlations: vec![0.6, 0.31, -0.17, 0.05],
    };
    let values = [1.5, -0.25, 3.75, 2.0, -1.125, 0.5, 4.25, -2.0];
    let bits = |values: &[f64], acf: &Autocorrelations| {
        innovations(values, acf)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(&values, &acf),
        [
            0x3ff8_0000_0000_0000,
            0xbffc_6eb4_53d8_8d52,
            0x400f_74d1_f130_4cfc,
            0xbfd3_a34f_25a9_6dc7,
            0xc013_3691_c221_d360,
            0x4015_fd17_cfd5_c28e,
            0x4006_1f34_3011_d48f,
            0xc022_2dfb_e612_dbac,
        ]
    );
    let zeros = Autocorrelations {
        variance: 1.0,
        correlations: vec![0.0, -0.0, 0.0],
    };
    assert_eq!(
        bits(&[-0.0, 0.0, -0.0, -0.0], &zeros),
        [0x8000_0000_0000_0000, 0, 0x8000_0000_0000_0000, 0]
    );
}

#[test]
fn a_nonfinite_step_of_the_innovations_is_a_failure() {
    let overflowing_variance = Autocorrelations {
        variance: 1e300,
        correlations: vec![1e10, 0.5],
    };
    assert_eq!(
        innovations(&[1.0, 2.0, 3.0], &overflowing_variance),
        Err(HrFailureReason::NonFinite)
    );
    let overflowing_residual = Autocorrelations {
        variance: 1.0,
        correlations: vec![-2.0],
    };
    assert_eq!(
        innovations(&[1e308, 1e308], &overflowing_residual),
        Err(HrFailureReason::NonFinite)
    );
}
