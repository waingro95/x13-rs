use super::*;
use crate::arima::model::ExpandedModel;
use crate::{ArimaModel, ArimaOrder, Frequency};

/// Exact AR initial rows, fresh initial conditions and checked roots.
const EXACT: FilterOptions =
    FilterOptions::computed(ArInitialization::Exact, RootValidation::Validate);

fn matrix(values: &[f64], columns: usize) -> RowMajorMatrix {
    RowMajorMatrix::new(values.to_vec(), columns)
}

#[test]
fn a_reused_workspace_forgets_the_previous_matrix_shape() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let mut workspace = ExactFilterWorkspace::default();
    let input = [1.0, 2.0, 4.0, 2.0, 3.0, 1.0];
    let first = exact_filter(matrix(&input, 1), &model, &[0.4], &mut workspace, EXACT).unwrap();
    let wide = [1.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 2.0, 3.0, 6.0, 1.0, 2.0];
    exact_filter(matrix(&wide, 3), &model, &[0.4], &mut workspace, EXACT).unwrap();
    let repeated = exact_filter(matrix(&input, 1), &model, &[0.4], &mut workspace, EXACT).unwrap();
    assert_eq!(first, repeated);
}

#[test]
fn log_determinant_sums_the_diagonal_logs_in_order() {
    assert_eq!(log_determinant(&[], 0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(log_determinant(&[1.0], 1).to_bits(), 0.0_f64.to_bits());
    let factor = [2.0, 7.0, 3.0, 11.0, 13.0, 5.0];
    let expected = ((0.0 + 2.0 * 2.0_f64.ln()) + 2.0 * 3.0_f64.ln()) + 2.0 * 5.0_f64.ln();
    assert_eq!(log_determinant(&factor, 3).to_bits(), expected.to_bits());
    assert_eq!(log_determinant(&factor, 2).to_bits(), 0x400cab0bfa2a2002);
    assert_eq!(log_determinant(&[-0.0], 1), f64::NEG_INFINITY);
    assert!(log_determinant(&[f64::NAN], 1).is_nan());
}

#[test]
fn exact_filter_of_an_arma_2_2_model() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(2, 0, 2).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let input = [
        1.0, -0.5, 2.0, 0.25, 4.0, 1.5, 2.0, -2.0, 3.0, 0.75, 1.0, 1.0, 5.0, -1.25, 2.5, 0.5,
    ];
    let result = exact_filter(
        matrix(&input, 2),
        &model,
        &[0.3, -0.2, 0.4, 0.1],
        &mut ExactFilterWorkspace::default(),
        EXACT,
    )
    .unwrap();
    assert_eq!(
        (result.matrix().row_count(), result.matrix().column_count()),
        (10, 2)
    );
    assert_eq!(
        result.log_determinant_correction().to_bits(),
        0x3fd1971e660534c6
    );
    assert_eq!(
        result
            .matrix()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0x3ff753df9a1d2f19,
            0xbfd99132217c5307,
            0x4013df60c099a07c,
            0x3fd8ed43959af3e1,
            0xbfe1538ff06b9c4a,
            0xbfb3e049caece010,
            0xc005bd8e235d6f64,
            0xbfc523e3abfbbc97,
            0x4003abb058a4bc56,
            0x3ff404cdc15521f8,
            0x3ffe96a3d97119ff,
            0xbffea8208b377e60,
            0x4010d7a600938c05,
            0x3ff023a1287250d9,
            0x400300d9cb554aea,
            0x3fe2c7ad9e1d2768,
            0x401aaf55c2530362,
            0xbff107b38f885670,
            0x40106c9397e5784d,
            0x3fe6a7681cfc3eca,
        ]
    );
}

#[test]
fn lag_expansions_treat_each_column_separately_and_zero_tiny_sums() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 2).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let mut forward = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    positive_lag_expand_in_place(&mut forward, 2, model.ma_operators(), &[0.5, 0.25]);
    assert_eq!(forward, [1.0, 2.0, 3.5, 5.0, 7.0, 9.0]);
    let mut backward = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    negative_lag_expand(&mut backward, 2, model.ma_operators(), &[0.5, 0.25]);
    assert_eq!(backward, [5.0, 7.0, 5.5, 7.0, 5.0, 6.0]);
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let mut cancel = [1.0, 1e-301, 2.0, 0.0];
    negative_lag_expand(&mut cancel, 2, model.ma_operators(), &[-0.5]);
    assert_eq!(cancel, [1.0, 0.0, 2.0, 0.0]);
}

#[test]
fn the_forecast_filter_equals_the_exact_one_without_its_ar_initial_rows() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 1, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let values: Vec<_> = (0..120)
        .flat_map(|i| [i as f64, (i as f64 * 0.7).sin() + i as f64 * 0.1])
        .collect();
    let parameters = [0.4, -0.3];
    let mut workspace = ExactFilterWorkspace::default();
    let full = exact_filter(
        matrix(&values, 2),
        &model,
        &parameters,
        &mut workspace,
        EXACT,
    )
    .unwrap();
    let tail = exact_filter(
        matrix(&values, 2),
        &model,
        &parameters,
        &mut workspace,
        FilterOptions::FORECAST_TAIL,
    )
    .unwrap();
    assert_eq!(tail.matrix().as_slice(), &full.matrix().as_slice()[2..]);
    workspace.ar_initial_factor.clear();
    workspace.initial_cross_filter.clear();
    let repeated = exact_filter(
        matrix(&values, 2),
        &model,
        &parameters,
        &mut workspace,
        FilterOptions::FORECAST_TAIL,
    )
    .unwrap();
    assert_eq!(tail.matrix().as_slice(), repeated.matrix().as_slice());
    let mut extended = values.clone();
    extended.extend_from_slice(&values[..24]);
    assert!(
        exact_filter(
            matrix(&extended, 2),
            &model,
            &parameters,
            &mut workspace,
            FilterOptions::FORECAST_TAIL
        )
        .is_ok()
    );
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The log determinant of the initial covariance (`euclid`, `uconv`,
/// `xpand`) for AR spans shorter than, longer than and equal to the MA span,
/// with differencing and a seasonal model.
#[test]
fn the_log_determinant_for_each_ordering_of_the_ar_and_ma_spans() {
    let cases = [
        ((1, 0, 3), (0, 0, 0), 0x3fe7_8e84_a186_e6d8),
        ((3, 0, 1), (0, 0, 0), 0x3fc0_2364_0ff5_52d9),
        ((2, 0, 0), (0, 0, 0), 0x3fc2_b61b_c63b_2462),
        ((1, 1, 2), (0, 0, 0), 0x3fd0_ee36_1be9_afbe),
        ((1, 0, 1), (1, 0, 1), 0x4011_d8c3_444c_0aea),
    ];
    let parameters = [0.3, -0.2, 0.1, 0.4, 0.2, -0.1];
    let values: Vec<_> = (0..40)
        .flat_map(|i| {
            let i = f64::from(i);
            [
                (i * 0.37).sin() * 3.0 + i * 0.05,
                (i * 0.61).cos() - i * 0.02,
            ]
        })
        .collect();
    for (regular, seasonal, log_determinant) in cases {
        let model = ExpandedModel::from_spec(
            ArimaModel::new(
                ArimaOrder::new(regular.0, regular.1, regular.2).unwrap(),
                ArimaOrder::new(seasonal.0, seasonal.1, seasonal.2).unwrap(),
            )
            .unwrap(),
            Frequency::Monthly,
        );
        let count = model.parameter_slots().len();
        let result = exact_filter(
            matrix(&values, 2),
            &model,
            &parameters[..count],
            &mut ExactFilterWorkspace::default(),
            EXACT,
        )
        .unwrap();
        assert_eq!(
            result.log_determinant_correction().to_bits(),
            log_determinant,
            "{regular:?}{seasonal:?}"
        );
    }
}

#[test]
fn an_explosive_ar_fails_the_euclid_recursion_when_roots_are_not_checked() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let unvalidated = FilterOptions::computed(ArInitialization::Exact, RootValidation::Skip);
    let values = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    assert_eq!(
        exact_filter(
            matrix(&values, 1),
            &model,
            &[1.5],
            &mut ExactFilterWorkspace::default(),
            unvalidated,
        ),
        Err(ExactFilterError::InvalidArCovariance)
    );
    // A unit reflection passes the `abs() > 1` check.
    assert!(
        exact_filter(
            matrix(&values, 1),
            &model,
            &[0.999],
            &mut ExactFilterWorkspace::default(),
            unvalidated,
        )
        .is_ok()
    );
}

#[test]
fn ma_autocovariances_keep_signed_zeros_and_nans() {
    assert_eq!(bits(&ma_autocovariances(&[1.0])), bits(&[1.0]));
    assert_eq!(bits(&ma_autocovariances(&[-0.0])), bits(&[-0.0]));
    assert_eq!(
        bits(&ma_autocovariances(&[1.0, -0.0, -0.0])),
        bits(&[1.0, 0.0, -0.0])
    );
    assert_eq!(
        bits(&ma_autocovariances(&[1.0, -0.4, 0.3, 0.1])),
        [
            0x3ff4_28f5_c28f_5c29,
            0xbfdf_5c28_f5c2_8f5c,
            0x3fd0_a3d7_0a3d_70a4,
            0x3fb9_9999_9999_999a,
        ]
    );
    let with_nan = ma_autocovariances(&[1.0, f64::NAN, 0.5]);
    assert!(with_nan[0].is_nan() && with_nan[1].is_nan());
    assert_eq!(with_nan[2].to_bits(), 0.5_f64.to_bits());
}

#[test]
fn expand_ar_covariance_recurs_in_place() {
    let full_ar = [1.0, -0.5, 0.25];
    let mut covariance = [2.0, 1.0, 0.5, 9.0];
    expand_ar_covariance(&full_ar, &mut covariance);
    assert_eq!(covariance, [2.0, 2.0, 1.0, 9.0]);
    let mut signed = [-0.0, -0.0, 7.0];
    expand_ar_covariance(&[1.0, 0.0], &mut signed);
    assert_eq!(bits(&signed), bits(&[-0.0, 0.0, 7.0]));
    let mut with_nan = [f64::NAN, 1.0, 3.0];
    expand_ar_covariance(&[1.0, -0.5], &mut with_nan);
    assert!(with_nan[0].is_nan() && with_nan[1].is_nan());
    assert_eq!(with_nan[2].to_bits(), 3.0_f64.to_bits());
    let mut untouched = [4.0];
    expand_ar_covariance(&[1.0], &mut untouched);
    assert_eq!(untouched, [4.0]);
}

/// Reuse reads the MA factor always, the AR factor only with exact AR
/// rows, and the cross filter only when both spans are positive.
#[test]
fn reused_factors_must_fit_the_call() {
    let model = |p, q| {
        ExpandedModel::from_spec(
            ArimaModel::new(
                ArimaOrder::new(p, 0, q).unwrap(),
                ArimaOrder::new(0, 0, 0).unwrap(),
            )
            .unwrap(),
            Frequency::Monthly,
        )
    };
    let (arma11, ar1, ma1) = (model(1, 1), model(1, 0), model(0, 1));
    let values: Vec<_> = (0..24).map(|i| (f64::from(i) * 0.9).sin()).collect();
    let longer: Vec<_> = (0..28).map(|i| (f64::from(i) * 0.9).sin()).collect();
    let filter = |values: &[f64],
                  model: &ExpandedModel,
                  parameters: &[f64],
                  workspace: &mut ExactFilterWorkspace,
                  options| {
        exact_filter(matrix(values, 2), model, parameters, workspace, options)
            .map(|result| result.matrix().row_count())
    };
    let invalid = Err(ExactFilterError::InvalidDimensions);
    let (tail, outlier) = (FilterOptions::FORECAST_TAIL, FilterOptions::OUTLIER_TEST);

    let mut empty = ExactFilterWorkspace::default();
    assert_eq!(
        filter(&values, &arma11, &[0.4, 0.3], &mut empty, tail),
        invalid
    );

    let mut saved = ExactFilterWorkspace::default();
    filter(&values, &arma11, &[0.4, 0.3], &mut saved, EXACT).unwrap();
    assert_eq!(
        filter(&values, &arma11, &[0.4, 0.3], &mut saved, tail),
        Ok(12)
    );
    assert_eq!(
        filter(&values, &arma11, &[0.4, 0.3], &mut saved, outlier),
        Ok(13)
    );
    assert_eq!(
        filter(&longer, &arma11, &[0.4, 0.3], &mut saved, tail),
        Ok(14)
    );
    assert_eq!(
        filter(&longer, &arma11, &[0.4, 0.3], &mut saved, outlier),
        invalid
    );

    let mut ma_only = ExactFilterWorkspace::default();
    filter(&values, &ma1, &[0.3], &mut ma_only, EXACT).unwrap();
    assert_eq!(
        filter(&values, &arma11, &[0.4, 0.3], &mut ma_only, tail),
        Ok(12)
    );
    assert_eq!(
        filter(&values, &arma11, &[0.4, 0.3], &mut ma_only, outlier),
        invalid
    );

    let mut ar_only = ExactFilterWorkspace::default();
    filter(&values, &ar1, &[0.4], &mut ar_only, EXACT).unwrap();
    assert_eq!(filter(&longer, &ar1, &[0.4], &mut ar_only, outlier), Ok(14));
}

#[test]
fn a_checked_filter_rejects_a_noninvertible_ma() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let values = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    let invalid = [1.1];
    let expected = ExactFilterError::InvalidRoots {
        kind: crate::regarima::ParameterKind::Ma,
    };

    assert_eq!(
        exact_filter(
            matrix(&values, 1),
            &model,
            &invalid,
            &mut ExactFilterWorkspace::default(),
            EXACT,
        ),
        Err(expected)
    );
}

#[test]
fn the_forecast_filter_takes_a_noninvertible_ma() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(0, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let values = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0];
    let invalid = [1.1];
    let mut workspace = ExactFilterWorkspace::default();
    let estimation = FilterOptions::computed(ArInitialization::Exact, RootValidation::Skip);

    let fitted = exact_filter(
        matrix(&values, 1),
        &model,
        &invalid,
        &mut workspace,
        estimation,
    )
    .unwrap();
    let forecast = exact_filter(
        matrix(&values, 1),
        &model,
        &invalid,
        &mut workspace,
        FilterOptions::FORECAST_TAIL,
    )
    .unwrap();

    assert_eq!(forecast.matrix().as_slice(), fitted.matrix().as_slice());
}

const CONDITIONAL: FilterOptions =
    FilterOptions::computed(ArInitialization::Conditional, RootValidation::Validate);

#[test]
fn a_conditional_ar_1_filter_drops_the_first_row() {
    let model = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let result = exact_filter(
        RowMajorMatrix::new(vec![2.0, 8.0, 6.0, 10.0, 14.0, 18.0], 2),
        &model,
        &[0.5],
        &mut ExactFilterWorkspace::default(),
        CONDITIONAL,
    )
    .unwrap();
    assert_eq!(result.matrix().as_slice(), [5.0, 6.0, 11.0, 13.0]);
    assert_eq!(result.matrix().row_count(), 2);
    assert_eq!(result.matrix().column_count(), 2);
    assert_eq!(result.effective_observations(), 2);
    assert_eq!(result.log_determinant_correction(), 0.0);
}

#[test]
fn a_conditional_filter_needs_more_rows_than_its_longest_lag() {
    let ar1 = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 0, 0).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let arma11 = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 0, 1).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    let arma12 = ExpandedModel::from_spec(
        ArimaModel::new(
            ArimaOrder::new(1, 0, 2).unwrap(),
            ArimaOrder::new(0, 0, 0).unwrap(),
        )
        .unwrap(),
        Frequency::Monthly,
    );
    // A one-column series; `RowMajorMatrix` itself rejects a zero width or
    // a partial row.
    let filter = |values: &[f64], model: &ExpandedModel, parameters: &[f64]| {
        exact_filter(
            RowMajorMatrix::new(values.to_vec(), 1),
            model,
            parameters,
            &mut ExactFilterWorkspace::default(),
            CONDITIONAL,
        )
    };
    assert_eq!(
        filter(&[1.0, 2.0], &ar1, &[]),
        Err(ExactFilterError::InvalidParameterCount {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        filter(&[1.0], &ar1, &[0.5]),
        Err(ExactFilterError::InsufficientObservations {
            actual: 1,
            minimum: 2
        })
    );
    assert_eq!(
        filter(&[], &ar1, &[0.5]),
        Err(ExactFilterError::InsufficientObservations {
            actual: 0,
            minimum: 2
        })
    );
    assert_eq!(
        filter(&[1.0, 2.0], &arma12, &[0.5, 0.2, 0.1]),
        Err(ExactFilterError::InsufficientObservations {
            actual: 2,
            minimum: 3
        })
    );
    let equality = filter(&[1.0, 2.0], &arma11, &[0.5, 0.2]).unwrap();
    assert_eq!(equality.effective_observations(), 1);
    assert_eq!(equality.matrix().row_count(), 2);
}
