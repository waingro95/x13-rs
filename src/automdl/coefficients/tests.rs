use super::*;
use crate::arima::roots::RootDetails;

fn model(regular: [u8; 3], seasonal: [u8; 3]) -> ResolvedArimaModel {
    ResolvedArimaModel::new(regular, seasonal).unwrap()
}

/// A check with unit innovation variance.
fn check(
    regular: [u8; 3],
    seasonal: [u8; 3],
    parameters: &[f64],
    diagonal: &[f64],
    observed_count: usize,
) -> CoefficientCheck {
    check_coefficients(
        model(regular, seasonal),
        observed_count,
        parameters,
        Some(ArmaVariances {
            innovation_variance: 1.0,
            diagonal,
        }),
    )
    .unwrap()
}

fn outcome(check: CoefficientCheck) -> (u8, Option<ResolvedArimaModel>) {
    (check.insignificant_count, check.reduced_model)
}

#[test]
fn a_four_way_tie_drops_the_regular_ar_term() {
    let result = check(
        [2, 1, 1],
        [1, 1, 1],
        &[0.25, 0.0625, 0.0625, 0.0625, 0.0625],
        &[1.0; 5],
        144,
    );
    assert_eq!(outcome(result), (4, Some(model([1, 1, 1], [1, 1, 1]))));
}

#[test]
fn a_smaller_later_t_value_clears_the_earlier_flags() {
    assert_eq!(
        outcome(check(
            [2, 1, 1],
            [1, 1, 1],
            &[0.25, 0.0625, 0.03125, 0.0625, 0.125],
            &[1.0; 5],
            144,
        )),
        (3, Some(model([2, 1, 1], [0, 1, 1])))
    );
    assert_eq!(
        outcome(check(
            [2, 1, 1],
            [1, 1, 1],
            &[0.25, 0.125, 0.0625, 0.03125, 0.015625],
            &[1.0; 5],
            144,
        )),
        (1, Some(model([2, 1, 1], [1, 1, 0])))
    );
}

#[test]
fn equal_t_values_count_without_reselecting() {
    // -0.0 and 0.0 have the same magnitude.
    assert_eq!(
        outcome(check(
            [2, 1, 1],
            [1, 1, 1],
            &[0.25, 0.0625, 0.125, -0.0, 0.0],
            &[1.0; 5],
            144,
        )),
        (2, Some(model([2, 1, 0], [1, 1, 1])))
    );
    assert_eq!(
        outcome(check(
            [1, 1, 1],
            [1, 1, 1],
            &[0.03125, 0.0625, 0.03125, 0.25],
            &[1.0; 4],
            144,
        )),
        (3, Some(model([0, 1, 1], [1, 1, 1])))
    );
}

#[test]
fn each_operator_can_be_the_one_dropped() {
    let cases = [
        ([1, 0, 0], [0, 0, 1], [0.0625, 0.25], [0, 0, 0], [0, 0, 1]),
        ([0, 0, 1], [1, 0, 0], [0.0625, 0.25], [0, 0, 1], [0, 0, 0]),
        ([0, 0, 1], [1, 0, 0], [0.25, 0.0625], [0, 0, 0], [1, 0, 0]),
        ([1, 0, 0], [0, 0, 1], [0.25, 0.0625], [1, 0, 0], [0, 0, 0]),
    ];
    for (regular, seasonal, parameters, reduced_regular, reduced_seasonal) in cases {
        assert_eq!(
            outcome(check(regular, seasonal, &parameters, &[1.0; 2], 144)),
            (1, Some(model(reduced_regular, reduced_seasonal)))
        );
    }
}

#[test]
fn only_the_last_coefficient_of_an_operator_is_tested() {
    assert_eq!(
        outcome(check([2, 0, 0], [0, 0, 0], &[0.0625, 0.25], &[1.0; 2], 144)),
        (0, None)
    );
}

#[test]
fn the_coefficient_cutoff_depends_on_the_sample_size() {
    for (observed_count, cutoff) in [(150, 0.15_f64), (151, 0.1)] {
        for sign in [1.0, -1.0] {
            let count = |coefficient: f64| {
                check(
                    [1, 0, 0],
                    [0, 0, 1],
                    &[sign * coefficient, 0.25],
                    &[1.0; 2],
                    observed_count,
                )
                .insignificant_count
            };
            assert_eq!(count(cutoff.next_down()), 1);
            assert_eq!(count(cutoff), 0);
            assert_eq!(count(cutoff.next_up()), 0);
        }
    }
}

#[test]
fn a_t_value_of_one_is_significant() {
    for sign in [1.0, -1.0] {
        let count = |coefficient: f64| {
            check(
                [1, 0, 0],
                [0, 0, 1],
                &[sign * coefficient, 0.25],
                &[0.015625, 1.0],
                144,
            )
            .insignificant_count
        };
        assert_eq!(t_value(sign * 0.125, 1.0, 0.015625).abs(), 1.0);
        assert_eq!(count(0.125_f64.next_down()), 1);
        assert_eq!(count(0.125), 0);
        assert_eq!(count(0.125_f64.next_up()), 0);
    }
}

#[test]
fn t_value_scales_the_variance_before_the_square_root() {
    // 0.0625 / sqrt(0.1 * 0.2).
    assert_eq!(t_value(0.0625, 0.1, 0.2).to_bits(), 0x3fdc48c6001f0abe);
}

#[test]
fn a_near_unit_root_vetoes_every_reduction() {
    for ar in [0.96, -0.96] {
        assert_eq!(
            outcome(check([1, 0, 1], [0, 0, 0], &[ar, 0.0625], &[1.0; 2], 144)),
            (0, None)
        );
    }
    // A complex pair of modulus 1 / 0.96.
    assert_eq!(
        outcome(check(
            [2, 0, 1],
            [0, 0, 0],
            &[0.0, -0.9216, 0.0625],
            &[1.0; 3],
            144,
        )),
        (0, None)
    );
    // Without the near-unit root the MA term is dropped.
    assert_eq!(
        outcome(check([1, 0, 1], [0, 0, 0], &[0.5, 0.0625], &[1.0; 2], 144)),
        (1, Some(model([1, 0, 0], [0, 0, 0])))
    );
}

#[test]
fn the_near_unit_limit_is_inclusive() {
    let limit: f64 = 1.0 / 0.95;
    let near = |modulus: f64| {
        has_near_unit_root(&DetailedRoots {
            roots: vec![RootDetails {
                real: -modulus,
                imaginary: 0.7,
                modulus,
                frequency: 0.0,
            }],
            all_outside_or_on_unit_circle: true,
        })
    };
    assert!(near(limit.next_down()));
    assert!(near(limit));
    assert!(!near(limit.next_up()));
}

#[test]
fn empty_and_single_parameter_models_are_not_reduced() {
    assert_eq!(
        outcome(check([0, 0, 0], [0, 0, 0], &[], &[], 144)),
        (0, None)
    );
    assert_eq!(
        outcome(check([1, 0, 0], [0, 0, 0], &[0.0625], &[1.0], 144)),
        (0, None)
    );
}

#[test]
fn a_singular_covariance_is_an_error_after_the_roots() {
    for (regular, seasonal, parameters) in [
        ([0, 0, 0], [0, 0, 0], &[][..]),
        ([1, 0, 0], [0, 0, 0], &[0.0625][..]),
        ([1, 0, 1], [0, 0, 0], &[0.96, 0.0625][..]),
    ] {
        let result = check_coefficients(model(regular, seasonal), 144, parameters, None);
        assert!(matches!(
            result,
            Err(error) if error == Error::numerical(NumericalOperation::Covariance)
        ));
    }
    // A failure in the roots comes first.
    let result = check_coefficients(model([1, 0, 0], [0, 0, 0]), 144, &[f64::NAN], None);
    assert!(matches!(
        result,
        Err(error) if error == Error::numerical(NumericalOperation::Estimation)
    ));
}

#[test]
fn a_nonfinite_t_value_does_not_stop_the_other_operators() {
    for first_diagonal in [0.0, -1.0] {
        assert!(t_value(0.0, 1.0, first_diagonal).is_nan());
        assert_eq!(
            outcome(check(
                [1, 0, 0],
                [0, 0, 1],
                &[0.0, 0.0625],
                &[first_diagonal, 1.0],
                144,
            )),
            (1, Some(model([1, 0, 0], [0, 0, 0])))
        );
    }
    // A zero innovation variance makes every t-value NaN or infinite.
    let zero_variance = check_coefficients(
        model([1, 0, 0], [0, 0, 1]),
        144,
        &[0.0, 0.0625],
        Some(ArmaVariances {
            innovation_variance: 0.0,
            diagonal: &[1.0; 2],
        }),
    )
    .unwrap();
    assert_eq!(outcome(zero_variance), (0, None));
}

#[test]
fn a_reduction_keeps_the_differencing_orders() {
    assert_eq!(
        check([2, 3, 0], [0, 2, 0], &[0.25, 0.0625], &[1.0; 2], 144).reduced_model,
        Some(model([1, 3, 0], [0, 2, 0]))
    );
}
