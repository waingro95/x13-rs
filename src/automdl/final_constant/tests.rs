use super::*;
use crate::Frequency;
use crate::error::{NumericalDetail, NumericalFailure, NumericalOperation};
use crate::regarima::OutlierType;
use crate::time::TimeAxis;

fn model() -> ResolvedArimaModel {
    ResolvedArimaModel::new([3, 1, 1], [0, 1, 1]).unwrap()
}

/// The t-value of a model whose only regressor is the constant.
fn lone_constant(coefficient: f64, factor: &[f64], variance: f64) -> Result<Option<f64>, Error> {
    constant_t_value(
        &[RegressorKind::Constant],
        &[coefficient],
        factor,
        variance,
        model(),
    )
}

fn is_unavailable(result: Result<Option<f64>, Error>) -> bool {
    matches!(
        result,
        Err(Error::Numerical(NumericalFailure(NumericalDetail::ModelSelectionConstantTestUnavailable {
            model: affected,
            test: ModelSelectionConstantTest::BeforeFinalCorrections,
        }))) if affected == model()
    )
}

#[test]
fn a_t_value_of_one_keeps_the_constant() {
    for sign in [1.0, -1.0] {
        for (coefficient, dropped) in [
            (2.0_f64.next_down(), true),
            (2.0, false),
            (2.0_f64.next_up(), false),
        ] {
            let coefficient = sign * coefficient;
            let t_value = lone_constant(coefficient, &[1.0, 0.0, 1.0], 4.0)
                .unwrap()
                .unwrap();
            assert_eq!(t_value.to_bits(), (coefficient / 2.0).to_bits());
            assert_eq!(is_insignificant(t_value), dropped);
        }
    }
}

#[test]
fn the_constant_is_found_after_other_regressors() {
    let outlier = RegressorKind::Outlier {
        kind: OutlierType::LevelShift,
        period: TimeAxis::quarterly(2002, 3).unwrap().period_at(0).unwrap(),
        frequency: Frequency::Quarterly,
    };
    // The last row of the factor is the response column, which the
    // standard error does not use.
    let t_value = constant_t_value(
        &[outlier, RegressorKind::Constant],
        &[100.0, 1.0],
        &[1.0, 0.0, 1.0, 0.0, 0.0, 99.0],
        4.0,
        model(),
    )
    .unwrap();
    assert_eq!(t_value, Some(0.5));
}

#[test]
fn a_model_without_a_constant_has_no_t_value() {
    for variance in [0.0, 4.0] {
        assert_eq!(
            constant_t_value(&[], &[], &[1.0], variance, model()).unwrap(),
            None
        );
    }
}

#[test]
fn a_variance_within_the_machine_limit_is_an_error() {
    let limit = 2.0 * BUILD_62_MACHINE_EPSILON;
    for variance in [0.0, limit] {
        assert!(is_unavailable(lone_constant(
            1.0,
            &[1.0, 0.0, 1.0],
            variance
        )));
    }
    assert!(lone_constant(1.0, &[1.0, 0.0, 1.0], limit.next_up()).is_ok());
    // The variance is checked before the factor is inverted, so a zero
    // pivot does not change the error.
    assert!(is_unavailable(lone_constant(1.0, &[0.0, 0.0, 1.0], 0.0)));
}

#[test]
fn the_variance_limit_is_dpmpar_not_f64_epsilon() {
    // `genrtt.f` compares with 2D0*dpmpar(1) = 2*2.220446d-16, just below
    // 2*f64::EPSILON; variances between the two run the t-test.
    let ieee_limit = 2.0 * f64::EPSILON;
    assert!(2.0 * BUILD_62_MACHINE_EPSILON < ieee_limit);
    for variance in [ieee_limit.next_down(), ieee_limit] {
        assert!(lone_constant(1.0, &[1.0, 0.0, 1.0], variance).is_ok());
    }
}

#[test]
fn a_zero_standard_error_gives_a_zero_t_value() {
    assert_eq!(
        lone_constant(1.0, &[1e308, 0.0, 1.0], 1.0).unwrap(),
        Some(0.0)
    );
}

#[test]
fn nonfinite_inputs_and_a_singular_factor_are_rejected() {
    for (coefficient, factor, variance) in [
        (f64::NAN, [1.0, 0.0, 1.0], 4.0),
        (1.0, [f64::INFINITY, 0.0, 1.0], 4.0),
        (1.0, [0.0, 0.0, 1.0], 4.0),
        (1.0, [1.0, 0.0, 1.0], f64::NAN),
    ] {
        assert!(matches!(
            lone_constant(coefficient, &factor, variance),
            Err(error) if error == Error::numerical(NumericalOperation::Estimation)
        ));
    }
}
