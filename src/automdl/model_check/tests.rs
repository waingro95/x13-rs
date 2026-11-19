use super::*;
use crate::error::{NumericalDetail, NumericalFailure, NumericalOperation};

fn airline_model() -> ResolvedArimaModel {
    ResolvedArimaModel::new([0, 1, 1], [0, 1, 1]).unwrap()
}

fn autoregressive_model() -> ResolvedArimaModel {
    ResolvedArimaModel::new([1, 0, 0], [0, 0, 1]).unwrap()
}

fn estimation() -> Error {
    Error::numerical(NumericalOperation::Estimation)
}

/// The residual-mean t statistic of `values` as a whole.
fn residual_mean_t(values: &[f64]) -> f64 {
    model_check(
        airline_model(),
        values,
        values.len(),
        Frequency::Monthly,
        0,
        1.0,
    )
    .unwrap()
    .residual_mean_t
}

#[test]
fn only_the_last_effective_residuals_are_checked() {
    let suffix = [2.0, 4.0, 6.0, 8.0];
    let alone = model_check(airline_model(), &suffix, 4, Frequency::Monthly, 1, 9.0).unwrap();
    let prefixed = model_check(
        airline_model(),
        &[
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            99.0,
            2.0,
            4.0,
            6.0,
            8.0,
        ],
        4,
        Frequency::Monthly,
        1,
        9.0,
    )
    .unwrap();
    assert_eq!(prefixed, alone);
    assert_eq!(prefixed.innovation_scale, 3.0);
    assert_eq!(prefixed.residual_mean_t, 4.47213595499958);
    assert_eq!(
        prefixed.confidence,
        1.0 - residual_acf(&suffix, 2, 1).unwrap()[1].p_value
    );
}

#[test]
fn a_sample_that_does_not_fit_the_residuals_is_an_internal_error() {
    let values = [2.0, 4.0, 6.0, 8.0];
    for count in [0, 1, 5] {
        assert!(matches!(
            model_check(airline_model(), &values, count, Frequency::Monthly, 0, 9.0),
            Err(Error::Internal(_))
        ));
    }
}

#[test]
fn a_degenerate_variance_is_an_estimation_failure() {
    let values = [2.0, 4.0, 6.0, 8.0];
    for variance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            model_check(airline_model(), &values, 4, Frequency::Monthly, 0, variance),
            Err(estimation())
        );
    }
    assert_eq!(
        model_check(airline_model(), &values, 4, Frequency::Monthly, 0, 0.0)
            .unwrap()
            .innovation_scale,
        0.0
    );
}

#[test]
fn nonfinite_residuals_are_an_estimation_failure() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            model_check(
                airline_model(),
                &[1.0, bad, 2.0, 4.0],
                4,
                Frequency::Monthly,
                0,
                1.0
            ),
            Err(estimation())
        );
    }
    // Finite residuals whose sums overflow.
    for values in [[f64::MAX; 4], [1e200, -1e200, 1e200, -1e200]] {
        assert_eq!(
            model_check(airline_model(), &values, 4, Frequency::Monthly, 0, 1.0),
            Err(estimation())
        );
    }
}

#[test]
fn residuals_without_spread_make_the_check_unavailable() {
    let cases: [(&[f64], usize); 5] = [
        (&[0.0, 0.0, 0.0, 0.0], 4),
        (&[5.0, 5.0, 5.0, 5.0], 4),
        (&[1.0, 5.0, 5.0, 5.0], 3),
        (&[1.0, 9.0, -2.0, 5.0, 5.0, 5.0], 3),
        (&[1e-200, -1e-200, 2e-200, -2e-200], 4),
    ];
    for (frequency, model) in [
        (Frequency::Monthly, airline_model()),
        (Frequency::Quarterly, autoregressive_model()),
    ] {
        for (values, effective) in cases {
            assert_eq!(
                model_check(model, values, effective, frequency, 0, 1.0),
                Err(Error::Numerical(NumericalFailure(
                    NumericalDetail::ModelSelectionResidualCheckUnavailable { model }
                ))),
                "{frequency:?} {values:?}"
            );
        }
    }
}

#[test]
fn a_tiny_spread_is_still_checked() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let values = [1e-150, -1e-150, 1e-150, -1e-150];
        let check = model_check(autoregressive_model(), &values, 4, frequency, 0, 0.0).unwrap();
        assert_eq!(check.innovation_scale, 0.0);
        assert_eq!(check.residual_mean_t, 0.0);
    }
}

#[test]
fn residual_mean_t_fuses_the_squares_and_the_mean_product() {
    assert_eq!(
        residual_mean_t(&[0.1, 0.1, 0.2, 0.3]).to_bits(),
        0x4010e2777572bb20
    );
    assert_eq!(
        residual_mean_t(&[1e154, 1e154 + 1e140, 1e154 - 1e140, 1e154]),
        0.0
    );
}

#[test]
fn a_raw_variance_of_zero_or_less_gives_an_infinite_or_nan_t() {
    assert_eq!(
        residual_mean_t(&[67108864.0, 67108865.0, 67108864.0, 67108863.0]),
        f64::INFINITY
    );
    assert_eq!(
        residual_mean_t(&[-67108864.0, -67108865.0, -67108864.0, -67108863.0]),
        f64::NEG_INFINITY
    );
    assert!(residual_mean_t(&[134217728.0, 134217729.0, 134217730.0]).is_nan());
}
