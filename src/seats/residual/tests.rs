use super::*;
use crate::seats::ar1_seasonal_ma::residual::ar1_residuals;
use crate::seats::ar2::residual::ar2_prefilter;
use crate::seats::decomposition::root_allocation::RegularArCoefficients::{Three, Two};
use crate::seats::seasonal_ma::residual::{regular_ma_residuals, seasonal_constrained_residuals};

const OVERFLOW: u64 = 0x7ff0_0000_0000_0000;

/// Exact rational values in [-1, 1].
fn series(n: usize, step: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * step + 11) % 101) as f64 / 50.0 - 1.0)
        .collect()
}

/// The residual count, the objective and Detpri of a solve, the last two as
/// bits.
fn summary<E>(result: Result<ResidualResult, E>) -> Result<(usize, u64, u64), E> {
    result.map(|result| {
        (
            result.scaled_residuals.len(),
            result.objective.to_bits(),
            result.determinant_scale.to_bits(),
        )
    })
}

fn objective_overflow(index: usize) -> Error {
    Failure::Unusable {
        quantity: "objective",
        index: Some(index),
        value: f64::INFINITY,
    }
    .at(Stage::Residual)
}

#[test]
fn objective_keeps_a_coefficient_dpeq_finds_unchanged() {
    let wd = [1.0; 75];
    let mut objective = AirlineObjective::new(Frequency::Quarterly, &wd);
    objective.evaluate([0.0, 0.6]).unwrap();
    let first = objective.evaluate([-0.0, 0.6]).unwrap();
    assert_eq!(objective.ma[0].to_bits(), 0.0_f64.to_bits());
    objective.evaluate([3e-20, 0.6]).unwrap();
    assert_eq!(
        objective.ma[0].to_bits(),
        0.0_f64.to_bits(),
        "dpeq finds 3e-20 equal to zero"
    );
    assert_eq!(objective.evaluations(), 3);
    let mut fresh = AirlineObjective::new(Frequency::Quarterly, &wd);
    fresh.evaluate([-0.0, 0.6]).unwrap();
    assert_eq!(
        fresh.ma[0].to_bits(),
        (-0.0_f64).to_bits(),
        "a new objective starts without a kept coefficient"
    );
    objective.evaluate([0.4, 0.6]).unwrap();
    let again = objective.evaluate([0.0, 0.6]).unwrap();
    assert_eq!(
        first
            .scaled_residuals
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        again
            .scaled_residuals
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let other = AirlineObjective::new(Frequency::Monthly, &[1.0; 167])
        .evaluate([0.4, 0.6])
        .unwrap();
    assert_eq!(other.scaled_residuals.len(), 180);
    assert_eq!(objective.evaluate([0.0, 0.6]).unwrap(), again);
}

#[test]
fn airline_residuals_accept_the_span_limits_and_reject_an_overflow() {
    for (frequency, period) in [(Frequency::Quarterly, 4), (Frequency::Monthly, 12)] {
        let model = ResolvedAirline {
            frequency,
            ma: [0.4, 0.6],
        };
        for n in [2 * period + 2, 780] {
            let wd = vec![0.0; n - period - 1];
            let result = resolved_residuals(&model, &wd).unwrap();
            assert_eq!(result.scaled_residuals.len(), n);
            assert_eq!(result.normalized_residuals.len(), n);
        }
        assert!(matches!(
            resolved_residuals(&model, &vec![f64::MAX; period + 1]),
            Err(Error {
                stage: Stage::Residual,
                failure: Failure::Unusable { value, .. },
            }) if !value.is_finite()
        ));
    }
}

#[test]
fn calcfx_solves_each_ma_layout() {
    let quarterly = series(60, 37);
    let monthly = series(150, 23);
    let mut negative_zero = vec![-0.0; 30];
    negative_zero[7] = 0.0;
    let one = 1.0_f64.to_bits();
    let cases = [
        (
            "airline quarterly",
            ordered_ma_residuals(4, &[0.3], Some(0.6), &quarterly),
            (65, 0x4031e7e34af0223d, 0x3ff040febbeabc51),
        ),
        (
            "airline monthly",
            ordered_ma_residuals(12, &[-0.4], Some(0.7), &monthly),
            (163, 0x4047fd7733c8b1c2, 0x3ff072412f908d2c),
        ),
        (
            "ma2 seasonal monthly",
            ordered_ma_residuals(12, &[0.5, -0.3], Some(0.4), &monthly),
            (164, 0x405026ea9687e7e3, 0x3ff02179efb2a953),
        ),
        (
            "ma2 seasonal quarterly",
            ordered_ma_residuals(4, &[-0.2, 0.45], Some(-0.5), &quarterly),
            (66, 0x4044eadf1bba1d95, 0x3ff02c993a44f970),
        ),
        (
            "ma2 regular quarterly",
            ordered_ma_residuals(4, &[0.5, -0.3], None, &quarterly),
            (62, 0x4030988eac48bd70, 0x3ff00beb7a4f1e40),
        ),
        (
            "ma2 regular monthly",
            ordered_ma_residuals(12, &[-0.6, 0.2], None, &monthly),
            (152, 0x40565997500fa104, 0x3ff00c6ba0eab9c6),
        ),
        (
            "airline tiny regular K",
            ordered_ma_residuals(12, &[1e-3], Some(0.5), &monthly),
            (163, 0x40464bdff71f2dae, 0x3ff02f67f4b053f2),
        ),
        (
            "airline unit roots",
            ordered_ma_residuals(4, &[1.0], Some(-1.0), &quarterly),
            (65, 0x405797f9af9cc410, 0x3ff1f87ccddd882b),
        ),
        (
            "airline negative zero",
            ordered_ma_residuals(4, &[-0.0], Some(0.0), &negative_zero),
            (35, 0, one),
        ),
        (
            "seasonal quarterly",
            seasonal_constrained_residuals(4, 0.6, &quarterly),
            (64, 0x40349f6b1ca7e649, 0x3ff03d637738303b),
        ),
        (
            "seasonal monthly",
            seasonal_constrained_residuals(12, -0.5, &monthly),
            (162, 0x4050c2836abd911e, 0x3ff02f67f3c89c9a),
        ),
        (
            "seasonal unit root",
            seasonal_constrained_residuals(12, 1.0, &monthly),
            (162, 0x404aa9d5926814ff, 0x3ff1c149a55f9cd4),
        ),
        (
            "seasonal negative zero",
            seasonal_constrained_residuals(4, -0.0, &negative_zero),
            (34, 0, one),
        ),
        (
            "regular",
            regular_ma_residuals(0.4, &quarterly),
            (61, 0x4030742d8155473d, 0x3ff005f4a166fe60),
        ),
        (
            "regular tiny K",
            regular_ma_residuals(1e-4, &monthly),
            (151, 0x4049a61e8697494a, 0x3ff0000000024a68),
        ),
        (
            "regular negative zero",
            regular_ma_residuals(-0.0, &negative_zero),
            (31, 0, one),
        ),
    ];
    for (label, result, expected) in cases {
        assert_eq!(summary(result), Ok(expected), "{label}");
    }
}

#[test]
fn calcfx_reports_an_objective_that_overflows() {
    let mut huge = vec![0.0; 40];
    huge[0] = f64::MAX / 2.0;
    huge[1] = f64::MAX / 2.0;
    let mut big_run = vec![0.0; 40];
    big_run[..12].fill(f64::MAX / 3.0);
    let cases = [
        (
            "airline",
            ordered_ma_residuals(4, &[0.5], Some(0.5), &huge),
            objective_overflow(1),
        ),
        (
            "ma2",
            ordered_ma_residuals(4, &[0.6, 0.0], None, &big_run),
            objective_overflow(2),
        ),
        (
            "seasonal",
            seasonal_constrained_residuals(4, 0.6, &big_run),
            objective_overflow(1),
        ),
        (
            "regular",
            regular_ma_residuals(0.5, &huge),
            objective_overflow(1),
        ),
        (
            "regular run",
            regular_ma_residuals(0.6, &big_run),
            objective_overflow(1),
        ),
    ];
    for (label, result, expected) in cases {
        assert_eq!(result, Err(expected), "{label}");
    }
}

#[test]
fn ar_prefilter_subtracts_each_lag_and_keeps_zero_signs() {
    let filter =
        |phi: &[f64], wd: &[f64]| ar_prefilter(phi, wd, |_, _, value| Ok::<_, ()>(value)).unwrap();
    // 3 − 0.5·2 + 0.25·1 and 4 − 0.5·3 + 0.25·2.
    assert_eq!(filter(&[0.5, -0.25], &[1.0, 2.0, 3.0, 4.0]), [2.25, 3.0]);
    assert!(filter(&[0.5, -0.25], &[1.0, 2.0]).is_empty());
    // The fused product decides the zero: −0 − (0·−0) is +0, −0 − (−0·−0) is −0.
    assert_eq!(
        filter(&[0.0], &[-0.0, -0.0])[0].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        filter(&[-0.0], &[-0.0, -0.0])[0].to_bits(),
        (-0.0_f64).to_bits()
    );

    // Each partial result reaches the check with its lag and position.
    let overflow = ar_prefilter(
        &[0.0, -1.0],
        &[f64::MAX, 0.0, f64::MAX],
        |lag, index, value| {
            if value.is_finite() {
                Ok(value)
            } else {
                Err((lag, index))
            }
        },
    );
    assert_eq!(overflow, Err((1, 0)));
}

#[test]
fn ar_conditioned_solves_run_on_the_prefiltered_series() {
    let wd = series(40, 37);
    let quarterly = Frequency::Quarterly;
    assert_eq!(
        summary(ar1_residuals(quarterly, [0.5, 0.6], &wd, true)),
        Ok((43, 0x403592dbdafbba57, 0x3ff05ed1eed019f9))
    );
    assert_eq!(
        summary(ar1_residuals(quarterly, [-0.7, 0.0], &wd, false)),
        Ok((39, 0x40297a4bdba0a526, 1.0_f64.to_bits()))
    );

    let mut overflow = vec![0.0; 20];
    overflow[6..10].fill(f64::MAX);
    assert_eq!(
        ar1_residuals(quarterly, [-1.0, 0.0], &overflow, false),
        Err(Failure::Unusable {
            quantity: "conditional AR prefilter",
            index: Some(6),
            value: f64::from_bits(OVERFLOW),
        }
        .at(Stage::Residual))
    );
    for (ar, quantity, index) in [
        (Two([-1.0, 0.0]), "Phi1 prefilter", 5),
        (Two([0.0, -1.0]), "Phi2 prefilter", 6),
        (Three([0.0, 0.0, -1.0]), "Phi3 prefilter", 6),
    ] {
        assert_eq!(
            ar2_prefilter(ar, &overflow),
            Err(Failure::Unusable {
                quantity,
                index: Some(index),
                value: f64::from_bits(OVERFLOW),
            }
            .at(Stage::Residual))
        );
    }
}
