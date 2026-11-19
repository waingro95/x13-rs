use super::*;

#[test]
fn grids_place_the_trading_day_frequencies_in_build_62_slots() {
    let regular = frequencies(SpectralGrid::Regular);
    let expanded = frequencies(SpectralGrid::Expanded);
    assert_eq!(regular.len(), 61);
    assert_eq!(expanded.len(), 67);
    let step: f64 = 1.0 / 120.0;
    for (grid, slots) in [
        (&regular, [41, 42, 43, 51, 52, 53]),
        (&expanded, [41, 43, 45, 54, 56, 58]),
    ] {
        let expected = [
            0.3482 - step,
            0.3482,
            0.3482 + step,
            0.432 - step,
            0.432,
            0.432 + step,
        ];
        for (slot, expected) in slots.into_iter().zip(expected) {
            assert_eq!(grid[slot].to_bits(), expected.to_bits());
        }
        let remaining: Vec<_> = grid
            .iter()
            .enumerate()
            .filter(|(i, _)| !slots.contains(i))
            .map(|(_, x)| *x)
            .collect();
        let base: Vec<_> = (0..61)
            .filter(|i| grid.len() == 67 || !slots.contains(i))
            .map(|i| i as f64 / 120.0)
            .collect();
        assert_eq!(remaining, base);
    }
}

#[test]
fn short_nonfinite_and_constant_inputs_are_errors() {
    for values in [&[][..], &[1.0][..]] {
        assert_eq!(
            estimate(values, SpectralGrid::Expanded),
            Err(SpectralUnavailable::InsufficientInput {
                length: values.len(),
                required: 2
            })
        );
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            estimate(&[1.0, bad, 2.0], SpectralGrid::Regular),
            Err(SpectralUnavailable::NonfiniteInput { index: 1 })
        );
        // Finiteness is checked before length: a lone nonfinite value is not
        // reported as too short.
        assert_eq!(
            estimate(&[bad], SpectralGrid::Regular),
            Err(SpectralUnavailable::NonfiniteInput { index: 0 })
        );
    }
    for values in [[3.0, 3.0, 3.0], [0.0, 1e-11, 0.0]] {
        assert_eq!(
            estimate(&values, SpectralGrid::Expanded),
            Err(SpectralUnavailable::ZeroCovariance)
        );
    }
    assert_eq!(
        estimate(&[f64::MAX, f64::MAX], SpectralGrid::Expanded),
        Err(SpectralUnavailable::NonfiniteArithmetic {
            stage: SpectralArithmeticStage::MeanSum
        })
    );
}

#[test]
fn recurrence_returns_maximum_order_even_when_aic_prefers_one() {
    let fitted = fit_autocovariances(&[1.0, 0.5, 0.25], 100).unwrap();
    let aic_one = 100.0 * 0.75_f64.ln() + 2.0;
    let aic_two = 100.0 * 0.75_f64.ln() + 4.0;
    assert!(aic_one < aic_two);
    assert_eq!(fitted.coefficients, [-0.5, 0.0]);
    assert_eq!(fitted.variance, 0.75);
}

#[test]
fn variance_collapse_before_the_last_order_is_an_error() {
    assert_eq!(
        fit_autocovariances(&[1.0, 0.999999, 0.999998, 0.999997], 100).unwrap_err(),
        SpectralUnavailable::EarlyVarianceCollapse {
            completed_order: 1,
            required_order: 3
        }
    );
    // The same small positive variance at the final requested order has no
    // unwritten tail: `sicp2` tests its cutoff before the next iteration.
    let completed = fit_autocovariances(&[1.0, 0.999999], 100).unwrap();
    assert_eq!(completed.coefficients, [-0.999999]);
    assert!(completed.variance > 0.0 && completed.variance < 0.00001);
}

#[test]
fn unit_covariance_denominator_and_coefficient_reversal() {
    // Mean2, centered[-3,-1,1,3], products divided by N=4 at every lag.
    assert_eq!(
        autocovariances(&[-1.0, 1.0, 3.0, 5.0], 67).unwrap(),
        [5.0, 1.25, -1.5, -2.25]
    );
    // Order1 a=.5, sd=.75; order2 d=-.5, a=[.75,-.5], sd=.5625.
    let fitted = fit_autocovariances(&[1.0, 0.5, -0.125], 100).unwrap();
    assert_eq!(fitted.coefficients, [-0.75, 0.5]);
    assert_eq!(fitted.variance, 0.5625);
}

#[test]
fn two_observations_use_order_one() {
    let values = [-1.0, 1.0];
    let spectrum = estimate(&values, SpectralGrid::Regular).unwrap();
    assert_eq!(spectrum.order, 1);
    assert_eq!(spectrum.frequencies.len(), 61);
    assert_eq!(spectrum.decibels.len(), 61);
    assert!(spectrum.decibels.iter().all(|x| x.is_finite()));
}

#[test]
fn nonpositive_variance_and_nonfinite_recurrence_are_errors() {
    for lag in [1.0, 2.0] {
        assert_eq!(
            fit_autocovariances(&[1.0, lag], 100).unwrap_err(),
            SpectralUnavailable::NonpositiveInnovationVariance { order: 1 }
        );
    }
    assert_eq!(
        fit_autocovariances(&[1.0, f64::MAX], 100).unwrap_err(),
        SpectralUnavailable::NonfiniteArithmetic {
            stage: SpectralArithmeticStage::ReflectionSquare
        }
    );
}

#[test]
fn order_four_recurrence_coefficient_and_variance_bits() {
    let fitted = fit_autocovariances(&[1.0, 0.6, 0.3, -0.1, 0.05], 50).unwrap();
    assert_eq!(
        fitted
            .coefficients
            .iter()
            .map(|coefficient| coefficient.to_bits())
            .collect::<Vec<_>>(),
        [
            0xbfea_69a6_9a69_a69a,
            0xbfb2_4924_9249_2493,
            0x3fe6_db6d_b6db_6db7,
            0xbfe1_4514_5145_1451,
        ]
    );
    assert_eq!(fitted.variance.to_bits(), 0x3fd8_a28a_28a2_8a2a);
}
