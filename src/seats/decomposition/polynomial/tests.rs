use super::{convolve, spectral_inputs};
use crate::Frequency;
use crate::seats::decomposition::test_support::airline;

#[test]
fn convolve_fuses_each_term_into_its_coefficient() {
    assert_eq!(convolve(&[1.0, -1.0], &[1.0, 2.0]), [1.0, 1.0, -2.0]);
    let e = f64::EPSILON;
    let actual = convolve(&[-1.0, 1.0 + e], &[1.0 - e, 1.0]);
    assert_eq!(actual[1].to_bits(), (-e * e).to_bits());
}

#[test]
fn convolve_accumulates_left_outer() {
    // Right-outer accumulation reverses the lag-two terms and loses the unit.
    let actual = convolve(&[1e16, -1e16, 1.0], &[1.0, 1.0, 1.0]);
    assert_eq!(actual, [1e16, 0.0, 1.0, -1e16, 1.0]);
}

#[test]
fn airline_polynomials() {
    for (frequency, seasonal_theta, aggregate_ma, seasonal, denominator) in [
        (
            Frequency::Quarterly,
            vec![1.0, 0.0, 0.0, 0.0, -0.6],
            vec![1.0, -0.4, 0.0, 0.0, -0.6, 0.4 * 0.6],
            vec![1.0; 4],
            vec![1.0, -1.0, 0.0, 0.0, -1.0, 1.0],
        ),
        (
            Frequency::Monthly,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.6,
            ],
            vec![
                1.0,
                -0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.6,
                0.4 * 0.6,
            ],
            vec![1.0; 12],
            vec![
                1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0,
            ],
        ),
    ] {
        let actual = spectral_inputs(&airline(frequency, [0.4, 0.6]));
        assert_eq!(actual.theta, [1.0, -0.4]);
        assert_eq!(actual.seasonal_theta, seasonal_theta);
        assert_eq!(actual.aggregate_ma, aggregate_ma);
        assert_eq!(actual.trend.stationary, [1.0]);
        assert_eq!(actual.trend.nonstationary, [1.0, -2.0, 1.0]);
        assert_eq!(actual.trend.combined, [1.0, -2.0, 1.0]);
        assert_eq!(actual.seasonal.stationary, [1.0]);
        assert_eq!(actual.seasonal.nonstationary, seasonal);
        assert_eq!(actual.seasonal.combined, seasonal);
        assert_eq!(actual.cycle.stationary, [1.0]);
        assert_eq!(actual.cycle.nonstationary, [1.0]);
        assert_eq!(actual.cycle.combined, [1.0]);
        assert_eq!(actual.trend_cycle, [1.0, -2.0, 1.0]);
        assert_eq!(actual.seasonal_cycle, seasonal);
        assert_eq!(actual.trend_seasonal, denominator);
        assert_eq!(actual.adjustment_stationary, [1.0]);
        assert_eq!(actual.adjustment_nonstationary, [1.0, -2.0, 1.0]);
        assert_eq!(actual.aggregate_denominator, denominator);
    }
}
