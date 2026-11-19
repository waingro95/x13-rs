use super::{build_harmonics, conjugate, multiply_harmonics};
use crate::Frequency;
use crate::seats::decomposition::polynomial::spectral_inputs;
use crate::seats::decomposition::test_support::{airline, assert_close};

#[test]
fn conjugate_folds_both_lag_signs_with_fused_terms() {
    let e = f64::EPSILON;
    let actual = conjugate(&[-1.0, 1.0 + e], &[1.0, 1.0 - e]);
    assert_eq!(actual.len(), 2);
    assert_eq!(actual[0].to_bits(), (-e * e).to_bits());
    assert_eq!(conjugate(&[1.0, 2.0, 3.0], &[4.0, 5.0]), [14.0, 28.0, 12.0]);
    assert_eq!(conjugate(&[4.0, 5.0], &[1.0, 2.0, 3.0]), [14.0, 28.0, 12.0]);
    assert_eq!(conjugate(&[0.0], &[1.0]), [0.0]);
}

#[test]
fn harmonic_product_covers_the_full_length() {
    let factor = [0.0, 0.0, 1.0];
    assert_eq!(
        multiply_harmonics(&factor, &factor),
        [0.5, 0.0, 0.0, 0.0, 0.5]
    );
    assert_eq!(multiply_harmonics(&[0.0], &[1.0]), [0.0]);
}

#[test]
fn harmonic_product_of_an_empty_factor_is_zero() {
    // MULTFN with a zero count leaves lplus1 cleared slots, not the operand.
    assert_eq!(multiply_harmonics(&[], &[1.0, 2.0, 3.0]), [0.0, 0.0]);
    assert_eq!(multiply_harmonics(&[4.0, 5.0], &[]), [0.0]);
    assert!(multiply_harmonics(&[], &[]).is_empty());
}

#[test]
fn conjugate_zero_signs() {
    assert_eq!(conjugate(&[-0.0], &[1.0])[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        conjugate(&[-f64::from_bits(1)], &[0.5])[0].to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn harmonic_product_zero_signs() {
    assert_eq!(
        multiply_harmonics(&[-0.0], &[1.0])[0].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        multiply_harmonics(&[-f64::from_bits(1)], &[0.5])[0].to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn airline_harmonics() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let period = usize::from(frequency.periods_per_year());
        let actual = build_harmonics(&spectral_inputs(&airline(frequency, [0.4, 0.6])));
        let mut numerator = vec![0.0; period + 2];
        numerator[0] = 1.5776;
        numerator[1] = -1.088;
        numerator[period - 1] = 0.48;
        numerator[period] = -1.392;
        numerator[period + 1] = 0.48;
        let mut denominator = vec![0.0; period + 2];
        denominator[0] = 4.0;
        denominator[1] = -4.0;
        denominator[period - 1] = 2.0;
        denominator[period] = -4.0;
        denominator[period + 1] = 2.0;
        let mut seasonal = vec![period as f64];
        seasonal.extend((1..period).map(|lag| (2 * (period - lag)) as f64));
        assert_close(&actual.numerator, &numerator);
        assert_close(&actual.trend, &[6.0, -8.0, 2.0]);
        assert_close(&actual.cycle, &[1.0]);
        assert_close(&actual.seasonal, &seasonal);
        assert_close(&actual.trend_cycle, &[6.0, -8.0, 2.0]);
        assert_close(&actual.denominator, &denominator);
    }
}
