use super::*;
use crate::seats::ar1_seasonal_ma::model::{
    Ar1SeasonalMaInput, ar1_seasonal_ma_geometry, import_ar1_seasonal_ma,
};
use crate::seats::decomposition::canonical::prepare_canonical;
use crate::seats::decomposition::rational::lower_degree_rational_prefix;
use crate::seats::decomposition::test_support::{airline, assert_close};

#[test]
fn seasonal_minimum_is_capped_at_ten_and_keeps_the_first_tie() {
    assert_eq!(seasonal_minimum(&[11.0, 12.0]), 10.0);
    assert_eq!(seasonal_minimum(&[3.0, 2.0]), 2.0);
    for (first, second) in [(-0.0, 0.0), (0.0, -0.0)] {
        assert_eq!(
            seasonal_minimum(&[first, second]).to_bits(),
            f64::to_bits(first)
        );
    }
}

#[test]
fn trend_minimum_takes_the_negative_zero_of_a_tie() {
    for (first, second, expected) in [
        (0.0, 0.0, 0.0),
        (-0.0, -0.0, -0.0),
        (-0.0, 0.0, -0.0),
        (0.0, -0.0, -0.0),
        (2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
    ] {
        assert_eq!(
            trend_minimum(first, second).to_bits(),
            f64::to_bits(expected)
        );
    }
}

#[test]
fn grid_test_is_fused_and_adoption_is_strict() {
    let e = f64::from_bits(0x3e40_0000_0000_0000); // 2^-27
    let (first, factor, minimum) = (-1.0, -(1.0 + e), 1.0 - e);
    assert_eq!((first - factor * minimum).to_bits(), 0.0_f64.to_bits());
    assert!(grid_required(first, factor, minimum));
    assert!(!grid_required(0.0, 1.0, 0.0));
    let mut minimum = 0.0;
    adopt_grid(&mut minimum, -0.0);
    assert_eq!(minimum.to_bits(), 0.0_f64.to_bits());
    let mut minimum = 2.0;
    adopt_grid(&mut minimum, 2.5);
    assert_eq!(minimum, 2.0);
    adopt_grid(&mut minimum, 1.0);
    assert_eq!(minimum, 1.0);
}

#[test]
fn irregular_sum_keeps_its_order_and_ends_on_a_positive_zero() {
    assert_eq!(irregular_without_cycle(1e16, -1e16, 1.0), 1.0);
    assert_eq!(
        irregular_without_cycle(-0.0, -0.0, -0.0).to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "pins exact results of the Apple silicon math library"
)]
fn airline_minima() {
    for (frequency, enot, estar, irregular) in [
        (
            Frequency::Quarterly,
            0x3fac_d9e8_3e42_5aec,
            0x3f82_12d7_7318_fc4c,
            0x3fd3_87fc_b923_a29c,
        ),
        (
            Frequency::Monthly,
            0x3fac_53ec_283b_58b6,
            0x3f92_c027_4c98_96d9,
            0x3fd4_12a8_ef93_83e1,
        ),
    ] {
        let result = spectral_minima(&airline(frequency, [0.4, 0.6])).unwrap();
        assert_close(&[result.enot], &[f64::from_bits(enot)]);
        assert_close(&[result.estar], &[f64::from_bits(estar)]);
        assert_close(&[result.irregular_candidate], &[f64::from_bits(irregular)]);
        assert_eq!(result.cycle, None);
    }
}

#[test]
fn ar1_cycle_minimum_enters_the_irregular_sum_and_the_canonical_filter() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let geometry = ar1_seasonal_ma_geometry(&import_ar1_seasonal_ma(
            &Ar1SeasonalMaInput::new(frequency, [-0.25, 0.4]).unwrap(),
        ))
        .unwrap();
        let prefix = lower_degree_rational_prefix(&geometry.inputs).unwrap();
        let quotient = prefix.quotient[0];
        let raw = prefix.cycle_numerator.as_ref().unwrap()[0];
        let denominator = if raw >= 0.0 {
            1.25_f64 * 1.25
        } else {
            0.75_f64 * 0.75
        };
        let minima = spectral_minima_prefix(frequency, prefix);
        let enoc = minima.cycle.as_ref().unwrap().minimum;
        assert!((enoc - raw / denominator).abs() < 1e-11);
        let ordered = ((quotient + minima.enot) + minima.estar) + enoc;
        assert_eq!(minima.irregular_candidate.to_bits(), ordered.to_bits());
        let cycle = prepare_canonical(minima).cycle.unwrap();
        assert_eq!(cycle.numerator.len(), 2);
        let lags = usize::from(frequency.periods_per_year()) + 3;
        assert!(cycle.filter_lags[..lags].iter().any(|v| *v != 0.0));
        assert!(cycle.filter_lags[lags..].iter().all(|v| v.to_bits() == 0));
    }
}
