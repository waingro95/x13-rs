use super::*;
use crate::Frequency;
use crate::seats::ar1_seasonal_ma::model::{
    Ar1SeasonalMaInput, ar1_seasonal_ma_geometry, import_ar1_seasonal_ma,
};
use crate::seats::decomposition::test_support::{airline, assert_close};

fn from_bits(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|&bits| f64::from_bits(bits)).collect()
}

#[test]
fn equal_order_division_keeps_an_unfused_remainder() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let period = usize::from(frequency.periods_per_year());
        let harmonics = build_harmonics(&spectral_inputs(&airline(frequency, [0.4, 0.6])));
        let (quotient, remainder) =
            divide_equal_order(&harmonics.numerator, &harmonics.denominator);
        assert_close(&quotient, &[0.24]);
        let mut expected = vec![0.0; period + 1];
        expected[0] = 0.6176;
        expected[1] = -0.128;
        expected[period] = -0.432;
        assert_close(&remainder, &expected);
    }
    let e = f64::EPSILON;
    let (quotient, remainder) = divide_equal_order(&[1.0, 1.0 - e], &[1.0 - e, 1.0]);
    assert_eq!(quotient, [1.0 - e]);
    let separate = 1.0 - (1.0 - e) * (1.0 - e);
    assert_eq!(remainder[0].to_bits(), separate.to_bits());
    assert_ne!(
        remainder[0].to_bits(),
        (-(1.0 - e)).mul_add(1.0 - e, 1.0).to_bits()
    );
}

#[test]
fn airline_partial_fractions_match_build_62() {
    // Build 62's PARFRA output. It differs from exact rational arithmetic in
    // the last bits, which these values keep.
    let quarterly = rational_prefix(&airline(Frequency::Quarterly, [0.4, 0.6])).unwrap();
    assert_close(
        &quarterly.trend_numerator,
        &from_bits(&[0x3fdc_f765_fd8a_dab7, 0xbfdc_bc6a_7ef9_db21]),
    );
    assert_close(
        &quarterly.seasonal_numerator,
        &from_bits(&[
            0x3fb0_3afb_7e90_ff95,
            0x3faf_212d_7731_8fbc,
            0x3f91_6872_b020_c495,
        ]),
    );
    let monthly = rational_prefix(&airline(Frequency::Monthly, [0.4, 0.6])).unwrap();
    assert_close(
        &monthly.trend_numerator,
        &from_bits(&[0x3fdc_5733_0498_bc3e, 0xbfdc_50a5_4bdd_f52e]),
    );
    assert_close(
        &monthly.seasonal_numerator,
        &from_bits(&[
            0x3fe6_1105_6a73_53be,
            0x3ff4_366e_68f2_d963,
            0x3ff1_6087_70e9_bf29,
            0x3fec_31b2_b8dd_bb9a,
            0x3fe5_9050_d3e6_5573,
            0x3fde_e6ec_627c_6e2a,
            0x3fd4_7a84_12c0_265e,
            0x3fc8_87ce_1eac_abe5,
            0x3fb9_73bc_18f0_aa48,
            0x3fa4_ef63_7118_59b3,
            0x3f85_583e_fcc4_cac0,
        ]),
    );
}

#[test]
fn an_ar1_root_below_one_half_adds_a_cycle_numerator() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        for phi in [0.75, -0.75, 0.25, -0.25, 0.0] {
            let input = Ar1SeasonalMaInput::new(frequency, [phi, 0.4]).unwrap();
            let geometry = ar1_seasonal_ma_geometry(&import_ar1_seasonal_ma(&input)).unwrap();
            let inputs = &geometry.inputs;
            let prefix = lower_degree_rational_prefix(inputs).unwrap();
            assert_eq!(prefix.quotient, [0.0]);
            assert_eq!(
                prefix.trend_numerator.len(),
                inputs.trend.combined.len() - 1
            );
            assert_eq!(
                prefix.seasonal_numerator.len(),
                inputs.seasonal.combined.len() - 1
            );
            assert_eq!(
                prefix.cycle_numerator.as_ref().map(Vec::len),
                (phi.abs() < 0.5).then_some(1)
            );
            if phi == 0.0 {
                // SPECTRU floors a constant cycle numerator at 1e-15.
                assert_eq!(prefix.cycle_numerator, Some(vec![1e-15]));
            }
        }
    }
}
