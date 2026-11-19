use super::*;
use crate::Frequency;
use crate::seats::ar1_seasonal_ma::model::{
    Ar1SeasonalMaInput, ar1_seasonal_ma_geometry, import_ar1_seasonal_ma,
};
use crate::seats::decomposition::canonical::{CanonicalPreparation, prepare_canonical};
use crate::seats::decomposition::minima::spectral_minima_prefix;
use crate::seats::decomposition::rational::lower_degree_rational_prefix;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The canonical components of an AR(1) model with a seasonal MA.
fn ar1_preparation(frequency: Frequency) -> (CanonicalPreparation, SpectralInputs) {
    let input = Ar1SeasonalMaInput::new(frequency, [-0.25, 0.4]).unwrap();
    let geometry = ar1_seasonal_ma_geometry(&import_ar1_seasonal_ma(&input)).unwrap();
    let prefix = lower_degree_rational_prefix(&geometry.inputs).unwrap();
    let minima = spectral_minima_prefix(frequency, prefix);
    (prepare_canonical(minima), geometry.inputs)
}

#[test]
fn trend_products_are_fused_into_the_scaled_irregular() {
    // (1+2^-52)(1-2^-52) - 1 is exactly -2^-104. Rounding the product first
    // gives 1 and loses it.
    let e = f64::EPSILON;
    let summed = assemble_numerator(&[-1.0, 0.0, 0.0], 1.0, &[1.0 - e, 0.0, 0.0], 1.0 + e, None);
    assert_eq!(bits(&summed), [0xb970_0000_0000_0000, 0, 0]);
}

#[test]
fn irregular_scaling_rounds_before_the_fused_add() {
    // The scale 1-2^-104 rounds to 1, and then fma(1, -1, 1) is +0. Fusing
    // across both steps would keep -2^-104.
    let e = f64::EPSILON;
    let summed = assemble_numerator(&[1.0 + e, 0.0, 0.0], 1.0 - e, &[-1.0, 0.0, 0.0], 1.0, None);
    assert_eq!(bits(&summed), [0, 0, 0]);
}

#[test]
fn negative_zero_products_and_sums_survive() {
    let summed = assemble_numerator(&[-0.0, 0.0, 0.0], 1.0, &[-0.0, 0.0, 0.0], 1.0, None);
    assert_eq!(bits(&summed), [1_u64 << 63, 0, 0]);
}

#[test]
fn cycle_is_padded_and_added_after_the_trend() {
    let base = [2.0, -4.0, 3.0];
    let trend = [1.0, 0.5, -0.25, 0.125];
    let cycle = [0.75, -0.125, 0.0625, -0.03125];
    let sum = assemble_numerator(&base, 0.1, &trend, 0.3, Some((&cycle, 0.7)));
    assert_eq!(sum.len(), 4);
    for i in 0..4 {
        let scaled = if i < 3 { base[i] * 0.1 } else { 0.0 };
        let expected = 0.7_f64.mul_add(cycle[i], 0.3_f64.mul_add(trend[i], scaled));
        assert_eq!(sum[i].to_bits(), expected.to_bits());
    }
}

#[test]
fn factors_every_component_of_an_ar1_model() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let (preparation, inputs) = ar1_preparation(frequency);
        let model = factor_adjusted(preparation, inputs, EffectiveTradingDay::Ordinary).unwrap();
        assert!(model.cycle.is_some());
        check_show_comp(&model).unwrap();
        for (component, variance) in show_comp_variances(&model).into_iter().flatten() {
            assert!(variance > 0.0, "{component:?} {variance}");
        }
    }
}

#[test]
fn negative_irregular_variance_is_refused() {
    let (mut preparation, inputs) = ar1_preparation(Frequency::Monthly);
    let negative = -f64::from_bits(1);
    preparation.minima.irregular_candidate = negative;
    assert_eq!(
        factor_adjusted(preparation, inputs, EffectiveTradingDay::Ordinary).unwrap_err(),
        Failure::NegativeVariance {
            component: CanonicalComponent::Irregular,
            variance: negative,
        }
        .at(Stage::CanonicalDecomposition)
    );
}

#[test]
fn a_nonpositive_adjusted_numerator_is_refused() {
    let (mut preparation, inputs) = ar1_preparation(Frequency::Monthly);
    preparation.minima.irregular_candidate = 0.0;
    preparation.trend.numerator = vec![-100.0, 0.0];
    assert_eq!(
        factor_adjusted(preparation, inputs, EffectiveTradingDay::Ordinary).unwrap_err(),
        Failure::UnusableNumerator {
            zero_lag: -106.04520001098633
        }
        .at(Stage::CanonicalDecomposition)
    );
}

#[test]
fn an_innovation_variance_above_one_is_refused() {
    let (preparation, inputs) = ar1_preparation(Frequency::Monthly);
    let mut model = factor_adjusted(preparation, inputs, EffectiveTradingDay::Ordinary).unwrap();
    model.adjusted.variance = 1.5;
    assert_eq!(
        check_show_comp(&model),
        Err(Failure::InnovationVariance {
            component: Component::SeasonallyAdjusted,
            direction: VarianceDirection::AboveOne,
            value: 1.5,
        }
        .at(Stage::CanonicalDecomposition))
    );
}

#[test]
fn innovation_variances_outside_the_unit_interval_are_refused() {
    assert_eq!(show_comp_direction(0.0), None);
    assert_eq!(show_comp_direction(1.0), None);
    assert_eq!(
        show_comp_direction(-f64::from_bits(1)),
        Some(VarianceDirection::BelowZero)
    );
    assert_eq!(
        show_comp_direction(1.0_f64.next_up()),
        Some(VarianceDirection::AboveOne)
    );
}
