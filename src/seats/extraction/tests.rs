use super::*;
use crate::seats::ar1_seasonal_ma::model::{
    Ar1SeasonalMaInput, ar1_seasonal_ma_geometry, import_ar1_seasonal_ma,
};
use crate::seats::mean::RequestedMean;
use crate::seats::tails::{ForwardTailState, difference_ordered};

#[test]
fn final_identity_recomputes_sa_and_adds_zero_effects() {
    let raw = ExtractedComponents {
        frequency: Frequency::Quarterly,
        observed_len: 1,
        trend: vec![3.0],
        seasonal: vec![2.0],
        irregular: vec![-0.0],
        cycle: None,
    };
    let saved = final_identity(&[10.0], &raw).unwrap();
    assert_eq!(saved.seasonal[0].to_bits(), 2.0_f64.to_bits());
    assert_eq!(saved.seasonally_adjusted[0].to_bits(), 8.0_f64.to_bits());
    assert_eq!(saved.trend[0].to_bits(), 3.0_f64.to_bits());
    assert_eq!(saved.irregular[0].to_bits(), 0.0_f64.to_bits());
}

#[test]
fn transitory_gains_a_zero_effect_only_above_the_variance_threshold() {
    let raw = ExtractedComponents {
        frequency: Frequency::Quarterly,
        observed_len: 4,
        trend: vec![10.0; 4],
        seasonal: vec![0.0; 4],
        irregular: vec![0.0; 4],
        cycle: Some(vec![-0.0, 0.25, 0.0, -0.25]),
    };
    for variance in [1e-10_f64.next_down(), 1e-10, 1e-10_f64.next_up()] {
        let components = final_identity_with_transitory(&[10.0; 4], &raw, variance).unwrap();
        let transitory = components.transitory.unwrap();
        assert_eq!(
            transitory[0].to_bits(),
            if variance > 1e-10 {
                0
            } else {
                (-0.0_f64).to_bits()
            }
        );
        assert_eq!(&transitory[1..], &[0.25, 0.0, -0.25]);
        assert_eq!(components.saved.trend, [10.0; 4]);
    }
    let mut zero = raw;
    zero.cycle = Some(vec![-0.0; 4]);
    assert!(
        final_identity_with_transitory(&[10.0; 4], &zero, 1.0)
            .unwrap()
            .transitory
            .is_none(),
        "an all-zero transitory is left out"
    );
    for threshold in [
        3.834e-20_f64.next_down(),
        3.834e-20,
        3.834e-20_f64.next_up(),
    ] {
        assert_eq!(component_present(&[threshold]), threshold >= 3.834e-20);
        assert_eq!(component_present(&[-threshold]), threshold >= 3.834e-20);
    }
}

#[test]
fn ar1_cycle_is_subtracted_from_the_irregular() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let n = if frequency == Frequency::Monthly {
            180
        } else {
            80
        };
        let s = usize::from(frequency.periods_per_year());
        let original = (0..n)
            .map(|i| 128.0 + ((i * 37 + 11) % 101) as f64 / 16.0)
            .collect::<Vec<_>>();
        let imported =
            import_ar1_seasonal_ma(&Ar1SeasonalMaInput::new(frequency, [-0.25, 0.4]).unwrap());
        let geometry = ar1_seasonal_ma_geometry(&imported).unwrap();
        let prefix =
            crate::seats::decomposition::rational::lower_degree_rational_prefix(&geometry.inputs)
                .unwrap();
        let canonical = prepare_canonical(
            crate::seats::decomposition::minima::spectral_minima_prefix(frequency, prefix),
        );
        let mut wd = difference_ordered(&original, s, DifferenceOrder::First).unwrap();
        let wm = crate::seats::mean::prepare_mean_wd(&mut wd, RequestedMean::No).unwrap();
        let residuals = crate::seats::ar1_seasonal_ma::residual::ar1_residuals(
            frequency,
            imported.parameters(),
            &wd,
            true,
        )
        .unwrap();
        let tails = crate::seats::ar1_seasonal_ma::finish_ar1_tails(
            &imported,
            &original,
            &wd,
            ForwardTailState {
                wm,
                requested_mean: RequestedMean::No,
                residuals: residuals.normalized_residuals,
            },
        )
        .unwrap();
        let raw = extract_ar1(&imported, &geometry.inputs, &canonical, &original, &tails).unwrap();
        let cycle = raw.cycle.as_ref().expect("the AR(1) root makes a cycle");
        assert_eq!(cycle.len(), n);
        assert!(cycle.iter().any(|v| *v != 0.0));
        for i in 0..n {
            assert_eq!(
                raw.irregular[i].to_bits(),
                (((original[i] - raw.seasonal[i]) - raw.trend[i]) - cycle[i]).to_bits()
            );
        }
    }
}

#[test]
fn negative_irregular_spectrum_is_rejected() {
    let model = ResolvedAirline {
        frequency: Frequency::Quarterly,
        ma: [0.4, 0.6],
    };
    let mut minima = spectral_minima(&model).unwrap();
    minima.irregular_candidate = -1.0;
    assert_eq!(
        airline_canonical(minima.clone()),
        Err(Error {
            stage: Stage::Extraction,
            failure: Failure::NegativeVariance {
                component: CanonicalComponent::Irregular,
                variance: -1.0,
            },
        })
    );
    for candidate in [0.0, -0.0, 1.0] {
        minima.irregular_candidate = candidate;
        assert!(airline_canonical(minima.clone()).is_ok());
    }
}
