use super::*;
use crate::seats::extraction::ExtractedComponents;

fn log_input(original: &[f64]) -> Result<LogInput, Error> {
    let transformed = original.iter().map(|value| value.ln()).collect::<Vec<_>>();
    prepare_transformed_log_input(Frequency::Quarterly, original, &transformed)
}

fn zero_components(n: usize) -> ExtractedComponents {
    ExtractedComponents {
        frequency: Frequency::Quarterly,
        observed_len: n,
        trend: vec![0.0; n],
        seasonal: vec![0.0; n],
        irregular: vec![0.0; n],
        cycle: None,
    }
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "pins exact results of the Apple silicon math library"
)]
fn log_input_is_the_log_of_the_exponentiated_log() {
    let source = f64::from_bits(0x3fff_cc88_3697_91ac);
    let prepared = log_input(&[source; 10]).unwrap();
    assert_eq!(source.ln().to_bits(), 0x3fe5_faa1_a531_7b63);
    assert_eq!(source.ln().exp().to_bits(), 0x3fff_cc88_3697_91ad);
    assert_eq!(prepared.logged()[0].to_bits(), 0x3fe5_faa1_a531_7b64);
}

#[test]
fn partial_year_seasonal_average_excludes_remainder_but_irregular_uses_all() {
    let input = log_input(&[2.0; 11]).unwrap();
    let raw = ExtractedComponents {
        frequency: Frequency::Quarterly,
        observed_len: 11,
        trend: vec![0.0; 11],
        seasonal: [vec![0.0; 8], vec![2.0; 3]].concat(),
        irregular: [vec![0.0; 8], vec![std::f64::consts::LN_2; 3]].concat(),
        cycle: None,
    };
    let components = final_log(&input, &raw).unwrap();
    assert_eq!(components.seasonal[0].to_bits(), 1.0_f64.to_bits());
    assert_eq!(components.seasonal[8].to_bits(), 2.0_f64.exp().to_bits());
    assert_ne!(
        components.seasonal[0].to_bits(),
        (1.0 / ((8.0 + 3.0 * 2.0_f64.exp()) / 11.0)).to_bits()
    );

    // EXP(0)=1 for the first eight slots and EXP(LN_2)=2 for the last
    // three. Ordered all-slot addition is exactly 14, so Bias2=14/11;
    // dropping the remainder would instead yield Bias2=8/8=1.
    assert_eq!(std::f64::consts::LN_2.exp().to_bits(), 2.0_f64.to_bits());
    let all_irregular_mean = 14.0_f64 / 11.0;
    assert_eq!(all_irregular_mean.to_bits(), 0x3ff4_5d17_45d1_745d);
    assert_eq!(components.trend[0].to_bits(), all_irregular_mean.to_bits());
    assert_eq!(components.irregular[0].to_bits(), 0x3ff9_2492_4924_9249);
    let truncated_mean = 8.0_f64 / 8.0;
    assert_ne!(components.trend[0].to_bits(), truncated_mean.to_bits());
    assert_ne!(components.irregular[0].to_bits(), 2.0_f64.to_bits());
}

/// The bias correction's refusal of `value`, the `quantity` at `index`.
fn unusable(quantity: &'static str, index: usize, value: f64) -> Result<FinalComponents, Error> {
    Err(Error {
        stage: Stage::BiasCorrection,
        failure: Failure::Unusable {
            quantity,
            index: Some(index),
            value,
        },
    })
}

#[test]
fn final_log_refuses_a_component_whose_exp_overflows_or_underflows() {
    let input = log_input(&[2.0; 10]).unwrap();
    assert!(
        final_log(&input, &zero_components(10)).is_ok(),
        "zero log components mean unit factors"
    );
    for (log, value) in [(1000.0, f64::INFINITY), (-1000.0, 0.0)] {
        let mut raw = zero_components(10);
        raw.seasonal[0] = log;
        assert_eq!(final_log(&input, &raw), unusable("seasonal exp", 0, value));
        let mut raw = zero_components(10);
        raw.irregular[0] = log;
        assert_eq!(final_log(&input, &raw), unusable("irregular exp", 0, value));
        let mut raw = zero_components(10);
        raw.trend[0] = log;
        assert_eq!(final_log(&input, &raw), unusable("trend exp", 0, value));
    }
}

#[test]
fn final_log_refuses_an_overflowing_bias_or_output_value() {
    let mut input = log_input(&[2.0; 10]).unwrap();
    let mut raw = zero_components(10);
    // exp(709) is finite, and the third term of the sum overflows.
    raw.irregular.fill(709.0);
    assert_eq!(
        final_log(&input, &raw),
        unusable("irregular accumulation", 2, f64::INFINITY)
    );
    raw.irregular.fill(0.0);
    input.logged[0] = 708.0;
    assert_eq!(
        final_log(&input, &raw),
        unusable("irregular percent", 0, f64::INFINITY)
    );
}

#[test]
fn log_transitory_is_left_out_at_the_variance_threshold() {
    let input = log_input(&[128.0; 16]).unwrap();
    let raw = ExtractedComponents {
        frequency: Frequency::Quarterly,
        observed_len: 16,
        trend: vec![128.0_f64.ln(); 16],
        seasonal: vec![0.0; 16],
        irregular: vec![0.0; 16],
        cycle: Some(vec![0.2; 16]),
    };
    let low = final_log_with_transitory(&input, &raw, 1e-10).unwrap();
    let high = final_log_with_transitory(&input, &raw, 1e-10_f64.next_up()).unwrap();
    assert!(low.transitory.is_none());
    assert_eq!(
        low.saved, high.saved,
        "SECOND uses its working cycle even when no S14 is saved"
    );
    let cycles = high.transitory.unwrap();
    assert!(
        cycles
            .iter()
            .all(|v| v.to_bits() == ((100.0 * 0.2_f64.exp()) / 100.0).to_bits())
    );
    assert!(
        high.saved.trend.iter().all(|v| *v == 128.0_f64.ln().exp()),
        "Bias1 irregular mean is EXP(raw irregular), not EXP(irregular+cycle)"
    );
    let mut neutral = raw;
    neutral.cycle = Some(vec![0.0; 16]);
    assert_eq!(
        final_log_with_transitory(&input, &neutral, 1.0)
            .unwrap()
            .transitory,
        Some(vec![1.0; 16])
    );
}
