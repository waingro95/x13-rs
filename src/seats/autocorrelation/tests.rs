use super::*;

fn wave(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 101) as f64 / 16.0)
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn auto(
    values: &mut [f64],
    stage: MeanDiagnosticStage,
    m: usize,
    iq: usize,
    nw: usize,
) -> Result<AutoResult, Error> {
    autocorrelations(values, stage, AutoCallDimensions { m, iq, nw })
}

#[test]
fn lag_counts_fit_a_short_series_and_persist_over_passes() {
    for (frequency, count, m, iq) in [
        (Frequency::Monthly, 36, 23, 21),
        (Frequency::Monthly, 37, 24, 22),
        (Frequency::Monthly, 180, 36, 24),
        (Frequency::Quarterly, 16, 11, 9),
        (Frequency::Quarterly, 17, 12, 10),
        (Frequency::Quarterly, 21, 16, 14),
        (Frequency::Quarterly, 80, 36, 16),
    ] {
        let mut state = AutoLagState::new(frequency);
        for mean in [RequestedMean::Yes, RequestedMean::No, RequestedMean::Yes] {
            assert_eq!(
                state.enter_difference_pass(frequency, count, mean, 2, DifferenceOrder::First),
                Ok(())
            );
            assert_eq!(state, AutoLagState { m, iq });
        }
    }
}

#[test]
fn autocorrelations_and_q_of_a_ramp() {
    let mut values = [1.0, 2.0, 3.0, 4.0];
    let result = auto(&mut values, MeanDiagnosticStage::ForwardResidual, 3, 2, 4).unwrap();
    assert_eq!(result.active_lags, 3);
    assert_eq!(result.acf[..3], [0.25, -0.3, -0.45]);
    assert!(result.acf[3..].iter().all(|value| value.to_bits() == 0));
    assert_eq!(
        result.qstat,
        (0.25 * 0.25 / 3.0 + -0.3 * -0.3 / 2.0) * 4.0 * 6.0
    );
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_short_call_gives_the_first_lags_of_a_full_one() {
    for (frequency, n, m, iq) in [
        (Frequency::Monthly, 36, 23, 21),
        (Frequency::Monthly, 37, 24, 22),
        (Frequency::Quarterly, 16, 11, 9),
        (Frequency::Quarterly, 17, 12, 10),
        (Frequency::Quarterly, 21, 16, 14),
    ] {
        let differenced = n - usize::from(frequency.periods_per_year()) - 1;
        for (stage, nw) in [
            (MeanDiagnosticStage::ForwardResidual, n),
            (MeanDiagnosticStage::SquaredResidual, differenced),
        ] {
            let input = match stage {
                MeanDiagnosticStage::SquaredResidual => {
                    wave(n).iter().map(|value| value * value).collect()
                }
                _ => wave(n),
            };
            let mut short_input = input.clone();
            let short = auto(&mut short_input, stage, m, iq, nw).unwrap();
            let mut full_input = input;
            let full = auto(&mut full_input, stage, 36, 0, nw).unwrap();
            assert_eq!(short.active_lags, m);
            assert_eq!(bits(&short.acf[..m]), bits(&full.acf[..m]));
            assert!(short.acf[m..].iter().all(|value| value.to_bits() == 0));
            assert_eq!(bits(&short_input), bits(&full_input));

            let mut q = 0.0;
            for lag in 1..=iq {
                q += short.acf[lag - 1] * short.acf[lag - 1] / (nw - lag) as f64;
            }
            q = q * nw as f64 * (nw + 2) as f64;
            assert_eq!(short.qstat.to_bits(), q.to_bits());
            assert_eq!(full.qstat, -1.0);
        }
    }
}

#[test]
fn lags_from_the_series_length_on_are_zero() {
    let mut values = wave(11);
    let result = auto(&mut values, MeanDiagnosticStage::Differenced, 36, 0, 11).unwrap();
    assert_eq!(result.active_lags, 36);
    assert!(result.acf[10..].iter().all(|value| value.to_bits() == 0));
}

#[test]
fn q_needs_a_sample_longer_than_its_lags() {
    for (iq, nw) in [(11, 11), (11, 0), (2, 1)] {
        assert_eq!(
            auto(
                &mut wave(11),
                MeanDiagnosticStage::ForwardResidual,
                11,
                iq,
                nw
            ),
            Err(Failure::AutocorrelationSample { count: nw }.at(Stage::Mean))
        );
    }
    assert!(
        auto(
            &mut wave(11),
            MeanDiagnosticStage::ForwardResidual,
            11,
            11,
            12
        )
        .is_ok()
    );
}

#[test]
fn restoring_the_mean_keeps_the_rounding_of_the_centred_values() {
    let mut values = [0.1, 2.1];
    auto(&mut values, MeanDiagnosticStage::ForwardResidual, 1, 0, 2).unwrap();
    assert_eq!(values[0].to_bits(), 0x3fb9_9999_9999_99a0);
    assert_eq!(values[1].to_bits(), ((2.1_f64 - 1.1) + 1.1).to_bits());
}

#[test]
fn the_differenced_series_is_not_centred() {
    let mut values = [0.1, 2.1];
    let result = auto(&mut values, MeanDiagnosticStage::Differenced, 1, 0, 2).unwrap();
    assert_eq!(bits(&values), bits(&[0.1, 2.1]));
    // Around the mean the lag-one autocorrelation of two values is -0.5.
    assert!(result.acf[0] > 0.0);
}

#[test]
fn a_series_without_variation_is_an_error_at_each_stage() {
    for (stage, values) in [
        (MeanDiagnosticStage::Differenced, [0.0, 0.0]),
        (MeanDiagnosticStage::ForwardResidual, [1.0, 1.0]),
        (MeanDiagnosticStage::SquaredResidual, [4.0, 4.0]),
    ] {
        let mut values = values;
        assert_eq!(
            auto(&mut values, stage, 1, 0, 2),
            Err(Failure::Unusable {
                quantity: stage.label(),
                index: None,
                value: 0.0,
            }
            .at(Stage::Autocorrelation))
        );
    }
}

#[test]
fn the_zero_energy_threshold_is_strict() {
    let mut below = [f64::from_bits(0x3dea_e94f_e549_41bf)];
    assert_eq!(
        auto(&mut below, MeanDiagnosticStage::Differenced, 1, 0, 1),
        Err(Failure::Unusable {
            quantity: "energy of the differenced series",
            index: None,
            value: f64::from_bits(0x3be6_a1c6_e8d9_8028),
        }
        .at(Stage::Autocorrelation))
    );
    let mut above = [f64::from_bits(0x3dea_e94f_e549_41c0)];
    auto(&mut above, MeanDiagnosticStage::Differenced, 1, 0, 1).unwrap();
    assert_eq!(above[0].to_bits(), 0x3dea_e94f_e549_41c0);
}
