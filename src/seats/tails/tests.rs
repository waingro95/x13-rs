use super::*;
use crate::seats::Failure;
use crate::seats::airline::finish_tails;
use crate::seats::airline::model::ResolvedAirline;
use crate::seats::ar1_seasonal_ma::finish_ar1_tails;
use crate::seats::ar1_seasonal_ma::model::{Ar1SeasonalMaInput, import_ar1_seasonal_ma};
use crate::seats::mean::{MeanFamily, mean_correction, prepare_mean_wd};
use crate::seats::residual::resolved_residuals;
use crate::seats::seasonal_ma::finish_seasonal_ma_tails;

fn first_differences(observed: &[f64], period: usize) -> Vec<f64> {
    difference_ordered(observed, period, DifferenceOrder::First).unwrap()
}

fn sawtooth(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 128.0 + ((i * 37 + 11) % 101) as f64 / 16.0)
        .collect()
}

#[test]
fn seasonal_ma_forecast_reads_the_mean_corrected_residuals() {
    // `a` is synthetic, not a CALCFX result: its large mean makes ANALTS's
    // residual-mean correction visible in the first forecast.
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let n = if frequency == Frequency::Monthly {
            36
        } else {
            16
        };
        let s = usize::from(frequency.periods_per_year());
        let original = sawtooth(n);
        let model = crate::seats::seasonal_ma::import_seasonal_ma(
            &crate::seats::seasonal_ma::SeasonalMaInput::new(frequency, 0.4).unwrap(),
        );
        let mut wd = first_differences(&original, s);
        let wm = prepare_mean_wd(&mut wd, RequestedMean::Yes).unwrap();
        let mut a = (0..n - 1)
            .map(|i| {
                if i == n - 2 {
                    1000.0
                } else if i % 2 == 0 {
                    999.0
                } else {
                    1001.0
                }
            })
            .collect::<Vec<_>>();
        let correction = mean_correction(&mut a, RequestedMean::Yes, wm, MeanFamily::SeasonalMa)
            .expect("only the airline refuses a residual-mean correction");
        assert_eq!(a[0..2], [-1.0, 1.0]);
        assert_eq!(a[n - 2].to_bits(), 0);
        assert_eq!(a[n - 1 - s], 1.0);
        let tails = finish_seasonal_ma_tails(
            &model,
            &original,
            &wd,
            ForwardTailState {
                wm: correction.wm,
                requested_mean: correction.statistics.requested,
                residuals: a,
            },
        )
        .unwrap()
        .tails;
        assert_eq!(tails.zaf.to_bits(), correction.wm.to_bits());
        // FCAST's nonzero terms: the seasonal MA residual is +1, not 1001;
        // then BPHIST=(1, zeros, 1, -1) in its order.
        let mut expected = (-0.4_f64).mul_add(1.0, tails.zaf);
        expected = 1.0_f64.mul_add(original[n - 1], expected);
        expected = 1.0_f64.mul_add(original[n - s], expected);
        expected = (-1.0_f64).mul_add(original[n - s - 1], expected);
        assert_eq!(tails.forward_extended[n].to_bits(), expected.to_bits());
        assert_eq!(&tails.forward_extended[..n], &original);
        assert_eq!(
            &tails.backward_extended[..n],
            &original.iter().rev().copied().collect::<Vec<_>>()
        );
    }
}

#[test]
fn ar1_forecast_reads_the_mean_corrected_residuals() {
    // ANALTS 2410–2457 centers `a` before AUTO and FCAST. `a` is synthetic.
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let n = if frequency == Frequency::Monthly {
            36
        } else {
            16
        };
        let s = usize::from(frequency.periods_per_year());
        let original = sawtooth(n);
        let imported =
            import_ar1_seasonal_ma(&Ar1SeasonalMaInput::new(frequency, [0.25, 0.4]).unwrap());
        let mut wd = first_differences(&original, s);
        let wm = prepare_mean_wd(&mut wd, RequestedMean::Yes).unwrap();
        let mut a = (0..n - 2)
            .map(|i| if i % 2 == 0 { 999.0 } else { 1001.0 })
            .collect::<Vec<_>>();
        let correction =
            mean_correction(&mut a, RequestedMean::Yes, wm, MeanFamily::Ar1(0.25)).unwrap();
        assert_eq!(a[0..2], [-1.0, 1.0]);
        let tails = finish_ar1_tails(
            &imported,
            &original,
            &wd,
            ForwardTailState {
                wm: correction.wm,
                requested_mean: correction.statistics.requested,
                residuals: a,
            },
        )
        .unwrap();
        // FCAST's loop: the seasonal MA first, then BPHIST in its order. The
        // corrected seasonal residual is -1, not 999.
        let mut expected = (-0.4_f64).mul_add(-1.0, tails.zaf);
        expected = 1.25_f64.mul_add(original[n - 1], expected);
        expected = (-0.25_f64).mul_add(original[n - 2], expected);
        expected = 1.0_f64.mul_add(original[n - s], expected);
        expected = (-1.25_f64).mul_add(original[n - s - 1], expected);
        expected = 0.25_f64.mul_add(original[n - s - 2], expected);
        assert_eq!(tails.forward_extended[n].to_bits(), expected.to_bits());
    }
}

#[test]
fn ar1_tails_carry_the_ar_history_and_mean_forcing() {
    for frequency in [Frequency::Monthly, Frequency::Quarterly] {
        let n = if frequency == Frequency::Monthly {
            180
        } else {
            80
        };
        let s = usize::from(frequency.periods_per_year());
        let original = sawtooth(n);
        let imported =
            import_ar1_seasonal_ma(&Ar1SeasonalMaInput::new(frequency, [0.25, 0.4]).unwrap());
        for requested in [RequestedMean::No, RequestedMean::Yes] {
            let mut wd = first_differences(&original, s);
            let wm = prepare_mean_wd(&mut wd, requested).unwrap();
            let residuals = crate::seats::ar1_seasonal_ma::residual::ar1_residuals(
                frequency,
                imported.parameters(),
                &wd,
                true,
            )
            .unwrap();
            assert_eq!(residuals.normalized_residuals.len(), n - 2);
            let tails = finish_ar1_tails(
                &imported,
                &original,
                &wd,
                ForwardTailState {
                    wm,
                    requested_mean: requested,
                    residuals: residuals.normalized_residuals,
                },
            )
            .unwrap();
            assert_eq!(tails.horizon, 3 * s);
            assert_eq!(
                tails.zaf.to_bits(),
                (if requested == RequestedMean::Yes {
                    (1.0 - 0.25) * wm
                } else {
                    0.0_f64
                })
                .to_bits()
            );
            assert_eq!(tails.zab, tails.zaf);
            let mut expected = vec![0.0; s + 2];
            expected[0] = 1.25;
            expected[1] = -0.25;
            expected[s - 1] = 1.0;
            expected[s] = -1.25;
            expected[s + 1] = 0.25;
            assert_eq!(tails.bphist, expected);
            assert_eq!(tails.bj_ma.len(), s);
            assert_eq!(tails.forward_extended.len(), n + 3 * s);
            assert_eq!(tails.backward_extended.len(), n + 3 * s);
        }
    }
}

#[test]
fn horizons_of_the_shortest_and_longest_series() {
    for (frequency, minimum, shortest) in [
        (Frequency::Quarterly, 10, Horizons { fh: 8, fhi: 10 }),
        (Frequency::Monthly, 26, Horizons { fh: 24, fhi: 26 }),
    ] {
        let period = usize::from(frequency.periods_per_year());
        assert_eq!(resolve_horizons(frequency, minimum), shortest);
        assert_eq!(
            resolve_horizons(frequency, 780),
            Horizons {
                fh: 3 * period,
                fhi: 3 * period
            }
        );
    }
}

#[test]
fn airline_tails_extend_the_shortest_and_longest_series() {
    for (frequency, minimum) in [(Frequency::Quarterly, 10), (Frequency::Monthly, 26)] {
        let model = ResolvedAirline {
            frequency,
            ma: [0.4, 0.6],
        };
        let period = usize::from(frequency.periods_per_year());
        for n in [minimum, 780] {
            let observed: Vec<_> = (0..n).map(|i| 100.0 + i as f64 * 0.01).collect();
            let wd = first_differences(&observed, period);
            let residuals = resolved_residuals(&model, &wd).unwrap();
            let tails = finish_tails(
                &model,
                &observed,
                &wd,
                ForwardTailState {
                    wm: 0.0,
                    requested_mean: RequestedMean::No,
                    residuals: residuals.normalized_residuals,
                },
            )
            .unwrap()
            .tails;
            assert_eq!(tails.observed_len, n);
            assert_eq!(tails.forward_extended.len(), n + tails.horizon);
        }
    }
}

#[test]
fn backward_wd_reverses_the_centered_wd_without_a_new_mean() {
    // Reversing WD and then removing a newly computed mean would change 71 of
    // the 75 backward values. Build 62 reverses the WD it already centered.
    let mut observed: Vec<_> = (0..80).map(|i| 100.0 + i as f64 * 0.19).collect();
    observed[34] += 1e9;
    observed[0] += 0.3;
    observed[34] += 1e9;
    assert_eq!(observed[0].to_bits(), 0x4059_1333_3333_3333);
    assert_eq!(observed[34].to_bits(), 0x41dd_cd65_1a9d_70a4);

    let frequency = Frequency::Quarterly;
    let model = ResolvedAirline {
        frequency,
        ma: [0.4, 0.6],
    };
    let differenced = first_differences(&observed, 4);
    let mut wd = differenced.clone();
    let wm = prepare_mean_wd(&mut wd, RequestedMean::Yes).unwrap();
    let build_62_mean = f64::from_bits(0x3f70_624d_a740_db4f);
    let reversed_mean = f64::from_bits(0x3f70_624d_d2f1_a8f6);
    assert_eq!(wm.to_bits(), build_62_mean.to_bits());

    let build_62_backward_wd: Vec<_> = differenced
        .iter()
        .map(|value| value - build_62_mean)
        .rev()
        .map(|value| value * 1.0)
        .collect();
    let recomputed_backward_wd: Vec<_> = differenced
        .iter()
        .rev()
        .map(|value| value - reversed_mean)
        .collect();
    assert_eq!(build_62_backward_wd[0].to_bits(), 0xbf70_624d_a740_9b4f);
    assert_eq!(recomputed_backward_wd[0].to_bits(), 0xbf70_624d_d2f1_68f6);
    assert_ne!(
        resolved_residuals(&model, &build_62_backward_wd).unwrap(),
        resolved_residuals(&model, &recomputed_backward_wd).unwrap()
    );

    let forward = resolved_residuals(&model, &wd).unwrap();
    let mut backward_wd = Vec::new();
    finish_ma_only_tails(
        frequency,
        &observed,
        &wd,
        ForwardTailState {
            wm,
            requested_mean: RequestedMean::Yes,
            residuals: forward.normalized_residuals,
        },
        TailInputs::ma_only(
            4,
            airline_bj_ma(4, model.ma),
            resolve_horizons(frequency, observed.len()),
            DifferenceOrder::First,
        ),
        |reversed| {
            backward_wd = reversed.to_vec();
            resolved_residuals(&model, reversed)
        },
    )
    .unwrap();
    assert_eq!(
        backward_wd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        build_62_backward_wd
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn bphist_expands_each_difference_order() {
    for (order, expected) in [
        (DifferenceOrder::Stationary, &[][..]),
        (DifferenceOrder::SeasonalOnly, &[0.0, 0.0, 0.0, 1.0][..]),
        (DifferenceOrder::First, &[1.0, 0.0, 0.0, 1.0, -1.0][..]),
        (
            DifferenceOrder::Second,
            &[2.0, -1.0, 0.0, 1.0, -2.0, 1.0][..],
        ),
        (DifferenceOrder::NonseasonalFirst, &[1.0][..]),
        (DifferenceOrder::NonseasonalSecond, &[2.0, -1.0][..]),
    ] {
        assert_eq!(bphist_ordered(4, &[1.0], order), expected, "{order:?}");
    }
}

#[test]
fn a_failed_backward_solve_is_reported_at_the_tails_stage() {
    let frequency = Frequency::Quarterly;
    let model = ResolvedAirline {
        frequency,
        ma: [0.4, 0.6],
    };
    let observed = sawtooth(40);
    let wd = first_differences(&observed, 4);
    let forward = resolved_residuals(&model, &wd).unwrap();
    let failure = Failure::Unusable {
        quantity: "objective",
        index: Some(1),
        value: f64::INFINITY,
    };
    let result = finish_ma_only_tails(
        frequency,
        &observed,
        &wd,
        ForwardTailState {
            wm: 0.0,
            requested_mean: RequestedMean::No,
            residuals: forward.normalized_residuals,
        },
        TailInputs::ma_only(
            4,
            airline_bj_ma(4, model.ma),
            resolve_horizons(frequency, observed.len()),
            DifferenceOrder::First,
        ),
        |_| Err(failure.clone().at(Stage::Residual)),
    );
    assert_eq!(result.err(), Some(failure.at(Stage::Tails)));
}
