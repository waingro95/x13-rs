use super::*;
use crate::seats::decomposition::root_allocation::RegularArCoefficients;

/// A hand-built statistics value, to place the t-value exactly.
fn statistics(t: f64) -> ResidualMean {
    ResidualMean {
        requested: RequestedMean::Yes,
        mean: 1000.0,
        t,
    }
}

#[test]
fn wd_loses_its_full_sample_mean() {
    let mut wd = [0.1, 2.1];
    let wm = prepare_mean_wd(&mut wd, RequestedMean::Yes).unwrap();
    assert_eq!(wm.to_bits(), 1.1_f64.to_bits());
    assert_eq!(wd[0].to_bits(), (0.1_f64 - 1.1).to_bits());
    assert_eq!(wd[1].to_bits(), (2.1_f64 - 1.1).to_bits());
}

#[test]
fn residual_statistics_follow_dmean_and_dvar() {
    let a = [1.0, 2.0, 3.0];
    let statistics = residual_mean_state(&a, RequestedMean::No).unwrap();
    assert_eq!(statistics.mean.to_bits(), 0x4000_0000_0000_0000);
    assert_eq!(dvar(&a).unwrap().to_bits(), 0x3fe5_5555_5555_5558);
    assert_eq!(statistics.t.to_bits(), 0x4010_f876_ccdf_6cd8);
}

#[test]
fn dvar_fuses_the_square_sum_and_clamps_negative_roundoff() {
    let values = [
        0x4043_9564_6dcf_1d78,
        0xc04d_514a_1ad0_3c98,
        0x4036_8bd2_9f41_9fbc,
        0xc053_e395_a3f2_a8d5,
        0x4054_7a7f_3e2b_9584,
    ]
    .map(f64::from_bits);
    assert_eq!(dvar(&values).unwrap().to_bits(), 0x40ac_ed72_b28c_0ff3);
    let negative_raw = [
        0x426d_1a94_a200_0000,
        0x426d_1a94_a1ff_8000,
        0x426d_1a94_a1ff_c000,
    ]
    .map(f64::from_bits);
    assert_eq!(dvar(&negative_raw).unwrap().to_bits(), 0);
    assert_eq!(
        residual_mean_state(&negative_raw, RequestedMean::No),
        Err(Failure::Unusable {
            quantity: "residual standard deviation",
            index: None,
            value: 0.0,
        }
        .at(Stage::Mean))
    );
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "pins exact results of the Apple silicon math library"
)]
fn residual_rstd_rounds_as_powf_not_sqrt() {
    let value = f64::from_bits(0x2ec8_87ce_ce83_7067);
    assert_eq!(
        residual_rstd(value, 1).unwrap().to_bits(),
        0x375c_0475_d1ec_1765
    );
    assert_eq!(value.sqrt().to_bits(), 0x375c_0475_d1ec_1766);
    assert_eq!(
        residual_rstd(f64::INFINITY, 1),
        Err(Failure::Unusable {
            quantity: "residual standard deviation",
            index: None,
            value: f64::INFINITY,
        }
        .at(Stage::Mean))
    );
}

#[test]
fn zero_residual_variance_is_an_error() {
    for request in [RequestedMean::No, RequestedMean::Yes] {
        assert_eq!(
            residual_mean_state(&[0.0, 0.0, 0.0], request),
            Err(Failure::Unusable {
                quantity: "residual standard deviation",
                index: None,
                value: 0.0,
            }
            .at(Stage::Mean))
        );
    }
}

#[test]
fn guard_refuses_only_t_above_ta() {
    for t in [100.0, -100.0, -1e300] {
        guard_correction(75, t).unwrap();
    }
    assert_eq!(
        guard_correction(75, 100.0_f64.next_up()),
        Err(Failure::UndefinedResidualMeanCorrection {
            residual_count: 75,
            t: 100.0_f64.next_up(),
        }
        .at(Stage::Mean))
    );
}

#[test]
fn undefined_correction_is_an_error_only_with_a_mean() {
    let a = [1.0, 1.0, 1.00001];
    let no = residual_mean_state(&a, RequestedMean::No).unwrap();
    assert!(no.t > RESIDUAL_MEAN_TA);
    assert_eq!(
        residual_mean_state(&a, RequestedMean::Yes),
        Err(Failure::UndefinedResidualMeanCorrection {
            residual_count: 3,
            t: no.t,
        }
        .at(Stage::Mean))
    );
}

#[test]
fn seasonal_ma_correction_centres_the_residuals_and_keeps_wm() {
    // With Q=0, CONV reads the initialized bth1(1)=0, so the correction is
    // defined and adds nothing to WM. It reads no frequency.
    let mut a = [1.0, 1.0, 1.00001];
    let correction = mean_correction(&mut a, RequestedMean::Yes, -0.0, MeanFamily::SeasonalMa)
        .expect("the seasonal-MA correction is defined");
    assert!(correction.statistics.t > 100.0);
    assert_eq!(correction.statistics.mean.to_bits(), 0x3ff0_0003_7ec8_ec26);
    assert_eq!(
        a.map(f64::to_bits),
        [
            0xbecb_f647_6130_0000,
            0xbecb_f647_6130_0000,
            0x3edb_f647_6130_0000,
        ]
    );
    let corrected = residual_statistics(&a, RequestedMean::Yes).unwrap();
    assert_eq!(corrected.mean.to_bits(), 0);
    assert_eq!(dvar(&a).unwrap().to_bits(), 0x3db8_6eff_de16_7a6c);
    assert_eq!(corrected.t.to_bits(), 0);
    assert_eq!(correction.wm.to_bits(), 0);
}

#[test]
fn ar1_correction_centres_the_residuals_and_keeps_wm() {
    let mut a = [999.0, 1001.0];
    let correction =
        mean_correction(&mut a, RequestedMean::Yes, 3.0, MeanFamily::Ar1(0.25)).unwrap();
    assert_eq!(a, [-1.0, 1.0]);
    assert_eq!(correction.statistics.mean, 1000.0);
    assert!(correction.statistics.t > 100.0);
    let corrected = residual_statistics(&a, RequestedMean::Yes).unwrap();
    assert_eq!(corrected.mean, 0.0);
    assert_eq!(dvar(&a).unwrap(), 1.0);
    assert_eq!(corrected.t, 0.0);
    assert_eq!(correction.wm.to_bits(), 3.0_f64.to_bits());
}

#[test]
fn nonseasonal_ar1_correction_adds_the_mean_over_the_ar_sum() {
    // With Q=Bq=0, ANALTS keeps bth1(1)=1, not the zero of Q=0 with Bq=1.
    let mut a = [999.0, 1001.0];
    let correction = mean_correction(
        &mut a,
        RequestedMean::Yes,
        3.0,
        MeanFamily::NonseasonalAr1(0.25),
    )
    .unwrap();
    assert_eq!(a, [-1.0, 1.0]);
    assert_eq!(correction.wm, (1.0_f64 / 0.75).mul_add(1000.0, 3.0));
}

#[test]
fn correction_adds_the_mean_times_the_polynomial_ratio_to_wm() {
    // (family, MA sum, AR sum): CONV's polynomials at one.
    let cases = [
        (MeanFamily::SeasonalMa, 0.0, 1.0),
        (MeanFamily::Ar1(0.25), 0.0, 0.75),
        (MeanFamily::NonseasonalAr1(-0.35), 1.0, 1.35),
        (
            MeanFamily::Ar2(RegularArCoefficients::Two([0.3, -0.2])),
            0.0,
            0.9,
        ),
        (
            MeanFamily::Ar2(RegularArCoefficients::Three([0.3, -0.2, 0.15])),
            0.0,
            0.75,
        ),
        (
            MeanFamily::Regular {
                ar: &[0.3, -0.2],
                seasonal_ar: None,
                seasonal_ma: true,
            },
            0.0,
            0.9,
        ),
        (
            MeanFamily::Regular {
                ar: &[0.3],
                seasonal_ar: None,
                seasonal_ma: false,
            },
            1.0,
            0.7,
        ),
        (
            MeanFamily::Regular {
                ar: &[0.3, -0.2],
                seasonal_ar: Some((12, 0.45)),
                seasonal_ma: true,
            },
            0.0,
            0.9 * 0.55,
        ),
        (
            MeanFamily::Regular {
                ar: &[],
                seasonal_ar: Some((4, -0.35)),
                seasonal_ma: false,
            },
            1.0,
            1.35,
        ),
        (
            MeanFamily::Regular {
                ar: &[0.3, -0.2],
                seasonal_ar: Some((12, 0.45)),
                seasonal_ma: false,
            },
            1.0,
            0.9 * 0.55,
        ),
    ];
    let wm = 0.375;
    for (family, ma_sum, ar_sum) in cases {
        let mut a = [999.0, 1001.0, 1000.25, 999.5, 1000.75];
        let correction = mean_correction(&mut a, RequestedMean::Yes, wm, family).unwrap();
        let mean = correction.statistics.mean;
        assert_eq!(mean, 1000.1, "{family:?}");
        assert!(a.iter().sum::<f64>().abs() < 1e-9, "{family:?}");
        let expected = wm + ma_sum / ar_sum * mean;
        assert!(
            (correction.wm - expected).abs() < 1e-9,
            "{family:?}: {} != {expected}",
            correction.wm
        );
        if ma_sum == 0.0 {
            assert_eq!(correction.wm.to_bits(), wm.to_bits(), "{family:?}");
        }
    }
}

#[test]
fn correction_is_entered_only_above_ta() {
    for family in [MeanFamily::SeasonalMa, MeanFamily::Ar1(0.25)] {
        for (t, enters) in [
            (100.0_f64.next_down(), false),
            (100.0, false),
            (100.0_f64.next_up(), true),
            (-100.0_f64.next_up(), false),
        ] {
            let mut a = [999.0, 1001.0];
            let correction = correct_mean(&mut a, statistics(t), -0.0, family).unwrap();
            assert_eq!(a, if enters { [-1.0, 1.0] } else { [999.0, 1001.0] });
            // The entered correction adds a zero to WM, which clears its sign.
            assert_eq!(
                correction.wm.to_bits(),
                if enters { 0 } else { (-0.0_f64).to_bits() }
            );
            assert_eq!(correction.statistics.t.to_bits(), t.to_bits());
        }
    }
}

#[test]
fn correction_is_skipped_without_a_mean_or_a_large_positive_t() {
    for family in [MeanFamily::SeasonalMa, MeanFamily::Ar1(0.25)] {
        for (values, requested, mean) in [
            ([999.0, 1001.0], RequestedMean::No, 1000.0_f64),
            ([-1001.0, -999.0], RequestedMean::Yes, -1000.0),
            ([-1.0, 1.0], RequestedMean::Yes, 0.0),
        ] {
            let mut a = values;
            let correction = mean_correction(&mut a, requested, -0.0, family).unwrap();
            assert_eq!(a.map(f64::to_bits), values.map(f64::to_bits));
            assert_eq!(correction.statistics.mean.to_bits(), mean.to_bits());
            assert_eq!(correction.wm.to_bits(), (-0.0_f64).to_bits());
        }
    }
}

#[test]
fn a_zero_ar_sum_fails_only_inside_the_correction() {
    // TRANS0 can leave an AR coefficient at one, where the AR sum is zero.
    for (values, requested) in [
        ([999.0, 1001.0], RequestedMean::No),
        ([-1001.0, -999.0], RequestedMean::Yes),
        ([-1.0, 1.0], RequestedMean::Yes),
    ] {
        let mut a = values;
        let correction = mean_correction(&mut a, requested, -0.0, MeanFamily::Ar1(1.0)).unwrap();
        assert_eq!(a.map(f64::to_bits), values.map(f64::to_bits));
        assert_eq!(correction.wm.to_bits(), (-0.0_f64).to_bits());
    }
    let failure = |family: MeanFamily<'_>| {
        let mut a = [999.0, 1001.0];
        mean_correction(&mut a, RequestedMean::Yes, 3.0, family).unwrap_err()
    };
    // A zero MA sum over the zero AR sum is 0/0; a unit MA sum is 1/0.
    for family in [
        MeanFamily::Ar1(1.0),
        MeanFamily::Ar2(RegularArCoefficients::Two([0.5, 0.5])),
    ] {
        assert!(matches!(
            failure(family),
            Error {
                stage: Stage::Mean,
                failure: Failure::Unusable {
                    quantity: "mean correction ratio",
                    index: None,
                    value,
                },
            } if value.is_nan()
        ));
    }
    assert_eq!(
        failure(MeanFamily::Regular {
            ar: &[1.0],
            seasonal_ar: None,
            seasonal_ma: false,
        }),
        Failure::Unusable {
            quantity: "mean correction ratio",
            index: None,
            value: f64::INFINITY,
        }
        .at(Stage::Mean)
    );
}
