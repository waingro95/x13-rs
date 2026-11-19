use crate::diagnostics::{
    SpectralNotApplicable as NotApplicable, SpectralPeakKind as Kind, SpectralPeaks,
    SpectralReason as Reason, SpectralSeries as Series, SpectralStage as Stage,
    SpectralTradingDayPeaks as TradingDay, Unavailable, UnavailableDetail, UnavailableKind,
};
use crate::time::TimeAxis;
use crate::x11;

/// Why a spectrum that would be examined is undefined.
fn reason(spectrum: Result<&SpectralPeaks, &Unavailable>) -> Reason {
    match spectrum {
        Err(Unavailable(UnavailableDetail::Spectrum { reason, .. })) => *reason,
        other => panic!("not an undefined spectrum: {other:?}"),
    }
}

#[test]
fn a_spectrum_without_peaks_says_why() {
    let not_examined = super::not_examined(
        Series::ModifiedIrregular,
        NotApplicable::TrendOnlyDecomposition,
    );
    assert_eq!(not_examined.kind(), UnavailableKind::NotApplicable);
    assert_eq!(
        not_examined.to_string(),
        "the spectrum of the modified irregular is not examined: trend-only decomposition has no \
         seasonally adjusted series or irregular to examine"
    );
    let short = super::undefined(
        Series::AdjustedOriginal,
        Reason::InsufficientInput {
            actual: 2,
            minimum: 3,
        },
        None,
    );
    assert_eq!(short.kind(), UnavailableKind::InsufficientData);
    assert_eq!(
        short.to_string(),
        "the spectrum of the adjusted original series is unavailable: 2 values, fewer than the 3 \
         required"
    );
    for reason in [
        Reason::ZeroCovariance,
        Reason::PartFPrerequisiteUnavailable,
        Reason::NonFiniteArithmetic {
            stage: Stage::Differencing,
        },
    ] {
        assert_eq!(
            super::undefined(Series::ModifiedSeasonallyAdjusted, reason, None).kind(),
            UnavailableKind::Numerical,
            "{reason:?}"
        );
    }
}

#[test]
fn ispeak_strict_boundaries_and_monthly_final_slot() {
    let regular = vec![0.0; 61];
    let frequencies: Vec<_> = (0..67).map(|i| i as f64 / 120.0).collect();
    let mut expanded = vec![0.0; 67];
    expanded[10] = 10.0;
    expanded[53] = 100.0; // excluded monthly final seasonal slot
    assert_eq!(
        super::peak_counts(&regular, &expanded, &frequencies, false),
        (1, 0)
    );
    expanded[9] = 10.0; // strict lower comparison, not <=
    assert_eq!(
        super::peak_counts(&regular, &expanded, &frequencies, false),
        (0, 0)
    );
    expanded[9] = 0.0;
    expanded[10] = 0.0; // strict median comparison
    assert_eq!(
        super::peak_counts(&regular, &expanded, &frequencies, false),
        (0, 0)
    );
}

#[test]
fn seasonal_local_exclusion_is_asymmetric() {
    let regular = vec![0.0; 61];
    let frequencies: Vec<_> = (0..67).map(|i| i as f64 / 120.0).collect();
    let mut expanded = vec![0.0; 67];
    expanded[40] = 10.0;
    expanded[41] = 20.0; // above seasonal candidate is not examined locally
    expanded[43] = 10.0;
    expanded[44] = 20.0; // above TD candidate is examined locally
    assert_eq!(
        super::peak_counts(&regular, &expanded, &frequencies, true),
        (1, 0)
    );
}

#[test]
fn log_then_two_backward_differences() {
    let axis = TimeAxis::monthly(2000, 3).unwrap();
    let values = [1.1_f64, 3.7, 9.2, 22.3];
    let (sample, actual) =
        super::prepare(&values, axis.span(values.len()).unwrap(), 2, true, false).unwrap();
    let logged = values.map(f64::ln);
    let once = [
        logged[1] - logged[0],
        logged[2] - logged[1],
        logged[3] - logged[2],
    ];
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [once[1] - once[0], once[2] - once[1]].map(f64::to_bits)
    );
    assert_eq!(sample.first(), axis.period_at(2).unwrap());
    assert_eq!(sample.count(), 2);
    assert!(sample.logarithmic_input());
}

#[test]
fn trading_day_peaks_need_more_than_sixty_differenced_values() {
    let values: Vec<_> = (0..61).map(|i| (i * 17 % 11) as f64).collect();
    let td = super::estimate_peaks(&values, 60).unwrap().trading_day;
    assert_eq!(td, TradingDay::NotTestedInsufficientSample { count: 60 });
    assert!(matches!(
        super::estimate_peaks(&values, 61).unwrap().trading_day,
        TradingDay::Tested { .. }
    ));
}

#[test]
fn warnings_for_peaks_and_for_an_original_without_seasonal_peaks() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let sample = super::prepare(&[1.0, 2.0, 3.0], axis.span(3).unwrap(), 1, false, false)
        .unwrap()
        .0;
    let available = |seasonal_count, td| {
        Ok(SpectralPeaks {
            sample: sample.clone(),
            ar_order: 1,
            seasonal_count,
            trading_day: TradingDay::Tested { count: td },
        })
    };
    let unavailable = Err(super::undefined(
        Series::AdjustedOriginal,
        Reason::ZeroCovariance,
        Some(sample.clone()),
    ));
    let mut diagnostics = crate::x11::SpectralDiagnostics {
        adjusted_original: available(1, 0),
        modified_seasonally_adjusted: available(0, 0),
        modified_irregular: available(0, 0),
    };
    // A seasonal peak in the original is expected unless trend-only.
    assert!(super::warnings(&diagnostics, false).is_empty());
    assert_eq!(
        super::warnings(&diagnostics, true),
        [crate::Warning::X11SpectralPeaks {
            kind: Kind::Seasonal
        }]
    );
    diagnostics.adjusted_original = unavailable;
    diagnostics.modified_irregular = available(1, 1);
    assert_eq!(
        super::warnings(&diagnostics, false),
        [crate::Warning::X11SpectralPeaks { kind: Kind::Both }]
    );
    diagnostics.adjusted_original = available(0, 0);
    assert_eq!(
        super::warnings(&diagnostics, false),
        [
            crate::Warning::X11SpectralPeaks { kind: Kind::Both },
            crate::Warning::X11NoOriginalSeasonalSpectralPeaks
        ]
    );
}

#[test]
fn prepare_and_estimate_report_unusable_input() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let err = super::prepare(&[1.0, 0.0, 2.0], axis.span(3).unwrap(), 1, true, false).unwrap_err();
    assert_eq!(
        err.0,
        Reason::NonPositiveLogInput {
            period: axis.period_at(1).unwrap()
        }
    );
    assert!(err.1.is_some());
    assert_eq!(
        super::prepare(
            &[1.0, f64::NAN, 2.0],
            axis.span(3).unwrap(),
            1,
            false,
            false
        )
        .unwrap_err()
        .0,
        Reason::NonFiniteInput {
            period: axis.period_at(1).unwrap()
        }
    );
    assert_eq!(
        super::prepare(
            &[f64::MAX, -f64::MAX, 1.0],
            axis.span(3).unwrap(),
            1,
            false,
            false
        )
        .unwrap_err()
        .0,
        Reason::NonFiniteArithmetic {
            stage: Stage::Differencing
        }
    );
    let empty = super::prepare(&[1.0, 2.0], axis.span(2).unwrap(), 2, false, false).unwrap_err();
    assert_eq!(
        empty,
        (
            Reason::InsufficientInput {
                actual: 2,
                minimum: 3
            },
            None
        )
    );
    assert_eq!(
        super::estimate_peaks(&[1.0; 96], 95).unwrap_err(),
        Reason::ZeroCovariance
    );
    assert_eq!(
        super::estimate_peaks(&[f64::MAX; 96], 95).unwrap_err(),
        Reason::NonFiniteArithmetic {
            stage: Stage::MeanDeletion
        }
    );
}

#[test]
fn original_operand_recombines_modified_and_extreme() {
    assert_eq!(
        super::original_operand(x11::Mode::Additive, 3.0, 7.0, 0.0, 0.0),
        Some(10.0)
    );
    assert_eq!(
        super::original_operand(x11::Mode::Multiplicative, 3.0, 7.0, 0.0, 0.0),
        Some(21.0)
    );
    // A modified value one ULP above 3 shows in the product.
    assert_eq!(
        super::original_operand(
            x11::Mode::Multiplicative,
            f64::from_bits(3.0f64.to_bits() + 1),
            7.0,
            0.0,
            0.0
        )
        .unwrap()
        .to_bits(),
        21.0f64.to_bits() + 1
    );
}

#[test]
fn log_additive_original_exponentiates_each_term() {
    assert_eq!(
        super::original_operand(x11::Mode::LogAdditive, 0.3, 0.0, 0.9, 0.8)
            .unwrap()
            .to_bits(),
        0x3ff38add9e58ac8e
    );
    assert_ne!((0.3f64 + 0.8 - 0.9).exp().to_bits(), 0x3ff38add9e58ac8e);
}

#[test]
fn pseudo_additive_original_from_final_components() {
    assert_eq!(
        super::pseudo_original_operand(1.0000000000000002, 1.4, 0.8, false).to_bits(),
        0x3ff3333333333334
    );
    assert_eq!(super::pseudo_original_operand(3.0, 100.0, 7.0, true), 21.0);
}

#[test]
fn trend_only_pseudo_additive_original_uses_the_part_f_trend() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let values: Vec<_> = (0..96).map(|i| 100.0 + i as f64 * 0.2).collect();
    let mut c = crate::x11::adjust(
        &values,
        axis,
        x11::Mode::PseudoAdditive,
        crate::x11::Options::default().with_decomposition(crate::x11::Decomposition::TrendOnly),
    )
    .unwrap();
    // Part F's logarithm and antilogarithm do not round back to the trend.
    let expected: Vec<_> = c
        .trend_cycle
        .iter()
        .zip(&c.irregular)
        .map(|(&t, &i)| super::pseudo_original_operand(t.ln().exp(), 1.0, i, true))
        .collect();
    assert!(
        c.trend_cycle
            .iter()
            .any(|t| t.to_bits() != t.ln().exp().to_bits())
    );
    let _ = crate::x11::quality::complete(
        &mut c,
        &values,
        axis.span(values.len()).unwrap(),
        Default::default(),
    );
    assert_eq!(
        c.spectral_workspace.level_shift_round_trip,
        super::LevelShiftRoundTrip::NotReached
    );
    super::complete(
        &mut c,
        axis.span(values.len()).unwrap(),
        super::SpectralCompletion {
            difference_order: 1,
            same_original: true,
            level_shift: None,
            remove_final_level_shift: false,
        },
    );
    assert_eq!(c.spectral_workspace.adjusted_original.unwrap(), expected);
}

#[test]
fn pseudo_additive_original_needs_part_f_to_reach_its_trend() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let values: Vec<_> = (0..180)
        .map(|i| 100.0 + i as f64 * 0.2 + (i * 17 % 11) as f64)
        .collect();
    for fails_before_trend in [true, false] {
        let mut c =
            crate::x11::adjust(&values, axis, x11::Mode::PseudoAdditive, Default::default())
                .unwrap();
        let modified = crate::x11::seasonality::modified_adjusted(
            &c.seasonally_adjusted,
            &c.trend_cycle,
            &c.intermediate_extreme_weights,
        )
        .unwrap();
        crate::x11::seasonality::complete_modified(&mut c, modified, &values, axis);
        let work = &mut c.quality_workspace;
        if fails_before_trend {
            work.base_irregular[7] = f64::INFINITY;
        } else {
            work.moving_seasonality_ratio = Err(
                crate::diagnostics::X11QualityUnavailableReason::NonFiniteArithmetic {
                    stage: crate::diagnostics::X11QualityStage::FinalMovingSeasonalityRatio,
                },
            );
        }
        let quality = crate::x11::quality::complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        assert!(quality.is_err());
        assert_eq!(
            c.spectral_workspace.part_f_trend.is_some(),
            !fails_before_trend
        );
        let diagnostics = super::complete(
            &mut c,
            axis.span(values.len()).unwrap(),
            super::SpectralCompletion {
                difference_order: 1,
                same_original: false,
                level_shift: None,
                remove_final_level_shift: false,
            },
        );
        if fails_before_trend {
            assert_eq!(
                diagnostics.adjusted_original(),
                Err(&super::undefined(
                    Series::AdjustedOriginal,
                    Reason::PartFPrerequisiteUnavailable,
                    None
                ))
            );
        } else {
            assert!(diagnostics.adjusted_original().is_ok());
        }
        assert!(diagnostics.modified_seasonally_adjusted().is_ok());
        assert!(diagnostics.modified_irregular().is_ok());
        assert_eq!(
            c.spectral_workspace.adjusted_original.is_some(),
            !fails_before_trend
        );
    }
}

fn adjusted_components() -> crate::x11::Components {
    let values: Vec<_> = (0..96)
        .map(|i| 100.0 + i as f64 * 0.2 + (i * 17 % 11) as f64)
        .collect();
    crate::x11::adjust(
        &values,
        TimeAxis::monthly(2000, 1).unwrap(),
        x11::Mode::Additive,
        Default::default(),
    )
    .unwrap()
}

#[test]
fn retained_series_are_not_replaced_by_the_final_irregular() {
    let mut c = adjusted_components();
    let modified_adjusted = vec![13.0; 96];
    super::retain_modified(&mut c, &modified_adjusted);
    c.intermediate_extreme_weights.fill(0.5);
    c.intermediate_extreme_weights[0] = 0.0;
    super::retain_irregular(&mut c, &vec![7.0; 96]);
    c.irregular.fill(900.0);
    super::complete(
        &mut c,
        TimeAxis::monthly(2000, 1).unwrap().span(96).unwrap(),
        super::SpectralCompletion {
            difference_order: 1,
            same_original: false,
            level_shift: None,
            remove_final_level_shift: false,
        },
    );
    let work = c.spectral_workspace;
    assert_eq!(work.modified_adjusted.unwrap(), modified_adjusted);
    let modified_irregular = work.modified_irregular.unwrap();
    assert_eq!(modified_irregular[0], 0.0);
    assert_eq!(&modified_irregular[1..], &[7.0; 95]);
}

#[test]
fn level_shift_round_trip_applies_only_when_part_f_reached_it() {
    use super::LevelShiftRoundTrip as RoundTrip;
    // Even a zero level shift shows the round trip: (-0.0 - 0.0) + 0.0 is
    // +0.0, and without it -0.0 - 0.0 stays -0.0.
    for (round_trip, expected) in [
        (RoundTrip::Reached, 0.0f64.to_bits()),
        (RoundTrip::NotReached, (-0.0f64).to_bits()),
    ] {
        let mut c = adjusted_components();
        let work = &mut c.spectral_workspace;
        work.modified_adjusted = Some(vec![-0.0; 96]);
        work.level_shift_round_trip = round_trip;
        super::complete(
            &mut c,
            TimeAxis::monthly(2000, 1).unwrap().span(96).unwrap(),
            super::SpectralCompletion {
                difference_order: 1,
                same_original: false,
                level_shift: Some(vec![0.0; 96]),
                remove_final_level_shift: false,
            },
        );
        assert!(
            c.spectral_workspace
                .modified_adjusted
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == expected)
        );
    }
}

#[test]
fn final_level_shift_removal_does_not_depend_on_part_f() {
    use super::LevelShiftRoundTrip as RoundTrip;

    // With `Finls`, `x11pt4` skips the level-shift round trip, and `spcdrv`
    // still takes the level shift out of the series once.
    let modified_adjusted: Vec<_> = (0..96)
        .map(|i| {
            100.0
                + 10.0 * (i as f64 * std::f64::consts::TAU / 12.0).sin()
                + (i * 17 % 11) as f64 * 0.1
        })
        .collect();
    let factors: Vec<_> = (0..96).map(|i| if i < 48 { 0.0 } else { 3.0 }).collect();
    let expected: Vec<_> = modified_adjusted
        .iter()
        .zip(&factors)
        .map(|(&v, &f)| v - f)
        .collect();
    // Only the modified seasonally adjusted series has a spectrum here: there
    // is no original, and the irregular is constant.
    let complete = |round_trip, remove_final_level_shift| {
        let mut c = adjusted_components();
        c.warnings.clear();
        let work = &mut c.spectral_workspace;
        work.adjusted_original = None;
        work.modified_adjusted = Some(modified_adjusted.clone());
        work.modified_irregular = Some(vec![0.0; 96]);
        work.level_shift_round_trip = round_trip;
        let diagnostics = super::complete(
            &mut c,
            TimeAxis::monthly(2000, 1).unwrap().span(96).unwrap(),
            super::SpectralCompletion {
                difference_order: 1,
                same_original: false,
                level_shift: Some(factors.clone()),
                remove_final_level_shift,
            },
        );
        (c, diagnostics)
    };
    let (reached, reached_diagnostics) = complete(RoundTrip::Reached, true);
    assert!(matches!(
        reached_diagnostics.modified_seasonally_adjusted(),
        Ok(peaks) if peaks.seasonal_count() > 0
    ));
    assert_eq!(
        reason(reached_diagnostics.adjusted_original()),
        Reason::PartFPrerequisiteUnavailable
    );
    assert_eq!(
        reason(reached_diagnostics.modified_irregular()),
        Reason::ZeroCovariance
    );
    assert_eq!(
        reached.spectral_workspace.modified_adjusted.as_ref(),
        Some(&expected)
    );
    assert!(matches!(
        reached.warnings.as_slice(),
        [crate::Warning::X11SpectralPeaks {
            kind: Kind::Seasonal | Kind::Both
        }]
    ));

    let (failed, failed_diagnostics) = complete(RoundTrip::Unknown, true);
    assert_eq!(
        failed.spectral_workspace.modified_adjusted.as_ref(),
        Some(&expected)
    );
    assert_eq!(failed_diagnostics, reached_diagnostics);
    assert_eq!(failed.warnings, reached.warnings);

    // Without `Finls`, a Part F that failed before the round trip leaves the
    // last bit of the series unknown, so it has no spectrum.
    let (kept, kept_diagnostics) = complete(RoundTrip::Unknown, false);
    assert_eq!(
        reason(kept_diagnostics.modified_seasonally_adjusted()),
        Reason::PartFPrerequisiteUnavailable
    );
    assert!(kept.warnings.is_empty());
}
