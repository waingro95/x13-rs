use crate::api::*;
use crate::error::{InvalidSpec, SpecProblem};
use crate::regarima::{ManualOutlier, OutlierTypes, RegressionVariable};

const GAP: usize = 17;

fn axis() -> TimeAxis {
    TimeAxis::monthly(2000, 1).unwrap()
}

fn airline() -> RegArimaSpec {
    let order = ArimaOrder::new(0, 1, 1).unwrap();
    RegArimaSpec::new(ArimaModel::new(order, order).unwrap())
}

/// A positive seasonal series with deterministic noise, so differencing
/// leaves something to estimate.
fn noisy_seasonal(count: usize) -> Vec<f64> {
    let mut state = 12_345_u64;
    (0..count)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let angle = 2.0 * std::f64::consts::PI * (index % 12) as f64 / 12.0;
            100.0 + 0.2 * index as f64 + 8.0 * angle.sin() + 3.0 * noise
        })
        .collect()
}

/// A gap in a constant series takes the constant, so the whole result,
/// refusals included, is that of the series without it.
#[test]
fn a_constant_series_with_gaps_matches_the_series_without_them() {
    let filler = crate::engine::missing::PLACEHOLDER;
    for constant in [5.0, -5.0, filler - 1.0, filler, filler + 1.0] {
        let flat = vec![constant; 96];
        let mut gapped = flat.clone();
        for gap in [1, GAP, 95] {
            gapped[gap] = f64::NAN;
        }
        for spec in [
            Spec::x11_regarima(
                airline(),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
            Spec::x11_regarima(
                airline(),
                x11::Options::new().with_mode(x11::Mode::Multiplicative),
            ),
            Spec::x11_regarima(
                airline(),
                x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
            ),
            Spec::x11_regarima(
                RegArimaSpec::automatic_model(),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
            Spec::x11_regarima(RegArimaSpec::automatic_model(), x11::Options::new()),
            Spec::seats(airline()),
            Spec::seats(RegArimaSpec::automatic_model()),
        ] {
            // `Adjustment` has no `PartialEq`; its `Debug` text covers every field.
            assert_eq!(
                format!("{:?}", adjust(Series::new(axis().start(), &gapped), &spec)),
                format!("{:?}", adjust(Series::new(axis().start(), &flat), &spec)),
                "{constant} {spec:?}"
            );
        }
    }

    let mut values = vec![5.0; 96];
    values[GAP] = f64::NAN;
    let spec = Spec::x11_regarima(
        airline(),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let adjustment = adjust(Series::new(axis().start(), &values), &spec).unwrap();
    assert!(adjustment.model().is_none());
    let adjusted = adjustment.seasonally_adjusted().values();
    assert!(adjusted.iter().all(|value| (value - 5.0).abs() < 1e-12));
}

/// Filling leaves only zeros, which are refused without gaps too.
#[test]
fn zeros_with_a_gap_are_all_zero_input() {
    let mut values = vec![0.0; 96];
    values[GAP] = f64::NAN;
    for spec in [
        Spec::x11_regarima(
            airline(),
            x11::Options::new().with_mode(x11::Mode::Additive),
        ),
        Spec::x11_regarima(
            airline(),
            x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
        ),
        Spec::x11_regarima(
            airline().with_transform(Transform::Log),
            x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
        ),
        Spec::seats(airline()),
    ] {
        assert_eq!(
            adjust(Series::new(axis().start(), &values), &spec).err(),
            Some(Error::AllZeroInput),
            "{spec:?}"
        );
    }
}

/// Build 62's refusal depends on the gap, not on what fills it.
#[test]
fn a_filled_gap_still_counts_as_a_pseudo_additive_regressor() {
    let mut values = vec![5.0; 96];
    values[GAP] = f64::NAN;
    let spec = Spec::x11_regarima(
        airline().with_constant(true),
        x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
    );
    assert_eq!(
        adjust(Series::new(axis().start(), &values), &spec).err(),
        Some(Error::InvalidSpec(InvalidSpec(
            SpecProblem::IncompatiblePseudoAdditivePreadjustment
        )))
    );
}

fn gapped() -> Vec<f64> {
    let mut values = noisy_seasonal(96);
    values[GAP] = f64::NAN;
    values
}

/// Gaps add missing-value regressors, which Build 62 lets through
/// pseudo-additive X-11 on the log scale.
#[test]
fn pseudo_additive_estimates_gaps_with_logs() {
    let log = airline().with_transform(Transform::Log);
    let values = gapped();
    for spec in [
        Spec::x11_regarima(
            log.clone(),
            x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
        ),
        Spec::x11_regarima(
            log.clone(),
            x11::Options::new()
                .with_mode(x11::Mode::PseudoAdditive)
                .with_final_outliers(x11::FinalOutliers::Remove(
                    OutlierTypes::LEVEL_SHIFT_AND_TEMPORARY_CHANGE,
                )),
        ),
        Spec::x11_regarima(
            RegArimaSpec::automatic_model().with_transform(Transform::Log),
            x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
        ),
    ] {
        let adjustment = adjust(Series::new(axis().start(), &values), &spec);
        assert!(adjustment.is_ok(), "{spec:?}: {:?}", adjustment.err());
    }
}

/// A gap leaves no effect to restore, so the final series keeps
/// pseudo-additive arithmetic rather than the log transform's.
#[test]
fn pseudo_additive_adjusted_series_with_a_gap_stays_pseudo_additive() {
    let values = gapped();
    let spec = Spec::x11_regarima(
        airline().with_transform(Transform::Log),
        x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
    );
    let adjustment = adjust(Series::new(axis().start(), &values), &spec).unwrap();
    let adjusted = adjustment.seasonally_adjusted();
    let factors = adjustment.seasonal().unwrap();
    let trend = adjustment.trend_cycle().unwrap();
    assert!(adjusted.values()[GAP].is_finite());
    for (index, &original) in values.iter().enumerate() {
        if index != GAP {
            let expected = original - trend.values()[index] * (factors.values()[index] - 1.0);
            let adjusted = adjusted.values()[index];
            assert!((adjusted - expected).abs() < 1e-9, "{index}: {adjusted}");
        }
    }
}

/// arima.f's mode gate tests Muladd, which pseudo-additive shares with
/// multiplicative, so the identity transform refuses the regressors.
#[test]
fn pseudo_additive_gaps_need_logs() {
    let spec = Spec::x11_regarima(
        airline(),
        x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
    );
    assert_eq!(
        adjust(Series::new(axis().start(), &gapped()), &spec).err(),
        Some(Error::InvalidSpec(InvalidSpec(
            SpecProblem::IncompatibleRegressionAdjustmentMode {
                transform: Transform::None,
                mode: x11::Mode::PseudoAdditive,
            }
        )))
    );
}

/// editor.f treats a missing-value regressor as an additive outlier, so a
/// constant, or final removal of additive outliers, refuses the gap.
#[test]
fn pseudo_additive_gaps_refuse_a_constant_and_additive_removal() {
    let log = airline().with_transform(Transform::Log);
    let values = gapped();
    for spec in [
        Spec::x11_regarima(
            log.clone().with_constant(true),
            x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
        ),
        Spec::x11_regarima(
            log.clone(),
            x11::Options::new()
                .with_mode(x11::Mode::PseudoAdditive)
                .with_final_outliers(x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE)),
        ),
        Spec::x11_regarima(
            log.clone().with_constant(true),
            x11::Options::new()
                .with_mode(x11::Mode::PseudoAdditive)
                .with_final_outliers(x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT)),
        ),
    ] {
        assert_eq!(
            adjust(Series::new(axis().start(), &values), &spec).err(),
            Some(Error::InvalidSpec(InvalidSpec(
                SpecProblem::IncompatiblePseudoAdditivePreadjustment
            ))),
            "{spec:?}"
        );
    }
}

/// Without a gap, the constant-model rule for final removal still holds.
#[test]
fn pseudo_additive_final_removal_without_gaps_needs_a_constant() {
    let spec = Spec::x11_regarima(
        airline().with_transform(Transform::Log),
        x11::Options::new()
            .with_mode(x11::Mode::PseudoAdditive)
            .with_final_outliers(x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT)),
    );
    assert!(matches!(
        adjust(Series::new(axis().start(), &noisy_seasonal(96)), &spec),
        Err(Error::InvalidSpec(InvalidSpec(
            SpecProblem::IncompatiblePseudoAdditiveFinalOutliers { .. }
        )))
    ));
}

/// Direct X-11 refuses every gap, and a constant series needs every season.
#[test]
fn a_constant_series_with_gaps_still_needs_a_model_and_every_season() {
    let mut values = vec![5.0; 96];
    values[GAP] = f64::NAN;
    assert_eq!(
        adjust(
            Series::new(axis().start(), &values),
            &Spec::x11(x11::Options::new().with_mode(x11::Mode::Additive))
        )
        .err(),
        Some(Error::MissingValue { index: GAP })
    );
    for index in (GAP..values.len()).step_by(12) {
        values[index] = f64::NAN;
    }
    for index in (5..GAP).step_by(12) {
        values[index] = f64::NAN;
    }
    assert!(matches!(
        adjust(
            Series::new(axis().start(), &values),
            &Spec::x11_regarima(
                airline(),
                x11::Options::new().with_mode(x11::Mode::Additive)
            )
        ),
        Err(Error::MissingSeason { .. })
    ));
}

/// `Same` is unset here, so the model estimates the leading gap.
#[test]
fn a_constant_series_with_a_leading_gap_is_estimated() {
    let mut values = vec![5.0; 96];
    values[0] = f64::NAN;
    let spec = Spec::x11_regarima(
        airline(),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    assert!(adjust(Series::new(axis().start(), &values), &spec).is_ok());
}

/// Filling leaves these values constant, but `Same` stays unset, so the
/// model runs: differencing annihilates the series, as in Build 62, and an
/// undifferenced model fits.
#[test]
fn a_leading_gap_before_filler_values_runs_the_model() {
    let mut values = vec![crate::engine::missing::PLACEHOLDER; 96];
    values[0] = f64::NAN;
    let series = Series::new(axis().start(), &values);
    for spec in [
        Spec::x11_regarima(
            airline(),
            x11::Options::new().with_mode(x11::Mode::Additive),
        ),
        Spec::x11_regarima(RegArimaSpec::automatic_model(), x11::Options::new()),
        Spec::seats(RegArimaSpec::automatic_model()),
    ] {
        assert_eq!(
            adjust(series, &spec).err(),
            Some(Error::DegenerateSeries),
            "{spec:?}"
        );
    }

    let model = ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let regarima = RegArimaSpec::new(model).with_constant(true);
    let spec = Spec::x11_regarima(regarima, x11::Options::new().with_mode(x11::Mode::Additive));
    assert!(adjust(series, &spec).unwrap().model().is_some());
}

/// SEATS runs the model after a leading gap too. The target is Build 62's
/// saved SA at the gap, which the model estimates.
#[test]
fn seats_fits_a_leading_gap_before_filler_values() {
    let mut values = vec![crate::engine::missing::PLACEHOLDER; 96];
    values[0] = f64::NAN;
    let model = ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = Spec::seats(RegArimaSpec::new(model));
    let adjustment = adjust(Series::new(axis().start(), &values), &spec).unwrap();
    let first = adjustment.seasonally_adjusted().values()[0];
    assert!((first - 999_999_993.335_383).abs() < 6e-6, "{first}");
}

/// `fcstxy.f` forecasts without `armafl.f`'s root check, so an estimate
/// that stops just outside the invertible region is kept. The targets are
/// Build 62's for three leading gaps.
#[test]
fn a_noninvertible_estimate_still_forecasts() {
    let mut values = vec![5.0; 96];
    values[..3].fill(f64::NAN);
    let spec = Spec::x11_regarima(
        airline(),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let adjustment = adjust(Series::new(axis().start(), &values), &spec).unwrap();

    let estimate = adjustment.model().unwrap().parameters()[0].estimate();
    assert!((estimate + 1.000_112_848_180_2).abs() < 1e-9, "{estimate}");
    let forecast = adjustment.forecasts().unwrap().series().values()[0];
    assert!((forecast - 5.000_000_239_917_04).abs() < 1e-9, "{forecast}");
}

/// `editor.f` tests the series span, which is the whole input here, so a
/// model span moves neither the first value nor the extremes.
#[test]
fn a_model_span_does_not_change_the_constant_rule() {
    let span = PeriodSpan::new(axis().period_at(12).unwrap(), axis().period_at(95).unwrap());
    let spec = Spec::x11_regarima(
        airline().with_model_span(span),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let run = |gap: usize, head: fn(usize) -> f64| {
        let mut values: Vec<f64> = (0..96).map(head).collect();
        values[gap] = f64::NAN;
        adjust(Series::new(axis().start(), &values), &spec)
    };

    // The model span starts on the gap, but the series' first value is observed.
    assert!(run(12, |_| 5.0).unwrap().model().is_none());
    // A gap outside the model span is refused whatever fills it.
    assert!(matches!(
        run(5, |_| 5.0),
        Err(Error::MissingOutsideModelSpan { .. })
    ));
    // A first value missing is a leading gap, whatever the model span says.
    assert!(matches!(
        run(0, |_| 5.0),
        Err(Error::MissingOutsideModelSpan { .. })
    ));
    // Constant inside the model span only: the series is not constant, and
    // the target is Build 62's first seasonally adjusted value.
    let varied_head = |index| {
        if index < 12 {
            7.0 + 0.37 * index as f64
        } else {
            5.0
        }
    };
    let adjustment = run(40, varied_head).unwrap();
    let adjusted = adjustment.seasonally_adjusted().values()[0];
    assert!((adjusted - 7.152_563_624_224_09).abs() < 1e-8, "{adjusted}");
}

fn regarima_specs() -> [Spec; 2] {
    [
        Spec::x11_regarima(
            airline(),
            x11::Options::new().with_mode(x11::Mode::Additive),
        ),
        Spec::seats(airline()),
    ]
}

#[test]
fn nan_is_estimated_under_regarima() {
    let mut values = noisy_seasonal(120);
    values[GAP] = f64::NAN;
    let gap = RegressionVariable::MissingValue {
        period: axis().period_at(GAP).unwrap(),
    };
    for spec in regarima_specs() {
        let adjustment = adjust(Series::new(axis().start(), &values), &spec).unwrap();
        assert_eq!(adjustment.seasonally_adjusted().len(), values.len());
        let model = adjustment.model().unwrap();
        assert!(
            model
                .regression_coefficients()
                .iter()
                .any(|estimate| estimate.variable() == gap),
            "{spec:?}"
        );
    }
}

#[test]
fn manual_outliers_outside_the_model_span_are_dropped_with_warnings() {
    // editor.f drops them with a warning each.
    let period = |index| axis().period_at(index).unwrap();
    let outliers = [
        ManualOutlier::new(OutlierType::Additive, period(5)),
        ManualOutlier::new(OutlierType::LevelShift, period(24)),
        ManualOutlier::new(OutlierType::LevelShift, period(60)),
    ];
    let regarima = airline()
        .with_manual_outliers(&outliers)
        .with_model_span(PeriodSpan::new(period(24), period(119)));
    let expected = [
        Warning::ManualOutlierOutsideModelSpan {
            kind: OutlierType::Additive,
            period: period(5),
        },
        Warning::ManualOutlierOutsideModelSpan {
            kind: OutlierType::LevelShift,
            period: period(24),
        },
    ];
    let kept = RegressionVariable::Outlier {
        kind: OutlierType::LevelShift,
        period: period(60),
    };
    for spec in [
        Spec::x11_regarima(
            regarima.clone(),
            x11::Options::new().with_mode(x11::Mode::Additive),
        ),
        Spec::seats(regarima),
    ] {
        let adjustment = adjust(Series::new(axis().start(), &noisy_seasonal(120)), &spec).unwrap();
        assert!(adjustment.warnings().starts_with(&expected), "{spec:?}");
        let variables: Vec<_> = adjustment
            .model()
            .unwrap()
            .regression_coefficients()
            .iter()
            .map(|estimate| estimate.variable())
            .filter(|variable| matches!(variable, RegressionVariable::Outlier { .. }))
            .collect();
        assert_eq!(variables, [kept], "{spec:?}");
    }
}

#[test]
fn nan_is_a_missing_value_under_direct_x11() {
    let mut values = noisy_seasonal(120);
    values[GAP] = f64::NAN;
    assert_eq!(
        adjust(
            Series::new(axis().start(), &values),
            &Spec::x11(x11::Options::new())
        )
        .unwrap_err(),
        Error::MissingValue { index: GAP }
    );
}

/// Build 62 reads `-99999.0` as its missing-value code, but here only NaN
/// is a gap.
#[test]
fn minus_99999_is_an_observation_on_every_pipeline() {
    // Shifted so that -99999 is an ordinary value of the series.
    let mut values: Vec<_> = noisy_seasonal(120)
        .into_iter()
        .map(|value| value - 100_099.0)
        .collect();
    values[GAP] = -99999.0;
    let [x11_regarima, seats] = regarima_specs();
    for spec in [
        Spec::x11(x11::Options::new().with_mode(x11::Mode::Additive)),
        x11_regarima,
        seats,
    ] {
        let adjustment = adjust(Series::new(axis().start(), &values), &spec)
            .unwrap_or_else(|error| panic!("{spec:?}: {error}"));
        let gaps = adjustment.model().map_or(0, |model| {
            model
                .regression_coefficients()
                .iter()
                .filter(|estimate| {
                    matches!(estimate.variable(), RegressionVariable::MissingValue { .. })
                })
                .count()
        });
        assert_eq!(gaps, 0, "{spec:?}");
    }
}

#[test]
fn infinity_is_nonfinite() {
    for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
        let mut values = noisy_seasonal(120);
        values[GAP] = infinity;
        assert_eq!(
            adjust(
                Series::new(axis().start(), &values),
                &Spec::x11(x11::Options::new())
            )
            .unwrap_err(),
            Error::NonFiniteValue { index: GAP }
        );
        assert_eq!(
            validate_values(&[f64::NAN, infinity]),
            Err(Error::NonFiniteValue { index: 1 })
        );
    }
    assert_eq!(validate_values(&[f64::NAN, -99999.0]), Ok(()));
    assert_eq!(validate_values(&[]), Err(Error::EmptyInput));
}

/// A gap in every period: direct X-11 refuses the first, and regARIMA
/// finds no observation of January.
#[test]
fn an_all_nan_input_is_rejected_as_all_missing() {
    let values = [f64::NAN; 120];
    assert_eq!(
        adjust(
            Series::new(axis().start(), &values),
            &Spec::x11(x11::Options::new())
        )
        .unwrap_err(),
        Error::MissingValue { index: 0 }
    );
    for spec in regarima_specs() {
        assert_eq!(
            adjust(Series::new(axis().start(), &values), &spec).unwrap_err(),
            Error::MissingSeason {
                frequency: crate::Frequency::Monthly,
                subperiod: 1,
            },
            "{spec:?}"
        );
    }
}
