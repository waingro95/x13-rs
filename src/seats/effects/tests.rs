use super::*;
use crate::Frequency;

const N: usize = 30;

fn wave(scale: f64, phase: f64) -> Vec<f64> {
    (0..N)
        .map(|i| scale * (i as f64).mul_add(0.7, phase).sin())
        .collect()
}

fn log(transform: SeatsTransform) -> bool {
    transform == SeatsTransform::Log
}

/// The ADJREG groups for `transform`; `level_shift` and `additive` replace
/// the default outlier waves.
fn input(
    transform: SeatsTransform,
    restoration: SeriesPriorRestoration,
    level_shift: &[f64],
    additive: &[f64],
) -> Result<FittedEffectsInput, Error> {
    let original = wave(10.0, 0.0)
        .into_iter()
        .map(|v| 100.0 + v)
        .collect::<Vec<_>>();
    let transformed = if log(transform) {
        original.iter().map(|v| v.ln()).collect()
    } else {
        original.clone()
    };
    let scale = if log(transform) { 0.01 } else { 1.0 };
    let fixed_length = wave(0.01, 0.4)
        .into_iter()
        .map(|v| if log(transform) { 1.0 + v } else { v })
        .collect::<Vec<_>>();
    FittedEffectsInput::with_outliers(
        TimeAxis::monthly(2000, 1).unwrap(),
        transform,
        &original,
        &transformed,
        RegressionEffects {
            trading_day: &wave(2.0 * scale, 0.1),
            holiday: &wave(1.5 * scale, 0.2),
            fixed_length: &fixed_length,
            missing: &wave(0.5 * scale, 0.3),
            user: &wave(0.25 * scale, 0.5),
        },
        restoration,
        OutlierLinearEffects {
            requires_final_adjustment: true,
            level_shift,
            additive,
            temporary_change: &wave(0.75 * scale, 0.6),
        },
    )
}

fn components(transform: SeatsTransform, cycle: Option<Vec<f64>>) -> ExtractedComponents {
    let (level, scale) = if log(transform) {
        (100.0_f64.ln(), 0.01)
    } else {
        (100.0, 1.0)
    };
    ExtractedComponents {
        frequency: Frequency::Monthly,
        observed_len: N,
        trend: wave(2.0 * scale, 0.9)
            .into_iter()
            .map(|v| level + v)
            .collect(),
        seasonal: wave(5.0 * scale, 1.0),
        irregular: wave(scale, 1.1),
        cycle,
    }
}

/// An unusable saved trend or irregular fails at the fitted-effects
/// restoration; other extraction and log failures keep their stage.
#[test]
fn saved_trend_and_irregular_failures_take_the_restoration_stage() {
    let identity = SeatsTransform::Identity;
    let mut level_shift = wave(3.0, 0.7);
    level_shift[5] = 1e308;
    let mut additive = wave(1.0, 0.8);
    additive[7] = 1e308;
    let input_identity = input(
        identity,
        SeriesPriorRestoration::Restore,
        &level_shift,
        &additive,
    )
    .unwrap();
    let mut trend = components(identity, None);
    trend.trend[5] = 1e308;
    let mut irregular = components(identity, None);
    irregular.irregular[7] = 1e308;

    let logged = SeatsTransform::Log;
    let mut level_shift = wave(0.03, 0.7);
    level_shift[5] = 311.0;
    let input_log = input(
        logged,
        SeriesPriorRestoration::Restore,
        &level_shift,
        &wave(0.01, 0.8),
    )
    .unwrap();
    let mut log_trend = components(logged, None);
    log_trend.trend[5] = 400.0;
    let mut log_trend_exp = components(logged, None);
    log_trend_exp.trend[3] = 800.0;

    let unusable = |stage, quantity, index| -> Result<FinalComponents, Error> {
        Err(Error {
            stage,
            failure: Failure::Unusable {
                quantity,
                index: Some(index),
                value: f64::INFINITY,
            },
        })
    };
    let restoration = Stage::FittedEffectsRestoration;
    assert_eq!(
        final_with_effects(&input_identity, &trend),
        unusable(restoration, "saved trend", 5)
    );
    assert_eq!(
        final_with_effects(&input_identity, &irregular),
        unusable(restoration, "saved irregular", 7)
    );
    assert_eq!(
        final_with_effects(&input_log, &log_trend),
        unusable(restoration, "saved trend", 5)
    );
    assert_eq!(
        final_with_effects(&input_log, &log_trend_exp),
        unusable(Stage::BiasCorrection, "trend exp", 3)
    );
    assert_eq!(
        final_with_effects_ar2(&input_log, &log_trend, 0.5, EffectiveTradingDay::Stochastic)
            .map(|restored| restored.saved),
        unusable(restoration, "saved trend", 5)
    );
}
