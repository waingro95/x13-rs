mod easter_state;
mod log_robust;
mod robust;

use super::extreme_scale::{RobustContext, ScaleSample};
use super::extreme_weights::{
    ExtremeSpans, calendar_extreme_weights, extreme_weight, irregular_standard_deviation,
    replace_extremes, rolling_extreme_weights, rolling_windows,
};
use super::mode::{pseudo_modified, pseudo_trend_modified};
use super::part_d::restore_final_si;
use super::seasonal_factors::{SeasonalSmoothingRequest, seasonal_filter};
use super::trend::{
    TrendFailure, replace_nonpositive_trend, trend_error, unscaled_trend_ratio, variable_trend,
};
use super::{Pass, ResolvedSeasonalFilter, SeasonalFilterSet, adjust, adjust_extended};

use crate::error::{NumericalDetail, NumericalFailure, X11ExtremeWeightingPass};
use crate::time::TimeAxis;
use crate::{numerics, x11};

#[test]
fn pseudo_additive_errors_name_the_period_on_the_extended_axis() {
    let axis = TimeAxis::quarterly(1998, 3).unwrap();
    let error = pseudo_modified(
        &[10.0; 4],
        &[1.5; 4],
        &[1.0; 4],
        &[1.0, 1.0, 0.0, 1.0],
        axis.span(4).unwrap(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        crate::Error::from(crate::error::NumericalDetail::X11PseudoAdditive {
            stage: crate::error::X11PseudoAdditiveStage::ExtremeReconstruction,
            index: 2,
            period: axis.period_at(2).unwrap(),
            reason: crate::error::X11PseudoAdditiveUnavailableReason::ZeroDivisor
        })
    );
    let failure = variable_trend(
        &[-1.0; 36],
        4,
        Pass::B,
        0.001,
        x11::Mode::PseudoAdditive,
        4..32,
        x11::Options::new(),
    )
    .err()
    .expect("no positive neighbor");
    assert_eq!(
        trend_error(
            axis.span(36).unwrap(),
            crate::x11::TrendStage::Preliminary,
            failure
        ),
        crate::Error::from(crate::error::NumericalDetail::X11Trend {
            stage: crate::x11::TrendStage::Preliminary,
            period: axis.start(),
            reason: crate::error::X11TrendUnavailableReason::NoPositiveNeighbor
        })
    );
}

#[test]
fn pseudo_additive_standard_deviation_rejects_a_nonfinite_scale() {
    let axis = TimeAxis::monthly(1999, 7).unwrap();
    let context = RobustContext {
        estimator: crate::x11::ExtremeEstimator::StandardDeviation,
        axis,
        fortran_start: 7,
        stage: crate::error::X11ExtremeStage::PreliminaryIrregular,
    };
    let error = context
        .scale(
            &[1e200; 12],
            &[1.0; 12],
            0..12,
            ScaleSample {
                first: 0,
                last: 11,
                stride: 1,
                pass: X11ExtremeWeightingPass::Initial,
                center: 1.0,
                group: None,
            },
        )
        .unwrap_err();
    assert_eq!(
        error,
        crate::Error::from(crate::error::NumericalDetail::X11ExtremeScale {
            estimator: crate::x11::ExtremeEstimator::StandardDeviation,
            reason: crate::error::X11ExtremeScaleUnavailableReason::NonFiniteScale,
            stage: crate::error::X11ExtremeStage::PreliminaryIrregular,
            pass: crate::error::X11ExtremeWeightingPass::Initial,
            first: axis.start(),
            last: axis.period_at(11).unwrap(),
            group: None,
        })
    );
}

#[test]
fn pseudo_additive_trend_checks_divisions_over_the_forecasts() {
    let axis = TimeAxis::monthly(1998, 7).unwrap();
    let mut values = [100.0; 60];
    values[36..].fill(0.0);
    let failure = variable_trend(
        &values,
        12,
        Pass::B,
        3.5,
        x11::Mode::PseudoAdditive,
        0..36,
        x11::Options::new(),
    )
    .err()
    .expect("a zero trend among the forecasts is a zero divisor");
    assert_eq!(
        trend_error(
            axis.span(60).unwrap(),
            crate::x11::TrendStage::Preliminary,
            failure
        ),
        crate::Error::from(crate::error::NumericalDetail::X11PseudoAdditive {
            stage: crate::error::X11PseudoAdditiveStage::TrendSelection,
            index: 42,
            period: axis.period_at(42).unwrap(),
            reason: crate::error::X11PseudoAdditiveUnavailableReason::ZeroDivisor,
        })
    );
}

#[test]
fn log_additive_final_bias_repairs_finite_nonpositive_trends() {
    // The bias filter's I/C is local to `finish_trend`, which cannot
    // reach the main trend's ratio.
    let axis = TimeAxis::quarterly(2000, 1).unwrap();
    let mut phase =
        crate::x11::log_additive::Phase::new(&[1.0; 24], axis.span(24).unwrap()).unwrap();
    phase.retain_c13(&[0.0; 24]);
    let trend = variable_trend(
        &[0.0; 24],
        4,
        Pass::D,
        2.7,
        x11::Mode::LogAdditive,
        2..22,
        x11::Options::new(),
    )
    .unwrap();
    let mut values = trend.values;
    values[0] = -1000.0;
    let mut seasonal = [1.0; 24];
    seasonal[4] = 100.0;
    phase
        .finish_trend(
            &mut values,
            &seasonal,
            4,
            crate::x11::SevenTermEndpoints::HendersonSevenTerm,
        )
        .unwrap();
    assert!(values.iter().any(|&v| v < 0.0));
    assert_eq!(values[0], 0.0);
    let count = replace_nonpositive_trend(&mut values, 2..22, None).unwrap();
    assert!(count >= 2);
    assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
}

#[test]
fn calendar_sigma_tests_read_the_estimation_span_then_backcasts_and_observations() {
    use crate::x11::AutomaticExtremeScale::{ByCalendarPeriod, RollingFiveYear};
    use crate::x11::ExtremeEstimationSpan as Span;
    let axis = TimeAxis::monthly(2000, 7).unwrap();
    // Twelve backcasts and twelve forecasts around 96 observations.
    let choices = |outlier: usize, policy| {
        let mut values: Vec<_> = (0..120)
            .map(|i| 100.0 + i as f64 * 0.25 + ((i * 17) % 23) as f64)
            .collect();
        values[outlier] += 60.0;
        let options = x11::Options::new()
            .with_extreme_scale(crate::x11::ExtremeScale::Automatic)
            .with_extreme_estimation_span(policy);
        let selection = adjust_extended(
            &values,
            axis,
            12..108,
            x11::Mode::Additive,
            super::PreparedOptions::new(options, axis, 96).unwrap(),
        )
        .unwrap()
        .calendar_sigma_selection
        .unwrap();
        (selection.initial_si(), selection.preliminary_irregular())
    };
    // An outlier among the backcasts: the first test reads it only when the
    // estimation span has the extensions; the second always does.
    assert_eq!(
        choices(8, Span::WithExtensions),
        (ByCalendarPeriod, ByCalendarPeriod)
    );
    assert_eq!(
        choices(8, Span::ObservedOnly),
        (RollingFiveYear, ByCalendarPeriod)
    );
    // An outlier among the forecasts: only the first test reads it, and only
    // with the extensions.
    assert_eq!(
        choices(111, Span::WithExtensions),
        (ByCalendarPeriod, RollingFiveYear)
    );
    assert_eq!(
        choices(111, Span::ObservedOnly),
        (RollingFiveYear, RollingFiveYear)
    );
}

#[test]
fn calendar_sigma_choice_reads_only_its_span() {
    use crate::x11::AutomaticExtremeScale as Choice;
    let mut irregular = vec![1.0; 48];
    irregular[44] = 10_000.0; // A forecast, outside 0..40.
    assert_eq!(
        crate::x11::calendar_sigma::select(&irregular, 0..40, 4, x11::Mode::Additive),
        Choice::RollingFiveYear
    );
    assert_eq!(
        crate::x11::calendar_sigma::select(&irregular, 0..48, 4, x11::Mode::Additive),
        Choice::ByCalendarPeriod
    );
    irregular[44] = 1.0;
    irregular[0] = 10_000.0; // A backcast, inside 0..40.
    assert_eq!(
        crate::x11::calendar_sigma::select(&irregular, 0..40, 4, x11::Mode::Additive),
        Choice::ByCalendarPeriod
    );
    assert_eq!(
        crate::x11::calendar_sigma::select(&irregular, 8..40, 4, x11::Mode::Additive),
        Choice::RollingFiveYear
    );
}

#[test]
fn standard_deviation_fuses_each_square_into_the_sum() {
    // The fused and unfused sums of these squares differ in the last bit.
    let values = [0.05711422845691383, 0.07722772277227723, 0.4064885496183206];
    let actual = irregular_standard_deviation(
        &values,
        &[1.0; 3],
        0..3,
        X11ExtremeWeightingPass::Initial,
        0.0,
    );
    let nonfused = ((values.iter().map(|value| value * value).sum::<f64>()) / 3.0).sqrt();
    assert_eq!(nonfused.to_bits(), 0x3fceddfbc60efc68);
    assert_eq!(actual.to_bits(), 0x3fceddfbc60efc69);
}

#[test]
fn standard_deviation_centers_and_skips_zero_weights_when_reweighting() {
    use X11ExtremeWeightingPass::{Initial, Reweighted};
    let values = [99.0, 1.5, 99.0, 2.5, 99.0, 4.5];
    let mut weights = [1.0; 6];
    weights[3] = 0.0;
    let selected = (1..6).step_by(2);
    assert_eq!(
        irregular_standard_deviation(&values, &weights, selected.clone(), Initial, 1.0).to_bits(),
        (14.75_f64 / 3.0).sqrt().to_bits()
    );
    assert_eq!(
        irregular_standard_deviation(&values, &weights, selected, Reweighted, 1.0).to_bits(),
        2.5_f64.to_bits()
    );
}

#[test]
fn final_si_restores_backcasts_and_observations_but_not_forecasts() {
    let mut si: Vec<_> = (1..=32).map(f64::from).collect();
    let mut weights = vec![1.0; 32];
    weights[0] = 0.0; // A backcast.
    weights[24] = 0.0; // An observation.
    weights[28] = 0.0; // A forecast.
    let state = super::FinalCExtremes {
        component: vec![1.0; 32],
        weights,
    };
    restore_final_si(&mut si, &state, 28, 4, x11::Mode::Additive);
    assert_eq!(si[0], 12.0); // The next four full-weight values: 6, 10, 14, 18.
    assert_eq!(si[24], 16.0); // The last four full-weight values: 22, 18, 14, 10.
    assert_eq!(&si[28..], &[29.0, 30.0, 31.0, 32.0]);
}

#[test]
fn groups_follow_calendar_and_shifted_starts() {
    use crate::{x11::ExtremeScale as Scale, x11::SigmaGroup as Group, x11::SigmaGroups as Groups};
    let mut irregular = [1.0, 4.0, 1.0, 4.0].repeat(8);
    irregular[0] = 9.0;
    let scale = Scale::Grouped(Groups::Quarterly([
        Group::Selected,
        Group::Complement,
        Group::Complement,
        Group::Complement,
    ]));
    let shifted = calendar_extreme_weights(
        &irregular,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
        scale,
    )
    .unwrap();
    let unshifted = calendar_extreme_weights(
        &irregular,
        ExtremeSpans::new(0..32, 0..32),
        4,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
        scale,
    )
    .unwrap();
    // A series starting in Q3 puts the spike in the complement, whose larger
    // scale grades it. Starting in Q1 puts it in the selected group, which
    // rejects it.
    assert!(shifted[0] > 0.0 && shifted[0] < 1.0);
    assert_eq!(unshifted[0], 0.0);
    assert!(shifted[1..].iter().all(|&weight| weight == 1.0));
}

#[test]
fn extreme_component_fuses_its_denominator() {
    // The fused and unfused denominators of these values differ.
    let irregular: f64 = 0.8023528027763073;
    let weight: f64 = 0.7615230460921844;
    let nonfused = irregular / (1.0 + weight * (irregular - 1.0));
    assert_ne!(nonfused.to_bits(), 0x3fee39764368c2f6);
    let actual = x11::Mode::Multiplicative.extreme_component(irregular, weight);
    assert_eq!(actual.to_bits(), 0x3fee39764368c2f6);
}

#[test]
fn calendar_period_reweighting_reads_weights_already_changed() {
    let mut irregular = vec![0.0; 24];
    irregular[4] = 2.0;
    for index in (1..24).step_by(4) {
        irregular[index] = 1.5;
    }
    let actual = calendar_extreme_weights(
        &irregular,
        ExtremeSpans::new(0..24, 4..20),
        4,
        1,
        x11::Mode::Additive,
        crate::x11::SigmaLimits::new(1.0, 2.0).unwrap(),
        crate::x11::ExtremeScale::ByCalendarPeriod,
    )
    .unwrap();
    assert_eq!(actual[4], 0.0);
    // Q1's reweighting has just rejected index 4. Q2's sample starts there,
    // so its scale is 0 and its weights stay. Scales computed before any
    // weight changed would give Q2 a weight of 0.5.
    for index in (1..24).step_by(4) {
        assert_eq!(actual[index], 1.0);
    }
}

#[test]
fn calendar_period_sample_before_the_estimation_span_starts_at_its_first_value() {
    let irregular = [1.0, 2.0, 4.0, 8.0].repeat(5);
    for start in [1, 3] {
        let actual = calendar_extreme_weights(
            &irregular,
            ExtremeSpans::new(0..20, 4..16),
            4,
            start,
            x11::Mode::Additive,
            x11::Options::new().sigma_limits(),
            crate::x11::ExtremeScale::ByCalendarPeriod,
        )
        .unwrap();
        // All four periods start before the estimation span, so each samples
        // 1, 1, 1 from index 4. Samples of their own periods would leave every
        // weight 1.
        assert_eq!(actual, [1.0, 0.5, 0.0, 0.0].repeat(5));
    }
}

#[test]
fn calendar_period_and_grouped_scales_guard_zero_differently() {
    use crate::{x11::ExtremeScale as Scale, x11::SigmaGroup as Group, x11::SigmaGroups as Groups};
    let mut tiny = vec![0.0; 120];
    tiny[0] = 1e-20;
    let period = calendar_extreme_weights(
        &tiny,
        ExtremeSpans::new(0..120, 0..120),
        12,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
        Scale::ByCalendarPeriod,
    )
    .unwrap();
    let grouped = calendar_extreme_weights(
        &tiny,
        ExtremeSpans::new(0..120, 0..120),
        12,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
        Scale::Grouped(Groups::Monthly([Group::Selected; 12])),
    )
    .unwrap();
    assert_eq!(period, vec![1.0; 120]);
    assert_eq!(grouped[0], 0.0);

    let mut empty_pass_two = vec![0.0; 20];
    empty_pass_two[4..16].fill(100.0);
    let limits = crate::x11::SigmaLimits::new(0.1, 0.2).unwrap();
    let period = calendar_extreme_weights(
        &empty_pass_two,
        ExtremeSpans::new(0..20, 4..16),
        4,
        1,
        x11::Mode::Additive,
        limits,
        Scale::ByCalendarPeriod,
    )
    .unwrap();
    let grouped = calendar_extreme_weights(
        &empty_pass_two,
        ExtremeSpans::new(0..20, 4..16),
        4,
        1,
        x11::Mode::Additive,
        limits,
        Scale::Grouped(Groups::Quarterly([Group::Selected; 4])),
    )
    .unwrap();
    assert_eq!(&period[4..16], &[0.0; 12]);
    assert_eq!(&grouped[4..16], &[0.0; 12]);
    for index in [0, 1, 2, 3, 16, 17, 18, 19] {
        assert_eq!(period[index], -1.0);
        assert_eq!(grouped[index], 1.0);
    }
}

#[test]
fn final_si_takes_back_the_c_extremes_and_replaces_under_the_c_weights() {
    // The stored factor 0.7 differs in its last bit from 0.49 / 0.7, the
    // factor recovered from the original and modified series.
    let state = super::FinalCExtremes {
        component: vec![0.7; 8],
        weights: vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    };
    let mut si = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 100.0, 200.0];
    restore_final_si(&mut si, &state, 6, 1, x11::Mode::Multiplicative);
    assert_eq!(si[0].to_bits(), 1.4_f64.to_bits());
    assert_eq!(si[1].to_bits(), 4.55_f64.to_bits()); // (1.4+4.2+5.6+7)/4
    assert_eq!(si[5].to_bits(), (12.0_f64 * 0.7).to_bits());
    assert_eq!(&si[6..], &[100.0, 200.0]);
}

#[test]
fn short_estimation_span_keeps_rejection_marks_on_the_extensions() {
    let mut irregular = vec![1.0; 20];
    for index in [0, 14, 19] {
        irregular[index] = 4.0;
    }
    irregular[15] = 10.0;
    let weights = extreme_weights(
        &irregular,
        ExtremeSpans::new(0..20, 4..16),
        4,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
    );
    let mut expected = vec![1.0; 20];
    expected[0] = -1.0;
    expected[19] = -1.0;
    expected[14] = 0.0;
    expected[15] = 0.0;
    assert_eq!(weights, expected);
    let mut replaced = irregular;
    replace_extremes(&mut replaced, &weights, 0..20, 4);
    assert_eq!(replaced[0], 0.0);
    assert_eq!(replaced[19], 3.4); // The period's mean: (1 + 1 + 1 + 10 + 4) / 5.
}

#[test]
fn rolling_windows_follow_the_calendar_not_the_series_start() {
    let mut irregular: Vec<_> = (0..80)
        .map(|i| ((i * 17 % 19) as f64 - 9.0) / 4.0)
        .collect();
    for index in [12, 31, 48, 70] {
        irregular[index] *= 2.0;
    }
    let weights = extreme_weights(
        &irregular,
        ExtremeSpans::new(0..80, 4..72),
        4,
        3,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
    );
    assert!(weights.iter().any(|&weight| weight > 0.0 && weight < 1.0));
    let mut shifted = vec![0.0; 2];
    shifted.extend(&irregular);
    let shifted_weights = extreme_weights(
        &shifted,
        ExtremeSpans::new(2..82, 6..74),
        4,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
    );
    assert_eq!(&shifted_weights[..2], &[1.0, 1.0]);
    assert!(
        weights
            .iter()
            .zip(&shifted_weights[2..])
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
fn first_section_trims_observed_span_before_estimation() {
    for (observed, application) in [(12..132, 6..144), (0..120, 6..126), (7..127, 6..121)] {
        let spans = ExtremeSpans::for_section(
            application.clone(),
            observed.clone(),
            6,
            crate::x11::ExtremeEstimationSpan::ObservedOnly,
        );
        assert_eq!(spans.application, application);
        assert_eq!(spans.estimation, observed.start + 6..observed.end - 6);
    }
}

fn extreme_weights(
    irregular: &[f64],
    spans: ExtremeSpans,
    period: usize,
    start_subperiod: usize,
    mode: x11::Mode,
    limits: crate::x11::SigmaLimits,
) -> Vec<f64> {
    rolling_extreme_weights(
        irregular,
        spans,
        period,
        start_subperiod,
        mode,
        limits,
        None,
    )
    .unwrap()
}

#[test]
fn section_resolution_preserves_explicit_filters_and_selected_msr() {
    use super::{ResolvedSeasonalFilter as Resolved, SeasonalFilterSet::Monthly};
    use crate::x11::SeasonalFilter::*;
    let requested = Monthly([
        X11Default, Msr, Stable, S3x1, S3x3, S3x5, S3x9, S3x15, X11Default, Msr, Stable, S3x9,
    ]);
    let resolve = |filters: &super::SeasonalFilterSet<x11::SeasonalFilter>, provisional| {
        filters.resolve(provisional).map(x11::SeasonalFilter::from)
    };
    assert_eq!(
        resolve(&requested, Resolved::S3x3),
        Monthly([
            S3x3, S3x3, Stable, S3x1, S3x3, S3x5, S3x9, S3x15, S3x3, S3x3, Stable, S3x9
        ])
    );
    assert_eq!(
        resolve(&requested, Resolved::S3x5),
        Monthly([
            S3x5, S3x5, Stable, S3x1, S3x3, S3x5, S3x9, S3x15, S3x5, S3x5, Stable, S3x9
        ])
    );
    let selected = crate::x11::msr::resolve(&requested, S3x9);
    assert_eq!(
        resolve(&selected, Resolved::S3x5),
        Monthly([
            S3x5, S3x9, Stable, S3x1, S3x3, S3x5, S3x9, S3x15, S3x5, S3x9, Stable, S3x9
        ])
    );
}

#[test]
fn custom_sigma_limits_control_graduation_and_equal_limit_cutoff() {
    // `wtxtrm.f`: the first pass keeps the weight of a value within the upper
    // limit; the second grades only lower < deviation <= upper.
    use X11ExtremeWeightingPass::{Initial, Reweighted};
    let limits = crate::x11::SigmaLimits::new(1.0, 3.0).unwrap();
    assert_eq!(extreme_weight(2.0, 1.0, Initial, 1.0, 0.0, limits), 1.0);
    assert_eq!(extreme_weight(2.0, 1.0, Reweighted, 1.0, 0.0, limits), 0.5);
    let equal = crate::x11::SigmaLimits::new(2.0, 2.0).unwrap();
    assert_eq!(extreme_weight(2.0, 1.0, Reweighted, 1.0, 0.0, equal), 1.0);
    assert_eq!(extreme_weight(3.0, 1.0, Initial, 1.0, 0.0, equal), 0.0);
    assert_eq!(extreme_weight(3.0, 1.0, Reweighted, 1.0, 0.0, equal), -1.0);
    assert_eq!(extreme_weight(3.0, 1.0, Reweighted, 0.0, 0.0, equal), 0.0);
}

#[test]
fn short_spans_use_stable_seasonality_before_centering() {
    let si = [
        1.0, 10.0, 20.0, 30.0, 2.0, 10.0, 20.0, 30.0, 3.0, 10.0, 20.0, 30.0, 8.0, 10.0, 20.0, 30.0,
    ];

    assert_eq!(
        seasonal_filter(
            &si,
            0..si.len(),
            x11::Mode::Additive,
            SeasonalSmoothingRequest {
                filters: &SeasonalFilterSet::uniform(
                    crate::Frequency::Quarterly,
                    ResolvedSeasonalFilter::S3x5,
                ),
                short_series_filters: crate::x11::ShortSeriesFilters::StableFallback,
            },
            TimeAxis::quarterly(2000, 1)
                .unwrap()
                .span(si.len())
                .unwrap(),
        )
        .unwrap()
        .values,
        [
            -12.375, -5.875, 4.125, 14.125, -12.375, -5.875, 4.125, 14.125, -12.375, -5.875, 4.125,
            14.125, -12.375, -5.875, 4.125, 14.125,
        ]
    );
}

#[test]
fn last_filter_is_the_last_subseries_smoothed_not_the_last_calendar_period() {
    let mut filters = [ResolvedSeasonalFilter::S3x3; 12];
    filters[5] = ResolvedSeasonalFilter::S3x9;
    let result = seasonal_filter(
        &vec![2.; 240],
        0..240,
        x11::Mode::Additive,
        SeasonalSmoothingRequest {
            filters: &SeasonalFilterSet::Monthly(filters),
            short_series_filters: x11::ShortSeriesFilters::StableFallback,
        },
        TimeAxis::monthly(2000, 7).unwrap().span(240).unwrap(),
    )
    .unwrap();
    assert_eq!(result.last_filter, ResolvedSeasonalFilter::S3x9);
}

#[test]
fn last_filter_keeps_3x15_and_reports_the_short_series_fallback() {
    for (count, filter, want) in [
        (
            239,
            ResolvedSeasonalFilter::S3x15,
            ResolvedSeasonalFilter::S3x15,
        ),
        (
            48,
            ResolvedSeasonalFilter::S3x3,
            ResolvedSeasonalFilter::Stable,
        ),
    ] {
        let result = seasonal_filter(
            &vec![2.; count],
            0..count,
            x11::Mode::Additive,
            SeasonalSmoothingRequest {
                filters: &SeasonalFilterSet::uniform(crate::Frequency::Monthly, filter),
                short_series_filters: x11::ShortSeriesFilters::StableFallback,
            },
            TimeAxis::monthly(2000, 7).unwrap().span(count).unwrap(),
        )
        .unwrap();
        assert_eq!(result.last_filter, want);
    }
}

#[test]
fn seasonal_history_counts_the_extensions_the_series_has() {
    let options = crate::x11::Options::new()
        .with_seasonal_filter(crate::x11::SeasonalFilter::S3x3)
        .with_short_series_filters(crate::x11::ShortSeriesFilters::UseRequested);
    let requested_axis = crate::time::TimeAxis::monthly(2001, 7).unwrap();
    let extended_axis = crate::time::TimeAxis::monthly(2001, 1).unwrap();
    let extended: Vec<_> = (0..48)
        .map(|index| 100.0 + 0.2 * index as f64 + (index % 12) as f64)
        .collect();
    let with_extensions = adjust_extended(
        &extended,
        extended_axis,
        6..42,
        x11::Mode::Additive,
        super::PreparedOptions::new(options, requested_axis, 36).unwrap(),
    );
    assert!(with_extensions.is_ok());
    let without = adjust_extended(
        &extended[6..42],
        requested_axis,
        0..36,
        x11::Mode::Additive,
        super::PreparedOptions::new(options, requested_axis, 36).unwrap(),
    );
    assert!(matches!(
        without,
        Err(crate::Error::InsufficientSeasonalHistory {
            frequency: crate::Frequency::Monthly,
            calendar_period: 1,
            requested: crate::x11::SeasonalFilter::S3x3,
            resolved: crate::x11::SeasonalFilter::S3x3,
            actual: 2,
            minimum: 3,
        })
    ));
}

#[test]
fn adjust_extended_returns_the_observed_span_of_the_full_run() {
    let values = (0..96)
        .map(|index| 100.0 + index as f64 * 0.2 + ((index * 7) % 12) as f64 * 1.5)
        .collect::<Vec<_>>();
    let axis = TimeAxis::monthly(1999, 1).unwrap();

    let full = adjust(&values, axis, x11::Mode::Additive, x11::Options::new()).unwrap();
    let cropped = adjust_extended(
        &values,
        axis,
        12..84,
        x11::Mode::Additive,
        super::PreparedOptions::new(x11::Options::new(), axis, values.len()).unwrap(),
    )
    .unwrap();

    assert_eq!(cropped.seasonal, full.seasonal[12..84]);
    assert_eq!(
        cropped.seasonally_adjusted,
        full.seasonally_adjusted[12..84]
    );
    assert_eq!(cropped.trend_cycle, full.trend_cycle[12..84]);
    assert_eq!(cropped.irregular, full.irregular[12..84]);
}

#[test]
fn forecast_extension_does_not_participate_in_henderson_selection() {
    let mut values = (0..96)
        .map(|index| {
            let seasonal = (index % 12) as f64 - 5.5;
            1_000.0 + index as f64 * 2.0 + seasonal * 2.0
        })
        .collect::<Vec<_>>();
    let mut forecast = *values.last().unwrap();
    for _ in 0..12 {
        forecast *= 1.01;
        values.push(forecast);
    }

    let actual = variable_trend(
        &values,
        12,
        Pass::D,
        3.5,
        x11::Mode::Multiplicative,
        0..96,
        x11::Options::new(),
    )
    .unwrap();

    assert_eq!(actual.terms, 13);
}

#[test]
fn tiny_nonzero_trend_change_uses_build_62_zero_threshold() {
    let values = (0..72)
        .map(|index| (index * index) as f64 * 1e-25)
        .collect::<Vec<_>>();

    let actual = variable_trend(
        &values,
        12,
        Pass::D,
        3.5,
        x11::Mode::Additive,
        0..values.len(),
        x11::Options::new(),
    )
    .unwrap();
    let (expected, _) = numerics::henderson_filter(
        &values,
        23,
        4.5,
        crate::x11::SevenTermEndpoints::LegacyFiveTerm,
    );

    assert_eq!(actual.values, expected);
}

#[test]
fn tiny_nonzero_standard_deviation_does_not_mark_extremes() {
    let mut irregular = [0.0; 72];
    irregular[36] = 1e-19;

    let weights = extreme_weights(
        &irregular,
        ExtremeSpans::new(0..72, 0..72),
        12,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
    );

    assert_eq!(weights, [1.0; 72]);
}

#[test]
fn extreme_windows_align_to_the_starting_calendar_period() {
    let mut irregular = (0..120)
        .map(|index| ((index * 37 % 23) as f64 - 11.0) / 5.0)
        .collect::<Vec<_>>();
    for index in [20, 55, 79, 103] {
        irregular[index] *= 3.0;
    }

    let weights = extreme_weights(
        &irregular,
        ExtremeSpans::new(0..120, 0..120),
        12,
        12,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
    );

    assert!((weights[5] - 0.995_574_239_416_913_8).abs() < 1e-15);
}

#[test]
fn quarterly_extreme_windows_align_to_the_starting_quarter() {
    let mut irregular = (0..80)
        .map(|index| ((index * 17 % 19) as f64 - 9.0) / 4.0)
        .collect::<Vec<_>>();
    for index in [12, 31, 48, 70] {
        irregular[index] *= 2.0;
    }

    let weights = extreme_weights(
        &irregular,
        ExtremeSpans::new(0..80, 0..80),
        4,
        3,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
    );

    assert!((weights[10] - 0.957_646_400_610_203_5).abs() < 1e-15);
}

#[test]
fn rolling_windows_cover_short_five_year_and_longer_spans() {
    let windows = |application, estimation| {
        rolling_windows(application, estimation, 4)
            .map(|window| (window.weighted, window.sample))
            .collect::<Vec<_>>()
    };
    // Fewer than five full years after the first full year: one window.
    assert_eq!(windows(0..24, 2..22), [(0..24, 2..22)]);
    // Exactly five years: the last window is the first.
    assert_eq!(windows(0..20, 0..20), [(0..20, 0..20)]);
    // Seven years: a first, two middle and a last window.
    assert_eq!(
        windows(0..30, 1..29),
        [
            (0..12, 1..24),
            (12..16, 4..24),
            (16..20, 8..28),
            (20..30, 8..29),
        ]
    );
}

/// The endpoint I/C ratio `variable_trend` leaves for the next pass, on every
/// length-selection branch.
#[test]
fn variable_trend_leaves_the_selected_endpoint_ratio() {
    let fixed = |terms| {
        x11::Options::new().with_trend_filter(crate::x11::TrendFilter::Henderson(
            crate::x11::HendersonLength::new(terms).unwrap(),
        ))
    };
    let seven_term = x11::Options::new()
        .with_seven_term_endpoints(crate::x11::SevenTermEndpoints::HendersonSevenTerm);
    // Per pass (B, D): the default, seven-term endpoints, and fixed 7 and
    // 13 terms. Fixed lengths keep the I/C unless legacy 7-term endpoints
    // reset it.
    let base = |terms| [(terms, 2.5), (terms, 2.5), (7, 0.001), (13, 2.5)];
    let quarterly_long = [(7, 0.001), (7, 4.5), (7, 0.001), (13, 2.5)];
    let expected = [
        // Quarterly: a ratio below 1 keeps 5 terms; above 3.5, D takes 7.
        [base(5), base(5)],
        [base(5), quarterly_long],
        [base(5), quarterly_long],
        // Monthly: below 1 takes 9 terms at I/C 1; above 3.5, D takes 23.
        [[(9, 1.0), (9, 1.0), (7, 0.001), (13, 2.5)]; 2],
        [base(13), base(13)],
        [base(13), [(23, 4.5), (23, 4.5), (7, 0.001), (13, 2.5)]],
    ];
    let mut actual = Vec::new();
    for period in [4, 12] {
        for irregular in [0.4, 1.5, 30.0] {
            let values = (0..period * 6)
                .map(|i| {
                    let sign = if (i * 7) % 3 == 0 { 1.0 } else { -1.0 };
                    100.0 + i as f64 + sign * irregular
                })
                .collect::<Vec<_>>();
            let passes = [Pass::B, Pass::D].map(|pass| {
                [x11::Options::new(), seven_term, fixed(7), fixed(13)].map(|options| {
                    let trend = variable_trend(
                        &values,
                        period,
                        pass,
                        2.5,
                        x11::Mode::Additive,
                        0..values.len(),
                        options,
                    )
                    .unwrap();
                    (trend.terms, trend.trend_ic)
                })
            });
            actual.push(passes);
        }
    }
    assert_eq!(actual, expected);
}

#[test]
fn base_length_keeps_the_inherited_endpoint_ratio() {
    let values = (0..72)
        .map(|index| index as f64 + if index % 2 == 0 { 0.75 } else { -0.75 })
        .collect::<Vec<_>>();
    let actual = variable_trend(
        &values,
        12,
        Pass::D,
        4.5,
        x11::Mode::Additive,
        0..values.len(),
        x11::Options::new(),
    )
    .unwrap();
    let (expected, _) = numerics::henderson_filter(
        &values,
        13,
        4.5,
        crate::x11::SevenTermEndpoints::LegacyFiveTerm,
    );

    assert_eq!(actual.values, expected);
    assert_eq!(actual.trend_ic, 4.5);
}

#[test]
fn multiplicative_variable_trend_replaces_nonpositive_estimates() {
    let mut values = vec![100.0; 25];
    values[12] = 1_000_000.0;

    let trend = variable_trend(
        &values,
        4,
        Pass::B,
        0.001,
        x11::Mode::Multiplicative,
        0..values.len(),
        x11::Options::new(),
    )
    .unwrap()
    .values;

    assert!(
        trend.iter().all(|value| *value > 0.0),
        "Build 62 replaces nonpositive multiplicative trend estimates: {trend:?}"
    );
}

#[test]
fn nonpositive_trend_replacement_reports_the_number_replaced() {
    let mut trend = [4.0, -2.0, 8.0, 0.0, 12.0];

    let replacements = replace_nonpositive_trend(&mut trend, 0..5, None).unwrap();

    assert_eq!(replacements, 2);
    assert_eq!(trend, [4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn trend_replacement_searches_each_side_within_the_observed_span() {
    type Case<'a> = (&'a [f64], std::ops::Range<usize>, &'a [f64], usize);
    let cases: &[Case<'_>] = &[
        (
            &[100.0, -1.0, 10.0, 20.0],
            1..4,
            &[100.0, 10.0, 10.0, 20.0],
            1,
        ),
        (
            &[10.0, 20.0, -1.0, 100.0],
            0..3,
            &[10.0, 20.0, 20.0, 100.0],
            1,
        ),
        (
            &[100.0, -1.0, 20.0, 30.0],
            2..4,
            &[100.0, 20.0, 20.0, 30.0],
            1,
        ),
        (
            &[10.0, 20.0, -1.0, 100.0],
            0..2,
            &[10.0, 20.0, 20.0, 100.0],
            1,
        ),
        (&[-1.0, 8.0, 20.0, 30.0], 2..4, &[8.0, 8.0, 20.0, 30.0], 1),
        (&[20.0, 30.0, 8.0, -1.0], 0..2, &[20.0, 30.0, 8.0, 8.0], 1),
        (
            &[100.0, -1.0, -1.0, 40.0, 200.0],
            1..4,
            &[100.0, 40.0, 40.0, 40.0, 200.0],
            2,
        ),
        (
            &[10.0, -1.0, -1.0, 40.0],
            0..4,
            &[10.0, 25.0, 32.5, 40.0],
            2,
        ),
    ];
    for (input, observed, expected, count) in cases {
        let mut trend = input.to_vec();
        let actual = replace_nonpositive_trend(&mut trend, observed.clone(), None).unwrap();
        assert_eq!(&trend, expected, "input {input:?}");
        assert_eq!(&actual, count, "input {input:?}");
    }
}

#[test]
fn trend_replacement_rejects_unavailable_and_nonfinite_results() {
    use crate::error::X11TrendUnavailableReason as Reason;

    let cases: &[(&[f64], std::ops::Range<usize>, usize, Reason)] = &[
        (
            &[10.0, -1.0, -2.0, 20.0],
            1..3,
            1,
            Reason::NoPositiveNeighbor,
        ),
        (&[-1.0, -2.0, 0.0], 0..3, 0, Reason::NoPositiveNeighbor),
        (&[f64::INFINITY, 10.0], 0..2, 0, Reason::NonFiniteEstimate),
        (&[10.0, f64::NAN], 0..2, 1, Reason::NonFiniteEstimate),
        (
            &[f64::MAX, -1.0, f64::MAX],
            0..3,
            1,
            Reason::NonFiniteReplacement,
        ),
    ];
    for (input, observed, index, reason) in cases {
        let actual = replace_nonpositive_trend(&mut input.to_vec(), observed.clone(), None);
        assert_eq!(
            actual,
            Err(TrendFailure {
                index: *index,
                reason: *reason
            })
        );
    }
}

#[test]
fn zeros_of_either_sign_are_replaced_in_both_extensions() {
    for zero in [0.0, -0.0] {
        let mut trend = [8.0, zero, 20.0, zero, 30.0];
        let count = replace_nonpositive_trend(&mut trend, 2..3, None).unwrap();
        assert_eq!(count, 2);
        assert_eq!(trend, [8.0, 20.0, 20.0, 20.0, 30.0]);
    }
}

#[cfg(feature = "__conditioning-scales")]
#[test]
fn trend_operation_scales_use_the_selected_directional_neighbors() {
    let mut trend = [100.0, -1.0, 10.0, 20.0];
    let mut scales = [101.0, 9.0, 11.0, 21.0];
    let count = replace_nonpositive_trend(&mut trend, 1..4, Some(&mut scales)).unwrap();
    assert_eq!(count, 1);
    assert_eq!(trend, [100.0, 10.0, 10.0, 20.0]);
    assert_eq!(scales, [101.0, 11.0, 11.0, 21.0]);

    let mut trend = [10.0, -1.0, 40.0];
    let mut scales = [11.0, 9.0, 41.0];
    replace_nonpositive_trend(&mut trend, 0..3, Some(&mut scales)).unwrap();
    assert_eq!(scales[1], 51.0);
}

#[test]
fn trend_failure_names_its_stage_period_and_reason() {
    let axis = crate::time::TimeAxis::monthly(2000, 11).unwrap();
    let failure = TrendFailure {
        index: 2,
        reason: crate::error::X11TrendUnavailableReason::NoPositiveNeighbor,
    };
    for stage in [
        crate::x11::TrendStage::FinalIteration,
        crate::x11::TrendStage::FinalTrendCycle,
    ] {
        let error = trend_error(axis.span(3).unwrap(), stage, failure);
        assert_eq!(
            error,
            crate::Error::from(crate::error::NumericalDetail::X11Trend {
                stage,
                period: axis.period_at(2).unwrap(),
                reason: crate::error::X11TrendUnavailableReason::NoPositiveNeighbor,
            })
        );
        assert_eq!(
            error.to_string(),
            format!(
                "X-11 cannot correct the {stage} estimate at 2001-01: no neighboring estimate is positive"
            )
        );
    }
}

#[test]
fn replacement_uses_extra_values_from_one_side_when_the_other_side_is_exhausted() {
    let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weights = [1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0];

    replace_extremes(&mut values, &weights, 0..8, 1);

    assert_eq!(values[3], 26.0 / 9.0);
}

#[test]
fn replacement_accumulates_weighted_value_before_neighbors() {
    let mut values = [1.0, 1.0, 1.0, 2.0e16, -1.0e16, 1.0, 1.0, 1.0];
    let weights = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];

    replace_extremes(&mut values, &weights, 0..8, 1);

    assert_eq!(values[3], 2.0 / 3.0);
}

#[test]
fn unscaled_trend_ratio_is_the_plain_quotient() {
    assert_eq!(unscaled_trend_ratio(1.0, 10.0), 0.1);
    assert_ne!(unscaled_trend_ratio(1.0, 10.0), 0.30000000000000004 / 3.0);
    assert_eq!(unscaled_trend_ratio(1.0, 0.0), 999.0);
}

#[test]
fn trend_only_pseudo_additive_modified_series_rounds_the_quotient_first() {
    use crate::Error;
    use crate::error::X11PseudoAdditiveStage as PseudoStage;
    use crate::x11::pseudo_additive as pseudo;
    let axis = TimeAxis::quarterly(2000, 3).unwrap();
    let tiny = 2.0_f64.powi(-54);
    assert_eq!(
        pseudo_trend_modified(&[8.0], &[tiny], &[1.0], axis.span(1).unwrap()).unwrap(),
        [2.0_f64.powi(-51)]
    );
    assert_ne!(pseudo::modified(8.0, 1.0, tiny, 1.0), 2.0_f64.powi(-51));
    assert!(
        matches!(pseudo_trend_modified(&[8.0, 8.0], &[1.0, 0.0], &[1.0, 0.0], axis.span(2).unwrap()),
        Err(Error::Numerical(NumericalFailure(NumericalDetail::X11PseudoAdditive { index: 1, period, stage: PseudoStage::ExtremeReconstruction, .. })))
        if period == axis.period_at(1).unwrap())
    );
}

/// A trend, a fixed seasonal pattern, noise that is larger in one calendar
/// period, and one outlier at index 41.
fn uneven_series(frequency: crate::Frequency) -> (TimeAxis, Vec<f64>) {
    let (axis, period, count) = match frequency {
        crate::Frequency::Monthly => (TimeAxis::monthly(1990, 3).unwrap(), 12, 144),
        crate::Frequency::Quarterly => (TimeAxis::quarterly(1990, 2).unwrap(), 4, 64),
    };
    let values = (0..count)
        .map(|i: usize| {
            let seasonal = [
                8.0, -3.0, 5.0, -6.0, 2.0, 7.0, -4.0, 1.0, -2.0, 6.0, -5.0, 3.0,
            ][i % period];
            let noise = f64::from(((i * 7919 + 104_729) % 1009) as u16) / 63.0 - 8.0;
            let scale = if i % period == 2 {
                3.0 + f64::from((i % 7) as u8)
            } else {
                0.4
            };
            let outlier = if i == 41 { 25.0 } else { 0.0 };
            100.0 + 0.3 * i as f64 + seasonal + noise * scale + outlier
        })
        .collect();
    (axis, values)
}

/// The seasonally adjusted value, trend, seasonal factor and intermediate
/// extreme weight at `index`.
fn values_at(components: &crate::x11::Components, index: usize) -> [f64; 4] {
    [
        components.seasonally_adjusted[index],
        components.trend_cycle[index],
        components.seasonal[index],
        components.intermediate_extreme_weights[index],
    ]
}

/// One run in each mode, each with a different extreme scale. Every index
/// read has a graded extreme weight, so its values depend on the scale as well
/// as on the filters.
#[test]
fn run_values_in_each_mode() {
    use crate::{
        Frequency, x11::Decomposition, x11::ExtremeEstimator as Estimator,
        x11::ExtremeScale as Scale, x11::SigmaGroup as Group, x11::SigmaGroups as Groups,
    };
    let (quarterly_axis, quarterly) = uneven_series(Frequency::Quarterly);
    let (monthly_axis, monthly) = uneven_series(Frequency::Monthly);
    let run = |axis, values: &[f64], mode, options| {
        adjust_extended(
            values,
            axis,
            0..values.len(),
            mode,
            super::PreparedOptions::new(options, axis, values.len()).unwrap(),
        )
        .unwrap()
    };

    let trend_only = run(
        quarterly_axis,
        &quarterly,
        x11::Mode::PseudoAdditive,
        x11::Options::new().with_decomposition(Decomposition::TrendOnly),
    );
    assert_eq!(
        values_at(&trend_only, 5),
        [
            95.53492063492064,
            110.2124467951161,
            1.0,
            0.40112058376274273
        ]
    );

    let robust = run(
        monthly_axis,
        &monthly,
        x11::Mode::LogAdditive,
        x11::Options::new()
            .with_extreme_estimator(Estimator::TauAdjustedMedianAbsoluteDeviation)
            .with_extreme_scale(Scale::RollingFiveYear),
    );
    assert_eq!(
        values_at(&robust, 40),
        [
            113.83285629304376,
            109.762716779696,
            1.014408490073813,
            0.36301413247981884
        ]
    );

    let groups = Groups::Quarterly([
        Group::Selected,
        Group::Complement,
        Group::Selected,
        Group::Complement,
    ]);
    let grouped = run(
        quarterly_axis,
        &quarterly,
        x11::Mode::Additive,
        x11::Options::new().with_extreme_scale(Scale::Grouped(groups)),
    );
    assert_eq!(
        values_at(&grouped, 5),
        [
            98.0947291705153,
            107.93841331024971,
            -2.5598085355946556,
            0.5238409224178775
        ]
    );

    let options = x11::Options::new().with_extreme_scale(Scale::Automatic);
    let (with_easter, easter) = super::adjust_easter(
        &monthly,
        monthly_axis,
        x11::Mode::Multiplicative,
        super::PreparedOptions::new(options, monthly_axis, monthly.len()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        values_at(&with_easter, 39),
        [
            116.38304906215538,
            104.63567849050752,
            0.9292113010949642,
            0.46269138721626524
        ]
    );
    let selection = with_easter.calendar_sigma_selection.unwrap();
    assert_eq!(
        (selection.initial_si(), selection.preliminary_irregular()),
        (
            crate::x11::AutomaticExtremeScale::ByCalendarPeriod,
            crate::x11::AutomaticExtremeScale::ByCalendarPeriod
        )
    );
    assert_eq!(
        easter.factors.iter().copied().fold(f64::NAN, f64::max),
        1.0068298530750361
    );
}
