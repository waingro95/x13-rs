use crate::error::{
    NumericalDetail, NumericalFailure, X11ExtremeScaleUnavailableReason as Reason,
    X11ExtremeStage as Stage, X11ExtremeWeightingPass as Pass,
};
use crate::time::TimeAxis;
use crate::x11::direct::extreme_scale::{
    ExtremeScaleState, ExtremeWeightPolicy, RobustContext, ScaleSample,
};
use crate::x11::direct::extreme_weights::{ExtremeSpans, calendar_extreme_weights};
use crate::x11::direct::seasonal_factors::{
    SeasonalSmoothingRequest, SeasonalWeightRequest, replace_extremes_and_filter,
};
use crate::x11::seasonal::{ResolvedSeasonalFilter, SeasonalFilterSet};
use crate::{Error, PeriodError, x11, x11::SigmaGroup as Group};

/// A unit-stride, zero-centered sample outside any sigma group.
fn window(first: usize, last: usize, pass: Pass) -> ScaleSample {
    ScaleSample {
        first,
        last,
        stride: 1,
        pass,
        center: 0.0,
        group: None,
    }
}

fn context(estimator: x11::ExtremeEstimator, axis: TimeAxis, stage: Stage) -> RobustContext {
    RobustContext::new(estimator, axis, axis)
        .unwrap()
        .unwrap()
        .at(stage)
}

fn robust_policy(scale: x11::ExtremeScale, context: RobustContext) -> ExtremeWeightPolicy {
    ExtremeWeightPolicy::from(scale).with_context(Some(context))
}

#[test]
fn fortran_start_counts_from_the_requested_year_and_errors_use_the_series_dates() {
    let requested = TimeAxis::monthly(1980, 2).unwrap();
    let axis = TimeAxis::monthly(1980, 7).unwrap();
    let first = RobustContext::new(
        x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
        requested,
        axis,
    )
    .unwrap()
    .unwrap();
    assert_eq!(first.fortran_start, 7);
    let older_request = TimeAxis::monthly(1979, 2).unwrap();
    let second = RobustContext::new(
        x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
        older_request,
        axis,
    )
    .unwrap()
    .unwrap();
    assert_eq!(second.fortran_start, 19);
    let backcast = RobustContext::new(
        x11::ExtremeEstimator::MedianAbsoluteDeviation,
        TimeAxis::quarterly(1980, 3).unwrap(),
        TimeAxis::quarterly(1980, 1).unwrap(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(backcast.fortran_start, 1);
    let values = [1.0, 2.0, 3.0];
    let weights = [1.0; 3];
    let scale7 = first
        .scale(&values, &weights, 0..3, window(0, 2, Pass::Initial))
        .unwrap();
    let scale19 = second
        .scale(&values, &weights, 0..3, window(0, 2, Pass::Initial))
        .unwrap();
    assert!(scale7 > scale19);
    assert_eq!(
        first.at(Stage::InitialSeasonalIrregular).scale(
            &[0.0; 3],
            &weights,
            0..3,
            window(0, 2, Pass::Initial),
        ),
        Err(Error::from(NumericalDetail::X11ExtremeScale {
            estimator: x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
            reason: Reason::ZeroTauBase,
            stage: Stage::InitialSeasonalIrregular,
            pass: Pass::Initial,
            first: axis.period_at(0).unwrap(),
            last: axis.period_at(2).unwrap(),
            group: None,
        }))
    );
}

#[test]
fn tau_post_loop_index_comes_from_the_window_bounds_and_stride() {
    let requested = TimeAxis::monthly(1980, 7).unwrap();
    let context = context(
        x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
        requested,
        Stage::PreliminaryIrregular,
    );
    let mut values = [0.0; 13];
    values[0] = 3.0;
    values[12] = 1.0;
    let actual = context
        .scale(
            &values,
            &[1.0; 13],
            (0..13).step_by(12),
            ScaleSample {
                stride: 12,
                ..window(0, 12, Pass::Initial)
            },
        )
        .unwrap();
    let expected = crate::x11::robust::scale(
        [3.0, 1.0],
        0.0,
        crate::x11::robust::RobustEstimator::Tau,
        31,
    )
    .unwrap();
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn empty_selected_group_uses_pooled_sample_for_both_scales() {
    let axis = TimeAxis::quarterly(1980, 3).unwrap();
    let values: Vec<_> = (0..32).map(|index| (index % 5) as f64 + 1.0).collect();
    let options = x11::Options::new();
    let context = context(
        x11::ExtremeEstimator::MedianAbsoluteDeviation,
        axis,
        Stage::PreliminaryIrregular,
    );
    let grouped = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Additive,
        options.sigma_limits(),
        robust_policy(
            x11::ExtremeScale::Grouped(x11::SigmaGroups::Quarterly([Group::Complement; 4])),
            context,
        ),
    )
    .unwrap();
    let pooled = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Additive,
        options.sigma_limits(),
        robust_policy(x11::ExtremeScale::RollingFiveYear, context),
    )
    .unwrap();
    assert_eq!(grouped, pooled);
}

#[test]
fn robust_calendar_period_sample_before_the_estimation_span_starts_at_its_first_value() {
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    let values = [1.0, 2.0, 4.0, 8.0].repeat(5);
    let weights = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..20, 4..16),
        4,
        1,
        x11::Mode::Additive,
        x11::Options::new().sigma_limits(),
        robust_policy(
            x11::ExtremeScale::ByCalendarPeriod,
            context(
                x11::ExtremeEstimator::MedianAbsoluteDeviation,
                axis,
                Stage::IntermediateIrregular,
            ),
        ),
    )
    .unwrap();
    assert_eq!(weights, [1.0, 1.0, 0.0, 0.0].repeat(5));
}

#[test]
fn tau_calendar_period_sample_before_the_estimation_span_keeps_its_post_loop_index() {
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    let weights = calendar_extreme_weights(
        &[1.0; 20],
        ExtremeSpans::new(0..20, 4..16),
        4,
        1,
        x11::Mode::Additive,
        x11::SigmaLimits::new(2.45, 2.47).unwrap(),
        robust_policy(
            x11::ExtremeScale::ByCalendarPeriod,
            context(
                x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
                axis,
                Stage::IntermediateIrregular,
            ),
        ),
    )
    .unwrap();
    assert_eq!(weights, [1.0; 20]);
}

#[test]
fn series_before_the_requested_year_and_period_overflow_are_distinct_errors() {
    let monthly = TimeAxis::monthly(1980, 7).unwrap();
    let estimator = x11::ExtremeEstimator::MedianAbsoluteDeviation;
    assert!(matches!(
        RobustContext::new(estimator, monthly, TimeAxis::monthly(1979, 12).unwrap()),
        Err(Error::Internal(_))
    ));
    let last_axis = TimeAxis::monthly(i32::MAX, 12).unwrap();
    let context = context(estimator, last_axis, Stage::RefinedSeasonalIrregular);
    assert_eq!(
        context.scale(&[1.0, 2.0], &[1.0; 2], 0..2, window(0, 1, Pass::Initial)),
        Err(Error::InvalidPeriod(PeriodError::PeriodOverflow {
            index: 1
        }))
    );
}

#[test]
fn grouped_weighting_reports_an_empty_group() {
    let axis = TimeAxis::quarterly(1980, 3).unwrap();
    let estimator = x11::ExtremeEstimator::MedianAbsoluteDeviation;
    let irregular = [1.0, 2.0, 3.0, 4.0].repeat(8);
    let error = calendar_extreme_weights(
        &irregular,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Additive,
        x11::SigmaLimits::new(1.5, 2.5).unwrap(),
        robust_policy(
            x11::ExtremeScale::Grouped(x11::SigmaGroups::Quarterly([Group::Selected; 4])),
            context(estimator, axis, Stage::PreliminaryIrregular),
        ),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::from(NumericalDetail::X11ExtremeScale {
            estimator,
            reason: Reason::EmptySample,
            stage: Stage::PreliminaryIrregular,
            pass: Pass::Initial,
            first: axis.period_at(0).unwrap(),
            last: axis.period_at(31).unwrap(),
            group: Some(Group::Complement),
        })
    );
}

#[test]
fn exact_zero_weights_are_omitted_only_on_reweighted_pass() {
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    let estimator = x11::ExtremeEstimator::MedianAbsoluteDeviation;
    let irregular = [1.0, 2.0, 3.0, 4.0].repeat(5);
    let error = calendar_extreme_weights(
        &irregular,
        ExtremeSpans::new(0..20, 0..20),
        4,
        1,
        x11::Mode::Additive,
        x11::SigmaLimits::new(0.1, 0.1).unwrap(),
        robust_policy(
            x11::ExtremeScale::ByCalendarPeriod,
            context(estimator, axis, Stage::IntermediateIrregular),
        ),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Numerical(NumericalFailure(NumericalDetail::X11ExtremeScale {
            reason: Reason::EmptySample,
            stage: Stage::IntermediateIrregular,
            pass: Pass::Reweighted,
            ..
        }))
    ));
}

#[test]
fn rejection_marks_stay_in_the_reweighted_sample() {
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    let context = context(
        x11::ExtremeEstimator::MedianAbsoluteDeviation,
        axis,
        Stage::IntermediateIrregular,
    );
    let scale = context
        .scale(
            &[2.0, 100.0, 4.0],
            &[-1.0, 0.0, 1.0],
            0..3,
            window(0, 2, Pass::Reweighted),
        )
        .unwrap();
    assert_eq!(scale, 3.0 / 0.6745);
}

#[test]
fn both_b_seasonal_sections_report_their_stage_on_a_tau_failure() {
    let axis = TimeAxis::monthly(1980, 7).unwrap();
    let estimator = x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation;
    let options = x11::Options::new().with_extreme_estimator(estimator);
    let filters = SeasonalFilterSet::uniform(axis.frequency(), ResolvedSeasonalFilter::S3x5);
    for stage in [
        Stage::InitialSeasonalIrregular,
        Stage::RefinedSeasonalIrregular,
    ] {
        let mut si = vec![0.0; 72];
        let mut state = ExtremeScaleState::new(options);
        let error = replace_extremes_and_filter(
            &mut si,
            ExtremeSpans::new(0..72, 0..72),
            x11::Mode::Additive,
            &mut state,
            SeasonalWeightRequest {
                dates: axis.span(72).unwrap(),
                smoothing: SeasonalSmoothingRequest {
                    filters: &filters,
                    short_series_filters: options.short_series_filters(),
                },
                robust: Some(context(estimator, axis, stage)),
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Numerical(NumericalFailure(NumericalDetail::X11ExtremeScale {
                estimator: x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
                reason: Reason::ZeroTauBase,
                stage: reported,
                pass: Pass::Initial,
                group: None,
                ..
            })) if reported == stage
        ));
    }
}

#[test]
fn tau_scale_distinguishes_a_tiny_base_from_nonfinite_failures() {
    let axis = TimeAxis::quarterly(1980, 3).unwrap();
    let tau = context(
        x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
        axis,
        Stage::PreliminaryIrregular,
    );
    assert!(
        tau.scale(
            &[1e-20, 1e-20, 2e-20],
            &[1.0; 3],
            0..3,
            window(0, 2, Pass::Initial),
        )
        .unwrap()
        .is_finite()
    );
    let reason = |result: Result<f64, Error>| match result.unwrap_err() {
        Error::Numerical(NumericalFailure(NumericalDetail::X11ExtremeScale { reason, .. })) => {
            reason
        }
        other => panic!("unexpected error: {other}"),
    };
    assert_eq!(
        reason(tau.scale(&[f64::INFINITY], &[1.0], 0..1, window(0, 0, Pass::Initial))),
        Reason::NonFiniteDeviation
    );
    assert_eq!(
        reason(tau.scale(&[f64::MAX; 3], &[1.0; 3], 0..3, window(0, 2, Pass::Initial))),
        Reason::NonFiniteScale
    );
}
