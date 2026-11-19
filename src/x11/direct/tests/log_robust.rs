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
use crate::{
    Error, x11,
    x11::{ExtremeEstimator as Estimator, ExtremeScale as Scale, SigmaGroup as Group},
};

fn context(estimator: Estimator, axis: TimeAxis, stage: Stage) -> RobustContext {
    RobustContext::new(estimator, axis, axis)
        .unwrap()
        .unwrap()
        .at(stage)
}

fn policy(scale: Scale, context: RobustContext) -> ExtremeWeightPolicy {
    ExtremeWeightPolicy::from(scale).with_context(Some(context))
}

fn relative(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        ((actual - expected) / expected).abs() <= 6e-15,
        "{actual:?} vs {expected:?}"
    );
}

#[test]
fn log_scale_checks_only_the_values_its_pass_keeps() {
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    for estimator in [
        Estimator::LogMedianAbsoluteDeviation,
        Estimator::TauAdjustedLogMedianAbsoluteDeviation,
    ] {
        let ctx = context(estimator, axis, Stage::IntermediateIrregular);
        let irregular = [0.0, 1.1, f64::NAN, 1.2];
        let weights = [1.0, -1.0, 0.0, 1.0];
        let sample = ScaleSample {
            first: 0,
            last: 3,
            stride: 1,
            pass: Pass::Reweighted,
            center: 1.0,
            group: Some(Group::Selected),
        };
        let scale = ctx.scale(&irregular, &weights, [1, 2, 3], sample).unwrap();
        assert!(scale.is_finite() && scale > 0.0);
        for (index, reason) in [
            (0, Reason::NonPositiveLogInput),
            (2, Reason::NonFiniteDeviation),
        ] {
            let error = ctx
                .scale(
                    &irregular,
                    &weights,
                    [index],
                    ScaleSample {
                        first: index,
                        last: index,
                        pass: Pass::Initial,
                        group: Some(Group::Complement),
                        ..sample
                    },
                )
                .unwrap_err();
            assert_eq!(
                error,
                Error::from(NumericalDetail::X11ExtremeScale {
                    estimator,
                    reason,
                    stage: Stage::IntermediateIrregular,
                    pass: Pass::Initial,
                    first: axis.period_at(index).unwrap(),
                    last: axis.period_at(index).unwrap(),
                    group: Some(Group::Complement),
                })
            );
        }
        assert_eq!(
            ctx.scale(&[0.0], &[-1.0], [0], ScaleSample { last: 0, ..sample }),
            Err(Error::from(NumericalDetail::X11ExtremeScale {
                estimator,
                reason: Reason::NonPositiveLogInput,
                stage: Stage::IntermediateIrregular,
                pass: Pass::Reweighted,
                first: axis.period_at(0).unwrap(),
                last: axis.period_at(0).unwrap(),
                group: Some(Group::Selected),
            }))
        );
        let error = ctx
            .scale(
                &[0.0, 1.1],
                &[0.0, 1.0],
                0..2,
                ScaleSample {
                    last: 1,
                    pass: Pass::Initial,
                    group: None,
                    ..sample
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Numerical(NumericalFailure(NumericalDetail::X11ExtremeScale {
                reason: Reason::NonPositiveLogInput,
                pass: Pass::Initial,
                ..
            }))
        ));
        let error = ctx
            .scale(
                &[0.0, 1.1],
                &[0.0, 1.0],
                0..2,
                ScaleSample {
                    last: 1,
                    group: None,
                    ..sample
                },
            )
            .unwrap();
        assert!(error.is_finite());
    }
}

#[test]
fn log_scales_take_positive_values_in_additive_mode() {
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    for estimator in [
        Estimator::LogMedianAbsoluteDeviation,
        Estimator::TauAdjustedLogMedianAbsoluteDeviation,
    ] {
        let weights = calendar_extreme_weights(
            &[1.1; 32],
            ExtremeSpans::new(0..32, 0..32),
            4,
            1,
            x11::Mode::Additive,
            x11::SigmaLimits::new(1000.0, 2000.0).unwrap(),
            policy(
                Scale::ByCalendarPeriod,
                context(estimator, axis, Stage::PreliminaryIrregular),
            ),
        )
        .unwrap();
        assert_eq!(weights, [1.0; 32]);
    }
}

#[test]
fn log_errors_name_the_stage_pass_group_and_dates() {
    let axis = TimeAxis::monthly(1980, 7).unwrap();
    let requested = TimeAxis::monthly(1980, 2).unwrap();
    let estimator = Estimator::LogMedianAbsoluteDeviation;
    for stage in [
        Stage::InitialSeasonalIrregular,
        Stage::RefinedSeasonalIrregular,
        Stage::PreliminaryIrregular,
        Stage::IntermediateIrregular,
    ] {
        for pass in [Pass::Initial, Pass::Reweighted] {
            let ctx = RobustContext::new(estimator, requested, axis)
                .unwrap()
                .unwrap()
                .at(stage);
            let error = ctx
                .scale(
                    &[1.1, 0.0],
                    &[1.0, -1.0],
                    [1],
                    ScaleSample {
                        first: 0,
                        last: 1,
                        stride: 1,
                        pass,
                        center: 0.0,
                        group: Some(Group::Complement),
                    },
                )
                .unwrap_err();
            assert_eq!(
                error,
                Error::from(NumericalDetail::X11ExtremeScale {
                    estimator,
                    reason: Reason::NonPositiveLogInput,
                    stage,
                    pass,
                    first: axis.period_at(0).unwrap(),
                    last: axis.period_at(1).unwrap(),
                    group: Some(Group::Complement),
                })
            );
        }
    }
}

#[test]
fn log_grouped_scale_pools_an_empty_selection_and_reports_an_empty_complement() {
    let axis = TimeAxis::quarterly(1980, 3).unwrap();
    let values = vec![1.1; 32];
    let ctx = context(
        Estimator::LogMedianAbsoluteDeviation,
        axis,
        Stage::PreliminaryIrregular,
    );
    let pooled = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Multiplicative,
        x11::Options::new().sigma_limits(),
        policy(Scale::RollingFiveYear, ctx),
    )
    .unwrap();
    let empty_selected = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Multiplicative,
        x11::Options::new().sigma_limits(),
        policy(
            Scale::Grouped(x11::SigmaGroups::Quarterly([Group::Complement; 4])),
            ctx,
        ),
    )
    .unwrap();
    assert_eq!(empty_selected, pooled);
    let error = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..32, 0..32),
        4,
        3,
        x11::Mode::Multiplicative,
        x11::Options::new().sigma_limits(),
        policy(
            Scale::Grouped(x11::SigmaGroups::Quarterly([Group::Selected; 4])),
            ctx,
        ),
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::from(NumericalDetail::X11ExtremeScale {
            estimator: Estimator::LogMedianAbsoluteDeviation,
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
fn both_b_seasonal_sections_report_their_stage_on_a_log_failure() {
    let axis = TimeAxis::monthly(1980, 7).unwrap();
    let estimator = Estimator::LogMedianAbsoluteDeviation;
    let options = x11::Options::new().with_extreme_estimator(estimator);
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
                    filters: &SeasonalFilterSet::uniform(
                        axis.frequency(),
                        ResolvedSeasonalFilter::S3x5,
                    ),
                    short_series_filters: options.short_series_filters(),
                },
                robust: Some(context(estimator, axis, stage)),
            },
        )
        .unwrap_err();
        assert!(
            matches!(error, Error::Numerical(NumericalFailure(NumericalDetail::X11ExtremeScale {
            estimator: Estimator::LogMedianAbsoluteDeviation, reason: Reason::NonPositiveLogInput,
            stage: reported, pass: Pass::Initial, group: None, ..
        })) if reported == stage)
        );
    }
}

#[test]
fn quarterly_log_mad_weight_uses_raw_deviation_after_scale_conversion() {
    // Worked from `sdxtrm.f` and `rho2.f`: the scale is 0.14343813776749645,
    // and the weight grades the raw deviation (0.059923703364536784), not its
    // log (0.4077768498585259).
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    let mut values = vec![1.1; 32];
    values[12] = 1.35;
    let weights = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..32, 0..32),
        4,
        1,
        x11::Mode::Multiplicative,
        x11::SigmaLimits::new(1.5, 2.5).unwrap(),
        policy(
            Scale::ByCalendarPeriod,
            context(
                Estimator::LogMedianAbsoluteDeviation,
                axis,
                Stage::PreliminaryIrregular,
            ),
        ),
    )
    .unwrap();
    relative(weights[12], 0.059923703364536784);
}

#[test]
fn log_tau_divides_by_the_post_loop_index_counted_from_the_requested_year() {
    // Worked from `sdxtrm.f` and `rho2.f`. The second quarter weighted takes
    // its sample [1.1, 1.2, 1.35] from indices 4, 8 and 12: converted MAD
    // 0.2855639047089442 and `rho2` sum 1.5982307626124292. A series
    // starting 1980-Q1 against a requested start of 1979-Q1 has first
    // Fortran index 5, so the loop over 4..13 ends at index 21. Dividing by
    // 21 gives scale 0.07877953394138247 and weight 0.009245660114980847 at
    // index 1. Dividing by 22 would give weight 0, by 17 (counting from the
    // series start) 0.3592447098535314, and by the sample count 3, 1.0.
    let axis = TimeAxis::quarterly(1980, 1).unwrap();
    let requested = TimeAxis::quarterly(1979, 1).unwrap();
    let mut values = vec![1.1; 20];
    values[1] = 1.55;
    values[8] = 1.2;
    values[12] = 1.35;
    let context = RobustContext::new(
        Estimator::TauAdjustedLogMedianAbsoluteDeviation,
        requested,
        axis,
    )
    .unwrap()
    .unwrap()
    .at(Stage::IntermediateIrregular);
    let weights = calendar_extreme_weights(
        &values,
        ExtremeSpans::new(0..20, 4..16),
        4,
        1,
        x11::Mode::Multiplicative,
        x11::SigmaLimits::new(5.0, 7.0).unwrap(),
        policy(Scale::ByCalendarPeriod, context),
    )
    .unwrap();
    relative(weights[1], 0.009245660114980847);
}
