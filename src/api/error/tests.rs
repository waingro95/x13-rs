use super::*;
use crate::error::{InvalidSpec, SpecProblem};
use crate::regarima::ArimaFactor;
use crate::regarima::ResolvedArimaModel;

#[test]
fn numerical_failure_keeps_its_source_out_of_the_message() {
    use std::error::Error as _;
    let inner = Error::InsufficientEffectiveObservations {
        actual: 3,
        minimum: 5,
    };
    let bare = Error::numerical(NumericalOperation::Extension);
    let sourced = Error::numerical_from(NumericalOperation::Extension, inner.clone());
    assert_eq!(sourced.to_string(), bare.to_string());
    assert_eq!(
        sourced.source().map(ToString::to_string),
        Some(inner.to_string())
    );
    assert!(bare.source().is_none());
    assert_eq!(sourced.kind(), ErrorKind::Numerical);
}

#[test]
fn an_internal_error_names_the_check_and_asks_for_a_report() {
    use std::error::Error as _;
    let error = Error::internal("prior adjustment length differs from the series");
    assert_eq!(error.kind(), ErrorKind::Internal);
    assert_eq!(
        error.to_string(),
        "internal error: prior adjustment length differs from the series; this is a bug in x13"
    );
    assert!(error.source().is_none());
}

#[test]
fn messages_name_periods_options_and_components() {
    use crate::regarima::OutlierType;
    use crate::{Warning, x11};
    let period = crate::Period::monthly(2001, 3).unwrap();
    let last = crate::Period::monthly(2005, 12).unwrap();
    let airline = ResolvedArimaModel::new([0, 1, 1], [0, 1, 1]).unwrap();
    for (error, expected) in [
        (
            Error::InvalidOutlierSpan {
                start: last,
                end: period,
            },
            "outlier span 2005-12 through 2001-03 is reversed or extends beyond the model span",
        ),
        (
            Error::PeriodFrequencyMismatch {
                period: crate::Period::quarterly(2001, 2).unwrap(),
                expected: Frequency::Monthly,
            },
            "2001Q2 is a quarterly period, but the input series is monthly",
        ),
        (
            Error::InvalidManualOutlier {
                kind: OutlierType::LevelShift,
                period,
            },
            "specified level shift at 2001-03 is duplicated, outside the series, or a level shift at its first or last observation",
        ),
        (
            Error::InvalidSpec(InvalidSpec(SpecProblem::ArimaOrder {
                factor: ArimaFactor::Seasonal,
                term: ArimaOrderTerm::MovingAverage,
                value: 4,
                maximum: 3,
            })),
            "seasonal MA order 4 exceeds the maximum of 3",
        ),
        (
            Error::MissingSeason {
                frequency: Frequency::Quarterly,
                subperiod: 2,
            },
            "every observation in quarter 2 is missing",
        ),
        (
            Error::CalendarYearOutOfRange {
                feature: CalendarFeature::Easter,
                year: 1850,
                minimum: 1901,
                maximum: Some(2100),
            },
            "Easter effects are not supported in 1850; supported years are 1901 through 2100",
        ),
        (
            Error::SingularRegression {
                pivot: 2,
                variable: RegressionVariable::Outlier {
                    kind: OutlierType::Additive,
                    period,
                },
            },
            "regression design is singular at pivot 2 (additive outlier at 2001-03)",
        ),
        (
            Error::Numerical(NumericalFailure(
                NumericalDetail::ModelSelectionUnavailable { candidate: airline },
            )),
            "automatic model selection unavailable: candidate (0 1 1)(0 1 1) has no valid \
             likelihood score",
        ),
        (
            Error::Numerical(NumericalFailure(
                NumericalDetail::ModelSelectionResidualCheckUnavailable { model: airline },
            )),
            "automatic model selection unavailable: the residual check for (0 1 1)(0 1 1) has \
             no positive centered variance",
        ),
        (
            Error::from(NumericalDetail::X11ExtremeScale {
                estimator: x11::ExtremeEstimator::MedianAbsoluteDeviation,
                reason: X11ExtremeScaleUnavailableReason::EmptySample,
                stage: X11ExtremeStage::PreliminaryIrregular,
                pass: X11ExtremeWeightingPass::Initial,
                first: period,
                last,
                group: Some(x11::SigmaGroup::Complement),
            }),
            "X-11 cannot compute the median absolute deviation extreme-value scale for the \
             preliminary irregular component (initial pass) over 2001-03 through 2005-12 in \
             the complementary calendar group: the sample is empty",
        ),
        (
            Error::InvalidPeriod(PeriodError::InvalidSubperiod {
                frequency: Frequency::Monthly,
                subperiod: 13,
            }),
            "month 13 is not between 1 and 12",
        ),
        (
            Error::TooShort {
                method: crate::Method::Seats,
                actual: 35,
                minimum: 36,
            },
            "SEATS needs at least 36 observations; the series has 35",
        ),
        (
            Error::SeatsTooLong {
                actual: 781,
                maximum: 780,
            },
            "SEATS accepts at most 780 observations; the series has 781",
        ),
        (
            Error::HendersonLongerThanSeries {
                terms: 101,
                count: 36,
            },
            "the 101-term Henderson filter is longer than the 36 values X-11 filters, \
             forecasts and backcasts included",
        ),
    ] {
        assert_eq!(error.to_string(), expected);
    }
    assert_eq!(
        Warning::SeasonalFilterReplaced {
            requested: x11::SeasonalFilter::S3x9,
            replacement: x11::SeasonalFilter::S3x5,
        }
        .to_string(),
        "the 3x9 seasonal filter was replaced by the 3x5 filter because the history is too \
         short"
    );
    assert_eq!(
        Warning::ManualOutlierOutsideModelSpan {
            kind: OutlierType::Additive,
            period,
        }
        .to_string(),
        "the manual additive outlier at 2001-03 was dropped because it is outside the model \
         span"
    );
    assert_eq!(
        Warning::ManualOutlierOutsideModelSpan {
            kind: OutlierType::LevelShift,
            period,
        }
        .to_string(),
        "the manual level shift at 2001-03 was dropped because the model span must contain \
         it and an earlier observation"
    );
    assert_eq!(
        Warning::NonPositiveTrendReplaced {
            stage: x11::TrendStage::FinalTrendCycle,
            replacements: 2,
        }
        .to_string(),
        "multiplicative X-11 replaced 2 nonpositive final trend-cycle estimates"
    );
}

#[test]
fn unsupported_kind_is_only_for_unsupported_configurations() {
    assert_eq!(
        Error::from(UnsupportedDetail::Configuration(
            UnsupportedConfiguration::ManualOutliers
        ))
        .kind(),
        ErrorKind::Unsupported
    );
    assert_ne!(Error::EmptyInput.kind(), ErrorKind::Unsupported);
    assert_ne!(
        Error::numerical(NumericalOperation::Estimation).kind(),
        ErrorKind::Unsupported
    );
    // An automatic selection that cannot score a candidate is a fit
    // failure, not an unsupported request.
    let airline = ResolvedArimaModel::new([0, 1, 1], [0, 1, 1]).unwrap();
    for error in [
        Error::Numerical(NumericalFailure(
            NumericalDetail::ModelSelectionResidualCheckUnavailable { model: airline },
        )),
        Error::Numerical(NumericalFailure(
            NumericalDetail::ModelSelectionConstantTestUnavailable {
                model: airline,
                test: ModelSelectionConstantTest::Initial,
            },
        )),
        Error::Numerical(NumericalFailure(
            NumericalDetail::ModelSelectionUnavailable { candidate: airline },
        )),
    ] {
        assert_eq!(error.kind(), ErrorKind::Numerical, "{error:?}");
    }
}

/// Too few observations for a pipeline is `TooShort` whatever the request;
/// a series the pipeline cannot take for another reason, or that only a
/// shorter filter fits, is `UnsuitableData`.
#[test]
fn length_errors_are_too_short_or_unsuitable_data() {
    use crate::engine::seats::seats_length_range;
    use crate::{
        ArimaModel, ArimaOrder, Frequency, Period, RegArimaSpec, Series, Spec, adjust, x11,
    };
    let start = Period::monthly(2000, 1).unwrap();
    let seats_limits = seats_length_range(Frequency::Monthly, 1);
    let (seats_minimum, seats_maximum) = (*seats_limits.start(), *seats_limits.end());
    let values: Vec<f64> = (0..=seats_maximum)
        .map(|i| {
            let season = 8.0 * (i as f64 * std::f64::consts::PI / 6.0).sin();
            let noise = ((i * 37) % 11) as f64 / 11.0 - 0.5;
            100.0 + 0.5 * i as f64 + season + noise
        })
        .collect();
    let run = |count: usize, spec: &Spec| adjust(Series::new(start, &values[..count]), spec);
    let airline = ArimaModel::new(
        ArimaOrder::new(0, 1, 1).unwrap(),
        ArimaOrder::new(0, 1, 1).unwrap(),
    )
    .unwrap();
    let seats = Spec::seats(RegArimaSpec::new(airline));
    let direct = Spec::x11(x11::Options::new());
    let regarima = Spec::x11_regarima(
        RegArimaSpec::new(airline),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let with_henderson = |spec: &Spec, terms: u8| {
        let options = spec
            .x11_options()
            .unwrap()
            .with_trend_filter(x11::TrendFilter::Henderson(
                x11::HendersonLength::new(terms).unwrap(),
            ));
        match spec.regarima() {
            Some(regarima) => Spec::x11_regarima(regarima.clone(), options),
            None => Spec::x11(options),
        }
    };

    // X-11 needs three years whatever the request: 35 months is short, 36
    // is not a length error. A long Henderson filter does not hide a
    // short series, since a shorter filter would not fix it.
    for (name, spec) in [("direct", &direct), ("regarima", &regarima)] {
        for count in [24, 35] {
            for spec in [spec.clone(), with_henderson(spec, 101)] {
                let error = run(count, &spec).unwrap_err();
                assert_eq!(
                    error,
                    Error::TooShort {
                        method: crate::Method::X11,
                        actual: count,
                        minimum: 36
                    },
                    "{name} {count}"
                );
                assert_eq!(error.kind(), ErrorKind::TooShort);
            }
        }
        if let Err(error) = run(36, spec) {
            panic!("{name}: 36 months is not a length error: {error:?}");
        }
    }

    // SEATS has a floor and a ceiling, and they mean different things.
    let error = run(seats_minimum - 1, &seats).unwrap_err();
    assert_eq!(
        error,
        Error::TooShort {
            method: crate::Method::Seats,
            actual: seats_minimum - 1,
            minimum: seats_minimum
        }
    );
    assert_eq!(error.kind(), ErrorKind::TooShort);
    let error = run(seats_maximum + 1, &seats).unwrap_err();
    assert_eq!(
        error,
        Error::SeatsTooLong {
            actual: seats_maximum + 1,
            maximum: seats_maximum
        }
    );
    assert_eq!(error.kind(), ErrorKind::UnsuitableData);

    // A Henderson filter longer than a series of three years or more: a
    // shorter one fits. The filter may be as long as the series.
    let error = run(36, &with_henderson(&direct, 101)).unwrap_err();
    assert_eq!(
        error,
        Error::HendersonLongerThanSeries {
            terms: 101,
            count: 36
        }
    );
    assert_eq!(error.kind(), ErrorKind::UnsuitableData);
    let error = run(36, &with_henderson(&regarima, 101)).unwrap_err();
    assert!(
        matches!(error, Error::HendersonLongerThanSeries { terms: 101, .. }),
        "{error:?}"
    );
    assert_eq!(error.kind(), ErrorKind::UnsuitableData);
    let error = run(36, &with_henderson(&direct, 37)).unwrap_err();
    assert_eq!(
        error,
        Error::HendersonLongerThanSeries {
            terms: 37,
            count: 36
        }
    );
    if let Err(error) = run(37, &with_henderson(&direct, 37)) {
        panic!("a filter as long as the series is not a length error: {error:?}");
    }
}

#[test]
fn each_kind_has_a_representative_error() {
    let period = crate::Period::monthly(2000, 1).unwrap();
    for (error, kind) in [
        (Error::NonFiniteValue { index: 3 }, ErrorKind::InvalidInput),
        (
            Error::InvalidModelSpan {
                start: period,
                end: period,
            },
            ErrorKind::InvalidSpec,
        ),
        (
            Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::AutomaticModel,
            )),
            ErrorKind::Unsupported,
        ),
        (
            Error::TooShort {
                method: crate::Method::Seats,
                actual: 1,
                minimum: 36,
            },
            ErrorKind::TooShort,
        ),
        (
            Error::NonPositiveValue { index: 0 },
            ErrorKind::UnsuitableData,
        ),
        (
            Error::EstimationDidNotConverge { iterations: 1500 },
            ErrorKind::Numerical,
        ),
        (
            Error::internal("regressor column length differs from the response"),
            ErrorKind::Internal,
        ),
    ] {
        assert_eq!(error.kind(), kind, "{error:?}");
    }
    // An inadmissible decomposition belongs to the fitted model.
    let inadmissible = Error::seats(
        crate::seats::Stage::Import,
        crate::seats::Failure::DegenerateModel {
            seasonal: false,
            ar: vec![0.5],
            ma: vec![0.5],
        },
    );
    assert_eq!(inadmissible.kind(), ErrorKind::Numerical);
}

#[test]
fn model_changes_report_the_fitted_and_proposed_orders() {
    use crate::seats::Orders;
    use crate::seats::failure::ModelChange;
    let error = Error::seats(
        crate::seats::Stage::Refit,
        crate::seats::Failure::ModelChange(ModelChange {
            current: Orders::new([2, 1, 0], [0, 1, 1]),
            proposed: Orders::new([1, 2, 0], [0, 1, 1]),
        }),
    );
    assert_eq!(
        error,
        Error::from(UnsupportedDetail::SeatsModelChange {
            current: ResolvedArimaModel::new([2, 1, 0], [0, 1, 1]).unwrap(),
            proposed: ResolvedArimaModel::new([1, 2, 0], [0, 1, 1]).unwrap(),
        })
    );
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(
        error.to_string(),
        "SEATS would re-estimate the model (2 1 0)(0 1 1) as (1 2 0)(0 1 1), as Build 62 does \
         when a fitted AR root reaches the unit circle or a seasonal AR factor cancels the \
         seasonal MA; this crate does not follow that change"
    );
}

#[test]
fn payloads_are_errors_and_a_cause_is_reported_once() {
    use std::error::Error as _;
    fn is_error<T: std::error::Error>() {}
    is_error::<InvalidSpec>();
    is_error::<Unsupported>();
    is_error::<InternalError>();

    // A cause that `source` returns stays out of the message.
    let cause = crate::PeriodError::PeriodOverflow { index: 1 }.to_string();
    let dated = Error::UserRegressorDateRange {
        kind: crate::regarima::UserRegressorType::User,
        start: crate::Period::monthly(2001, 3).unwrap(),
        count: 2,
        source: crate::PeriodError::PeriodOverflow { index: 1 },
    };
    assert!(!dated.to_string().contains(&cause));
    assert_eq!(dated.source().map(ToString::to_string), Some(cause));

    // A cause the message already gives is not a source as well.
    let unavailable = Error::from(NumericalDetail::ExactLikelihoodUnavailable {
        failure: ExactLikelihoodFailure::ArmaAutocovariance,
    });
    assert!(
        unavailable
            .to_string()
            .starts_with("exact ARMA likelihood unavailable: ")
    );
    assert!(unavailable.source().is_none());
}
