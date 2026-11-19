use super::*;
use crate::diagnostics::UnavailableKind;

fn axis() -> TimeAxis {
    TimeAxis::monthly(2000, 1).unwrap()
}

/// `super::compute` on an axis that dates every observation.
fn compute(
    axis: TimeAxis,
    observed_count: usize,
    residuals: &[f64],
    converged: bool,
    model_variance: f64,
) -> Result<Peaks, Unavailable> {
    super::compute(axis, observed_count, residuals, converged, model_variance).unwrap()
}

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (i as f64 * 0.72).sin() + (i as f64 * 0.11).cos() * 0.3)
        .collect()
}

#[test]
fn rejects_an_axis_that_cannot_date_the_observations() {
    let axis = TimeAxis::monthly(i32::MAX, 12).unwrap();
    assert_eq!(
        super::compute(axis, 2, &signal(2), true, 1.0),
        Err(Error::InvalidPeriod(crate::PeriodError::PeriodOverflow {
            index: 1
        }))
    );
}

/// The sample is the latest residuals: no more than the observations, and
/// no more than eight years.
#[test]
fn sample_is_the_latest_residuals_up_to_eight_years() {
    for (observed, residual_count, count, first) in [
        (72, 70, 70, 2),
        (72, 72, 72, 0),
        (72, 74, 72, 0),
        (97, 99, 96, 1),
    ] {
        let residuals = signal(residual_count);
        let record = compute(axis(), observed, &residuals, true, 1.0);
        let Ok(peaks) = record else {
            panic!("{record:?}");
        };
        assert_eq!(
            peaks.sample,
            residual_sample(
                axis().period_at(first).unwrap(),
                axis().period_at(observed - 1).unwrap(),
                count
            )
        );
        assert_eq!(
            window(observed, &residuals),
            &residuals[residual_count - count..]
        );
    }
}

#[test]
fn inapplicable_and_unavailable_spectra_say_why() {
    let residuals = signal(97);
    for (axis, converged, variance, expected) in [
        (
            TimeAxis::quarterly(2000, 1).unwrap(),
            false,
            f64::NAN,
            NotApplicable::QuarterlyFrequency,
        ),
        (axis(), false, 0.0, NotApplicable::ModelNotConverged),
        (axis(), true, 0.0, NotApplicable::ZeroModelVariance),
    ] {
        let record = compute(axis, 97, &residuals, converged, variance);
        assert_eq!(
            record,
            Err(Unavailable(UnavailableDetail::SpectrumNotApplicable {
                series: Series::Residuals,
                reason: expected,
            }))
        );
        assert_eq!(
            record.as_ref().unwrap_err().kind(),
            UnavailableKind::NotApplicable
        );
        assert!(warning(&record).is_none());
    }
    for variance in [-1.0, f64::NAN, f64::INFINITY] {
        let record = compute(axis(), 97, &residuals, true, variance);
        assert!(matches!(
            record,
            Err(Unavailable(UnavailableDetail::Spectrum {
                series: Series::Residuals,
                reason: Reason::InvalidModelVariance,
                ..
            }))
        ));
        assert!(warning(&record).is_none());
    }
    for (values, expected, count) in [
        (
            vec![],
            Reason::InsufficientInput {
                actual: 0,
                minimum: 2,
            },
            None,
        ),
        (
            vec![1.0],
            Reason::InsufficientInput {
                actual: 1,
                minimum: 2,
            },
            Some(1),
        ),
        (vec![1.0; 72], Reason::ZeroCovariance, Some(72)),
        (
            vec![f64::NAN; 72],
            Reason::NonFiniteInput {
                period: axis().start(),
            },
            Some(72),
        ),
        (
            vec![f64::MAX; 72],
            Reason::NonFiniteArithmetic {
                stage: Stage::MeanDeletion,
            },
            Some(72),
        ),
    ] {
        let record = compute(axis(), values.len(), &values, true, 1.0);
        let Err(Unavailable(UnavailableDetail::Spectrum {
            reason, ref sample, ..
        })) = record
        else {
            panic!("{record:?}");
        };
        assert_eq!(reason, expected);
        assert_eq!(sample.as_ref().map(|sample| sample.count), count);
        assert!(warning(&record).is_none());
    }
    // The first nonfinite residual of the window names the period, dated from
    // the window's offset into the observed span.
    let mut values = signal(72);
    values[5] = f64::NAN;
    values[9] = f64::INFINITY;
    let record = compute(axis(), 80, &values, true, 1.0);
    assert_eq!(
        record,
        Err(Unavailable(UnavailableDetail::Spectrum {
            series: Series::Residuals,
            reason: Reason::NonFiniteInput {
                period: axis().period_at(13).unwrap(),
            },
            sample: Some(residual_sample(
                axis().period_at(8).unwrap(),
                axis().period_at(79).unwrap(),
                72
            )),
        }))
    );
    let unavailable = record.unwrap_err();
    assert_eq!(unavailable.kind(), UnavailableKind::Numerical);
    assert_eq!(
        unavailable.to_string(),
        "the spectrum of the regARIMA residuals is unavailable: the value at 2001-02 is not finite"
    );
    // Trading-day peaks need more than five years of residuals.
    for count in [60, 61, 96, 97] {
        let record = compute(axis(), count, &signal(count), true, 1.0);
        let Ok(peaks) = record else {
            panic!("{record:?}");
        };
        assert_eq!(peaks.sample.count, count.min(96));
        assert_eq!(peaks.trading_day_count().is_some(), count > 60);
    }
}

#[test]
fn spectrum_failures_map_to_reasons() {
    use crate::numerics::spectral::{SpectralArithmeticStage as S, SpectralUnavailable as E};
    let core_reason = |error| super::core_reason(error, axis().span(72).unwrap(), 2);
    for (source, expected) in [
        (
            E::InsufficientInput {
                length: 1,
                required: 2,
            },
            Reason::InsufficientInput {
                actual: 1,
                minimum: 2,
            },
        ),
        (E::ZeroCovariance, Reason::ZeroCovariance),
        (
            E::NonfiniteInput { index: 3 },
            Reason::NonFiniteInput {
                period: axis().period_at(5).unwrap(),
            },
        ),
        (
            E::EarlyVarianceCollapse {
                completed_order: 2,
                required_order: 30,
            },
            Reason::EarlyVarianceCollapse {
                completed_order: 2,
                required_order: 30,
            },
        ),
        (
            E::NonpositiveInnovationVariance { order: 3 },
            Reason::NonPositiveInnovationVariance { order: 3 },
        ),
    ] {
        assert_eq!(core_reason(source), expected);
    }
    for (stages, expected) in [
        (vec![S::MeanSum, S::Mean, S::Demeaning], Stage::MeanDeletion),
        (vec![S::CovarianceSum, S::Covariance], Stage::Autocovariance),
        (
            vec![
                S::InitialVariance,
                S::InitialAic,
                S::InitialLagCovariance,
                S::VarianceRatio,
                S::Reflection,
                S::ReflectionSquare,
                S::InnovationVariance,
                S::Aic,
                S::ArCoefficient,
                S::NextLagCovariance,
                S::CovarianceRecurrence,
            ],
            Stage::ArRecursion,
        ),
        (
            vec![
                S::CosineSum,
                S::SineSum,
                S::PowerDenominator,
                S::SpectralPower,
            ],
            Stage::SpectralPower,
        ),
        (vec![S::Decibels], Stage::DecibelConversion),
    ] {
        for stage in stages {
            assert_eq!(
                core_reason(E::NonfiniteArithmetic { stage }),
                Reason::NonFiniteArithmetic { stage: expected }
            );
        }
    }
}

#[test]
fn peak_counts_choose_the_warning() {
    use crate::diagnostics::SpectralPeakKind as K;
    for (seasonal, td, kind) in [
        (0, 0, None),
        (1, 0, Some(K::Seasonal)),
        (0, 1, Some(K::TradingDay)),
        (2, 2, Some(K::Both)),
    ] {
        let record = Ok(Peaks {
            sample: residual_sample(axis().start(), axis().period_at(71).unwrap(), 72),
            ar_order: 30,
            seasonal_count: seasonal,
            trading_day: TradingDay::Tested { count: td },
        });
        assert_eq!(
            warning(&record),
            kind.map(|kind| crate::Warning::ResidualSpectralPeaks { kind })
        );
    }
}

/// Where Build 62 reports another fit's residuals, as automdl does after a
/// fallback, the spectrum and the model result use those.
#[test]
fn spectrum_uses_the_residuals_the_model_search_supplies() {
    use crate::engine::post_fit::{PostFitContext, PostFitWarnings, finish};
    use crate::regarima::{ForecastHorizon, ForecastSpec};
    use crate::{ArimaModel, ArimaOrder, RegArimaSpec, Warning};
    let values: Vec<_> = signal(72).into_iter().map(|v| 100.0 + v).collect();
    let ar1 = ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    let spec = RegArimaSpec::new(ar1)
        .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(0), 0).unwrap())
        .resolve_explicit()
        .unwrap();
    let regression =
        crate::regression::model::RegressionModel::from_spec(spec, axis(), values.len()).unwrap();
    let fit = crate::arima::estimate(
        &values,
        (&values).into(),
        axis(),
        spec,
        regression,
        crate::arima::ArmaStart::Default,
    )
    .unwrap();
    let supplied: Vec<_> = signal(63)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * if i % 3 == 0 { 3.0 } else { 0.5 })
        .collect();
    let expected = compute(
        axis(),
        values.len(),
        &supplied,
        true,
        fit.innovation_variance,
    );
    assert!(expected.is_ok(), "{expected:?}");

    let earlier = Warning::DevianceIncreased;
    let context = PostFitContext {
        series_axis: axis(),
        easter: None,
        mode: crate::x11::Mode::Additive,
        final_options: crate::engine::FinalX11Options::new(
            crate::x11::FinalOutliers::Retain,
            crate::x11::TemporaryChangeAllocation::Irregular,
            crate::x11::HolidayAdjustment::Remove,
        ),
        prepared_options: crate::x11::PreparedOptions::new(
            crate::x11::Options::default(),
            axis(),
            values.len(),
        )
        .unwrap(),
        transform_selection: None,
        restore_missing_prior: true,
        warnings: PostFitWarnings::BeforeFit(vec![earlier.clone()]),
        residual_override: Some(supplied.clone()),
    };
    let result = finish(&values, fit, context).unwrap();
    let model = result.model().unwrap();
    assert_eq!(model.residuals(), supplied);
    assert_eq!(model.residual_spectral(), expected.as_ref());
    assert_eq!(result.warnings()[0], earlier);
    let peak_warnings: Vec<_> = result
        .warnings()
        .iter()
        .filter(|warning| matches!(warning, Warning::ResidualSpectralPeaks { .. }))
        .cloned()
        .collect();
    assert_eq!(peak_warnings, Vec::from_iter(warning(&expected)));
}
