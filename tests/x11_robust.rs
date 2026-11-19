use x13::regarima::{ForecastHorizon, ForecastSpec};
use x13::{ErrorKind, Period, RegArimaSpec, Series, Spec, adjust, x11};

#[test]
fn estimator_option_is_independent_of_scale_and_span() {
    assert_eq!(
        x11::Options::new().extreme_estimator(),
        x11::ExtremeEstimator::StandardDeviation
    );
    let options = x11::Options::new()
        .with_extreme_scale(x11::ExtremeScale::ByCalendarPeriod)
        .with_extreme_estimation_span(x11::ExtremeEstimationSpan::ObservedOnly)
        .with_extreme_estimator(x11::ExtremeEstimator::MedianAbsoluteDeviation);
    assert_eq!(
        options.extreme_estimator(),
        x11::ExtremeEstimator::MedianAbsoluteDeviation
    );
    assert_eq!(options.extreme_scale(), x11::ExtremeScale::ByCalendarPeriod);
    assert_eq!(
        options.extreme_estimation_span(),
        x11::ExtremeEstimationSpan::ObservedOnly
    );
}

#[test]
fn mad_estimators_change_the_adjustment() {
    let values: Vec<_> = (0..120)
        .map(|index| {
            let seasonal = [
                0.0, 3.0, 7.0, 4.0, 1.0, -2.0, -4.0, -5.0, -3.0, 0.0, 2.0, 5.0,
            ];
            100.0 + index as f64 * 0.2 + seasonal[index % 12] + if index == 67 { 35.0 } else { 0.0 }
        })
        .collect();
    let start = Period::monthly(1980, 7).unwrap();
    let standard = adjust(
        Series::new(start, &values),
        &Spec::x11(x11::Options::new().with_mode(x11::Mode::Additive)),
    )
    .unwrap();
    for estimator in [
        x11::ExtremeEstimator::MedianAbsoluteDeviation,
        x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
    ] {
        let options = x11::Options::new().with_extreme_estimator(estimator);
        let robust = adjust(
            Series::new(start, &values),
            &Spec::x11(options.with_mode(x11::Mode::Additive)),
        )
        .unwrap();
        assert_eq!(robust.seasonal().unwrap().len(), values.len());
        assert_ne!(robust.seasonally_adjusted(), standard.seasonally_adjusted());
    }
}

#[test]
fn tau_adjusted_estimator_rejects_a_constant_series() {
    let error = adjust(
        Series::new(Period::monthly(1980, 7).unwrap(), &[100.0; 72]),
        &Spec::x11(
            x11::Options::new()
                .with_mode(x11::Mode::Additive)
                .with_extreme_estimator(x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation),
        ),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Numerical);
    let message = error.to_string();
    assert!(
        message.starts_with(
            "X-11 cannot compute the tau-adjusted median absolute deviation extreme-value scale"
        ) && message.ends_with(": the tau base is zero"),
        "{message}"
    );
}

#[test]
fn constant_series_drops_extensions_before_extreme_weighting() {
    let values = [100.0; 120];
    let start = Period::monthly(1980, 7).unwrap();
    for backcasts in [5, 17] {
        let regarima = RegArimaSpec::automatic_model()
            .with_forecast(ForecastSpec::new(ForecastHorizon::OneYear, backcasts).unwrap());
        let standard = adjust(
            Series::new(start, &values),
            &Spec::x11_regarima(regarima.clone(), x11::Options::new()),
        )
        .unwrap();
        assert!(standard.forecasts().is_none());
        assert!(standard.backcasts().is_none());
        let error = adjust(
            Series::new(start, &values),
            &Spec::x11_regarima(
                regarima,
                x11::Options::new().with_extreme_estimator(
                    x11::ExtremeEstimator::TauAdjustedMedianAbsoluteDeviation,
                ),
            ),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Numerical);
        assert_eq!(
            error.to_string(),
            format!(
                "X-11 cannot compute the tau-adjusted median absolute deviation extreme-value \
                 scale for the initial seasonal-irregular component (initial pass) over {} \
                 through {}: the tau base is zero",
                Period::monthly(1981, 1).unwrap(),
                Period::monthly(1985, 12).unwrap()
            )
        );
    }
}
