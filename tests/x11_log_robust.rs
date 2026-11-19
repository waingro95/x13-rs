use x13::{ErrorKind, Period, Series, Spec, adjust, x11, x11::ExtremeEstimator as Estimator};

#[test]
fn log_estimators_change_the_adjustment() {
    for start in [
        Period::monthly(1980, 7).unwrap(),
        Period::quarterly(1980, 3).unwrap(),
    ] {
        let period = usize::from(start.frequency().periods_per_year());
        let values: Vec<_> = (0..period * 10)
            .map(|index| {
                100.0
                    + index as f64 * 0.13
                    + (index % period) as f64 * 0.8
                    + if index == period * 5 { 28.0 } else { 0.0 }
            })
            .collect();
        let run = |estimator| {
            adjust(
                Series::new(start, &values),
                &Spec::x11(
                    x11::Options::new()
                        .with_mode(x11::Mode::Multiplicative)
                        .with_extreme_estimator(estimator),
                ),
            )
            .unwrap()
        };
        let standard = run(Estimator::StandardDeviation);
        for estimator in [
            Estimator::LogMedianAbsoluteDeviation,
            Estimator::TauAdjustedLogMedianAbsoluteDeviation,
        ] {
            assert_ne!(
                run(estimator).seasonally_adjusted(),
                standard.seasonally_adjusted(),
                "{estimator:?} from {start:?}"
            );
        }
    }
}

#[test]
fn tau_adjusted_log_estimator_rejects_a_constant_series() {
    let error = adjust(
        Series::new(Period::monthly(1980, 7).unwrap(), &[100.0; 72]),
        &Spec::x11(
            x11::Options::new()
                .with_mode(x11::Mode::Multiplicative)
                .with_extreme_estimator(Estimator::TauAdjustedLogMedianAbsoluteDeviation),
        ),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Numerical);
    let message = error.to_string();
    assert!(
        message.starts_with(
            "X-11 cannot compute the tau-adjusted log median absolute deviation extreme-value \
             scale"
        ) && message.ends_with(": the tau base is zero"),
        "{message}"
    );
}
