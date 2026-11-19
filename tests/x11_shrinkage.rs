use x13::{ErrorKind, Period, Series, Spec, x11};

#[test]
fn global_shrinkage_changes_seasonal_factors() {
    assert_eq!(
        x11::Options::new().seasonal_shrinkage(),
        x11::SeasonalShrinkage::None
    );
    let values: Vec<_> = (0..120)
        .map(|i| 100.0 + i as f64 * 0.2 + [5.0, -3.0, 2.0, -4.0][i % 4] * (1.0 + i as f64 / 300.0))
        .collect();
    let start = Period::quarterly(2000, 1).unwrap();
    let baseline =
        x13::adjust(Series::new(start, &values), &Spec::x11(x11::Options::new())).unwrap();
    let global = x13::adjust(
        Series::new(start, &values),
        &Spec::x11(x11::Options::new().with_seasonal_shrinkage(x11::SeasonalShrinkage::Global)),
    )
    .unwrap();
    assert_ne!(baseline.seasonal(), global.seasonal());
}

#[test]
fn local_shrinkage_needs_forecasts() {
    let start = Period::monthly(-2000, 7).unwrap();
    let last = start.shift(119).unwrap();
    let options = x11::Options::new().with_seasonal_shrinkage(x11::SeasonalShrinkage::Local);
    for spec in [
        Spec::x11(options),
        Spec::x11_regarima(
            x13::RegArimaSpec::automatic_model().with_forecast(
                x13::regarima::ForecastSpec::new(x13::regarima::ForecastHorizon::OneYear, 5)
                    .unwrap(),
            ),
            options,
        ),
    ] {
        let error = x13::adjust(Series::new(start, &[100.; 120]), &spec).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Numerical);
        assert_eq!(
            error.to_string(),
            format!(
                "local seasonal shrinkage over {start} through {last} failed during \
                 local-likelihood calculation: it needs 5 accepted forecasts, but there are 0"
            )
        );
    }
}

#[test]
fn global_shrinkage_rejects_the_stable_filter() {
    use x13::x11;
    let start = Period::quarterly(1998, 3).unwrap();
    let last = start.shift(79).unwrap();
    let spec = Spec::x11(
        x11::Options::new()
            .with_seasonal_filter(x11::SeasonalFilter::Stable)
            .with_seasonal_shrinkage(x11::SeasonalShrinkage::Global),
    );
    let error = x13::adjust(Series::new(start, &[100.; 80]), &spec).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Numerical);
    assert_eq!(
        error.to_string(),
        format!(
            "global seasonal shrinkage over {start} through {last} failed during \
             filter-weight calculation: shrinkage is not defined after the stable seasonal \
             filter"
        )
    );
    let mut invalid = [100.; 80];
    invalid[7] = f64::NAN;
    let invalid = x13::adjust(Series::new(start, &invalid), &spec);
    assert!(!invalid.is_err_and(|error| error.kind() == ErrorKind::Numerical));
}
