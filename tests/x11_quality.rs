use x13::diagnostics::UnavailableKind;
use x13::{Period, Series, Spec, x11};

#[test]
fn constant_series_quality_names_the_constant_inputs() {
    let start = Period::monthly(1980, 7).unwrap();
    let actual = x13::adjust(
        Series::new(start, &[100.0; 120]),
        &Spec::x11_regarima(
            x13::RegArimaSpec::automatic_model().with_forecast(
                x13::regarima::ForecastSpec::new(x13::regarima::ForecastHorizon::Periods(12), 12)
                    .unwrap(),
            ),
            x11::Options::new().with_mode(x11::Mode::PseudoAdditive),
        ),
    )
    .unwrap();
    let unavailable = actual
        .diagnostics()
        .x11()
        .unwrap()
        .quality()
        .expect_err("quality measures need a nonconstant series");
    assert_eq!(unavailable.kind(), UnavailableKind::NotApplicable);
    assert_eq!(
        unavailable.to_string(),
        "the M and Q measures are unavailable: constant series: modified original series, \
         irregular, original series, detrended modified original series"
    );
}

#[test]
fn quality_measures_are_finite_and_in_range() {
    let values = (0..180)
        .map(|i| {
            100.0
                + i as f64 * 0.3
                + [
                    3.0, -2.0, 1.0, -4.0, 2.0, 7.0, -1.0, 4.0, 2.0, -3.0, 1.0, 6.0,
                ][i % 12]
                + ((i * 17) % 11) as f64 * 0.17
        })
        .collect::<Vec<_>>();
    let actual = x13::adjust(
        Series::new(Period::monthly(2000, 1).unwrap(), &values),
        &Spec::x11(
            x13::x11::Options::new()
                .with_mode(x11::Mode::Additive)
                .with_seasonal_filter(x13::x11::SeasonalFilter::S3x5),
        ),
    )
    .unwrap();
    let value = actual
        .diagnostics()
        .x11()
        .unwrap()
        .quality()
        .expect("quality measures should be available");
    for v in [
        value.m1(),
        value.m2(),
        value.m3(),
        value.m4(),
        value.m5(),
        value.m6().expect("M6 exists for a seasonal adjustment"),
        value.m7(),
        value.q(),
        value.q_without_m2(),
    ] {
        assert!(v.is_finite() && (0.0..=3.0).contains(&v));
    }
    assert!(value.m6_included_in_q());
    assert!(value.seasonal_evolution().is_some());
    assert_eq!(value.observed_span().count(), 180);
}
