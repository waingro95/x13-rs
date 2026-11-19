use x13::regarima::{ForecastHorizon, ForecastSpec};
use x13::{
    ArimaModel, ArimaOrder, Period, RegArimaSpec, Series, Spec, x11,
    x11::ExtremeEstimationSpan as Span,
};

/// `spec` with its extreme values estimated over `span`.
fn with_span(spec: &Spec, span: Span) -> Spec {
    let options = spec
        .x11_options()
        .unwrap()
        .with_extreme_estimation_span(span);
    match spec.regarima() {
        Some(regarima) => Spec::x11_regarima(regarima.clone(), options),
        None => Spec::x11(options),
    }
}

fn assert_same(actual: &x13::Adjustment, expected: &x13::Adjustment) {
    for (a, b) in [
        (actual.seasonal().unwrap(), expected.seasonal().unwrap()),
        (actual.seasonally_adjusted(), expected.seasonally_adjusted()),
        (
            actual.trend_cycle().unwrap(),
            expected.trend_cycle().unwrap(),
        ),
        (actual.irregular(), expected.irregular()),
    ] {
        assert_eq!(a.start(), b.start());
        assert_eq!(a.len(), b.len());
        assert!(
            a.values()
                .iter()
                .zip(b.values())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    assert_eq!(actual.diagnostics(), expected.diagnostics());
    assert_eq!(actual.warnings(), expected.warnings());
}

fn ar1() -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(1, 0, 0).unwrap(),
        ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap()
}

#[test]
fn extreme_span_is_inert_without_extensions() {
    for start in [
        Period::monthly(2000, 7).unwrap(),
        Period::quarterly(2000, 3).unwrap(),
    ] {
        let period = usize::from(start.frequency().periods_per_year());
        let values: Vec<_> = (0..20 * period)
            .map(|i| 100.0 + i as f64 * 0.2 + ((i * 17) % 23) as f64)
            .collect();
        for base in [
            Spec::x11(x11::Options::new()),
            Spec::x11(x11::Options::new().with_mode(x11::Mode::Multiplicative)),
            Spec::x11_regarima(
                RegArimaSpec::new(ar1())
                    .with_constant(true)
                    .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(0), 0).unwrap()),
                x11::Options::new().with_mode(x11::Mode::Additive),
            ),
        ] {
            let expected = x13::adjust(Series::new(start, &values), &base).unwrap();
            for span in [Span::ObservedOnly, Span::WithExtensions] {
                assert_same(
                    &x13::adjust(Series::new(start, &values), &with_span(&base, span)).unwrap(),
                    &expected,
                );
            }
        }
    }
}

#[test]
fn extreme_span_includes_extensions_by_default() {
    let values: Vec<_> = (0..120)
        .map(|i| 100.0 + i as f64 * 0.1 + ((i * 17) % 19) as f64)
        .collect();
    let start = Period::monthly(2000, 7).unwrap();
    let base = Spec::x11_regarima(
        RegArimaSpec::new(ar1())
            .with_constant(true)
            .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(12), 5).unwrap()),
        x11::Options::new().with_mode(x11::Mode::Additive),
    );
    let run = |spec: &Spec| x13::adjust(Series::new(start, &values), spec).unwrap();
    let default = run(&base);
    assert_same(&run(&with_span(&base, Span::WithExtensions)), &default);
    let observed = run(&with_span(&base, Span::ObservedOnly));
    assert_ne!(
        observed.seasonally_adjusted(),
        default.seasonally_adjusted()
    );
}
