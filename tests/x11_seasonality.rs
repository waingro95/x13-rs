use x13::diagnostics::UnavailableKind;
use x13::{Period, Series, Spec, Warning, diagnostics::SeasonalityTest as Test, x11};

#[test]
fn constant_series_seasonality_tests_say_why_they_are_unavailable() {
    let actual = x13::adjust(
        Series::new(Period::monthly(2000, 1).unwrap(), &[100.0; 120]),
        &Spec::x11(
            x11::Options::new()
                .with_mode(x11::Mode::Additive)
                .with_seasonal_filter(x11::SeasonalFilter::S3x5),
        ),
    )
    .unwrap();
    let d = actual.diagnostics().x11().unwrap().seasonality();
    for (name, test) in [
        ("preliminary", d.preliminary_stable()),
        ("final", d.final_stable()),
    ] {
        let unavailable = test.expect_err(name);
        assert_eq!(unavailable.kind(), UnavailableKind::Numerical);
        assert_eq!(
            unavailable.to_string(),
            format!(
                "the {name} stable-seasonality test is unavailable: the residual mean square is \
                 zero"
            )
        );
    }
    assert!(d.moving().is_err());
    let kw = d
        .kruskal_wallis()
        .expect("Kruskal-Wallis is defined for a constant series");
    assert_eq!(kw.degrees_of_freedom(), 11);
    assert_eq!(kw.sample().count(), 120);
    assert!(kw.upper_tail_probability_percent() > 90.0);
    assert!(d.combined().is_err());
    for (full, recent, name) in [
        (
            d.residual_full(),
            d.residual_recent_three_years(),
            "residual",
        ),
        (
            d.modified_residual_full(),
            d.modified_residual_recent_three_years(),
            "modified residual",
        ),
    ] {
        let full = full.expect_err("a constant series has no residual seasonality test");
        assert_eq!(full.kind(), UnavailableKind::NotApplicable);
        assert_eq!(
            full.to_string(),
            format!(
                "the {name}-seasonality test over the full span is unavailable: the original \
                 series is constant"
            )
        );
        let recent = recent.expect_err("the recent test needs the full-span test");
        assert_eq!(recent.kind(), UnavailableKind::Numerical);
        assert_eq!(
            recent.to_string(),
            format!(
                "the {name}-seasonality test over the last three years is unavailable: the \
                 {name}-seasonality test over the full span is unavailable"
            )
        );
    }
    let expected = [
        Test::PreliminaryStable,
        Test::FinalStable,
        Test::Moving,
        Test::ResidualFull,
        Test::ModifiedResidualFull,
    ];
    let warnings = actual.warnings();
    assert_eq!(warnings.len(), expected.len(), "{warnings:?}");
    for (warning, expected) in warnings.iter().zip(expected) {
        assert!(
            matches!(
                warning,
                Warning::SeasonalityTestUnavailable { test, .. } if *test == expected
            ),
            "{warning:?}"
        );
    }
}
