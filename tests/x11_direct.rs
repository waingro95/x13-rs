use x13::{Period, Series, Spec, adjust, x11};

fn seasonal_values(count: usize, period: usize) -> Vec<f64> {
    (0..count)
        .map(|index| {
            let seasonal = (index % period) as f64 - (period - 1) as f64 / 2.0;
            100.0 + 0.2 * index as f64 + seasonal
        })
        .collect()
}

#[test]
fn direct_x11_has_no_model_calendar_effects_or_outliers() {
    for (values, start, mode) in [
        (
            seasonal_values(72, 12),
            Period::monthly(2000, 1).unwrap(),
            x11::Mode::Additive,
        ),
        (
            seasonal_values(24, 4),
            Period::quarterly(2000, 1).unwrap(),
            x11::Mode::Multiplicative,
        ),
    ] {
        let adjustment = adjust(
            Series::new(start, &values),
            &Spec::x11(x11::Options::new().with_mode(mode)),
        )
        .unwrap();

        assert!(adjustment.model().is_none());
        assert!(adjustment.calendar_effects().is_none());
        assert!(adjustment.outliers().is_empty());
    }
}
