use x13::{
    ErrorKind, Frequency, Period, Series, Spec, x11, x11::ExtremeScale as Scale,
    x11::SigmaGroup as Group, x11::SigmaGroups as Groups,
};

const PERIOD: x11::Options = x11::Options::new().with_extreme_scale(Scale::ByCalendarPeriod);
const QUARTERS: Groups = Groups::Quarterly([Group::Selected; 4]);

#[test]
fn groups_of_the_wrong_frequency_are_rejected() {
    let model = x13::ArimaModel::new(
        x13::ArimaOrder::new(1, 0, 0).unwrap(),
        x13::ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    for (start, groups, expected) in [
        (
            Period::monthly(2000, 7).unwrap(),
            QUARTERS,
            Frequency::Quarterly,
        ),
        (
            Period::quarterly(2000, 3).unwrap(),
            Groups::Monthly([Group::Selected; 12]),
            Frequency::Monthly,
        ),
    ] {
        let options = x11::Options::new().with_extreme_scale(Scale::Grouped(groups));
        for spec in [
            Spec::x11(options),
            Spec::x11_regarima(
                x13::RegArimaSpec::new(model),
                options.with_mode(x13::x11::Mode::Additive),
            ),
        ] {
            let error = x13::adjust(Series::new(start, &[100.0; 120]), &spec).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidSpec);
            assert_eq!(
                error.to_string(),
                format!(
                    "the calendar groups for extreme-value scales are for {expected} series, \
                     but the input series is {}",
                    start.frequency()
                )
            );
        }
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
    assert_same_fit(actual, expected);
    assert_eq!(actual.diagnostics(), expected.diagnostics());
    assert_eq!(actual.warnings(), expected.warnings());
}

fn assert_same_fit(actual: &x13::Adjustment, expected: &x13::Adjustment) {
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        actual.forecasts().map(|f| (
            f.start(),
            bits(f.series().values()),
            bits(f.transformed().values())
        )),
        expected.forecasts().map(|f| (
            f.start(),
            bits(f.series().values()),
            bits(f.transformed().values())
        ))
    );
    assert_eq!(
        actual.backcasts().map(|f| (
            f.start(),
            bits(f.series().values()),
            bits(f.transformed().values())
        )),
        expected.backcasts().map(|f| (
            f.start(),
            bits(f.series().values()),
            bits(f.transformed().values())
        ))
    );
    assert_eq!(
        actual.model().map(|m| m.model()),
        expected.model().map(|m| m.model())
    );
    if let (Some(a), Some(b)) = (actual.model(), expected.model()) {
        assert_eq!(a.transform(), b.transform());
        assert_eq!(a.transform_selection(), b.transform_selection());
        assert_eq!(a.parameters(), b.parameters());
        assert_eq!(a.regression_coefficients(), b.regression_coefficients());
        assert_eq!(
            a.innovation_variance().to_bits(),
            b.innovation_variance().to_bits()
        );
        assert_eq!(
            a.log_determinant_correction().to_bits(),
            b.log_determinant_correction().to_bits()
        );
        assert_eq!(
            a.log_likelihood().map(f64::to_bits),
            b.log_likelihood().map(f64::to_bits)
        );
        assert_eq!(a.information_criteria(), b.information_criteria());
        assert_eq!(a.arma_covariance(), b.arma_covariance());
        assert_eq!(
            a.regression_covariance().map(bits),
            b.regression_covariance().map(bits)
        );
        assert_eq!(bits(a.residuals()), bits(b.residuals()));
        assert_eq!(a.convergence(), b.convergence());
    }
    if let (Some(a), Some(b)) = (actual.forecasts(), expected.forecasts()) {
        for (a, b) in [
            (a.lower_95().values(), b.lower_95().values()),
            (a.upper_95().values(), b.upper_95().values()),
            (
                a.transformed_standard_errors(),
                b.transformed_standard_errors(),
            ),
            (a.regression_variances(), b.regression_variances()),
            (a.stochastic_variances(), b.stochastic_variances()),
        ] {
            assert_eq!(bits(a), bits(b));
        }
    }
    if let (Some(a), Some(b)) = (actual.backcasts(), expected.backcasts()) {
        for (a, b) in [
            (a.lower_95().values(), b.lower_95().values()),
            (a.upper_95().values(), b.upper_95().values()),
            (
                a.transformed_standard_errors(),
                b.transformed_standard_errors(),
            ),
        ] {
            assert_eq!(bits(a), bits(b));
        }
    }
}

#[test]
fn a_single_group_pools_every_period() {
    for (start, empty, all) in [
        (
            Period::monthly(2000, 7).unwrap(),
            Groups::Monthly([Group::Complement; 12]),
            Groups::Monthly([Group::Selected; 12]),
        ),
        (
            Period::quarterly(2000, 3).unwrap(),
            Groups::Quarterly([Group::Complement; 4]),
            Groups::Quarterly([Group::Selected; 4]),
        ),
    ] {
        let period = usize::from(start.frequency().periods_per_year());
        let mut values: Vec<_> = (0..20 * period)
            .map(|i| 100.0 + i as f64 * 0.2 + ((i * 17) % 23) as f64)
            .collect();
        values[8 * period] += 150.0;
        for mode in [x13::x11::Mode::Additive, x13::x11::Mode::Multiplicative] {
            let base = x11::Options::new().with_mode(mode);
            let run = |options: x11::Options| {
                x13::adjust(Series::new(start, &values), &Spec::x11(options)).unwrap()
            };
            let default = run(base);
            assert_same(
                &default,
                &run(base.with_extreme_scale(Scale::RollingFiveYear)),
            );
            let pooled = run(base.with_extreme_scale(Scale::Grouped(empty)));
            assert_same(&pooled, &run(base.with_extreme_scale(Scale::Grouped(all))));
            assert!(
                pooled
                    .seasonal()
                    .unwrap()
                    .values()
                    .iter()
                    .zip(default.seasonal().unwrap().values())
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            );
        }
    }
}

#[test]
fn scale_sigma_and_span_options_are_independent() {
    let limits = x13::x11::SigmaLimits::new(1.0, 3.0).unwrap();
    let options = PERIOD
        .with_sigma_limits(limits)
        .with_extreme_estimation_span(x13::x11::ExtremeEstimationSpan::ObservedOnly);
    assert_eq!(options.extreme_scale(), Scale::ByCalendarPeriod);
    assert_eq!(options.sigma_limits(), limits);
    assert_eq!(
        options.extreme_estimation_span(),
        x13::x11::ExtremeEstimationSpan::ObservedOnly
    );
    let reordered = x11::Options::new()
        .with_extreme_estimation_span(x13::x11::ExtremeEstimationSpan::ObservedOnly)
        .with_sigma_limits(limits)
        .with_extreme_scale(Scale::ByCalendarPeriod);
    assert_eq!(options, reordered);
}

#[test]
fn extreme_scale_changes_x11_but_not_the_regarima_fit() {
    let model = x13::ArimaModel::new(
        x13::ArimaOrder::new(1, 0, 0).unwrap(),
        x13::ArimaOrder::new(0, 0, 0).unwrap(),
    )
    .unwrap();
    for start in [
        Period::monthly(2000, 7).unwrap(),
        Period::quarterly(2000, 3).unwrap(),
    ] {
        let period = usize::from(start.frequency().periods_per_year());
        let mut values: Vec<_> = (0..20 * period)
            .map(|i| 100.0 + i as f64 * 0.1 + ((i * 17) % 19) as f64)
            .collect();
        values[8 * period] += 100.0;
        let groups = match start.frequency() {
            Frequency::Monthly => Groups::Monthly([
                Group::Selected,
                Group::Selected,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Complement,
                Group::Selected,
            ]),
            Frequency::Quarterly => Groups::Quarterly([
                Group::Selected,
                Group::Complement,
                Group::Complement,
                Group::Selected,
            ]),
            other => unreachable!("no groups for {other:?}"),
        };
        let regarima = x13::RegArimaSpec::new(model)
            .with_constant(true)
            .with_forecast(
                x13::regarima::ForecastSpec::new(
                    x13::regarima::ForecastHorizon::Periods(period as u16),
                    (period / 2) as u16,
                )
                .unwrap(),
            );
        let specs: Vec<_> = [
            Scale::RollingFiveYear,
            Scale::ByCalendarPeriod,
            Scale::Grouped(groups),
        ]
        .into_iter()
        .map(|scale| {
            Spec::x11_regarima(
                regarima.clone(),
                x11::Options::new()
                    .with_mode(x13::x11::Mode::Additive)
                    .with_extreme_scale(scale),
            )
        })
        .collect();
        let results: Vec<_> = specs
            .iter()
            .map(|spec| x13::adjust(Series::new(start, &values), spec).unwrap())
            .collect();
        for result in &results {
            assert_eq!(result.forecasts().unwrap().series().len(), period);
            assert_eq!(result.backcasts().unwrap().series().len(), period / 2);
        }
        for result in &results[1..] {
            assert!(
                result
                    .seasonal()
                    .unwrap()
                    .values()
                    .iter()
                    .zip(results[0].seasonal().unwrap().values())
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            );
            assert_same_fit(result, &results[0]);
        }
    }
}
