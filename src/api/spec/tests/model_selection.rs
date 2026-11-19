use crate::api::regarima::{ForecastHorizon, LaborDay, LengthEffect};
use crate::api::spec::*;
use crate::error::{InvalidSpec, SpecProblem, UnsupportedConfiguration};
use crate::regarima::{ConstantSelection, ModelSelection, TransformSelection};
use crate::{Period, Series, x11};

fn automatic_transform() -> TransformSelection {
    TransformSelection::automatic()
}

fn explicit_model() -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(1, 1, 0).unwrap(),
        ArimaOrder::new(0, 1, 1).unwrap(),
    )
    .unwrap()
}

fn unsupported(configuration: UnsupportedConfiguration) -> Error {
    Error::from(crate::error::UnsupportedDetail::Configuration(
        configuration,
    ))
}

#[test]
fn defaults_differ_only_in_model_and_constant() {
    let model = explicit_model();
    let explicit = RegArimaSpec::new(model);
    let automatic = RegArimaSpec::automatic_model();
    assert_eq!(explicit.model(), ModelSelection::Explicit(model));
    assert_eq!(explicit.constant(), ConstantSelection::Excluded);
    assert_eq!(automatic.model(), ModelSelection::Automatic);
    assert_eq!(automatic.constant(), ConstantSelection::Automatic);
    for requested in [explicit, automatic] {
        assert_eq!(
            requested.transform(),
            TransformSelection::Explicit(Transform::None)
        );
        assert_eq!(requested.forecast().horizon(), ForecastHorizon::OneYear);
        assert_eq!(
            requested.forecast().forecast_periods(Frequency::Monthly),
            12
        );
        assert_eq!(
            requested.forecast().forecast_periods(Frequency::Quarterly),
            4
        );
        assert_eq!(requested.forecast().backcast_periods(), 0);
        assert_eq!(requested.calendar(), None);
        assert_eq!(requested.outliers(), None);
    }
}

#[test]
fn automatic_model_with_a_forced_constant_is_unsupported() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    for (include, policy) in [
        (true, ConstantSelection::Included),
        (false, ConstantSelection::Excluded),
    ] {
        let requested = RegArimaSpec::automatic_model().with_constant(include);
        assert_eq!(requested.model(), ModelSelection::Automatic);
        assert_eq!(requested.constant(), policy);
        let expected = unsupported(UnsupportedConfiguration::AutomaticModelConstant);
        assert_eq!(requested.validate_model_policy(), Err(expected.clone()));
        assert_eq!(
            requested
                .clone()
                .with_transform(Transform::Log)
                .validate(&[-1.0], axis),
            Err(expected.clone())
        );
        assert_eq!(requested.resolve(Transform::None), Err(expected.clone()));
        assert_eq!(requested.resolve_explicit(), Err(expected.clone()));
        assert_eq!(
            requested
                .with_transform(automatic_transform())
                .resolve_explicit(),
            Err(expected)
        );
    }
}

#[test]
fn automatic_model_does_not_resolve_to_an_explicit_model() {
    for requested in [
        RegArimaSpec::automatic_model(),
        RegArimaSpec::automatic_model().with_transform(automatic_transform()),
    ] {
        assert_eq!(requested.validate_model_policy(), Ok(()));
        assert_eq!(
            requested.resolve(Transform::None),
            Err(unsupported(UnsupportedConfiguration::AutomaticModel))
        );
        assert_eq!(
            requested.resolve(Transform::Log),
            Err(unsupported(UnsupportedConfiguration::AutomaticModel))
        );
        assert_eq!(
            requested.resolve_explicit(),
            Err(unsupported(UnsupportedConfiguration::AutomaticModel))
        );
    }
    let transform_only = RegArimaSpec::new(explicit_model()).with_transform(automatic_transform());
    assert_eq!(
        transform_only.model(),
        ModelSelection::Explicit(explicit_model())
    );
    assert_eq!(transform_only.constant(), ConstantSelection::Excluded);
    assert_eq!(
        transform_only.resolve_explicit(),
        Err(unsupported(UnsupportedConfiguration::AutomaticTransform))
    );
    assert!(transform_only.resolve(Transform::Log).is_ok());
}

#[test]
fn setters_keep_the_model_and_constant_choices() {
    let calendar = CalendarSpec::new().with_basis(CalendarBasis::Length(LengthEffect::LeapYear));
    let outliers = OutlierSpec::default();
    let extensions = ForecastSpec::new(ForecastHorizon::Periods(8), 4).unwrap();
    for requested in [
        RegArimaSpec::new(explicit_model()),
        RegArimaSpec::new(explicit_model()).with_constant(true),
        RegArimaSpec::automatic_model(),
        RegArimaSpec::automatic_model().with_constant(true),
        RegArimaSpec::automatic_model().with_constant(false),
    ] {
        for changed in [
            requested.clone().with_transform(Transform::Log),
            requested.clone().with_transform(automatic_transform()),
            requested.clone().with_forecast(extensions),
            requested.clone().with_calendar(calendar),
            requested.clone().with_outliers(outliers),
        ] {
            assert_eq!(changed.model(), requested.model());
            assert_eq!(changed.constant(), requested.constant());
        }
    }
    for include in [true, false] {
        let requested = RegArimaSpec::new(explicit_model())
            .with_constant(include)
            .with_transform(Transform::Log)
            .with_forecast(extensions)
            .with_calendar(calendar)
            .with_outliers(outliers);
        let resolved = requested.resolve_explicit().unwrap();
        assert_eq!(resolved.model(), explicit_model().into());
        assert_eq!(resolved.includes_constant(), include);
        assert_eq!(resolved.transform(), Transform::Log);
        assert_eq!(resolved.extensions(), extensions);
        assert_eq!(resolved.calendar(), Some(calendar));
        assert_eq!(resolved.outliers(), Some(outliers));
    }
}

#[test]
fn automatic_model_requests_are_validated_in_order() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let requested = RegArimaSpec::automatic_model();
    let spec = Spec::x11_regarima(requested.clone(), x11::Options::new());
    assert!(matches!(
        crate::adjust(Series::new(axis.start(), &[1.0]), &spec),
        Err(Error::TooShort { .. })
    ));
    let wrong_frequency_options = x11::Options::new().with_seasonal_filters(
        crate::x11::SeasonalFilters::Quarterly([crate::x11::SeasonalFilter::S3x5; 4]),
    );
    let spec = Spec::x11_regarima(requested.clone(), wrong_frequency_options);
    assert_eq!(
        crate::adjust(Series::new(axis.start(), &[1.0]), &spec).unwrap_err(),
        Error::InvalidSpec(InvalidSpec(SpecProblem::SeasonalFilterFrequencyMismatch {
            expected: Frequency::Quarterly,
            actual: Frequency::Monthly,
        }))
    );
    for include in [true, false] {
        let spec = Spec::x11_regarima(
            requested.clone().with_constant(include),
            x11::Options::new(),
        );
        assert_eq!(
            crate::adjust(Series::new(axis.start(), &[1.0]), &spec).unwrap_err(),
            unsupported(UnsupportedConfiguration::AutomaticModelConstant)
        );
        assert_eq!(
            crate::adjust(Series::new(axis.start(), &[f64::INFINITY]), &spec).unwrap_err(),
            Error::NonFiniteValue { index: 0 }
        );
        assert_eq!(
            crate::adjust(Series::new(axis.start(), &[0.0]), &spec).unwrap_err(),
            Error::AllZeroInput
        );
        assert!(matches!(
            crate::adjust(
                Series::new(Period::monthly(i32::MAX, 12).unwrap(), &[1.0, 1.0]),
                &spec
            ),
            Err(Error::InvalidPeriod(_))
        ));
    }
    let both = requested
        .clone()
        .with_constant(true)
        .with_transform(automatic_transform());
    let spec = Spec::x11_regarima(both, x11::Options::new().with_mode(x11::Mode::Additive));
    assert_eq!(
        crate::adjust(Series::new(axis.start(), &[1.0]), &spec).unwrap_err(),
        Error::InvalidSpec(InvalidSpec(SpecProblem::AutomaticTransformWithExplicitMode))
    );
    let log_spec = Spec::x11_regarima(
        requested.with_transform(Transform::Log),
        x11::Options::new(),
    );
    assert_eq!(
        crate::adjust(Series::new(axis.start(), &[-1.0]), &log_spec).unwrap_err(),
        Error::NonPositiveValue { index: 0 }
    );
}

#[test]
fn log_and_calendar_checks_apply_to_explicit_and_automatic_models() {
    let monthly = TimeAxis::monthly(2000, 1).unwrap();
    let quarterly = TimeAxis::quarterly(2000, 1).unwrap();
    for requested in [
        RegArimaSpec::new(explicit_model()),
        RegArimaSpec::automatic_model(),
    ] {
        assert_eq!(
            requested
                .clone()
                .with_transform(Transform::Log)
                .validate(&[-1.0], monthly),
            Err(Error::NonPositiveValue { index: 0 })
        );
        let calendar = CalendarSpec::new().with_labor_day(LaborDay::new(8).unwrap());
        assert_eq!(
            requested
                .clone()
                .with_calendar(calendar)
                .validate(&[1.0], quarterly),
            Err(Error::InvalidSpec(InvalidSpec(
                SpecProblem::UnsupportedCalendarFrequency {
                    feature: CalendarFeature::LaborDay,
                    frequency: Frequency::Quarterly
                }
            )))
        );
        let spec = Spec::x11_regarima(requested.with_calendar(calendar), x11::Options::new());
        assert_eq!(
            crate::adjust(Series::new(quarterly.start(), &[1.0]), &spec).unwrap_err(),
            Error::InvalidSpec(InvalidSpec(SpecProblem::UnsupportedCalendarFrequency {
                feature: CalendarFeature::LaborDay,
                frequency: Frequency::Quarterly,
            }))
        );
    }
}

#[test]
fn with_transform_takes_a_transform_a_selection_or_automatic_options() {
    assert_eq!(
        TransformSelection::automatic(),
        TransformSelection::Automatic(crate::regarima::AutoTransformOptions::new())
    );
    assert_eq!(
        RegArimaSpec::automatic_model()
            .with_transform(Transform::Log)
            .transform(),
        TransformSelection::Explicit(Transform::Log)
    );
    let options = crate::regarima::AutoTransformOptions::new()
        .with_aic_difference(-1.0)
        .unwrap();
    assert_eq!(
        RegArimaSpec::automatic_model()
            .with_transform(options)
            .transform(),
        TransformSelection::Automatic(options)
    );

    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let values = (0..48).map(|i| 100.0 + f64::from(i)).collect::<Vec<_>>();
    for spec in [
        RegArimaSpec::automatic_model().with_transform(automatic_transform()),
        RegArimaSpec::new(explicit_model()).with_transform(automatic_transform()),
    ] {
        assert_eq!(spec.transform(), automatic_transform());
        assert_eq!(spec.validate_model_policy(), Ok(()));
        assert_eq!(spec.validate(&values, axis), Ok(()));
        // The log-domain check belongs to an explicit log transform; the
        // automatic choice handles nonpositive values itself.
        assert_eq!(spec.validate(&[-1.0], axis), Ok(()));
    }

    // It replaces an explicit transform set earlier.
    assert_eq!(
        RegArimaSpec::automatic_model()
            .with_transform(Transform::Log)
            .with_transform(TransformSelection::automatic()),
        RegArimaSpec::automatic_model().with_transform(TransformSelection::automatic())
    );

    // A manual outlier still needs an explicit transform.
    let outlier = crate::regarima::ManualOutlier::new(
        crate::regarima::OutlierType::Additive,
        Period::monthly(2001, 1).unwrap(),
    );
    assert_eq!(
        RegArimaSpec::new(explicit_model())
            .with_transform(TransformSelection::automatic())
            .with_manual_outliers(&[outlier])
            .validate_model_policy(),
        Err(unsupported(UnsupportedConfiguration::ManualOutliers))
    );
}
