use super::*;
use crate::regarima::{ForecastHorizon, ForecastSpec, ModelSelectionResult, TransformSelection};
use crate::{Transform, x11};

const FILTER_REPLACED: Warning = Warning::SeasonalFilterReplaced {
    requested: x11::SeasonalFilter::S3x15,
    replacement: x11::SeasonalFilter::Stable,
};

fn monthly() -> TimeAxis {
    TimeAxis::monthly(2000, 1).unwrap()
}

fn automatic_transform() -> RegArimaSpec {
    RegArimaSpec::automatic_model().with_transform(TransformSelection::automatic())
}

fn extended(forecasts: u16, backcasts: u16) -> RegArimaSpec {
    RegArimaSpec::automatic_model()
        .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(forecasts), backcasts).unwrap())
}

/// Runs the pipeline on a series without gaps, with the default final options.
fn run(
    values: &[f64],
    axis: TimeAxis,
    requested: RegArimaSpec,
    mode: X11ModeSelection,
    options: x11::Options,
) -> Result<Adjustment, Error> {
    let final_options = FinalX11Options::new(
        crate::x11::FinalOutliers::Retain,
        crate::x11::TemporaryChangeAllocation::Irregular,
        crate::x11::HolidayAdjustment::Remove,
    );
    adjust(values, &[], axis, requested, mode, options, final_options)
}

/// The seasonality tests Build 62 cannot compute for a constant series, in
/// the order it warns about them: the two stable-seasonality F tests and the
/// moving-seasonality test have a zero residual mean square, and the two
/// residual-seasonality tests are skipped.
fn constant_diagnostic_warnings() -> [Warning; 5] {
    use crate::diagnostics::SeasonalityTest as Test;
    [
        Test::PreliminaryStable,
        Test::FinalStable,
        Test::Moving,
        Test::ResidualFull,
        Test::ModifiedResidualFull,
    ]
    .map(|test| Warning::SeasonalityTestUnavailable { test })
}

fn with_constant_diagnostics<const N: usize>(first: [Warning; N]) -> Vec<Warning> {
    first
        .into_iter()
        .chain(constant_diagnostic_warnings())
        .collect()
}

#[test]
fn constant_series_skips_model_selection() {
    let values = [100.0; 120];
    let result = run(
        &values,
        monthly(),
        RegArimaSpec::automatic_model(),
        X11ModeSelection::FromTransform,
        x11::Options::default(),
    )
    .unwrap();
    assert_eq!(
        result.model_selection(),
        Some(&ModelSelectionResult::ConstantSeries {
            transform: Transform::None
        })
    );
    assert!(result.model().is_none());
    assert!(result.forecasts().is_none());
    assert!(result.backcasts().is_none());
    assert_eq!(result.seasonal().unwrap().len(), values.len());
}

/// `editor.f` reports a nonpositive value, a constant series and a replaced
/// seasonal filter before any warning from fitting a candidate model.
#[test]
fn editor_warnings_precede_candidate_warnings() {
    let options = x11::Options::default().with_seasonal_filter(x11::SeasonalFilter::S3x15);
    let nonpositive = Warning::AutomaticTransformNonPositiveInput { index: 7 };
    let mut prepared = x11::PreparedOptions::new(options, monthly(), 120).unwrap();
    let warnings = editor_and_candidate_warnings(
        vec![Warning::DevianceIncreased, nonpositive.clone()],
        true,
        &mut prepared,
        false,
    );
    assert_eq!(
        warnings,
        [
            nonpositive,
            Warning::ConstantSeries,
            FILTER_REPLACED,
            Warning::DevianceIncreased
        ]
    );
    // X-11 must not report the replaced filter a second time.
    assert!(prepared.take_warning().is_none());
}

/// A nonpositive series skips the automatic transform test, so a negative
/// constant series still skips the model.
#[test]
fn negative_constant_series_reports_each_warning_once() {
    let result = run(
        &[-100.0; 120],
        monthly(),
        automatic_transform(),
        X11ModeSelection::FromTransform,
        x11::Options::default().with_seasonal_filter(x11::SeasonalFilter::S3x15),
    )
    .unwrap();
    assert_eq!(
        result.warnings(),
        with_constant_diagnostics([
            Warning::AutomaticTransformNonPositiveInput { index: 0 },
            Warning::ConstantSeries,
            FILTER_REPLACED,
        ])
    );
}

/// A positive constant series reaches the automatic transform test, whose
/// fit fails on a series that differencing annihilates.
#[test]
fn positive_constant_series_fails_the_automatic_transform_test() {
    for axis in [monthly(), TimeAxis::quarterly(2000, 1).unwrap()] {
        let result = run(
            &[100.0; 120],
            axis,
            automatic_transform(),
            X11ModeSelection::FromTransform,
            x11::Options::default(),
        );
        assert_eq!(result.err(), Some(Error::DegenerateSeries), "{axis:?}");
    }
}

#[test]
fn multiplicative_mode_rejects_a_negative_constant_series() {
    let result = run(
        &[-100.0; 120],
        monthly(),
        RegArimaSpec::automatic_model(),
        X11ModeSelection::Explicit(x11::Mode::Multiplicative),
        x11::Options::default(),
    );
    assert_eq!(result.err(), Some(Error::NonPositiveValue { index: 0 }));
}

/// A 3x15 filter needs twenty years. Nineteen years qualify only with the
/// twelve requested forecasts, which count although a constant series never
/// produces them.
#[test]
fn constant_series_counts_requested_forecasts_toward_the_seasonal_filter() {
    let options = x11::Options::default().with_seasonal_filter(x11::SeasonalFilter::S3x15);
    let run = |forecasts| {
        run(
            &[100.0; 228],
            monthly(),
            extended(forecasts, 0),
            X11ModeSelection::FromTransform,
            options,
        )
        .unwrap()
    };
    let with_forecasts = run(12);
    assert!(with_forecasts.forecasts().is_none());
    assert_eq!(
        with_forecasts.warnings(),
        with_constant_diagnostics([Warning::ConstantSeries])
    );
    assert_eq!(
        run(0).warnings(),
        with_constant_diagnostics([Warning::ConstantSeries, FILTER_REPLACED])
    );
}

/// A constant series produces no forecasts or backcasts, so the spans the
/// quality measures would cover are the observed series.
#[test]
fn constant_series_quality_spans_exclude_the_requested_extensions() {
    use crate::diagnostics::{Unavailable, UnavailableDetail::Quality};
    let result = run(
        &[100.0; 120],
        monthly(),
        extended(12, 12),
        X11ModeSelection::Explicit(x11::Mode::PseudoAdditive),
        x11::Options::default(),
    )
    .unwrap();
    let quality = result.diagnostics().x11().unwrap().quality();
    let Err(Unavailable(Quality {
        observed_span,
        seasonal_history_span,
        ..
    })) = quality
    else {
        panic!("the measures need a series that varies: {quality:?}")
    };
    assert_eq!(observed_span.first(), monthly().start());
    assert_eq!(observed_span.count(), 120);
    assert_eq!(observed_span, seasonal_history_span);
}

#[test]
fn constant_series_rejects_a_henderson_filter_longer_than_the_series() {
    let henderson = x11::Options::default().with_trend_filter(x11::TrendFilter::Henderson(
        x11::HendersonLength::new(83).unwrap(),
    ));
    let run = |options| {
        run(
            &[100.0; 72],
            monthly(),
            extended(12, 12),
            X11ModeSelection::FromTransform,
            options,
        )
    };
    assert_eq!(
        run(henderson).err(),
        Some(Error::HendersonLongerThanSeries {
            terms: 83,
            count: 72
        })
    );
    assert!(run(x11::Options::default()).is_ok());
}
