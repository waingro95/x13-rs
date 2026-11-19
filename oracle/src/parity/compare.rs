//! Compares Rust's outputs with Build 62's.
//!
//! A comparison yields a [`Failure`] for each value outside its tolerance and
//! one [`Headroom`] per output: the worst error as a fraction of the error the
//! tolerance allows. `tests/parity/tolerances.toml` holds the tolerances.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::output::{Coefficient, Components, Output, SeatsModel};
use crate::{Frequency, Series, SeriesPoint};

/// The contents of `tests/parity/tolerances.toml`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Tolerances {
    seats: SeatsTolerance,
    unconditioned: ComponentTolerances,
    additive: ComponentTolerances,
    multiplicative: ComponentTolerances,
    saved: SavedTolerances,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeatsTolerance {
    relative: f64,
}

/// Relative tolerances for the scalars Build 62 saves as decimal text.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedTolerances {
    fifteen_digits: f64,
    fourteen_digits: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentTolerances {
    seasonal: ComponentTolerance,
    seasonally_adjusted: ComponentTolerance,
    trend_cycle: ComponentTolerance,
    irregular: ComponentTolerance,
}

impl ComponentTolerances {
    /// In the order of [`components`].
    const fn in_order(&self) -> [ComponentTolerance; 4] {
        [
            self.seasonal,
            self.seasonally_adjusted,
            self.trend_cycle,
            self.irregular,
        ]
    }
}

/// The tolerance for one value of a dated series.
///
/// A value passes when its error is at most `absolute`, or at most `relative`
/// times `max(|expected|, scale_floor)`, or, where the comparison has a
/// conditioning scale for that date, at most `conditioned_epsilon_multiplier`
/// times `f64::EPSILON` times the scale.
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentTolerance {
    absolute: f64,
    relative: f64,
    scale_floor: f64,
    #[serde(default)]
    conditioned_epsilon_multiplier: Option<f64>,
}

impl ComponentTolerance {
    /// Relative to the expected value alone, so an expected zero must be
    /// matched exactly.
    const fn relative(relative: f64) -> Self {
        Self {
            absolute: 0.0,
            relative,
            scale_floor: 0.0,
            conditioned_epsilon_multiplier: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Failure {
    pub(crate) output: String,
    pub(crate) index: Option<usize>,
    pub(crate) expected: String,
    pub(crate) actual: String,
    pub(crate) relative_error: Option<f64>,
}

impl Failure {
    pub(crate) fn text(output: &str, expected: impl ToString, actual: impl ToString) -> Self {
        Self {
            output: output.into(),
            index: None,
            expected: expected.to_string(),
            actual: actual.to_string(),
            relative_error: None,
        }
    }

    fn value(output: &str, index: usize, expected: f64, actual: f64, relative_error: f64) -> Self {
        Self {
            output: output.into(),
            index: Some(index),
            expected: format!("{expected:.17e}"),
            actual: format!("{actual:.17e}"),
            relative_error: relative_error.is_finite().then_some(relative_error),
        }
    }

    fn shape(detail: String) -> Self {
        Self::text(
            "component shape",
            "matching finite dated components",
            detail,
        )
    }
}

/// The worst error of one output as a fraction of the error its tolerance
/// allows. It is infinite when a value cannot pass at any error: an expected
/// zero that was missed, or a non-finite value.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Headroom {
    pub(crate) output: String,
    #[serde(serialize_with = "serialize_ratio")]
    pub(crate) ratio: f64,
}

fn serialize_ratio<S: serde::Serializer>(
    ratio: &f64,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    if ratio.is_finite() {
        serializer.serialize_f64(*ratio)
    } else {
        serializer.serialize_str("unbounded")
    }
}

pub(crate) fn worst(metrics: &[Headroom]) -> Option<Headroom> {
    metrics
        .iter()
        .max_by(|a, b| a.ratio.total_cmp(&b.ratio))
        .cloned()
}

/// The largest share of a tolerance that Rust may move from a recorded
/// baseline under `PARITY_COMPARE_RUST_TOLERANT` or `_CUMULATIVE`.
pub(crate) const DRIFT_LIMIT: f64 = 0.1;

pub(crate) fn drift_failures(scope: &str, metrics: &[Headroom]) -> Vec<Failure> {
    metrics
        .iter()
        .filter(|m| !m.ratio.is_finite() || m.ratio > DRIFT_LIMIT)
        .map(|m| {
            Failure::text(
                &format!("Rust {scope} drift: {}", m.output),
                "<= 0.1",
                m.ratio,
            )
        })
        .collect()
}

/// Compares every output both sides record: the components, the forecasts and
/// the regARIMA model.
pub(crate) fn compare(
    expected: &Output,
    actual: &Output,
    adjustment: &x13::Adjustment,
    series: &Series,
    tolerances: &Tolerances,
) -> (Vec<Failure>, Vec<Headroom>) {
    let seats = ComponentTolerance::relative(tolerances.seats.relative);
    let additive_scales;
    let (component_tolerances, scales): (_, [Option<&[f64]>; 4]) =
        match adjustment.diagnostics().x11().map(|x11| x11.mode()) {
            None => ([seats; 4], [None; 4]),
            Some(x13::x11::Mode::Additive) => {
                additive_scales =
                    additive_conditioning_scales(&expected.components, &series.values);
                (
                    tolerances.additive.in_order(),
                    [Some(additive_scales.as_slice()); 4],
                )
            }
            Some(x13::x11::Mode::Multiplicative) => {
                let (trend_cycle, irregular) = adjustment.conditioning_scales();
                (
                    tolerances.multiplicative.in_order(),
                    [None, None, Some(trend_cycle), Some(irregular)],
                )
            }
            Some(_) => (tolerances.unconditioned.in_order(), [None; 4]),
        };
    let mut failures = vec![];
    let mut metrics = vec![];
    for ((name, expected, actual), (tolerance, scales)) in
        components(&expected.components, &actual.components)
            .into_iter()
            .zip(component_tolerances.into_iter().zip(scales))
    {
        metrics.push(compare_points(
            name,
            expected,
            actual,
            tolerance,
            scales,
            &mut failures,
        ));
    }
    metrics.push(compare_points(
        "forecasts",
        &expected.forecasts,
        &actual.forecasts,
        ComponentTolerance::relative(tolerances.saved.fifteen_digits),
        None,
        &mut failures,
    ));
    if let (Some(expected), Some(actual)) = (&expected.transitory, &actual.transitory) {
        metrics.push(compare_points(
            "transitory",
            expected,
            actual,
            seats,
            None,
            &mut failures,
        ));
    }
    if expected.transitory.is_some() != actual.transitory.is_some() {
        failures.push(Failure::text(
            "transitory presence",
            expected.transitory.is_some(),
            actual.transitory.is_some(),
        ));
    }
    match (&expected.model, &actual.model) {
        (Some(expected), Some(actual)) => {
            for (name, a, b) in [
                (
                    "transform",
                    format!("{:?}", expected.transform),
                    format!("{:?}", actual.transform),
                ),
                (
                    "model",
                    format!("{:?}", expected.orders),
                    format!("{:?}", actual.orders),
                ),
                (
                    "outliers",
                    format!("{:?}", expected.outliers),
                    format!("{:?}", actual.outliers),
                ),
            ] {
                if a != b {
                    failures.push(Failure::text(name, a, b));
                }
            }
            metrics.push(compare_coefficients(
                "arma",
                series.frequency,
                &expected.arma,
                &actual.arma,
                if expected.strict_arma_gate {
                    tolerances.saved.fifteen_digits
                } else {
                    tolerances.saved.fourteen_digits
                },
                &mut failures,
            ));
            metrics.push(compare_coefficients(
                "regression",
                series.frequency,
                &expected.regression,
                &actual.regression,
                tolerances.saved.fifteen_digits,
                &mut failures,
            ));
        }
        (None, None) => (),
        _ => failures.push(Failure::text(
            "model presence",
            expected.model.is_some(),
            actual.model.is_some(),
        )),
    }
    (failures, metrics)
}

/// Compares the backcasts and the SEATS model, which Rust bit records leave
/// out (see [`Output`]). Rust's values come from the live `Adjustment`, so a
/// recorded baseline has nothing to compare them with.
pub(crate) fn compare_unrecorded(
    expected: &Output,
    adjustment: &x13::Adjustment,
    series: &Series,
    tolerances: &Tolerances,
) -> Result<(Vec<Failure>, Vec<Headroom>)> {
    let mut failures = vec![];
    let mut metrics = vec![compare_points(
        "backcasts",
        &expected.backcasts,
        &super::output::rust_backcasts(series, adjustment)?,
        ComponentTolerance::relative(tolerances.saved.fifteen_digits),
        None,
        &mut failures,
    )];
    // seatdg.f logs the coefficients with 14 significant digits.
    compare_seats_model(
        expected.seats_model.as_ref(),
        super::output::rust_seats_model(adjustment).as_ref(),
        series.frequency,
        tolerances.saved.fourteen_digits,
        &mut failures,
        &mut metrics,
    );
    Ok((failures, metrics))
}

/// The SEATS model: presence, then orders, then coefficients.
fn compare_seats_model(
    expected: Option<&SeatsModel>,
    actual: Option<&SeatsModel>,
    frequency: Frequency,
    tolerance: f64,
    failures: &mut Vec<Failure>,
    metrics: &mut Vec<Headroom>,
) {
    match (expected, actual) {
        (Some(expected), Some(actual)) => {
            if expected.orders != actual.orders {
                failures.push(Failure::text(
                    "seats model",
                    format!("{:?}", expected.orders),
                    format!("{:?}", actual.orders),
                ));
            }
            metrics.push(compare_coefficients(
                "seats coefficients",
                frequency,
                &expected.coefficients,
                &actual.coefficients,
                tolerance,
                failures,
            ));
        }
        (None, None) => (),
        (expected, actual) => failures.push(Failure::text(
            "seats model presence",
            expected.is_some(),
            actual.is_some(),
        )),
    }
}

fn components<'a>(
    expected: &'a Components,
    actual: &'a Components,
) -> [(&'static str, &'a [SeriesPoint], &'a [SeriesPoint]); 4] {
    [
        ("seasonal", &expected.seasonal, &actual.seasonal),
        (
            "seasonally_adjusted",
            &expected.seasonally_adjusted,
            &actual.seasonally_adjusted,
        ),
        ("trend_cycle", &expected.trend_cycle, &actual.trend_cycle),
        ("irregular", &expected.irregular, &actual.irregular),
    ]
}

/// The additive conditioning scale at each date: the largest of 1, the input
/// and the four expected components in absolute value, or NaN (no scale) when
/// one of them is missing or not finite.
fn additive_conditioning_scales(expected: &Components, input: &[f64]) -> Vec<f64> {
    input
        .iter()
        .enumerate()
        .map(|(index, &input)| {
            let at = |points: &[SeriesPoint]| points.get(index).map_or(f64::NAN, |p| p.value);
            let values = [
                input,
                at(&expected.seasonal),
                at(&expected.seasonally_adjusted),
                at(&expected.trend_cycle),
                at(&expected.irregular),
            ];
            if values.iter().all(|value| value.is_finite()) {
                values.into_iter().map(f64::abs).fold(1.0, f64::max)
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Compares one dated series value by value. `scales` holds a conditioning
/// scale per date; an entry that is not finite and positive is no scale.
fn compare_points(
    output: &str,
    expected: &[SeriesPoint],
    actual: &[SeriesPoint],
    tolerance: ComponentTolerance,
    scales: Option<&[f64]>,
    failures: &mut Vec<Failure>,
) -> Headroom {
    if expected.len() != actual.len() {
        failures.push(Failure::shape(format!(
            "{output}: {} values, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    let mut ratio = 0.0_f64;
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        if expected.period != actual.period {
            failures.push(Failure::shape(format!(
                "{output}[{index}]: period {:?}, expected {:?}",
                actual.period, expected.period
            )));
        }
        if !expected.value.is_finite() || !actual.value.is_finite() {
            ratio = f64::INFINITY;
            failures.push(Failure::shape(format!(
                "{output}[{index}]: value {}, expected {}",
                actual.value, expected.value
            )));
            continue;
        }
        let error = (actual.value - expected.value).abs();
        let relative_scale = expected.value.abs().max(tolerance.scale_floor);
        let relative_limit = tolerance.relative * relative_scale;
        let conditioned = scales
            .and_then(|scales| scales.get(index).copied())
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .zip(tolerance.conditioned_epsilon_multiplier);
        let allowed = tolerance
            .absolute
            .max(relative_limit)
            .max(conditioned.map_or(0.0, |(scale, multiplier)| f64::EPSILON * scale * multiplier));
        if error != 0.0 {
            ratio = ratio.max(error / allowed);
        }
        let passes = error <= tolerance.absolute
            || error <= relative_limit
            || conditioned
                .is_some_and(|(scale, multiplier)| error / (f64::EPSILON * scale) <= multiplier);
        if !passes {
            failures.push(Failure::value(
                output,
                index,
                expected.value,
                actual.value,
                error / relative_scale,
            ));
        }
    }
    Headroom {
        output: output.into(),
        ratio,
    }
}

/// Compares named scalars by relative error against the expected value. An
/// expected zero must be matched exactly.
fn compare_coefficients(
    name: &str,
    frequency: Frequency,
    expected: &[Coefficient],
    actual: &[Coefficient],
    tolerance: f64,
    failures: &mut Vec<Failure>,
) -> Headroom {
    if expected.len() != actual.len() {
        failures.push(Failure::text(
            &format!("{name}.length"),
            expected.len(),
            actual.len(),
        ));
    }
    let mut ratio = 0.0_f64;
    for (i, (a, b)) in expected.iter().zip(actual).enumerate() {
        // Build 62 labels an outlier the spec names with a numeric month and
        // one it detects with a month abbreviation, as Rust always does, so
        // outlier names are compared as parsed dates.
        let same_outlier = super::output::outlier_label(&a.name, frequency)
            .ok()
            .zip(super::output::outlier_label(&b.name, frequency).ok())
            .is_some_and(|(a, b)| a == b);
        if a.name != b.name && !same_outlier {
            failures.push(Failure::text(
                &format!("{name}[{i}].name"),
                &a.name,
                &b.name,
            ));
        }
        let error = (b.value - a.value).abs();
        ratio = ratio.max(if !a.value.is_finite() || !b.value.is_finite() {
            f64::INFINITY
        } else if error == 0.0 {
            0.0
        } else {
            error / (tolerance * a.value.abs())
        });
        if let Some(failure) = coefficient_failure(name, i, a.value, b.value, tolerance) {
            failures.push(failure);
        }
    }
    Headroom {
        output: name.into(),
        ratio,
    }
}

fn coefficient_failure(
    output: &str,
    index: usize,
    expected: f64,
    actual: f64,
    tolerance: f64,
) -> Option<Failure> {
    let error = (actual - expected).abs();
    if expected.is_finite() && actual.is_finite() && error <= tolerance * expected.abs() {
        return None;
    }
    Some(Failure::value(
        output,
        index,
        expected,
        actual,
        error / expected.abs(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_limit_is_inclusive_and_unbounded_is_explicit() {
        let metric = |ratio| Headroom {
            output: "trend".into(),
            ratio,
        };
        assert!(drift_failures("incremental", &[metric(0.1)]).is_empty());
        assert_eq!(
            drift_failures("cumulative", &[metric(0.10000000000000002)]).len(),
            1
        );
        assert_eq!(
            drift_failures("incremental", &[metric(f64::INFINITY)]).len(),
            1
        );
        assert_eq!(
            serde_json::to_value(metric(f64::INFINITY)).unwrap()["ratio"],
            "unbounded"
        );
        assert_eq!(serde_json::to_value(metric(0.1)).unwrap()["ratio"], 0.1);
    }

    #[test]
    fn headroom_divides_by_the_loosest_applicable_limit() {
        let point = |value| SeriesPoint {
            period: crate::Period::new(2000, 1),
            value,
        };
        let ratio = |expected, actual, scales: Option<&[f64]>| {
            compare_points(
                "test",
                &[point(expected)],
                &[point(actual)],
                ComponentTolerance {
                    absolute: 0.0,
                    relative: 0.25,
                    scale_floor: 0.0,
                    conditioned_epsilon_multiplier: Some(2.0),
                },
                scales,
                &mut vec![],
            )
            .ratio
        };
        assert_eq!(ratio(2.0, 2.25, None), 0.5);
        assert_eq!(ratio(2.0, 2.25, Some(&[1.0 / f64::EPSILON])), 0.125);
        assert_eq!(ratio(2.0, 2.25, Some(&[f64::NAN])), 0.5);
        assert_eq!(ratio(0.0, 0.0, None), 0.0);
        assert!(ratio(0.0, 1e-100, None).is_infinite());

        let coefficient = |value| Coefficient {
            name: "ar".into(),
            value,
        };
        let mut failures = vec![];
        let headroom = compare_coefficients(
            "arma",
            Frequency::Monthly,
            &[coefficient(2.0)],
            &[coefficient(2.25)],
            0.25,
            &mut failures,
        );
        assert_eq!(headroom.ratio, 0.5);
        assert!(failures.is_empty());
    }

    #[test]
    fn a_value_outside_every_limit_fails_once_with_its_index() {
        let point = |value| SeriesPoint {
            period: crate::Period::new(2000, 1),
            value,
        };
        let tolerance = ComponentTolerance {
            absolute: 1e-13,
            relative: 1e-13,
            scale_floor: 1.0,
            conditioned_epsilon_multiplier: Some(16.0),
        };
        let failures = |actual: f64, scales: Option<&[f64]>| {
            let mut failures = vec![];
            compare_points(
                "seasonal",
                &[point(1.0), point(1e3)],
                &[point(1.0), point(actual)],
                tolerance,
                scales,
                &mut failures,
            );
            failures
        };
        // 1e3 allows 1e-10 relatively; a scale of 1e6 allows 16e6 epsilons.
        assert!(failures(1e3 + 9e-11, None).is_empty());
        let failed = failures(1e3 + 2e-9, None);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].output, "seasonal");
        assert_eq!(failed[0].index, Some(1));
        assert!(failures(1e3 + 2e-9, Some(&[1e6, 1e6])).is_empty());
        assert_eq!(failures(f64::NAN, None)[0].output, "component shape");
    }

    #[test]
    fn coefficients_require_an_exact_zero_and_finite_values() {
        assert!(coefficient_failure("arma", 0, 0., 1e-100, 6e-15).is_some());
        assert!(coefficient_failure("arma", 0, 1., 1. + 4e-15, 6e-15).is_none());
        assert!(coefficient_failure("arma", 0, 1., 1. + 1e-13, 6e-15).is_some());
        assert!(coefficient_failure("arma", 0, f64::INFINITY, f64::INFINITY, 6e-15).is_some());
    }

    #[test]
    fn seats_model_flags_presence_and_order_mismatches() {
        let model = |orders, value| SeatsModel {
            orders,
            coefficients: vec![Coefficient {
                name: "Nonseasonal MA(01)".into(),
                value,
            }],
        };
        let airline = model([[0, 1, 1], [0, 1, 1]], 0.5);
        let run = |expected: Option<&SeatsModel>, actual: Option<&SeatsModel>| {
            let (mut failures, mut metrics) = (vec![], vec![]);
            compare_seats_model(
                expected,
                actual,
                Frequency::Monthly,
                6e-14,
                &mut failures,
                &mut metrics,
            );
            (failures, metrics)
        };

        let (failures, metrics) = run(None, None);
        assert!(failures.is_empty() && metrics.is_empty());
        for (expected, actual) in [(Some(&airline), None), (None, Some(&airline))] {
            let (failures, metrics) = run(expected, actual);
            assert_eq!(failures.len(), 1);
            assert_eq!(failures[0].output, "seats model presence");
            assert_eq!(failures[0].expected, expected.is_some().to_string());
            assert_eq!(failures[0].actual, actual.is_some().to_string());
            assert!(metrics.is_empty());
        }

        let (failures, metrics) = run(Some(&airline), Some(&airline));
        assert!(failures.is_empty());
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].output, "seats coefficients");
        assert_eq!(metrics[0].ratio, 0.0);

        // A SEATS model change with equal coefficients fails on orders alone.
        let changed = model([[0, 1, 1], [0, 0, 1]], 0.5);
        let (failures, metrics) = run(Some(&airline), Some(&changed));
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].output, "seats model");
        assert_eq!(failures[0].expected, "[[0, 1, 1], [0, 1, 1]]");
        assert_eq!(failures[0].actual, "[[0, 1, 1], [0, 0, 1]]");
        assert_eq!(metrics.len(), 1);

        // Both checks run: different orders and coefficients fail twice.
        let (failures, _) = run(Some(&airline), Some(&model([[0, 1, 1], [0, 0, 1]], 0.25)));
        let outputs = failures
            .iter()
            .map(|f| f.output.as_str())
            .collect::<Vec<_>>();
        assert_eq!(outputs, ["seats model", "seats coefficients"]);
    }
}
