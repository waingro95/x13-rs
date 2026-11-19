//! The checks `strategy::finish` makes on an airline fit before SEATS runs.
use crate::arima::{ArmaStart, EstimatedRegArima, estimate};
use crate::engine::seats::pipeline::SummaryQSource;
use crate::regression::model::RegressionModel;
use crate::seats::{Failure, Stage};
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, Error, RegArimaSpec};

struct Case {
    original: Vec<f64>,
    axis: TimeAxis,
    fit: EstimatedRegArima,
}

fn spec(regular: [u8; 3], seasonal: [u8; 3]) -> crate::api::ResolvedRegArimaSpec {
    let order = |[p, d, q]: [u8; 3]| ArimaOrder::new(p, d, q).unwrap();
    RegArimaSpec::new(ArimaModel::new(order(regular), order(seasonal)).unwrap())
        .with_constant(false)
        .resolve_explicit()
        .unwrap()
}

/// A fit of the family's model on a positive monthly series.
fn fitted(regular: [u8; 3], seasonal: [u8; 3]) -> Case {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let original = (0..120)
        .map(|t| {
            let angle = std::f64::consts::TAU * (t % 12) as f64 / 12.0;
            100.0 + 0.3 * t as f64 + 10.0 * angle.sin() + ((t * 37 % 101) as f64 - 50.0) * 0.05
        })
        .collect::<Vec<_>>();
    let fit = estimate(
        &original,
        (&original).into(),
        axis,
        spec(regular, seasonal),
        RegressionModel::new(),
        ArmaStart::Default,
    )
    .unwrap();
    Case {
        original,
        axis,
        fit,
    }
}

/// `strategy::finish` on an explicit airline run of `case`.
fn finish_airline(case: Case) -> Result<crate::Adjustment, Error> {
    use crate::engine::seats::pipeline::{
        SeatsFinishContext, SeatsFitWarnings, SeatsTransformedInput,
    };
    let family = crate::api::selected_family(case.fit.spec.model()).unwrap();
    super::finish(
        family,
        &case.original,
        case.axis,
        case.fit,
        SeatsFinishContext {
            summary_q: SummaryQSource::FromFit,
            transformed_input: SeatsTransformedInput::ReconstructOriginal,
            estimation_span: None,
            residual_override: None,
            transform_selection: None,
            warnings: SeatsFitWarnings::BeforeFit(Vec::new()),
        },
    )
}

#[test]
fn a_nonfinite_fitted_coefficient_is_unusable_at_import() {
    let mut case = fitted([0, 1, 1], [0, 1, 1]);
    case.fit.parameters[0] = f64::INFINITY;
    assert_eq!(
        finish_airline(case).unwrap_err(),
        Error::from(
            Failure::Unusable {
                quantity: "fitted model coefficient",
                index: Some(0),
                value: f64::INFINITY,
            }
            .at(Stage::Import)
        )
    );
}
