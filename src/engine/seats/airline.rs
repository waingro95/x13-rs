//! The airline family's SEATS run on a regARIMA fit.
use super::pipeline::SummaryQSource;
use super::strategy::{
    Check, Components, ConstantValues, FamilyInput, FamilyStrategy, FittedPreconditions,
    PassSummary, requested_mean,
};
use crate::arima::EstimatedRegArima;
use crate::arima::model::ParameterSlot;
use crate::regarima::{ArimaFactor, ParameterKind};
use crate::seats::airline::model::{AirlineInput, import_airline};
use crate::seats::airline::{AirlineRun, RetryDecision, compose_airline};
use crate::seats::effects::FittedEffectsInput;
use crate::seats::pipeline::RunInput;
use crate::time::TimeAxis;
use crate::{Error, seats};

pub(super) struct Strategy;

impl FamilyStrategy for Strategy {
    fn compose(input: FamilyInput<'_>) -> Result<Components, Error> {
        let FamilyInput {
            original,
            axis,
            fit,
            effects,
            summary_q,
        } = input;
        let observed = compose_fitted_airline(original, axis, fit, effects, summary_q)?;
        let passes = PassSummary::new(
            [false; 2],
            observed.passes.iter().map(|pass| {
                (
                    pass.decision == RetryDecision::RemoveMeanAndRepeat,
                    pass.refit.as_ref().map(|search| search.bound_mask),
                )
            }),
        );
        let diagnostics = seats::Diagnostics {
            mean_requested: observed.requested_mean.into(),
            mean_included: observed.mean.requested.into(),
            model: seats::Model::new(
                observed.model.frequency(),
                [0, 1, 1],
                [0, 1, 1],
                &observed.model.ma_coefficients(),
            )?,
            refitted: passes.refit_occurred,
            mean_temporarily_removed: passes.temporary_mean_removal,
            coefficients_at_bound: passes.bound_fixed.to_vec(),
        };
        Ok(Components {
            saved: observed.saved,
            diagnostics,
            transitory: observed.transitory,
        })
    }
}

/// Whether both orders of the fitted model are `(0 1 1)`.
fn airline_orders(fit: &EstimatedRegArima) -> bool {
    let model = fit.spec.model();
    [model.regular(), model.seasonal()]
        .into_iter()
        .all(|order| (order.p(), order.d(), order.q()) == (0, 1, 1))
}

fn compose_fitted_airline(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    effects: Option<&FittedEffectsInput>,
    summary_q: SummaryQSource,
) -> Result<AirlineRun, Error> {
    let fitted = FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    };
    debug_assert!(airline_orders(fit));
    let period = usize::from(axis.frequency().periods_per_year());
    let slots = [
        ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Regular, 1, 1),
        ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Seasonal, period, period),
    ];
    debug_assert!(fitted.holds(&[
        Check::Span,
        Check::Layout(&slots),
        Check::NoEffects,
        Check::Constant(ConstantValues::Coefficients),
        Check::Uncertainty { parameters: 2 },
    ]));
    let &[nonseasonal, seasonal] = fit.parameters.as_slice() else {
        return Err(Error::internal(
            "the fit does not carry the model's parameters",
        ));
    };
    let imported = import_airline(&AirlineInput::new(
        axis.frequency(),
        [nonseasonal, seasonal],
    )?);
    let observed_input = RunInput {
        original,
        transformed: &fit.transformed,
        transform: fit.spec.transform().into(),
        requested_mean: requested_mean(fit),
        summary_q: summary_q.resolve(fit)?,
    };
    Ok(compose_airline(&imported, observed_input, effects)?)
}
