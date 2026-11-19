//! The AR(1) family's SEATS run on a regARIMA fit.
use super::pipeline::SummaryQSource;
use super::strategy::{Check, ConstantValues, FittedPreconditions};
use crate::Error;
use crate::arima::EstimatedRegArima;
use crate::arima::model::ParameterSlot;
use crate::regarima::{ArimaFactor, ParameterKind};
use crate::seats::ar1_seasonal_ma::model::{Ar1SeasonalMaInput, import_ar1_seasonal_ma};
use crate::seats::ar1_seasonal_ma::{Ar1Run, compose_ar1};
use crate::seats::effects::FittedEffectsInput;
use crate::seats::pipeline::RunInput;
use crate::time::TimeAxis;

/// Whether the fitted model is `(1 1 0)(0 1 1)` or `(1 1 0)(0 0 0)`.
fn ar1_orders(fit: &EstimatedRegArima) -> bool {
    let model = fit.spec.model();
    let [regular, seasonal] =
        [model.regular(), model.seasonal()].map(|order| (order.p(), order.d(), order.q()));
    regular == (1, 1, 0) && (seasonal == (0, 1, 1) || seasonal == (0, 0, 0))
}

fn compose_fitted_ar1(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    effects: Option<&FittedEffectsInput>,
    summary_q: SummaryQSource,
) -> Result<Ar1Run, Error> {
    let fitted = FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    };
    debug_assert!(ar1_orders(fit));
    let seasonal = fit.spec.model().seasonal();
    let has_seasonal = (seasonal.p(), seasonal.d(), seasonal.q()) == (0, 1, 1);
    let period = usize::from(axis.frequency().periods_per_year());
    let ar = ParameterSlot::new(ParameterKind::Ar, ArimaFactor::Regular, 1, 1);
    let seasonal_ma = ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Seasonal, period, period);
    let slots: &[ParameterSlot] = if has_seasonal {
        &[ar, seasonal_ma]
    } else {
        &[ar]
    };
    debug_assert!(fitted.holds(&[
        Check::Span,
        Check::Layout(slots),
        Check::NoEffects,
        Check::Constant(ConstantValues::CoefficientsAndColumns),
        Check::FiniteEstimates {
            parameters: slots.len(),
        },
    ]));
    let input = match (has_seasonal, fit.parameters.as_slice()) {
        (true, &[phi, seasonal_ma]) => {
            Ar1SeasonalMaInput::new(axis.frequency(), [phi, seasonal_ma])?
        }
        (false, &[phi]) => Ar1SeasonalMaInput::nonseasonal(axis.frequency(), phi)?,
        _ => {
            return Err(Error::internal(
                "the fit does not carry the model's parameters",
            ));
        }
    };
    let imported = import_ar1_seasonal_ma(&input);
    let observed_input = RunInput {
        original,
        transformed: &fit.transformed,
        transform: fit.spec.transform().into(),
        requested_mean: super::strategy::requested_mean(fit),
        summary_q: summary_q.resolve(fit)?,
    };
    Ok(compose_ar1(&imported, observed_input, effects)?)
}

pub(super) struct Strategy;

impl super::strategy::FamilyStrategy for Strategy {
    fn compose(
        input: super::strategy::FamilyInput<'_>,
    ) -> Result<super::strategy::Components, crate::Error> {
        let super::strategy::FamilyInput {
            original,
            axis,
            fit,
            effects,
            summary_q,
        } = input;
        let observed = compose_fitted_ar1(original, axis, fit, effects, summary_q)?;
        let passes = super::strategy::PassSummary::new(
            [false; 2],
            observed.passes.iter().map(|pass| {
                (
                    pass.decision == crate::seats::airline::RetryDecision::RemoveMeanAndRepeat,
                    pass.refit.as_ref().map(|search| search.bound_mask),
                )
            }),
        );
        // Without the seasonal MA, only the AR coordinate is live.
        let (seasonal, count) = if observed.model.has_seasonal() {
            ([0, 1, 1], 2)
        } else {
            ([0, 0, 0], 1)
        };
        let diagnostics = crate::seats::Diagnostics {
            mean_requested: observed.requested_mean.into(),
            mean_included: observed.mean.requested.into(),
            model: crate::seats::Model::new(
                axis.frequency(),
                [1, 1, 0],
                seasonal,
                &observed.model.parameters()[..count],
            )?,
            refitted: passes.refit_occurred,
            mean_temporarily_removed: passes.temporary_mean_removal,
            coefficients_at_bound: passes.bound_fixed[..count].to_vec(),
        };
        Ok(super::strategy::Components {
            saved: observed.saved,
            diagnostics,
            transitory: observed.transitory,
        })
    }
}
