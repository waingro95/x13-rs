//! The single-MA family's SEATS run on a regARIMA fit.
use super::pipeline::SummaryQSource;
use super::strategy::{Check, ConstantValues, FittedPreconditions};
use crate::Error;
use crate::arima::EstimatedRegArima;
use crate::arima::model::ParameterSlot;
use crate::regarima::{ArimaFactor, ParameterKind};
use crate::seats::effects::FittedEffectsInput;
use crate::seats::pipeline::RunInput;
use crate::seats::seasonal_ma::{
    SeasonalMaInput, SeasonalMaRun, SingleMaShape, compose_seasonal_ma, import_seasonal_ma,
};
use crate::time::TimeAxis;

fn compose_fitted_seasonal_ma(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    effects: Option<&FittedEffectsInput>,
    summary_q: SummaryQSource,
) -> Result<SeasonalMaRun, Error> {
    let fitted = FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    };
    let orders = fit.spec.model();
    let regular = orders.regular();
    let seasonal = orders.seasonal();
    let nonseasonal = match (
        (regular.p(), regular.d(), regular.q()),
        (seasonal.p(), seasonal.d(), seasonal.q()),
    ) {
        ((0, 1, 0), (0, 1, 1)) => false,
        ((0, 1, 1), (0, 0, 0)) => true,
        _ => {
            return Err(Error::internal(
                "the single-MA family received a model of other orders",
            ));
        }
    };
    let slot = if nonseasonal {
        ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Regular, 1, 1)
    } else {
        let period = usize::from(axis.frequency().periods_per_year());
        ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Seasonal, period, period)
    };
    debug_assert!(fitted.holds(&[
        Check::Span,
        Check::Layout(&[slot]),
        Check::NoEffects,
        Check::Constant(ConstantValues::CoefficientsAndColumns),
        Check::FiniteEstimates { parameters: 1 },
    ]));
    let &[ma] = fit.parameters.as_slice() else {
        return Err(Error::internal(
            "the fit does not carry the model's parameters",
        ));
    };
    let imported = import_seasonal_ma(&if nonseasonal {
        SeasonalMaInput::nonseasonal(axis.frequency(), ma)?
    } else {
        SeasonalMaInput::new(axis.frequency(), ma)?
    });
    let input = RunInput {
        summary_q: summary_q.resolve(fit)?,
        original,
        transformed: &fit.transformed,
        transform: fit.spec.transform().into(),
        requested_mean: super::strategy::requested_mean(fit),
    };
    Ok(compose_seasonal_ma(&imported, input, effects)?)
}

pub(super) struct Strategy;

impl super::strategy::FamilyStrategy for Strategy {
    fn compose(
        input: super::strategy::FamilyInput<'_>,
    ) -> Result<super::strategy::Components, Error> {
        let super::strategy::FamilyInput {
            original,
            axis,
            fit,
            effects,
            summary_q,
        } = input;
        let observed = compose_fitted_seasonal_ma(original, axis, fit, effects, summary_q)?;
        let passes = super::strategy::PassSummary::new(
            [false; 1],
            observed.passes.iter().map(|pass| {
                (
                    pass.decision == crate::seats::airline::RetryDecision::RemoveMeanAndRepeat,
                    pass.refit.as_ref().map(|search| search.bound_mask),
                )
            }),
        );
        let (regular, seasonal) = match observed.model.shape {
            SingleMaShape::Nonseasonal => ([0, 1, 1], [0, 0, 0]),
            SingleMaShape::Seasonal => ([0, 1, 0], [0, 1, 1]),
        };
        let diagnostics = crate::seats::Diagnostics {
            mean_requested: observed.requested_mean.into(),
            mean_included: observed.mean.requested.into(),
            model: crate::seats::Model::new(
                axis.frequency(),
                regular,
                seasonal,
                &[observed.model.seasonal_ma()],
            )?,
            refitted: passes.refit_occurred,
            mean_temporarily_removed: passes.temporary_mean_removal,
            coefficients_at_bound: passes.bound_fixed.to_vec(),
        };
        Ok(super::strategy::Components {
            saved: observed.saved,
            diagnostics,
            transitory: None,
        })
    }
}
