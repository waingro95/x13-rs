//! The second-difference family's SEATS run on a regARIMA fit.
use super::pipeline::SummaryQSource;
use super::strategy::{Check, ConstantValues, FittedPreconditions};
use crate::Error;
use crate::arima::EstimatedRegArima;
use crate::arima::model::ParameterSlot;
use crate::regarima::{ArimaFactor, ParameterKind};
use crate::seats::effects::FittedEffectsInput;
use crate::seats::pipeline::RunInput;
use crate::seats::second_difference::model::{SecondDifferenceInput, import_second_difference};
use crate::seats::second_difference::{SecondDifferenceRun, compose_second_difference};
use crate::time::TimeAxis;

/// Whether both orders of the fitted model are the second difference's,
/// `(0 2 1)(0 1 1)`.
fn second_difference_orders(fit: &EstimatedRegArima) -> bool {
    let model = fit.spec.model();
    let regular = model.regular();
    let seasonal = model.seasonal();
    (regular.p(), regular.d(), regular.q()) == (0, 2, 1)
        && (seasonal.p(), seasonal.d(), seasonal.q()) == (0, 1, 1)
}

/// Whether each uncertainty value the fit carries has the model's shape and
/// is finite, as are the regression coefficients and columns, the innovation
/// variance and the determinant correction.
fn fitted_values_usable(fitted: &FittedPreconditions<'_>) -> bool {
    let fit = fitted.fit;
    let k = fitted.regression_count();
    let finite = |values: &[f64]| values.iter().all(|value| value.is_finite());
    [
        (fit.parameter_standard_errors.as_deref(), 2),
        (fit.regression_standard_errors.as_deref(), k),
        (fit.arma_covariance.matrix(), 4),
        (fit.regression_covariance.as_deref(), k * (k + 1) / 2),
    ]
    .into_iter()
    .all(|(values, expected)| {
        values.is_none_or(|values| values.len() == expected && finite(values))
    }) && finite(&fit.regression_coefficients)
        && fit
            .regression_model
            .columns()
            .all(|(_, column)| finite(column))
        && fit.innovation_variance.is_finite()
        && fit.log_determinant_correction.is_finite()
}

fn compose_fitted_second_difference(
    original: &[f64],
    axis: TimeAxis,
    fit: &EstimatedRegArima,
    effects: Option<&FittedEffectsInput>,
    summary_q: SummaryQSource,
) -> Result<SecondDifferenceRun, Error> {
    let fitted = FittedPreconditions {
        original,
        axis,
        fit,
        effects,
    };
    debug_assert!(second_difference_orders(fit));
    let period = usize::from(axis.frequency().periods_per_year());
    let slots = [
        ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Regular, 1, 1),
        ParameterSlot::new(ParameterKind::Ma, ArimaFactor::Seasonal, period, period),
    ];
    debug_assert!(fitted.holds(&[
        Check::Span,
        Check::Layout(&slots),
        Check::NoEffects,
        Check::Constant(ConstantValues::Shape),
    ]));
    debug_assert!(fitted_values_usable(&fitted));
    debug_assert!(fitted.holds(&[Check::Residuals, Check::Likelihood]));
    let &[nonseasonal, seasonal] = fit.parameters.as_slice() else {
        return Err(Error::internal(
            "the fit does not carry the model's parameters",
        ));
    };
    let imported = import_second_difference(&SecondDifferenceInput::new(
        axis.frequency(),
        [nonseasonal, seasonal],
    )?);
    let observed_input = RunInput {
        summary_q: summary_q.resolve(fit)?,
        original,
        transformed: &fit.transformed,
        transform: fit.spec.transform().into(),
        requested_mean: super::strategy::requested_mean(fit),
    };
    Ok(compose_second_difference(
        &imported,
        observed_input,
        effects,
    )?)
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
        let observed = compose_fitted_second_difference(original, axis, fit, effects, summary_q)?;
        let diagnostics = diagnostics(&observed)?;
        Ok(super::strategy::Components {
            saved: observed.saved,
            diagnostics,
            transitory: None,
        })
    }
}

fn diagnostics(observed: &SecondDifferenceRun) -> Result<crate::seats::Diagnostics, Error> {
    let passes = super::strategy::PassSummary::new(
        [false; 2],
        observed.passes.iter().map(|pass| {
            (
                pass.decision == crate::seats::airline::RetryDecision::RemoveMeanAndRepeat,
                pass.refit.as_ref().map(|search| search.bound_mask),
            )
        }),
    );
    Ok(crate::seats::Diagnostics {
        mean_requested: observed.requested_mean.into(),
        mean_included: observed.mean.requested.into(),
        model: crate::seats::Model::new(
            observed.model.frequency(),
            [0, 2, 1],
            [0, 1, 1],
            &observed.model.ma_coefficients(),
        )?,
        refitted: passes.refit_occurred,
        mean_temporarily_removed: passes.temporary_mean_removal,
        coefficients_at_bound: passes.bound_fixed.to_vec(),
    })
}
