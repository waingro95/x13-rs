//! Family dispatch and the steps every family shares: the fitted effects,
//! the consistency checks on the fit and the result assembly. Each family
//! module runs its family in `crate::seats`; the `pipeline` module describes
//! the layers.
//!
//! # Consistency checks
//!
//! The estimation returns a fit that spans the series, carries the model's
//! parameter layout and the requested regression, and has finite estimates.
//! The families ask overlapping questions about these, so each is a
//! [`Check`] that [`FittedPreconditions::holds`] answers, and a family
//! asserts the ones it reads with `debug_assert!`. A value a family reads
//! that the estimation does not guarantee, such as a finite fitted
//! coefficient, is a typed error in the family instead.
use super::pipeline::{SeatsFinishContext, SummaryQSource};
use super::{airline, ar1_seasonal_ma, ar2, arma, seasonal_ma, second_difference};
use crate::api::{ArmaCoordinates, SeatsFamily, SeatsStrategy};
use crate::arima::EstimatedRegArima;
use crate::arima::model::{ExpandedModel, ParameterSlot};
use crate::regression::model::RegressorKind;
use crate::seats::FinalComponents;
use crate::seats::airline::{RequestedMean, SeatsTransform};
use crate::seats::effects::FittedEffectsInput;
use crate::time::TimeAxis;
use crate::{Adjustment, Error, Transform, seats};

#[cfg(test)]
mod tests;

/// What a family runs on: the series, the fit, any prepared fitted effects
/// and where the summary Q comes from.
pub(super) struct FamilyInput<'a> {
    pub original: &'a [f64],
    pub axis: TimeAxis,
    pub fit: &'a EstimatedRegArima,
    pub effects: Option<&'a FittedEffectsInput>,
    pub summary_q: SummaryQSource,
}

/// A family's result: the final components, its diagnostics and any
/// transitory component.
pub(super) struct Components {
    pub saved: FinalComponents,
    pub diagnostics: seats::Diagnostics,
    pub transitory: Option<Vec<f64>>,
}

/// A family's checks on the fit, its SEATS run and its diagnostics. SEARCH
/// and CALCFX state stays in `crate::seats`.
pub(super) trait FamilyStrategy {
    fn compose(input: FamilyInput<'_>) -> Result<Components, Error>;
}

type FinishFamily =
    fn(&[f64], TimeAxis, EstimatedRegArima, SeatsFinishContext) -> Result<Adjustment, Error>;

/// Runs `family` on the fit and assembles the adjustment.
pub(super) fn finish(
    family: SeatsFamily,
    original: &[f64],
    axis: TimeAxis,
    fit: EstimatedRegArima,
    context: SeatsFinishContext,
) -> Result<Adjustment, Error> {
    let finish: FinishFamily = match family.strategy() {
        SeatsStrategy::Airline => finish_family::<airline::Strategy>,
        SeatsStrategy::SecondDifference => finish_family::<second_difference::Strategy>,
        SeatsStrategy::SingleMa => finish_family::<seasonal_ma::Strategy>,
        SeatsStrategy::Ar1 => finish_family::<ar1_seasonal_ma::Strategy>,
        SeatsStrategy::Ar2 => finish_family::<ar2::Strategy>,
        SeatsStrategy::Arma(count) => finish_arma(count),
    };
    finish(original, axis, fit, context)
}

/// The ARMA family's finish for `count` coordinates.
fn finish_arma(count: ArmaCoordinates) -> FinishFamily {
    match count {
        ArmaCoordinates::Zero => finish_family::<arma::Strategy<0>>,
        ArmaCoordinates::One => finish_family::<arma::Strategy<1>>,
        ArmaCoordinates::Two => finish_family::<arma::Strategy<2>>,
        ArmaCoordinates::Three => finish_family::<arma::Strategy<3>>,
        ArmaCoordinates::Four => finish_family::<arma::Strategy<4>>,
        ArmaCoordinates::Five => finish_family::<arma::Strategy<5>>,
        ArmaCoordinates::Six => finish_family::<arma::Strategy<6>>,
    }
}

fn finish_family<F: FamilyStrategy>(
    original: &[f64],
    axis: TimeAxis,
    mut fit: EstimatedRegArima,
    context: SeatsFinishContext,
) -> Result<Adjustment, Error> {
    let effects = if fit.spec.calendar().is_some() || has_noncalendar_effects(&fit.regressor_kinds)
    {
        Some(super::effects::prepare_with_transformed_input(
            original,
            axis,
            &fit,
            &context.transformed_input,
        )?)
    } else {
        if !constant_alone(&fit) {
            return Err(Error::internal(
                "the fitted regression is not the requested constant alone",
            ));
        }
        None
    };
    let components = F::compose(FamilyInput {
        original,
        axis,
        fit: &fit,
        effects: effects.as_ref().map(|effects| &effects.input),
        summary_q: context.summary_q,
    })?;
    // The residuals the regARIMA result reports, whose spectrum is computed
    // below: automd's when it overrides the fit's, else the fit's own. A
    // SEATS refit does not change them.
    let reported_residuals = context
        .residual_override
        .as_deref()
        .unwrap_or(&fit.residuals);
    let (estimation_axis, estimation_count) =
        context.estimation_span.unwrap_or((axis, original.len()));
    let residual_spectral = crate::engine::regarima_spectral::compute(
        estimation_axis,
        estimation_count,
        reported_residuals,
        fit.report().status.converged,
        fit.innovation_variance,
    )?;
    let scale = match fit.spec.transform() {
        Transform::None => crate::seats::TransitoryScale::Additive,
        Transform::Log => crate::seats::TransitoryScale::Multiplicative,
    };
    let mut warnings = context.warnings.into_warnings(&mut fit);
    warnings.extend(crate::engine::regarima_spectral::warning(
        &residual_spectral,
    ));
    let mut model =
        crate::api::ModelResult::from_estimated(fit, context.residual_override, residual_spectral);
    model.set_transform_selection(context.transform_selection);
    let mut result = Adjustment::from_seats_components(
        axis.start(),
        components.saved,
        model,
        components.diagnostics,
        warnings,
    );
    if let Some(values) = components.transitory {
        result.set_seats_transitory(crate::seats::Transitory {
            start: axis.start(),
            scale,
            values,
        })?;
    }
    if let Some(effects) = effects
        && let Some(calendar) = effects.calendar_effects
    {
        result.set_seats_calendar_effects(calendar);
    }
    Ok(result)
}

/// Whether the fitted regression is the requested constant alone, with no
/// fixed length factor: what SEATS reads when it prepares no fitted effects.
fn constant_alone(fit: &EstimatedRegArima) -> bool {
    let expected: &[RegressorKind] = if fit.spec.includes_constant() {
        &[RegressorKind::Constant]
    } else {
        &[]
    };
    fit.regression_model.fixed_length().is_empty()
        && fit.regression_model.kinds().eq(expected.iter().cloned())
}

/// Whether the fit carries regression effects besides the constant and the
/// calendar: outliers, missing-value columns or supplied regressors. Those
/// need the fitted-effects preparation, as a configured calendar does.
pub(super) fn has_noncalendar_effects(kinds: &[RegressorKind]) -> bool {
    kinds.iter().any(|kind| {
        matches!(
            kind,
            RegressorKind::Outlier { .. }
                | RegressorKind::MissingValue { .. }
                | RegressorKind::User { .. }
        )
    })
}

/// The fit a family runs on, with the fitted effects prepared for it, if
/// any.
#[derive(Clone, Copy)]
pub(super) struct FittedPreconditions<'a> {
    pub original: &'a [f64],
    pub axis: TimeAxis,
    pub fit: &'a EstimatedRegArima,
    pub effects: Option<&'a FittedEffectsInput>,
}

/// One consistency check on a fit.
#[derive(Clone, Copy)]
pub(super) enum Check<'a> {
    /// The fit covers exactly this axis and series.
    Span,
    /// The model's free parameters sit in exactly these slots, and its ARMA
    /// workspace is the spec's model. The parameter count is not checked:
    /// each family reads the parameters through a pattern of the model's
    /// length.
    Layout(&'a [ParameterSlot]),
    /// Without prepared fitted effects, no calendar, outlier, missing-value,
    /// supplied or fixed-length regression.
    NoEffects,
    /// Without prepared fitted effects, the regression is the requested
    /// constant alone, its columns span the series, and the given values
    /// are finite.
    Constant(ConstantValues),
    /// Any standard errors cover the `parameters` ARMA and every regression
    /// coefficient, and are finite.
    Uncertainty { parameters: usize },
    /// Any standard errors and covariances have the shape for `parameters`
    /// ARMA and every regression coefficient, and are finite, as are the
    /// innovation variance and determinant correction.
    FiniteEstimates { parameters: usize },
    /// Every residual is finite.
    Residuals,
    /// Every likelihood statistic is finite.
    Likelihood,
}

/// The values of a constant-only regression [`Check::Constant`] requires
/// finite.
#[derive(Clone, Copy)]
pub(super) enum ConstantValues {
    /// None: kinds and lengths only.
    Shape,
    /// The coefficients.
    Coefficients,
    /// The coefficients and columns.
    CoefficientsAndColumns,
}

/// Whether `values` are present with other than `count` values, or with a
/// nonfinite one.
fn misshapen(values: Option<&[f64]>, count: usize) -> bool {
    values.is_some_and(|values| values.len() != count || values.iter().any(|v| !v.is_finite()))
}

impl FittedPreconditions<'_> {
    /// Whether every check in `checks` holds.
    pub(super) fn holds(&self, checks: &[Check<'_>]) -> bool {
        checks.iter().all(|&check| self.check_holds(check))
    }

    /// The regression coefficients the standard errors and covariances
    /// cover: every fitted regressor with prepared effects, else the
    /// constant, if any.
    pub(super) fn regression_count(&self) -> usize {
        if self.effects.is_some() {
            self.fit.regressor_kinds.len()
        } else {
            usize::from(self.fit.spec.includes_constant())
        }
    }

    fn check_holds(&self, check: Check<'_>) -> bool {
        let Self {
            original,
            axis,
            fit,
            effects,
        } = *self;
        match check {
            Check::Span => axis == fit.axis && original.len() == fit.transformed.len(),
            Check::Layout(slots) => {
                let model = fit.spec.model();
                ExpandedModel::from_spec(model, axis.frequency()).parameter_slots() == slots
                    && fit.arma_workspace.model() == model
            }
            Check::NoEffects => {
                effects.is_some()
                    || !(fit.spec.calendar().is_some()
                        || has_noncalendar_effects(&fit.regressor_kinds)
                        || !fit.regression_model.fixed_length().is_empty())
            }
            Check::Constant(values) => effects.is_some() || self.constant_only(values),
            Check::Uncertainty { parameters } => {
                !misshapen(fit.parameter_standard_errors.as_deref(), parameters)
                    && !misshapen(
                        fit.regression_standard_errors.as_deref(),
                        self.regression_count(),
                    )
            }
            Check::FiniteEstimates { parameters } => {
                let k = self.regression_count();
                !misshapen(fit.parameter_standard_errors.as_deref(), parameters)
                    && !misshapen(fit.regression_standard_errors.as_deref(), k)
                    && !misshapen(fit.arma_covariance.matrix(), parameters * parameters)
                    && !misshapen(fit.regression_covariance.as_deref(), k * (k + 1) / 2)
                    && fit.innovation_variance.is_finite()
                    && fit.log_determinant_correction.is_finite()
            }
            Check::Residuals => fit.residuals.iter().all(|value| value.is_finite()),
            Check::Likelihood => fit.likelihood.is_none_or(|likelihood| {
                [
                    likelihood.log_likelihood,
                    likelihood.aic,
                    likelihood.hannan_quinn,
                    likelihood.bic,
                ]
                .into_iter()
                .chain(likelihood.aicc)
                .all(f64::is_finite)
            }),
        }
    }

    /// Whether the regression is the requested constant alone, its columns
    /// span the series, and `values` are finite.
    fn constant_only(&self, values: ConstantValues) -> bool {
        let fit = self.fit;
        let constant = fit.spec.includes_constant();
        let expected: &[RegressorKind] = if constant {
            &[RegressorKind::Constant]
        } else {
            &[]
        };
        let finite_coefficients = !matches!(values, ConstantValues::Shape);
        let finite_columns = matches!(values, ConstantValues::CoefficientsAndColumns);
        fit.regressor_kinds == expected
            && fit.regression_model.kinds().eq(expected.iter().cloned())
            && fit.regression_coefficients.len() == usize::from(constant)
            && !(finite_coefficients && fit.regression_coefficients.iter().any(|v| !v.is_finite()))
            && fit.regression_model.columns().all(|(_, column)| {
                column.len() == self.original.len()
                    && !(finite_columns && column.iter().any(|v| !v.is_finite()))
            })
    }
}

/// SEATS imports the regARIMA constant as its requested mean.
pub(super) fn requested_mean(fit: &EstimatedRegArima) -> RequestedMean {
    if fit.spec.includes_constant() {
        RequestedMean::Yes
    } else {
        RequestedMean::No
    }
}

impl From<Transform> for SeatsTransform {
    fn from(transform: Transform) -> Self {
        match transform {
            Transform::None => Self::Identity,
            Transform::Log => Self::Log,
        }
    }
}

impl From<RequestedMean> for bool {
    fn from(mean: RequestedMean) -> Self {
        mean == RequestedMean::Yes
    }
}

/// The pass bookkeeping every family reports: the bound mask of the last
/// SEARCH refit (`initial` when none ran), whether any pass refit, and
/// whether any pass removed the mean and repeated.
pub(super) struct PassSummary<M> {
    pub bound_fixed: M,
    pub refit_occurred: bool,
    pub temporary_mean_removal: bool,
}

impl<M> PassSummary<M> {
    /// Folds the passes in order, each given as whether it removed the mean
    /// and, when SEARCH ran, its bound mask.
    pub(super) fn new(initial: M, passes: impl IntoIterator<Item = (bool, Option<M>)>) -> Self {
        let mut summary = Self {
            bound_fixed: initial,
            refit_occurred: false,
            temporary_mean_removal: false,
        };
        for (removed_mean, refit_mask) in passes {
            summary.temporary_mean_removal |= removed_mean;
            if let Some(mask) = refit_mask {
                summary.refit_occurred = true;
                summary.bound_fixed = mask;
            }
        }
        summary
    }
}
