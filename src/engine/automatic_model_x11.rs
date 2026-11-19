//! Automatic-model regARIMA X-11: `x11ari.f` around automdl's model
//! selection. The shared prologue is `X11Preparation`.

use super::FinalX11Options;
use super::automatic_transform::CandidateModelContext;
use super::post_fit::{PostFitContext, PostFitWarnings, finish};
use super::x11_configuration::X11Preparation;
use crate::regarima::{ModelSelection, ModelSelectionResult, X11ModeSelection};
use crate::time::TimeAxis;
use crate::{Adjustment, Error, RegArimaSpec, Warning, automdl, x11};

pub(crate) fn adjust(
    values: &[f64],
    missing: &[usize],
    axis: TimeAxis,
    requested: RegArimaSpec,
    requested_mode: X11ModeSelection,
    options: x11::Options,
    final_options: FinalX11Options,
) -> Result<Adjustment, Error> {
    debug_assert_eq!(requested.model(), ModelSelection::Automatic);
    requested.validate_model_policy()?;
    let mut preparation = X11Preparation::new(
        values,
        missing,
        axis,
        requested,
        requested_mode,
        options,
        CandidateModelContext::AutomaticModel,
    )?;
    let same = super::missing::skips_model(values, missing);
    let spec = preparation.resolved.spec;
    let pseudo_without_forecasts =
        preparation.mode == x11::Mode::PseudoAdditive && preparation.without_forecasts();
    let mut warnings = editor_and_candidate_warnings(
        std::mem::take(&mut preparation.resolved.warnings),
        same,
        &mut preparation.prepared,
        pseudo_without_forecasts,
    );
    if let Some(warning) = preparation.backcast_warning.take() {
        warnings.insert(0, warning);
    }
    preparation.ensure_registration_capacity()?;
    let easter = preparation.prepare_easter()?;
    if same {
        let mut adjustment =
            preparation.finish_same(easter, final_options.holiday_adjustment, warnings)?;
        adjustment.set_model_selection(ModelSelectionResult::ConstantSeries {
            transform: spec.transform(),
        });
        return Ok(adjustment);
    }

    let regression = preparation.regression()?;
    let (estimation_values, fit_context) = preparation.response(&regression, easter.as_ref())?;
    let input = &preparation.input;
    let fit_context = fit_context
        .with_missing(&input.missing)
        .with_user_regressors(preparation.requested.user_regressors());
    // The selector installs its own canonical baseline; candidate starts are not reused.
    let selected = automdl::select(&estimation_values, fit_context, input.axis, spec)?;
    let outcome = ModelSelectionResult::Selected {
        first_model: selected.first_selected_model,
        mean_significant: selected.procedural_mean,
    };
    warnings.extend(selected.warnings);
    // trnaic/ssprep save Kfmt before enabling the log prior; automd's
    // same-baseline restor recovers that flag. adjreg then omits Sprior
    // when reconstructing the missing-adjusted original (Orixmv).
    let transform_selection = preparation.resolved.selection;
    let restore_missing_prior = !(transform_selection.is_some() && selected.restored_baseline);
    let mut adjustment = finish(
        values,
        selected.terminal.fit,
        PostFitContext {
            series_axis: axis,
            easter,
            mode: preparation.mode,
            final_options,
            prepared_options: preparation.prepared,
            transform_selection,
            restore_missing_prior,
            warnings: PostFitWarnings::Complete(warnings),
            residual_override: selected.terminal.residual_override,
        },
    )?;
    adjustment.set_model_selection(outcome);
    Ok(adjustment)
}

fn editor_and_candidate_warnings(
    transformation: Vec<Warning>,
    same: bool,
    prepared: &mut x11::PreparedOptions,
    pseudo_without_forecasts: bool,
) -> Vec<Warning> {
    let (mut editor, candidate): (Vec<_>, Vec<_>) = transformation
        .into_iter()
        .partition(|warning| matches!(warning, Warning::AutomaticTransformNonPositiveInput { .. }));
    if same {
        editor.push(Warning::ConstantSeries);
    }
    if let Some(warning) = prepared.take_warning() {
        editor.push(warning);
    }
    if pseudo_without_forecasts {
        // editor.f warns in this order: constant series, replaced filter,
        // pseudo-additive without forecasts.
        editor.push(Warning::PseudoAdditiveWithoutForecasts);
    }
    editor.extend(candidate);
    editor
}

#[cfg(test)]
mod tests;
