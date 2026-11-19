//! The automatic model selection of `automd.f`.

use super::corrections::complete_corrections;
use super::first_pass::first_pass;
use super::terminal::{TerminalFit, finish_fit};
use super::warnings::push_many_iterations;
use crate::api::ResolvedRegArimaSpec;
use crate::time::TimeAxis;
use crate::{ArimaModel, Error, Warning};

pub(crate) struct SelectedModel {
    pub(crate) terminal: TerminalFit,
    /// The model the first order search chose.
    pub(crate) first_selected_model: ArimaModel,
    /// Whether the automatic procedure gave the model a constant.
    pub(crate) procedural_mean: bool,
    /// `restor` recovered the baseline's estimates for the first pass's
    /// model.
    pub(crate) restored_baseline: bool,
    pub(crate) warnings: Vec<Warning>,
}

/// Selects and fits a model for the transformed `series`.
pub(crate) fn select(
    series: &[f64],
    original: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
) -> Result<SelectedModel, Error> {
    let mut warnings = Vec::new();
    let first = first_pass(series, original, axis, spec, &mut warnings)?;
    let run = complete_corrections(&first, original, &mut warnings)?;
    debug_assert!(run.state.is_consistent());
    let procedural_mean = run.state.current.procedural_mean;
    let terminal = finish_fit(run.state.fit, run.state.residual_override, original)?;
    push_many_iterations(&mut warnings, terminal.fit.report());
    Ok(SelectedModel {
        terminal,
        first_selected_model: first.search.model,
        procedural_mean,
        restored_baseline: run.restored_baseline,
        warnings,
    })
}
