//! `pass2.f`: the comparison of the identified model with the reference
//! model, and the last-resort model that replaces a survivor the residual
//! check rejects. Also the state `automd.f` carries from that comparison
//! through its final checks.

use super::first_pass::FirstPass;
use super::model_check::{ModelCheck, model_check};
use super::outlier_policy::{no_request_outlier_policy, reduced_outlier_cutoffs};
use super::refit::Refit;
use super::restoration::FirstRestoration;
use super::warnings::push_reported_fit;
use crate::api::ResolvedRegArimaSpec;
use crate::arima::append_dynamic_fit_warnings;
use crate::arima::model::ExpandedModel;
use crate::arima::{
    ArmaStart, AutomaticFit, AutomaticInput, EstimatedRegArima, FitContext, FittedNumericalView,
};
use crate::outliers::critical::OutlierCutoffs;
use crate::outliers::search::{
    ClearedRegressionFit, SelectedOutlier, resolve_cutoffs, search_automatic_finalized_input,
};
use crate::regarima::{OutlierSpec, ResolvedArimaModel};
use crate::regression::model::{RegressionModel, RegressorKind};
use crate::time::TimeAxis;
use crate::{Error, Warning};

/// The two statistics of a model check that `pass2.f` compares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ComparisonMetrics {
    pub(crate) confidence: f64,
    pub(crate) innovation_scale: f64,
}

impl From<ModelCheck> for ComparisonMetrics {
    fn from(check: ModelCheck) -> Self {
        Self {
            confidence: check.confidence,
            innovation_scale: check.innovation_scale,
        }
    }
}

/// One side of `pass2.f`'s comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ComparedModel {
    pub(crate) model: ResolvedArimaModel,
    /// Whether the automatic procedure gave the model a constant.
    pub(crate) procedural_mean: bool,
    pub(crate) automatic_outlier_count: usize,
    pub(crate) metrics: ComparisonMetrics,
}

/// The fit the selector currently holds.
pub(crate) enum FallbackFit {
    /// A converged fit.
    Existing(Box<EstimatedRegArima>),
    /// A refit that may not have converged.
    Refit(AutomaticFit),
    /// A fit whose automatic outliers `clrotl` removed for a new search.
    ClearedRegression(ClearedRegressionFit),
}

impl FallbackFit {
    pub(super) fn spec(&self) -> ResolvedRegArimaSpec {
        match self {
            Self::Existing(fit) => fit.spec,
            Self::Refit(fit) => fit.spec(),
            Self::ClearedRegression(fit) => fit.last_fitted().spec,
        }
    }

    pub(crate) fn numerical_view(&self) -> FittedNumericalView<'_> {
        match self {
            Self::Existing(fit) => fit.numerical_view(),
            Self::Refit(fit) => fit.numerical_view(),
            Self::ClearedRegression(fit) => fit.numerical_view(),
        }
    }

    pub(crate) fn innovation_variance(&self) -> f64 {
        match self {
            Self::Existing(fit) => fit.innovation_variance,
            Self::Refit(fit) => fit.innovation_variance(),
            Self::ClearedRegression(fit) => fit.last_fitted().innovation_variance,
        }
    }

    pub(crate) fn residuals(&self) -> &[f64] {
        self.numerical_view().residuals
    }

    pub(crate) fn as_converged(&self) -> Option<&EstimatedRegArima> {
        match self {
            Self::Existing(fit) => Some(fit),
            Self::Refit(fit) => fit.as_converged(),
            Self::ClearedRegression(fit) => fit.as_synchronized(),
        }
    }

    pub(super) fn regression_model(&self) -> &RegressionModel {
        match self {
            Self::Existing(fit) => &fit.regression_model,
            Self::Refit(fit) => fit.regression_model(),
            Self::ClearedRegression(fit) => fit.regression_model(),
        }
    }

    /// `stpitr.f`'s convergence flag as this fit left it.
    pub(super) fn converged(&self) -> bool {
        match self {
            Self::Existing(_) | Self::ClearedRegression(_) => true,
            Self::Refit(fit) => fit.pre_covariance_status().converged,
        }
    }
}

/// How `automd.f` re-enters after a rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetryRoute {
    /// Only the outlier search reruns.
    OutlierSearch,
    /// The differencing and ARMA orders are identified again.
    Identification,
}

/// Which pass of the automatic-model loop is running: Build 62's `nloop` in
/// `automd.f` and `pass2.f`.
///
/// Build 62's `nround` is incremented and reset but never read, so it has no
/// counterpart here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopStage {
    /// `nloop = 1`: the first pass's identified model.
    First,
    /// `nloop = 2`: after one residual rejection or coefficient restart.
    Second,
    /// `nloop = 3`: no retry remains. `pass2.f`'s last resort and `automd.f`'s
    /// constant removal (`IF(nloop.lt.3)nloop=3`) jump here directly.
    Final,
}

impl LoopStage {
    /// The pass a residual rejection or coefficient restart moves to.
    pub(crate) fn next(self) -> Option<Self> {
        match self {
            Self::First => Some(Self::Second),
            Self::Second => Some(Self::Final),
            Self::Final => None,
        }
    }
}

/// What `automd.f` carries from the first comparison through the final
/// checks.
pub(crate) struct FallbackState {
    pub(crate) fit: FallbackFit,
    /// The residuals to report when they are not the fit's own: `pass2.f`
    /// refits the reference model it goes back to but keeps the residuals
    /// that model had before.
    pub(crate) residual_override: Option<Vec<f64>>,
    pub(crate) selected_outliers: Vec<SelectedOutlier>,
    pub(crate) current: ComparedModel,
    pub(crate) reference: ComparedModel,
    /// The residual-mean t statistic of the last `mdlchk` that ran. Going
    /// back to the reference model does not rerun it.
    pub(crate) residual_mean_t: f64,
    /// The outlier critical values the spec resolves to, and what the
    /// reductions so far have left of them.
    pub(crate) original_cutoffs: OutlierCutoffs,
    pub(crate) active_cutoffs: OutlierCutoffs,
}

/// A model ready for the final constant, root, adjustment and coefficient
/// checks.
pub(crate) struct FinalChecks {
    pub(crate) state: FallbackState,
    /// Build 62's `nloop` as the checks start; the constant check can move it
    /// to [`LoopStage::Final`].
    pub(crate) stage: LoopStage,
}

/// The first comparison's rejection with outlier search on: `automd.f`
/// re-enters at pass two with reduced critical values.
pub(crate) struct OutlierRetry {
    pub(crate) state: FallbackState,
    pub(crate) route: RetryRoute,
    pub(crate) policy: OutlierSpec,
}

/// Where the first residual comparison leaves the correction loop.
pub(crate) enum FirstComparison {
    /// Accepted (pass one), or replaced by the last resort (final pass).
    FinalChecks(FinalChecks),
    /// Rejected with outlier search on.
    OutlierRetry(OutlierRetry),
}

impl FallbackState {
    pub(crate) fn reported_residuals(&self) -> &[f64] {
        self.residual_override
            .as_deref()
            .unwrap_or_else(|| self.fit.residuals())
    }

    /// Reruns `mdlchk` on the current model and fit. `n` is the series
    /// length. Every step that replaces the fit ends here.
    pub(crate) fn recheck(&mut self, n: usize) -> Result<(), Error> {
        let model = self.current.model;
        let frequency = self.fit.numerical_view().axis.frequency();
        let expanded = ExpandedModel::from_spec(model, frequency);
        let check = model_check(
            model,
            self.reported_residuals(),
            n - expanded.differencing_loss(),
            frequency,
            expanded.parameter_slots().len(),
            self.fit.innovation_variance(),
        )?;
        self.current.metrics = check.into();
        self.residual_mean_t = check.residual_mean_t;
        Ok(())
    }

    /// Removes the selected automatic outliers from `regression` before a
    /// refit and clears the selection. Does nothing when the current model
    /// counts no automatic outlier.
    pub(crate) fn remove_selected_outliers(&mut self, regression: &mut RegressionModel) {
        if self.current.automatic_outlier_count == 0 {
            return;
        }
        // After `clrotl` with no refit since, the columns are already gone.
        let already_cleared = matches!(
            &self.fit,
            FallbackFit::ClearedRegression(fit) if fit.as_synchronized().is_none()
        );
        for selected in &self.selected_outliers {
            let removed = regression.remove_outlier(selected.kind, selected.period);
            debug_assert!(removed || already_cleared);
        }
        self.selected_outliers.clear();
    }

    /// Whether the state agrees with itself: the current model is the fit's,
    /// and the selected outliers are the fit's outlier regressors.
    pub(super) fn is_consistent(&self) -> bool {
        if self.current.model != self.fit.spec().model() {
            return false;
        }
        if let FallbackFit::ClearedRegression(fit) = &self.fit
            && fit.as_synchronized().is_none()
        {
            let last = fit.last_fitted();
            if selected_match_regressors(
                &last.regressor_kinds,
                last.axis,
                last.transformed.len(),
                &self.selected_outliers,
            ) {
                return true;
            }
        }
        let view = self.fit.numerical_view();
        selected_match_regressors(
            view.regressor_kinds,
            view.axis,
            view.transformed.len(),
            &self.selected_outliers,
        )
    }
}

/// Whether `selected` lists exactly the outlier regressors, in order, each
/// at an index inside the series whose period matches the axis.
fn selected_match_regressors(
    regressor_kinds: &[RegressorKind],
    axis: TimeAxis,
    transformed_len: usize,
    selected: &[SelectedOutlier],
) -> bool {
    let mut outliers = regressor_kinds.iter().filter_map(|kind| match kind {
        RegressorKind::Outlier { kind, period, .. } => Some((*kind, *period)),
        _ => None,
    });
    selected.iter().all(|entry| {
        outliers.next() == Some((entry.kind, entry.period))
            && axis.period_at(entry.index).ok() == Some(entry.period)
            && entry.index < transformed_len
    }) && outliers.next().is_none()
}

/// The fit `pass2.f` restarts from when it goes back to the reference model.
#[derive(Clone, Copy)]
struct SavedReference<'a> {
    fit: &'a EstimatedRegArima,
    selected_outliers: &'a [SelectedOutlier],
    /// The residuals reported with the refit in place of its own.
    residuals: &'a [f64],
}

struct Comparison {
    restored_reference: bool,
    accepted: bool,
}

/// `pass2.f`'s first comparison: the first pass's model against the
/// baseline.
pub(crate) fn first_fallback(
    first: &FirstPass,
    restored: FirstRestoration,
    original: FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<FirstComparison, Error> {
    let baseline = &first.baseline;
    let n = original.raw().len();
    let model = restored.fit.spec.model();
    let automatic_outlier_count = restored.selected_outliers.len();
    let mut state = FallbackState {
        fit: FallbackFit::Existing(Box::new(restored.fit)),
        residual_override: None,
        selected_outliers: restored.selected_outliers,
        current: ComparedModel {
            model,
            procedural_mean: first.search.mean_significant,
            automatic_outlier_count,
            metrics: restored.check.into(),
        },
        reference: ComparedModel {
            model: baseline.fit.spec.model(),
            procedural_mean: baseline
                .fit
                .regressor_kinds
                .contains(&RegressorKind::Constant),
            automatic_outlier_count: baseline.selected_outliers.len(),
            metrics: baseline.check.into(),
        },
        residual_mean_t: restored.check.residual_mean_t,
        original_cutoffs: OutlierCutoffs::NONE,
        active_cutoffs: OutlierCutoffs::NONE,
    };
    debug_assert!(state.is_consistent());
    let policy = baseline.fit.spec.outliers();
    if let Some(policy) = policy {
        state.original_cutoffs = resolve_cutoffs(policy, baseline.fit.axis, n)?;
        state.active_cutoffs = state.original_cutoffs;
    }
    let comparison = compare(
        &mut state,
        SavedReference {
            fit: &baseline.fit,
            selected_outliers: &baseline.selected_outliers,
            residuals: &baseline.fit.residuals,
        },
        LoopStage::First,
        original,
        warnings,
    )?;
    if comparison.accepted {
        return Ok(FirstComparison::FinalChecks(FinalChecks {
            state,
            stage: LoopStage::First,
        }));
    }
    if let Some(policy) = policy {
        state.active_cutoffs =
            reduced_outlier_cutoffs(state.active_cutoffs).unwrap_or(state.active_cutoffs);
        let route = if restored.same_baseline || comparison.restored_reference {
            RetryRoute::OutlierSearch
        } else {
            RetryRoute::Identification
        };
        return Ok(FirstComparison::OutlierRetry(OutlierRetry {
            state,
            route,
            policy,
        }));
    }
    let response = state.fit.numerical_view().transformed.to_vec();
    let selected = state.selected_outliers.clone();
    let fit = fit_last_resort(&mut state, original, baseline.fit.spec, warnings)?;
    let search = search_automatic_finalized_input(
        AutomaticInput::Transformed(&response),
        original,
        fit,
        no_request_outlier_policy(),
        selected,
    )?;
    warnings.extend_from_slice(&search.fit.warnings);
    state.fit = FallbackFit::Existing(Box::new(search.fit));
    state.selected_outliers = search.selected;
    state.residual_override = None;
    state.recheck(n)?;
    state.current.automatic_outlier_count = state.selected_outliers.len();
    // `pass2.f` sets `Nloop=3` after its last-resort model: no retry follows.
    Ok(FirstComparison::FinalChecks(FinalChecks {
        state,
        stage: LoopStage::Final,
    }))
}

/// One comparison of `pass2.f`: goes back to the reference model if
/// [`prefers_reference`] says so, then tests the survivor's Ljung-Box
/// confidence against the limit of `stage`.
fn compare(
    state: &mut FallbackState,
    saved: SavedReference<'_>,
    stage: LoopStage,
    original: FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<Comparison, Error> {
    let model = state.current.model;
    let first_ar = if model.regular().p() > 0 || model.seasonal().p() > 0 {
        state.fit.numerical_view().parameters.first().copied()
    } else {
        None
    };
    let restored_reference = prefers_reference(&state.current, &state.reference, first_ar, stage);
    if restored_reference {
        restore_reference(state, saved, original, warnings)?;
    }
    // The survivor's statistics and outlier count become the reference's;
    // its orders do so only if it is rejected.
    state.reference.metrics = state.current.metrics;
    state.reference.automatic_outlier_count = state.current.automatic_outlier_count;
    let accepted = state.current.metrics.confidence <= acceptance_limit(stage);
    if !accepted {
        state.reference.model = state.current.model;
        state.reference.procedural_mean = state.current.procedural_mean;
    }
    Ok(Comparison {
        restored_reference,
        accepted,
    })
}

/// Refits the reference model from the fit `bkdfmd` saved and makes it the
/// current model.
fn restore_reference(
    state: &mut FallbackState,
    saved: SavedReference<'_>,
    original: FitContext<'_>,
    warnings: &mut Vec<Warning>,
) -> Result<(), Error> {
    // After a reentry the reference model need not be the one `bkdfmd`
    // saved, and then there is no fit to restart from.
    let target = state.reference.model;
    let saved_model = saved.fit.spec.model();
    if target != saved_model {
        return Err(Error::internal(
            "model selection restored state saved for another model",
        ));
    }
    let response = state.fit.numerical_view().transformed.to_vec();
    let axis = saved.fit.axis;
    let regression = saved
        .fit
        .regression_model
        .evaluated_over(axis, original.raw().len())?;
    let refit = Refit {
        response: &response,
        original,
        axis,
        spec: saved.fit.spec,
        regression,
        start: saved.fit.numerical_view().refit_start(),
        carried_convergence: state.fit.converged(),
    }
    .fit()?;
    push_reported_fit(warnings, &refit);
    state.fit = FallbackFit::Refit(refit);
    state.residual_override = Some(saved.residuals.to_vec());
    state.selected_outliers = saved.selected_outliers.to_vec();
    state.current.model = state.reference.model;
    state.current.procedural_mean = state.reference.procedural_mean;
    state.current.metrics = state.reference.metrics;
    Ok(())
}

/// `pass2.f`'s last resort for a model the residual check rejects: a
/// (3,d,1)(P,D,1) model, with one regular AR order fewer each time the fit
/// does not converge. The seasonal AR order is dropped when the model has a
/// seasonal difference.
fn fit_last_resort(
    state: &mut FallbackState,
    original: FitContext<'_>,
    spec: ResolvedRegArimaSpec,
    warnings: &mut Vec<Warning>,
) -> Result<EstimatedRegArima, Error> {
    let axis = state.fit.numerical_view().axis;
    let n = original.raw().len();
    let response = state.fit.numerical_view().transformed.to_vec();
    let seasonal = state.current.model.seasonal();
    let mut p = 3;
    let seasonal_p = if seasonal.d() > 0 { 0 } else { seasonal.p() };
    let mut regression = state.fit.regression_model().clone();
    let mut carried_convergence = state.fit.converged();
    loop {
        let model = ResolvedArimaModel::new(
            [p, state.current.model.regular().d(), 1],
            [seasonal_p, seasonal.d(), 1],
        )?;
        regression.set_constant(state.current.procedural_mean, model, axis.frequency(), n)?;
        regression = regression.evaluated_over(axis, n)?;
        let attempt = Refit {
            response: &response,
            original,
            axis,
            spec: spec
                .with_model(model)
                .with_constant(state.current.procedural_mean),
            regression: regression.clone(),
            start: ArmaStart::Default,
            carried_convergence,
        }
        .fit()?;
        let iterations = attempt.nonlinear_iterations();
        // Build 62 reports the estimation of the first attempt only, and
        // only if it converged.
        if p == 3 && attempt.pre_covariance_status().converged {
            push_reported_fit(warnings, &attempt);
        } else {
            append_dynamic_fit_warnings(warnings, attempt.warnings());
        }
        state.current.model = model;
        match attempt.into_converged() {
            Ok(fit) => return Ok(fit),
            Err(_) if p == 0 => return Err(Error::EstimationDidNotConverge { iterations }),
            Err(_) => {
                p -= 1;
                carried_convergence = false;
            }
        }
    }
}

/// The Ljung-Box confidence a model may have and still pass. `pass2.f`
/// raises `Pcr`, 0.95 by default (`gtinpt.f`), by 0.025 before the first
/// comparison and by 0.015 before each later one; the sums are taken in that
/// order.
fn acceptance_limit(stage: LoopStage) -> f64 {
    let first = 0.95 + 0.025;
    match stage {
        LoopStage::First => first,
        LoopStage::Second | LoopStage::Final => first + 0.015,
    }
}

/// `pass2.f`'s test for going back to the reference model, which it makes
/// only when the two models differ and the reference has no more automatic
/// outliers than the current one. `first_ar` is the current model's first AR
/// coefficient.
fn prefers_reference(
    current: &ComparedModel,
    reference: &ComparedModel,
    first_ar: Option<f64>,
    stage: LoopStage,
) -> bool {
    if reference.automatic_outlier_count > current.automatic_outlier_count
        || (current.model == reference.model
            && current.procedural_mean == reference.procedural_mean)
    {
        return false;
    }
    // `pass2.f`'s names: `Plbox` and `Rvr` are the current model's Ljung-Box
    // confidence and residual scale, `Plbox0` and `Rvr0` the reference's.
    // `Fct` lets the reference's scale be up to 1/(1 - 0.0125) of the
    // current model's.
    let plbox = current.metrics.confidence;
    let plbox0 = reference.metrics.confidence;
    let rvr = current.metrics.innovation_scale;
    let rvr0 = reference.metrics.innovation_scale;
    let fct = 1.0 / (1.0 - 0.0125);
    let regular = current.model.regular();
    let seasonal = current.model.seasonal();

    let both_pass_with_smaller_scale = plbox < 0.95 && plbox0 < 0.75 && rvr0 < rvr;
    let only_reference_passes_first_loop =
        stage == LoopStage::First && plbox >= 0.95 && plbox0 < 0.95;
    let both_pass_with_better_confidence =
        plbox < 0.95 && plbox0 < 0.75 && plbox0 < plbox && rvr0 < fct * rvr;
    let only_reference_passes_with_similar_scale =
        plbox >= 0.95 && plbox0 < 0.95 && rvr0 < fct * rvr;
    // A (1,0,q)(0,1,1) model whose AR coefficient is nearly a regular
    // difference, and a (0,1,1)(1,0,Q) model whose seasonal AR coefficient
    // is nearly a seasonal difference.
    let near_regular_difference = regular.d() == 0
        && seasonal.d() == 1
        && regular.p() == 1
        && seasonal.p() == 0
        && regular.q() <= 1
        && seasonal.q() == 1
        && first_ar.is_some_and(|value| value >= 0.82);
    let near_seasonal_difference = regular.d() == 1
        && seasonal.d() == 0
        && regular.p() == 0
        && seasonal.p() == 1
        && regular.q() == 1
        && seasonal.q() <= 1
        && first_ar.is_some_and(|value| value >= 0.65);

    both_pass_with_smaller_scale
        || only_reference_passes_first_loop
        || both_pass_with_better_confidence
        || only_reference_passes_with_similar_scale
        || near_regular_difference
        || near_seasonal_difference
}

pub(super) mod second;
#[cfg(test)]
mod tests;
