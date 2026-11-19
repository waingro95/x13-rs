use crate::api::ResolvedRegArimaSpec;
use crate::api::result::ArmaCovariance;
use crate::error::{ExactLikelihoodFailure, NumericalDetail, NumericalOperation};
use crate::numerics::BUILD_62_MACHINE_EPSILON;
use crate::numerics::minpack::{LmCounters, LmEvaluation, LmExit, LmOptions, lmdif};
use crate::regression::design::RegressionMatrix;
use crate::regression::gls::{
    RegressionSolution, SingularRegression, regression_covariance, regression_residuals,
    solve_filtered_regression,
};
use crate::regression::model::{RegressionModel, RegressorKind};
use crate::time::TimeAxis;
use crate::transform;
use crate::{Error, Warning};

use super::covariance::{ParameterCovariance, parameter_covariance};
use super::exact::{
    ArInitialization, ExactFilterError, ExactFilterResult, ExactFilterWorkspace, FilterOptions,
    RootValidation, difference_unscreened, exact_filter,
};
use super::likelihood::{
    FitLikelihood, LikelihoodStatistics, gaussian_log_likelihood, likelihood_statistics,
};
use super::model::ExpandedModel;

mod policy;
use policy::{EstimationPolicy, LikelihoodMode};
mod attempt;
pub(crate) use attempt::FitStop;
use attempt::{FitAttempt, PostOptimizerDecision, post_optimizer_decision};
pub(crate) mod automatic;
mod status;
pub(crate) use status::{FitDiagnostic, FitStatus};
mod reporting;
pub(crate) use reporting::{
    FitReport, FitReportDiagnostic, append_dynamic_fit_warnings, append_reported_fit_warnings,
};
mod workspace;
pub(crate) use workspace::{ArmaCovarianceDiagnostic, ArmaWorkspace};
mod view;
pub(crate) use view::FittedNumericalView;
pub(crate) mod identification;
pub(crate) mod search;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EstimatedRegArima {
    pub(crate) axis: TimeAxis,
    pub(crate) spec: ResolvedRegArimaSpec,
    pub(crate) transformed: Vec<f64>,
    pub(crate) parameters: Vec<f64>,
    pub(crate) parameter_standard_errors: Option<Vec<f64>>,
    pub(crate) regression_coefficients: Vec<f64>,
    pub(crate) regression_standard_errors: Option<Vec<f64>>,
    pub(crate) regressor_kinds: Vec<RegressorKind>,
    pub(crate) regression_model: RegressionModel,
    pub(crate) innovation_variance: f64,
    pub(crate) log_determinant_correction: f64,
    pub(crate) likelihood: Option<LikelihoodStatistics>,
    pub(crate) arma_covariance: ArmaCovariance,
    pub(crate) arma_workspace: ArmaWorkspace,
    pub(crate) arma_covariance_diagnostic: ArmaCovarianceDiagnostic,
    pub(crate) pre_covariance_status: FitStatus,
    pub(crate) regression_covariance: Option<Vec<f64>>,
    pub(crate) regression_factor: Vec<f64>,
    /// The final estimation pass's factor, kept apart because `idotlr.f`
    /// combines it with candidate columns filtered after covariance.
    pub(crate) outlier_test_factor: Vec<f64>,
    pub(crate) forecast_filter_workspace: ExactFilterWorkspace,
    pub(crate) residuals: Vec<f64>,
    pub(crate) nonlinear_tolerances: Vec<f64>,
    pub(crate) overall_iterations: usize,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) warnings: Vec<Warning>,
}

impl EstimatedRegArima {
    /// The status and counts this fit's final warnings are reported from.
    pub(crate) fn report(&self) -> FitReport {
        FitReport {
            status: self.pre_covariance_status,
            covariance_diagnostic: self.arma_covariance_diagnostic,
            nonlinear_iterations: self.nonlinear_iterations,
            arma_parameter_count: self.parameters.len(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ArmaStart {
    Default,
    Previous(Vec<f64>),
    PreviousWithWorkspace {
        parameters: Vec<f64>,
        workspace: ArmaWorkspace,
    },
}

/// The `rgarma.f`/`prlkhd.f` output used by `trnaic.f`, before report-only
/// covariance callbacks. Candidate selection must not depend on `prtmdl.f`.
pub(crate) struct TransformCandidateFit {
    pub(crate) parameters: Vec<f64>,
    pub(crate) innovation_variance: f64,
    pub(crate) likelihood: Option<LikelihoodStatistics>,
    pub(crate) warnings: Vec<Warning>,
}

pub(crate) fn fit_transform_candidate(
    series: &[f64],
    original_series: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
) -> Result<TransformCandidateFit, Error> {
    let request = FitRequest {
        original: original_series,
        axis,
        spec,
        regression_model,
        start,
    };
    let fit = fit(series, request)?;
    Ok(TransformCandidateFit {
        parameters: fit.parameters,
        innovation_variance: fit.innovation_variance,
        likelihood: fit.likelihood.and_then(FitLikelihood::exact_statistics),
        warnings: fit.state.warnings,
    })
}

pub(crate) fn estimate(
    series: &[f64],
    original_series: crate::arima::FitContext<'_>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
) -> Result<EstimatedRegArima, Error> {
    let request = FitRequest {
        original: original_series,
        axis,
        spec,
        regression_model,
        start,
    };
    finish_estimate(fit(series, request)?)
}

struct FittedRegArima {
    status: FitStatus,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    transformed: Vec<f64>,
    model: ExpandedModel,
    matrix: RegressionMatrix,
    regression_model: RegressionModel,
    parameters: Vec<f64>,
    arma_workspace: ArmaWorkspace,
    filter_workspace: ExactFilterWorkspace,
    state: EstimationState,
    nonlinear_iterations: usize,
    effective_observations: usize,
    largest_scaled_residual: f64,
    final_stage: FinalStage,
    innovation_variance: f64,
    likelihood: Option<FitLikelihood>,
    policy: EstimationPolicy,
    initial_log_determinant_correction: f64,
}

/// What one `rgarma.f` fit estimates. The response goes beside it,
/// untransformed to [`fit_attempt`] or transformed to [`fit_transformed`].
#[derive(Clone)]
struct FitRequest<'a> {
    /// The original-scale series the likelihood's transformation adjustment reads.
    original: crate::arima::FitContext<'a>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    start: ArmaStart,
}

/// How one fit runs: the policy's tolerances and budgets, the determinant a
/// conditional retry carries, and whether `stpitr.f`'s convergence flag
/// starts fresh or carries over from the previous automatic fit.
#[derive(Clone, Copy)]
struct FitControl {
    policy: EstimationPolicy,
    initialization: FitInitialization,
    convergence: ConvergenceMode,
}

impl FitControl {
    const fn fresh(policy: EstimationPolicy) -> Self {
        Self {
            policy,
            initialization: FitInitialization::Fresh,
            convergence: ConvergenceMode::Fresh,
        }
    }

    const fn carried(policy: EstimationPolicy, converged: bool) -> Self {
        Self {
            policy,
            initialization: FitInitialization::Fresh,
            convergence: ConvergenceMode::Carried(converged),
        }
    }
}

#[derive(Clone, Copy)]
enum FitInitialization {
    Fresh,
    AfterExact { raw_log_determinant_correction: f64 },
}

impl FitInitialization {
    fn log_determinant_correction(self, mode: LikelihoodMode, model: &ExpandedModel) -> f64 {
        match self {
            Self::AfterExact {
                raw_log_determinant_correction,
            } if mode == LikelihoodMode::ConditionalAr && model.ma_operators().is_empty() => {
                raw_log_determinant_correction
            }
            Self::Fresh | Self::AfterExact { .. } => 0.0,
        }
    }
}

#[derive(Clone, Copy)]
enum ConvergenceMode {
    Fresh,
    Carried(bool),
}

/// An explicit fit, which must complete.
fn fit(series: &[f64], request: FitRequest<'_>) -> Result<FittedRegArima, Error> {
    // rgarma.f `Nefobs.lt.Nextvl` abends here, before the annihilation check
    // and the exact filter: explicit models, every trnaic.f transform
    // candidate (it clears `Lautom` before passing `argok`) and the calendar
    // AIC tests of an explicit model all pass `argok=F`.
    attempt::validate_effective_length(
        series.len(),
        &ExpandedModel::from_spec(request.spec.model(), request.axis.frequency()),
    )?;
    fit_attempt(
        series,
        request,
        FitControl::fresh(EstimationPolicy::EXPLICIT),
    )?
    .into_completed()
}

/// Transforms `series`, then fits it with [`fit_transformed`].
fn fit_attempt(
    series: &[f64],
    request: FitRequest<'_>,
    control: FitControl,
) -> Result<FitAttempt, AttemptFailure> {
    let transformed = transform::forward(series, request.spec.transform())?;
    fit_transformed(transformed, request, control)
}

/// `rgarma.f` on a transformed response: outer GLS passes alternate with
/// nonlinear ARMA steps until the outer loop or the optimizer ends the fit.
fn fit_transformed(
    transformed: Vec<f64>,
    request: FitRequest<'_>,
    control: FitControl,
) -> Result<FitAttempt, AttemptFailure> {
    let mut fit = FitLoop::setup(transformed, request, control)?;
    let final_stage = loop {
        let pass = fit.outer_gls_pass()?;
        if pass.is_final {
            break pass.into_final_stage();
        }
        if let Some(final_stage) = fit.nonlinear_step(pass)? {
            break final_stage;
        }
    };
    Ok(fit.finish(final_stage)?)
}

#[derive(Debug)]
enum AttemptFailure {
    Public(Error),
    Filter {
        cause: ExactFilterError,
        convergence_after_check: Option<bool>,
        warnings: Vec<Warning>,
    },
}

impl AttemptFailure {
    fn into_public(self) -> Error {
        match self {
            Self::Public(error) => error,
            Self::Filter { cause, .. } => map_filter_error(cause),
        }
    }
}

impl From<Error> for AttemptFailure {
    fn from(error: Error) -> Self {
        Self::Public(error)
    }
}

impl From<AttemptFailure> for Error {
    fn from(failure: AttemptFailure) -> Self {
        failure.into_public()
    }
}

/// The locals `rgarma.f` carries across its outer GLS passes and nonlinear
/// steps.
///
/// The steps keep Build 62's order: setup's validation checks, the first
/// pass's capture of the residual a rejected trial reports
/// (`overall_iterations == 0`), and the `estimate_nonlinearly = has_regression`
/// reset after each lmdif run.
struct FitLoop<'a> {
    original: crate::arima::FitContext<'a>,
    axis: TimeAxis,
    spec: ResolvedRegArimaSpec,
    regression_model: RegressionModel,
    transformed: Vec<f64>,
    policy: EstimationPolicy,
    convergence: ConvergenceMode,
    model: ExpandedModel,
    matrix: RegressionMatrix,
    has_regression: bool,
    initial_log_determinant_correction: f64,
    effective_observations: usize,
    overall_deviance_tolerance: f64,
    parameters: Vec<f64>,
    arma_workspace: ArmaWorkspace,
    filter_workspace: ExactFilterWorkspace,
    state: EstimationState,
    nonlinear_counters: LmCounters,
    estimate_nonlinearly: bool,
    /// The residual a rejected nonlinear trial reports, fixed by the first
    /// outer pass.
    largest_scaled_residual: f64,
    last_nonlinear_exit: Option<LmExit>,
    stop: FitStop,
    outer_converged: bool,
    convergence_checked: bool,
}

/// One outer GLS pass: the regression on the filtered matrix, and whether
/// the fit ends on it.
struct GlsPass {
    regression: RegressionSolution,
    filtered_rows: usize,
    raw_log_determinant_correction: f64,
    is_final: bool,
}

impl GlsPass {
    /// Ends the fit on this pass's GLS residuals and determinant.
    fn into_final_stage(self) -> FinalStage {
        let output_residuals = self.regression.residuals().to_vec();
        let determinant = self.raw_log_determinant_correction;
        self.end(output_residuals, determinant)
    }

    fn end(
        self,
        output_residuals: Vec<f64>,
        likelihood_log_determinant_correction: f64,
    ) -> FinalStage {
        FinalStage {
            regression: self.regression,
            filtered_rows: self.filtered_rows,
            raw_log_determinant_correction: self.raw_log_determinant_correction,
            likelihood_log_determinant_correction,
            output_residuals,
        }
    }
}

impl<'a> FitLoop<'a> {
    /// The model, the starting values, the annihilation check and the
    /// root-validated initial filter that fixes the effective sample.
    fn setup(
        transformed: Vec<f64>,
        request: FitRequest<'a>,
        control: FitControl,
    ) -> Result<Self, AttemptFailure> {
        let FitRequest {
            original,
            axis,
            spec,
            regression_model,
            start,
        } = request;
        let policy = control.policy;
        let model = ExpandedModel::from_spec(spec.model(), axis.frequency());
        let initial_log_determinant_correction = control
            .initialization
            .log_determinant_correction(policy.mode(), &model);
        let matrix = regression_model.matrix(&transformed)?;
        let has_regression = matrix.column_count() > 1;
        let (parameters, arma_workspace) =
            starting_parameters_with_workspace(start, &model, spec.model())?;
        // `rgarma.f` rejects a raw series annihilated by differencing before
        // regression can produce a legitimate fitted zero-variance residual.
        if !model.differencing_operators().is_empty()
            && difference_unscreened(&transformed, &model).is_ok_and(|differenced| {
                differenced
                    .iter()
                    .all(|value| crate::numerics::build_62_equal(*value, 0.0))
            })
        {
            return Err(Error::DegenerateSeries.into());
        }
        let mut filter_workspace = ExactFilterWorkspace::default();
        let initial_filter = filter_for(
            &matrix,
            &model,
            &parameters,
            &mut filter_workspace,
            policy.mode(),
            RootValidation::Validate,
        )
        .map_err(|cause| AttemptFailure::Filter {
            cause,
            convergence_after_check: None,
            warnings: Vec::new(),
        })?;
        let effective_observations = initial_filter.effective_observations();
        if effective_observations == 0 {
            return Err(Error::InsufficientEffectiveObservations {
                actual: 0,
                minimum: 1,
            }
            .into());
        }
        let overall_deviance_tolerance =
            policy.overall_deviance_tolerance(effective_observations as f64);
        Ok(Self {
            original,
            axis,
            spec,
            regression_model,
            transformed,
            policy,
            convergence: control.convergence,
            model,
            matrix,
            has_regression,
            initial_log_determinant_correction,
            effective_observations,
            overall_deviance_tolerance,
            parameters,
            arma_workspace,
            filter_workspace,
            state: EstimationState::default(),
            nonlinear_counters: LmCounters::default(),
            estimate_nonlinearly: true,
            // `Lrgrsd` in `rgarma.f`, until the first pass sets it.
            largest_scaled_residual: 1.0e6,
            last_nonlinear_exit: None,
            stop: FitStop::Completed,
            outer_converged: match control.convergence {
                ConvergenceMode::Fresh => true,
                ConvergenceMode::Carried(converged) => converged,
            },
            convergence_checked: false,
        })
    }

    /// Filters and regresses at the current parameters, then runs the outer
    /// convergence check.
    fn outer_gls_pass(&mut self) -> Result<GlsPass, AttemptFailure> {
        // `rgarma.f` re-enters outer GLS with `Lckrts=F`; constrained
        // nonlinear trials and the initial setup remain root-validated.
        let filtered = filter_for(
            &self.matrix,
            &self.model,
            &self.parameters,
            &mut self.filter_workspace,
            self.policy.mode(),
            RootValidation::Skip,
        )
        .map_err(|cause| AttemptFailure::Filter {
            cause,
            convergence_after_check: self.convergence_checked.then_some(self.outer_converged),
            warnings: std::mem::take(&mut self.state.warnings),
        })?;
        let filtered_rows = filtered.matrix().row_count();
        let raw_log_determinant_correction =
            filtered.log_determinant_correction() + self.initial_log_determinant_correction;
        let regression = solve_filtered_regression(&RegressionMatrix::from(filtered.into_matrix()))
            .map_err(|error| singular_regression_error(error, &self.regression_model))?;
        let effective = self.effective_observations as f64;
        let objective =
            regression.residual_sum_squares() * (raw_log_determinant_correction / effective).exp();

        if self.state.overall_iterations == 0 {
            self.largest_scaled_residual = regression
                .residuals()
                .iter()
                .fold(0.0_f64, |largest, value| largest.max(value.abs()))
                * (raw_log_determinant_correction / (2.0 * effective)).exp();
        }

        let continue_outer = self.check_convergence(objective);
        self.state.overall_iterations += 1;
        Ok(GlsPass {
            regression,
            filtered_rows,
            raw_log_determinant_correction,
            is_final: !continue_outer
                || !self.estimate_nonlinearly
                || self.parameters.is_empty()
                || objective == 0.0,
        })
    }

    /// `stpitr.f`; returns whether the outer loop continues.
    fn check_convergence(&mut self, objective: f64) -> bool {
        // A fresh fit always runs `stpitr.f`. In the carried search only,
        // `rgarma.f` skips it unless nonlinear estimation is active with free
        // ARMA parameters.
        let carried = matches!(self.convergence, ConvergenceMode::Carried(_));
        if carried && !(self.estimate_nonlinearly && !self.parameters.is_empty()) {
            return false;
        }
        if carried {
            self.outer_converged = true;
        }
        self.convergence_checked = true;
        let decision = self.state.check_outer_convergence(
            objective,
            self.overall_deviance_tolerance,
            self.nonlinear_counters.iterations,
            self.policy.maximum_overall_iterations(),
        );
        match decision {
            OuterDecision::Continue => true,
            OuterDecision::Complete => false,
            OuterDecision::Stop(reason) => {
                self.stop = reason;
                self.outer_converged = false;
                false
            }
        }
    }

    /// One lmdif run from the pass's raw regression residuals. Returns the
    /// final stage when the fit ends here, or `None` to refit.
    fn nonlinear_step(&mut self, pass: GlsPass) -> Result<Option<FinalStage>, Error> {
        let effective = self.effective_observations as f64;
        let nonlinear_tolerance = self.policy.nonlinear_deviance_tolerance(
            effective,
            self.has_regression,
            self.state.overall_iterations,
        );
        self.state.nonlinear_tolerances.push(nonlinear_tolerance);

        let raw_regression_residuals =
            regression_residuals(&self.matrix, pass.regression.coefficients())?;
        let raw_matrix = RegressionMatrix::without_regressors(&raw_regression_residuals);
        let residuals = ScaledResiduals {
            matrix: &raw_matrix,
            model: &self.model,
            mode: self.policy.mode(),
            effective_observations: effective,
            invalid_residual: self.largest_scaled_residual,
            initial_log_determinant_correction: self.initial_log_determinant_correction,
        };
        let starting_nonlinear_iterations = self.nonlinear_counters.iterations;
        let maximum_iterations = self
            .policy
            .nonlinear_iteration_limit(self.has_regression, starting_nonlinear_iterations);
        let mut nonlinear_residuals = vec![0.0; pass.filtered_rows];
        let mut callback_workspace = ExactFilterWorkspace::default();
        let mut callback_log_determinant_correction = None;
        let outcome = lmdif(
            &mut self.parameters,
            &mut nonlinear_residuals,
            LmOptions {
                function_tolerance: nonlinear_tolerance,
                parameter_tolerance: 0.0,
                gradient_tolerance: 0.0,
                maximum_iterations,
                finite_difference_epsilon: 0.0,
                // The step bound `rgarma.f` passes to `lmdif`.
                factor: 100.0,
            },
            &mut self.nonlinear_counters,
            |candidate, output, evaluation| {
                let roots = if evaluation == LmEvaluation::ConstrainedTrial {
                    RootValidation::Validate
                } else {
                    RootValidation::Skip
                };
                if let Some(correction) =
                    residuals.evaluate(candidate, roots, output, &mut callback_workspace)
                {
                    callback_log_determinant_correction = Some(correction);
                }
                Ok(())
            },
        )
        // The only lmdif error is a nonfinite trial residual during estimation.
        .map_err(|_| Error::numerical(NumericalOperation::Estimation))?;

        if let Some(diagonal) = outcome.work_diagonal {
            self.arma_workspace.replace_diagonal(diagonal);
        }
        self.last_nonlinear_exit = Some(outcome.exit);
        self.estimate_nonlinearly = self.has_regression;
        let decision = post_optimizer_decision(
            outcome.exit,
            starting_nonlinear_iterations,
            self.nonlinear_counters.iterations,
            self.policy.maximum_overall_iterations(),
        );
        Ok(match decision {
            PostOptimizerDecision::Refit => None,
            PostOptimizerDecision::Stop(reason) => {
                self.stop = reason;
                Some(pass.into_final_stage())
            }
            PostOptimizerDecision::Complete => {
                let determinant = callback_log_determinant_correction
                    .unwrap_or(pass.raw_log_determinant_correction);
                Some(pass.end(nonlinear_residuals, determinant))
            }
        })
    }

    /// The `rgarma.f` status and the `prlkhd.f` likelihood, before any
    /// covariance callbacks.
    fn finish(self, final_stage: FinalStage) -> Result<FitAttempt, Error> {
        let status = FitStatus::classify(
            self.outer_converged,
            self.has_regression,
            self.last_nonlinear_exit,
            self.nonlinear_counters.iterations,
            self.policy.maximum_overall_iterations(),
        );
        let innovation_variance = reported_innovation_variance(
            final_stage.regression.residual_sum_squares(),
            self.effective_observations,
        );
        let likelihood = if status.converged && innovation_variance > 0.0 {
            Some(self.likelihood(&final_stage, innovation_variance)?)
        } else {
            None
        };
        Ok(FitAttempt {
            fitted: Box::new(FittedRegArima {
                status,
                axis: self.axis,
                spec: self.spec,
                transformed: self.transformed,
                model: self.model,
                matrix: self.matrix,
                regression_model: self.regression_model,
                parameters: self.parameters,
                arma_workspace: self.arma_workspace,
                filter_workspace: self.filter_workspace,
                state: self.state,
                nonlinear_iterations: self.nonlinear_counters.iterations,
                effective_observations: self.effective_observations,
                largest_scaled_residual: self.largest_scaled_residual,
                final_stage,
                innovation_variance,
                likelihood,
                policy: self.policy,
                initial_log_determinant_correction: self.initial_log_determinant_correction,
            }),
            stop: self.stop,
        })
    }

    fn likelihood(
        &self,
        final_stage: &FinalStage,
        innovation_variance: f64,
    ) -> Result<FitLikelihood, Error> {
        let adjustment = self.original.transformation_adjustment(
            self.regression_model.fixed_length(),
            self.transformed.len() - self.effective_observations,
            self.spec.transform(),
        )?;
        Ok(
            if self.policy.mode() == LikelihoodMode::ConditionalAr
                && !self.model.ar_operators().is_empty()
            {
                FitLikelihood::ConditionalAr {
                    log_likelihood: gaussian_log_likelihood(
                        self.effective_observations,
                        final_stage.likelihood_log_determinant_correction,
                        innovation_variance,
                    ),
                    transformation_adjustment: adjustment,
                }
            } else {
                FitLikelihood::Exact(likelihood_statistics(
                    self.effective_observations,
                    self.parameters.len() + final_stage.regression.coefficients().len() + 1,
                    final_stage.likelihood_log_determinant_correction,
                    innovation_variance,
                    adjustment,
                ))
            },
        )
    }
}

fn finish_estimate(mut fitted: FittedRegArima) -> Result<EstimatedRegArima, Error> {
    let final_stage = &fitted.final_stage;
    let innovation_variance = fitted.innovation_variance;
    let effective_observations = fitted.effective_observations;
    // `prtmdl.f` uses a strict threshold, independently of ARMA covariance.
    let report_uncertainty = uncertainty_is_available(innovation_variance);
    let regression_standard_errors =
        report_uncertainty.then(|| final_stage.regression.standard_errors(innovation_variance));
    let regression_covariance = if report_uncertainty {
        let columns = final_stage.regression.coefficients().len();
        // The covariance scale divides by `n_eff - columns`, so reject here
        // instead of inside `regression_covariance`. A perfect fit skips the
        // covariance and never reaches this check.
        if columns >= effective_observations {
            return Err(Error::RegressionColumnsExceedObservations {
                columns,
                effective_observations,
            });
        }
        Some(regression_covariance(
            final_stage.regression.normal_inverse(),
            innovation_variance,
            effective_observations,
            columns,
        )?)
    } else {
        None
    };
    let regression_factor_length = final_stage.regression.coefficients().len()
        * (final_stage.regression.coefficients().len() + 1)
        / 2;
    let regression_factor =
        final_stage.regression.packed_factor()[..regression_factor_length].to_vec();
    let outlier_test_factor = final_stage.regression.packed_factor().to_vec();
    // Runs only once the regression covariance above succeeds, and pushes
    // its singular warning before the iteration warning below.
    let ArmaCovarianceOutcome {
        covariance: arma_covariance,
        standard_errors: parameter_standard_errors,
        forecast_filter_workspace,
        log_determinant_correction,
        diagnostic: arma_covariance_diagnostic,
    } = fitted.arma_covariance(report_uncertainty)?;
    let FittedRegArima {
        status,
        axis,
        spec,
        transformed,
        model: _,
        matrix: _,
        regression_model,
        parameters,
        arma_workspace,
        filter_workspace: _,
        mut state,
        nonlinear_iterations,
        effective_observations: _,
        largest_scaled_residual: _,
        final_stage,
        innovation_variance,
        likelihood,
        policy: _,
        initial_log_determinant_correction: _,
    } = fitted;
    append_iteration_warning(nonlinear_iterations, &mut state.warnings);
    let regressor_kinds = regression_model.kinds().collect();
    Ok(EstimatedRegArima {
        axis,
        spec,
        transformed,
        parameters,
        parameter_standard_errors,
        regression_coefficients: final_stage.regression.coefficients().to_vec(),
        regression_standard_errors,
        regressor_kinds,
        regression_model,
        innovation_variance,
        likelihood: likelihood.and_then(FitLikelihood::exact_statistics),
        log_determinant_correction,
        arma_covariance,
        arma_workspace,
        arma_covariance_diagnostic,
        pre_covariance_status: status,
        regression_covariance,
        regression_factor,
        outlier_test_factor,
        forecast_filter_workspace,
        residuals: final_stage.output_residuals,
        nonlinear_tolerances: state.nonlinear_tolerances,
        overall_iterations: state.overall_iterations,
        nonlinear_iterations,
        warnings: state.warnings,
    })
}

/// What the ARMA covariance step of [`finish_estimate`] produces.
struct ArmaCovarianceOutcome {
    covariance: ArmaCovariance,
    /// Reported only when the innovation variance is large enough and the
    /// covariance is not singular.
    standard_errors: Option<Vec<f64>>,
    forecast_filter_workspace: ExactFilterWorkspace,
    /// `.lks`'s Lndtcv after the covariance callbacks.
    log_determinant_correction: f64,
    diagnostic: ArmaCovarianceDiagnostic,
}

impl FittedRegArima {
    /// The ARMA parameter covariance from `fcnar.f`'s callbacks around the
    /// final estimates. It replaces the ARMA workspace's diagonal and, when
    /// the covariance is singular, pushes [`Warning::SingularArmaCovariance`].
    /// A model without ARMA parameters needs none.
    fn arma_covariance(
        &mut self,
        report_uncertainty: bool,
    ) -> Result<ArmaCovarianceOutcome, Error> {
        // `.lks` saves Lndtcv after the covariance callbacks; likelihood
        // uses the earlier value from `rgarma.f` before those callbacks.
        let mut log_determinant_correction = self.final_stage.likelihood_log_determinant_correction;
        if self.parameters.is_empty() {
            return Ok(ArmaCovarianceOutcome {
                covariance: ArmaCovariance::NotRequired,
                standard_errors: None,
                forecast_filter_workspace: self.filter_workspace.clone(),
                log_determinant_correction,
                diagnostic: ArmaCovarianceDiagnostic::None,
            });
        }
        let effective = self.effective_observations as f64;
        let overall_deviance_tolerance = self.policy.overall_deviance_tolerance(effective);
        let raw_regression_residuals =
            regression_residuals(&self.matrix, self.final_stage.regression.coefficients())?;
        let residual_matrix = RegressionMatrix::without_regressors(&raw_regression_residuals);
        let covariance_tolerance = self
            .state
            .nonlinear_tolerances
            .last()
            .copied()
            .unwrap_or(overall_deviance_tolerance);
        // The covariance callbacks reuse `fcnar.f` without root checks.
        let residuals = ScaledResiduals {
            matrix: &residual_matrix,
            model: &self.model,
            mode: self.policy.mode(),
            effective_observations: effective,
            invalid_residual: self.largest_scaled_residual,
            initial_log_determinant_correction: self.initial_log_determinant_correction,
        };
        let mut covariance_residuals = vec![0.0; self.final_stage.filtered_rows];
        let mut covariance_workspace = ExactFilterWorkspace::default();
        if let Some(correction) = residuals.evaluate(
            &self.parameters,
            RootValidation::Skip,
            &mut covariance_residuals,
            &mut covariance_workspace,
        ) {
            log_determinant_correction = correction;
        }
        let covariance = parameter_covariance(
            &mut self.parameters,
            &covariance_residuals,
            covariance_tolerance,
            self.innovation_variance,
            |candidate, output| {
                if let Some(correction) = residuals.evaluate(
                    candidate,
                    RootValidation::Skip,
                    output,
                    &mut covariance_workspace,
                ) {
                    log_determinant_correction = correction;
                }
            },
        );
        let (covariance, standard_errors, diagnostic) = match covariance {
            ParameterCovariance::Calculated {
                matrix: covariance,
                unscaled_diagonal,
            } => {
                self.arma_workspace.replace_diagonal(unscaled_diagonal);
                let count = self.parameters.len();
                let standard_errors = report_uncertainty.then(|| {
                    (0..count)
                        .map(|index| covariance[index + index * count].sqrt())
                        .collect()
                });
                (
                    ArmaCovariance::Calculated(covariance),
                    standard_errors,
                    ArmaCovarianceDiagnostic::None,
                )
            }
            ParameterCovariance::Singular { unscaled_diagonal } => {
                self.arma_workspace.replace_diagonal(unscaled_diagonal);
                self.state.warnings.push(Warning::SingularArmaCovariance);
                (
                    ArmaCovariance::Singular,
                    None,
                    ArmaCovarianceDiagnostic::Singular,
                )
            }
        };
        Ok(ArmaCovarianceOutcome {
            covariance,
            standard_errors,
            forecast_filter_workspace: covariance_workspace,
            log_determinant_correction,
            diagnostic,
        })
    }
}

fn singular_regression_error(
    error: SingularRegression,
    regression_model: &RegressionModel,
) -> Error {
    let Some(variable) = regression_model
        .kinds()
        .nth(error.pivot)
        .map(RegressorKind::variable)
    else {
        return Error::internal("singular regression pivot is past the last regressor");
    };
    Error::SingularRegression {
        pivot: error.pivot,
        variable,
    }
}

fn starting_parameters(start: ArmaStart, model: &ExpandedModel) -> Result<Vec<f64>, Error> {
    let parameter_count = model.parameter_slots().len();
    match start {
        ArmaStart::Default => Ok(vec![0.1; parameter_count]),
        ArmaStart::Previous(mut parameters)
        | ArmaStart::PreviousWithWorkspace { mut parameters, .. }
            if parameters.len() == parameter_count
                && parameters.iter().all(|parameter| parameter.is_finite()) =>
        {
            // `setmdl.f` records strict inside-root failure before applying
            // its later-call near-unit shrinkage.
            let inside_ma_factor = model
                .first_inside_ma_factor(&parameters)
                .map_err(|_| Error::numerical(NumericalOperation::Estimation))?;
            if let Some(factor) = inside_ma_factor {
                return Err(Error::from(NumericalDetail::InvalidMaStartingRoots {
                    factor,
                }));
            }
            model.prepare_previous_parameters(&mut parameters);
            Ok(parameters)
        }
        ArmaStart::Previous(_) | ArmaStart::PreviousWithWorkspace { .. } => Err(Error::internal(
            "previous ARMA parameters do not fit the model",
        )),
    }
}

fn starting_parameters_with_workspace(
    start: ArmaStart,
    model: &ExpandedModel,
    resolved_model: crate::regarima::ResolvedArimaModel,
) -> Result<(Vec<f64>, ArmaWorkspace), Error> {
    let workspace = match &start {
        ArmaStart::PreviousWithWorkspace { workspace, .. }
            if workspace.model() == resolved_model
                && workspace.diagonal().len() == model.parameter_slots().len() =>
        {
            workspace.clone()
        }
        ArmaStart::PreviousWithWorkspace { .. } => {
            return Err(Error::internal(
                "previous ARMA workspace was built for another model",
            ));
        }
        _ => ArmaWorkspace::reset(resolved_model),
    };
    Ok((starting_parameters(start, model)?, workspace))
}

pub(crate) fn uncertainty_is_available(innovation_variance: f64) -> bool {
    innovation_variance > 2.0 * BUILD_62_MACHINE_EPSILON
}

fn reported_innovation_variance(residual_sum_squares: f64, effective_observations: usize) -> f64 {
    let variance = residual_sum_squares / effective_observations as f64;
    if variance < 2.0 * BUILD_62_MACHINE_EPSILON {
        0.0
    } else {
        variance
    }
}

#[derive(Debug, Default)]
struct EstimationState {
    previous_objective: Option<f64>,
    overall_iterations: usize,
    nonlinear_tolerances: Vec<f64>,
    warnings: Vec<Warning>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OuterDecision {
    Continue,
    Complete,
    Stop(FitStop),
}

impl EstimationState {
    fn check_outer_convergence(
        &mut self,
        objective: f64,
        deviance_tolerance: f64,
        nonlinear_iterations: usize,
        maximum_overall_iterations: usize,
    ) -> OuterDecision {
        if nonlinear_iterations >= maximum_overall_iterations {
            return OuterDecision::Stop(FitStop::OverallIterations);
        }

        let mut continue_estimation = true;
        if self.overall_iterations > 1 && !crate::numerics::build_62_equal(objective, 0.0) {
            let ratio = self.previous_objective.unwrap_or(0.0) / objective - 1.0;
            if deviance_tolerance / 2.0 < BUILD_62_MACHINE_EPSILON {
                return OuterDecision::Stop(FitStop::ToleranceBelowPrecision);
            }
            if ratio < 0.0 {
                self.warnings.push(Warning::DevianceIncreased);
            }
            if ratio.abs() < deviance_tolerance {
                continue_estimation = false;
            } else if objective < BUILD_62_MACHINE_EPSILON {
                // `stpitr.f` stops with Convrg true; `rgarma.f` subsequently
                // replaces intermediate PDVTER with the last MINPACK status.
                continue_estimation = false;
            }
        }
        self.previous_objective = Some(objective);
        if continue_estimation {
            OuterDecision::Continue
        } else {
            OuterDecision::Complete
        }
    }
}

/// What ends a fit: the last GLS regression, with the filter result's row
/// count and determinant kept in place of the filtered matrix it consumed.
struct FinalStage {
    regression: RegressionSolution,
    filtered_rows: usize,
    /// The outer pass's determinant; identification's conditional retry
    /// starts from it.
    raw_log_determinant_correction: f64,
    likelihood_log_determinant_correction: f64,
    output_residuals: Vec<f64>,
}

/// `fcnar.f`: the filtered residuals at trial parameters, scaled by the
/// determinant so their sum of squares is the deviance lmdif minimizes. The
/// nonlinear step and the covariance callbacks share it and differ only in
/// root validation.
#[derive(Clone, Copy)]
struct ScaledResiduals<'a> {
    matrix: &'a RegressionMatrix,
    model: &'a ExpandedModel,
    mode: LikelihoodMode,
    effective_observations: f64,
    /// Fills the output when the filter rejects the parameters.
    invalid_residual: f64,
    initial_log_determinant_correction: f64,
}

impl ScaledResiduals<'_> {
    /// Writes the residuals at `parameters` into `output`, returning the
    /// determinant correction when the filter accepts them.
    fn evaluate(
        &self,
        parameters: &[f64],
        roots: RootValidation,
        output: &mut [f64],
        workspace: &mut ExactFilterWorkspace,
    ) -> Option<f64> {
        match filter_for(
            self.matrix,
            self.model,
            parameters,
            workspace,
            self.mode,
            roots,
        ) {
            Ok(filtered) => {
                let correction =
                    filtered.log_determinant_correction() + self.initial_log_determinant_correction;
                let scale = (correction / (2.0 * self.effective_observations)).exp();
                let values = filtered.matrix().as_slice();
                debug_assert_eq!(output.len(), values.len());
                for (output, value) in output.iter_mut().zip(values) {
                    *output = scale * value;
                }
                Some(correction)
            }
            Err(_) => {
                output.fill(self.invalid_residual);
                None
            }
        }
    }
}

fn filter_for(
    matrix: &RegressionMatrix,
    model: &ExpandedModel,
    parameters: &[f64],
    workspace: &mut ExactFilterWorkspace,
    mode: LikelihoodMode,
    roots: RootValidation,
) -> Result<ExactFilterResult, ExactFilterError> {
    let ar_initialization = match mode {
        LikelihoodMode::Exact => ArInitialization::Exact,
        LikelihoodMode::ConditionalAr => ArInitialization::Conditional,
    };
    exact_filter(
        matrix.as_matrix().clone(),
        model,
        parameters,
        workspace,
        FilterOptions::computed(ar_initialization, roots),
    )
}

fn append_iteration_warning(iterations: usize, warnings: &mut Vec<Warning>) {
    if iterations > 200 {
        warnings.push(Warning::EstimationRequiredManyIterations { iterations });
    }
}

impl From<ExactFilterError> for Error {
    fn from(error: ExactFilterError) -> Self {
        map_filter_error(error)
    }
}

fn map_filter_error(error: ExactFilterError) -> Error {
    match error {
        ExactFilterError::InvalidRoots { kind } => {
            Error::from(NumericalDetail::InvalidRoots { kind })
        }
        ExactFilterError::InsufficientObservations { actual, minimum } => {
            Error::InsufficientEffectiveObservations { actual, minimum }
        }
        ExactFilterError::SingularMaInitialCovariance { pivot } => {
            exact_likelihood(ExactLikelihoodFailure::SingularMaInitialCovariance { pivot })
        }
        ExactFilterError::SingularArInitialCovariance { pivot } => {
            exact_likelihood(ExactLikelihoodFailure::SingularArInitialCovariance { pivot })
        }
        ExactFilterError::InvalidArCovariance => {
            exact_likelihood(ExactLikelihoodFailure::ArmaAutocovariance)
        }
        // Matrix shape and parameter count are internal invariants.
        ExactFilterError::InvalidDimensions | ExactFilterError::InvalidParameterCount { .. } => {
            Error::internal("exact filter input has the wrong shape")
        }
    }
}

fn exact_likelihood(failure: ExactLikelihoodFailure) -> Error {
    Error::from(NumericalDetail::ExactLikelihoodUnavailable { failure })
}

#[cfg(test)]
mod tests;
