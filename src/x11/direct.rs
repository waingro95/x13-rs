//! The direct X-11 run: Build 62's `x11pt2.f` and `x11pt3.f`.
//!
//! `RunContext` holds what one run validates once (span, axis, robust-scale
//! geometry, section filters). `Iteration` carries what the passes hand
//! forward: the modified series, the extreme-scale state, the Henderson I/C
//! ratio, warnings and the C extremes. Each of B, C and D runs `initial_si`
//! (centered SI) and `estimate_trend` (seasonal, Henderson trend, SI against
//! it); B and C then run `weight_extremes` (refined seasonal, irregular,
//! extreme weights) and `remove_extremes` (next modified series). `part_d` is
//! Part D's shared step: the D8 diagnostics, the MSR choice, then the final
//! seasonal (D10), trend-cycle (D12), and adjusted series and irregular (D11,
//! D13).
//!
//! A run ends one of two ways. `run_components` (ordinary runs and the legacy
//! Easter final run) makes the seasonality tests and finishes with
//! `into_components`; `run_preliminary` (the legacy Easter prepass) skips them
//! and finishes with `into_preliminary`, keeping what the final run resumes
//! from. `SeasonalityTests` carries that difference through `run`.
//!
//! Fallible calls and warning pushes follow Build 62's order, which decides
//! the error an input gets and the order of the warning list.

use std::ops::Range;

use crate::error::X11ExtremeStage;
use crate::time::{AxisSpan, TimeAxis};
use crate::{Error, Warning, x11};

use super::seasonal::{ResolvedSeasonalFilter, SeasonalFilterSet};
use super::{Components, PreparedOptions};

mod extreme_scale;
mod extreme_weights;
mod mode;
mod part_d;
mod passes;
mod seasonal_factors;
mod trend;

#[cfg(test)]
mod tests;

use extreme_scale::{ExtremeScaleState, RobustContext, SigmaState};
use extreme_weights::{ExtremeSpans, extreme_components};
use part_d::PartD;
use seasonal_factors::{SeasonalSmoothingRequest, validate_initial_seasonal_history};

fn extension_geometry() -> Error {
    Error::internal("X-11 span lies outside the extended series")
}

/// Which of `x11ari.f`'s X-11 runs this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunPurpose {
    /// A run without legacy Easter adjustment.
    Ordinary,
    /// The legacy Easter prepass, whose irregular estimates the Easter effect.
    LegacyEasterPreliminary,
    /// The run on the Easter-adjusted series, resuming from the prepass.
    LegacyEasterFinal,
}

/// What the legacy Easter prepass leaves for the final run. An ordinary run
/// starts from the default.
#[derive(Default)]
struct PrepassState {
    /// The calendar-sigma state after the prepass's tests.
    sigma: Option<SigmaState>,
    /// Whether the prepass chose an MSR filter, which the final run then
    /// does not choose again.
    consumed_msr: bool,
    history: Option<super::quality::QualityFilterHistory>,
    /// Which observations the prepass left eligible for the quality measures.
    eligible: Option<Vec<bool>>,
}

/// One run's series and options.
struct RunInput<'a> {
    /// The series the passes decompose, with any extensions.
    values: &'a [f64],
    /// The series D11 adjusts in the legacy Easter final run: `values` before
    /// the Easter effect was removed.
    raw: &'a [f64],
    axis: TimeAxis,
    /// The observed span of `values`.
    observed: Range<usize>,
    mode: x11::Mode,
    prepared: PreparedOptions,
}

/// The legacy Easter prepass's results.
struct Prepass {
    irregular: Vec<f64>,
    warnings: Vec<crate::diagnostics::Unavailable>,
    trend_warnings: Vec<crate::x11::EasterPreliminaryTrendWarning>,
    msr: Option<crate::x11::MsrSelection>,
    sigma: Option<crate::x11::CalendarSigmaSelection>,
    state: PrepassState,
}

/// An X-11 run with legacy Easter adjustment (`x11ari.f`) on a series without
/// extensions: the prepass, the Easter estimate, then the final run.
pub(crate) fn adjust_easter(
    values: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    prepared: PreparedOptions,
) -> Result<(Components, crate::x11::EasterEffect), Error> {
    let state = prepare_easter(values, axis, mode, &prepared, 0, 0)?;
    let working = state.remove_observed_prior(values)?;
    finish_easter(
        &working,
        values,
        axis,
        0..values.len(),
        mode,
        prepared,
        state,
    )
}

/// The legacy Easter prepass and the Easter factors estimated from its
/// irregular, over the observed span and the requested extensions.
pub(crate) struct PreparedEaster {
    prior: super::easter::EstimatedEasterPrior,
    preliminary: Prepass,
    /// The axis of the observed series.
    axis: TimeAxis,
    /// The axis of `prior.factors`, which starts with the requested backcasts.
    factor_axis: TimeAxis,
    /// The observed span within `prior.factors`.
    observed: Range<usize>,
}

impl PreparedEaster {
    /// The Easter factors of the observed span.
    pub(crate) fn observed_factors(&self) -> &[f64] {
        &self.prior.factors[self.observed.clone()]
    }

    /// The `count` Easter factors from the start of `axis`.
    pub(crate) fn factors_at(&self, axis: TimeAxis, count: usize) -> Result<&[f64], Error> {
        let first = (i64::from(axis.start().year()) - i64::from(self.factor_axis.start().year()))
            * 12
            + i64::from(axis.start().subperiod())
            - i64::from(self.factor_axis.start().subperiod());
        let first = usize::try_from(first).map_err(|_| extension_geometry())?;
        let end = first.checked_add(count).ok_or_else(extension_geometry)?;
        self.prior
            .factors
            .get(first..end)
            .ok_or_else(extension_geometry)
    }

    /// `values` with the observed span's Easter factors removed.
    pub(crate) fn remove_observed_prior(&self, values: &[f64]) -> Result<Vec<f64>, Error> {
        super::easter::apply_prior(values, self.observed_factors(), self.axis)
    }
}

/// The legacy Easter prepass on the observed series, and the Easter factors
/// its irregular gives over that span and the requested extensions.
pub(crate) fn prepare_easter(
    values: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    prepared: &PreparedOptions,
    requested_forecasts: usize,
    requested_backcasts: usize,
) -> Result<PreparedEaster, Error> {
    let mut prepared = prepared.clone();
    // The final run reports the editor's warning.
    prepared.take_warning();
    let preliminary = run_preliminary(RunInput {
        values,
        raw: values,
        axis,
        observed: 0..values.len(),
        mode,
        prepared,
    })?;
    let prior = super::easter::estimate_requested(
        &preliminary.irregular,
        axis,
        requested_forecasts,
        requested_backcasts,
    )?;
    Ok(PreparedEaster {
        prior,
        preliminary,
        axis,
        factor_axis: axis
            .shift(-i64::try_from(requested_backcasts).map_err(|_| extension_geometry())?)?,
        observed: requested_backcasts..requested_backcasts + values.len(),
    })
}

/// The final run of a legacy Easter adjustment: X-11 on `working`, the series
/// with the Easter effect removed, resuming from the prepass. D11 adjusts
/// `raw`.
pub(crate) fn finish_easter(
    working: &[f64],
    raw: &[f64],
    axis: TimeAxis,
    observed: Range<usize>,
    mode: x11::Mode,
    prepared: PreparedOptions,
    state: PreparedEaster,
) -> Result<(Components, crate::x11::EasterEffect), Error> {
    let factors = state.observed_factors().to_vec();
    let PreparedEaster {
        prior,
        preliminary,
        axis: observed_axis,
        ..
    } = state;
    let components = run_components(
        RunInput {
            values: working,
            raw,
            axis,
            observed,
            mode,
            prepared,
        },
        RunPurpose::LegacyEasterFinal,
        preliminary.state,
    )?;
    Ok((
        components,
        crate::x11::EasterEffect {
            start: observed_axis.start(),
            factors,
            status: prior.status,
            preliminary_warnings: preliminary.warnings,
            preliminary_trend_warnings: preliminary.trend_warnings,
            preliminary_msr_selection: preliminary.msr,
            preliminary_calendar_sigma_selection: preliminary.sigma,
        },
    ))
}

/// An X-11 run on a series without extensions.
pub(crate) fn adjust(
    values: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    options: x11::Options,
) -> Result<Components, Error> {
    adjust_extended(
        values,
        axis,
        0..values.len(),
        mode,
        PreparedOptions::new(options, axis, values.len())?,
    )
}

/// An X-11 run on a series whose `observed` span is extended with forecasts
/// and backcasts.
pub(crate) fn adjust_extended(
    values: &[f64],
    axis: TimeAxis,
    observed: Range<usize>,
    mode: x11::Mode,
    prepared: PreparedOptions,
) -> Result<Components, Error> {
    run_components(
        RunInput {
            values,
            raw: values,
            axis,
            observed,
            mode,
            prepared,
        },
        RunPurpose::Ordinary,
        PrepassState::default(),
    )
}

/// A run that ends in components: an ordinary run, or the legacy Easter
/// final run resuming from the prepass's `state`.
fn run_components(
    input: RunInput<'_>,
    purpose: RunPurpose,
    state: PrepassState,
) -> Result<Components, Error> {
    debug_assert_ne!(purpose, RunPurpose::LegacyEasterPreliminary);
    let (context, iteration, part_d) = run::<WithSeasonalityTests>(input, purpose, state)?;
    Ok(iteration.into_components(&context, part_d))
}

/// The legacy Easter prepass, which ends in the state the final run resumes
/// from instead of components.
fn run_preliminary(input: RunInput<'_>) -> Result<Prepass, Error> {
    let (context, iteration, part_d) = run::<WithoutSeasonalityTests>(
        input,
        RunPurpose::LegacyEasterPreliminary,
        PrepassState::default(),
    )?;
    iteration.into_preliminary(&context, part_d)
}

/// Whether a run makes the seasonality tests.
trait SeasonalityTests {
    /// What Part D hands on.
    type Diagnostics;

    /// The test on pass B's initial SI.
    fn preliminary(context: &RunContext<'_>, si: &[f64], warnings: &mut Vec<Warning>) -> Self;

    /// Part D's tests on the unmodified D8.
    fn complete(
        self,
        context: &RunContext<'_>,
        d8: &[f64],
        warnings: &mut Vec<Warning>,
    ) -> Self::Diagnostics;
}

/// A run that makes the seasonality tests. It holds the stable-seasonality
/// test of pass B's initial SI until Part D completes the set.
struct WithSeasonalityTests(
    Result<crate::diagnostics::StableSeasonalityTest, crate::diagnostics::Unavailable>,
);

impl SeasonalityTests for WithSeasonalityTests {
    type Diagnostics = crate::diagnostics::SeasonalityTests;

    fn preliminary(context: &RunContext<'_>, si: &[f64], warnings: &mut Vec<Warning>) -> Self {
        let &RunContext {
            axis,
            mode,
            period,
            ref observed,
            ..
        } = context;
        let stable = super::seasonality::stable(
            si,
            axis,
            observed.start + period / 2..observed.end - period / 2,
            mode,
            crate::diagnostics::SeasonalityTest::PreliminaryStable,
        );
        super::seasonality::append_warning(&stable, warnings);
        Self(stable)
    }

    fn complete(
        self,
        context: &RunContext<'_>,
        d8: &[f64],
        warnings: &mut Vec<Warning>,
    ) -> Self::Diagnostics {
        let seasonality = super::seasonality::core(
            self.0,
            d8,
            context.axis,
            context.observed.clone(),
            context.mode,
        );
        super::seasonality::append_warning(&seasonality.final_stable, warnings);
        super::seasonality::append_warning(&seasonality.moving, warnings);
        seasonality
    }
}

/// A run that skips the seasonality tests: the legacy Easter prepass.
struct WithoutSeasonalityTests;

impl SeasonalityTests for WithoutSeasonalityTests {
    type Diagnostics = ();

    fn preliminary(_: &RunContext<'_>, _: &[f64], _: &mut Vec<Warning>) -> Self {
        Self
    }

    fn complete(self, _: &RunContext<'_>, _: &[f64], _: &mut Vec<Warning>) -> Self::Diagnostics {}
}

/// One direct X-11 run: B and C each estimate a trend and reweight the
/// extremes, D estimates the final-iteration trend, and Part D's shared step
/// builds the final series.
fn run<'a, T: SeasonalityTests>(
    mut input: RunInput<'a>,
    purpose: RunPurpose,
    state: PrepassState,
) -> Result<(RunContext<'a>, Iteration, PartD<T::Diagnostics>), Error> {
    let warning = input.prepared.take_warning();
    let context = RunContext::new(input, purpose)?;
    let mut iteration = Iteration::new(&context, state.sigma, warning)?;

    let initial = iteration.initial_si(&context)?;
    let tests = T::preliminary(&context, &initial.si, &mut iteration.warnings);
    let (si, trend) = iteration.estimate_trend(&context, Pass::B, initial)?;
    let weighted = iteration.weight_extremes(&context, Pass::B, si, &trend.values)?;
    iteration.remove_extremes(&context, Pass::B, &trend.values, weighted)?;

    let initial = iteration.initial_si(&context)?;
    let (si, trend) = iteration.estimate_trend(&context, Pass::C, initial)?;
    let weighted = iteration.weight_extremes(&context, Pass::C, si, &trend.values)?;
    // Part D's tests read D8 rebuilt from these same rounded C extremes.
    let diagnostic_c_extremes =
        extreme_components(context.mode, &weighted.irregular, &weighted.weights).collect();
    iteration.remove_extremes(&context, Pass::C, &trend.values, weighted)?;

    let initial = iteration.initial_si(&context)?;
    let (final_si, final_iteration_trend) = iteration.estimate_trend(&context, Pass::D, initial)?;
    let part_d = iteration.part_d(
        &context,
        final_si,
        &final_iteration_trend.values,
        diagnostic_c_extremes,
        tests,
        state,
    )?;
    Ok((context, iteration, part_d))
}

/// Build 62's B, C and D iterations (`x11pt2.f`'s `Kpart` 2, 3 and 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    B,
    C,
    D,
}

impl Pass {
    const fn trend_stage(self) -> x11::TrendStage {
        match self {
            Self::B => x11::TrendStage::Preliminary,
            Self::C => x11::TrendStage::Intermediate,
            Self::D => x11::TrendStage::FinalIteration,
        }
    }
}

/// One run's validated inputs, fixed through every pass.
struct RunContext<'a> {
    values: &'a [f64],
    raw: &'a [f64],
    axis: TimeAxis,
    /// `axis` checked over `values`: the dates of every observation index.
    dates: AxisSpan,
    observed: Range<usize>,
    mode: x11::Mode,
    options: x11::Options,
    requested_axis: TimeAxis,
    purpose: RunPurpose,
    /// Declared filters, before `vsfb.f` resolves the provisional ones.
    filters: SeasonalFilterSet<x11::SeasonalFilter>,
    /// `vsfb.f` resolves the provisional filters by section of a pass, not by
    /// pass: 3x3 in the first section, 3x5 in the second. Declared stable
    /// entries stay stable, which `vsfc.f`'s endpoint centering reads.
    first_section_filters: SeasonalFilterSet<ResolvedSeasonalFilter>,
    second_section_filters: SeasonalFilterSet<ResolvedSeasonalFilter>,
    robust: Option<RobustContext>,
    period: usize,
    start_subperiod: usize,
    trend_only: bool,
}

impl<'a> RunContext<'a> {
    /// Validate the run. Errors take this precedence: axis, robust-scale
    /// geometry, observation count, fixed trend length, then initial seasonal
    /// history.
    fn new(input: RunInput<'a>, purpose: RunPurpose) -> Result<Self, Error> {
        let RunInput {
            values,
            raw,
            axis,
            observed,
            mode,
            prepared,
        } = input;
        let options = prepared.options;
        let filters = prepared.seasonal_filters;
        debug_assert!(observed.start <= observed.end && observed.end <= values.len());
        let dates = axis.span(values.len())?;
        let robust = if mode == x11::Mode::PseudoAdditive {
            Some(RobustContext::required(
                options.extreme_estimator(),
                prepared.requested_axis,
                axis,
            )?)
        } else {
            RobustContext::new(options.extreme_estimator(), prepared.requested_axis, axis)?
        };
        debug_assert_eq!(filters.frequency(), axis.frequency());
        let period = usize::from(axis.frequency().periods_per_year());
        super::require_three_years(axis.frequency(), observed.len())?;
        // `PreparedOptions` made this check on the requested span; the series
        // is shorter when its requested extension was not generated.
        if let Some(length) = options.trend_filter().fixed_length()
            && values.len() < usize::from(length.terms())
        {
            return Err(Error::HendersonLongerThanSeries {
                terms: usize::from(length.terms()),
                count: values.len(),
            });
        }
        let context = Self {
            values,
            raw,
            axis,
            dates,
            observed,
            mode,
            options,
            requested_axis: prepared.requested_axis,
            purpose,
            filters,
            first_section_filters: filters.resolve(ResolvedSeasonalFilter::S3x3),
            second_section_filters: filters.resolve(ResolvedSeasonalFilter::S3x5),
            robust,
            period,
            start_subperiod: usize::from(axis.start().subperiod()),
            trend_only: options.decomposition() == crate::x11::Decomposition::TrendOnly,
        };
        if !context.trend_only {
            validate_initial_seasonal_history(
                values.len(),
                axis,
                &context.filters,
                context.first_smoothing(),
            )?;
        }
        Ok(context)
    }

    fn first_smoothing(&self) -> SeasonalSmoothingRequest<'_> {
        SeasonalSmoothingRequest {
            filters: &self.first_section_filters,
            short_series_filters: self.options.short_series_filters(),
        }
    }

    fn second_smoothing(&self) -> SeasonalSmoothingRequest<'_> {
        SeasonalSmoothingRequest {
            filters: &self.second_section_filters,
            short_series_filters: self.options.short_series_filters(),
        }
    }

    fn robust_at(&self, stage: X11ExtremeStage) -> Option<RobustContext> {
        self.robust.map(|context| context.at(stage))
    }

    /// Weighting spans for a section applied over `application`, estimated
    /// over the observed span less `trim` at each end when requested.
    fn extreme_spans(&self, application: Range<usize>, trim: usize) -> ExtremeSpans {
        ExtremeSpans::for_section(
            application,
            self.observed.clone(),
            trim,
            self.options.extreme_estimation_span(),
        )
    }

    /// The seasonal a trend-only decomposition uses in place of a filter.
    fn neutral_seasonal(&self) -> Vec<f64> {
        vec![self.mode.center(); self.values.len()]
    }
}

/// What the B and C passes carry forward, and what they keep for Part D.
struct Iteration {
    log_additive: Option<super::log_additive::Phase>,
    /// The original series, logged for log-additive runs.
    original: Vec<f64>,
    /// `original` less the latest extremes; each pass starts from it.
    modified: Vec<f64>,
    extreme_scale: ExtremeScaleState,
    trend_ic: f64,
    warnings: Vec<Warning>,
    intermediate_extreme_weights: Vec<f64>,
    /// C extremes and weights that rebuild the final SI.
    final_c_extremes: Option<FinalCExtremes>,
}

/// Pass C's extreme components and weights.
struct FinalCExtremes {
    component: Vec<f64>,
    weights: Vec<f64>,
}
