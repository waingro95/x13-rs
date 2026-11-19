//! X-11: the options of an X-11 adjustment and the diagnostics it reports.
//!
//! [`Options`] holds every X-11 setting: the [`Mode`], the seasonal and trend
//! filters, the limits for extreme values, and the treatment of Easter,
//! holidays and outliers. [`Spec::x11`](crate::Spec::x11) and
//! [`Spec::x11_regarima`](crate::Spec::x11_regarima) take it.
//!
//! An adjustment's X-11 [`Diagnostics`] come from
//! [`Diagnostics::x11`](crate::Diagnostics::x11): the mode that ran, the
//! seasonal filter the moving seasonality ratio chose ([`MsrSelection`]) and
//! the spectral peaks of the output ([`SpectralDiagnostics`]). An X-11 Easter
//! estimate is an [`EasterEffect`], from
//! [`Adjustment::x11_easter`](crate::Adjustment::x11_easter).

pub use crate::api::x11::{
    AutomaticExtremeScale, CalendarSigmaSelection, Decomposition, Diagnostics, EasterDateGroup,
    EasterEffect, EasterPreliminaryTrendWarning, EasterStatus, ExtremeEstimationSpan,
    ExtremeEstimator, ExtremeScale, FinalOutliers, HendersonIcRatio, HendersonLength,
    HolidayAdjustment, Mode, MsrFilter, MsrSelection, Options, SeasonalFilter, SeasonalFilters,
    SeasonalShrinkage, SevenTermEndpoints, ShortSeriesFilters, SigmaGroup, SigmaGroups,
    SigmaLimits, SpectralDiagnostics, TemporaryChangeAllocation, TrendFilter, TrendStage,
};
pub(crate) use crate::api::x11::{X11EasterConflict, X11EasterStage, X11EasterUnavailableReason};

mod calendar_sigma;
mod direct;
mod easter;
mod log_additive;
mod msr;
pub(crate) mod pseudo_additive;
pub(crate) mod quality;
mod robust;
mod seasonal;
pub(crate) mod seasonality;
mod shrinkage;
pub(crate) mod spectral;

pub(crate) use direct::{PreparedEaster, adjust_easter, finish_easter, prepare_easter};
pub(crate) use direct::{adjust, adjust_extended};

/// `editor.f`'s `Length.lt.3*Ny`: every X-11 run needs three years of
/// observations, whatever the options.
pub(crate) fn require_three_years(
    frequency: crate::Frequency,
    observed: usize,
) -> Result<(), crate::Error> {
    let minimum = 3 * usize::from(frequency.periods_per_year());
    if observed < minimum {
        return Err(crate::Error::TooShort {
            method: crate::Method::X11,
            actual: observed,
            minimum,
        });
    }
    Ok(())
}

/// One run's options after `editor.f`'s checks, which are made on the
/// requested span: the observations with the forecasts and backcasts asked
/// for, whether or not they are later generated.
#[derive(Clone)]
pub(crate) struct PreparedOptions {
    options: Options,
    /// The axis of the requested span.
    requested_axis: crate::time::TimeAxis,
    /// The seasonal filter of each calendar period, January or the first
    /// quarter first, wherever the series starts.
    seasonal_filters: seasonal::SeasonalFilterSet<SeasonalFilter>,
    warning: Option<crate::Warning>,
}

impl PreparedOptions {
    /// Take `editor.f`'s warning, which a run reports once.
    pub(crate) fn take_warning(&mut self) -> Option<crate::Warning> {
        self.warning.take()
    }

    /// Options for a run on a series with no extensions: its observed count is
    /// the requested count.
    pub(crate) fn new(
        options: Options,
        requested_axis: crate::time::TimeAxis,
        requested_count: usize,
    ) -> Result<Self, crate::Error> {
        Self::with_observed(options, requested_axis, requested_count, requested_count)
    }

    /// Options for a run on `observed_count` observations that the requested
    /// span `requested_count` extends with forecasts and backcasts. The
    /// three-year check comes before the Henderson one: a series too short for
    /// X-11 is not helped by a shorter filter.
    pub(crate) fn with_observed(
        options: Options,
        requested_axis: crate::time::TimeAxis,
        requested_count: usize,
        observed_count: usize,
    ) -> Result<Self, crate::Error> {
        use crate::Warning;
        use seasonal::SeasonalFilterSet;
        if let Some(filters) = options.seasonal_filters()
            && filters.frequency() != requested_axis.frequency()
        {
            return Err(crate::Error::from(
                crate::error::SpecProblem::SeasonalFilterFrequencyMismatch {
                    expected: filters.frequency(),
                    actual: requested_axis.frequency(),
                },
            ));
        }
        if let ExtremeScale::Grouped(groups) = options.extreme_scale()
            && groups.frequency() != requested_axis.frequency()
        {
            return Err(crate::Error::from(
                crate::error::SpecProblem::ExtremeScaleFrequencyMismatch {
                    expected: groups.frequency(),
                    actual: requested_axis.frequency(),
                },
            ));
        }
        requested_axis.periods(requested_count)?;
        require_three_years(requested_axis.frequency(), observed_count)?;
        if let Some(length) = options.trend_filter().fixed_length()
            && requested_count < usize::from(length.terms())
        {
            return Err(crate::Error::HendersonLongerThanSeries {
                terms: usize::from(length.terms()),
                count: requested_count,
            });
        }
        if options.decomposition() == Decomposition::TrendOnly {
            return Ok(Self {
                options,
                requested_axis,
                seasonal_filters: SeasonalFilterSet::uniform(
                    requested_axis.frequency(),
                    SeasonalFilter::S3x5,
                ),
                warning: None,
            });
        }
        let period = usize::from(requested_axis.frequency().periods_per_year());
        let calendar_years = (usize::from(requested_axis.start().subperiod()) - 1
            + requested_count)
            .div_ceil(period);
        // An explicit set's frequency was checked against the axis above.
        let mut seasonal_filters = options.seasonal_filters().map_or_else(
            || SeasonalFilterSet::uniform(requested_axis.frequency(), options.seasonal_filter()),
            SeasonalFilterSet::from,
        );
        // `editor.f` replaces every affected `Lter` entry and prints one
        // warning, whichever calendar period's entry it found first.
        let mut warning = None;
        if calendar_years < 20 {
            for filter in seasonal_filters.as_mut_slice() {
                if *filter == SeasonalFilter::S3x15 {
                    *filter = SeasonalFilter::Stable;
                    warning = Some(Warning::SeasonalFilterReplaced {
                        requested: SeasonalFilter::S3x15,
                        replacement: SeasonalFilter::Stable,
                    });
                }
            }
        }
        Ok(Self {
            options,
            requested_axis,
            seasonal_filters,
            warning,
        })
    }
}

/// A final X-11 run before its diagnostic completions. The residual
/// seasonality tests hold placeholders, and quality and spectra are not yet
/// computed, until [`Components::complete`] turns it into
/// [`CompletedComponents`], the only form a public result accepts.
pub(crate) struct Components {
    pub(crate) spectral_workspace: spectral::SpectralWorkspace,
    pub(crate) decomposition: Decomposition,
    pub(crate) mode: Mode,
    pub(crate) seasonal: Vec<f64>,
    pub(crate) seasonally_adjusted: Vec<f64>,
    pub(crate) trend_cycle: Vec<f64>,
    pub(crate) irregular: Vec<f64>,
    pub(crate) final_henderson_terms: usize,
    pub(crate) msr_selection: Option<MsrSelection>,
    pub(crate) calendar_sigma_selection: Option<CalendarSigmaSelection>,
    pub(crate) seasonality: crate::diagnostics::SeasonalityTests,
    pub(crate) quality_workspace: quality::QualityWorkspace,
    pub(crate) warnings: Vec<crate::Warning>,
    pub(crate) intermediate_extreme_weights: Vec<f64>,
    pub(crate) conditioning_scales: ConditioningScales,
}

/// What the diagnostic completions take from whatever produced the final
/// D11: the modified seasonally adjusted series (E11 with extremes replaced),
/// the quality adjustment factors and the spectral settings.
pub(crate) struct Completion {
    pub(crate) modified: Vec<f64>,
    pub(crate) quality: quality::QualityAdjustments,
    pub(crate) spectral: spectral::SpectralCompletion,
}

/// A final X-11 run with its residual seasonality, quality and spectral
/// diagnostics complete.
pub(crate) struct CompletedComponents {
    pub(crate) components: Components,
    pub(crate) quality:
        Result<crate::diagnostics::QualityMeasures, crate::diagnostics::Unavailable>,
    pub(crate) spectral: SpectralDiagnostics,
}

impl Components {
    /// Runs the diagnostic completions on the final D11, D12 and D13:
    /// residual seasonality of D11, then of the modified series, then
    /// quality, then spectra. Each appends its warnings in that order, and
    /// quality sets the Part F operands the spectra read.
    pub(crate) fn complete(
        mut self,
        original: &[f64],
        dates: crate::time::AxisSpan,
        completion: Completion,
    ) -> CompletedComponents {
        let Completion {
            modified,
            quality,
            spectral,
        } = completion;
        let axis = crate::time::TimeAxis::from_start(dates.start());
        seasonality::complete_residual(&mut self, original, axis);
        seasonality::complete_modified(&mut self, modified, original, axis);
        let quality = quality::complete(&mut self, original, dates, quality);
        let spectral = spectral::complete(&mut self, dates, spectral);
        CompletedComponents {
            components: self,
            quality,
            spectral,
        }
    }
}

/// Henderson operation scales of D12 and D13 over the observed span, which
/// this repository's development tests read. Empty for log-additive and
/// pseudo-additive decompositions.
///
/// The two vectors exist only with the hidden `__conditioning-scales` feature,
/// which computes them in `direct::Iteration::into_components`. Without it the
/// type is zero-sized, so `Components` and `Adjustment` carry it at no cost
/// and the construction sites need no `cfg`.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct ConditioningScales {
    #[cfg(feature = "__conditioning-scales")]
    pub(crate) trend_cycle: Vec<f64>,
    #[cfg(feature = "__conditioning-scales")]
    pub(crate) irregular: Vec<f64>,
}
