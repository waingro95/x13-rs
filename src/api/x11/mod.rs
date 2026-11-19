//! The X-11 options and the diagnostics of an X-11 adjustment.

mod diagnostics;
mod easter;
mod final_outliers;
mod holiday_adjustment;
mod spectral;
mod temporary_change;

pub use diagnostics::{Diagnostics, TrendStage};
pub use easter::{EasterDateGroup, EasterEffect, EasterPreliminaryTrendWarning, EasterStatus};
pub(crate) use easter::{X11EasterConflict, X11EasterStage, X11EasterUnavailableReason};
pub use final_outliers::FinalOutliers;
pub use holiday_adjustment::HolidayAdjustment;
pub use spectral::SpectralDiagnostics;
pub use temporary_change::TemporaryChangeAllocation;

use crate::Error;
use crate::error::SpecProblem;
use crate::regarima::X11ModeSelection;

/// How X-11 combines the trend-cycle, seasonal and irregular components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Components add up to the series; seasonal effects are in series
    /// units. Accepts any finite values.
    Additive,
    /// Components multiply to the series; seasonal factors are ratios near
    /// one. Needs positive values. The X-13ARIMA-SEATS default.
    Multiplicative,
    /// X-11 on the logarithms of the series, mapped back to ratios with
    /// a bias correction of the trend. Needs positive values. This
    /// is separate from the regARIMA [`Transform`](crate::Transform).
    LogAdditive,
    /// For series with seasonal values at or near zero; observations may be
    /// zero but not negative. Cannot be combined with calendar, outlier or
    /// other regression preadjustment.
    PseudoAdditive,
}

display_phrases!(Mode {
    Additive => "additive",
    Multiplicative => "multiplicative",
    LogAdditive => "log-additive",
    PseudoAdditive => "pseudo-additive",
});

/// Whether X-11 removes seasonal variation or estimates only trend and irregular.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Decomposition {
    /// Perform seasonal adjustment (default).
    #[default]
    SeasonalAdjustment,
    /// Estimate trend and irregular without seasonal smoothing. Explicit
    /// seasonal-filter requests and LogAdditive mode are incompatible.
    TrendOnly,
}

/// Optional shrinkage of the final seasonal factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SeasonalShrinkage {
    /// Preserve the unshrunk seasonal factors (default).
    #[default]
    None,
    /// Calendar-year global shrinkage. A final stable seasonal filter has no
    /// defined shrinkage weight and returns [`Error::Numerical`].
    Global,
    /// Calendar-period local likelihood shrinkage. In seasonal adjustment it
    /// needs at least five monthly forecasts or one quarterly forecast that
    /// X-11 uses. Direct X-11, regARIMA X-11 whose forecasts were not used,
    /// undefined weights and unusable scales all return [`Error::Numerical`].
    /// All shrinkage settings are inert under trend-only decomposition.
    Local,
}

display_phrases!(SeasonalShrinkage {
    None => "no",
    Global => "global",
    Local => "local",
});

/// Scalar estimator for X-11 extreme-value weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExtremeEstimator {
    /// Standard deviation (default).
    StandardDeviation,
    /// Fixed-center median absolute deviation.
    MedianAbsoluteDeviation,
    /// Fixed-center MAD with the tau adjustment.
    TauAdjustedMedianAbsoluteDeviation,
    /// Median absolute log deviation of eligible internal SI/irregular samples,
    /// independent of regARIMA input transformation. Sampled values must be
    /// finite and positive. A zero converted scale is valid; unusable samples
    /// return [`Error::Numerical`].
    LogMedianAbsoluteDeviation,
    /// Log MAD of eligible internal SI/irregular samples, converted before the
    /// tau adjustment, independently of regARIMA input transformation.
    /// Sampled values must be finite and positive. If the converted tau base
    /// is exactly zero (for example, all eligible samples equal one or
    /// exponential conversion rounds to one), estimation returns
    /// [`Error::Numerical`].
    TauAdjustedLogMedianAbsoluteDeviation,
}

display_phrases!(ExtremeEstimator {
    StandardDeviation => "standard deviation",
    MedianAbsoluteDeviation => "median absolute deviation",
    TauAdjustedMedianAbsoluteDeviation => "tau-adjusted median absolute deviation",
    LogMedianAbsoluteDeviation => "log median absolute deviation",
    TauAdjustedLogMedianAbsoluteDeviation => "tau-adjusted log median absolute deviation",
});

/// Sampling policy for X-11 extreme-value scale estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExtremeScale {
    /// Rolling five-year X-11 scale (default).
    RollingFiveYear,
    /// Select rolling or calendar-period scales with two Cochran tests.
    /// Both decisions are available through [`Diagnostics::calendar_sigma_selection`].
    ///
    /// ```
    /// use x13::{Period, Series, Spec, x11};
    /// let options = x11::Options::new().with_extreme_scale(x11::ExtremeScale::Automatic);
    /// let series = Series::new(Period::monthly(2000, 4)?, &[100.0; 72]);
    /// let result = x13::adjust(series, &Spec::x11(options))?;
    /// assert!(result.diagnostics().x11().unwrap().calendar_sigma_selection().is_some());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    Automatic,
    /// Separate scale for each calendar month or quarter.
    ByCalendarPeriod,
    /// One scale for each of two calendar groups.
    Grouped(SigmaGroups),
}

/// Scale selected by one automatic calendar-sigma test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AutomaticExtremeScale {
    /// Rolling five-year scale.
    RollingFiveYear,
    /// Separate scale for each calendar month or quarter.
    ByCalendarPeriod,
}

/// The two automatic choices of extreme-value scale, made under
/// [`ExtremeScale::Automatic`] and read from
/// [`Diagnostics::calendar_sigma_selection`](Diagnostics::calendar_sigma_selection).
///
/// Both choices are always present, also for a constant series. The test
/// statistics behind them are not reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CalendarSigmaSelection {
    initial_si: AutomaticExtremeScale,
    preliminary_irregular: AutomaticExtremeScale,
}

impl CalendarSigmaSelection {
    pub(crate) const fn new(
        initial_si: AutomaticExtremeScale,
        preliminary_irregular: AutomaticExtremeScale,
    ) -> Self {
        Self {
            initial_si,
            preliminary_irregular,
        }
    }

    /// Decision from the first B seasonal-irregular calculation, before weighting.
    #[must_use]
    pub const fn initial_si(self) -> AutomaticExtremeScale {
        self.initial_si
    }

    /// Decision from B irregular over accepted backcasts and observations.
    /// Forecasts never enter this test, even when used to estimate weights.
    #[must_use]
    pub const fn preliminary_irregular(self) -> AutomaticExtremeScale {
        self.preliminary_irregular
    }
}

/// Membership in one of two calendar groups for extreme-scale estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SigmaGroup {
    /// In the selected group.
    Selected,
    /// In the complementary group.
    Complement,
}

display_phrases!(SigmaGroup {
    Selected => "selected",
    Complement => "complementary",
});

/// Calendar-order groups, independent of the series start date.
/// Empty selection pools the estimation span for both logical groups;
/// all-selected arrays use the same pooled scale for the applied group under
/// standard deviation. Robust estimators return [`Error::Numerical`] for an
/// empty complement.
/// Frequency mismatch returns [`Error::InvalidSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SigmaGroups {
    /// January through December.
    Monthly([SigmaGroup; 12]),
    /// First through fourth calendar quarter.
    Quarterly([SigmaGroup; 4]),
}

impl SigmaGroups {
    /// Frequency required by these calendar groups.
    #[must_use]
    pub const fn frequency(self) -> crate::Frequency {
        match self {
            Self::Monthly(_) => crate::Frequency::Monthly,
            Self::Quarterly(_) => crate::Frequency::Quarterly,
        }
    }
}

/// Span used to estimate X-11 extreme-value scales.
/// Seasonal smoothing and extreme replacement still use accepted extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtremeEstimationSpan {
    /// Estimate using the observed series and all accepted extensions (default).
    WithExtensions,
    /// Estimate using only observations; apply weights over accepted extensions too.
    ObservedOnly,
}

/// A seasonal filter usable uniformly or for an individual month or quarter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SeasonalFilter {
    /// Three-term seasonal moving average.
    S3x1,
    /// 3-by-3 seasonal moving average.
    S3x3,
    /// 3-by-5 seasonal moving average.
    S3x5,
    /// 3-by-9 seasonal moving average.
    S3x9,
    /// 3-by-15 seasonal moving average, subject to the minimum-history rules of
    /// X-13ARIMA-SEATS.
    S3x15,
    /// Constant seasonal factors estimated from the whole span.
    Stable,
    /// The original X-11 sequence: S3x3 in the first section
    /// and S3x5 in the second section of each pass, then final S3x5.
    /// This is distinct from MSR selection, the default.
    X11Default,
    /// Choice of filter by the global moving seasonality ratio
    /// (MSR), the default in [`Options::new`] as in X-13ARIMA-SEATS when no
    /// seasonal filter is given. Selects S3x3, S3x5 or S3x9 for the periods
    /// that request it. See [`MsrSelection`] for the fallback with a short
    /// history, and for the rule that skips the selection when
    /// January (or the first quarter) explicitly requests S3x1.
    Msr,
}

display_phrases!(SeasonalFilter {
    S3x1 => "3x1",
    S3x3 => "3x3",
    S3x5 => "3x5",
    S3x9 => "3x9",
    S3x15 => "3x15",
    Stable => "stable",
    X11Default => "X-11 default",
    Msr => "MSR-selected",
});

/// How X-11 smooths seasonal subseries with fewer than five complete years.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ShortSeriesFilters {
    /// Use stable seasonal factors on short spans (`sfshort=no` in
    /// X-13ARIMA-SEATS).
    #[default]
    StableFallback,
    /// Use the requested filter where its endpoint rules are defined
    /// (`sfshort=yes`). Exactly three S3x5 entries still use Stable.
    UseRequested,
}

/// Seasonal-filter requests in calendar order, independent of the series start.
/// Explicit filters, the original [`SeasonalFilter::X11Default`] sequence,
/// and [`SeasonalFilter::Msr`] selection may be mixed. MSR replaces only
/// periods requesting MSR; original-filter periods retain final S3x5.
///
/// The array length is enforced by the type. Adjustment rejects a setting whose
/// frequency differs from the input series with [`Error::InvalidSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SeasonalFilters {
    /// January through December, including when observations start midyear.
    Monthly([SeasonalFilter; 12]),
    /// First through fourth calendar quarter.
    Quarterly([SeasonalFilter; 4]),
}

/// The three possible global MSR filter choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MsrFilter {
    /// 3x3 seasonal filter.
    S3x3,
    /// 3x5 seasonal filter.
    S3x5,
    /// 3x9 seasonal filter.
    S3x9,
}

/// Outcome of the global MSR selection for the calendar periods that request
/// it. Explicit filters in other calendar periods are not changed by this
/// choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MsrSelection {
    /// A ratio met a selection threshold after this many ratio evaluations.
    Selected {
        /// Selected filter.
        filter: MsrFilter,
        /// Number of ratio evaluations.
        passes: usize,
    },
    /// Fewer than five years remained; use S3x5. `passes` is zero when
    /// the initial calendar-truncated span was already too short.
    ShortHistoryFallback {
        /// Number of ratio evaluations.
        passes: usize,
    },
    /// Global selection is skipped when January/Q1 explicitly requests
    /// S3x1, even if later periods request MSR. Those periods retain the
    /// provisional S3x3/S3x5 sequence and use S3x5 for final smoothing.
    SuppressedByFirstPeriodS3x1,
    /// Global MSR was consumed by the preliminary pass of the X-11 Easter
    /// estimate.
    SuppressedByEasterEstimation,
}

impl SeasonalFilters {
    /// Frequency required by these calendar-specific filters.
    #[must_use]
    pub const fn frequency(self) -> crate::Frequency {
        match self {
            Self::Monthly(_) => crate::Frequency::Monthly,
            Self::Quarterly(_) => crate::Frequency::Quarterly,
        }
    }
}

/// Validated explicit Henderson length: an odd number from 3 through 101.
///
/// The filter must fit the series X-11 receives, including the forecasts and
/// backcasts it uses. If too few values remain once unusable extensions are
/// dropped, adjustment returns [`Error::HendersonLongerThanSeries`]; a shorter
/// filter fixes it. Build 62 differs: it may continue with windows outside the
/// remaining span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HendersonLength(u8);

impl HendersonLength {
    /// Validates the number of terms. The input-span check occurs at adjustment.
    pub fn new(terms: u8) -> Result<Self, Error> {
        if !(3..=101).contains(&terms) || terms.is_multiple_of(2) {
            return Err(Error::from(SpecProblem::HendersonLength { terms }));
        }
        Ok(Self(terms))
    }

    /// Number of Henderson terms.
    #[must_use]
    pub const fn terms(self) -> u8 {
        self.0
    }
}

/// Positive finite I/C ratio used to generate Henderson endpoint weights. This
/// is separate from the measured I/C used for automatic length selection.
/// Values strictly below `3.834e-20` count as an omitted ratio, as in
/// X-13ARIMA-SEATS, and the default for the filter length and frequency is
/// used. Legacy seven-term endpoints reset the ratio to `0.001` when applied,
/// even if a custom ratio was requested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HendersonIcRatio(f64);

// Construction excludes NaNs, so equality is reflexive.
impl Eq for HendersonIcRatio {}

// Construction excludes NaNs and zeros, so equal ratios have equal bits.
impl std::hash::Hash for HendersonIcRatio {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl HendersonIcRatio {
    /// Validates a positive finite endpoint-weight ratio.
    pub fn new(value: f64) -> Result<Self, Error> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::from(SpecProblem::HendersonIcRatio));
        }
        Ok(Self(value))
    }

    /// Requested value, before any defaulting or legacy endpoint reset.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.0
    }
}

/// Endpoint behavior when a seven-term Henderson filter is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SevenTermEndpoints {
    /// The X-13ARIMA-SEATS default: five-term endpoints and an I/C reset to
    /// `0.001`.
    LegacyFiveTerm,
    /// Seven-term Henderson endpoints, preserving the endpoint I/C ratio.
    HendersonSevenTerm,
}

/// Selection of the Henderson trend filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TrendFilter {
    /// Automatic Henderson selection from the I/C ratio (the default).
    Automatic,
    /// A fixed Henderson length.
    Henderson(HendersonLength),
    /// Fixed length with a custom endpoint I/C ratio.
    HendersonWithIcRatio {
        /// Henderson filter length.
        length: HendersonLength,
        /// Endpoint I/C ratio.
        ratio: HendersonIcRatio,
    },
}

impl TrendFilter {
    pub(crate) const fn fixed_length(self) -> Option<HendersonLength> {
        match self {
            Self::Automatic => None,
            Self::Henderson(length) | Self::HendersonWithIcRatio { length, .. } => Some(length),
        }
    }
}

/// Finite sigma limits satisfying `0 < lower <= upper`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmaLimits {
    lower: f64,
    upper: f64,
}

impl SigmaLimits {
    /// Validates the limits used for X-11 extreme-value weights.
    /// Equal limits select a step cutoff, as in X-13ARIMA-SEATS.
    pub fn new(lower: f64, upper: f64) -> Result<Self, Error> {
        if !lower.is_finite() || !upper.is_finite() || lower <= 0.0 || lower > upper {
            return Err(Error::from(SpecProblem::SigmaLimits));
        }
        Ok(Self { lower, upper })
    }

    /// Lower sigma limit.
    #[must_use]
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// Upper sigma limit.
    #[must_use]
    pub const fn upper(self) -> f64 {
        self.upper
    }
}

/// X-11 settings, for [`Spec::x11`](crate::Spec::x11) and
/// [`Spec::x11_regarima`](crate::Spec::x11_regarima).
///
/// [`Options::new`], also the [`Default`], gives the X-13ARIMA-SEATS
/// defaults: seasonal adjustment in the default mode, the seasonal filter
/// chosen automatically from the moving seasonality ratio, an automatically
/// chosen Henderson trend filter, sigma limits of 1.5 and 2.5 for extreme
/// values, and no X-11 Easter adjustment. Each `with_*` setter changes one
/// setting and returns the options; a getter of the same name reads it back.
///
/// ```
/// use x13::{Spec, x11};
///
/// let options = x11::Options::new()
///     .with_seasonal_filter(x11::SeasonalFilter::S3x9)
///     .with_trend_filter(x11::TrendFilter::Henderson(x11::HendersonLength::new(11)?))
///     .with_sigma_limits(x11::SigmaLimits::new(1.5, 3.0)?);
/// let spec = Spec::x11(options);
/// # Ok::<(), x13::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Options {
    mode: Option<Mode>,
    holiday_adjustment: HolidayAdjustment,
    final_outliers: FinalOutliers,
    temporary_change_allocation: TemporaryChangeAllocation,
    easter_estimation: bool,
    decomposition: Decomposition,
    explicit_seasonal_filter: bool,
    seasonal_shrinkage: SeasonalShrinkage,
    seasonal_filter: SeasonalFilter,
    seasonal_filters: Option<SeasonalFilters>,
    short_series_filters: ShortSeriesFilters,
    trend_filter: TrendFilter,
    seven_term_endpoints: SevenTermEndpoints,
    sigma_limits: SigmaLimits,
    extreme_estimation_span: ExtremeEstimationSpan,
    extreme_scale: ExtremeScale,
    extreme_estimator: ExtremeEstimator,
}

impl Options {
    /// The X-13ARIMA-SEATS defaults: seasonal adjustment with automatic MSR
    /// seasonal-filter selection, an automatic Henderson filter with five-term
    /// endpoints when the seven-term filter is chosen
    /// ([`SevenTermEndpoints::LegacyFiveTerm`]), and 1.5/2.5 sigma limits.
    /// Trend-only decomposition ignores the seasonal filter.
    pub const fn new() -> Self {
        Self {
            mode: None,
            holiday_adjustment: HolidayAdjustment::Remove,
            final_outliers: FinalOutliers::Retain,
            temporary_change_allocation: TemporaryChangeAllocation::Irregular,
            decomposition: Decomposition::SeasonalAdjustment,
            easter_estimation: false,
            explicit_seasonal_filter: false,
            seasonal_shrinkage: SeasonalShrinkage::None,
            seasonal_filter: SeasonalFilter::Msr,
            seasonal_filters: None,
            short_series_filters: ShortSeriesFilters::StableFallback,
            trend_filter: TrendFilter::Automatic,
            seven_term_endpoints: SevenTermEndpoints::LegacyFiveTerm,
            extreme_estimation_span: ExtremeEstimationSpan::WithExtensions,
            extreme_scale: ExtremeScale::RollingFiveYear,
            extreme_estimator: ExtremeEstimator::StandardDeviation,
            sigma_limits: SigmaLimits {
                lower: 1.5,
                upper: 2.5,
            },
        }
    }

    /// Sets the decomposition mode.
    ///
    /// With none set, [`Spec::x11`](crate::Spec::x11) is multiplicative, and
    /// [`Spec::x11_regarima`](crate::Spec::x11_regarima) follows the model's
    /// transform: additive with no transform, multiplicative with logs. A
    /// mode cannot be combined with an automatic transform, which is not
    /// known until the series is fitted.
    pub const fn with_mode(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// The decomposition mode, or `None` when it is left to the default.
    /// [`Diagnostics::mode`](Diagnostics::mode) reports the mode
    /// used.
    #[must_use]
    pub const fn mode(self) -> Option<Mode> {
        self.mode
    }

    /// The mode for X-11 after a regARIMA model: the one set, or the one
    /// that matches the transform.
    pub(crate) const fn mode_selection(self) -> X11ModeSelection {
        match self.mode {
            Some(mode) => X11ModeSelection::Explicit(mode),
            None => X11ModeSelection::FromTransform,
        }
    }

    /// Chooses whether estimated holiday effects are removed from the final
    /// seasonally adjusted series (the default, as in X-13ARIMA-SEATS) or kept
    /// in it and in the irregular.
    ///
    /// An X-11 Easter effect estimated without calendar regressors in the
    /// model stays in the adjusted series under either choice; see
    /// [`with_easter_estimation`](Self::with_easter_estimation).
    pub const fn with_holiday_adjustment(mut self, policy: HolidayAdjustment) -> Self {
        self.holiday_adjustment = policy;
        self
    }

    /// Whether estimated holiday effects are removed from the final
    /// seasonally adjusted series.
    #[must_use]
    pub const fn holiday_adjustment(self) -> HolidayAdjustment {
        self.holiday_adjustment
    }

    /// Chooses which fitted regARIMA outlier types to take out of the final
    /// X-11 components. By default ([`FinalOutliers::Retain`]) every
    /// fitted outlier stays in the final series, as in X-13ARIMA-SEATS.
    ///
    /// [`Spec::x11`](crate::Spec::x11) fits no outliers and ignores this
    /// setting.
    pub const fn with_final_outliers(mut self, policy: FinalOutliers) -> Self {
        self.final_outliers = policy;
        self
    }

    /// Which fitted outlier types are taken out of the final X-11
    /// components.
    #[must_use]
    pub const fn final_outliers(self) -> FinalOutliers {
        self.final_outliers
    }

    /// Chooses the final component that receives fitted temporary-change
    /// outliers: the irregular (the default, as in X-13ARIMA-SEATS) or the
    /// trend.
    ///
    /// [`Spec::x11`](crate::Spec::x11) fits no outliers and ignores this
    /// setting.
    pub const fn with_temporary_change_allocation(
        mut self,
        allocation: TemporaryChangeAllocation,
    ) -> Self {
        self.temporary_change_allocation = allocation;
        self
    }

    /// The final component that receives fitted temporary-change outliers.
    #[must_use]
    pub const fn temporary_change_allocation(self) -> TemporaryChangeAllocation {
        self.temporary_change_allocation
    }

    /// Requests X-11's own Easter adjustment (monthly series only; off by
    /// default): the Easter effect is estimated from a preliminary
    /// decomposition and removed before the adjustment, with or without a
    /// regARIMA model.
    ///
    /// [`Adjustment::x11_easter`](crate::Adjustment::x11_easter) reports the
    /// estimate. As in X-13ARIMA-SEATS, with no regARIMA model, or with one
    /// that has no calendar regressors, the effect shapes the seasonal
    /// factors but stays in the seasonally adjusted series, whatever the
    /// [`HolidayAdjustment`].
    ///
    /// This is separate from the regARIMA Easter regressor in
    /// [`CalendarSpec`](crate::regarima::CalendarSpec), and a request with both is
    /// refused, as is one with an automatic transform.
    pub const fn with_easter_estimation(mut self, estimate: bool) -> Self {
        self.easter_estimation = estimate;
        self
    }

    /// Whether X-11's own Easter adjustment is requested.
    #[must_use]
    pub const fn easter_estimation(self) -> bool {
        self.easter_estimation
    }

    /// Sets one seasonal filter for all months or quarters, clearing any
    /// calendar-specific override.
    pub const fn with_seasonal_filter(mut self, filter: SeasonalFilter) -> Self {
        self.explicit_seasonal_filter = true;
        self.seasonal_filter = filter;
        self.seasonal_filters = None;
        self
    }

    /// Selects optional final seasonal-factor shrinkage.
    pub const fn with_seasonal_shrinkage(mut self, policy: SeasonalShrinkage) -> Self {
        self.seasonal_shrinkage = policy;
        self
    }

    /// Requested seasonal-factor shrinkage policy.
    #[must_use]
    pub const fn seasonal_shrinkage(self) -> SeasonalShrinkage {
        self.seasonal_shrinkage
    }

    /// Overrides the uniform filter with calendar-specific filter requests.
    ///
    /// ```
    /// use x13::x11;
    /// let mut months = [x11::SeasonalFilter::X11Default; 12];
    /// months[0] = x11::SeasonalFilter::Msr;
    /// months[11] = x11::SeasonalFilter::Stable; // December, even for a July start.
    /// let options = x11::Options::new()
    ///     .with_seasonal_filters(x11::SeasonalFilters::Monthly(months));
    /// ```
    pub const fn with_seasonal_filters(mut self, filters: SeasonalFilters) -> Self {
        self.explicit_seasonal_filter = true;
        self.seasonal_filters = Some(filters);
        self
    }

    /// Selects the decomposition independently of its arithmetic mode.
    pub const fn with_decomposition(mut self, decomposition: Decomposition) -> Self {
        self.decomposition = decomposition;
        self
    }

    /// Requested decomposition; ordinary seasonal adjustment is the default.
    #[must_use]
    pub const fn decomposition(self) -> Decomposition {
        self.decomposition
    }

    /// Whether either seasonal-filter setter was called, even with S3x5.
    pub(crate) const fn has_explicit_seasonal_filter(self) -> bool {
        self.explicit_seasonal_filter
    }

    /// Sets how seasonal filters treat a short history.
    pub const fn with_short_series_filters(mut self, policy: ShortSeriesFilters) -> Self {
        self.short_series_filters = policy;
        self
    }

    /// Requested short-history seasonal-filter policy.
    #[must_use]
    pub const fn short_series_filters(self) -> ShortSeriesFilters {
        self.short_series_filters
    }

    /// Sets automatic or explicit Henderson selection.
    pub const fn with_trend_filter(mut self, filter: TrendFilter) -> Self {
        self.trend_filter = filter;
        self
    }

    /// Sets endpoint behavior for seven-term filters, including automatic selection.
    /// Other Henderson lengths are unaffected.
    ///
    /// ```
    /// use x13::x11;
    /// let options = x11::Options::new()
    ///     .with_trend_filter(x11::TrendFilter::HendersonWithIcRatio {
    ///         length: x11::HendersonLength::new(7)?, ratio: x11::HendersonIcRatio::new(0.75)?,
    ///     })
    ///     .with_seven_term_endpoints(x11::SevenTermEndpoints::HendersonSevenTerm);
    /// # Ok::<(), x13::Error>(())
    /// ```
    pub const fn with_seven_term_endpoints(mut self, policy: SevenTermEndpoints) -> Self {
        self.seven_term_endpoints = policy;
        self
    }

    /// Requested endpoint behavior for seven-term Henderson filters.
    #[must_use]
    pub const fn seven_term_endpoints(self) -> SevenTermEndpoints {
        self.seven_term_endpoints
    }

    /// Sets validated extreme-value sigma limits.
    pub const fn with_sigma_limits(mut self, limits: SigmaLimits) -> Self {
        self.sigma_limits = limits;
        self
    }

    /// Uniform seasonal-filter request. Not used while [`Self::seasonal_filters`]
    /// returns a calendar-specific override.
    #[must_use]
    pub const fn seasonal_filter(self) -> SeasonalFilter {
        self.seasonal_filter
    }

    /// Calendar-specific override, if one was requested.
    #[must_use]
    pub const fn seasonal_filters(self) -> Option<SeasonalFilters> {
        self.seasonal_filters
    }

    /// Henderson-filter request.
    #[must_use]
    pub const fn trend_filter(self) -> TrendFilter {
        self.trend_filter
    }

    /// Extreme-value sigma limits.
    #[must_use]
    pub const fn sigma_limits(self) -> SigmaLimits {
        self.sigma_limits
    }

    /// Sets the span for extreme estimation, without removing accepted extensions
    /// from seasonal smoothing or replacement.
    ///
    /// ```
    /// use x13::x11;
    /// let options = x11::Options::new()
    ///     .with_extreme_estimation_span(x11::ExtremeEstimationSpan::ObservedOnly);
    /// assert_eq!(options.extreme_estimation_span(), x11::ExtremeEstimationSpan::ObservedOnly);
    /// // Accepted forecasts/backcasts remain available to seasonal smoothing.
    /// ```
    pub const fn with_extreme_estimation_span(mut self, span: ExtremeEstimationSpan) -> Self {
        self.extreme_estimation_span = span;
        self
    }

    /// Requested span for extreme-value scale estimation.
    #[must_use]
    pub const fn extreme_estimation_span(self) -> ExtremeEstimationSpan {
        self.extreme_estimation_span
    }

    /// Sets the scale policy independently of sigma limits and estimation span.
    ///
    /// ```
    /// use x13::{Period, Series, Spec, x11};
    /// let mut months = [x11::SigmaGroup::Complement; 12];
    /// months[0] = x11::SigmaGroup::Selected; // January, even for a July start.
    /// let options = x11::Options::new()
    ///     .with_extreme_scale(x11::ExtremeScale::Grouped(x11::SigmaGroups::Monthly(months)));
    /// let series = Series::new(Period::monthly(2000, 7)?, &[100.0; 120]);
    /// let result = x13::adjust(series, &Spec::x11(options))?;
    /// assert_eq!(result.seasonal().map(Series::len), Some(120));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn with_extreme_scale(mut self, scale: ExtremeScale) -> Self {
        self.extreme_scale = scale;
        self
    }

    /// Requested extreme-value scale policy.
    #[must_use]
    pub const fn extreme_scale(self) -> ExtremeScale {
        self.extreme_scale
    }

    /// Selects the estimator independently of the scale and sampling policies.
    pub const fn with_extreme_estimator(mut self, estimator: ExtremeEstimator) -> Self {
        self.extreme_estimator = estimator;
        self
    }

    /// Requested extreme-value estimator.
    #[must_use]
    pub const fn extreme_estimator(self) -> ExtremeEstimator {
        self.extreme_estimator
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}
