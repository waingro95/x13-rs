//! The public [`Error`](enum@Error) and its payload types.

use thiserror::Error;

mod detail;

pub(crate) use detail::{NumericalDetail, SpecProblem, UnsupportedDetail};

use crate::regarima::RegressionVariable;
use crate::{Frequency, Period, PeriodError, x11};

/// Seasonal-shrinkage step that could not produce a usable value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11ShrinkageStage {
    /// Weight for the last seasonal filter.
    FilterWeight,
    /// Variance of the seasonal-irregular values about the seasonal factors.
    Variance,
    /// Global shrinkage weight.
    GlobalWeight,
    /// Local shrinkage likelihood.
    LocalLikelihood,
    /// Shrunk seasonal factors.
    SeasonalFactors,
}

/// Why requested seasonal shrinkage could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11ShrinkageUnavailableReason {
    /// Shrinkage is not defined after this seasonal filter.
    UnsupportedFilter {
        /// Last seasonal filter applied before shrinkage.
        filter: x11::SeasonalFilter,
    },
    /// Local shrinkage needs more accepted forecasts than were produced.
    InsufficientAcceptedForecasts {
        /// Accepted forecasts available.
        actual: usize,
        /// Accepted forecasts needed.
        minimum: usize,
    },
    /// A local variance is zero or negative.
    NonPositiveLocalVariance,
    /// A denominator is zero.
    ZeroDenominator,
    /// The sum of the local likelihoods is zero or negative.
    NonPositiveLikelihoodSum,
    /// An intermediate value is NaN or infinite.
    NonFiniteArithmetic,
}

/// Why a requested robust X-11 scale could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11ExtremeScaleUnavailableReason {
    /// No values remain in the estimation sample.
    EmptySample,
    /// The tau correction's base is zero.
    ZeroTauBase,
    /// A deviation from the center is NaN or infinite.
    NonFiniteDeviation,
    /// A log-scale estimator met a zero or negative value.
    NonPositiveLogInput,
    /// The estimated scale is NaN or infinite.
    NonFiniteScale,
}

/// Why a multiplicative X-11 trend cannot be made usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11TrendUnavailableReason {
    /// No neighboring trend estimate is positive to replace a nonpositive one.
    NoPositiveNeighbor,
    /// The trend estimate is NaN or infinite.
    NonFiniteEstimate,
    /// The replacement estimate is NaN or infinite.
    NonFiniteReplacement,
}

/// X-11 step whose extreme-value scale was being estimated.
#[expect(
    clippy::enum_variant_names,
    reason = "each variant names the X-11 component the scale is taken from"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11ExtremeStage {
    /// Seasonal-irregular values smoothed by the first seasonal filter.
    InitialSeasonalIrregular,
    /// Seasonal-irregular values smoothed by the second seasonal filter.
    RefinedSeasonalIrregular,
    /// Irregular values of the preliminary (B-table) pass.
    PreliminaryIrregular,
    /// Irregular values of the intermediate (C-table) pass.
    IntermediateIrregular,
}

/// First or reweighted estimate of an extreme scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11ExtremeWeightingPass {
    /// First estimate, from unweighted values.
    Initial,
    /// Second estimate, after extreme values are downweighted.
    Reweighted,
}

/// ARIMA operator whose order or expanded lag is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ArimaOrderTerm {
    /// Autoregressive operator.
    Autoregressive,
    /// Differencing operator.
    Differencing,
    /// Moving-average operator.
    MovingAverage,
}

/// Forecast or backcast direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExtensionDirection {
    /// Periods after the observed span.
    Forecast,
    /// Periods before the observed span.
    Backcast,
}

/// Operation in the pseudo-additive X-11 passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11PseudoAdditiveStage {
    /// Initial seasonal adjustment of the original series.
    InitialSeasonalAdjustment,
    /// Seasonal-irregular calculation.
    SeasonalIrregular,
    /// Irregular calculation.
    Irregular,
    /// Reconstruction of extreme values.
    ExtremeReconstruction,
    /// Seasonal filter selection.
    SeasonalSelection,
    /// Trend filter selection.
    TrendSelection,
    /// Final seasonal adjustment.
    FinalAdjustment,
}

/// Unusable pseudo-additive operand or calculated intermediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11PseudoAdditiveUnavailableReason {
    /// A divisor is zero.
    ZeroDivisor,
    /// An operand or result is NaN or infinite.
    NonFiniteValue,
}

/// Step of log-additive X-11 that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11LogAdditiveStage {
    /// Logarithm of accepted original-scale input.
    InputLog,
    /// Exponentiation of final working seasonal values.
    SeasonalExponentiation,
    /// Exponentiation of final working trend values.
    TrendExponentiation,
    /// Sum of squared irregular values (table C13) over the full span.
    IrregularVariance,
    /// Common exponential or smoothed seasonal bias factor.
    BiasFactor,
    /// Product of exponentiated trend and bias factor.
    TrendBias,
}

/// Unusable input or intermediate in log-additive X-11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11LogAdditiveUnavailableReason {
    /// An accepted input is outside the positive logarithm domain.
    NonPositiveInput,
    /// A required input or calculated intermediate is not finite.
    NonFiniteValue,
    /// Seasonal exponentiation underflowed to an unusable zero divisor.
    ZeroSeasonalFactor,
}

/// Configuration that is recognized but not available in the current path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum UnsupportedConfiguration {
    /// Automatic transformation before selection is resolved.
    AutomaticTransform,
    /// Automatic model selection is unresolved or unavailable in the current path.
    AutomaticModel,
    /// An automatic model request forces a nondefault constant policy.
    AutomaticModelConstant,
    /// Calendar AICC testing combined with an unimplemented selection policy.
    CalendarSelection,
    /// Specified outliers combined with an unimplemented automatic policy.
    ManualOutliers,
}

/// Why an automatic transformation candidate has no usable AICC score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransformSelectionUnavailableReason {
    /// Fitted innovation variance is zero, so likelihood is unavailable.
    ZeroVariance,
    /// Too few effective observations for the fitted parameter count's AICC.
    InsufficientEffectiveObservations,
    /// Likelihood or AICC arithmetic produced a non-finite result.
    NonFiniteScore,
}

/// Calendar feature associated with a date or frequency validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CalendarFeature {
    /// Flow trading-day calculation.
    TradingDay,
    /// Monthly stock trading-day calculation.
    StockTradingDay,
    /// Easter moving-holiday effect.
    Easter,
    /// Labor Day moving-holiday effect.
    LaborDay,
    /// Thanksgiving moving-holiday effect.
    Thanksgiving,
}

/// Exact ARMA likelihood state that cannot be initialized for the current
/// parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum ExactLikelihoodFailure {
    /// The MA initial-state matrix G'G is not positive definite, which
    /// usually means the MA estimates have become noninvertible.
    #[error("MA initial-state covariance (G'G) is not positive definite at pivot {pivot}")]
    SingularMaInitialCovariance {
        /// Zero-based pivot (column) at which the Cholesky factorization failed.
        pivot: usize,
    },
    /// The AR initial-state covariance var(w_p|z) is not positive definite.
    #[error("AR initial-state covariance var(w_p|z) is not positive definite at pivot {pivot}")]
    SingularArInitialCovariance {
        /// Zero-based pivot (column) at which the Cholesky factorization failed.
        pivot: usize,
    },
    /// The theoretical ARMA autocovariances cannot be computed.
    #[error("theoretical ARMA autocovariances cannot be computed")]
    ArmaAutocovariance,
}

/// Numerical stage that failed without a more specific public error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumericalOperation {
    /// Exact ARMA filtering or likelihood evaluation.
    ExactLikelihood,
    /// Nonlinear parameter estimation.
    Estimation,
    /// Parameter covariance calculation.
    Covariance,
    /// Forecast or backcast calculation.
    Extension,
    /// Automatic-outlier filtering, testing, or deletion.
    OutlierSearch,
}

/// Step of automatic model selection whose constant-term test was unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelSelectionConstantTest {
    /// The initial constant-term test, before outlier search.
    Initial,
    /// The constant-term retest after outlier search.
    AfterOutlierSearch,
    /// The constant-term retest before final model corrections.
    BeforeFinalCorrections,
    /// The constant-term test after the final AR-root repair.
    AfterFinalRootRepair,
    /// The constant-term test after the final nonseasonal overdifferencing
    /// repair.
    AfterOverdifferenceRepair,
}

/// A final decomposition component returned by an adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    /// Seasonal component (table D10, or S10 for SEATS).
    Seasonal,
    /// Seasonally adjusted series (table D11, or S11 for SEATS).
    SeasonallyAdjusted,
    /// Trend-cycle component (table D12, or S12 for SEATS).
    TrendCycle,
    /// Irregular component (table D13, or S13 for SEATS).
    Irregular,
}

impl std::fmt::Display for Component {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Seasonal => "seasonal",
            Self::SeasonallyAdjusted => "seasonally adjusted",
            Self::TrendCycle => "trend-cycle",
            Self::Irregular => "irregular",
        };
        formatter.write_str(name)
    }
}

/// A published floating-point output of an adjustment. Scalars report index 0;
/// list outputs report the list position, and matrices their packed index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdjustmentOutput {
    /// A final decomposition component.
    Component(Component),
    /// The SEATS transitory component.
    Transitory,
    /// Forecast on the original scale.
    ForecastValue,
    /// Lower 95% bound of a forecast.
    ForecastLower95,
    /// Upper 95% bound of a forecast.
    ForecastUpper95,
    /// Forecast on the transformed scale.
    ForecastTransformedValue,
    /// Standard error of a transformed-scale forecast.
    ForecastTransformedStandardError,
    /// Regression part of a forecast's variance.
    ForecastRegressionVariance,
    /// Stochastic (ARIMA) part of a forecast's variance.
    ForecastStochasticVariance,
    /// Backcast on the original scale.
    BackcastValue,
    /// Lower 95% bound of a backcast.
    BackcastLower95,
    /// Upper 95% bound of a backcast.
    BackcastUpper95,
    /// Backcast on the transformed scale.
    BackcastTransformedValue,
    /// Standard error of a transformed-scale backcast.
    BackcastTransformedStandardError,
    /// ARMA parameter estimate.
    ParameterEstimate,
    /// ARMA parameter standard error.
    ParameterStandardError,
    /// Regression coefficient estimate.
    RegressionEstimate,
    /// Regression coefficient standard error.
    RegressionStandardError,
    /// Innovation variance.
    InnovationVariance,
    /// Log-determinant correction to the likelihood.
    LogDeterminantCorrection,
    /// Log-likelihood.
    LogLikelihood,
    /// Akaike information criterion.
    Aic,
    /// Bias-corrected Akaike information criterion.
    Aicc,
    /// Hannan-Quinn information criterion.
    HannanQuinn,
    /// Bayesian information criterion.
    Bic,
    /// ARMA parameter covariance matrix entry.
    ArmaCovariance,
    /// Regression coefficient covariance matrix entry.
    RegressionCovariance,
    /// Model residual.
    Residual,
    /// Outlier effect estimate.
    OutlierEstimate,
    /// Outlier effect standard error.
    OutlierStandardError,
    /// Outlier t-value.
    OutlierTValue,
    /// Estimated calendar effect.
    CalendarEstimated,
    /// Fixed length-of-period effect.
    CalendarFixedLength,
    /// Combined calendar effect.
    CalendarCombined,
}

impl std::fmt::Display for AdjustmentOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Component(component) => return write!(formatter, "{component} component"),
            Self::Transitory => "transitory component",
            Self::ForecastValue => "forecast",
            Self::ForecastLower95 => "forecast lower 95% bound",
            Self::ForecastUpper95 => "forecast upper 95% bound",
            Self::ForecastTransformedValue => "transformed forecast",
            Self::ForecastTransformedStandardError => "transformed forecast standard error",
            Self::ForecastRegressionVariance => "forecast regression variance",
            Self::ForecastStochasticVariance => "forecast stochastic variance",
            Self::BackcastValue => "backcast",
            Self::BackcastLower95 => "backcast lower 95% bound",
            Self::BackcastUpper95 => "backcast upper 95% bound",
            Self::BackcastTransformedValue => "transformed backcast",
            Self::BackcastTransformedStandardError => "transformed backcast standard error",
            Self::ParameterEstimate => "ARMA parameter estimate",
            Self::ParameterStandardError => "ARMA parameter standard error",
            Self::RegressionEstimate => "regression estimate",
            Self::RegressionStandardError => "regression standard error",
            Self::InnovationVariance => "innovation variance",
            Self::LogDeterminantCorrection => "log-determinant correction",
            Self::LogLikelihood => "log-likelihood",
            Self::Aic => "AIC",
            Self::Aicc => "AICC",
            Self::HannanQuinn => "Hannan-Quinn criterion",
            Self::Bic => "BIC",
            Self::ArmaCovariance => "ARMA covariance",
            Self::RegressionCovariance => "regression covariance",
            Self::Residual => "model residual",
            Self::OutlierEstimate => "outlier estimate",
            Self::OutlierStandardError => "outlier standard error",
            Self::OutlierTValue => "outlier t-value",
            Self::CalendarEstimated => "estimated calendar effect",
            Self::CalendarFixedLength => "fixed length-of-period effect",
            Self::CalendarCombined => "combined calendar effect",
        };
        formatter.write_str(name)
    }
}

/// Incompatible settings for the requested X-11 decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum X11DecompositionConflict {
    /// Seasonal filters cannot be explicitly requested for trend-only X-11.
    ExplicitSeasonalFilter,
    /// Trend-only log-additive X-11, which is undefined in Build 62.
    UnsafeLogAdditive,
}

// Public docs and messages use plain names ("SEATS optimizer", not the Fortran
// routine `SEARCH`). X-13 table codes (D11) and spec options may follow in
// brackets.

/// Why [`adjust`](crate::adjust) could not adjust a series.
///
/// A variant with named fields is one failed check and carries what a caller
/// needs to act on it: an observation index, a period or a count.
/// [`InvalidSpec`](Error::InvalidSpec), [`Unsupported`](Error::Unsupported),
/// [`Numerical`](Error::Numerical) and [`Internal`](Error::Internal) each
/// cover many checks behind an opaque payload, whose message says which one
/// failed. `Display` gives a one-line lowercase message for logs; match on
/// the variant or [`kind`](Error::kind), never on the message text. The enum
/// and its struct variants are `#[non_exhaustive]`, so more variants and
/// fields may be added.
///
/// `Error` implements `PartialEq` but not `Eq`: variants carry `f64` values,
/// and a NaN payload never equals itself.
///
/// Route on [`kind`](Error::kind), which puts every variant in one of seven
/// [`ErrorKind`]s, and match individual variants only where you handle them
/// specially:
///
/// | Kind | Meaning | Typical response |
/// |---|---|---|
/// | [`InvalidInput`](ErrorKind::InvalidInput) | the series or user-regressor data break the input contract | fix the caller |
/// | [`InvalidSpec`](ErrorKind::InvalidSpec) | the spec is invalid or does not fit the series | fix the spec |
/// | [`Unsupported`](ErrorKind::Unsupported) | a valid request this crate does not run | try another spec, such as an explicit model |
/// | [`TooShort`](ErrorKind::TooShort) | the series has too few observations for this pipeline | skip the series until it has more data |
/// | [`UnsuitableData`](ErrorKind::UnsuitableData) | this series cannot be adjusted this way | skip the series or change the request |
/// | [`Numerical`](ErrorKind::Numerical) | estimation or decomposition failed | retry with another model, such as the airline model |
/// | [`Internal`](ErrorKind::Internal) | a bug in this crate | report it |
///
/// Where X-13ARIMA-SEATS refuses an input, so does this crate, with its own
/// message. Where it would hang, crash or print a nonfinite or undefined
/// result, this crate returns an error instead.
///
/// See [`adjust`](crate::adjust) for an example of routing on the kind.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A user regressor has no values.
    #[error("{kind} regressor starting at {start} has no values")]
    #[non_exhaustive]
    EmptyUserRegressor {
        /// Role of the regressor.
        kind: crate::regarima::UserRegressorType,
        /// Period of the regressor's first value.
        start: Period,
    },
    /// A user regressor value is NaN or infinite.
    #[error("{kind} regressor value at {period} (index {index}) is not finite")]
    #[non_exhaustive]
    NonFiniteUserRegressor {
        /// Role of the regressor.
        kind: crate::regarima::UserRegressorType,
        /// Zero-based position of the value in the regressor.
        index: usize,
        /// Period of the value.
        period: Period,
    },
    /// A user regressor has a different frequency from the input series.
    #[error("{kind} regressor starting at {start} is {actual}, but the input series is {expected}")]
    #[non_exhaustive]
    UserRegressorFrequency {
        /// Role of the regressor.
        kind: crate::regarima::UserRegressorType,
        /// Frequency of the input series.
        expected: Frequency,
        /// Frequency of the regressor.
        actual: Frequency,
        /// Period of the regressor's first value.
        start: Period,
    },
    /// A user regressor does not cover the periods the adjustment needs, including
    /// forecasts and backcasts.
    #[error(
        "{kind} regressor covers {supplied_start} through {supplied_end}, but the adjustment needs {required_start} through {required_end}"
    )]
    #[non_exhaustive]
    UserRegressorCoverage {
        /// Role of the regressor.
        kind: crate::regarima::UserRegressorType,
        /// First period the regressor supplies.
        supplied_start: Period,
        /// Last period the regressor supplies.
        supplied_end: Period,
        /// First period the adjustment needs.
        required_start: Period,
        /// Last period the adjustment needs.
        required_end: Period,
    },
    /// A user regressor's dates run outside the supported year range.
    #[error("{kind} regressor of {count} values starting at {start} has an invalid date range")]
    #[non_exhaustive]
    UserRegressorDateRange {
        /// Role of the regressor.
        kind: crate::regarima::UserRegressorType,
        /// Period of the regressor's first value.
        start: Period,
        /// Number of values in the regressor.
        count: usize,
        /// The period failure.
        source: PeriodError,
    },
    /// The model span is reversed or extends beyond the series.
    #[error("model span {start} through {end} is reversed or extends beyond the series")]
    #[non_exhaustive]
    InvalidModelSpan {
        /// First period of the span.
        start: Period,
        /// Last period of the span.
        end: Period,
    },
    /// The outlier span is reversed or extends beyond the model span.
    #[error("outlier span {start} through {end} is reversed or extends beyond the model span")]
    #[non_exhaustive]
    InvalidOutlierSpan {
        /// First period of the span.
        start: Period,
        /// Last period of the span.
        end: Period,
    },
    /// A requested date (span bound or specified outlier) has a different
    /// frequency from the input series.
    #[error("{period} is a {} period, but the input series is {expected}", period.frequency())]
    #[non_exhaustive]
    PeriodFrequencyMismatch {
        /// The requested date.
        period: Period,
        /// Frequency of the input series.
        expected: Frequency,
    },
    /// A missing observation lies outside the model span, so the model cannot
    /// estimate it.
    #[error(
        "missing observation at {period} is outside the model span, so the model cannot estimate it"
    )]
    #[non_exhaustive]
    MissingOutsideModelSpan {
        /// Period of the missing observation.
        period: Period,
    },
    /// The series has more observations than SEATS accepts (see
    /// [`Spec::seats`](super::Spec::seats)).
    #[error("SEATS accepts at most {maximum} observations; the series has {actual}")]
    #[non_exhaustive]
    SeatsTooLong {
        /// Observations in the series.
        actual: usize,
        /// Most observations SEATS accepts.
        maximum: usize,
    },
    /// SEATS needs a series that starts in a nonnegative year.
    #[error("SEATS needs a nonnegative start year, not {year}")]
    #[non_exhaustive]
    SeatsNegativeYear {
        /// First year of the series.
        year: i32,
    },
    /// Pseudo-additive observations must be nonnegative.
    #[error(
        "pseudo-additive X-11 needs nonnegative observations; the observation at index {index} is negative"
    )]
    #[non_exhaustive]
    NegativeValue {
        /// Zero-based position of the first negative observation.
        index: usize,
    },
    /// A requested short-history filter has undefined endpoints for a calendar
    /// subseries in the initial centered seasonal-irregular span.
    #[error("the {requested} seasonal filter (resolved to {resolved}) needs {minimum} centered seasonal-irregular values in {} {calendar_period}, but there are {actual}; choose a shorter seasonal filter or the stable short-series fallback", frequency.subperiod_name())]
    #[non_exhaustive]
    InsufficientSeasonalHistory {
        /// Frequency of the input series.
        frequency: Frequency,
        /// One-based month or quarter whose history is too short.
        calendar_period: u8,
        /// Seasonal filter the spec requested.
        requested: x11::SeasonalFilter,
        /// Seasonal filter the request resolved to for this month or quarter.
        resolved: x11::SeasonalFilter,
        /// Centered seasonal-irregular values available.
        actual: usize,
        /// Centered seasonal-irregular values the filter needs.
        minimum: usize,
    },
    /// No observations were supplied.
    #[error("seasonal adjustment requires at least one observation")]
    EmptyInput,
    /// An input observation is infinite. (NaN marks a missing observation.)
    #[error("observation at index {index} is not finite")]
    #[non_exhaustive]
    NonFiniteValue {
        /// Zero-based position of the observation.
        index: usize,
    },
    /// A missing (NaN) observation in a request with no regARIMA model to
    /// estimate it: direct X-11 rejects gaps, as X-13ARIMA-SEATS does.
    #[error("missing observation at index {index} requires a regARIMA model")]
    #[non_exhaustive]
    MissingValue {
        /// Zero-based position of the first missing observation.
        index: usize,
    },
    /// No observations remain for a calendar month or quarter.
    #[error("every observation in {} {subperiod} is missing", frequency.subperiod_name())]
    #[non_exhaustive]
    MissingSeason {
        /// Frequency of the input series.
        frequency: Frequency,
        /// One-based month or quarter with no observations.
        subperiod: u8,
    },
    /// An all-zero series is invalid input, as in X-13ARIMA-SEATS. A series of
    /// zeros with gaps returns this too, unless a gap comes first.
    #[error("all observations are equal to zero")]
    AllZeroInput,
    /// An observation is zero or negative where the request takes logarithms
    /// or divides by the series: the log transform, and multiplicative or
    /// log-additive X-11.
    #[error(
        "observation at index {index} is zero or negative; the log transform and multiplicative or log-additive X-11 need positive values"
    )]
    #[non_exhaustive]
    NonPositiveValue {
        /// Zero-based position of the first zero or negative observation.
        index: usize,
    },
    /// A period operation overflowed or used an invalid subperiod.
    #[error(transparent)]
    InvalidPeriod(#[from] PeriodError),
    /// The series has fewer observations than the method accepts. X-11
    /// needs three years, and SEATS 36 months or 16 quarters (see
    /// [`Spec::seats`](super::Spec::seats)).
    #[error("{method} needs at least {minimum} observations; the series has {actual}")]
    #[non_exhaustive]
    TooShort {
        /// Method that refused the series.
        method: super::Method,
        /// Observations in the series.
        actual: usize,
        /// Fewest observations the method accepts.
        minimum: usize,
    },
    /// The series has the three years X-11 needs, but the fixed Henderson
    /// trend filter has more terms than the values X-11 filters. A shorter
    /// filter fits.
    #[error(
        "the {terms}-term Henderson filter is longer than the {count} values X-11 filters, \
         forecasts and backcasts included"
    )]
    #[non_exhaustive]
    HendersonLongerThanSeries {
        /// Terms in the requested Henderson filter.
        terms: usize,
        /// Values X-11 filters: the observations plus any forecasts and
        /// backcasts.
        count: usize,
    },
    /// An observation too large for the widest print format of X-13ARIMA-SEATS:
    /// more than 21 columns, so a magnitude of at least 1e19.
    #[error(
        "observation at index {index} has magnitude of at least 1e19, the largest this crate accepts"
    )]
    #[non_exhaustive]
    ObservationTooLarge {
        /// Zero-based position of the observation.
        index: usize,
    },
    /// The series plus its forecasts and backcasts exceeds the reference
    /// program's capacity in positions or in calendar years.
    #[error(
        "series with extensions spans {positions} positions over {years} calendar years; \
         the limits are {maximum_positions} positions and {maximum_years} years"
    )]
    #[non_exhaustive]
    SeriesTooLong {
        /// Positions spanned by the series with its extensions.
        positions: usize,
        /// Calendar years spanned by the series with its extensions.
        years: usize,
        /// The position limit.
        maximum_positions: usize,
        /// The calendar-year limit.
        maximum_years: usize,
    },
    /// A regression design exceeds the 80-variable capacity of X-13ARIMA-SEATS.
    #[error("regression design has {actual} variables; maximum is {maximum}")]
    #[non_exhaustive]
    TooManyRegressionVariables {
        /// Regression variables in the design.
        actual: usize,
        /// The limit.
        maximum: usize,
    },
    /// A specified outlier is duplicated, outside the observed series, or a
    /// level shift at the first or last observation. Outliers outside the
    /// model span are dropped with
    /// [`Warning::ManualOutlierOutsideModelSpan`](crate::Warning::ManualOutlierOutsideModelSpan),
    /// not rejected.
    #[error(
        "specified {kind} at {period} is duplicated, outside the series, or a level shift at its first or last observation"
    )]
    #[non_exhaustive]
    InvalidManualOutlier {
        /// Type of the specified outlier.
        kind: crate::regarima::OutlierType,
        /// Period of the specified outlier.
        period: Period,
    },
    /// A calendar effect requires a year within its reference-table domain.
    #[error("{feature} effects are not supported in {year}; supported years are {}", Years { minimum: *minimum, maximum: *maximum })]
    #[non_exhaustive]
    CalendarYearOutOfRange {
        /// Requested calendar effect.
        feature: CalendarFeature,
        /// Year outside the effect's table.
        year: i32,
        /// First supported year.
        minimum: i32,
        /// Last supported year, or `None` when the table has no upper limit.
        maximum: Option<i32>,
    },
    /// Differencing or model structure leaves too few effective observations.
    #[error("model requires at least {minimum} effective observations; received {actual}")]
    #[non_exhaustive]
    InsufficientEffectiveObservations {
        /// Effective observations available.
        actual: usize,
        /// Effective observations needed.
        minimum: usize,
    },
    /// The regression normal equations are singular at one model variable.
    #[error("regression design is singular at pivot {pivot} ({variable})")]
    #[non_exhaustive]
    SingularRegression {
        /// Zero-based pivot (column) at which the factorization failed.
        pivot: usize,
        /// Regression variable at that pivot.
        variable: RegressionVariable,
    },
    /// The regression has as many estimated columns as effective observations
    /// or more, leaving no residual degrees of freedom for its covariance.
    #[error(
        "regression has {columns} estimated columns but only {effective_observations} effective observations"
    )]
    #[non_exhaustive]
    RegressionColumnsExceedObservations {
        /// Estimated regression columns.
        columns: usize,
        /// Effective observations after differencing.
        effective_observations: usize,
    },
    /// Differencing reduces the series to zeros (a linear trend under two
    /// differences, for example), so the regARIMA objective is identically
    /// zero. SEATS also returns it for a constant series, which leaves no
    /// estimated model to decompose.
    #[error("the series is constant or differences to zeros, so no model can be estimated")]
    DegenerateSeries,
    /// Nonlinear estimation exhausted its iteration limit.
    #[error("regARIMA estimation did not converge after {iterations} iterations")]
    #[non_exhaustive]
    EstimationDidNotConverge {
        /// Iterations performed.
        iterations: usize,
    },
    /// An out-of-range setting, or settings that cannot be combined with each
    /// other or with the series' frequency. The message names the setting.
    #[error("{0}")]
    InvalidSpec(InvalidSpec),
    /// A valid request this crate does not run. The message says what and,
    /// where known, what to try instead.
    #[error("{0}")]
    Unsupported(Unsupported),
    /// Estimation or decomposition failed. The message gives the step, the
    /// period where one applies, and the cause;
    /// [`std::error::Error::source`] returns the lower-level failure when
    /// there is one.
    #[error(transparent)]
    Numerical(NumericalFailure),
    /// An internal consistency check failed. This is a bug in this crate, not
    /// a property of the input: please report it.
    #[error("{0}")]
    Internal(InternalError),
}

/// Why a spec was refused: the payload of [`Error::InvalidSpec`]. Its
/// `Display` names the setting or the combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InvalidSpec(pub(crate) SpecProblem);

/// What this crate does not run: the payload of [`Error::Unsupported`]. Its
/// `Display` says what and, where known, what to try instead.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unsupported(pub(crate) UnsupportedDetail);

/// Where estimation or decomposition failed: the payload of
/// [`Error::Numerical`]. Its `Display` gives the step, the period where one
/// applies, and the cause; [`source`](std::error::Error::source) returns the
/// lower-level failure when there is one.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericalFailure(pub(crate) NumericalDetail);

/// A failed internal consistency check: the payload of [`Error::Internal`].
/// Its `Display` names the check.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InternalError(pub(crate) &'static str);

impl std::error::Error for InvalidSpec {}

impl std::error::Error for Unsupported {}

impl std::error::Error for InternalError {}

impl std::fmt::Display for InvalidSpec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::fmt::Display for NumericalFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for NumericalFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.0)
    }
}

impl std::fmt::Display for InternalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "internal error: {}; this is a bug in x13",
            self.0
        )
    }
}

impl From<SpecProblem> for Error {
    fn from(problem: SpecProblem) -> Self {
        Self::InvalidSpec(InvalidSpec(problem))
    }
}

impl From<UnsupportedDetail> for Error {
    fn from(detail: UnsupportedDetail) -> Self {
        Self::Unsupported(Unsupported(detail))
    }
}

impl From<NumericalDetail> for Error {
    fn from(detail: NumericalDetail) -> Self {
        Self::Numerical(NumericalFailure(detail))
    }
}

impl From<crate::seats::Error> for Error {
    fn from(error: crate::seats::Error) -> Self {
        Self::seats(error.stage, error.failure)
    }
}

/// Coarse class of an [`Error`](enum@Error), for callers that route on the
/// cause rather than the variant (see [`Error::kind`]).
///
/// Every variant belongs to exactly one kind. More kinds may be added, so
/// match with a wildcard arm.
///
/// A check that reads only the series or user-regressor data is
/// [`InvalidInput`](ErrorKind::InvalidInput). One that reads only the spec, or
/// the spec against the series' frequency and span, is
/// [`InvalidSpec`](ErrorKind::InvalidSpec). One that needs both to fail, such
/// as a log of a nonpositive value, is
/// [`UnsuitableData`](ErrorKind::UnsuitableData), except that a series with
/// fewer observations than the pipeline accepts at all is
/// [`TooShort`](ErrorKind::TooShort).
///
/// ```
/// use x13::{ErrorKind, Period, Series, Spec, adjust, x11};
///
/// let start = Period::monthly(2020, 1)?;
/// let spec = Spec::x11(x11::Options::new());
/// let error = adjust(Series::new(start, &[]), &spec).unwrap_err();
/// assert_eq!(error.kind(), ErrorKind::InvalidInput);
///
/// let error = adjust(Series::new(start, &[100.0; 12]), &spec).unwrap_err();
/// assert_eq!(error.kind(), ErrorKind::TooShort);
///
/// let error = adjust(Series::new(start, &[-1.0; 36]), &spec).unwrap_err();
/// assert_eq!(error.kind(), ErrorKind::UnsuitableData);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The caller's data breaks the input contract whatever the spec: no
    /// observations, an infinite value, a series that runs past the supported
    /// years, or user-regressor data that is empty, nonfinite, of another
    /// frequency or too short. This is a bug in the caller; retrying cannot
    /// help.
    InvalidInput,
    /// The spec is invalid in itself or does not fit the series' frequency or
    /// dates: an out-of-range setting, a conflicting combination, or a span
    /// or date that misses the series. Fix the spec.
    InvalidSpec,
    /// A valid request that this crate does not run, although X-13ARIMA-SEATS
    /// may, so this is not a verdict on the data. Another spec may succeed:
    /// when automatic selection picks a model SEATS does not run here, or
    /// SEATS would re-estimate the fitted model with smaller orders, an
    /// explicit model such as the airline model may work, so a caller that
    /// retries only on [`ErrorKind::Numerical`] gives those series up.
    Unsupported,
    /// The series has fewer observations than the pipeline accepts at all,
    /// whatever the request: X-11's three years, or SEATS's minimum. No
    /// change of options helps; the series needs more data. A series that
    /// is long enough for the pipeline but too short for one request, such as
    /// a model with many parameters or a Henderson filter longer than the
    /// series, is [`UnsuitableData`](ErrorKind::UnsuitableData) instead,
    /// because a smaller request may fit.
    TooShort,
    /// The input and spec are valid, but this series cannot be adjusted this
    /// way: too long or too wide for the reference program's capacity; too
    /// short for the requested model, regressors or Henderson filter, although
    /// long enough for the pipeline; nonpositive under a log or multiplicative
    /// decomposition; all zero, constant or degenerate; gaps the request cannot
    /// estimate; or dates outside a calendar table. Skip the series or change
    /// the request: a regARIMA pipeline fills gaps, no transform or additive
    /// X-11 accepts nonpositive values, and a smaller model, fewer regressors
    /// or a shorter filter fits a short sample.
    UnsuitableData,
    /// Estimation or decomposition failed for the fitted model: no
    /// convergence, a singular or otherwise unusable numerical state, invalid
    /// roots, an automatic selection that cannot score or test a candidate,
    /// no admissible SEATS decomposition, an undefined X-11 intermediate, or
    /// a nonfinite result. A different model, such as the airline model, may
    /// succeed.
    Numerical,
    /// A bug in this crate: an internal consistency check failed
    /// ([`Error::Internal`]). No change of input or spec is known to help.
    /// Please report it.
    Internal,
}

impl Error {
    /// A failed internal consistency check, named by a short phrase such as
    /// `"prior adjustment length differs from the series"`.
    pub(crate) const fn internal(check: &'static str) -> Self {
        Self::Internal(InternalError(check))
    }

    /// A numerical failure of `operation` with no more specific cause and no
    /// lower-level failure to keep. Use [`Error::numerical_from`] where an
    /// inner error exists.
    pub(crate) const fn numerical(operation: NumericalOperation) -> Self {
        Self::Numerical(NumericalFailure(NumericalDetail::Unspecified {
            operation,
            source: None,
        }))
    }

    /// A numerical failure of `operation` that keeps the lower-level failure
    /// as its source.
    pub(crate) fn numerical_from(operation: NumericalOperation, source: Self) -> Self {
        Self::Numerical(NumericalFailure(NumericalDetail::Unspecified {
            operation,
            source: Some(Box::new(source)),
        }))
    }

    /// A SEATS step failure at `stage`: a model change SEATS proposes is
    /// unsupported, every other failure numerical.
    pub(crate) fn seats(stage: crate::seats::Stage, failure: crate::seats::Failure) -> Self {
        let crate::seats::Failure::ModelChange(change) = failure else {
            return NumericalDetail::Seats { stage, failure }.into();
        };
        let model = |orders: crate::seats::Orders| {
            let [p, d, q, bp, bd, bq] = orders.to_array();
            crate::regarima::ResolvedArimaModel::new([p, d, q], [bp, bd, bq])
        };
        let current = match model(change.current) {
            Ok(current) => current,
            Err(error) => return error,
        };
        match model(change.proposed) {
            Ok(proposed) => UnsupportedDetail::SeatsModelChange { current, proposed }.into(),
            Err(error) => error,
        }
    }

    /// The coarse class of this error.
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        use ErrorKind as K;
        // No wildcard: a new variant must be classified here.
        match self {
            Self::EmptyInput
            | Self::NonFiniteValue { .. }
            | Self::InvalidPeriod(_)
            | Self::EmptyUserRegressor { .. }
            | Self::NonFiniteUserRegressor { .. }
            | Self::UserRegressorFrequency { .. }
            | Self::UserRegressorCoverage { .. }
            | Self::UserRegressorDateRange { .. } => K::InvalidInput,
            Self::InvalidModelSpan { .. }
            | Self::InvalidOutlierSpan { .. }
            | Self::PeriodFrequencyMismatch { .. }
            | Self::InvalidManualOutlier { .. }
            | Self::InvalidSpec(_) => K::InvalidSpec,
            Self::Unsupported(_) => K::Unsupported,
            Self::TooShort { .. } => K::TooShort,
            Self::MissingValue { .. }
            | Self::MissingSeason { .. }
            | Self::MissingOutsideModelSpan { .. }
            | Self::AllZeroInput
            | Self::NonPositiveValue { .. }
            | Self::NegativeValue { .. }
            | Self::HendersonLongerThanSeries { .. }
            | Self::InsufficientEffectiveObservations { .. }
            | Self::RegressionColumnsExceedObservations { .. }
            | Self::InsufficientSeasonalHistory { .. }
            | Self::SeatsTooLong { .. }
            | Self::SeatsNegativeYear { .. }
            | Self::ObservationTooLarge { .. }
            | Self::SeriesTooLong { .. }
            | Self::TooManyRegressionVariables { .. }
            | Self::CalendarYearOutOfRange { .. }
            | Self::DegenerateSeries => K::UnsuitableData,
            Self::EstimationDidNotConverge { .. }
            | Self::Numerical(_)
            | Self::SingularRegression { .. } => K::Numerical,
            Self::Internal(_) => K::Internal,
        }
    }
}

// Message helpers: optional context that reads as part of a clause.

/// A calendar table's year range: `1901 through 2100`, or `1901 onward`.
struct Years {
    minimum: i32,
    maximum: Option<i32>,
}

impl std::fmt::Display for Years {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.maximum {
            Some(maximum) => write!(formatter, "{} through {maximum}", self.minimum),
            None => write!(formatter, "{} onward", self.minimum),
        }
    }
}

display_phrases!(super::Method {
    X11 => "X-11",
    Seats => "SEATS",
});

display_phrases!(X11ShrinkageStage {
    FilterWeight => "filter-weight calculation",
    Variance => "variance calculation",
    GlobalWeight => "global-weight calculation",
    LocalLikelihood => "local-likelihood calculation",
    SeasonalFactors => "seasonal-factor calculation",
});

impl std::fmt::Display for X11ShrinkageUnavailableReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFilter { filter } => {
                write!(
                    formatter,
                    "shrinkage is not defined after the {filter} seasonal filter"
                )
            }
            Self::InsufficientAcceptedForecasts { actual, minimum } => write!(
                formatter,
                "it needs {minimum} accepted forecasts, but there are {actual}"
            ),
            Self::NonPositiveLocalVariance => {
                formatter.write_str("a local variance is not positive")
            }
            Self::ZeroDenominator => formatter.write_str("a denominator is zero"),
            Self::NonPositiveLikelihoodSum => {
                formatter.write_str("the likelihood sum is not positive")
            }
            Self::NonFiniteArithmetic => formatter.write_str("the arithmetic is not finite"),
        }
    }
}

display_phrases!(X11ExtremeScaleUnavailableReason {
    EmptySample => "the sample is empty",
    ZeroTauBase => "the tau base is zero",
    NonFiniteDeviation => "a deviation is not finite",
    NonPositiveLogInput => "a log-scale input is not positive",
    NonFiniteScale => "the scale is not finite",
});

display_phrases!(X11TrendUnavailableReason {
    NoPositiveNeighbor => "no neighboring estimate is positive",
    NonFiniteEstimate => "the estimate is not finite",
    NonFiniteReplacement => "the replacement estimate is not finite",
});

display_phrases!(X11ExtremeStage {
    InitialSeasonalIrregular => "initial seasonal-irregular component",
    RefinedSeasonalIrregular => "refined seasonal-irregular component",
    PreliminaryIrregular => "preliminary irregular component",
    IntermediateIrregular => "intermediate irregular component",
});

display_phrases!(X11ExtremeWeightingPass {
    Initial => "initial",
    Reweighted => "reweighted",
});

display_phrases!(ArimaOrderTerm {
    Autoregressive => "AR",
    Differencing => "differencing",
    MovingAverage => "MA",
});

display_phrases!(ExtensionDirection {
    Forecast => "forecast",
    Backcast => "backcast",
});

display_phrases!(X11PseudoAdditiveStage {
    InitialSeasonalAdjustment => "initial seasonal adjustment",
    SeasonalIrregular => "seasonal-irregular calculation",
    Irregular => "irregular calculation",
    ExtremeReconstruction => "extreme-value reconstruction",
    SeasonalSelection => "seasonal selection",
    TrendSelection => "trend selection",
    FinalAdjustment => "final adjustment",
});

display_phrases!(X11PseudoAdditiveUnavailableReason {
    ZeroDivisor => "a divisor is zero",
    NonFiniteValue => "a value is not finite",
});

display_phrases!(X11LogAdditiveStage {
    InputLog => "the logarithm of the input",
    SeasonalExponentiation => "seasonal exponentiation",
    TrendExponentiation => "trend exponentiation",
    IrregularVariance => "the irregular variance (C13)",
    BiasFactor => "the bias-factor calculation",
    TrendBias => "the trend bias correction",
});

display_phrases!(X11LogAdditiveUnavailableReason {
    NonPositiveInput => "an input is not positive",
    NonFiniteValue => "a value is not finite",
    ZeroSeasonalFactor => "a seasonal factor underflowed to zero",
});

display_phrases!(UnsupportedConfiguration {
    AutomaticTransform => "automatic transformation selection on this path",
    AutomaticModel => "automatic model selection on this path",
    AutomaticModelConstant => "a fixed constant with automatic model selection",
    CalendarSelection => "AICC calendar selection combined with automatic model or transformation \
        selection, outlier search, no calendar effect to test, a standalone leap-year or \
        length-of-period basis, a length-of-period treatment or legacy X-11 Easter",
    ManualOutliers => "specified outliers combined with automatic model or transformation \
        selection or outlier search",
});

display_phrases!(TransformSelectionUnavailableReason {
    ZeroVariance => "the fitted innovation variance is zero",
    InsufficientEffectiveObservations => "too few effective observations for the AICC",
    NonFiniteScore => "the likelihood or AICC is not finite",
});

display_phrases!(CalendarFeature {
    TradingDay => "trading-day",
    StockTradingDay => "stock trading-day",
    Easter => "Easter",
    LaborDay => "Labor Day",
    Thanksgiving => "Thanksgiving",
});

display_phrases!(NumericalOperation {
    ExactLikelihood => "exact likelihood evaluation",
    Estimation => "parameter estimation",
    Covariance => "parameter covariance calculation",
    Extension => "forecasting or backcasting",
    OutlierSearch => "automatic outlier search",
});

display_phrases!(ModelSelectionConstantTest {
    Initial => "initial constant test",
    AfterOutlierSearch => "constant test after outlier search",
    BeforeFinalCorrections => "constant test before the final model corrections",
    AfterFinalRootRepair => "constant test after the final AR-root repair",
    AfterOverdifferenceRepair => "constant test after the overdifferencing repair",
});

display_phrases!(X11DecompositionConflict {
    ExplicitSeasonalFilter => "trend-only X-11 does not accept an explicit seasonal filter",
    UnsafeLogAdditive => "trend-only log-additive X-11 is undefined in Build 62",
});

#[cfg(test)]
mod tests;
