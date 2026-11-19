//! What the opaque payloads of [`Error`] hold: one enum per payload, with a
//! variant for each failed check and the values that describe it.

use super::{
    AdjustmentOutput, ArimaOrderTerm, CalendarFeature, Error, ExactLikelihoodFailure,
    ExtensionDirection, ModelSelectionConstantTest, NumericalOperation,
    TransformSelectionUnavailableReason, UnsupportedConfiguration, X11DecompositionConflict,
    X11ExtremeScaleUnavailableReason, X11ExtremeStage, X11ExtremeWeightingPass,
    X11LogAdditiveStage, X11LogAdditiveUnavailableReason, X11PseudoAdditiveStage,
    X11PseudoAdditiveUnavailableReason, X11ShrinkageStage, X11ShrinkageUnavailableReason,
    X11TrendUnavailableReason,
};
use crate::api::seats::{SeatsForecastField, SeatsUnsupportedOption};
use crate::api::{
    ArimaModel, Transform, X11EasterConflict, X11EasterStage, X11EasterUnavailableReason,
};
use crate::regarima::{
    ArimaFactor, ConstantSelection, ModelSelection, ParameterKind, ResolvedArimaModel,
};
use crate::x11;
use crate::{Frequency, Period};

/// Why a spec was refused: a setting out of range, or settings that cannot be
/// combined with each other or with the series' frequency.
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
pub(crate) enum SpecProblem {
    /// An explicit Henderson length is even or outside 3 through 101.
    #[error("Henderson length must be odd and between 3 and 101, not {terms}")]
    HendersonLength {
        /// Requested Henderson length.
        terms: u8,
    },
    /// A Henderson endpoint I/C ratio is not finite and positive.
    #[error("Henderson endpoint I/C ratio must be finite and positive")]
    HendersonIcRatio,
    /// X-11 sigma limits are not finite, positive and ordered.
    #[error("X-11 sigma limits must be finite and satisfy 0 < lower <= upper")]
    SigmaLimits,
    /// An ARIMA order exceeds its limit. An order does not know which factor
    /// it will fill, so `ArimaOrder::new` checks the regular limits and
    /// reports the nonseasonal factor; `ArimaModel::new` then checks its
    /// seasonal factor against the tighter seasonal limits.
    #[error("{factor} {term} order {value} exceeds the maximum of {maximum}")]
    ArimaOrder {
        /// Nonseasonal or seasonal factor.
        factor: ArimaFactor,
        /// Operator whose order is too large.
        term: ArimaOrderTerm,
        /// Requested order.
        value: u8,
        /// Largest supported order.
        maximum: u8,
    },
    /// A forecast or backcast horizon exceeds the maximum.
    #[error("{direction} horizon of {periods} periods exceeds the maximum of {maximum}")]
    ExtensionHorizon {
        /// Forecast or backcast.
        direction: ExtensionDirection,
        /// Requested horizon in periods.
        periods: u16,
        /// Largest supported horizon.
        maximum: u16,
    },
    /// A stock trading-day sample day is outside `1..=31`.
    #[error("stock trading-day sample day {sample_day} is not between 1 and 31")]
    StockTradingDaySampleDay {
        /// Requested sample day.
        sample_day: u8,
    },
    /// An Easter window is outside `0..=25`.
    #[error("Easter window {window} is not between 0 and 25")]
    EasterWindow {
        /// Requested window in days.
        window: u8,
    },
    /// A Labor Day window is outside `1..=25`.
    #[error("Labor Day window {window} is not between 1 and 25")]
    LaborDayWindow {
        /// Requested window in days.
        window: u8,
    },
    /// A Thanksgiving offset is outside the two supported ranges.
    #[error("Thanksgiving offset {offset} is not between -8 and -1 or between 1 and 17")]
    ThanksgivingOffset {
        /// Requested offset in days.
        offset: i8,
    },
    /// An automatic-outlier critical value is nonpositive or nonfinite.
    #[error("automatic-outlier critical value must be finite and positive")]
    OutlierCriticalValue,
    /// A temporary-change rate is outside the open interval from zero to one.
    #[error("temporary-change rate must be finite and strictly between zero and one")]
    TemporaryChangeRate,
    /// The AICC difference for automatic transformation selection is NaN or
    /// infinite.
    #[error("automatic transformation AICC difference must be finite")]
    TransformAicDifference,
    /// A requested decomposition conflicts with another X-11 setting.
    #[error("incompatible X-11 decomposition: {reason}")]
    IncompatibleX11Decomposition {
        /// The conflicting setting.
        reason: X11DecompositionConflict,
    },
    /// Legacy X-11 Easter estimation conflicts with another part of the
    /// request.
    #[error("legacy X-11 Easter estimation is incompatible with this request: {reason}")]
    IncompatibleX11Easter {
        /// The conflicting setting.
        reason: X11EasterConflict,
    },
    /// Pseudo-additive X-11 cannot take regression preadjustment;
    /// X-13ARIMA-SEATS refuses preadjustment factors derived from a regARIMA
    /// model. Raised before the fit for calendar effects, manual outliers or an
    /// automatic outlier search, and after it when the fitted model holds
    /// calendar, outlier or user regressors. A gap is accepted only without a
    /// constant term or removal of additive outliers, because the gap's
    /// regressor counts as an additive outlier.
    #[error(
        "pseudo-additive X-11 cannot take regression preadjustment: calendar effects, outliers (manual or automatic search), user regressors, or gaps in the series together with a constant term or additive-outlier removal"
    )]
    IncompatiblePseudoAdditivePreadjustment,
    /// On a series without gaps, pseudo-additive final-outlier removal needs a
    /// fixed model with a constant term, as X-13ARIMA-SEATS requires.
    #[error(
        "pseudo-additive X-11 final-outlier removal on a series without gaps needs a fixed model with a constant term (model {model}, constant {constant})"
    )]
    IncompatiblePseudoAdditiveFinalOutliers {
        /// Requested model.
        model: ModelSelection,
        /// Requested constant.
        constant: ConstantSelection,
    },
    /// Automatic selection determines the X-11 mode, so an explicit mode
    /// cannot be combined with it.
    #[error(
        "automatic transformation selection cannot be combined with an explicit X-11 mode; let the mode follow the selected transform"
    )]
    AutomaticTransformWithExplicitMode,
    /// Calendar-specific X-11 filters do not match the input frequency.
    #[error(
        "the calendar-specific seasonal filters are for {expected} series, but the input series is {actual}"
    )]
    SeasonalFilterFrequencyMismatch {
        /// Frequency the seasonal filters are written for.
        expected: Frequency,
        /// Frequency of the input series.
        actual: Frequency,
    },
    /// Calendar-specific extreme scales do not match the input frequency.
    #[error(
        "the calendar groups for extreme-value scales are for {expected} series, but the input series is {actual}"
    )]
    ExtremeScaleFrequencyMismatch {
        /// Frequency the calendar groups are written for.
        expected: Frequency,
        /// Frequency of the input series.
        actual: Frequency,
    },
    /// A calendar effect is unavailable at the input frequency.
    #[error("{feature} effects are not supported for {frequency} series")]
    UnsupportedCalendarFrequency {
        /// Requested calendar effect.
        feature: CalendarFeature,
        /// Frequency of the input series.
        frequency: Frequency,
    },
    /// Regression adjustments need a transform that matches the X-11 mode.
    #[error(
        "regression effects estimated with transform {transform} are incompatible with {mode} X-11 decomposition"
    )]
    IncompatibleRegressionAdjustmentMode {
        /// Transform the regression effects were estimated with.
        transform: Transform,
        /// Requested X-11 decomposition.
        mode: x11::Mode,
    },
}

/// A valid request this crate does not run.
#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
pub(crate) enum UnsupportedDetail {
    /// A combination of settings this crate does not run.
    #[error("unsupported configuration: {0}")]
    Configuration(UnsupportedConfiguration),
    /// A SEATS request this crate does not run: an explicit model outside the
    /// SEATS models, a calendar spec with no effect, or more than three years
    /// of forecasts.
    #[error("SEATS does not support {0}")]
    SeatsOption(SeatsUnsupportedOption),
    /// Automatic selection chose a valid model that SEATS does not run in
    /// this crate. An explicit model such as the airline model may succeed.
    #[error("SEATS does not support the automatically selected model {selected} (first identified as {first_model}, mean {}); an explicit model such as (0 1 1)(0 1 1) may succeed", if *mean_significant { "significant" } else { "not significant" })]
    SeatsSelectedModel {
        /// Model chosen by automatic selection.
        selected: ResolvedArimaModel,
        /// First model identified by automatic selection.
        first_model: ArimaModel,
        /// Whether the selected model's mean is significant.
        mean_significant: bool,
    },
    /// SEATS would re-estimate the fitted model with smaller orders, as
    /// Build 62's CHMODEL and ChangeModel do.
    #[error(
        "SEATS would re-estimate the model {current} as {proposed}, as Build 62 does when a fitted AR root reaches the unit circle or a seasonal AR factor cancels the seasonal MA; this crate does not follow that change"
    )]
    SeatsModelChange {
        /// The fitted model.
        current: ResolvedArimaModel,
        /// The orders Build 62 would re-estimate.
        proposed: ResolvedArimaModel,
    },
}

/// Where estimation or decomposition failed, and on what.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub(crate) enum NumericalDetail {
    /// Legacy X-11 Easter estimation reached an undefined or unusable value.
    #[error("legacy X-11 Easter estimation failed during {stage}{}: {reason}", At(*period))]
    X11Easter {
        /// Easter estimation step that failed.
        stage: X11EasterStage,
        /// Period involved, when the failure has one.
        period: Option<Period>,
        /// What was undefined or unusable.
        reason: X11EasterUnavailableReason,
    },
    /// Requested seasonal shrinkage reached an undefined or unusable value.
    #[error(
        "{policy} seasonal shrinkage over {first} through {last} failed during {stage}: {reason}"
    )]
    X11Shrinkage {
        /// Requested shrinkage.
        policy: x11::SeasonalShrinkage,
        /// First period of the shrinkage span.
        first: Period,
        /// Last period of the shrinkage span.
        last: Period,
        /// Shrinkage step that failed.
        stage: X11ShrinkageStage,
        /// What was undefined or unusable.
        reason: X11ShrinkageUnavailableReason,
    },
    /// A pseudo-additive operation is undefined on the accepted X-11 span.
    #[error("pseudo-additive X-11 failed during {stage} at {period} (index {index}): {reason}")]
    X11PseudoAdditive {
        /// Pseudo-additive step that failed.
        stage: X11PseudoAdditiveStage,
        /// Zero-based position in the X-11 span.
        index: usize,
        /// Period of the failing value.
        period: Period,
        /// What was undefined or unusable.
        reason: X11PseudoAdditiveUnavailableReason,
    },
    /// A log-additive operation cannot produce a usable value on this span.
    #[error("log-additive X-11 failed during {stage} over {first} through {last}: {reason}")]
    X11LogAdditive {
        /// Log-additive step that failed.
        stage: X11LogAdditiveStage,
        /// First period of the affected span.
        first: Period,
        /// Last period of the affected span.
        last: Period,
        /// What was undefined or unusable.
        reason: X11LogAdditiveUnavailableReason,
    },
    /// A nonpositive multiplicative trend estimate cannot be corrected at
    /// this X-11 step.
    #[error("X-11 cannot correct the {stage} estimate at {period}: {reason}")]
    X11Trend {
        /// Trend estimate that could not be corrected.
        stage: x11::TrendStage,
        /// Period of the estimate.
        period: Period,
        /// Why the estimate could not be corrected.
        reason: X11TrendUnavailableReason,
    },
    /// A robust X-11 extreme-value scale has no usable result over this span.
    #[error("X-11 cannot compute the {estimator} extreme-value scale for the {stage} ({pass} pass) over {first} through {last}{}: {reason}", InGroup(*group))]
    X11ExtremeScale {
        /// Requested scale estimator.
        estimator: x11::ExtremeEstimator,
        /// Why the scale is unavailable.
        reason: X11ExtremeScaleUnavailableReason,
        /// X-11 values whose scale was being estimated.
        stage: X11ExtremeStage,
        /// Initial or reweighted estimate.
        pass: X11ExtremeWeightingPass,
        /// First period of the estimation span.
        first: Period,
        /// Last period of the estimation span.
        last: Period,
        /// Calendar group of the scale, or `None` for a pooled scale.
        group: Option<x11::SigmaGroup>,
    },
    /// Every candidate of an automatic ARMA order search for `factor` failed
    /// estimation, so no order can be chosen. X-13ARIMA-SEATS stops with
    /// "Cannot make a choice of seasonal ARMA order due to model estimation
    /// errors" in its seasonal search, and "Cannot make a choice of
    /// nonseasonal ARMA order …" in its regular search.
    #[error(
        "automatic model identification cannot choose {factor} ARMA orders: every candidate failed estimation"
    )]
    NoEstimableArmaCandidate {
        /// Nonseasonal or seasonal order search.
        factor: ArimaFactor,
    },
    /// The centered residual ACF/Ljung–Box check has no positive variance.
    #[error(
        "automatic model selection unavailable: the residual check for {model} has no positive centered variance"
    )]
    ModelSelectionResidualCheckUnavailable {
        /// Model whose residual check could not be computed.
        model: ResolvedArimaModel,
    },
    /// A required constant test has no usable statistic.
    #[error("automatic model selection unavailable: the {test} for {model} has no valid statistic")]
    ModelSelectionConstantTestUnavailable {
        /// Model whose constant was being tested.
        model: ResolvedArimaModel,
        /// Test that had no valid statistic.
        test: ModelSelectionConstantTest,
    },
    /// An otherwise eligible automatic-model candidate has no valid
    /// likelihood score.
    #[error(
        "automatic model selection unavailable: candidate {candidate} has no valid likelihood score"
    )]
    ModelSelectionUnavailable {
        /// Candidate model without a valid score.
        candidate: ResolvedArimaModel,
    },
    /// An automatic transformation candidate has no valid AICC.
    #[error(
        "automatic transformation selection has no valid AICC for candidate transform {candidate}: {reason}"
    )]
    TransformSelectionUnavailable {
        /// Candidate transform without a valid score.
        candidate: Transform,
        /// Why the score is unavailable.
        reason: TransformSelectionUnavailableReason,
    },
    /// A calendar candidate has no finite AICC, so the candidates cannot be
    /// compared.
    #[error("calendar selection has no valid AICC for the model {} the {feature} effect", if *included { "with" } else { "without" })]
    CalendarSelectionUnavailable {
        /// Calendar effect being tested.
        feature: CalendarFeature,
        /// Whether the failing candidate model includes the effect.
        included: bool,
    },
    /// A fitted AR or MA polynomial has invalid roots.
    #[error("the fitted {} polynomial has invalid roots", match kind { ParameterKind::Ar => "AR", ParameterKind::Ma => "MA" })]
    InvalidRoots {
        /// AR or MA roots.
        kind: ParameterKind,
    },
    /// Retained MA estimates have a root strictly inside the unit circle at
    /// refit setup, before near-unit shrinkage. This is not an on-unit root or
    /// a generic filtering failure.
    #[error("retained {factor} MA starting parameters have roots inside the unit circle")]
    InvalidMaStartingRoots {
        /// Nonseasonal or seasonal MA factor.
        factor: ArimaFactor,
    },
    /// The exact ARMA likelihood cannot be initialized at the current
    /// parameter estimates.
    #[error("exact ARMA likelihood unavailable: {failure}")]
    ExactLikelihoodUnavailable {
        /// Why the likelihood cannot be initialized.
        failure: ExactLikelihoodFailure,
    },
    /// The likelihood's adjustment for the transform and prior factors (its
    /// Jacobian term) is unusable at this observation.
    #[error("likelihood prior adjustment is unusable at observation index {index}")]
    LikelihoodPriorUnavailable {
        /// Zero-based position of the observation.
        index: usize,
    },
    /// Automatic-outlier residuals have no nonzero robust scale.
    #[error("automatic-outlier search requires a nonzero robust residual scale")]
    ZeroOutlierRobustScale,
    /// Automatic-outlier backward deletion has no nonzero residual scale.
    #[error(
        "automatic-outlier backward deletion requires a nonzero residual root mean square error"
    )]
    ZeroOutlierDeletionScale,
    /// The automatic critical-value calculation failed.
    #[error("automatic-outlier critical value could not be computed")]
    OutlierCriticalValueComputation,
    /// A published output is NaN or infinite. Build 62 may print such a
    /// value (+Infinity, -Infinity) and continue; it is never returned here.
    #[error("{output} is not finite at index {index}")]
    NonFiniteOutput {
        /// The output that is not finite.
        output: AdjustmentOutput,
        /// Position within the output; see [`AdjustmentOutput`].
        index: usize,
    },
    /// A regARIMA forecast passed to SEATS is not finite. `index` is
    /// zero-based within the requested forecasts.
    #[error("the regARIMA {field} at {period} (forecast index {index}) passed to SEATS is {value}")]
    SeatsForecast {
        /// Which forecast output is not finite.
        field: SeatsForecastField,
        /// Zero-based position in the requested forecasts.
        index: usize,
        /// Period of the forecast.
        period: Period,
        /// The rejected value: NaN or an infinity.
        value: f64,
    },
    /// A SEATS step failed on the fitted model.
    #[error("SEATS failed during {stage}: {failure}")]
    Seats {
        /// The step that failed.
        stage: crate::seats::Stage,
        /// What failed, with the values involved.
        failure: crate::seats::Failure,
    },
    /// A numerical operation failed without a more specific cause.
    #[error("numerical failure during {operation}")]
    Unspecified {
        /// Operation that failed.
        operation: NumericalOperation,
        /// The lower-level failure, when the operation saw one that this
        /// crate can name, such as a singular initial-condition covariance in
        /// the exact ARMA filter.
        source: Option<Box<Error>>,
    },
}

// Optional context that reads as part of a clause.

/// ` at 2020-03`, or nothing when the period is unknown.
struct At(Option<Period>);

impl std::fmt::Display for At {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(period) => write!(formatter, " at {period}"),
            None => Ok(()),
        }
    }
}

/// ` in the selected calendar group`, or nothing for a pooled scale.
struct InGroup(Option<x11::SigmaGroup>);

impl std::fmt::Display for InGroup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(group) => write!(formatter, " in the {group} calendar group"),
            None => Ok(()),
        }
    }
}
