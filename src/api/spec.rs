//! Adjustment requests: [`Spec`], [`RegArimaSpec`] and their parts.

use super::regarima::{
    ArimaFactor, CalendarBasis, CalendarSelection, CalendarSpec, FlowLengthTreatment, ForecastSpec,
    ManualOutlier, OutlierSpec, TemporaryChangeRate,
};
use super::{ArimaOrderTerm, Error, UnsupportedConfiguration, UnsupportedDetail};
use crate::Frequency;
use crate::error::SpecProblem;
use crate::regarima::{CalendarFeature, ConstantSelection, ModelSelection};
use crate::time::TimeAxis;
use crate::x11;
use std::sync::Arc;

pub(super) const MAX_REGULAR_AR_ORDER: u8 = 3;
pub(super) const MAX_REGULAR_DIFFERENCING_ORDER: u8 = 2;
pub(super) const MAX_REGULAR_MA_ORDER: u8 = 3;
pub(super) const MAX_SEASONAL_AR_ORDER: u8 = 2;
pub(super) const MAX_SEASONAL_DIFFERENCING_ORDER: u8 = 1;
pub(super) const MAX_SEASONAL_MA_ORDER: u8 = 2;

/// The `(p d q)` orders of one ARIMA factor: autoregressive, differencing
/// and moving-average.
///
/// [`ArimaOrder::new`] enforces the regular-factor limits; [`ArimaModel::new`]
/// applies the tighter seasonal limits to its seasonal factor. Displays as
/// `(p d q)`, such as `(0 1 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArimaOrder {
    p: u8,
    d: u8,
    q: u8,
}

/// Displays as `(p d q)`, such as `(0 1 1)`.
impl std::fmt::Display for ArimaOrder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "({} {} {})", self.p, self.d, self.q)
    }
}

impl ArimaOrder {
    /// Constructs an order with `p` and `q` at most 3 and `d` at most 2,
    /// otherwise [`Error::InvalidSpec`]. These are the regular-factor
    /// limits, and the error names the regular factor even for an order meant
    /// for the seasonal one; [`ArimaModel::new`] applies the seasonal limits.
    pub fn new(p: u8, d: u8, q: u8) -> Result<Self, Error> {
        validate_order(
            ArimaFactor::Regular,
            p,
            d,
            q,
            MAX_REGULAR_AR_ORDER,
            MAX_REGULAR_DIFFERENCING_ORDER,
            MAX_REGULAR_MA_ORDER,
        )?;
        Ok(Self { p, d, q })
    }

    /// Autoregressive order.
    #[must_use]
    pub const fn p(self) -> u8 {
        self.p
    }

    /// Differencing order.
    #[must_use]
    pub const fn d(self) -> u8 {
        self.d
    }

    /// Moving-average order.
    #[must_use]
    pub const fn q(self) -> u8 {
        self.q
    }
}

/// A seasonal ARIMA model `(p d q)(P D Q)`: a regular factor and a seasonal
/// factor whose lags are whole years (12 months or 4 quarters).
///
/// The airline model `(0 1 1)(0 1 1)`, the X-13ARIMA-SEATS default, suits many
/// monthly and quarterly economic series; [`ArimaModel::airline`] builds it,
/// and [`RegArimaSpec::new`] takes it. SEATS accepts a narrower set of models
/// than X-11 (see [`Spec::seats`]). Displays as
/// `(p d q)(P D Q)`.
///
/// ```
/// use x13::{ArimaModel, ArimaOrder, Error};
///
/// let airline = ArimaModel::airline();
/// assert_eq!(airline.to_string(), "(0 1 1)(0 1 1)");
///
/// let other = ArimaModel::new(ArimaOrder::new(2, 1, 0)?, ArimaOrder::new(0, 1, 1)?)?;
/// assert_eq!(other.to_string(), "(2 1 0)(0 1 1)");
///
/// // Seasonal orders are limited to (2 1 2).
/// let too_large = ArimaModel::new(ArimaOrder::new(0, 1, 1)?, ArimaOrder::new(3, 1, 1)?);
/// assert!(matches!(too_large, Err(Error::InvalidSpec(_))));
/// # Ok::<(), Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArimaModel {
    regular: ArimaOrder,
    seasonal: ArimaOrder,
}

/// Displays as `(p d q)(P D Q)`, such as `(0 1 1)(0 1 1)`.
impl std::fmt::Display for ArimaModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}{}", self.regular, self.seasonal)
    }
}

impl ArimaModel {
    /// The airline model `(0 1 1)(0 1 1)`, the X-13ARIMA-SEATS default.
    ///
    /// ```
    /// use x13::regarima::ModelSelection;
    /// use x13::{ArimaModel, RegArimaSpec};
    ///
    /// let spec = RegArimaSpec::new(ArimaModel::airline());
    /// assert_eq!(spec.model(), ModelSelection::Explicit(ArimaModel::airline()));
    /// ```
    #[must_use]
    pub const fn airline() -> Self {
        const UNIT_MA: ArimaOrder = ArimaOrder { p: 0, d: 1, q: 1 };
        Self {
            regular: UNIT_MA,
            seasonal: UNIT_MA,
        }
    }

    /// Constructs the model `regular` × `seasonal`. The seasonal orders must
    /// satisfy `P ≤ 2`, `D ≤ 1` and `Q ≤ 2`, otherwise
    /// [`Error::InvalidSpec`].
    pub fn new(regular: ArimaOrder, seasonal: ArimaOrder) -> Result<Self, Error> {
        validate_order(
            ArimaFactor::Seasonal,
            seasonal.p,
            seasonal.d,
            seasonal.q,
            MAX_SEASONAL_AR_ORDER,
            MAX_SEASONAL_DIFFERENCING_ORDER,
            MAX_SEASONAL_MA_ORDER,
        )?;
        Ok(Self { regular, seasonal })
    }

    /// Nonseasonal factor.
    #[must_use]
    pub const fn regular(self) -> ArimaOrder {
        self.regular
    }

    /// Seasonal factor.
    #[must_use]
    pub const fn seasonal(self) -> ArimaOrder {
        self.seasonal
    }
}

/// Transform applied to the series before the regARIMA model is fitted.
///
/// Components, forecasts and backcasts are reported on the original scale.
/// Only [`Forecasts::transformed`](crate::Forecasts::transformed),
/// [`Backcasts::transformed`](crate::Backcasts::transformed) and
/// their standard errors are on the transformed scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Transform {
    /// Use observations without transformation.
    None,
    /// Apply the natural logarithm before modeling.
    Log,
}

display_phrases!(Transform {
    None => "none",
    Log => "log",
});

/// The regARIMA model that preprocesses a series before X-11 or SEATS.
///
/// A regression with ARIMA errors: the regression part holds calendar
/// effects ([`CalendarSpec`]), outliers (found by an automatic search,
/// [`OutlierSpec`], or placed at known dates, [`ManualOutlier`]), user
/// regressors and an optional constant; an ARIMA model describes the rest.
/// The fitted model estimates NaN gaps and extends the series with
/// forecasts, and optionally backcasts.
///
/// Start from [`new`](RegArimaSpec::new) with a fixed model or from
/// [`automatic_model`](RegArimaSpec::automatic_model) to let the
/// automatic search choose one, then chain setters. Both start with no
/// transform, no calendar effects, outliers or user regressors, one year of
/// forecasts and no backcasts; `new` has no constant term, and
/// `automatic_model` lets the search choose it. Pass the result to
/// [`Spec::x11_regarima`] or [`Spec::seats`].
///
/// Setters do not validate combinations; [`adjust`](crate::adjust) does.
/// Manual outliers or [`CalendarSelection::AicTest`] combined with an
/// automatic model, an automatic transform or an outlier search, and a
/// constant setting on an automatic model, are
/// [`Error::Unsupported`].
///
/// ```
/// use x13::regarima::{
///     CalendarBasis, CalendarSpec, Easter, FlowTradingDay, ForecastHorizon, ForecastSpec, OutlierSpec,
///     OutlierTypes, TradingDayContrast, TransformSelection,
/// };
/// use x13::{ArimaModel, RegArimaSpec, Transform};
///
/// let trading_day = FlowTradingDay::new(TradingDayContrast::SixCoefficient);
/// let calendar = CalendarSpec::new()
///     .with_basis(CalendarBasis::FlowTradingDay(trading_day))
///     .with_easter(Easter::new(8)?);
/// let regarima = RegArimaSpec::new(ArimaModel::airline())
///     .with_transform(Transform::Log)
///     .with_calendar(calendar)
///     .with_outliers(OutlierSpec::new().with_types(OutlierTypes::ALL))
///     .with_forecast(ForecastSpec::new(ForecastHorizon::Periods(24), 0)?);
///
/// assert_eq!(regarima.transform(), TransformSelection::Explicit(Transform::Log));
/// assert_eq!(regarima.forecast().horizon(), ForecastHorizon::Periods(24));
/// # Ok::<(), x13::Error>(())
/// ```
///
/// `RegArimaSpec` is `PartialEq` but not `Eq`, because the outlier settings
/// hold floating-point values.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct RegArimaSpec {
    model: ModelSelection,
    transform: crate::regarima::TransformSelection,
    constant: ConstantSelection,
    extensions: ForecastSpec,
    calendar: Option<CalendarSpec>,
    outliers: Option<OutlierSpec>,
    manual_outliers: Arc<[ManualOutlier]>,
    temporary_change_rate: Option<TemporaryChangeRate>,
    user_regressors: Arc<[crate::regarima::UserRegressor]>,
    model_span: Option<super::PeriodSpan>,
}

impl RegArimaSpec {
    /// A fixed ARIMA `model` with no transform and no constant term, one
    /// year of forecasts, no backcasts, and no calendar effects, outliers or
    /// user regressors.
    pub fn new(model: ArimaModel) -> Self {
        Self {
            model: ModelSelection::Explicit(model),
            transform: crate::regarima::TransformSelection::Explicit(Transform::None),
            constant: ConstantSelection::Excluded,
            extensions: ForecastSpec::default(),
            calendar: None,
            outliers: None,
            manual_outliers: Arc::from([]),
            temporary_change_rate: None,
            user_regressors: Arc::from([]),
            model_span: None,
        }
    }

    /// The automatic model search: it chooses the ARIMA orders and
    /// whether to include a constant term
    /// ([`ModelSelection::Automatic`](crate::regarima::ModelSelection::Automatic)).
    ///
    /// Other settings start as in [`new`](RegArimaSpec::new); an automatic
    /// transform is a separate choice
    /// ([`with_transform`](RegArimaSpec::with_transform)).
    /// Leave the constant unset: the search decides it, and
    /// [`with_constant`](RegArimaSpec::with_constant) on an automatic model is
    /// [`Error::Unsupported`]. The chosen model is reported by
    /// [`Adjustment::model`](crate::Adjustment::model) and the search by
    /// [`Adjustment::model_selection`](crate::Adjustment::model_selection).
    pub fn automatic_model() -> Self {
        Self {
            model: ModelSelection::Automatic,
            transform: crate::regarima::TransformSelection::Explicit(Transform::None),
            constant: ConstantSelection::Automatic,
            extensions: ForecastSpec::default(),
            calendar: None,
            outliers: None,
            manual_outliers: Arc::from([]),
            temporary_change_rate: None,
            user_regressors: Arc::from([]),
            model_span: None,
        }
    }

    /// Estimates the model on these inclusive dates only; adjustment still
    /// covers the whole input and forecasts still begin after its last
    /// observation. A span that starts after the first observation cancels
    /// backcasts, as in X-13ARIMA-SEATS
    /// ([`Warning::ModelSpanBackcastsIgnored`](crate::Warning::ModelSpanBackcastsIgnored)).
    pub const fn with_model_span(mut self, span: super::PeriodSpan) -> Self {
        self.model_span = Some(span);
        self
    }

    /// Requested estimation span, or the whole series by default.
    #[must_use]
    pub const fn model_span(&self) -> Option<super::PeriodSpan> {
        self.model_span
    }

    /// Sets the transform: a fixed [`Transform`], none (the default) or
    /// natural logarithms, or the automatic choice between the two by AICC.
    /// [`TransformSelection::automatic`](crate::regarima::TransformSelection::automatic)
    /// is the automatic choice with the default AICC difference, −2; pass an
    /// [`AutoTransformOptions`](crate::regarima::AutoTransformOptions) to change the
    /// difference.
    ///
    /// Logs need positive observations ([`Error::NonPositiveValue`]). Under
    /// X-11, logs pair with a multiplicative or log-additive mode.
    ///
    /// The automatic choice is reported by
    /// [`Adjustment::transform_selection`](crate::Adjustment::transform_selection).
    /// With a zero or negative observation it picks no transform without
    /// comparing, and warns
    /// ([`Warning::AutomaticTransformNonPositiveInput`](crate::Warning::AutomaticTransformNonPositiveInput)).
    ///
    /// ```
    /// use x13::regarima::{AutoTransformOptions, TransformSelection};
    /// use x13::{ArimaModel, RegArimaSpec, Transform};
    ///
    /// let logs = RegArimaSpec::new(ArimaModel::airline()).with_transform(Transform::Log);
    /// assert_eq!(logs.transform(), TransformSelection::Explicit(Transform::Log));
    ///
    /// let automatic =
    ///     RegArimaSpec::new(ArimaModel::airline()).with_transform(TransformSelection::automatic());
    /// assert_eq!(
    ///     automatic.transform(),
    ///     TransformSelection::Automatic(AutoTransformOptions::new())
    /// );
    /// ```
    pub fn with_transform(
        mut self,
        transform: impl Into<crate::regarima::TransformSelection>,
    ) -> Self {
        self.transform = transform.into();
        self
    }

    /// Includes or excludes a constant term (excluded by default).
    ///
    /// Only for a fixed model: [`automatic_model`](RegArimaSpec::automatic_model)
    /// decides the constant itself, and setting it there is
    /// [`Error::Unsupported`].
    pub const fn with_constant(mut self, include_constant: bool) -> Self {
        self.constant = if include_constant {
            ConstantSelection::Included
        } else {
            ConstantSelection::Excluded
        };
        self
    }

    /// Sets the forecast and backcast horizons (by default one year of
    /// forecasts and no backcasts).
    ///
    /// Forecasts extend the series X-11 adjusts and are returned by
    /// [`Adjustment::forecasts`](crate::Adjustment::forecasts); backcasts
    /// likewise by [`Adjustment::backcasts`](crate::Adjustment::backcasts).
    /// SEATS takes at most three years of forecasts and ignores backcasts.
    pub const fn with_forecast(mut self, forecast: ForecastSpec) -> Self {
        self.extensions = forecast;
        self
    }

    /// Adds the trading-day, length-of-period and holiday regressors in
    /// `calendar`, replacing any set before.
    ///
    /// Their combined estimates are reported by
    /// [`Adjustment::calendar_effects`](crate::Adjustment::calendar_effects)
    /// and the coefficients by
    /// [`ModelResult::regression_coefficients`](crate::ModelResult::regression_coefficients).
    pub const fn with_calendar(mut self, calendar: CalendarSpec) -> Self {
        self.calendar = Some(calendar);
        self
    }

    /// Turns on the automatic outlier search with these settings, replacing
    /// any set before; [`OutlierSpec::new`] is the X-13ARIMA-SEATS default
    /// search.
    ///
    /// Outliers it finds are reported by
    /// [`Adjustment::outliers`](crate::Adjustment::outliers).
    pub const fn with_outliers(mut self, outliers: OutlierSpec) -> Self {
        self.outliers = Some(outliers);
        self
    }

    /// Fits outliers at known dates, in the order given, replacing any list
    /// set before. The spec keeps its own copy.
    ///
    /// Each date must have the series' frequency
    /// ([`Error::PeriodFrequencyMismatch`]) and fall within the series; a
    /// level shift cannot fall on the first or last observation, and a
    /// repeated outlier is rejected ([`Error::InvalidManualOutlier`]). An outlier the
    /// model span cannot fit is dropped with a warning, as in X-13ARIMA-SEATS
    /// ([`Warning::ManualOutlierOutsideModelSpan`](crate::Warning::ManualOutlierOutsideModelSpan)).
    /// Manual outliers need a fixed model and transform and no automatic
    /// outlier search ([`Error::Unsupported`]).
    pub fn with_manual_outliers(mut self, outliers: &[ManualOutlier]) -> Self {
        self.manual_outliers = Arc::from(outliers);
        self
    }

    /// Outliers set by [`with_manual_outliers`](RegArimaSpec::with_manual_outliers),
    /// in the order given.
    #[must_use]
    pub fn manual_outliers(&self) -> &[ManualOutlier] {
        &self.manual_outliers
    }

    /// Sets the temporary-change decay rate, `tcrate` in either the regression
    /// or the outlier spec of X-13ARIMA-SEATS. There is one rate for every
    /// TC column, so this rate drives specified TCs and automatic TC
    /// candidates alike, with or without an [`OutlierSpec`].
    pub const fn with_temporary_change_rate(mut self, rate: TemporaryChangeRate) -> Self {
        self.temporary_change_rate = Some(rate);
        self
    }

    /// Explicit temporary-change rate, or `None` for the
    /// frequency-dependent default, `0.7^(12/s)` for `s` periods per year.
    #[must_use]
    pub const fn temporary_change_rate(&self) -> Option<TemporaryChangeRate> {
        self.temporary_change_rate
    }

    /// Adds user-supplied regression columns, in the order given,
    /// replacing any set before. The spec keeps its own copy.
    ///
    /// Each column must cover the whole series and every forecast and
    /// backcast date the pipeline uses; SEATS forecasts three years
    /// internally, whatever its forecast setting. Columns are never
    /// extrapolated.
    pub fn with_user_regressors(mut self, regressors: &[crate::regarima::UserRegressor]) -> Self {
        self.user_regressors = Arc::from(regressors);
        self
    }

    /// Columns set by [`with_user_regressors`](RegArimaSpec::with_user_regressors);
    /// their positions identify the fitted coefficients.
    #[must_use]
    pub fn user_regressors(&self) -> &[crate::regarima::UserRegressor] {
        &self.user_regressors
    }

    /// Requested model: fixed or automatic. The fitted orders are in
    /// [`ModelResult::model`](crate::ModelResult::model).
    #[must_use]
    pub const fn model(&self) -> ModelSelection {
        self.model
    }

    /// Requested transform: fixed or automatic. The transform used is in
    /// [`ModelResult::transform`](crate::ModelResult::transform).
    #[must_use]
    pub const fn transform(&self) -> crate::regarima::TransformSelection {
        self.transform
    }

    /// Requested constant policy. Whether the fitted model has one is in
    /// [`ModelResult::includes_constant`](crate::ModelResult::includes_constant).
    #[must_use]
    pub const fn constant(&self) -> ConstantSelection {
        self.constant
    }

    /// Requested forecast and backcast horizons.
    #[must_use]
    pub const fn forecast(&self) -> ForecastSpec {
        self.extensions
    }

    /// Calendar regressors, if set.
    #[must_use]
    pub const fn calendar(&self) -> Option<CalendarSpec> {
        self.calendar
    }

    /// Automatic outlier search settings, if the search is on.
    #[must_use]
    pub const fn outliers(&self) -> Option<OutlierSpec> {
        self.outliers
    }

    pub(super) fn validate(&self, values: &[f64], axis: TimeAxis) -> Result<(), Error> {
        self.validate_model_policy()?;
        let model_range = self.model_range(axis, values.len())?;
        if let Some(outliers) = self.outliers {
            outliers.resolved_span(axis.shift(model_range.start as i64)?, model_range.len())?;
        }
        if self.transform == crate::regarima::TransformSelection::Explicit(Transform::Log)
            && let Some(index) = crate::numerics::first_nonpositive(values)
        {
            return Err(Error::NonPositiveValue { index });
        }
        // Calendar years and supplied-column coverage follow the backcasts
        // Build 62 keeps after editor.f's model-span cancellation.
        let extensions = self.extensions.for_model_start(model_range.start);
        if let Some(calendar) = self.calendar {
            calendar.validate(axis, values.len(), extensions)?;
        }
        self.validate_user_regressors(axis, values.len(), extensions)?;
        // adpdrg validates supplied outliers before editor's constant bypass.
        // The TC rate needs no check here: `TemporaryChangeRate` is valid by
        // construction and the gtinpt.f default is valid for both frequencies.
        ManualOutlier::validate_list(&self.manual_outliers, axis, values.len())?;
        Ok(())
    }

    /// `extensions` are the effective horizons: after the model-span backcast
    /// cancellation ([`ForecastSpec::for_model_start`]), and with the internal
    /// horizon for methods that raise it (SEATS).
    pub(crate) fn validate_user_regressors(
        &self,
        axis: TimeAxis,
        count: usize,
        extensions: ForecastSpec,
    ) -> Result<(), Error> {
        if self.user_regressors.is_empty() {
            return Ok(());
        }
        let (required_axis, required_count) = extensions.extended_span(axis, count)?;
        for regressor in self.user_regressors.iter() {
            regressor.values_over(required_axis, required_count)?;
        }
        Ok(())
    }

    pub(crate) fn model_range(
        &self,
        axis: TimeAxis,
        count: usize,
    ) -> Result<std::ops::Range<usize>, Error> {
        self.model_span
            .map_or(Ok(0..count), |span| span.resolve_model_span(axis, count))
    }

    /// The spec the automatic-model path's transform test fits: `model`
    /// replaces the automatic selection and every other policy carries over,
    /// except where noted. The destructure is exhaustive so that a new field
    /// fails to compile here until its automatic-model treatment is decided.
    pub(crate) fn transform_test_spec(&self, model: ArimaModel) -> Self {
        let Self {
            model: _,
            transform,
            // An automatic model's constant is `Automatic`
            // (`validate_model_policy` rejects the others); the test fits
            // without one.
            constant: _,
            extensions,
            calendar,
            outliers,
            // `validate_model_policy` rejects manual outliers with an
            // automatic model.
            manual_outliers: _,
            temporary_change_rate,
            user_regressors,
            // The caller's values are already narrowed to the model span.
            model_span: _,
        } = self;
        Self {
            model: ModelSelection::Explicit(model),
            transform: *transform,
            constant: ConstantSelection::Excluded,
            extensions: *extensions,
            calendar: *calendar,
            outliers: *outliers,
            manual_outliers: Arc::from([]),
            temporary_change_rate: *temporary_change_rate,
            user_regressors: Arc::clone(user_regressors),
            model_span: None,
        }
    }

    pub(crate) fn validate_model_policy(&self) -> Result<(), Error> {
        if !self.manual_outliers.is_empty()
            && (self.model == ModelSelection::Automatic
                || matches!(
                    self.transform,
                    crate::regarima::TransformSelection::Automatic(_)
                )
                || self.outliers.is_some())
        {
            return Err(Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::ManualOutliers,
            )));
        }
        if let Some(calendar) = self.calendar
            && calendar.selection() == CalendarSelection::AicTest
            && (self.model == ModelSelection::Automatic
                || matches!(
                    self.transform,
                    crate::regarima::TransformSelection::Automatic(_)
                )
                || self.outliers.is_some()
                || (calendar.basis().is_none() && calendar.easter().is_none())
                || matches!(calendar.basis(), Some(CalendarBasis::Length(_)))
                || matches!(calendar.basis(), Some(CalendarBasis::FlowTradingDay(flow))
                    if flow.length_treatment() == FlowLengthTreatment::LengthOfPeriod))
        {
            return Err(Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::CalendarSelection,
            )));
        }
        // Only `automatic_model` leaves the constant automatic, and
        // `with_constant` always sets it.
        debug_assert!(
            self.model == ModelSelection::Automatic
                || self.constant != ConstantSelection::Automatic
        );
        if self.model == ModelSelection::Automatic && self.constant != ConstantSelection::Automatic
        {
            return Err(Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::AutomaticModelConstant,
            )));
        }
        Ok(())
    }

    pub(crate) fn resolve(&self, transform: Transform) -> Result<ResolvedRegArimaSpec, Error> {
        self.validate_model_policy()?;
        let ModelSelection::Explicit(model) = self.model else {
            return Err(Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::AutomaticModel,
            )));
        };
        Ok(ResolvedRegArimaSpec {
            flow_length_context: FlowLengthContext::LeapYear,
            model: crate::regarima::ResolvedArimaModel::from_explicit(model),
            transform,
            include_constant: self.constant == ConstantSelection::Included,
            extensions: self.extensions,
            calendar: self.calendar,
            outliers: self.outliers,
            temporary_change_rate: self.temporary_change_rate,
        })
    }

    pub(crate) fn resolve_explicit(&self) -> Result<ResolvedRegArimaSpec, Error> {
        self.validate_model_policy()?;
        if self.model == ModelSelection::Automatic {
            return Err(Error::from(UnsupportedDetail::Configuration(
                UnsupportedConfiguration::AutomaticModel,
            )));
        }
        match self.transform {
            crate::regarima::TransformSelection::Explicit(transform) => self.resolve(transform),
            crate::regarima::TransformSelection::Automatic(_) => Err(Error::from(
                UnsupportedDetail::Configuration(UnsupportedConfiguration::AutomaticTransform),
            )),
        }
    }
}

/// The length regressor that a flow trading-day effect with the automatic
/// length treatment uses in one fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlowLengthContext {
    LeapYear,
    LengthOfPeriod,
}

/// Internal fit input: numerical routines cannot receive an unresolved policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedRegArimaSpec {
    flow_length_context: FlowLengthContext,
    model: crate::regarima::ResolvedArimaModel,
    transform: Transform,
    include_constant: bool,
    extensions: ForecastSpec,
    calendar: Option<CalendarSpec>,
    outliers: Option<OutlierSpec>,
    temporary_change_rate: Option<TemporaryChangeRate>,
}

impl ResolvedRegArimaSpec {
    pub(crate) const fn with_calendar(mut self, calendar: Option<CalendarSpec>) -> Self {
        self.calendar = calendar;
        self
    }

    pub(crate) const fn with_extensions(mut self, extensions: ForecastSpec) -> Self {
        self.extensions = extensions;
        self
    }

    pub(crate) const fn with_flow_length_context(mut self, context: FlowLengthContext) -> Self {
        self.flow_length_context = context;
        self
    }

    pub(crate) const fn flow_length_context(self) -> FlowLengthContext {
        self.flow_length_context
    }
    pub(crate) fn with_model(
        mut self,
        model: impl Into<crate::regarima::ResolvedArimaModel>,
    ) -> Self {
        self.model = model.into();
        self
    }

    pub(crate) const fn with_constant(mut self, include: bool) -> Self {
        self.include_constant = include;
        self
    }
    pub(crate) const fn transform(self) -> Transform {
        self.transform
    }
    pub(crate) const fn model(self) -> crate::regarima::ResolvedArimaModel {
        self.model
    }
    pub(crate) const fn includes_constant(self) -> bool {
        self.include_constant
    }
    pub(crate) const fn extensions(self) -> ForecastSpec {
        self.extensions
    }
    pub(crate) const fn calendar(self) -> Option<CalendarSpec> {
        self.calendar
    }
    pub(crate) const fn outliers(self) -> Option<OutlierSpec> {
        self.outliers
    }
    /// The rate of every TC column, specified or automatic. Build 62 keeps one
    /// global `Tcalfa` (model.cmn), which `tcrate` sets in either the
    /// regression or the outlier spec (getreg.f, gtotlr.f); gtinpt.f turns an
    /// unset one into `0.7**(12/Sp)` once every spec is read.
    pub(crate) fn resolved_temporary_change_rate(self, frequency: Frequency) -> f64 {
        self.temporary_change_rate.map_or_else(
            || 0.7_f64.powf(12.0 / f64::from(frequency.periods_per_year())),
            TemporaryChangeRate::value,
        )
    }
}

/// Largest calendar year accepted for any calendar effect. X-13ARIMA-SEATS
/// prints four-digit years, and the trading-day arithmetic overflows `i32` far
/// above this, so the bound is checked once here.
const MAXIMUM_CALENDAR_YEAR: i32 = 9999;

/// Smallest year with trading-day regressors: `adpdrg.f` stops with "No
/// trading variables before 1776".
pub(crate) const MINIMUM_TRADING_DAY_YEAR: i32 = 1776;

pub(super) fn validate_year_range(
    feature: CalendarFeature,
    first_year: Option<i32>,
    last_year: Option<i32>,
    minimum: i32,
    maximum: Option<i32>,
) -> Result<(), Error> {
    let maximum = maximum.map_or(MAXIMUM_CALENDAR_YEAR, |maximum| {
        maximum.min(MAXIMUM_CALENDAR_YEAR)
    });
    for year in [first_year, last_year].into_iter().flatten() {
        if year < minimum || year > maximum {
            return Err(Error::CalendarYearOutOfRange {
                feature,
                year,
                minimum,
                maximum: Some(maximum),
            });
        }
    }
    Ok(())
}

/// The seasonal-adjustment method a [`Spec`] runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    /// X-11.
    X11,
    /// SEATS.
    Seats,
}

/// What [`adjust`](crate::adjust) does with a series: the pipeline and its
/// options.
///
/// A constructor picks the pipeline:
///
/// | Constructor | Pipeline |
/// |---|---|
/// | [`x11`](Spec::x11) | X-11 on the series as given: no model, forecasts, calendar regression or gap filling |
/// | [`x11_regarima`](Spec::x11_regarima) | a regARIMA model ([`RegArimaSpec`]) estimates calendar effects, outliers and gaps and extends the series with forecasts; X-11 then adjusts the result |
/// | [`seats`](Spec::seats) | the same regARIMA preprocessing, then a SEATS decomposition of the fitted model |
///
/// The X-11 constructors take every X-11 setting, the decomposition mode
/// included, as [`x11::Options`](crate::x11::Options), and
/// [`x11::Options::new`](crate::x11::Options::new) gives the X-13ARIMA-SEATS
/// defaults. SEATS has no settings beyond its regARIMA model.
///
/// ```
/// use x13::{RegArimaSpec, Spec, x11};
///
/// let options = x11::Options::new()
///     .with_mode(x11::Mode::Additive)
///     .with_seasonal_filter(x11::SeasonalFilter::S3x5);
/// let spec = Spec::x11(options);
/// assert_eq!(spec.x11_options(), Some(options));
/// assert!(spec.regarima().is_none());
///
/// let spec = Spec::seats(RegArimaSpec::automatic_model());
/// assert_eq!(spec.x11_options(), None);
/// assert!(spec.regarima().is_some());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Spec {
    kind: SpecKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum SpecKind {
    X11 {
        options: x11::Options,
    },
    X11RegArima {
        regarima: RegArimaSpec,
        options: x11::Options,
    },
    Seats {
        regarima: RegArimaSpec,
    },
}

impl Spec {
    /// X-11 on the series as given, with no regARIMA model.
    ///
    /// There are no forecasts, fitted model, calendar regression or
    /// outliers, and a NaN gap is [`Error::MissingValue`]. The series needs
    /// at least three years of observations and at most 85 calendar years.
    ///
    /// With no [mode](crate::x11::Options::with_mode) set, X-11 is
    /// multiplicative, the default of X-13ARIMA-SEATS, and needs positive
    /// values ([`Error::NonPositiveValue`]). Additive X-11 accepts any finite
    /// values, including zero and negative ones; pseudo-additive accepts
    /// zeros but not negative values; log-additive needs positive values.
    ///
    /// The [final-outlier](crate::x11::Options::with_final_outliers) and
    /// [temporary-change](crate::x11::Options::with_temporary_change_allocation)
    /// settings act on the outliers of a regARIMA model, so they are ignored
    /// here.
    ///
    /// ```
    /// use x13::{Period, Series, Spec, adjust, x11};
    ///
    /// // Ten years of quarterly data with a fixed seasonal pattern.
    /// let pattern = [0.9, 1.05, 1.1, 0.95];
    /// let values: Vec<f64> = (0..40)
    ///     .map(|i| {
    ///         let noise = 1.0 + (((i * 37) % 11) as f64 / 11.0 - 0.5) / 100.0;
    ///         (200.0 + 2.0 * i as f64) * pattern[i % 4] * noise
    ///     })
    ///     .collect();
    /// let series = Series::new(Period::quarterly(2010, 1)?, &values);
    /// let adjustment = adjust(series, &Spec::x11(x11::Options::new()))?;
    ///
    /// // Seasonal factors are ratios near one, and the adjusted series is
    /// // the original divided by them.
    /// let seasonal = adjustment.seasonal().expect("X-11 returns every component");
    /// assert!((seasonal.values()[2] - 1.1).abs() < 0.02);
    /// for ((original, factor), adjusted) in values
    ///     .iter()
    ///     .zip(seasonal.values())
    ///     .zip(adjustment.seasonally_adjusted().values())
    /// {
    ///     assert!((original / factor - adjusted).abs() < 1e-9 * original);
    /// }
    /// assert!(adjustment.model().is_none() && adjustment.forecasts().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Another mode:
    ///
    /// ```
    /// use x13::{Period, Series, Spec, adjust, x11};
    ///
    /// // A net flow that changes sign: only additive X-11 can adjust it.
    /// let values: Vec<f64> = (0..60)
    ///     .map(|i| {
    ///         let season = 12.0 * (i as f64 * std::f64::consts::PI / 6.0).cos();
    ///         let noise = ((i * 37) % 11) as f64 / 11.0 - 0.5;
    ///         0.1 * i as f64 - 3.0 + season + noise
    ///     })
    ///     .collect();
    /// let spec = Spec::x11(x11::Options::new().with_mode(x11::Mode::Additive));
    /// let flow = Series::new(Period::monthly(2019, 1)?, &values);
    /// let adjustment = adjust(flow, &spec)?;
    ///
    /// // Additive seasonal effects are in series units and sum with the
    /// // adjusted series to the original.
    /// let seasonal = adjustment.seasonal().expect("X-11 returns every component");
    /// for ((original, effect), adjusted) in values
    ///     .iter()
    ///     .zip(seasonal.values())
    ///     .zip(adjustment.seasonally_adjusted().values())
    /// {
    ///     assert!((effect + adjusted - original).abs() < 1e-9);
    /// }
    /// assert!(adjust(flow, &Spec::x11(x11::Options::new())).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn x11(options: x11::Options) -> Self {
        Self {
            kind: SpecKind::X11 { options },
        }
    }

    /// X-11 after regARIMA preprocessing.
    ///
    /// The regARIMA model ([`RegArimaSpec`]) estimates the calendar effects,
    /// outliers and user regressors it is given, estimates NaN gaps, and
    /// extends the series with forecasts (one year by default). X-11 then
    /// adjusts the preadjusted, extended series, which steadies the factors at
    /// the end of the series. This is the usual X-11 setup in X-13ARIMA-SEATS.
    ///
    /// With no [mode](crate::x11::Options::with_mode) set, the X-11 mode follows
    /// the model's transform: additive with no transform, multiplicative with
    /// logs. The mode used is in
    /// [`x11::Diagnostics::mode`](crate::x11::Diagnostics::mode). An automatic
    /// transform
    /// ([`TransformSelection::automatic`](crate::regarima::TransformSelection::automatic))
    /// is not known until the series is fitted, so it cannot be combined with
    /// a mode.
    ///
    /// When the fitted model has regression effects (calendar effects, user
    /// regressors, gaps, or outliers, whether specified or found by the
    /// search), a mode that is set must agree with the transform: additive
    /// with no transform, any other mode with logs. Pseudo-additive X-11
    /// takes no regression effects other than gaps, and gaps exclude a
    /// constant. A request that breaks one of these rules returns
    /// [`Error::InvalidSpec`]. The series needs at least three years of
    /// observations, and at most 85 calendar years including forecasts and
    /// backcasts.
    ///
    /// ```
    /// use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, adjust, x11};
    ///
    /// let regarima = RegArimaSpec::new(ArimaModel::airline()).with_transform(Transform::Log);
    /// let options = x11::Options::new().with_mode(x11::Mode::Multiplicative);
    /// let spec = Spec::x11_regarima(regarima, options);
    ///
    /// let mut values: Vec<f64> = (0..96)
    ///     .map(|i| {
    ///         let season = 1.0 + 0.1 * (i as f64 * std::f64::consts::PI / 6.0).sin();
    ///         let noise = 1.0 + (((i * 37) % 11) as f64 / 11.0 - 0.5) / 50.0;
    ///         (100.0 + 0.8 * i as f64) * season * noise
    ///     })
    ///     .collect();
    /// values[40] = f64::NAN; // a gap for the model to estimate
    /// let series = Series::new(Period::monthly(2012, 1)?, &values);
    /// let adjustment = adjust(series, &spec)?;
    ///
    /// assert!(adjustment.seasonally_adjusted().values()[40].is_finite());
    /// let model = adjustment.model().expect("a fitted model");
    /// assert_eq!(model.model().to_string(), "(0 1 1)(0 1 1)");
    /// let forecasts = adjustment.forecasts().expect("one year by default");
    /// assert_eq!(forecasts.start(), Period::monthly(2020, 1)?);
    /// assert_eq!(forecasts.series().len(), 12);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// An automatic transform, with the mode that matches the choice:
    ///
    /// ```
    /// use x13::regarima::TransformSelection;
    /// use x13::{ArimaModel, Period, RegArimaSpec, Series, Spec, Transform, adjust, x11};
    ///
    /// let regarima =
    ///     RegArimaSpec::new(ArimaModel::airline()).with_transform(TransformSelection::automatic());
    /// let spec = Spec::x11_regarima(regarima, x11::Options::new());
    ///
    /// // Growth with a seasonal swing proportional to the level favors logs.
    /// let values: Vec<f64> = (0..96)
    ///     .map(|i| {
    ///         let season = 1.0 + 0.2 * (i as f64 * std::f64::consts::PI / 6.0).sin();
    ///         let noise = 1.0 + (((i * 37) % 11) as f64 / 11.0 - 0.5) / 50.0;
    ///         100.0 * (0.015 * i as f64).exp() * season * noise
    ///     })
    ///     .collect();
    /// let series = Series::new(Period::monthly(2012, 1)?, &values);
    /// let adjustment = adjust(series, &spec)?;
    ///
    /// assert_eq!(adjustment.model().expect("a fitted model").transform(), Transform::Log);
    /// let x11 = adjustment.diagnostics().x11().expect("an X-11 run");
    /// assert_eq!(x11.mode(), x11::Mode::Multiplicative);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub const fn x11_regarima(regarima: RegArimaSpec, options: x11::Options) -> Self {
        Self {
            kind: SpecKind::X11RegArima { regarima, options },
        }
    }

    /// SEATS: regARIMA preprocessing, then a model-based decomposition of the
    /// fitted model into trend-cycle, seasonal, transitory and irregular
    /// components.
    ///
    /// The model is explicit ([`RegArimaSpec::new`]) or chosen by the
    /// automatic search ([`RegArimaSpec::automatic_model`]). SEATS runs the
    /// common seasonal models, the airline model `(0 1 1)(0 1 1)` among them:
    /// regular orders up to `(3 2 2)` and seasonal orders up to `(1 1 1)`,
    /// within limits on their combined size. Another model, explicit or
    /// selected by the search, is an error of kind
    /// [`ErrorKind::Unsupported`]. A constant series has no fitted model to
    /// decompose ([`Error::DegenerateSeries`]).
    ///
    /// [`adjust`](crate::adjust) also checks:
    ///
    /// - **Length.** 36 to 780 monthly observations, or 16 to 328 quarterly
    ///   ones (one fewer for each quarter the series starts after the
    ///   first); otherwise [`Error::TooShort`] or [`Error::SeatsTooLong`].
    /// - **Start.** The first year is not negative
    ///   ([`Error::SeatsNegativeYear`]).
    /// - **Forecasts.** At most three years of forecasts (one by default;
    ///   zero returns none). Requested backcasts are ignored with
    ///   [`Warning::SeatsBackcastsIgnored`](crate::Warning::SeatsBackcastsIgnored),
    ///   as in X-13ARIMA-SEATS.
    /// - **Calendar.** Every [`CalendarSpec`] effect is available, but a
    ///   calendar spec with no effect is reported as unsupported.
    ///
    /// Forecasts, calendar effects and outliers in the [`Adjustment`] come
    /// from the regARIMA fit. SEATS may revise its own copy of the model for
    /// the decomposition, reported in
    /// [`seats::Diagnostics`](crate::seats::Diagnostics); a revision this crate
    /// does not follow is an error of kind [`ErrorKind::Unsupported`]. The
    /// combined adjustment factors (table S16) are not returned. A valid
    /// request can still fail to fit or decompose, usually as
    /// [`ErrorKind::Numerical`].
    ///
    /// [`Adjustment`]: crate::Adjustment
    /// [`ErrorKind::Numerical`]: crate::ErrorKind::Numerical
    /// [`ErrorKind::Unsupported`]: crate::ErrorKind::Unsupported
    ///
    /// An explicit airline model:
    ///
    /// ```
    /// use x13::{Adjustment, ArimaModel, Error, Period, RegArimaSpec, Series, Spec};
    ///
    /// fn adjust_monthly(values: &[f64]) -> Result<Adjustment, Error> {
    ///     let preprocessing = RegArimaSpec::new(ArimaModel::airline());
    ///     let series = Series::new(Period::monthly(2000, 1)?, values);
    ///     x13::adjust(series, &Spec::seats(preprocessing))
    /// }
    /// # let _request: fn(&[f64]) -> Result<Adjustment, Error> = adjust_monthly;
    /// ```
    ///
    /// The same model, with an automatic choice between no transform and logs:
    ///
    /// ```
    /// use x13::regarima::TransformSelection;
    /// use x13::{Adjustment, ArimaModel, Error, Period, RegArimaSpec, Series, Spec};
    /// fn adjust_automatic(values: &[f64]) -> Result<Adjustment, Error> {
    ///     let request = RegArimaSpec::new(ArimaModel::airline())
    ///         .with_transform(TransformSelection::automatic());
    ///     let series = Series::new(Period::monthly(2000, 1)?, values);
    ///     x13::adjust(series, &Spec::seats(request))
    /// }
    /// # let _request: fn(&[f64]) -> Result<Adjustment, Error> = adjust_automatic;
    /// ```
    ///
    /// Automatic model and constant selection:
    ///
    /// ```
    /// use x13::{Adjustment, Error, Period, RegArimaSpec, Series, Spec};
    /// fn adjust_selected(values: &[f64]) -> Result<Adjustment, Error> {
    ///     let series = Series::new(Period::monthly(2000, 1)?, values);
    ///     x13::adjust(series, &Spec::seats(RegArimaSpec::automatic_model()))
    /// }
    /// # let _request: fn(&[f64]) -> Result<Adjustment, Error> = adjust_selected;
    /// ```
    pub const fn seats(regarima: RegArimaSpec) -> Self {
        Self {
            kind: SpecKind::Seats { regarima },
        }
    }

    /// Seasonal-adjustment method this spec runs: X-11 or SEATS.
    #[must_use]
    pub const fn method(&self) -> Method {
        match self.kind {
            SpecKind::Seats { .. } => Method::Seats,
            SpecKind::X11 { .. } | SpecKind::X11RegArima { .. } => Method::X11,
        }
    }

    /// The X-11 settings given to [`Spec::x11`] or [`Spec::x11_regarima`], or
    /// `None` for SEATS.
    #[must_use]
    pub const fn x11_options(&self) -> Option<x11::Options> {
        match self.kind {
            SpecKind::X11 { options } | SpecKind::X11RegArima { options, .. } => Some(options),
            SpecKind::Seats { .. } => None,
        }
    }

    /// The regARIMA model specification, or `None` for [`Spec::x11`].
    #[must_use]
    pub const fn regarima(&self) -> Option<&RegArimaSpec> {
        match &self.kind {
            SpecKind::X11 { .. } => None,
            SpecKind::X11RegArima { regarima, .. } | SpecKind::Seats { regarima } => Some(regarima),
        }
    }

    pub(super) fn kind(&self) -> SpecKind {
        self.kind.clone()
    }
}

pub(super) fn validate_order(
    factor: ArimaFactor,
    p: u8,
    d: u8,
    q: u8,
    maximum_p: u8,
    maximum_d: u8,
    maximum_q: u8,
) -> Result<(), Error> {
    for (term, value, maximum) in [
        (ArimaOrderTerm::Autoregressive, p, maximum_p),
        (ArimaOrderTerm::Differencing, d, maximum_d),
        (ArimaOrderTerm::MovingAverage, q, maximum_q),
    ] {
        if value > maximum {
            return Err(Error::from(SpecProblem::ArimaOrder {
                factor,
                term,
                value,
                maximum,
            }));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
