//! The regARIMA model: the options of a [`RegArimaSpec`](crate::RegArimaSpec)
//! and what the fit reports.
//!
//! A request adds to its ARIMA orders a transform ([`TransformSelection`]),
//! calendar effects ([`CalendarSpec`]), outliers to detect ([`OutlierSpec`])
//! or to include ([`ManualOutlier`]), regressors of its own
//! ([`UserRegressor`]) and forecasts ([`ForecastSpec`]).
//!
//! The fit is read from [`ModelResult`](crate::ModelResult) and
//! [`Adjustment`](crate::Adjustment): the coefficients
//! ([`ParameterEstimate`], [`RegressionEstimate`]), the outliers found
//! ([`DetectedOutlier`]), the calendar effects ([`CalendarEffects`]) and the
//! automatic choices ([`ModelSelectionResult`],
//! [`TransformSelectionResult`]).

pub use crate::api::error::CalendarFeature;
pub use crate::api::regarima::{
    ArimaFactor, AutoTransformOptions, CalendarBasis, CalendarEffectScale, CalendarEffects,
    CalendarSelection, CalendarSpec, ConstantSelection, Convergence, DetectedOutlier, Easter,
    FlowLengthTreatment, FlowTradingDay, ForecastHorizon, ForecastSpec, HolidayVariable,
    InformationCriteria, LaborDay, LengthEffect, ManualOutlier, ModelSelection,
    ModelSelectionResult, OutlierCriticalValue, OutlierCriticalValues, OutlierMethod, OutlierSpec,
    OutlierType, OutlierTypes, ParameterEstimate, ParameterKind, PrecisionCriterion,
    RegressionEstimate, RegressionVariable, ResolvedArimaModel, ResolvedArimaOrder,
    StockTradingDay, TemporaryChangeRate, Thanksgiving, TradingDayContrast, TradingDayVariable,
    TransformSelection, TransformSelectionResult, UserRegressor, UserRegressorType, Weekday,
};

pub(crate) use crate::api::regarima::X11ModeSelection;
