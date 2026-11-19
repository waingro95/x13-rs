//! The regARIMA model: its request options and fitted results.

mod calendar;
mod estimates;
mod forecast;
mod model_selection;
mod outliers;
mod resolved_model;
mod transform;
mod user_regressor;

pub use calendar::{
    CalendarBasis, CalendarSelection, CalendarSpec, Easter, FlowLengthTreatment, FlowTradingDay,
    LaborDay, LengthEffect, StockTradingDay, Thanksgiving, TradingDayContrast,
};
pub use estimates::{
    ArimaFactor, CalendarEffectScale, CalendarEffects, Convergence, HolidayVariable,
    InformationCriteria, ParameterEstimate, ParameterKind, PrecisionCriterion, RegressionEstimate,
    RegressionVariable, TradingDayVariable, Weekday,
};
pub use forecast::{ForecastHorizon, ForecastSpec};
pub use model_selection::{ConstantSelection, ModelSelection, ModelSelectionResult};
pub use outliers::{
    DetectedOutlier, ManualOutlier, OutlierCriticalValue, OutlierCriticalValues, OutlierMethod,
    OutlierSpec, OutlierType, OutlierTypes, TemporaryChangeRate,
};
pub use resolved_model::{ResolvedArimaModel, ResolvedArimaOrder};
pub(crate) use transform::X11ModeSelection;
pub use transform::{AutoTransformOptions, TransformSelection, TransformSelectionResult};
pub use user_regressor::{UserRegressor, UserRegressorType};
