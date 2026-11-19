//! A regARIMA request: the model, regressors and spans a row asks for, ahead
//! of a final adjustment by X-11 or SEATS.
//!
//! A row may ask for a Build 62 option the public Rust API lacks. The spec
//! file renders it in full, so the row measures what Build 62 does, and the
//! Rust spec builder returns `UnsupportedConfiguration`, which the harness
//! scores as a failure. The request is not mapped onto the nearest Rust
//! spelling: the row would then compare two different models and report a
//! numeric mismatch instead of the missing feature.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use super::{
    OracleArimaModel, OracleAutomaticModelTransform, OracleCalendarSpec, OracleConfiguration,
    OracleForecastHorizon, OracleOutlierSpec, OracleTemporaryChangeRate, OracleTransform,
};
use crate::{OracleError, Series, user_regressors::UserRegressorRequest};

/// A model span the call table states relative to the series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelSpanPolicy {
    ExcludeLastThreeYears,
    ExcludeFirstThird,
}

impl ModelSpanPolicy {
    pub(crate) fn range(self, series: &Series) -> Result<Range<usize>, OracleError> {
        let count = series.values.len();
        let range = match self {
            Self::ExcludeLastThreeYears => {
                let excluded = 3 * usize::from(series.frequency.periods_per_year());
                0..count
                    .checked_sub(excluded)
                    .ok_or_else(|| invalid("model span requires more than three years"))?
            }
            Self::ExcludeFirstThird => count / 3..count,
        };
        if range.is_empty() {
            return Err(invalid("model span must contain observations"));
        }
        Ok(range)
    }
}

/// An automatic outlier search span the call table states relative to the
/// series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutlierSpanPolicy {
    MiddleThird,
}

impl OutlierSpanPolicy {
    pub(crate) fn range(self, series: &Series) -> Result<Range<usize>, OracleError> {
        let count = series.values.len();
        let range = match self {
            Self::MiddleThird => count / 3..2 * count / 3,
        };
        if range.is_empty() {
            return Err(invalid("outlier span must contain observations"));
        }
        Ok(range)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RegArimaRequest {
    /// None requests default automatic model/constant selection.
    pub(crate) model: Option<OracleArimaModel>,
    pub(crate) transform: OracleAutomaticModelTransform,
    #[serde(default)]
    pub(crate) include_constant: bool,
    pub(crate) calendar: Option<OracleCalendarSpec>,
    pub(crate) outliers: Option<OracleOutlierSpec>,
    pub(crate) forecast_horizon: OracleForecastHorizon,
    pub(crate) backcast_horizon: u16,
    #[serde(default)]
    pub(crate) aic_test: bool,
    #[serde(default)]
    pub(crate) manual_outliers: bool,
    /// Build 62 `regression { tcrate }` for the manual TC; requires `manual_outliers`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) manual_temporary_change_rate: Option<OracleTemporaryChangeRate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_span: Option<ModelSpanPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) outlier_span: Option<OutlierSpanPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) user_regressors: Option<UserRegressorRequest>,
}

impl RegArimaRequest {
    /// The regARIMA and X-11 configuration with this request's model,
    /// transform, calendar and outliers, so the spec file and the Rust spec
    /// each have one mapping of those settings.
    pub(crate) fn x11_equivalent(self) -> Result<OracleConfiguration, OracleError> {
        let mut value = serde_json::json!({
            "forecast_horizon":self.forecast_horizon, "backcast_horizon":self.backcast_horizon,
            "seasonal_filter":"s3x5", "calendar":self.calendar, "outliers":self.outliers,
        });
        if let Some(model) = self.model {
            value["model"] = serde_json::to_value(model).map_err(invalid)?;
            value["include_constant"] = self.include_constant.into();
            match self.transform {
                OracleAutomaticModelTransform::Explicit { transform } => {
                    value["kind"] = if self.calendar.is_some() || self.outliers.is_some() {
                        "calendar_outlier_reg_arima_x11"
                    } else {
                        "explicit_reg_arima_x11"
                    }
                    .into();
                    value["transform"] = serde_json::to_value(transform).map_err(invalid)?;
                    value["mode"] = if transform == OracleTransform::Log {
                        "multiplicative"
                    } else {
                        "additive"
                    }
                    .into();
                }
                OracleAutomaticModelTransform::Automatic { options } => {
                    value["kind"] = "automatic_transform_reg_arima_x11".into();
                    value["automatic_transform"] =
                        serde_json::to_value(options).map_err(invalid)?;
                }
            }
        } else {
            value["kind"] = "automatic_model_x11".into();
            value["transform"] = serde_json::to_value(self.transform).map_err(invalid)?;
        }
        serde_json::from_value(value).map_err(invalid)
    }
}

fn invalid(error: impl std::fmt::Display) -> OracleError {
    OracleError::InvalidConfiguration(error.to_string())
}
