//! The `Spec` of a [`RegArimaRequest`](crate::RegArimaRequest): the spec of
//! its X-11 equivalent, then the options only a request can ask for.
//!
//! `regression { tcrate }` and `outlier { tcrate }` both become
//! `RegArimaSpec::with_temporary_change_rate`, the one Rust spelling of
//! Build 62's single `Tcalfa`. Build 62 rejects a `tcrate` in both specs, so
//! the spec file renderer refuses that request and so does this module.

use std::ops::Range;

use super::RustError;
use crate::{Method, OracleConfiguration, Series, user_regressors::Kind};

pub(super) fn spec(config: &OracleConfiguration, series: &Series) -> Result<x13::Spec, RustError> {
    let OracleConfiguration::RegArima { method, request } = *config else {
        return Err(RustError::UnsupportedConfiguration);
    };
    let x11 = request.x11_equivalent()?;
    let equivalent = if request.model.is_none() {
        super::automatic_model_request(&x11)?
    } else {
        super::explicit_model_request(&x11)?
    };
    // The X-11 equivalent asks for `seasonalma = s3x5`; each rebuild below
    // keeps those X-11 options rather than the library default, and takes the
    // mode from the transform.
    let x11_options = equivalent.options;
    let mut spec = equivalent.into_spec();
    let rebuild = |regarima| x13::Spec::x11_regarima(regarima, x11_options);
    if request.aic_test {
        let regarima = spec
            .regarima()
            .cloned()
            .ok_or(RustError::UnsupportedConfiguration)?;
        let calendar = regarima
            .calendar()
            .ok_or(RustError::UnsupportedConfiguration)?;
        let regarima = regarima
            .with_calendar(calendar.with_selection(x13::regarima::CalendarSelection::AicTest));
        spec = rebuild(regarima);
    }
    if request.manual_outliers {
        let start = super::start_period(series)?;
        let count = series.values.len();
        if count < 9 {
            return Err(RustError::UnsupportedConfiguration);
        }
        let outliers = [
            x13::regarima::ManualOutlier::new(
                x13::regarima::OutlierType::Additive,
                super::period_at(start, count / 3 - 1)?,
            ),
            x13::regarima::ManualOutlier::new(
                x13::regarima::OutlierType::LevelShift,
                super::period_at(start, 2 * count / 3 - 1)?,
            ),
            x13::regarima::ManualOutlier::new(
                x13::regarima::OutlierType::TemporaryChange,
                super::period_at(start, count - 7)?,
            ),
        ];
        let mut regarima = spec
            .regarima()
            .cloned()
            .ok_or(RustError::UnsupportedConfiguration)?
            .with_manual_outliers(&outliers);
        // Build 62's `regression { tcrate }`.
        if let Some(rate) = request.manual_temporary_change_rate {
            // Mirrors the renderer's guard against a second `tcrate`.
            if regarima.temporary_change_rate().is_some() {
                return Err(RustError::UnsupportedConfiguration);
            }
            regarima = regarima
                .with_temporary_change_rate(x13::regarima::TemporaryChangeRate::new(rate.value())?);
        }
        spec = rebuild(regarima);
    } else if request.manual_temporary_change_rate.is_some() {
        // Mirrors the renderer's guard (a manual TC rate requires manual
        // outliers), not a missing Rust feature: Rust can spell this request.
        return Err(RustError::UnsupportedConfiguration);
    }
    if request.model_span.is_some() || request.outlier_span.is_some() {
        let start = super::start_period(series)?;
        let dated_span = |range: Range<usize>| -> Result<x13::PeriodSpan, RustError> {
            Ok(x13::PeriodSpan::new(
                super::period_at(start, range.start)?,
                super::period_at(start, range.end - 1)?,
            ))
        };
        let mut regarima = spec
            .regarima()
            .cloned()
            .ok_or(RustError::UnsupportedConfiguration)?;
        if let Some(policy) = request.model_span {
            regarima = regarima.with_model_span(dated_span(policy.range(series)?)?);
        }
        if let Some(policy) = request.outlier_span {
            let outliers = regarima
                .outliers()
                .ok_or(RustError::UnsupportedConfiguration)?;
            regarima =
                regarima.with_outliers(outliers.with_span(dated_span(policy.range(series)?)?));
        }
        spec = rebuild(regarima);
    }
    if let Some(supplied) = request.user_regressors {
        let start = super::start_period(series)?;
        let regressors = supplied
            .columns(series)?
            .into_iter()
            .map(|column| {
                let kind = match column.kind {
                    Kind::TradingDay => x13::regarima::UserRegressorType::TradingDay,
                    Kind::Holiday => x13::regarima::UserRegressorType::Holiday,
                    Kind::User => x13::regarima::UserRegressorType::User,
                };
                x13::regarima::UserRegressor::new(kind, x13::Series::new(start, &column.values))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let regarima = spec
            .regarima()
            .cloned()
            .ok_or(RustError::UnsupportedConfiguration)?;
        spec = rebuild(regarima.with_user_regressors(&regressors));
    }
    if method == Method::Seats {
        Ok(x13::Spec::seats(
            spec.regarima()
                .cloned()
                .ok_or(RustError::UnsupportedConfiguration)?,
        ))
    } else {
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Frequency, OracleConfiguration, Period, Series};

    #[test]
    fn a_seats_request_builds_a_seats_spec() {
        let config: OracleConfiguration = serde_json::from_value(serde_json::json!({
            "kind":"reg_arima", "method":"seats", "request": {
                "model":{"regular":{"p":0,"d":1,"q":1},"seasonal":{"p":0,"d":1,"q":1}},
                "transform":{"kind":"explicit","transform":"log"},
                "forecast_horizon":{"kind":"one_year"}, "backcast_horizon":0
            }
        }))
        .unwrap();
        let series = Series {
            series_id: "test".into(),
            frequency: Frequency::Monthly,
            start: Period::new(2000, 1),
            values: vec![100.; 144],
        };
        let spec = super::spec(&config, &series).unwrap();
        assert!(spec.x11_options().is_none());
        assert!(spec.regarima().is_some());
    }
}
