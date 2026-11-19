use std::sync::Arc;

use crate::time::TimeAxis;
use crate::{Error, Period, Series};

/// How a supplied regression column's fitted effect enters the adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UserRegressorType {
    /// The fitted effect joins the trading-day component.
    TradingDay,
    /// The fitted effect joins the holiday component.
    Holiday,
    /// The fitted effect is a separate user-defined regression effect.
    User,
}

display_phrases!(UserRegressorType {
    TradingDay => "trading-day",
    Holiday => "holiday",
    User => "user-defined",
});

/// One finite, dated regression column, including all required extension dates.
/// Values are used as supplied, without centering or extrapolation.
#[derive(Debug, Clone, PartialEq)]
pub struct UserRegressor {
    kind: UserRegressorType,
    start: Period,
    values: Arc<[f64]>,
}

impl UserRegressor {
    /// Builds a regressor of `kind` from a dated column of values.
    ///
    /// The series must cover the observed span and every forecast or backcast
    /// the request needs. Copies the values. Errors when `series` is empty,
    /// runs past the supported year range, or holds a nonfinite value.
    pub fn new(kind: UserRegressorType, series: Series<'_>) -> Result<Self, Error> {
        let start = series.start();
        let values = series.values();
        let axis = TimeAxis::from_start(start);
        let Some(last) = values.len().checked_sub(1) else {
            return Err(Error::EmptyUserRegressor { kind, start });
        };
        axis.period_at(last)
            .map_err(|source| Error::UserRegressorDateRange {
                kind,
                start,
                count: values.len(),
                source,
            })?;
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteUserRegressor {
                kind,
                index,
                period: axis.period_at(index)?,
            });
        }
        Ok(Self {
            kind,
            start,
            values: Arc::from(values),
        })
    }

    /// Adjustment classification.
    #[must_use]
    pub const fn kind(&self) -> UserRegressorType {
        self.kind
    }

    /// Period of the first value.
    #[must_use]
    pub const fn start(&self) -> Period {
        self.start
    }

    /// Regressor values, from [`start`](Self::start) onward.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// `regvar.f` requires supplied values at every requested matrix date.
    pub(crate) fn values_over(&self, axis: TimeAxis, count: usize) -> Result<&[f64], Error> {
        if axis.frequency() != self.start.frequency() {
            return Err(Error::UserRegressorFrequency {
                kind: self.kind,
                expected: axis.frequency(),
                actual: self.start.frequency(),
                start: self.start,
            });
        }
        let Some(last) = count.checked_sub(1) else {
            return Ok(&self.values[..0]);
        };
        let required_end =
            axis.period_at(last)
                .map_err(|source| Error::UserRegressorDateRange {
                    kind: self.kind,
                    start: axis.start(),
                    count,
                    source,
                })?;
        let frequency = i64::from(axis.frequency().periods_per_year());
        let offset = (i64::from(axis.start().year()) - i64::from(self.start.year())) * frequency
            + i64::from(axis.start().subperiod())
            - i64::from(self.start.subperiod());
        let covered = usize::try_from(offset)
            .ok()
            .and_then(|start| start.checked_add(count).map(|end| start..end));
        if let Some(range) = covered
            && let Some(values) = self.values.get(range)
        {
            return Ok(values);
        }
        Err(Error::UserRegressorCoverage {
            kind: self.kind,
            supplied_start: self.start,
            supplied_end: TimeAxis::from_start(self.start).period_at(self.values.len() - 1)?,
            required_start: axis.start(),
            required_end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_sliced_by_calendar_date() {
        for (start, requested) in [
            (
                TimeAxis::monthly(2000, 11).unwrap(),
                TimeAxis::monthly(2001, 1).unwrap(),
            ),
            (
                TimeAxis::quarterly(2000, 3).unwrap(),
                TimeAxis::quarterly(2001, 1).unwrap(),
            ),
        ] {
            let column = UserRegressor::new(
                UserRegressorType::Holiday,
                Series::new(start.start(), &[2.0, 3.0, 5.0, 7.0, 11.0]),
            )
            .unwrap();
            assert_eq!(column.kind(), UserRegressorType::Holiday);
            assert_eq!(column.start(), start.start());
            assert_eq!(column.values_over(requested, 2).unwrap(), &[5.0, 7.0]);
            assert_eq!(column.values(), &[2.0, 3.0, 5.0, 7.0, 11.0]);
        }
    }

    #[test]
    fn empty_nonfinite_and_out_of_range_columns_are_rejected() {
        let start = TimeAxis::monthly(2000, 1).unwrap();
        assert_eq!(
            UserRegressor::new(UserRegressorType::User, Series::new(start.start(), &[])),
            Err(Error::EmptyUserRegressor {
                kind: UserRegressorType::User,
                start: start.start(),
            })
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                UserRegressor::new(
                    UserRegressorType::TradingDay,
                    Series::new(start.start(), &[1.0, value]),
                ),
                Err(Error::NonFiniteUserRegressor {
                    kind: UserRegressorType::TradingDay,
                    index: 1,
                    period: start.period_at(1).unwrap(),
                })
            );
        }
        let last_year = TimeAxis::monthly(i32::MAX, 12).unwrap().start();
        assert_eq!(
            UserRegressor::new(UserRegressorType::User, Series::new(last_year, &[1.0, 2.0])),
            Err(Error::UserRegressorDateRange {
                kind: UserRegressorType::User,
                start: last_year,
                count: 2,
                source: crate::PeriodError::PeriodOverflow { index: 1 },
            })
        );
    }

    #[test]
    fn a_column_of_another_frequency_or_too_short_is_rejected() {
        let start = TimeAxis::quarterly(2000, 1).unwrap();
        let column = UserRegressor::new(
            UserRegressorType::User,
            Series::new(start.start(), &[1.0; 8]),
        )
        .unwrap();
        assert!(matches!(
            column.values_over(TimeAxis::monthly(2000, 1).unwrap(), 4),
            Err(Error::UserRegressorFrequency { .. })
        ));
        for (requested, count) in [(start.shift(-1).unwrap(), 4), (start, 9)] {
            assert_eq!(
                column.values_over(requested, count),
                Err(Error::UserRegressorCoverage {
                    kind: UserRegressorType::User,
                    supplied_start: start.start(),
                    supplied_end: start.period_at(7).unwrap(),
                    required_start: requested.start(),
                    required_end: requested.period_at(count - 1).unwrap(),
                })
            );
        }
    }

    #[test]
    fn a_column_must_cover_the_forecasts_and_backcasts_in_use() {
        use crate::regarima::{ForecastHorizon, ForecastSpec};
        use crate::{ArimaModel, ArimaOrder, PeriodSpan, RegArimaSpec};

        let start = TimeAxis::quarterly(2000, 1).unwrap();
        let order = ArimaOrder::new(0, 1, 1).unwrap();
        let request = RegArimaSpec::new(ArimaModel::new(order, order).unwrap());
        let column = |count| {
            let values = vec![1.0; count];
            UserRegressor::new(UserRegressorType::User, Series::new(start.start(), &values))
                .unwrap()
        };
        let observations = [2.0; 8];
        assert!(matches!(
            request
                .clone()
                .with_user_regressors(&[column(11)])
                .validate(&observations, start),
            Err(Error::UserRegressorCoverage { .. })
        ));
        let complete = request.with_user_regressors(&[column(12)]);
        complete.validate(&observations, start).unwrap();
        let backcasts =
            complete.with_forecast(ForecastSpec::new(ForecastHorizon::OneYear, 2).unwrap());
        assert!(matches!(
            backcasts.validate(&observations, start),
            Err(Error::UserRegressorCoverage { .. })
        ));
        backcasts
            .with_model_span(PeriodSpan::new(
                start.period_at(2).unwrap(),
                start.period_at(7).unwrap(),
            ))
            .validate(&observations, start)
            .unwrap();
    }
}
