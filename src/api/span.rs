use std::ops::Range;

use crate::time::TimeAxis;
use crate::{Error, Period};

/// Inclusive dated bounds for model estimation or automatic outlier detection.
/// Neither bound is an observation index. When adjusting, each bound must have
/// the input series' frequency ([`Error::PeriodFrequencyMismatch`] otherwise),
/// and the span must be ordered and lie inside its containing span
/// ([`Error::InvalidModelSpan`] or [`Error::InvalidOutlierSpan`] otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeriodSpan {
    start: Period,
    end: Period,
}

impl PeriodSpan {
    /// Span from `start` through `end`, both included. Validation happens when
    /// the request is adjusted.
    #[must_use]
    pub const fn new(start: Period, end: Period) -> Self {
        Self { start, end }
    }

    /// First period.
    #[must_use]
    pub const fn start(self) -> Period {
        self.start
    }

    /// Last period.
    #[must_use]
    pub const fn end(self) -> Period {
        self.end
    }

    /// Offsets of a model span within a series of `count` observations that
    /// starts at `axis`.
    pub(crate) fn resolve_model_span(
        self,
        axis: TimeAxis,
        count: usize,
    ) -> Result<Range<usize>, Error> {
        self.resolve(axis, count, || Error::InvalidModelSpan {
            start: self.start,
            end: self.end,
        })
    }

    /// Offsets of an outlier span within a model span of `count`
    /// observations that starts at `axis`.
    pub(crate) fn resolve_outlier_span(
        self,
        axis: TimeAxis,
        count: usize,
    ) -> Result<Range<usize>, Error> {
        self.resolve(axis, count, || Error::InvalidOutlierSpan {
            start: self.start,
            end: self.end,
        })
    }

    fn resolve(
        self,
        axis: TimeAxis,
        count: usize,
        invalid: impl Fn() -> Error,
    ) -> Result<Range<usize>, Error> {
        let frequency = i64::from(axis.frequency().periods_per_year());
        let offset = |date: Period| -> Result<usize, Error> {
            if date.frequency() != axis.frequency() {
                return Err(Error::PeriodFrequencyMismatch {
                    period: date,
                    expected: axis.frequency(),
                });
            }
            let difference = (i64::from(date.year()) - i64::from(axis.start().year())) * frequency
                + i64::from(date.subperiod())
                - i64::from(axis.start().subperiod());
            usize::try_from(difference).map_err(|_| invalid())
        };
        let start = offset(self.start)?;
        let end = offset(self.end)?;
        if start > end || end >= count {
            return Err(invalid());
        }
        Ok(start..end + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Frequency;

    type Resolver = fn(PeriodSpan, TimeAxis, usize) -> Result<Range<usize>, Error>;
    const RESOLVERS: [Resolver; 2] = [
        PeriodSpan::resolve_model_span,
        PeriodSpan::resolve_outlier_span,
    ];

    #[test]
    fn inclusive_dates_resolve_across_years() {
        let axis = TimeAxis::quarterly(2000, 3).unwrap();
        let span = PeriodSpan::new(axis.period_at(3).unwrap(), axis.period_at(8).unwrap());
        assert_eq!(span.resolve_model_span(axis, 12).unwrap(), 3..9);
    }

    #[test]
    fn bounds_outside_the_series_or_out_of_order_are_rejected() {
        let axis = TimeAxis::monthly(2000, 7).unwrap();
        for (start, end) in [(0, 12), (5, 4)] {
            let span =
                PeriodSpan::new(axis.period_at(start).unwrap(), axis.period_at(end).unwrap());
            assert_eq!(
                span.resolve_model_span(axis, 12),
                Err(Error::InvalidModelSpan {
                    start: span.start(),
                    end: span.end(),
                })
            );
        }
        let before = TimeAxis::monthly(2000, 6).unwrap().start();
        let span = PeriodSpan::new(before, axis.start());
        assert_eq!(
            span.resolve_outlier_span(axis, 12),
            Err(Error::InvalidOutlierSpan {
                start: before,
                end: axis.start(),
            })
        );
    }

    #[test]
    fn bounds_of_another_frequency_are_rejected() {
        // March 2001 must not resolve as 2001Q3, nor 2001Q2 as February 2001.
        let quarterly = TimeAxis::quarterly(2000, 1).unwrap();
        let march = Period::monthly(2001, 3).unwrap();
        for resolve in RESOLVERS {
            assert_eq!(
                resolve(PeriodSpan::new(quarterly.start(), march), quarterly, 20),
                Err(Error::PeriodFrequencyMismatch {
                    period: march,
                    expected: Frequency::Quarterly,
                })
            );
        }
        let monthly = TimeAxis::monthly(2000, 1).unwrap();
        let second_quarter = Period::quarterly(2001, 2).unwrap();
        for resolve in RESOLVERS {
            assert_eq!(
                resolve(
                    PeriodSpan::new(second_quarter, monthly.period_at(30).unwrap()),
                    monthly,
                    60
                ),
                Err(Error::PeriodFrequencyMismatch {
                    period: second_quarter,
                    expected: Frequency::Monthly,
                })
            );
        }
    }
}
