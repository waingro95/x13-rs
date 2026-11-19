//! Monthly and quarterly periods, and the series that carry them.

use std::cmp::Ordering;
use std::fmt;
use std::iter::FusedIterator;

use thiserror::Error;

/// Observation frequency of a series: monthly or quarterly.
///
/// These are the two frequencies Build 62 adjusts. A [`Series`] and every
/// [`Period`] carry one, and [`adjust`](crate::adjust) rejects a date or
/// regressor whose frequency differs from the input series rather than
/// reinterpreting it.
///
/// Displays as `monthly` or `quarterly`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Frequency {
    /// Twelve periods per year.
    Monthly,
    /// Four periods per year.
    Quarterly,
}

impl Frequency {
    /// Number of observations in one calendar year.
    #[must_use]
    pub const fn periods_per_year(self) -> u8 {
        match self {
            Self::Monthly => 12,
            Self::Quarterly => 4,
        }
    }

    /// Name of one subperiod, for messages: `month` or `quarter`.
    pub(crate) const fn subperiod_name(self) -> &'static str {
        match self {
            Self::Monthly => "month",
            Self::Quarterly => "quarter",
        }
    }

    // Orders frequencies for `Period`'s `Ord` only; `Frequency` itself stays
    // unordered, so callers don't see a meaningless "monthly < quarterly".
    const fn rank(self) -> u8 {
        match self {
            Self::Monthly => 0,
            Self::Quarterly => 1,
        }
    }
}

/// Displays as `monthly` or `quarterly`.
impl fmt::Display for Frequency {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Monthly => "monthly",
            Self::Quarterly => "quarterly",
        })
    }
}

/// A year and one-based month or quarter, tagged with its frequency.
///
/// Periods name dates in a request (span bounds, manual outliers) and in a
/// result (forecast starts, detected outliers). Because the frequency is part
/// of the value, March 2001 and the third quarter of 2001 are different
/// periods, and a date of the wrong frequency is an error, never a silent
/// reinterpretation.
///
/// Periods order by frequency first (monthly before quarterly), then
/// chronologically. Within one frequency the order is calendar order; across
/// frequencies it only groups each frequency's periods together and carries
/// no calendar meaning.
///
/// Displays as `2020-03` (monthly) or `2020Q3` (quarterly); `Debug` prints
/// the same form.
///
/// ```
/// use x13::{Frequency, Period};
///
/// let march = Period::monthly(2020, 3)?;
/// let third_quarter = Period::quarterly(2020, 3)?;
/// assert_ne!(march, third_quarter);
/// assert_eq!(march.to_string(), "2020-03");
/// assert_eq!(third_quarter.to_string(), "2020Q3");
/// assert_eq!(third_quarter.frequency(), Frequency::Quarterly);
/// assert!(Period::monthly(2020, 2)? < march);
/// assert!(Period::quarterly(2020, 5).is_err());
/// # Ok::<(), x13::PeriodError>(())
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Period {
    year: i32,
    subperiod: u8,
    frequency: Frequency,
}

impl Period {
    /// Constructs `month` (1 to 12) of `year`.
    ///
    /// Returns [`PeriodError::InvalidSubperiod`] for any other month.
    pub const fn monthly(year: i32, month: u8) -> Result<Self, PeriodError> {
        Self::new(Frequency::Monthly, year, month)
    }

    /// Constructs `quarter` (1 to 4) of `year`.
    ///
    /// Returns [`PeriodError::InvalidSubperiod`] for any other quarter.
    pub const fn quarterly(year: i32, quarter: u8) -> Result<Self, PeriodError> {
        Self::new(Frequency::Quarterly, year, quarter)
    }

    /// Calendar year containing this period.
    #[must_use]
    pub const fn year(self) -> i32 {
        self.year
    }

    /// One-based month or quarter within the year.
    #[must_use]
    pub const fn subperiod(self) -> u8 {
        self.subperiod
    }

    /// Whether this is a month or a quarter.
    #[must_use]
    pub const fn frequency(self) -> Frequency {
        self.frequency
    }

    const fn new(frequency: Frequency, year: i32, subperiod: u8) -> Result<Self, PeriodError> {
        if subperiod == 0 || subperiod > frequency.periods_per_year() {
            return Err(PeriodError::InvalidSubperiod {
                frequency,
                subperiod,
            });
        }
        Ok(Self {
            year,
            subperiod,
            frequency,
        })
    }

    /// Moves this period by a signed number of periods of its own frequency:
    /// negative values move it earlier.
    ///
    /// Returns [`PeriodError::ShiftOverflow`] when the result's year would
    /// leave the `i32` range; no `periods` value panics.
    ///
    /// ```
    /// use x13::Period;
    ///
    /// let november = Period::monthly(2015, 11)?;
    /// assert_eq!(november.shift(3)?, Period::monthly(2016, 2)?);
    /// assert_eq!(november.shift(-11)?, Period::monthly(2014, 12)?);
    /// assert!(Period::quarterly(i32::MAX, 4)?.shift(1).is_err());
    /// # Ok::<(), x13::PeriodError>(())
    /// ```
    pub fn shift(self, periods: i64) -> Result<Self, PeriodError> {
        let overflow = PeriodError::ShiftOverflow { periods };
        let ordinal = self.ordinal().checked_add(periods).ok_or(overflow)?;
        Self::from_ordinal(self.frequency, ordinal).ok_or(overflow)
    }

    /// Periods since year zero's first subperiod, at this period's frequency.
    fn ordinal(self) -> i64 {
        i64::from(self.year) * i64::from(self.frequency.periods_per_year())
            + i64::from(self.subperiod - 1)
    }

    fn from_ordinal(frequency: Frequency, ordinal: i64) -> Option<Self> {
        let periods_per_year = i64::from(frequency.periods_per_year());
        let year = i32::try_from(ordinal.div_euclid(periods_per_year)).ok()?;
        let subperiod = ordinal.rem_euclid(periods_per_year) as u8 + 1;
        Some(Self {
            year,
            subperiod,
            frequency,
        })
    }
}

impl Ord for Period {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency
            .rank()
            .cmp(&other.frequency.rank())
            .then(self.year.cmp(&other.year))
            .then(self.subperiod.cmp(&other.subperiod))
    }
}

impl PartialOrd for Period {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Period {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.frequency {
            Frequency::Monthly => write!(formatter, "{}-{:02}", self.year, self.subperiod),
            Frequency::Quarterly => write!(formatter, "{}Q{}", self.year, self.subperiod),
        }
    }
}

/// Prints the same compact date as `Display`, so a period inside another
/// value's `Debug` output reads as a date.
impl fmt::Debug for Period {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

/// A monthly or quarterly series: observations at consecutive periods,
/// beginning at one [`Period`].
///
/// It borrows its values, so building one copies nothing, and it is `Copy`.
/// Observation `i` falls `i` periods after [`start`](Self::start), so the
/// values carry no dates of their own. [`adjust`](crate::adjust) takes one as
/// its input and returns its dated outputs as `Series` too.
///
/// Construction checks nothing: the checks that need the values, such as an
/// empty series or a year past the supported range, run in
/// [`adjust`](crate::adjust).
///
/// A gap is a NaN observation, never a hole in the dates. `PartialEq` compares
/// the `f64` values, so a series with a gap never equals itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Series<'a> {
    start: Period,
    values: &'a [f64],
}

impl<'a> Series<'a> {
    /// Dates `values` from `start`: the first value falls at `start`, the next
    /// one period later, and so on, in `start`'s frequency.
    ///
    /// ```
    /// use x13::{Frequency, Period, Series};
    ///
    /// let values = [100.0, 102.5, f64::NAN, 104.0];
    /// let series = Series::new(Period::quarterly(2019, 3)?, &values);
    /// assert_eq!(series.frequency(), Frequency::Quarterly);
    /// assert_eq!(series.len(), 4);
    ///
    /// let dates: Vec<String> = series.periods()?.map(|period| period.to_string()).collect();
    /// assert_eq!(dates, ["2019Q3", "2019Q4", "2020Q1", "2020Q2"]);
    /// # Ok::<(), x13::PeriodError>(())
    /// ```
    #[must_use]
    pub const fn new(start: Period, values: &'a [f64]) -> Self {
        Self { start, values }
    }

    /// Period of the first observation.
    #[must_use]
    pub const fn start(self) -> Period {
        self.start
    }

    /// The observations, in period order.
    #[must_use]
    pub const fn values(self) -> &'a [f64] {
        self.values
    }

    /// Frequency of every observation.
    #[must_use]
    pub const fn frequency(self) -> Frequency {
        self.start.frequency()
    }

    /// Number of observations, gaps included.
    #[must_use]
    pub const fn len(self) -> usize {
        self.values.len()
    }

    /// Whether there are no observations.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }

    /// Iterates over the period of each observation, in order.
    ///
    /// The whole span is checked before the iterator is returned, so
    /// iteration itself cannot fail partway through. Returns
    /// [`PeriodError::PeriodOverflow`] when the last period would leave the
    /// `i32` year range.
    pub fn periods(self) -> Result<Periods, PeriodError> {
        TimeAxis::from_start(self.start).periods(self.values.len())
    }
}

/// The dates of an input series inside the crate: a contiguous run of months
/// or quarters beginning at one period.
///
/// An axis is just its first period. Observation `i` falls at
/// [`period_at(i)`](Self::period_at), and the axis carries no length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TimeAxis {
    start: Period,
}

impl TimeAxis {
    pub(crate) const fn from_start(start: Period) -> Self {
        Self { start }
    }

    /// Constructs a monthly axis beginning at `month` (1 to 12) in `year`.
    pub(crate) fn monthly(year: i32, month: u8) -> Result<Self, PeriodError> {
        Period::monthly(year, month).map(Self::from_start)
    }

    /// Constructs a quarterly axis beginning at `quarter` (1 to 4) in `year`.
    #[cfg(test)]
    pub(crate) fn quarterly(year: i32, quarter: u8) -> Result<Self, PeriodError> {
        Period::quarterly(year, quarter).map(Self::from_start)
    }

    /// Frequency of every observation on this axis.
    pub(crate) const fn frequency(self) -> Frequency {
        self.start.frequency
    }

    /// Period of the first observation (index zero).
    pub(crate) const fn start(self) -> Period {
        self.start
    }

    /// Returns the period of the zero-based observation `index`, or
    /// [`PeriodError::PeriodOverflow`] when the year would leave the `i32`
    /// range.
    pub(crate) fn period_at(self, index: usize) -> Result<Period, PeriodError> {
        let overflow = PeriodError::PeriodOverflow { index };
        let offset = i64::try_from(index).map_err(|_| overflow)?;
        self.start.shift(offset).map_err(|_| overflow)
    }

    /// Checks that all `count` periods from the start are representable, and
    /// returns the span that proves it.
    ///
    /// Returns [`PeriodError::PeriodOverflow`] for the last period when its
    /// year would leave the `i32` range.
    pub(crate) fn span(self, count: usize) -> Result<AxisSpan, PeriodError> {
        if let Some(last_index) = count.checked_sub(1) {
            self.period_at(last_index)?;
        }
        Ok(AxisSpan {
            start: self.start,
            count,
        })
    }

    /// Iterates over exactly `count` contiguous periods.
    ///
    /// The complete span is checked before the iterator is returned, so
    /// iteration itself cannot fail partway through.
    pub(crate) fn periods(self, count: usize) -> Result<Periods, PeriodError> {
        self.span(count)?;
        Ok(Periods {
            frequency: self.frequency(),
            next_ordinal: self.start.ordinal(),
            remaining: count,
        })
    }

    /// Moves the axis start by a signed number of periods: negative values
    /// move it earlier.
    pub(crate) fn shift(self, periods: i64) -> Result<Self, PeriodError> {
        self.start.shift(periods).map(Self::from_start)
    }
}

/// The first `count` periods of an axis, checked once by [`TimeAxis::span`].
///
/// Code that labels an observation index with its date (error values and
/// diagnostics) takes the span in place of an axis and an index. The span
/// has already proved every date in range, so [`period`](Self::period) is
/// total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AxisSpan {
    start: Period,
    count: usize,
}

impl AxisSpan {
    /// Period of the first observation (index zero).
    pub(crate) const fn start(self) -> Period {
        self.start
    }

    /// Number of observations the span covers.
    pub(crate) const fn count(self) -> usize {
        self.count
    }

    /// Period of the zero-based observation `index`, which must be below
    /// [`count`](Self::count).
    pub(crate) fn period(self, index: usize) -> Period {
        debug_assert!(
            index < self.count,
            "period index {index} is outside a span of {} observations",
            self.count
        );
        // `TimeAxis::span` checked the last period, so every index below
        // `count` is in range.
        TimeAxis::from_start(self.start)
            .period_at(index)
            .unwrap_or(self.start)
    }
}

/// Iterator over consecutive periods, returned by [`Series::periods`].
///
/// It yields exactly the requested number of periods in calendar order and
/// reports that length through [`ExactSizeIterator`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Periods {
    frequency: Frequency,
    next_ordinal: i64,
    remaining: usize,
}

impl Iterator for Periods {
    type Item = Period;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        self.remaining -= 1;
        // `Series::periods` checked the last period, so every ordinal fits.
        Period::from_ordinal(self.frequency, ordinal)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Periods {}
impl FusedIterator for Periods {}

/// Failure to construct a [`Period`], or to move a period or a series' dates
/// outside the representable years.
///
/// [`adjust`](crate::adjust) wraps it as
/// [`Error::InvalidPeriod`](crate::Error::InvalidPeriod), whose
/// [`kind`](crate::Error::kind) is
/// [`ErrorKind::InvalidInput`](crate::ErrorKind::InvalidInput). More variants
/// may be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[non_exhaustive]
pub enum PeriodError {
    /// A month or quarter number is out of range.
    #[error(
        "{} {subperiod} is not between 1 and {}",
        frequency.subperiod_name(),
        frequency.periods_per_year()
    )]
    InvalidSubperiod {
        /// Frequency of the period.
        frequency: Frequency,
        /// Requested month or quarter number.
        subperiod: u8,
    },
    /// A period index falls outside the supported year range.
    #[error("period at observation index {index} is outside the supported year range")]
    PeriodOverflow {
        /// Zero-based observation index.
        index: usize,
    },
    /// A shift moves a period outside the supported year range.
    #[error("shifting by {periods} periods is outside the supported year range")]
    ShiftOverflow {
        /// Requested shift in periods.
        periods: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn series_is_copy_and_reports_its_parts() -> Result<(), PeriodError> {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let start = Period::quarterly(2019, 3)?;
        let series = Series::new(start, &values);
        let copy = series;
        assert_eq!(series, copy);
        assert_eq!(series.start(), start);
        assert_eq!(series.values(), &values);
        assert_eq!(series.frequency(), Frequency::Quarterly);
        assert_eq!(series.len(), 5);
        assert!(!series.is_empty());
        let dates: Vec<String> = series.periods()?.map(|period| period.to_string()).collect();
        assert_eq!(dates, ["2019Q3", "2019Q4", "2020Q1", "2020Q2", "2020Q3"]);

        let empty = Series::new(start, &[]);
        assert!(empty.is_empty());
        assert_eq!(empty.periods()?.count(), 0);
        Ok(())
    }

    #[test]
    fn period_shift_crosses_years() -> Result<(), PeriodError> {
        assert_eq!(
            Period::monthly(2015, 11)?.shift(3)?,
            Period::monthly(2016, 2)?
        );
        assert_eq!(
            Period::quarterly(2015, 1)?.shift(-1)?,
            Period::quarterly(2014, 4)?
        );
        Ok(())
    }

    #[test]
    fn period_shift_reports_overflow() -> Result<(), PeriodError> {
        assert_eq!(
            Period::monthly(i32::MAX, 12)?.shift(1),
            Err(PeriodError::ShiftOverflow { periods: 1 })
        );
        assert_eq!(
            Period::quarterly(i32::MIN, 1)?.shift(-1),
            Err(PeriodError::ShiftOverflow { periods: -1 })
        );
        Ok(())
    }

    #[test]
    fn period_shift_never_panics_at_the_extremes() -> Result<(), PeriodError> {
        let starts = [
            Period::monthly(i32::MIN, 1)?,
            Period::monthly(2020, 6)?,
            Period::quarterly(i32::MAX, 4)?,
        ];
        for start in starts {
            for periods in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX] {
                match start.shift(periods) {
                    Ok(shifted) => assert_eq!(shifted.frequency(), start.frequency()),
                    Err(error) => assert_eq!(error, PeriodError::ShiftOverflow { periods }),
                }
            }
            assert_eq!(start.shift(0)?, start);
        }
        // The ends of the year range are reachable from inside it.
        assert_eq!(
            Period::monthly(i32::MIN, 1)?.shift(11)?,
            Period::monthly(i32::MIN, 12)?
        );
        assert_eq!(
            Period::quarterly(i32::MAX, 4)?.shift(-3)?,
            Period::quarterly(i32::MAX, 1)?
        );
        Ok(())
    }

    #[test]
    fn period_constructors_work_in_const_context() {
        const MARCH: Result<Period, PeriodError> = Period::monthly(2020, 3);
        const BAD_QUARTER: Result<Period, PeriodError> = Period::quarterly(2020, 5);
        assert_eq!(MARCH.map(|period| period.subperiod()), Ok(3));
        assert_eq!(
            BAD_QUARTER,
            Err(PeriodError::InvalidSubperiod {
                frequency: Frequency::Quarterly,
                subperiod: 5,
            })
        );
    }

    fn ymd(period: Period) -> (i32, u8) {
        (period.year(), period.subperiod())
    }

    #[test]
    fn monthly_axis_maps_offsets_across_year_boundaries() -> Result<(), PeriodError> {
        let axis = TimeAxis::monthly(1973, 1)?;

        assert_eq!(axis.frequency(), Frequency::Monthly);
        assert_eq!(ymd(axis.start()), (1973, 1));
        assert_eq!(ymd(axis.period_at(11)?), (1973, 12));
        assert_eq!(ymd(axis.period_at(12)?), (1974, 1));
        assert_eq!(ymd(axis.period_at(639)?), (2026, 4));
        Ok(())
    }

    #[test]
    fn quarterly_axis_maps_offsets_across_year_boundaries() -> Result<(), PeriodError> {
        let axis = TimeAxis::quarterly(2000, 3)?;

        assert_eq!(axis.frequency(), Frequency::Quarterly);
        assert_eq!(ymd(axis.period_at(0)?), (2000, 3));
        assert_eq!(ymd(axis.period_at(1)?), (2000, 4));
        assert_eq!(ymd(axis.period_at(2)?), (2001, 1));
        Ok(())
    }

    #[test]
    fn constructors_reject_invalid_subperiods() {
        assert_eq!(
            TimeAxis::monthly(2020, 0),
            Err(PeriodError::InvalidSubperiod {
                frequency: Frequency::Monthly,
                subperiod: 0,
            })
        );
        assert_eq!(
            TimeAxis::monthly(2020, 13),
            Err(PeriodError::InvalidSubperiod {
                frequency: Frequency::Monthly,
                subperiod: 13,
            })
        );
        assert_eq!(
            TimeAxis::quarterly(2020, 5),
            Err(PeriodError::InvalidSubperiod {
                frequency: Frequency::Quarterly,
                subperiod: 5,
            })
        );
    }

    #[test]
    fn periods_yields_the_requested_count() -> Result<(), PeriodError> {
        let axis = TimeAxis::quarterly(2023, 3)?;

        let periods = axis.periods(5)?.map(ymd).collect::<Vec<_>>();

        assert_eq!(
            periods,
            vec![(2023, 3), (2023, 4), (2024, 1), (2024, 2), (2024, 3)]
        );
        Ok(())
    }

    #[test]
    fn periods_past_the_year_range_are_errors() -> Result<(), PeriodError> {
        let axis = TimeAxis::monthly(i32::MAX, 12)?;

        assert_eq!(
            axis.period_at(1),
            Err(PeriodError::PeriodOverflow { index: 1 })
        );
        assert_eq!(
            axis.periods(2).unwrap_err(),
            PeriodError::PeriodOverflow { index: 1 }
        );
        assert_eq!(axis.periods(0)?.count(), 0);
        // `Series::periods` reports the same overflow.
        let series = Series::new(axis.start(), &[1.0, 2.0]);
        assert_eq!(
            series.periods().unwrap_err(),
            PeriodError::PeriodOverflow { index: 1 }
        );
        Ok(())
    }

    #[test]
    fn span_rejects_overflow() -> Result<(), PeriodError> {
        let axis = TimeAxis::monthly(i32::MAX, 12)?;

        assert_eq!(axis.span(0)?.count(), 0);
        assert_eq!(axis.span(1)?.count(), 1);
        assert_eq!(axis.span(2), Err(PeriodError::PeriodOverflow { index: 1 }));
        // The last period may sit exactly on the final representable month.
        let nearly = TimeAxis::monthly(i32::MAX, 11)?;
        assert_eq!(nearly.span(2)?.count(), 2);
        assert_eq!(
            nearly.span(3),
            Err(PeriodError::PeriodOverflow { index: 2 })
        );
        assert_eq!(
            TimeAxis::quarterly(i32::MIN, 1)?.span(usize::MAX),
            Err(PeriodError::PeriodOverflow {
                index: usize::MAX - 1
            })
        );
        Ok(())
    }

    #[test]
    fn span_period_matches_period_at() -> Result<(), PeriodError> {
        for (axis, count) in [
            (TimeAxis::monthly(1999, 11)?, 40),
            (TimeAxis::quarterly(2000, 4)?, 17),
            (TimeAxis::monthly(i32::MAX, 11)?, 2),
            (TimeAxis::quarterly(i32::MIN, 1)?, 9),
        ] {
            let span = axis.span(count)?;
            assert_eq!(span.start(), axis.start());
            assert_eq!(span.count(), count);
            for index in 0..count {
                assert_eq!(span.period(index), axis.period_at(index)?);
            }
        }
        Ok(())
    }

    #[test]
    fn axis_shift_crosses_years() -> Result<(), PeriodError> {
        let monthly = TimeAxis::monthly(2000, 1)?;
        let quarterly = TimeAxis::quarterly(2000, 2)?;

        assert_eq!(ymd(monthly.shift(-12)?.start()), (1999, 1));
        assert_eq!(ymd(monthly.shift(14)?.start()), (2001, 3));
        assert_eq!(ymd(quarterly.shift(-3)?.start()), (1999, 3));
        assert_eq!(ymd(quarterly.shift(7)?.start()), (2002, 1));
        Ok(())
    }

    #[test]
    fn axis_shift_reports_overflow() -> Result<(), PeriodError> {
        let maximum = TimeAxis::monthly(i32::MAX, 12)?;
        let minimum = TimeAxis::quarterly(i32::MIN, 1)?;

        assert_eq!(
            maximum.shift(1),
            Err(PeriodError::ShiftOverflow { periods: 1 })
        );
        assert_eq!(
            minimum.shift(-1),
            Err(PeriodError::ShiftOverflow { periods: -1 })
        );
        Ok(())
    }

    #[test]
    fn periods_carry_their_frequency() -> Result<(), PeriodError> {
        let march = Period::monthly(2020, 3)?;
        let third_quarter = Period::quarterly(2020, 3)?;
        assert_eq!(march.frequency(), Frequency::Monthly);
        assert_eq!(third_quarter.frequency(), Frequency::Quarterly);
        assert_eq!((march.year(), march.subperiod()), (2020, 3));
        // Same year and subperiod, different dates.
        assert_ne!(march, third_quarter);
        assert_eq!(TimeAxis::monthly(2020, 3)?.start(), march);
        assert!(
            TimeAxis::quarterly(2019, 4)?
                .periods(5)?
                .all(|period| period.frequency() == Frequency::Quarterly)
        );
        for (result, frequency, subperiod) in [
            (Period::monthly(2020, 0), Frequency::Monthly, 0),
            (Period::monthly(2020, 13), Frequency::Monthly, 13),
            (Period::quarterly(2020, 0), Frequency::Quarterly, 0),
            (Period::quarterly(2020, 5), Frequency::Quarterly, 5),
        ] {
            assert_eq!(
                result,
                Err(PeriodError::InvalidSubperiod {
                    frequency,
                    subperiod
                })
            );
        }
        Ok(())
    }

    #[test]
    fn periods_display_as_calendar_dates() -> Result<(), PeriodError> {
        assert_eq!(Period::monthly(2020, 3)?.to_string(), "2020-03");
        assert_eq!(Period::monthly(1999, 12)?.to_string(), "1999-12");
        assert_eq!(Period::quarterly(2020, 3)?.to_string(), "2020Q3");
        // `Debug` prints the same form.
        assert_eq!(format!("{:?}", Period::monthly(1992, 7)?), "1992-07");
        assert_eq!(format!("{:?}", Period::quarterly(2001, 3)?), "2001Q3");
        Ok(())
    }

    #[test]
    fn periods_order_by_frequency_then_date() -> Result<(), PeriodError> {
        let monthly = TimeAxis::monthly(1999, 11)?.periods(4)?;
        let mut sorted = monthly.clone().collect::<Vec<_>>();
        sorted.reverse();
        sorted.sort();
        assert_eq!(sorted, monthly.collect::<Vec<_>>());
        // Every monthly period sorts before every quarterly one, whatever the
        // calendar says: cross-frequency order has no calendar meaning.
        assert!(Period::monthly(2100, 1)? < Period::quarterly(1900, 1)?);
        Ok(())
    }
}
