//! Calendar regressors: a trading-day or length-of-period basis and moving
//! holidays.

use super::ForecastSpec;
use crate::api::error::CalendarFeature;
use crate::api::spec::{MINIMUM_TRADING_DAY_YEAR, validate_year_range};
use crate::error::SpecProblem;
use crate::time::TimeAxis;
use crate::{Error, Frequency};

/// How configured trading-day and Easter effects enter an explicit model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum CalendarSelection {
    /// Keep every configured effect (the default).
    #[default]
    Specified,
    /// Test configured trading days, then configured Easter, against omission
    /// using the default AICC difference of zero. Other holidays remain.
    /// Requires an explicit model and transform without automatic outliers.
    /// Standalone length and explicit length-of-period testing are not supported.
    AicTest,
}

/// Calendar regressors for the regARIMA model: a trading-day or
/// length-of-period basis and moving holidays. Set with
/// [`RegArimaSpec::with_calendar`](crate::RegArimaSpec::with_calendar).
///
/// Start from [`CalendarSpec::new`], which has no effects, and add what the
/// series needs. Each effect covers a fixed range of years, counting
/// forecasts and backcasts: Easter and Labor Day 1901 to 2100, Thanksgiving
/// 1939 to 2100, trading days from 1776. Under SEATS, an automatic model (or
/// an explicit model with an automatic transform and Easter) also evaluates
/// holidays over the three years of forecasts SEATS generates internally, so a
/// series ending within three years of 2100 can be rejected even when it
/// requests fewer forecasts. Stock trading days, Labor Day and Thanksgiving
/// are monthly only. [`adjust`](crate::adjust) reports a request outside
/// these limits as an error rather than dropping the effect.
///
/// ```
/// use x13::regarima::{
///     CalendarBasis, CalendarSpec, Easter, FlowTradingDay, LaborDay, TradingDayContrast,
/// };
///
/// let trading_day = FlowTradingDay::new(TradingDayContrast::WeekdayWeekend);
/// let calendar = CalendarSpec::new()
///     .with_basis(CalendarBasis::FlowTradingDay(trading_day))
///     .with_easter(Easter::new(8)?)
///     .with_labor_day(LaborDay::new(8)?);
/// assert_eq!(calendar.easter().map(Easter::window), Some(8));
/// assert!(Easter::new(26).is_err());
/// # Ok::<(), x13::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[must_use]
pub struct CalendarSpec {
    selection: CalendarSelection,
    basis: Option<CalendarBasis>,
    easter: Option<Easter>,
    labor_day: Option<LaborDay>,
    thanksgiving: Option<Thanksgiving>,
}

impl CalendarSpec {
    /// A calendar specification with no effects, keeping every effect added
    /// later ([`CalendarSelection::Specified`]).
    pub const fn new() -> Self {
        Self {
            selection: CalendarSelection::Specified,
            basis: None,
            easter: None,
            labor_day: None,
            thanksgiving: None,
        }
    }

    /// Selects whether configured trading-day/Easter effects are tested by AICC.
    pub const fn with_selection(mut self, selection: CalendarSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Requested calendar-selection policy.
    #[must_use]
    pub const fn selection(self) -> CalendarSelection {
        self.selection
    }

    pub(crate) const fn without_basis(mut self) -> Self {
        self.basis = None;
        self
    }

    pub(crate) const fn without_easter(mut self) -> Self {
        self.easter = None;
        self
    }

    /// Sets or replaces the trading-day or standalone length basis.
    pub const fn with_basis(mut self, basis: CalendarBasis) -> Self {
        self.basis = Some(basis);
        self
    }

    /// Sets or replaces the Easter effect.
    pub const fn with_easter(mut self, easter: Easter) -> Self {
        self.easter = Some(easter);
        self
    }

    /// Sets or replaces the Labor Day effect.
    pub const fn with_labor_day(mut self, labor_day: LaborDay) -> Self {
        self.labor_day = Some(labor_day);
        self
    }

    /// Sets or replaces the Thanksgiving effect.
    pub const fn with_thanksgiving(mut self, thanksgiving: Thanksgiving) -> Self {
        self.thanksgiving = Some(thanksgiving);
        self
    }

    /// Configured trading-day or standalone length basis.
    #[must_use]
    pub const fn basis(self) -> Option<CalendarBasis> {
        self.basis
    }

    /// Configured Easter effect.
    #[must_use]
    pub const fn easter(self) -> Option<Easter> {
        self.easter
    }

    /// Configured Labor Day effect.
    #[must_use]
    pub const fn labor_day(self) -> Option<LaborDay> {
        self.labor_day
    }

    /// Configured Thanksgiving effect.
    #[must_use]
    pub const fn thanksgiving(self) -> Option<Thanksgiving> {
        self.thanksgiving
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.basis.is_none()
            && self.easter.is_none()
            && self.labor_day.is_none()
            && self.thanksgiving.is_none()
    }

    // `extensions` are the horizons Build 62 generates, after editor.f cancels
    // backcasts for a late model span. With no model span, backcasts can reach
    // a year before the holiday tables (adestr.f reads `kdate(1900)`): rejected.
    pub(crate) fn validate(
        self,
        axis: TimeAxis,
        observed_count: usize,
        extensions: ForecastSpec,
    ) -> Result<(), Error> {
        let frequency = axis.frequency();
        if frequency != Frequency::Monthly {
            if matches!(self.basis, Some(CalendarBasis::StockTradingDay(_))) {
                return Err(Error::from(SpecProblem::UnsupportedCalendarFrequency {
                    feature: CalendarFeature::StockTradingDay,
                    frequency,
                }));
            }
            if self.labor_day.is_some() {
                return Err(Error::from(SpecProblem::UnsupportedCalendarFrequency {
                    feature: CalendarFeature::LaborDay,
                    frequency,
                }));
            }
            if self.thanksgiving.is_some() {
                return Err(Error::from(SpecProblem::UnsupportedCalendarFrequency {
                    feature: CalendarFeature::Thanksgiving,
                    frequency,
                }));
            }
        }

        let (start, span_count) = extensions.extended_span(axis, observed_count)?;
        start.periods(span_count)?;
        let first_year = (span_count != 0).then_some(start.start().year());
        let last_year = if let Some(last_index) = span_count.checked_sub(1) {
            Some(start.period_at(last_index)?.year())
        } else {
            None
        };

        if let Some(feature) = match self.basis {
            Some(CalendarBasis::FlowTradingDay(_)) => Some(CalendarFeature::TradingDay),
            Some(CalendarBasis::StockTradingDay(_)) => Some(CalendarFeature::StockTradingDay),
            Some(CalendarBasis::Length(_)) | None => None,
        } {
            validate_year_range(
                feature,
                first_year,
                last_year,
                MINIMUM_TRADING_DAY_YEAR,
                None,
            )?;
        }
        if self.easter.is_some() {
            validate_year_range(
                CalendarFeature::Easter,
                first_year,
                last_year,
                1901,
                Some(2100),
            )?;
        }
        if self.labor_day.is_some() {
            validate_year_range(
                CalendarFeature::LaborDay,
                first_year,
                last_year,
                1901,
                Some(2100),
            )?;
        }
        if self.thanksgiving.is_some() {
            validate_year_range(
                CalendarFeature::Thanksgiving,
                first_year,
                last_year,
                1939,
                Some(2100),
            )?;
        }
        Ok(())
    }
}

/// Mutually exclusive trading-day or standalone length basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CalendarBasis {
    /// Flow trading-day contrasts with an attached length treatment.
    FlowTradingDay(FlowTradingDay),
    /// Monthly stock trading-day contrasts.
    StockTradingDay(StockTradingDay),
    /// A standalone leap-year or length-of-period variable.
    Length(LengthEffect),
}

/// Shape of a trading-day regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TradingDayContrast {
    /// Monday through Saturday contrasts against Sunday.
    SixCoefficient,
    /// One weekday-versus-weekend contrast.
    WeekdayWeekend,
}

/// Length treatment attached to a flow trading-day regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FlowLengthTreatment {
    /// The default leap-year treatment.
    LeapYear,
    /// Do not include a length effect.
    NoLengthEffect,
    /// Include length of month or quarter.
    LengthOfPeriod,
}

/// Flow trading-day configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowTradingDay {
    contrast: TradingDayContrast,
    length_treatment: FlowLengthTreatment,
}

impl FlowTradingDay {
    /// Flow trading day with the default leap-year treatment.
    #[must_use]
    pub const fn new(contrast: TradingDayContrast) -> Self {
        Self {
            contrast,
            length_treatment: FlowLengthTreatment::LeapYear,
        }
    }

    /// Selects the attached length treatment.
    #[must_use]
    pub const fn with_length_treatment(mut self, treatment: FlowLengthTreatment) -> Self {
        self.length_treatment = treatment;
        self
    }

    /// Trading-day contrast shape.
    #[must_use]
    pub const fn contrast(self) -> TradingDayContrast {
        self.contrast
    }

    /// Attached length treatment.
    #[must_use]
    pub const fn length_treatment(self) -> FlowLengthTreatment {
        self.length_treatment
    }
}

/// Monthly stock trading-day configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StockTradingDay {
    contrast: TradingDayContrast,
    sample_day: u8,
}

impl StockTradingDay {
    /// Constructs stock trading day for a sample day in `1..=31`.
    pub fn new(contrast: TradingDayContrast, sample_day: u8) -> Result<Self, Error> {
        if !(1..=31).contains(&sample_day) {
            return Err(Error::from(SpecProblem::StockTradingDaySampleDay {
                sample_day,
            }));
        }
        Ok(Self {
            contrast,
            sample_day,
        })
    }

    /// Trading-day contrast shape.
    #[must_use]
    pub const fn contrast(self) -> TradingDayContrast {
        self.contrast
    }

    /// Requested day of month, clipped to the last day of short months.
    #[must_use]
    pub const fn sample_day(self) -> u8 {
        self.sample_day
    }
}

/// Standalone calendar-length effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LengthEffect {
    /// Leap-year effect.
    LeapYear,
    /// Length-of-month or length-of-quarter effect.
    LengthOfPeriod,
}

display_phrases!(LengthEffect {
    LeapYear => "leap-year",
    LengthOfPeriod => "length-of-period",
});

/// Easter moving-holiday configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Easter {
    window: u8,
}

impl Easter {
    /// Constructs an Easter effect with a window in `0..=25`.
    pub fn new(window: u8) -> Result<Self, Error> {
        if window > 25 {
            return Err(Error::from(SpecProblem::EasterWindow { window }));
        }
        Ok(Self { window })
    }

    /// Easter window length.
    #[must_use]
    pub const fn window(self) -> u8 {
        self.window
    }
}

/// Labor Day moving-holiday configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaborDay {
    window: u8,
}

impl LaborDay {
    /// Constructs a Labor Day effect with a window in `1..=25`.
    pub fn new(window: u8) -> Result<Self, Error> {
        if !(1..=25).contains(&window) {
            return Err(Error::from(SpecProblem::LaborDayWindow { window }));
        }
        Ok(Self { window })
    }

    /// Labor Day window length.
    #[must_use]
    pub const fn window(self) -> u8 {
        self.window
    }
}

/// Thanksgiving moving-holiday configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Thanksgiving {
    offset: i8,
}

impl Thanksgiving {
    /// Constructs a Thanksgiving effect with offset `-8..=-1` or `1..=17`.
    pub fn new(offset: i8) -> Result<Self, Error> {
        if !((-8..=-1).contains(&offset) || (1..=17).contains(&offset)) {
            return Err(Error::from(SpecProblem::ThanksgivingOffset { offset }));
        }
        Ok(Self { offset })
    }

    /// Thanksgiving offset.
    #[must_use]
    pub const fn offset(self) -> i8 {
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::regarima::ForecastHorizon;

    #[test]
    fn trading_day_years_are_capped_at_9999() {
        let calendar = CalendarSpec::new().with_basis(CalendarBasis::FlowTradingDay(
            FlowTradingDay::new(TradingDayContrast::WeekdayWeekend),
        ));
        let extensions = ForecastSpec::default();
        let far = TimeAxis::quarterly(2_147_483_000, 3).unwrap();
        assert_eq!(
            calendar.validate(far, 40, extensions),
            Err(Error::CalendarYearOutOfRange {
                feature: CalendarFeature::TradingDay,
                year: 2_147_483_000,
                minimum: 1776,
                maximum: Some(9999),
            })
        );
        // Four quarters from 9999Q1 end in 9999; a fifth period lands in 10000.
        let no_forecasts = ForecastSpec::new(ForecastHorizon::Periods(0), 0).unwrap();
        let axis = TimeAxis::quarterly(9999, 1).unwrap();
        assert_eq!(calendar.validate(axis, 4, no_forecasts), Ok(()));
        assert_eq!(
            calendar.validate(axis, 5, no_forecasts),
            Err(Error::CalendarYearOutOfRange {
                feature: CalendarFeature::TradingDay,
                year: 10000,
                minimum: 1776,
                maximum: Some(9999),
            })
        );
    }
}
