//! What the fitted regARIMA model reports: coefficients, regression variables,
//! calendar effects and fit statistics.

use super::{Easter, LaborDay, LengthEffect, OutlierType, Thanksgiving};
use crate::{Frequency, Period, Series};

/// ARIMA factor containing a parameter or order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArimaFactor {
    /// Regular (nonseasonal) factor.
    Regular,
    /// Seasonal factor.
    Seasonal,
}

display_phrases!(ArimaFactor {
    Regular => "nonseasonal",
    Seasonal => "seasonal",
});

/// Kind of fitted ARIMA coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterKind {
    /// Autoregressive coefficient.
    Ar,
    /// Moving-average coefficient.
    Ma,
}

/// One fitted AR or MA coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterEstimate {
    pub(crate) kind: ParameterKind,
    pub(crate) factor: ArimaFactor,
    pub(crate) period: u8,
    pub(crate) lag: u8,
    pub(crate) estimate: f64,
    pub(crate) standard_error: Option<f64>,
}

impl ParameterEstimate {
    /// AR or MA coefficient kind.
    #[must_use]
    pub const fn kind(&self) -> ParameterKind {
        self.kind
    }

    /// Regular or seasonal factor.
    #[must_use]
    pub const fn factor(&self) -> ArimaFactor {
        self.factor
    }

    /// Period of the factor: 1 for the regular factor, the seasonal period
    /// for the seasonal one.
    #[must_use]
    pub const fn period(&self) -> u8 {
        self.period
    }

    /// Lag of the coefficient in observations: its lag within the factor times
    /// [`period`](Self::period).
    #[must_use]
    pub const fn lag(&self) -> u8 {
        self.lag
    }

    /// Estimated coefficient.
    #[must_use]
    pub const fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Estimated standard error, when fitted uncertainty is available.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }
}

/// One fitted regression coefficient.
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionEstimate {
    pub(crate) variable: RegressionVariable,
    pub(crate) name: String,
    pub(crate) estimate: f64,
    pub(crate) standard_error: Option<f64>,
}

impl RegressionEstimate {
    /// Which regressor this coefficient belongs to.
    #[must_use]
    pub const fn variable(&self) -> RegressionVariable {
        self.variable
    }

    /// Stable regressor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Estimated coefficient.
    #[must_use]
    pub const fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Estimated standard error, when regression uncertainty is available.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.standard_error
    }
}

/// Day represented by one of the six trading-day contrasts against Sunday.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weekday {
    /// Monday.
    Monday,
    /// Tuesday.
    Tuesday,
    /// Wednesday.
    Wednesday,
    /// Thursday.
    Thursday,
    /// Friday.
    Friday,
    /// Saturday.
    Saturday,
}

display_phrases!(Weekday {
    Monday => "Monday",
    Tuesday => "Tuesday",
    Wednesday => "Wednesday",
    Thursday => "Thursday",
    Friday => "Friday",
    Saturday => "Saturday",
});

/// A trading-day regression column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TradingDayVariable {
    /// One flow weekday-minus-Sunday column.
    FlowSixCoefficient {
        /// Weekday contrasted with Sunday.
        weekday: Weekday,
    },
    /// One flow weekday-versus-weekend column.
    FlowWeekdayWeekend,
    /// One stock weekday-minus-Sunday column.
    StockSixCoefficient {
        /// Weekday contrasted with Sunday.
        weekday: Weekday,
        /// Day of month the stock is sampled, clipped to the last day of short
        /// months.
        sample_day: u8,
    },
    /// One stock weekday-versus-weekend column.
    StockWeekdayWeekend {
        /// Day of month the stock is sampled, clipped to the last day of short
        /// months.
        sample_day: u8,
    },
}

impl std::fmt::Display for TradingDayVariable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlowSixCoefficient { weekday } => {
                write!(formatter, "{weekday} trading-day coefficient")
            }
            Self::FlowWeekdayWeekend => formatter.write_str("weekday-weekend trading-day contrast"),
            Self::StockSixCoefficient {
                weekday,
                sample_day,
            } => write!(
                formatter,
                "{weekday} stock trading-day coefficient (day {sample_day})"
            ),
            Self::StockWeekdayWeekend { sample_day } => write!(
                formatter,
                "weekday-weekend stock trading-day contrast (day {sample_day})"
            ),
        }
    }
}

/// A moving-holiday regression column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HolidayVariable {
    /// Easter effect.
    Easter(Easter),
    /// Labor Day effect.
    LaborDay(LaborDay),
    /// Thanksgiving effect.
    Thanksgiving(Thanksgiving),
}

impl std::fmt::Display for HolidayVariable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Easter(effect) => write!(formatter, "Easter effect (window {})", effect.window()),
            Self::LaborDay(effect) => {
                write!(formatter, "Labor Day effect (window {})", effect.window())
            }
            Self::Thanksgiving(effect) => {
                write!(
                    formatter,
                    "Thanksgiving effect (offset {})",
                    effect.offset()
                )
            }
        }
    }
}

/// The regressor a fitted coefficient belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RegressionVariable {
    /// Regression constant.
    Constant,
    /// Supplied-list index (zero-based) and declared adjustment classification.
    User {
        /// Zero-based position in the supplied regressor list.
        index: usize,
        /// Declared adjustment classification.
        kind: crate::regarima::UserRegressorType,
    },
    /// Trading-day column.
    TradingDay(TradingDayVariable),
    /// Calendar-length column.
    Length(LengthEffect),
    /// Moving-holiday column.
    Holiday(HolidayVariable),
    /// Outlier regressor.
    Outlier {
        /// Outlier type.
        kind: OutlierType,
        /// Period of the outlier.
        period: Period,
    },
    /// A dated unit pulse used to estimate a missing observation, not an outlier.
    MissingValue {
        /// Period of the missing observation.
        period: Period,
    },
}

impl std::fmt::Display for RegressionVariable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant => formatter.write_str("constant"),
            Self::User { index, kind } => {
                write!(formatter, "{kind} regressor at list index {index}")
            }
            Self::TradingDay(variable) => variable.fmt(formatter),
            Self::Length(effect) => write!(formatter, "{effect} effect"),
            Self::Holiday(variable) => variable.fmt(formatter),
            Self::Outlier { kind, period } => write!(formatter, "{kind} at {period}"),
            Self::MissingValue { period } => {
                write!(formatter, "missing-value regressor at {period}")
            }
        }
    }
}

impl RegressionVariable {
    pub(crate) fn stable_name(self, frequency: Frequency) -> String {
        match self {
            Self::Constant => "const".to_owned(),
            Self::User { index, .. } => format!("user{}", index + 1),
            Self::TradingDay(
                TradingDayVariable::FlowSixCoefficient { weekday }
                | TradingDayVariable::StockSixCoefficient { weekday, .. },
            ) => match weekday {
                Weekday::Monday => "Mon",
                Weekday::Tuesday => "Tue",
                Weekday::Wednesday => "Wed",
                Weekday::Thursday => "Thu",
                Weekday::Friday => "Fri",
                Weekday::Saturday => "Sat",
            }
            .to_owned(),
            Self::TradingDay(
                TradingDayVariable::FlowWeekdayWeekend
                | TradingDayVariable::StockWeekdayWeekend { .. },
            ) => "Weekday".to_owned(),
            Self::Length(LengthEffect::LeapYear) => "Leap Year".to_owned(),
            Self::Length(LengthEffect::LengthOfPeriod) => match frequency {
                Frequency::Monthly => "Length-of-Month",
                Frequency::Quarterly => "Length-of-Quarter",
            }
            .to_owned(),
            Self::Holiday(HolidayVariable::Easter(effect)) => {
                format!("Easter[{}]", effect.window())
            }
            Self::Holiday(HolidayVariable::LaborDay(effect)) => {
                format!("Labor[{}]", effect.window())
            }
            Self::Holiday(HolidayVariable::Thanksgiving(effect)) => {
                format!("Thanksgiving[{}]", effect.offset())
            }
            Self::Outlier { kind, period } => {
                let prefix = match kind {
                    OutlierType::Additive => "AO",
                    OutlierType::LevelShift => "LS",
                    OutlierType::TemporaryChange => "TC",
                };
                dated_name(prefix, period, frequency)
            }
            Self::MissingValue { period } => dated_name("MV", period, frequency),
        }
    }
}

/// `prefix` followed by the year and month or quarter, such as `AO2000.Jan`.
fn dated_name(prefix: &str, period: Period, frequency: Frequency) -> String {
    let suffix = match frequency {
        Frequency::Monthly => [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][usize::from(period.subperiod() - 1)]
        .to_owned(),
        Frequency::Quarterly => period.subperiod().to_string(),
    };
    format!("{prefix}{}.{suffix}", period.year())
}

/// Scale used by returned calendar-effect series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CalendarEffectScale {
    /// Additive differences, whose identity value is zero.
    Difference,
    /// Multiplicative factors, whose identity value is one.
    Factor,
}

/// Calendar effects over the observed input span, each column a [`Series`]
/// starting at the input's first period.
#[derive(Debug, Clone, PartialEq)]
pub struct CalendarEffects {
    pub(crate) start: Period,
    pub(crate) scale: CalendarEffectScale,
    pub(crate) estimated: Vec<f64>,
    pub(crate) fixed_length: Vec<f64>,
    pub(crate) combined: Vec<f64>,
}

impl CalendarEffects {
    /// Whether effects are additive differences or multiplicative factors.
    #[must_use]
    pub const fn scale(&self) -> CalendarEffectScale {
        self.scale
    }

    /// Estimated calendar effect, one value per input observation.
    #[must_use]
    pub fn estimated(&self) -> Series<'_> {
        Series::new(self.start, &self.estimated)
    }

    /// Fixed leap-year prior factor over the input span, or the scale identity
    /// when absent.
    #[must_use]
    pub fn fixed_length(&self) -> Series<'_> {
        Series::new(self.start, &self.fixed_length)
    }

    /// Combined calendar adjustment withheld from the adjusted series, over
    /// the input span.
    #[must_use]
    pub fn combined(&self) -> Series<'_> {
        Series::new(self.start, &self.combined)
    }
}

/// Likelihood-based model-selection criteria.
#[derive(Debug, Clone, PartialEq)]
pub struct InformationCriteria {
    pub(crate) aic: f64,
    pub(crate) aicc: Option<f64>,
    pub(crate) hannan_quinn: f64,
    pub(crate) bic: f64,
}

impl InformationCriteria {
    /// Akaike information criterion.
    #[must_use]
    pub const fn aic(&self) -> f64 {
        self.aic
    }

    /// Finite-sample corrected AIC, when it is defined.
    #[must_use]
    pub const fn aicc(&self) -> Option<f64> {
        self.aicc
    }

    /// Hannan–Quinn information criterion.
    #[must_use]
    pub const fn hannan_quinn(&self) -> f64 {
        self.hannan_quinn
    }

    /// Bayesian information criterion.
    #[must_use]
    pub const fn bic(&self) -> f64 {
        self.bic
    }
}

/// Successful estimation iteration counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Convergence {
    pub(crate) overall_iterations: usize,
    pub(crate) nonlinear_iterations: usize,
}

impl Convergence {
    /// Number of iterative generalized least squares (IGLS) iterations.
    #[must_use]
    pub const fn overall_iterations(self) -> usize {
        self.overall_iterations
    }

    /// Cumulative nonlinear iterations.
    #[must_use]
    pub const fn nonlinear_iterations(self) -> usize {
        self.nonlinear_iterations
    }
}

/// Optimizer condition that stopped further likelihood improvement.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionCriterion {
    /// Improvement in the likelihood was below the tolerance.
    Likelihood,
    /// Parameter changes were below the tolerance.
    Parameters,
    /// The gradient criterion was met: the residuals are orthogonal to the
    /// Jacobian columns within the tolerance.
    Gradient,
}

display_phrases!(PrecisionCriterion {
    Likelihood => "likelihood-change",
    Parameters => "parameter-change",
    Gradient => "gradient",
});

#[cfg(test)]
mod tests {
    use super::RegressionVariable;
    use crate::Frequency;
    use crate::regarima::OutlierType;
    use crate::time::TimeAxis;

    #[test]
    fn outlier_regressors_are_named_by_type_and_date() {
        let monthly = TimeAxis::monthly(2005, 1).unwrap();
        let quarterly = TimeAxis::quarterly(2005, 1).unwrap();

        assert_eq!(
            RegressionVariable::Outlier {
                kind: OutlierType::Additive,
                period: monthly.start(),
            }
            .stable_name(Frequency::Monthly),
            "AO2005.Jan"
        );
        assert_eq!(
            RegressionVariable::Outlier {
                kind: OutlierType::LevelShift,
                period: monthly.period_at(11).unwrap(),
            }
            .stable_name(Frequency::Monthly),
            "LS2005.Dec"
        );
        assert_eq!(
            RegressionVariable::Outlier {
                kind: OutlierType::TemporaryChange,
                period: quarterly.period_at(2).unwrap(),
            }
            .stable_name(Frequency::Quarterly),
            "TC2005.3"
        );
        assert_eq!(
            RegressionVariable::MissingValue {
                period: monthly.period_at(2).unwrap(),
            }
            .stable_name(Frequency::Monthly),
            "MV2005.Mar"
        );
    }
}
