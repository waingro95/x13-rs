//! What a call-table row asks for, as `tests/parity/specs.toml` states it.
//!
//! A configuration is independent of both implementations: `spec_file`
//! renders it for Build 62 and `rust_spec` builds the `x13::Spec` from it.

use serde::{Deserialize, Serialize};

use crate::{Frequency, OracleError};

pub(crate) mod regarima_request;

pub(crate) use regarima_request::RegArimaRequest;

/// The method of the final seasonal adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Method {
    X11,
    Seats,
}

const MAX_REGULAR_AR_ORDER: u8 = 3;
const MAX_REGULAR_DIFFERENCING_ORDER: u8 = 2;
const MAX_REGULAR_MA_ORDER: u8 = 3;
const MAX_SEASONAL_AR_ORDER: u8 = 2;
const MAX_SEASONAL_DIFFERENCING_ORDER: u8 = 1;
const MAX_SEASONAL_MA_ORDER: u8 = 2;
const MAX_EXPANDED_LAG: u16 = 36;
const MAX_EXTENSION_HORIZON: u16 = 120;
const MAX_EASTER_WINDOW: u8 = 25;
const MAX_LABOR_DAY_WINDOW: u8 = 25;
const MIN_THANKSGIVING_OFFSET: i8 = -8;
const MAX_THANKSGIVING_OFFSET: i8 = 17;

/// Requested Build 62 X-11 decomposition, independently of arithmetic mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11Decomposition {
    SeasonalAdjustment,
    TrendOnly,
}

/// Build 62 fixed-center estimator for X-11 extreme-value adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(
    clippy::enum_variant_names,
    reason = "the names are x13::x11::ExtremeEstimator's and the ones specs.toml spells"
)]
pub(crate) enum OracleX11ExtremeEstimator {
    StandardDeviation,
    MedianAbsoluteDeviation,
    TauAdjustedMedianAbsoluteDeviation,
    LogMedianAbsoluteDeviation,
    TauAdjustedLogMedianAbsoluteDeviation,
}

impl OracleX11ExtremeEstimator {
    pub(crate) const fn spec_value(self) -> &'static str {
        match self {
            Self::StandardDeviation => "std",
            Self::MedianAbsoluteDeviation => "wmad",
            Self::TauAdjustedMedianAbsoluteDeviation => "tau",
            Self::LogMedianAbsoluteDeviation => "wmadlog",
            Self::TauAdjustedLogMedianAbsoluteDeviation => "taulog",
        }
    }
}

/// Whether forecast/backcast extensions contribute to X-11 extreme scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11ExtremeEstimationSpan {
    WithExtensions,
    ObservedOnly,
}

/// Destination for fitted temporary-change effects in final X-11 components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11TemporaryChangeAllocation {
    Irregular,
    Trend,
}

/// Whether fitted holiday effects are removed from final X-11 components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11HolidayAdjustment {
    Remove,
    Retain,
}

/// Build 62 policy for seasonal filters when the SI span is short.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11ShortSeriesFilters {
    StableFallback,
    UseRequested,
}

/// Requested Build 62 calendar-specific extreme scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum OracleX11ExtremeScale {
    Automatic,
    RollingFiveYear,
    ByCalendarPeriod,
    Grouped { groups: OracleX11SigmaGroups },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11SigmaGroup {
    Selected,
    Complement,
}

/// Calendar order, independent of the series start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "frequency", content = "groups", rename_all = "snake_case")]
pub(crate) enum OracleX11SigmaGroups {
    Monthly([OracleX11SigmaGroup; 12]),
    Quarterly([OracleX11SigmaGroup; 4]),
}

impl OracleX11SigmaGroups {
    pub(crate) const fn frequency(self) -> Frequency {
        match self {
            Self::Monthly(_) => Frequency::Monthly,
            Self::Quarterly(_) => Frequency::Quarterly,
        }
    }

    pub(crate) fn selected_names(self) -> Vec<&'static str> {
        let (groups, names): (&[OracleX11SigmaGroup], &[&str]) = match &self {
            Self::Monthly(groups) => (
                groups,
                &[
                    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                    "dec",
                ],
            ),
            Self::Quarterly(groups) => (groups, &["q1", "q2", "q3", "q4"]),
        };
        groups
            .iter()
            .zip(names)
            .filter_map(|(group, name)| (*group == OracleX11SigmaGroup::Selected).then_some(*name))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum X11Mode {
    Additive,
    Multiplicative,
    LogAdditive,
    PseudoAdditive,
}

impl X11Mode {
    pub(crate) const fn spec_value(self) -> &'static str {
        match self {
            Self::Additive => "add",
            Self::Multiplicative => "mult",
            Self::LogAdditive => "logadd",
            Self::PseudoAdditive => "pseudoadd",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum X11SeasonalFilter {
    X11Default,
    S3x1,
    S3x3,
    S3x5,
    S3x9,
    S3x15,
    Stable,
    Msr,
}

/// Fixed calendar-order filters: January through December, or Q1 through Q4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11SeasonalFilters {
    Monthly([X11SeasonalFilter; 12]),
    Quarterly([X11SeasonalFilter; 4]),
}

impl OracleX11SeasonalFilters {
    pub(crate) const fn frequency(self) -> Frequency {
        match self {
            Self::Monthly(_) => Frequency::Monthly,
            Self::Quarterly(_) => Frequency::Quarterly,
        }
    }

    pub(crate) fn filters(&self) -> &[X11SeasonalFilter] {
        match self {
            Self::Monthly(filters) => filters,
            Self::Quarterly(filters) => filters,
        }
    }
}

impl X11SeasonalFilter {
    pub(crate) const fn spec_value(self) -> &'static str {
        match self {
            Self::X11Default => "x11default",
            Self::S3x1 => "s3x1",
            Self::S3x3 => "s3x3",
            Self::S3x5 => "s3x5",
            Self::S3x9 => "s3x9",
            Self::S3x15 => "s3x15",
            Self::Stable => "stable",
            Self::Msr => "msr",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(crate) struct OracleHendersonLength(u8);

impl OracleHendersonLength {
    pub(crate) fn new(terms: u8) -> Result<Self, OracleError> {
        if !(3..=101).contains(&terms) || terms.is_multiple_of(2) {
            return Err(invalid_configuration(format!(
                "Henderson length {terms} must be odd and within 3..=101"
            )));
        }
        Ok(Self(terms))
    }

    pub(crate) const fn terms(self) -> u8 {
        self.0
    }
}

impl<'de> Deserialize<'de> for OracleHendersonLength {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let terms = u8::deserialize(deserializer)?;
        Self::new(terms).map_err(serde::de::Error::custom)
    }
}

/// Requested endpoint I/C parameter; distinct from measured trend-selection I/C.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(transparent)]
pub(crate) struct OracleHendersonIcRatio(f64);

// Construction and deserialization exclude NaN, making equality reflexive.
impl Eq for OracleHendersonIcRatio {}

impl OracleHendersonIcRatio {
    pub(crate) fn new(value: f64) -> Result<Self, OracleError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_configuration(format!(
                "Henderson I/C ratio requires a finite positive value; got {value}"
            )));
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> f64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for OracleHendersonIcRatio {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(f64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "OracleX11TrendFilterSerde", into = "OracleX11TrendFilterSerde")]
pub(crate) enum OracleX11TrendFilter {
    Automatic,
    Henderson(OracleHendersonLength),
    AutomaticWithTrueSevenTerm,
    HendersonWithOptions {
        length: OracleHendersonLength,
        ic_ratio: Option<OracleHendersonIcRatio>,
        true_seven_term: bool,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum OracleX11TrendFilterSerde {
    Automatic {},
    Henderson {
        terms: OracleHendersonLength,
    },
    AutomaticWithTrueSevenTerm {},
    HendersonWithOptions {
        options: OracleHendersonOptionsSerde,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleHendersonOptionsSerde {
    length: OracleHendersonLength,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ic_ratio: Option<OracleHendersonIcRatio>,
    true_seven_term: bool,
}

impl From<OracleX11TrendFilterSerde> for OracleX11TrendFilter {
    fn from(value: OracleX11TrendFilterSerde) -> Self {
        match value {
            OracleX11TrendFilterSerde::Automatic {} => Self::Automatic,
            OracleX11TrendFilterSerde::Henderson { terms } => Self::Henderson(terms),
            OracleX11TrendFilterSerde::AutomaticWithTrueSevenTerm {} => {
                Self::AutomaticWithTrueSevenTerm
            }
            OracleX11TrendFilterSerde::HendersonWithOptions { options } => {
                Self::HendersonWithOptions {
                    length: options.length,
                    ic_ratio: options.ic_ratio,
                    true_seven_term: options.true_seven_term,
                }
            }
        }
    }
}

impl From<OracleX11TrendFilter> for OracleX11TrendFilterSerde {
    fn from(value: OracleX11TrendFilter) -> Self {
        match value {
            OracleX11TrendFilter::Automatic => Self::Automatic {},
            OracleX11TrendFilter::Henderson(terms) => Self::Henderson { terms },
            OracleX11TrendFilter::AutomaticWithTrueSevenTerm => Self::AutomaticWithTrueSevenTerm {},
            OracleX11TrendFilter::HendersonWithOptions {
                length,
                ic_ratio,
                true_seven_term,
            } => Self::HendersonWithOptions {
                options: OracleHendersonOptionsSerde {
                    length,
                    ic_ratio,
                    true_seven_term,
                },
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct OracleSigmaLimits {
    lower: f64,
    upper: f64,
}

impl OracleSigmaLimits {
    pub(crate) fn new(lower: f64, upper: f64) -> Result<Self, OracleError> {
        if !lower.is_finite() || !upper.is_finite() || lower <= 0.0 || lower > upper {
            return Err(invalid_configuration(format!(
                "sigma limits require finite bounds satisfying 0 < lower <= upper; got ({lower}, {upper})"
            )));
        }
        Ok(Self { lower, upper })
    }

    pub(crate) const fn lower(self) -> f64 {
        self.lower
    }

    pub(crate) const fn upper(self) -> f64 {
        self.upper
    }
}

#[derive(Deserialize)]
struct OracleSigmaLimitsSerde {
    lower: f64,
    upper: f64,
}

impl<'de> Deserialize<'de> for OracleSigmaLimits {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = OracleSigmaLimitsSerde::deserialize(deserializer)?;
        Self::new(value.lower, value.upper).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OracleArimaOrder {
    p: u8,
    d: u8,
    q: u8,
}

impl OracleArimaOrder {
    pub(crate) fn new(p: u8, d: u8, q: u8) -> Result<Self, OracleError> {
        if p > MAX_REGULAR_AR_ORDER {
            return Err(invalid_configuration(format!(
                "regular AR order {p} exceeds {MAX_REGULAR_AR_ORDER}"
            )));
        }
        if d > MAX_REGULAR_DIFFERENCING_ORDER {
            return Err(invalid_configuration(format!(
                "regular differencing order {d} exceeds {MAX_REGULAR_DIFFERENCING_ORDER}"
            )));
        }
        if q > MAX_REGULAR_MA_ORDER {
            return Err(invalid_configuration(format!(
                "regular MA order {q} exceeds {MAX_REGULAR_MA_ORDER}"
            )));
        }
        Ok(Self { p, d, q })
    }

    pub(crate) const fn p(self) -> u8 {
        self.p
    }

    pub(crate) const fn d(self) -> u8 {
        self.d
    }

    pub(crate) const fn q(self) -> u8 {
        self.q
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OracleArimaModel {
    regular: OracleArimaOrder,
    seasonal: OracleArimaOrder,
}

impl OracleArimaModel {
    pub(crate) const fn regular(self) -> OracleArimaOrder {
        self.regular
    }

    pub(crate) const fn seasonal(self) -> OracleArimaOrder {
        self.seasonal
    }

    fn validate_for_frequency(self, frequency: Frequency) -> Result<(), OracleError> {
        let period = u16::from(frequency.periods_per_year());
        let expanded_ar_lag = u16::from(self.regular.p) + u16::from(self.seasonal.p) * period;
        let expanded_differencing_lag =
            u16::from(self.regular.d) + u16::from(self.seasonal.d) * period;
        let expanded_ma_lag = u16::from(self.regular.q) + u16::from(self.seasonal.q) * period;
        validate_expanded_lag("AR", expanded_ar_lag)?;
        validate_expanded_lag("differencing", expanded_differencing_lag)?;
        validate_expanded_lag("MA", expanded_ma_lag)?;
        validate_regular_order(self.regular)?;
        validate_seasonal_order(self.seasonal)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleTransform {
    None,
    Log,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum OracleAutomaticModelTransform {
    Explicit {
        transform: OracleTransform,
    },
    Automatic {
        options: OracleAutomaticTransformOptions,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct OracleAutomaticTransformOptions {
    aic_difference: f64,
}

impl OracleAutomaticTransformOptions {
    pub(crate) fn new(aic_difference: f64) -> Result<Self, OracleError> {
        if !aic_difference.is_finite() {
            return Err(invalid_configuration(format!(
                "automatic-transform AIC difference must be finite; got {aic_difference}"
            )));
        }
        Ok(Self { aic_difference })
    }

    pub(crate) const fn aic_difference(self) -> f64 {
        self.aic_difference
    }
}

impl Default for OracleAutomaticTransformOptions {
    fn default() -> Self {
        Self {
            aic_difference: -2.0,
        }
    }
}

#[derive(Deserialize)]
struct OracleAutomaticTransformOptionsSerde {
    aic_difference: f64,
}

impl<'de> Deserialize<'de> for OracleAutomaticTransformOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = OracleAutomaticTransformOptionsSerde::deserialize(deserializer)?;
        Self::new(value.aic_difference).map_err(serde::de::Error::custom)
    }
}

impl OracleTransform {
    pub(crate) const fn spec_value(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Log => "log",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "periods", rename_all = "snake_case")]
pub(crate) enum OracleForecastHorizon {
    OneYear,
    Periods(u16),
}

impl OracleForecastHorizon {
    pub(crate) fn resolve(self, frequency: Frequency) -> Result<u16, OracleError> {
        let periods = match self {
            Self::OneYear => u16::from(frequency.periods_per_year()),
            Self::Periods(periods) => periods,
        };
        validate_horizon("forecast", periods)?;
        Ok(periods)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleTradingDayContrast {
    SixCoefficient,
    WeekdayWeekend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleFlowLengthTreatment {
    LeapYear,
    NoLengthEffect,
    LengthOfPeriod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OracleFlowTradingDay {
    contrast: OracleTradingDayContrast,
    length_treatment: OracleFlowLengthTreatment,
}

impl OracleFlowTradingDay {
    pub(crate) const fn contrast(self) -> OracleTradingDayContrast {
        self.contrast
    }

    pub(crate) const fn length_treatment(self) -> OracleFlowLengthTreatment {
        self.length_treatment
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "OracleStockTradingDaySerde",
    into = "OracleStockTradingDaySerde"
)]
pub(crate) struct OracleStockTradingDay {
    contrast: OracleTradingDayContrast,
    sample_day: u8,
}

impl OracleStockTradingDay {
    pub(crate) fn new(
        contrast: OracleTradingDayContrast,
        sample_day: u8,
    ) -> Result<Self, OracleError> {
        if !(1..=31).contains(&sample_day) {
            return Err(invalid_configuration(format!(
                "stock trading-day sample day {sample_day} is outside 1..=31"
            )));
        }
        Ok(Self {
            contrast,
            sample_day,
        })
    }

    pub(crate) const fn contrast(self) -> OracleTradingDayContrast {
        self.contrast
    }

    pub(crate) const fn sample_day(self) -> u8 {
        self.sample_day
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct OracleStockTradingDaySerde {
    contrast: OracleTradingDayContrast,
    sample_day: u8,
}

impl TryFrom<OracleStockTradingDaySerde> for OracleStockTradingDay {
    type Error = OracleError;

    fn try_from(value: OracleStockTradingDaySerde) -> Result<Self, Self::Error> {
        Self::new(value.contrast, value.sample_day)
    }
}

impl From<OracleStockTradingDay> for OracleStockTradingDaySerde {
    fn from(value: OracleStockTradingDay) -> Self {
        Self {
            contrast: value.contrast,
            sample_day: value.sample_day,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleLengthEffect {
    LeapYear,
    LengthOfPeriod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub(crate) enum OracleCalendarBasis {
    FlowTradingDay(OracleFlowTradingDay),
    StockTradingDay(OracleStockTradingDay),
    Length(OracleLengthEffect),
}

macro_rules! bounded_calendar_value {
    ($name:ident, $primitive:ty, $field:ident, $predicate:expr, $message:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
        #[serde(transparent)]
        pub(crate) struct $name($primitive);

        impl $name {
            pub(crate) fn new($field: $primitive) -> Result<Self, OracleError> {
                if !($predicate)($field) {
                    return Err(invalid_configuration(format!($message, $field)));
                }
                Ok(Self($field))
            }

            pub(crate) const fn $field(self) -> $primitive {
                self.0
            }
        }

        impl TryFrom<$primitive> for $name {
            type Error = OracleError;

            fn try_from(value: $primitive) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let value = <$primitive>::deserialize(deserializer)?;
                Self::new(value).map_err(serde::de::Error::custom)
            }
        }

        impl From<$name> for $primitive {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

bounded_calendar_value!(
    OracleEaster,
    u8,
    window,
    |value| value <= MAX_EASTER_WINDOW,
    "Easter window {} is outside 0..=25"
);
bounded_calendar_value!(
    OracleLaborDay,
    u8,
    window,
    |value| (1..=MAX_LABOR_DAY_WINDOW).contains(&value),
    "Labor Day window {} is outside 1..=25"
);
bounded_calendar_value!(
    OracleThanksgiving,
    i8,
    offset,
    |value| (MIN_THANKSGIVING_OFFSET..=-1).contains(&value)
        || (1..=MAX_THANKSGIVING_OFFSET).contains(&value),
    "Thanksgiving offset {} is outside -8..=-1 or 1..=17"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct OracleCalendarSpec {
    basis: Option<OracleCalendarBasis>,
    easter: Option<OracleEaster>,
    labor_day: Option<OracleLaborDay>,
    thanksgiving: Option<OracleThanksgiving>,
}

impl OracleCalendarSpec {
    pub(crate) const fn basis(self) -> Option<OracleCalendarBasis> {
        self.basis
    }

    pub(crate) const fn easter(self) -> Option<OracleEaster> {
        self.easter
    }

    pub(crate) const fn labor_day(self) -> Option<OracleLaborDay> {
        self.labor_day
    }

    pub(crate) const fn thanksgiving(self) -> Option<OracleThanksgiving> {
        self.thanksgiving
    }

    fn validate_for_frequency(self, frequency: Frequency) -> Result<(), OracleError> {
        if frequency != Frequency::Monthly {
            if matches!(self.basis, Some(OracleCalendarBasis::StockTradingDay(_))) {
                return Err(invalid_configuration(
                    "stock trading day requires monthly input".to_owned(),
                ));
            }
            if self.labor_day.is_some() {
                return Err(invalid_configuration(
                    "Labor Day requires monthly input".to_owned(),
                ));
            }
            if self.thanksgiving.is_some() {
                return Err(invalid_configuration(
                    "Thanksgiving requires monthly input".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) const fn is_empty(self) -> bool {
        self.basis.is_none()
            && self.easter.is_none()
            && self.labor_day.is_none()
            && self.thanksgiving.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleOutlierType {
    Additive,
    LevelShift,
    TemporaryChange,
}

impl OracleOutlierType {
    pub(crate) const fn spec_value(self) -> &'static str {
        match self {
            Self::Additive => "ao",
            Self::LevelShift => "ls",
            Self::TemporaryChange => "tc",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<OracleOutlierType>", into = "Vec<OracleOutlierType>")]
pub(crate) struct OracleOutlierTypes(u8);

impl OracleOutlierTypes {
    const ADDITIVE_BIT: u8 = 1;
    const LEVEL_SHIFT_BIT: u8 = 2;
    const TEMPORARY_CHANGE_BIT: u8 = 4;

    pub(crate) const ADDITIVE: Self = Self(Self::ADDITIVE_BIT);
    pub(crate) const LEVEL_SHIFT: Self = Self(Self::LEVEL_SHIFT_BIT);
    pub(crate) const TEMPORARY_CHANGE: Self = Self(Self::TEMPORARY_CHANGE_BIT);
    pub(crate) const ADDITIVE_AND_LEVEL_SHIFT: Self =
        Self(Self::ADDITIVE_BIT | Self::LEVEL_SHIFT_BIT);
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub(crate) const fn contains(self, outlier_type: OracleOutlierType) -> bool {
        let bit = match outlier_type {
            OracleOutlierType::Additive => Self::ADDITIVE_BIT,
            OracleOutlierType::LevelShift => Self::LEVEL_SHIFT_BIT,
            OracleOutlierType::TemporaryChange => Self::TEMPORARY_CHANGE_BIT,
        };
        self.0 & bit != 0
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = OracleOutlierType> {
        [
            OracleOutlierType::Additive,
            OracleOutlierType::LevelShift,
            OracleOutlierType::TemporaryChange,
        ]
        .into_iter()
        .filter(move |outlier_type| self.contains(*outlier_type))
    }
}

impl TryFrom<Vec<OracleOutlierType>> for OracleOutlierTypes {
    type Error = OracleError;

    fn try_from(values: Vec<OracleOutlierType>) -> Result<Self, Self::Error> {
        let mut result = None;
        for value in values {
            let next = match value {
                OracleOutlierType::Additive => Self::ADDITIVE,
                OracleOutlierType::LevelShift => Self::LEVEL_SHIFT,
                OracleOutlierType::TemporaryChange => Self::TEMPORARY_CHANGE,
            };
            result = Some(result.map_or(next, |current: Self| current.union(next)));
        }
        result.ok_or_else(|| invalid_configuration("outlier type set cannot be empty".to_owned()))
    }
}

impl From<OracleOutlierTypes> for Vec<OracleOutlierType> {
    fn from(value: OracleOutlierTypes) -> Self {
        value.iter().collect()
    }
}

/// Build 62 policy for selected fitted outliers in final X-11 output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum OracleX11FinalOutliers {
    Retain,
    Remove { types: OracleOutlierTypes },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleOutlierMethod {
    AddOne,
    AddAll,
}

impl OracleOutlierMethod {
    pub(crate) const fn spec_value(self) -> &'static str {
        match self {
            Self::AddOne => "addone",
            Self::AddAll => "addall",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub(crate) struct OracleOutlierCriticalValue(f64);

impl OracleOutlierCriticalValue {
    pub(crate) fn new(value: f64) -> Result<Self, OracleError> {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_configuration(
                "outlier critical value must be finite and positive".to_owned(),
            ));
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for OracleOutlierCriticalValue {
    type Error = OracleError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<OracleOutlierCriticalValue> for f64 {
    fn from(value: OracleOutlierCriticalValue) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub(crate) struct OracleTemporaryChangeRate(f64);

impl OracleTemporaryChangeRate {
    pub(crate) fn new(value: f64) -> Result<Self, OracleError> {
        if !value.is_finite() || value <= 0.0 || value >= 1.0 {
            return Err(invalid_configuration(
                "temporary-change rate must be finite and strictly between zero and one".to_owned(),
            ));
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for OracleTemporaryChangeRate {
    type Error = OracleError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<OracleTemporaryChangeRate> for f64 {
    fn from(value: OracleTemporaryChangeRate) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum OracleOutlierCriticalValues {
    Automatic,
    Shared {
        value: OracleOutlierCriticalValue,
    },
    PerType {
        additive: Option<OracleOutlierCriticalValue>,
        level_shift: Option<OracleOutlierCriticalValue>,
        temporary_change: Option<OracleOutlierCriticalValue>,
    },
}

impl OracleOutlierCriticalValues {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct OracleOutlierSpec {
    types: OracleOutlierTypes,
    method: OracleOutlierMethod,
    critical_values: OracleOutlierCriticalValues,
    temporary_change_rate: Option<OracleTemporaryChangeRate>,
}

impl Default for OracleOutlierSpec {
    fn default() -> Self {
        Self {
            types: OracleOutlierTypes::ADDITIVE_AND_LEVEL_SHIFT,
            method: OracleOutlierMethod::AddOne,
            critical_values: OracleOutlierCriticalValues::Automatic,
            temporary_change_rate: None,
        }
    }
}

impl OracleOutlierSpec {
    pub(crate) const fn types(self) -> OracleOutlierTypes {
        self.types
    }

    pub(crate) const fn method(self) -> OracleOutlierMethod {
        self.method
    }

    pub(crate) const fn critical_values(self) -> OracleOutlierCriticalValues {
        self.critical_values
    }

    pub(crate) const fn temporary_change_rate(self) -> Option<OracleTemporaryChangeRate> {
        self.temporary_change_rate
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11SeasonalShrinkage {
    #[default]
    None,
    Global,
    Local,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OracleX11Easter {
    #[default]
    None,
    Estimate,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum OracleConfiguration {
    /// Common preprocessing for SEATS and pending regression features.
    RegArima {
        method: Method,
        request: RegArimaRequest,
    },
    DirectX11 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decomposition: Option<OracleX11Decomposition>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_shrinkage: Option<OracleX11SeasonalShrinkage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legacy_easter: Option<OracleX11Easter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_outliers: Option<OracleX11FinalOutliers>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temporary_change_allocation: Option<OracleX11TemporaryChangeAllocation>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        holiday_adjustment: Option<OracleX11HolidayAdjustment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_series_filters: Option<OracleX11ShortSeriesFilters>,
        mode: X11Mode,
        seasonal_filter: Option<X11SeasonalFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_filters: Option<OracleX11SeasonalFilters>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend_filter: Option<OracleX11TrendFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sigma_limits: Option<OracleSigmaLimits>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimation_span: Option<OracleX11ExtremeEstimationSpan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_scale: Option<OracleX11ExtremeScale>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimator: Option<OracleX11ExtremeEstimator>,
    },
    AutomaticModelX11 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decomposition: Option<OracleX11Decomposition>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_shrinkage: Option<OracleX11SeasonalShrinkage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legacy_easter: Option<OracleX11Easter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_outliers: Option<OracleX11FinalOutliers>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temporary_change_allocation: Option<OracleX11TemporaryChangeAllocation>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        holiday_adjustment: Option<OracleX11HolidayAdjustment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_series_filters: Option<OracleX11ShortSeriesFilters>,
        transform: OracleAutomaticModelTransform,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mode: Option<X11Mode>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        calendar: Option<OracleCalendarSpec>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        outliers: Option<OracleOutlierSpec>,
        forecast_horizon: OracleForecastHorizon,
        backcast_horizon: u16,
        seasonal_filter: Option<X11SeasonalFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_filters: Option<OracleX11SeasonalFilters>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend_filter: Option<OracleX11TrendFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sigma_limits: Option<OracleSigmaLimits>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimation_span: Option<OracleX11ExtremeEstimationSpan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_scale: Option<OracleX11ExtremeScale>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimator: Option<OracleX11ExtremeEstimator>,
    },
    ExplicitRegArimaX11 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decomposition: Option<OracleX11Decomposition>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_shrinkage: Option<OracleX11SeasonalShrinkage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legacy_easter: Option<OracleX11Easter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_outliers: Option<OracleX11FinalOutliers>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temporary_change_allocation: Option<OracleX11TemporaryChangeAllocation>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        holiday_adjustment: Option<OracleX11HolidayAdjustment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_series_filters: Option<OracleX11ShortSeriesFilters>,
        transform: OracleTransform,
        model: OracleArimaModel,
        include_constant: bool,
        forecast_horizon: OracleForecastHorizon,
        backcast_horizon: u16,
        mode: X11Mode,
        seasonal_filter: Option<X11SeasonalFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_filters: Option<OracleX11SeasonalFilters>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend_filter: Option<OracleX11TrendFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sigma_limits: Option<OracleSigmaLimits>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimation_span: Option<OracleX11ExtremeEstimationSpan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_scale: Option<OracleX11ExtremeScale>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimator: Option<OracleX11ExtremeEstimator>,
    },
    AutomaticTransformRegArimaX11 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decomposition: Option<OracleX11Decomposition>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_shrinkage: Option<OracleX11SeasonalShrinkage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legacy_easter: Option<OracleX11Easter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_outliers: Option<OracleX11FinalOutliers>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temporary_change_allocation: Option<OracleX11TemporaryChangeAllocation>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        holiday_adjustment: Option<OracleX11HolidayAdjustment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_series_filters: Option<OracleX11ShortSeriesFilters>,
        #[serde(default)]
        automatic_transform: OracleAutomaticTransformOptions,
        model: OracleArimaModel,
        include_constant: bool,
        calendar: Option<OracleCalendarSpec>,
        outliers: Option<OracleOutlierSpec>,
        forecast_horizon: OracleForecastHorizon,
        backcast_horizon: u16,
        seasonal_filter: Option<X11SeasonalFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_filters: Option<OracleX11SeasonalFilters>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend_filter: Option<OracleX11TrendFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sigma_limits: Option<OracleSigmaLimits>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimation_span: Option<OracleX11ExtremeEstimationSpan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_scale: Option<OracleX11ExtremeScale>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimator: Option<OracleX11ExtremeEstimator>,
    },
    CalendarOutlierRegArimaX11 {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decomposition: Option<OracleX11Decomposition>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_shrinkage: Option<OracleX11SeasonalShrinkage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        legacy_easter: Option<OracleX11Easter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        final_outliers: Option<OracleX11FinalOutliers>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temporary_change_allocation: Option<OracleX11TemporaryChangeAllocation>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        holiday_adjustment: Option<OracleX11HolidayAdjustment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        short_series_filters: Option<OracleX11ShortSeriesFilters>,
        transform: OracleTransform,
        model: OracleArimaModel,
        include_constant: bool,
        calendar: Option<OracleCalendarSpec>,
        outliers: Option<OracleOutlierSpec>,
        forecast_horizon: OracleForecastHorizon,
        backcast_horizon: u16,
        mode: X11Mode,
        seasonal_filter: Option<X11SeasonalFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seasonal_filters: Option<OracleX11SeasonalFilters>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trend_filter: Option<OracleX11TrendFilter>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sigma_limits: Option<OracleSigmaLimits>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimation_span: Option<OracleX11ExtremeEstimationSpan>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_scale: Option<OracleX11ExtremeScale>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extreme_estimator: Option<OracleX11ExtremeEstimator>,
    },
}

impl OracleConfiguration {
    pub(crate) const fn legacy_easter(self) -> Option<OracleX11Easter> {
        match self {
            Self::DirectX11 { legacy_easter, .. }
            | Self::AutomaticModelX11 { legacy_easter, .. }
            | Self::ExplicitRegArimaX11 { legacy_easter, .. }
            | Self::AutomaticTransformRegArimaX11 { legacy_easter, .. }
            | Self::CalendarOutlierRegArimaX11 { legacy_easter, .. } => legacy_easter,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn decomposition(self) -> Option<OracleX11Decomposition> {
        match self {
            Self::DirectX11 { decomposition, .. }
            | Self::AutomaticModelX11 { decomposition, .. }
            | Self::ExplicitRegArimaX11 { decomposition, .. }
            | Self::AutomaticTransformRegArimaX11 { decomposition, .. }
            | Self::CalendarOutlierRegArimaX11 { decomposition, .. } => decomposition,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn is_trend_only(self) -> bool {
        matches!(
            self.decomposition(),
            Some(OracleX11Decomposition::TrendOnly)
        )
    }

    pub(crate) const fn seasonal_shrinkage(self) -> Option<OracleX11SeasonalShrinkage> {
        match self {
            Self::DirectX11 {
                seasonal_shrinkage, ..
            }
            | Self::AutomaticModelX11 {
                seasonal_shrinkage, ..
            }
            | Self::ExplicitRegArimaX11 {
                seasonal_shrinkage, ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                seasonal_shrinkage, ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                seasonal_shrinkage, ..
            } => seasonal_shrinkage,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn short_series_filters(self) -> Option<OracleX11ShortSeriesFilters> {
        match self {
            Self::DirectX11 {
                short_series_filters,
                ..
            }
            | Self::AutomaticModelX11 {
                short_series_filters,
                ..
            }
            | Self::ExplicitRegArimaX11 {
                short_series_filters,
                ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                short_series_filters,
                ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                short_series_filters,
                ..
            } => short_series_filters,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn holiday_adjustment(self) -> Option<OracleX11HolidayAdjustment> {
        match self {
            Self::DirectX11 {
                holiday_adjustment, ..
            }
            | Self::AutomaticModelX11 {
                holiday_adjustment, ..
            }
            | Self::ExplicitRegArimaX11 {
                holiday_adjustment, ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                holiday_adjustment, ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                holiday_adjustment, ..
            } => holiday_adjustment,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn temporary_change_allocation(
        self,
    ) -> Option<OracleX11TemporaryChangeAllocation> {
        match self {
            Self::DirectX11 {
                temporary_change_allocation,
                ..
            }
            | Self::AutomaticModelX11 {
                temporary_change_allocation,
                ..
            }
            | Self::ExplicitRegArimaX11 {
                temporary_change_allocation,
                ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                temporary_change_allocation,
                ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                temporary_change_allocation,
                ..
            } => temporary_change_allocation,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn final_outliers(self) -> Option<OracleX11FinalOutliers> {
        match self {
            Self::DirectX11 { final_outliers, .. }
            | Self::AutomaticModelX11 { final_outliers, .. }
            | Self::ExplicitRegArimaX11 { final_outliers, .. }
            | Self::AutomaticTransformRegArimaX11 { final_outliers, .. }
            | Self::CalendarOutlierRegArimaX11 { final_outliers, .. } => final_outliers,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn extreme_estimator(self) -> Option<OracleX11ExtremeEstimator> {
        match self {
            Self::DirectX11 {
                extreme_estimator, ..
            }
            | Self::AutomaticModelX11 {
                extreme_estimator, ..
            }
            | Self::ExplicitRegArimaX11 {
                extreme_estimator, ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                extreme_estimator, ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                extreme_estimator, ..
            } => extreme_estimator,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn extreme_scale(self) -> Option<OracleX11ExtremeScale> {
        match self {
            Self::DirectX11 { extreme_scale, .. }
            | Self::AutomaticModelX11 { extreme_scale, .. }
            | Self::ExplicitRegArimaX11 { extreme_scale, .. }
            | Self::AutomaticTransformRegArimaX11 { extreme_scale, .. }
            | Self::CalendarOutlierRegArimaX11 { extreme_scale, .. } => extreme_scale,
            Self::RegArima { .. } => None,
        }
    }

    pub(crate) const fn requests_x11_extreme_weights(self) -> bool {
        matches!(
            self,
            Self::DirectX11 {
                mode: X11Mode::PseudoAdditive,
                ..
            } | Self::ExplicitRegArimaX11 {
                mode: X11Mode::PseudoAdditive,
                ..
            } | Self::CalendarOutlierRegArimaX11 {
                mode: X11Mode::PseudoAdditive,
                ..
            } | Self::AutomaticModelX11 {
                mode: Some(X11Mode::PseudoAdditive),
                ..
            }
        ) || self.final_outliers().is_some()
            || self.temporary_change_allocation().is_some()
            || self.holiday_adjustment().is_some()
            || self.short_series_filters().is_some()
            || self.extreme_estimator().is_some()
            || requests_x11_extreme_weights(self.extreme_estimation_span(), self.extreme_scale())
    }

    pub(crate) const fn requests_automatic_calendar_sigma(self) -> bool {
        matches!(self.extreme_scale(), Some(OracleX11ExtremeScale::Automatic))
    }

    pub(crate) const fn extreme_estimation_span(self) -> Option<OracleX11ExtremeEstimationSpan> {
        match self {
            Self::DirectX11 {
                extreme_estimation_span,
                ..
            }
            | Self::AutomaticModelX11 {
                extreme_estimation_span,
                ..
            }
            | Self::ExplicitRegArimaX11 {
                extreme_estimation_span,
                ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                extreme_estimation_span,
                ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                extreme_estimation_span,
                ..
            } => extreme_estimation_span,
            Self::RegArima { .. } => None,
        }
    }
    pub(crate) fn requests_msr(&self) -> bool {
        self.msr_filters().is_some_and(|(uniform, filters)| {
            filters.map_or(uniform == Some(X11SeasonalFilter::Msr), |filters| {
                filters.filters().contains(&X11SeasonalFilter::Msr)
            })
        })
    }

    fn msr_filters(&self) -> Option<(Option<X11SeasonalFilter>, Option<OracleX11SeasonalFilters>)> {
        match *self {
            Self::DirectX11 {
                seasonal_filter,
                seasonal_filters,
                ..
            }
            | Self::ExplicitRegArimaX11 {
                seasonal_filter,
                seasonal_filters,
                ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                seasonal_filter,
                seasonal_filters,
                ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                seasonal_filter,
                seasonal_filters,
                ..
            }
            | Self::AutomaticModelX11 {
                seasonal_filter,
                seasonal_filters,
                ..
            } => Some((seasonal_filter, seasonal_filters)),
            Self::RegArima { .. } => None,
        }
    }

    #[cfg(test)]
    pub(crate) const fn direct_x11() -> Self {
        Self::DirectX11 {
            final_outliers: None,
            temporary_change_allocation: None,
            holiday_adjustment: None,
            short_series_filters: None,
            decomposition: None,
            seasonal_shrinkage: None,
            legacy_easter: None,
            mode: X11Mode::Additive,
            seasonal_filter: Some(X11SeasonalFilter::S3x5),
            seasonal_filters: None,
            trend_filter: None,
            extreme_estimation_span: None,
            extreme_scale: None,
            extreme_estimator: None,
            sigma_limits: None,
        }
    }

    pub(crate) const fn method(self) -> Method {
        match self {
            Self::RegArima { method, .. } => method,
            Self::DirectX11 { .. }
            | Self::AutomaticModelX11 { .. }
            | Self::ExplicitRegArimaX11 { .. }
            | Self::AutomaticTransformRegArimaX11 { .. }
            | Self::CalendarOutlierRegArimaX11 { .. } => Method::X11,
        }
    }

    pub(crate) fn validate_for_frequency(self, frequency: Frequency) -> Result<(), OracleError> {
        if let Some((uniform, calendar)) = self.msr_filters() {
            if self.is_trend_only() {
                if uniform.is_some() || calendar.is_some() {
                    return Err(invalid_configuration(
                        "trend-only forbids explicit seasonal filters".into(),
                    ));
                }
            } else if uniform.is_none() {
                return Err(invalid_configuration(
                    "ordinary X11 requires an explicit oracle seasonal filter".into(),
                ));
            }
        }
        if let Self::AutomaticModelX11 {
            mode: Some(_),
            transform: OracleAutomaticModelTransform::Automatic { .. },
            ..
        } = self
        {
            return Err(invalid_configuration(
                "automatic transformation cannot have an explicit X-11 mode".into(),
            ));
        }
        if let Some(OracleX11ExtremeScale::Grouped { groups }) = self.extreme_scale()
            && groups.frequency() != frequency
        {
            return Err(invalid_configuration(format!(
                "extreme scale frequency {:?} does not match series frequency {frequency:?}",
                groups.frequency()
            )));
        }
        let filters = match self {
            Self::RegArima { .. } => None,
            Self::DirectX11 {
                seasonal_filters, ..
            }
            | Self::AutomaticModelX11 {
                seasonal_filters, ..
            }
            | Self::ExplicitRegArimaX11 {
                seasonal_filters, ..
            }
            | Self::AutomaticTransformRegArimaX11 {
                seasonal_filters, ..
            }
            | Self::CalendarOutlierRegArimaX11 {
                seasonal_filters, ..
            } => seasonal_filters,
        };
        if let Some(filters) = filters
            && filters.frequency() != frequency
        {
            return Err(invalid_configuration(format!(
                "seasonal filter frequency {:?} does not match series frequency {frequency:?}",
                filters.frequency()
            )));
        }
        if let Self::CalendarOutlierRegArimaX11 {
            calendar, outliers, ..
        } = self
            && calendar.is_none_or(OracleCalendarSpec::is_empty)
            && outliers.is_none()
        {
            return Err(invalid_configuration(
                "calendar/outlier regARIMA requires calendar effects or outlier detection"
                    .to_owned(),
            ));
        }
        if let Self::AutomaticModelX11 {
            calendar,
            forecast_horizon,
            backcast_horizon,
            ..
        } = self
        {
            if let Some(calendar) = calendar {
                if calendar.is_empty() {
                    return Err(invalid_configuration(
                        "automatic-model calendar request is empty".to_owned(),
                    ));
                }
                calendar.validate_for_frequency(frequency)?;
            }
            forecast_horizon.resolve(frequency)?;
            validate_horizon("backcast", backcast_horizon)?;
        }
        if let Some(configuration) = self.regarima() {
            configuration.model.validate_for_frequency(frequency)?;
            configuration.forecast_horizon.resolve(frequency)?;
            validate_horizon("backcast", configuration.backcast_horizon)?;
            if let Some(calendar) = configuration.calendar {
                calendar.validate_for_frequency(frequency)?;
            }
        }
        Ok(())
    }

    pub(crate) const fn regarima(self) -> Option<OracleRegArimaConfiguration> {
        let decomposition = self.decomposition();
        let seasonal_shrinkage = self.seasonal_shrinkage();
        let final_outliers = self.final_outliers();
        let temporary_change_allocation = self.temporary_change_allocation();
        let holiday_adjustment = self.holiday_adjustment();
        let short_series_filters = self.short_series_filters();
        let extreme_estimation_span = self.extreme_estimation_span();
        let extreme_scale = self.extreme_scale();
        let extreme_estimator = self.extreme_estimator();
        match self {
            Self::ExplicitRegArimaX11 {
                transform,
                model,
                include_constant,
                forecast_horizon,
                backcast_horizon,
                mode,
                seasonal_filter,
                seasonal_filters,
                trend_filter,
                sigma_limits,
                ..
            } => Some(OracleRegArimaConfiguration {
                decomposition,
                seasonal_shrinkage,
                final_outliers,
                temporary_change_allocation,
                holiday_adjustment,
                short_series_filters,
                extreme_estimation_span,
                extreme_scale,
                extreme_estimator,
                transform,
                automatic_transform: None,
                model,
                include_constant,
                calendar: None,
                outliers: None,
                forecast_horizon,
                backcast_horizon,
                mode,
                seasonal_filter,
                seasonal_filters,
                trend_filter,
                sigma_limits,
            }),
            Self::AutomaticTransformRegArimaX11 {
                automatic_transform,
                model,
                include_constant,
                calendar,
                outliers,
                forecast_horizon,
                backcast_horizon,
                seasonal_filter,
                seasonal_filters,
                trend_filter,
                sigma_limits,
                ..
            } => Some(OracleRegArimaConfiguration {
                decomposition,
                final_outliers,
                temporary_change_allocation,
                holiday_adjustment,
                short_series_filters,
                extreme_estimation_span,
                extreme_scale,
                extreme_estimator,
                transform: OracleTransform::None,
                automatic_transform: Some(automatic_transform),
                seasonal_shrinkage,
                model,
                include_constant,
                calendar,
                outliers,
                forecast_horizon,
                backcast_horizon,
                mode: X11Mode::Additive,
                seasonal_filter,
                seasonal_filters,
                trend_filter,
                sigma_limits,
            }),
            Self::CalendarOutlierRegArimaX11 {
                transform,
                model,
                include_constant,
                calendar,
                outliers,
                forecast_horizon,
                backcast_horizon,
                mode,
                seasonal_filter,
                seasonal_filters,
                trend_filter,
                sigma_limits,
                ..
            } => Some(OracleRegArimaConfiguration {
                decomposition,
                seasonal_shrinkage,
                final_outliers,
                temporary_change_allocation,
                holiday_adjustment,
                short_series_filters,
                extreme_estimation_span,
                extreme_scale,
                extreme_estimator,
                transform,
                automatic_transform: None,
                model,
                include_constant,
                calendar,
                outliers,
                forecast_horizon,
                backcast_horizon,
                mode,
                seasonal_filter,
                seasonal_filters,
                trend_filter,
                sigma_limits,
            }),
            Self::RegArima { .. } | Self::DirectX11 { .. } | Self::AutomaticModelX11 { .. } => None,
        }
    }

    pub(crate) fn is_calendar_outlier_regarima(self) -> bool {
        match self {
            Self::CalendarOutlierRegArimaX11 { .. } => true,
            Self::AutomaticTransformRegArimaX11 {
                calendar, outliers, ..
            } => calendar.is_some_and(|calendar| !calendar.is_empty()) || outliers.is_some(),
            _ => false,
        }
    }

    pub(crate) const fn is_automatic_model(self) -> bool {
        matches!(self, Self::AutomaticModelX11 { .. })
    }

    pub(crate) const fn automatic_model_calendar(self) -> Option<OracleCalendarSpec> {
        match self {
            Self::AutomaticModelX11 { calendar, .. } => calendar,
            _ => None,
        }
    }

    pub(crate) const fn automatic_model_outliers(self) -> Option<OracleOutlierSpec> {
        match self {
            Self::AutomaticModelX11 { outliers, .. } => outliers,
            _ => None,
        }
    }
}

pub(crate) const fn requests_x11_extreme_weights(
    span: Option<OracleX11ExtremeEstimationSpan>,
    scale: Option<OracleX11ExtremeScale>,
) -> bool {
    span.is_some() || scale.is_some()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OracleRegArimaConfiguration {
    pub(crate) decomposition: Option<OracleX11Decomposition>,
    pub(crate) seasonal_shrinkage: Option<OracleX11SeasonalShrinkage>,
    pub(crate) final_outliers: Option<OracleX11FinalOutliers>,
    pub(crate) temporary_change_allocation: Option<OracleX11TemporaryChangeAllocation>,
    pub(crate) holiday_adjustment: Option<OracleX11HolidayAdjustment>,
    pub(crate) short_series_filters: Option<OracleX11ShortSeriesFilters>,
    pub(crate) transform: OracleTransform,
    pub(crate) automatic_transform: Option<OracleAutomaticTransformOptions>,
    pub(crate) model: OracleArimaModel,
    pub(crate) include_constant: bool,
    pub(crate) calendar: Option<OracleCalendarSpec>,
    pub(crate) outliers: Option<OracleOutlierSpec>,
    pub(crate) forecast_horizon: OracleForecastHorizon,
    pub(crate) backcast_horizon: u16,
    pub(crate) mode: X11Mode,
    pub(crate) seasonal_filter: Option<X11SeasonalFilter>,
    pub(crate) seasonal_filters: Option<OracleX11SeasonalFilters>,
    pub(crate) trend_filter: Option<OracleX11TrendFilter>,
    pub(crate) sigma_limits: Option<OracleSigmaLimits>,
    pub(crate) extreme_estimation_span: Option<OracleX11ExtremeEstimationSpan>,
    pub(crate) extreme_scale: Option<OracleX11ExtremeScale>,
    pub(crate) extreme_estimator: Option<OracleX11ExtremeEstimator>,
}

fn validate_regular_order(order: OracleArimaOrder) -> Result<(), OracleError> {
    OracleArimaOrder::new(order.p, order.d, order.q).map(|_| ())
}

fn validate_seasonal_order(order: OracleArimaOrder) -> Result<(), OracleError> {
    if order.p > MAX_SEASONAL_AR_ORDER {
        return Err(invalid_configuration(format!(
            "seasonal AR order {} exceeds {MAX_SEASONAL_AR_ORDER}",
            order.p
        )));
    }
    if order.d > MAX_SEASONAL_DIFFERENCING_ORDER {
        return Err(invalid_configuration(format!(
            "seasonal differencing order {} exceeds {MAX_SEASONAL_DIFFERENCING_ORDER}",
            order.d
        )));
    }
    if order.q > MAX_SEASONAL_MA_ORDER {
        return Err(invalid_configuration(format!(
            "seasonal MA order {} exceeds {MAX_SEASONAL_MA_ORDER}",
            order.q
        )));
    }
    Ok(())
}

fn validate_expanded_lag(name: &'static str, lag: u16) -> Result<(), OracleError> {
    if lag > MAX_EXPANDED_LAG {
        return Err(invalid_configuration(format!(
            "expanded {name} lag {lag} exceeds {MAX_EXPANDED_LAG}"
        )));
    }
    Ok(())
}

fn validate_horizon(name: &'static str, periods: u16) -> Result<(), OracleError> {
    if periods > MAX_EXTENSION_HORIZON {
        return Err(invalid_configuration(format!(
            "{name} horizon {periods} exceeds {MAX_EXTENSION_HORIZON}"
        )));
    }
    Ok(())
}

fn invalid_configuration(message: String) -> OracleError {
    OracleError::InvalidConfiguration(message)
}
