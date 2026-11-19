use super::{CalendarSigmaSelection, Mode, MsrSelection, TrendStage};
use crate::diagnostics::Unavailable;
use crate::{Frequency, Period, Series};

/// Easter date categories, in the order X-13ARIMA-SEATS reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EasterDateGroup {
    /// Easter on or before April 1.
    OnOrBeforeApril1,
    /// Easter on or after April 16.
    OnOrAfterApril16,
    /// Easter from April 2 through April 8.
    April2Through8,
    /// Easter from April 9 through April 15.
    April9Through15,
}

/// Whether X-11 could estimate the Easter factors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EasterStatus {
    /// The factors were estimated.
    Estimated,
    /// These date groups have no year in the sample, so the estimate
    /// is skipped and every factor is 1.
    MissingDateGroups(Vec<EasterDateGroup>),
}

/// Nonpositive trend replacement during the decomposition that estimates H1.
/// The stage is local to that preliminary decomposition, not the final pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EasterPreliminaryTrendWarning {
    pub(crate) stage: TrendStage,
    pub(crate) replacements: usize,
}

impl EasterPreliminaryTrendWarning {
    /// Trend stage that replaced values.
    #[must_use]
    pub fn stage(&self) -> TrendStage {
        self.stage
    }

    /// Number of nonpositive trend values replaced.
    #[must_use]
    pub fn replacements(&self) -> usize {
        self.replacements
    }
}

/// Result of the X-11 Easter adjustment: the factors (table H1), the status of
/// the estimate, and the warnings and selections of the preliminary X-11 run
/// it used.
///
/// Unless the regARIMA model has calendar regressors, the seasonally adjusted
/// series still contains this effect.
#[derive(Debug, Clone, PartialEq)]
pub struct EasterEffect {
    pub(crate) start: Period,
    pub(crate) factors: Vec<f64>,
    pub(crate) status: EasterStatus,
    pub(crate) preliminary_warnings: Vec<Unavailable>,
    pub(crate) preliminary_trend_warnings: Vec<EasterPreliminaryTrendWarning>,
    pub(crate) preliminary_msr_selection: Option<MsrSelection>,
    pub(crate) preliminary_calendar_sigma_selection: Option<CalendarSigmaSelection>,
}
impl EasterEffect {
    /// Easter factors (H1) over the observed span, dated from its first
    /// period.
    #[must_use]
    pub fn factors(&self) -> Series<'_> {
        Series::new(self.start, &self.factors)
    }
    /// Whether the factors were estimated.
    #[must_use]
    pub fn status(&self) -> &EasterStatus {
        &self.status
    }
    /// The residual-seasonality test of the preliminary seasonally adjusted
    /// series (D11), when X-13ARIMA-SEATS notes that it has no value: the
    /// original series is constant or the residual mean square is zero.
    #[must_use]
    pub fn preliminary_warnings(&self) -> &[Unavailable] {
        &self.preliminary_warnings
    }
    /// Trend replacements made in the preliminary decomposition, before the
    /// Easter effect is estimated, reported apart from
    /// [`preliminary_warnings`](Self::preliminary_warnings).
    #[must_use]
    pub fn preliminary_trend_warnings(&self) -> &[EasterPreliminaryTrendWarning] {
        &self.preliminary_trend_warnings
    }
    /// MSR seasonal filter selection in the preliminary decomposition, when
    /// it ran.
    #[must_use]
    pub fn preliminary_msr_selection(&self) -> Option<&MsrSelection> {
        self.preliminary_msr_selection.as_ref()
    }
    /// Calendar sigma selection in the preliminary decomposition, when it ran.
    #[must_use]
    pub fn preliminary_calendar_sigma_selection(&self) -> Option<&CalendarSigmaSelection> {
        self.preliminary_calendar_sigma_selection.as_ref()
    }
}

/// Why a request cannot use the X-11 Easter prior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum X11EasterConflict {
    /// Automatic transformation selection is requested.
    AutomaticTransformation,
    /// The regARIMA model also fits an Easter effect.
    FittedEaster,
    /// The series is not monthly.
    Frequency {
        /// Frequency of the series.
        actual: Frequency,
    },
    /// The X-11 mode is neither multiplicative nor pseudo-additive.
    Mode {
        /// Requested mode.
        actual: Mode,
    },
    /// Pseudo-additive X-11 needs fitted holidays retained in the adjusted
    /// series.
    PseudoRequiresHolidayRetention,
}

impl std::fmt::Display for X11EasterConflict {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AutomaticTransformation => {
                formatter.write_str("automatic transformation selection is requested")
            }
            Self::FittedEaster => {
                formatter.write_str("the regARIMA model also fits an Easter effect")
            }
            Self::Frequency { actual } => {
                write!(formatter, "it needs a monthly series, not a {actual} one")
            }
            Self::Mode { actual } => write!(
                formatter,
                "it needs multiplicative or pseudo-additive X-11, not {actual}"
            ),
            Self::PseudoRequiresHolidayRetention => {
                formatter.write_str("pseudo-additive X-11 needs the holiday effect retained")
            }
        }
    }
}
/// Step of the Easter factor estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11EasterStage {
    /// Looking up the Easter date.
    CalendarLookup,
    /// Reading the Easter indicator.
    IndicatorRead,
    /// Trimming extreme values.
    Trimming,
    /// Normalizing the factors.
    Normalization,
}

display_phrases!(X11EasterStage {
    CalendarLookup => "Easter date lookup",
    IndicatorRead => "Easter indicator read",
    Trimming => "trimming",
    Normalization => "normalization",
});
/// Why an Easter factor estimate is undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum X11EasterUnavailableReason {
    /// The date is outside the Easter date table.
    OutsideDateTable,
    /// Build 62 would read a value it never set, so its result is undefined.
    UninitializedOperand,
    /// A date group retains no values after trimming.
    EmptyRetainedGroup,
    /// A divisor is zero.
    ZeroDivisor,
    /// Arithmetic produced a nonfinite value.
    NonFiniteArithmetic,
}

display_phrases!(X11EasterUnavailableReason {
    OutsideDateTable => "the date is outside the Easter date table",
    UninitializedOperand => "Build 62 would read an uninitialized operand",
    EmptyRetainedGroup => "a retained date group is empty",
    ZeroDivisor => "a divisor is zero",
    NonFiniteArithmetic => "the arithmetic is not finite",
});
