//! X-11 M and Q quality diagnostics.
use super::{Sample, SeasonalityTest};
use crate::Period;

/// X-11 M1 through M11 and Q quality measures (table F3).
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMeasures {
    pub(crate) m: [f64; 6],
    pub(crate) m6: Option<f64>,
    pub(crate) seasonal_evolution: Option<SeasonalEvolution>,
    pub(crate) q: f64,
    pub(crate) q_without_m2: f64,
    pub(crate) failed_measure_count: usize,
    pub(crate) m6_included_in_q: bool,
    pub(crate) observed_span: Sample,
    pub(crate) seasonal_history_span: Sample,
}
impl QualityMeasures {
    /// M1: relative contribution of the irregular over three-month (one-quarter)
    /// spans.
    #[must_use]
    pub const fn m1(&self) -> f64 {
        self.m[0]
    }
    /// M2: relative contribution of the irregular to the stationary part of the
    /// variance.
    #[must_use]
    pub const fn m2(&self) -> f64 {
        self.m[1]
    }
    /// M3: change in the irregular relative to the trend-cycle.
    #[must_use]
    pub const fn m3(&self) -> f64 {
        self.m[2]
    }
    /// M4: autocorrelation of the irregular, from the average duration of run.
    #[must_use]
    pub const fn m4(&self) -> f64 {
        self.m[3]
    }
    /// M5: months or quarters before the trend-cycle change exceeds the
    /// irregular change.
    #[must_use]
    pub const fn m5(&self) -> f64 {
        self.m[4]
    }
    /// Not computed under trend-only decomposition.
    #[must_use]
    pub const fn m6(&self) -> Option<f64> {
        self.m6
    }
    /// M7: moving seasonality relative to stable seasonality.
    #[must_use]
    pub const fn m7(&self) -> f64 {
        self.m[5]
    }
    /// M8 through M11. `None` for a trend-only decomposition, after a stable
    /// seasonal filter, or with fewer than six complete years.
    #[must_use]
    pub const fn seasonal_evolution(&self) -> Option<&SeasonalEvolution> {
        self.seasonal_evolution.as_ref()
    }
    /// Q, the weighted average of the M measures.
    #[must_use]
    pub const fn q(&self) -> f64 {
        self.q
    }
    /// Q2, the Q average without M2.
    #[must_use]
    pub const fn q_without_m2(&self) -> f64 {
        self.q_without_m2
    }
    /// Number of M measures at or above 1.
    #[must_use]
    pub const fn failed_measure_count(&self) -> usize {
        self.failed_measure_count
    }
    /// Whether M6 enters Q, which X-13ARIMA-SEATS allows only when every
    /// seasonal filter is 3x5.
    #[must_use]
    pub const fn m6_included_in_q(&self) -> bool {
        self.m6_included_in_q
    }
    /// Span of the observed series.
    #[must_use]
    pub const fn observed_span(&self) -> &Sample {
        &self.observed_span
    }
    /// Span of the seasonal factors that M8 through M11 read: from the start of
    /// the X-11 working series, backcasts included, through the last observation.
    #[must_use]
    pub const fn seasonal_history_span(&self) -> &Sample {
        &self.seasonal_history_span
    }
}

/// M8 through M11 seasonal evolution measures.
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalEvolution {
    pub(crate) m: [f64; 4],
}
impl SeasonalEvolution {
    /// M8: size of the fluctuations in the seasonal component over the whole
    /// series.
    #[must_use]
    pub const fn m8(&self) -> f64 {
        self.m[0]
    }
    /// M9: average linear movement in the seasonal component over the whole
    /// series.
    #[must_use]
    pub const fn m9(&self) -> f64 {
        self.m[1]
    }
    /// M10: size of the fluctuations in the seasonal component in recent years.
    #[must_use]
    pub const fn m10(&self) -> f64 {
        self.m[2]
    }
    /// M11: average linear movement in the seasonal component in recent years.
    #[must_use]
    pub const fn m11(&self) -> f64 {
        self.m[3]
    }
}

/// Why X-11 cannot compute the M and Q measures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum X11QualityUnavailableReason {
    /// The listed series are constant over their eligible values, so the
    /// measures are skipped.
    ConstantSeries {
        /// Series without variation.
        series: Vec<X11QualitySeries>,
    },
    /// Arithmetic produced a nonfinite value.
    NonFiniteArithmetic {
        /// Stage where the value became nonfinite.
        stage: X11QualityStage,
    },
    /// A multiplicative measure needs the logarithm of a nonpositive value.
    NonPositiveLogInput {
        /// Series holding the value.
        series: X11QualitySeries,
        /// Period of the value.
        period: Period,
    },
    /// A stage has too few values.
    InsufficientHistory {
        /// Stage that needs more values.
        stage: X11QualityStage,
        /// Number of values available.
        actual: usize,
        /// Number of values the stage needs.
        minimum: usize,
    },
    /// A stage has no eligible values.
    NoEligibleValues {
        /// Stage without eligible values.
        stage: X11QualityStage,
    },
    /// A stage divides by zero.
    ZeroDenominator {
        /// Stage with the zero denominator.
        stage: X11QualityStage,
    },
    /// A seasonality diagnostic the measures need is unavailable.
    PrerequisiteUnavailable {
        /// Seasonality stage that is unavailable.
        stage: SeasonalityTest,
    },
}
impl std::fmt::Display for X11QualityUnavailableReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstantSeries { series } => {
                formatter.write_str("constant series:")?;
                for (position, series) in series.iter().enumerate() {
                    let separator = if position == 0 { " " } else { ", " };
                    write!(formatter, "{separator}{series}")?;
                }
                Ok(())
            }
            Self::NonFiniteArithmetic { stage } => {
                write!(formatter, "{stage}: a value is not finite")
            }
            Self::NonPositiveLogInput { series, period } => write!(
                formatter,
                "the {series} at {period} is not positive, so it has no logarithm"
            ),
            Self::InsufficientHistory {
                stage,
                actual,
                minimum,
            } => write!(
                formatter,
                "{stage}: {actual} values, fewer than the {minimum} needed"
            ),
            Self::NoEligibleValues { stage } => write!(formatter, "{stage}: no eligible values"),
            Self::ZeroDenominator { stage } => write!(formatter, "{stage}: division by zero"),
            Self::PrerequisiteUnavailable { stage } => {
                write!(formatter, "the {stage} is unavailable")
            }
        }
    }
}
/// Series a quality measure reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum X11QualitySeries {
    /// Original series modified for extremes.
    ModifiedOriginal,
    /// Irregular component.
    Irregular,
    /// Trend-cycle.
    Trend,
    /// Original series.
    Original {
        /// Whether regARIMA outlier effects were removed first.
        outliers_removed: bool,
    },
    /// Modified original series with its linear trend removed.
    DetrendedOriginal,
}
impl std::fmt::Display for X11QualitySeries {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ModifiedOriginal => "modified original series",
            Self::Irregular => "irregular",
            Self::Trend => "trend-cycle",
            Self::Original {
                outliers_removed: false,
            } => "original series",
            Self::Original {
                outliers_removed: true,
            } => "original series without outlier effects",
            Self::DetrendedOriginal => "detrended modified original series",
        })
    }
}
/// Step of the M and Q computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum X11QualityStage {
    /// Final moving seasonality ratio.
    FinalMovingSeasonalityRatio,
    /// Changes in the prior adjustment factors.
    PriorChanges,
    /// Changes in the calendar effects.
    CalendarChanges,
    /// Changes in the irregular.
    IrregularChanges,
    /// Changes in the trend-cycle.
    TrendChanges,
    /// Changes in the modified original series.
    ModifiedOriginalChanges,
    /// Removing the linear trend from the modified original series.
    Detrending,
    /// Normalizing the component variances.
    VarianceNormalization,
    /// Months or quarters for cyclical dominance.
    MonthOrQuarterDominance,
    /// M8 through M11.
    SeasonalEvolution,
    /// M1.
    M1,
    /// M2.
    M2,
    /// M3.
    M3,
    /// M4.
    M4,
    /// M5.
    M5,
    /// M6.
    M6,
    /// M7.
    M7,
}
display_phrases!(X11QualityStage {
    FinalMovingSeasonalityRatio => "final moving seasonality ratio",
    PriorChanges => "changes in the prior adjustment factors",
    CalendarChanges => "changes in the calendar effects",
    IrregularChanges => "changes in the irregular",
    TrendChanges => "changes in the trend-cycle",
    ModifiedOriginalChanges => "changes in the modified original series",
    Detrending => "detrending the modified original series",
    VarianceNormalization => "normalizing the component variances",
    MonthOrQuarterDominance => "months or quarters for cyclical dominance",
    SeasonalEvolution => "M8 through M11",
    M1 => "M1",
    M2 => "M2",
    M3 => "M3",
    M4 => "M4",
    M5 => "M5",
    M6 => "M6",
    M7 => "M7",
});
