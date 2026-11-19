//! Spectral peak diagnostics: the peaks of one autoregressive spectrum, the
//! sample it is estimated from, and why a spectrum has no peaks to report.
use crate::Period;

/// Sample a spectrum is estimated from.
///
/// The spectrum of the regARIMA residuals takes no logarithms or differences,
/// so its input span is the sample itself.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpectralSample {
    pub(crate) input_first: Period,
    pub(crate) input_last: Period,
    pub(crate) input_count: usize,
    pub(crate) first: Period,
    pub(crate) last: Period,
    pub(crate) count: usize,
    pub(crate) difference_order: usize,
    pub(crate) logarithmic_input: bool,
}
impl SpectralSample {
    /// First period of the series before differencing.
    #[must_use]
    pub const fn input_first(&self) -> Period {
        self.input_first
    }
    /// Last period of the series before differencing.
    #[must_use]
    pub const fn input_last(&self) -> Period {
        self.input_last
    }
    /// Number of values before differencing.
    #[must_use]
    pub const fn input_count(&self) -> usize {
        self.input_count
    }
    /// First period of the differenced sample.
    #[must_use]
    pub const fn first(&self) -> Period {
        self.first
    }
    /// Last period of the differenced sample.
    #[must_use]
    pub const fn last(&self) -> Period {
        self.last
    }
    /// Number of values in the differenced sample.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
    /// Number of differences taken.
    #[must_use]
    pub const fn difference_order(&self) -> usize {
        self.difference_order
    }
    /// Whether the series was log-transformed before differencing.
    #[must_use]
    pub const fn logarithmic_input(&self) -> bool {
        self.logarithmic_input
    }
}

/// Visually significant peaks of one spectrum.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpectralPeaks {
    pub(crate) sample: SpectralSample,
    pub(crate) ar_order: usize,
    pub(crate) seasonal_count: usize,
    pub(crate) trading_day: SpectralTradingDayPeaks,
}
impl SpectralPeaks {
    /// Sample the spectrum was estimated from.
    #[must_use]
    pub const fn sample(&self) -> &SpectralSample {
        &self.sample
    }
    /// Order of the autoregressive spectrum.
    #[must_use]
    pub const fn ar_order(&self) -> usize {
        self.ar_order
    }
    /// Number of seasonal peaks.
    #[must_use]
    pub const fn seasonal_count(&self) -> usize {
        self.seasonal_count
    }
    /// Number of trading-day peaks. `None` when the differenced sample is too
    /// short to test for them.
    #[must_use]
    pub const fn trading_day_count(&self) -> Option<usize> {
        match self.trading_day {
            SpectralTradingDayPeaks::Tested { count } => Some(count),
            SpectralTradingDayPeaks::NotTestedInsufficientSample { .. } => None,
        }
    }
}

/// Kind of spectral peak a warning reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpectralPeakKind {
    /// Seasonal peaks.
    Seasonal,
    /// Trading-day peaks.
    TradingDay,
    /// Seasonal and trading-day peaks.
    Both,
}
display_phrases!(SpectralPeakKind {
    Seasonal => "seasonal",
    TradingDay => "trading-day",
    Both => "seasonal and trading-day",
});

/// Trading-day peak count, or why it was not tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpectralTradingDayPeaks {
    /// The differenced sample has 60 or fewer values, too few for the
    /// trading-day frequencies.
    NotTestedInsufficientSample {
        /// Number of differenced values.
        count: usize,
    },
    /// Number of trading-day peaks.
    Tested {
        /// Number of peaks.
        count: usize,
    },
}

/// Series whose spectrum is examined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpectralSeries {
    /// Prior-adjusted original series, the first X-11 spectrum (table sp0).
    AdjustedOriginal,
    /// Seasonally adjusted series modified for extremes.
    ModifiedSeasonallyAdjusted,
    /// Irregular modified for extremes.
    ModifiedIrregular,
    /// Residuals of the final regARIMA fit.
    Residuals,
}
display_phrases!(SpectralSeries {
    AdjustedOriginal => "adjusted original series",
    ModifiedSeasonallyAdjusted => "modified seasonally adjusted series",
    ModifiedIrregular => "modified irregular",
    Residuals => "regARIMA residuals",
});

/// Why a spectrum is not examined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpectralNotApplicable {
    /// Spectral diagnostics are monthly only.
    QuarterlyFrequency,
    /// The original series is constant.
    ConstantOriginalSeries,
    /// Trend-only decomposition has no seasonally adjusted series or irregular
    /// to examine.
    TrendOnlyDecomposition,
    /// The model estimation did not converge.
    ModelNotConverged,
    /// The model innovation variance is zero.
    ZeroModelVariance,
}
display_phrases!(SpectralNotApplicable {
    QuarterlyFrequency => "spectral diagnostics are monthly only",
    ConstantOriginalSeries => "the original series is constant",
    TrendOnlyDecomposition => "trend-only decomposition has no seasonally adjusted series or irregular to examine",
    ModelNotConverged => "the model estimation did not converge",
    ZeroModelVariance => "the model innovation variance is zero",
});

/// Step of the spectrum computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpectralStage {
    /// Building the series to examine.
    OperandConstruction,
    /// Taking logarithms.
    LogTransform,
    /// Differencing.
    Differencing,
    /// Removing the mean.
    MeanDeletion,
    /// Autocovariances.
    Autocovariance,
    /// Autoregressive recursion.
    ArRecursion,
    /// Spectral power.
    SpectralPower,
    /// Conversion to decibels.
    DecibelConversion,
}
display_phrases!(SpectralStage {
    OperandConstruction => "building the series to examine",
    LogTransform => "taking logarithms",
    Differencing => "differencing",
    MeanDeletion => "removing the mean",
    Autocovariance => "the autocovariances",
    ArRecursion => "the autoregressive recursion",
    SpectralPower => "the spectral power",
    DecibelConversion => "the conversion to decibels",
});

/// Why a spectrum that would be examined is undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpectralReason {
    /// The sample is too short.
    InsufficientInput {
        /// Number of values available.
        actual: usize,
        /// Number of values required.
        minimum: usize,
    },
    /// The log transform meets a nonpositive value.
    NonPositiveLogInput {
        /// Period of the value.
        period: Period,
    },
    /// The model innovation variance is negative or nonfinite.
    InvalidModelVariance,
    /// The series has a nonfinite value.
    NonFiniteInput {
        /// Period of the value.
        period: Period,
    },
    /// The sample has zero autocovariance.
    ZeroCovariance,
    /// The innovation variance collapsed before the required AR order.
    EarlyVarianceCollapse {
        /// Last AR order completed.
        completed_order: usize,
        /// AR order required.
        required_order: usize,
    },
    /// The innovation variance is nonpositive.
    NonPositiveInnovationVariance {
        /// AR order where it became nonpositive.
        order: usize,
    },
    /// Arithmetic produced a nonfinite value.
    NonFiniteArithmetic {
        /// Step where the value became nonfinite.
        stage: SpectralStage,
    },
    /// The X-11 quality computation that builds this series is unavailable.
    PartFPrerequisiteUnavailable,
}
impl std::fmt::Display for SpectralReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientInput { actual, minimum } => {
                write!(
                    formatter,
                    "{actual} values, fewer than the {minimum} required"
                )
            }
            Self::NonPositiveLogInput { period } => write!(
                formatter,
                "the value at {period} is not positive, so it has no logarithm"
            ),
            Self::InvalidModelVariance => {
                formatter.write_str("the model innovation variance is negative or not finite")
            }
            Self::NonFiniteInput { period } => {
                write!(formatter, "the value at {period} is not finite")
            }
            Self::ZeroCovariance => formatter.write_str("the sample has zero autocovariance"),
            Self::EarlyVarianceCollapse {
                completed_order,
                required_order,
            } => write!(
                formatter,
                "the innovation variance collapsed at AR order {completed_order}, before the \
                 required order {required_order}"
            ),
            Self::NonPositiveInnovationVariance { order } => write!(
                formatter,
                "the innovation variance is not positive at AR order {order}"
            ),
            Self::NonFiniteArithmetic { stage } => {
                write!(formatter, "{stage} produced a value that is not finite")
            }
            Self::PartFPrerequisiteUnavailable => formatter
                .write_str("the X-11 quality computation that builds this series is unavailable"),
        }
    }
}
