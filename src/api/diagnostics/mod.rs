//! What the diagnostics share: the sample a diagnostic covers, the peaks of a
//! spectrum, and why a diagnostic has no value.
use crate::Period;

mod quality;
mod seasonality;
mod spectral;
pub use quality::{QualityMeasures, SeasonalEvolution};
pub(crate) use quality::{X11QualitySeries, X11QualityStage, X11QualityUnavailableReason};
pub(crate) use seasonality::X11DiagnosticUnavailableReason;
pub use seasonality::{
    CombinedSeasonality, FTest, KruskalWallisSeasonality, KruskalWallisTest, MovingSeasonality,
    MovingSeasonalityTest, ResidualSeasonality, ResidualSeasonalityTest, SeasonalityTest,
    SeasonalityTests, StableSeasonality, StableSeasonalityTest,
};
pub(crate) use spectral::{
    SpectralNotApplicable, SpectralReason, SpectralSeries, SpectralStage, SpectralTradingDayPeaks,
};
pub use spectral::{SpectralPeakKind, SpectralPeaks, SpectralSample};

/// The periods a diagnostic covers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sample {
    pub(crate) first: Period,
    pub(crate) last: Period,
    pub(crate) count: usize,
}
impl Sample {
    /// First period.
    #[must_use]
    pub const fn first(&self) -> Period {
        self.first
    }
    /// Last period.
    #[must_use]
    pub const fn last(&self) -> Period {
        self.last
    }
    /// Number of periods.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
}

/// Why a diagnostic has no value. `Display` gives the reason.
///
/// A diagnostic without a value does not invalidate the adjustment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unavailable(pub(crate) UnavailableDetail);

impl Unavailable {
    /// The class of the reason, for deciding what to do about it.
    #[must_use]
    pub fn kind(&self) -> UnavailableKind {
        match &self.0 {
            UnavailableDetail::SeasonalityTest { reason, .. } => reason.kind(),
            UnavailableDetail::Quality { reason, .. } => reason.kind(),
            UnavailableDetail::SpectrumNotApplicable { .. } => UnavailableKind::NotApplicable,
            UnavailableDetail::Spectrum { reason, .. } => reason.kind(),
        }
    }
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            UnavailableDetail::SeasonalityTest { test, reason, .. } => {
                write!(formatter, "the {test} is unavailable: {reason}")
            }
            UnavailableDetail::Quality { reason, .. } => {
                write!(formatter, "the M and Q measures are unavailable: {reason}")
            }
            UnavailableDetail::SpectrumNotApplicable { series, reason } => {
                write!(
                    formatter,
                    "the spectrum of the {series} is not examined: {reason}"
                )
            }
            UnavailableDetail::Spectrum { series, reason, .. } => {
                write!(
                    formatter,
                    "the spectrum of the {series} is unavailable: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for Unavailable {}

/// The class of reason a diagnostic has no value, from
/// [`Unavailable::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnavailableKind {
    /// The request or the series rules the diagnostic out: a quarterly series
    /// has no spectrum, a trend-only run has no spectrum of the seasonally
    /// adjusted series, a constant series has nothing to test. Expected, and
    /// not a problem.
    NotApplicable,
    /// The series has too few observations for this diagnostic.
    InsufficientData,
    /// The computation met a degenerate or nonfinite value.
    Numerical,
}

/// Which diagnostic has no value, and why.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum UnavailableDetail {
    /// A seasonality test. An empty sample has no dates.
    SeasonalityTest {
        test: SeasonalityTest,
        reason: X11DiagnosticUnavailableReason,
        /// The sample the test would use.
        sample: Option<Sample>,
        /// The lag of the difference a residual seasonality test takes first.
        differencing_lag: Option<usize>,
    },
    /// The M and Q quality measures, with the spans they would cover.
    Quality {
        reason: X11QualityUnavailableReason,
        /// The span of the observed series.
        observed_span: Sample,
        /// The span of the seasonal factors that M8 through M11 read.
        seasonal_history_span: Sample,
    },
    /// A spectrum X-13ARIMA-SEATS does not examine in this case.
    SpectrumNotApplicable {
        series: SpectralSeries,
        reason: SpectralNotApplicable,
    },
    /// A spectrum that would be examined but is undefined.
    Spectrum {
        series: SpectralSeries,
        reason: SpectralReason,
        /// The sample the spectrum would use, when it could be built.
        sample: Option<SpectralSample>,
    },
}

impl X11DiagnosticUnavailableReason {
    const fn kind(self) -> UnavailableKind {
        match self {
            Self::ConstantOriginalSeries => UnavailableKind::NotApplicable,
            Self::InsufficientSample => UnavailableKind::InsufficientData,
            Self::ZeroResidualMeanSquare
            | Self::NonFiniteArithmetic
            | Self::RankIntegerOverflow
            | Self::PrerequisiteUnavailable { .. } => UnavailableKind::Numerical,
        }
    }
}

impl X11QualityUnavailableReason {
    const fn kind(&self) -> UnavailableKind {
        match self {
            Self::ConstantSeries { .. } => UnavailableKind::NotApplicable,
            Self::InsufficientHistory { .. } | Self::NoEligibleValues { .. } => {
                UnavailableKind::InsufficientData
            }
            Self::NonFiniteArithmetic { .. }
            | Self::NonPositiveLogInput { .. }
            | Self::ZeroDenominator { .. }
            | Self::PrerequisiteUnavailable { .. } => UnavailableKind::Numerical,
        }
    }
}

impl SpectralReason {
    const fn kind(self) -> UnavailableKind {
        match self {
            Self::InsufficientInput { .. } => UnavailableKind::InsufficientData,
            Self::NonPositiveLogInput { .. }
            | Self::InvalidModelVariance
            | Self::NonFiniteInput { .. }
            | Self::ZeroCovariance
            | Self::EarlyVarianceCollapse { .. }
            | Self::NonPositiveInnovationVariance { .. }
            | Self::NonFiniteArithmetic { .. }
            | Self::PartFPrerequisiteUnavailable => UnavailableKind::Numerical,
        }
    }
}
