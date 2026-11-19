use super::{CalendarSigmaSelection, Decomposition, Mode, MsrSelection, SpectralDiagnostics};
use crate::diagnostics::Unavailable;

/// Diagnostics of an X-11 adjustment, from [`Diagnostics::x11`](crate::Diagnostics::x11): the mode
/// and filters used, seasonality tests, quality measures and spectral
/// peaks.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostics {
    pub(in crate::api) spectral: SpectralDiagnostics,
    pub(in crate::api) decomposition: Decomposition,
    pub(in crate::api) mode: Mode,
    pub(in crate::api) final_henderson_terms: usize,
    pub(in crate::api) msr_selection: Option<MsrSelection>,
    pub(in crate::api) calendar_sigma_selection: Option<CalendarSigmaSelection>,
    pub(in crate::api) seasonality: crate::diagnostics::SeasonalityTests,
    pub(in crate::api) quality: Result<crate::diagnostics::QualityMeasures, Unavailable>,
}

impl Diagnostics {
    /// Spectral peaks from the default spectra, over the last eight
    /// years (96 observations) or the whole series if shorter. Monthly
    /// series only: quarterly and constant series are not applicable. Each
    /// spectrum may be unavailable on its own.
    #[must_use]
    pub const fn spectral(&self) -> &SpectralDiagnostics {
        &self.spectral
    }
    /// The decomposition X-11 performed.
    #[must_use]
    pub const fn decomposition(&self) -> Decomposition {
        self.decomposition
    }
    /// The M and Q quality measures, where X-13ARIMA-SEATS computes them.
    /// Measures that are unavailable do not make the adjustment invalid.
    pub const fn quality(&self) -> Result<&crate::diagnostics::QualityMeasures, &Unavailable> {
        self.quality.as_ref()
    }
    /// The nine seasonality tests, including the final-D11 and E2 tests.
    #[must_use]
    pub const fn seasonality(&self) -> &crate::diagnostics::SeasonalityTests {
        &self.seasonality
    }
    /// Both automatic calendar-sigma decisions, present only for an Automatic
    /// request.
    #[must_use]
    pub const fn calendar_sigma_selection(&self) -> Option<&CalendarSigmaSelection> {
        self.calendar_sigma_selection.as_ref()
    }
    /// Decomposition mode used for this adjustment.
    #[must_use]
    pub const fn mode(&self) -> Mode {
        self.mode
    }
    /// Number of terms in the final selected Henderson filter.
    #[must_use]
    pub const fn final_henderson_terms(&self) -> usize {
        self.final_henderson_terms
    }

    /// Global MSR outcome, or `None` when no calendar period requests MSR.
    #[must_use]
    pub const fn msr_selection(&self) -> Option<&MsrSelection> {
        self.msr_selection.as_ref()
    }
}

/// Stage of direct X-11 trend estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrendStage {
    /// First trend estimate used in the decomposition iterations.
    Preliminary,
    /// Intermediate trend estimate used in the decomposition iterations.
    Intermediate,
    /// Last trend estimate used in the decomposition iterations.
    FinalIteration,
    /// Final trend-cycle component returned by the adjustment.
    FinalTrendCycle,
}

display_phrases!(TrendStage {
    Preliminary => "preliminary trend",
    Intermediate => "intermediate trend",
    FinalIteration => "final-iteration trend",
    FinalTrendCycle => "final trend-cycle",
});
