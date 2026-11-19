//! Spectral peak diagnostics for the X-11 series.
use crate::diagnostics::{SpectralPeaks, Unavailable};

/// Spectral peaks of the X-11 input and output series.
///
/// Each spectrum may be unavailable on its own, which does not invalidate
/// the adjustment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpectralDiagnostics {
    pub(crate) adjusted_original: Result<SpectralPeaks, Unavailable>,
    pub(crate) modified_seasonally_adjusted: Result<SpectralPeaks, Unavailable>,
    pub(crate) modified_irregular: Result<SpectralPeaks, Unavailable>,
}
impl SpectralDiagnostics {
    /// Peaks of the prior-adjusted original series, the first spectrum (table
    /// sp0).
    pub const fn adjusted_original(&self) -> Result<&SpectralPeaks, &Unavailable> {
        self.adjusted_original.as_ref()
    }
    /// Peaks of the seasonally adjusted series modified for extremes.
    pub const fn modified_seasonally_adjusted(&self) -> Result<&SpectralPeaks, &Unavailable> {
        self.modified_seasonally_adjusted.as_ref()
    }
    /// Peaks of the irregular modified for extremes.
    pub const fn modified_irregular(&self) -> Result<&SpectralPeaks, &Unavailable> {
        self.modified_irregular.as_ref()
    }
}
