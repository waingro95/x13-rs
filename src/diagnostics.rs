//! The tests and measures behind the diagnostics of an
//! [`Adjustment`](crate::Adjustment): the seasonality tests
//! ([`SeasonalityTests`]) and quality measures ([`QualityMeasures`]) of an
//! X-11 run, the [`SpectralPeaks`] of a spectrum, the [`Sample`] a diagnostic
//! covers, and [`Unavailable`], which says why a diagnostic has no value.
//!
//! A diagnostic that may have no value is a `Result<&T, &Unavailable>`. The
//! [`Unavailable`] describes its reason through `Display` and has no
//! accessors beyond [`Unavailable::kind`]: branch on the [`UnavailableKind`]
//! and report the message.

pub use crate::api::diagnostics::{
    CombinedSeasonality, FTest, KruskalWallisSeasonality, KruskalWallisTest, MovingSeasonality,
    MovingSeasonalityTest, QualityMeasures, ResidualSeasonality, ResidualSeasonalityTest, Sample,
    SeasonalEvolution, SeasonalityTest, SeasonalityTests, SpectralPeakKind, SpectralPeaks,
    SpectralSample, StableSeasonality, StableSeasonalityTest, Unavailable, UnavailableKind,
};

pub(crate) use crate::api::diagnostics::{
    SpectralNotApplicable, SpectralReason, SpectralSeries, SpectralStage, SpectralTradingDayPeaks,
    UnavailableDetail, X11DiagnosticUnavailableReason, X11QualitySeries, X11QualityStage,
    X11QualityUnavailableReason,
};
