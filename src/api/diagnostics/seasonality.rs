//! X-11 seasonality diagnostics. Probabilities are in percent.
use super::{Sample, Unavailable};

/// The point in the X-11 run at which a seasonality test is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SeasonalityTest {
    /// Stable-seasonality F test on the preliminary SI ratios of the B
    /// iteration.
    PreliminaryStable,
    /// Stable-seasonality F test on the final SI ratios (table D8.A).
    FinalStable,
    /// Moving-seasonality F test (table D8.A).
    Moving,
    /// Kruskal-Wallis test (table D8.A).
    KruskalWallis,
    /// Combined seasonality test.
    Combined,
    /// Residual-seasonality F test on the seasonally adjusted series over the
    /// full span.
    ResidualFull,
    /// Residual-seasonality F test on the seasonally adjusted series over the
    /// last three years.
    ResidualRecentThreeYears,
    /// Residual-seasonality F test on the extreme-modified adjusted series over
    /// the full span.
    ModifiedResidualFull,
    /// Residual-seasonality F test on the extreme-modified adjusted series over
    /// the last three years.
    ModifiedResidualRecentThreeYears,
}

display_phrases!(SeasonalityTest {
    PreliminaryStable => "preliminary stable-seasonality test",
    FinalStable => "final stable-seasonality test",
    Moving => "moving-seasonality test",
    KruskalWallis => "Kruskal-Wallis test",
    Combined => "combined seasonality test",
    ResidualFull => "residual-seasonality test over the full span",
    ResidualRecentThreeYears => "residual-seasonality test over the last three years",
    ModifiedResidualFull => "modified residual-seasonality test over the full span",
    ModifiedResidualRecentThreeYears => "modified residual-seasonality test over the last three years",
});

/// Why an X-11 diagnostic test could not be computed: a case where the
/// reference program stops the test early, or one where its result would be
/// undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum X11DiagnosticUnavailableReason {
    /// The residual mean square is zero.
    ZeroResidualMeanSquare,
    /// The original series is constant.
    ConstantOriginalSeries,
    /// The sample is too short.
    InsufficientSample,
    /// Arithmetic produced a nonfinite value.
    NonFiniteArithmetic,
    /// A rank sum overflows the reference program's integer arithmetic.
    RankIntegerOverflow,
    /// A test this one depends on is unavailable.
    PrerequisiteUnavailable {
        /// Test that is unavailable.
        stage: SeasonalityTest,
    },
}

impl std::fmt::Display for X11DiagnosticUnavailableReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroResidualMeanSquare => formatter.write_str("the residual mean square is zero"),
            Self::ConstantOriginalSeries => formatter.write_str("the original series is constant"),
            Self::InsufficientSample => formatter.write_str("the sample is too short"),
            Self::NonFiniteArithmetic => formatter.write_str("the arithmetic is not finite"),
            Self::RankIntegerOverflow => formatter.write_str("a rank sum overflows"),
            Self::PrerequisiteUnavailable { stage } => {
                write!(formatter, "the {stage} is unavailable")
            }
        }
    }
}

/// An F test from X-11's seasonality tests: the statistic, its upper-tail
/// probability and degrees of freedom.
///
/// As in X-13ARIMA-SEATS, a statistic above 90 (or above 40 with more than 150
/// denominator degrees of freedom) reports a probability of 0, and a
/// nonpositive one reports a statistic of 0 and a probability of 100%.
#[derive(Debug, Clone, PartialEq)]
pub struct FTest {
    pub(crate) statistic: f64,
    pub(crate) upper_tail_probability_percent: f64,
    pub(crate) numerator_degrees_of_freedom: usize,
    pub(crate) denominator_degrees_of_freedom: usize,
    pub(crate) sample: Sample,
    pub(crate) differencing_lag: Option<usize>,
    pub(crate) small_residual_mean_square: bool,
}
impl FTest {
    /// F statistic.
    #[must_use]
    pub const fn statistic(&self) -> f64 {
        self.statistic
    }
    /// Upper-tail probability in percent, not a fraction.
    #[must_use]
    pub const fn upper_tail_probability_percent(&self) -> f64 {
        self.upper_tail_probability_percent
    }
    /// Numerator degrees of freedom.
    #[must_use]
    pub const fn numerator_degrees_of_freedom(&self) -> usize {
        self.numerator_degrees_of_freedom
    }
    /// Denominator degrees of freedom.
    #[must_use]
    pub const fn denominator_degrees_of_freedom(&self) -> usize {
        self.denominator_degrees_of_freedom
    }
    /// Sample the test used.
    #[must_use]
    pub const fn sample(&self) -> &Sample {
        &self.sample
    }
    /// Lag of the difference a residual-seasonality test takes first: 3 for a
    /// monthly series, 1 for a quarterly one. `None` for the other tests.
    #[must_use]
    pub const fn differencing_lag(&self) -> Option<usize> {
        self.differencing_lag
    }
    /// Whether the residual mean square of a multiplicative stable-seasonality
    /// test is at most 0.01, where X-13ARIMA-SEATS prints a note on the test's
    /// reliability. It is not a [`Warning`](crate::Warning).
    #[must_use]
    pub const fn small_residual_mean_square(&self) -> bool {
        self.small_residual_mean_square
    }
}

/// Stable-seasonality conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StableSeasonality {
    /// Stable seasonality is present at the 0.1% level.
    PresentAtPointOnePercent,
    /// Stable seasonality is not present at the 0.1% level.
    NotPresentAtPointOnePercent,
}
/// Moving-seasonality conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MovingSeasonality {
    /// Moving seasonality is present at the 1% level.
    PresentAtOnePercent,
    /// Moving seasonality is present at the 5% level.
    PresentAtFivePercent,
    /// Moving seasonality is not present at the 5% level.
    NotPresentAtFivePercent,
}
/// Conclusion of a residual-seasonality test.
///
/// Full-span tests report only the 1% variants. Tests over the last three
/// years report [`PresentAtOnePercent`](Self::PresentAtOnePercent),
/// [`PresentAtFivePercent`](Self::PresentAtFivePercent) or
/// [`NotPresentAtFivePercent`](Self::NotPresentAtFivePercent), each
/// classifying a probability exactly at a threshold as X-13ARIMA-SEATS does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualSeasonality {
    /// Residual seasonality is present at the 1% level.
    PresentAtOnePercent,
    /// Residual seasonality is not present at the 1% level.
    NotPresentAtOnePercent,
    /// Residual seasonality is present at the 5% level.
    PresentAtFivePercent,
    /// Residual seasonality is not present at the 5% level.
    NotPresentAtFivePercent,
}
/// Kruskal-Wallis conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KruskalWallisSeasonality {
    /// Seasonality is present at the 1% level.
    PresentAtOnePercent,
    /// Seasonality is not present at the 1% level.
    NotPresentAtOnePercent,
}
/// Combined seasonality conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombinedSeasonality {
    /// Identifiable seasonality is present.
    Present,
    /// Identifiable seasonality is probably not present.
    ProbablyNotPresent,
    /// Identifiable seasonality is not present.
    NotPresent,
}

/// Stable-seasonality F test and its conclusion.
#[derive(Debug, Clone, PartialEq)]
pub struct StableSeasonalityTest {
    pub(crate) test: FTest,
    pub(crate) conclusion: StableSeasonality,
}
impl StableSeasonalityTest {
    /// F test and its sample.
    #[must_use]
    pub const fn test(&self) -> &FTest {
        &self.test
    }
    /// The test's conclusion.
    #[must_use]
    pub const fn conclusion(&self) -> StableSeasonality {
        self.conclusion
    }
}
/// Moving-seasonality F test and its conclusion.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingSeasonalityTest {
    pub(crate) test: FTest,
    pub(crate) conclusion: MovingSeasonality,
}
impl MovingSeasonalityTest {
    /// F test and its sample.
    #[must_use]
    pub const fn test(&self) -> &FTest {
        &self.test
    }
    /// The test's conclusion.
    #[must_use]
    pub const fn conclusion(&self) -> MovingSeasonality {
        self.conclusion
    }
}
/// Residual-seasonality F test and its conclusion.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualSeasonalityTest {
    pub(crate) test: FTest,
    pub(crate) conclusion: ResidualSeasonality,
}
impl ResidualSeasonalityTest {
    /// F test and its sample.
    #[must_use]
    pub const fn test(&self) -> &FTest {
        &self.test
    }
    /// The test's conclusion.
    #[must_use]
    pub const fn conclusion(&self) -> ResidualSeasonality {
        self.conclusion
    }
}

/// Kruskal–Wallis test for stable seasonality, as X-13ARIMA-SEATS computes it:
/// tied values get distinct ranks from its sort rather than average ranks, and
/// there is no tie correction.
#[derive(Debug, Clone, PartialEq)]
pub struct KruskalWallisTest {
    pub(crate) statistic: f64,
    pub(crate) upper_tail_probability_percent: f64,
    pub(crate) degrees_of_freedom: usize,
    pub(crate) sample: Sample,
    pub(crate) conclusion: KruskalWallisSeasonality,
}
impl KruskalWallisTest {
    /// Kruskal-Wallis statistic.
    #[must_use]
    pub const fn statistic(&self) -> f64 {
        self.statistic
    }
    /// Upper-tail probability in percent, not a fraction.
    #[must_use]
    pub const fn upper_tail_probability_percent(&self) -> f64 {
        self.upper_tail_probability_percent
    }
    /// Degrees of freedom.
    #[must_use]
    pub const fn degrees_of_freedom(&self) -> usize {
        self.degrees_of_freedom
    }
    /// Sample the test used.
    #[must_use]
    pub const fn sample(&self) -> &Sample {
        &self.sample
    }
    /// The test's conclusion.
    #[must_use]
    pub const fn conclusion(&self) -> KruskalWallisSeasonality {
        self.conclusion
    }
}

/// The nine seasonality tests of an X-11 run. Each may be unavailable on its
/// own, which does not invalidate the adjustment.
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalityTests {
    pub(crate) preliminary_stable: Result<StableSeasonalityTest, Unavailable>,
    pub(crate) final_stable: Result<StableSeasonalityTest, Unavailable>,
    pub(crate) moving: Result<MovingSeasonalityTest, Unavailable>,
    pub(crate) kruskal_wallis: Result<KruskalWallisTest, Unavailable>,
    pub(crate) combined: Result<CombinedSeasonality, Unavailable>,
    pub(crate) residual_full: Result<ResidualSeasonalityTest, Unavailable>,
    pub(crate) residual_recent_three_years: Result<ResidualSeasonalityTest, Unavailable>,
    pub(crate) modified_residual_full: Result<ResidualSeasonalityTest, Unavailable>,
    pub(crate) modified_residual_recent_three_years: Result<ResidualSeasonalityTest, Unavailable>,
}
impl SeasonalityTests {
    /// Stable-seasonality test on the preliminary SI ratios of the B
    /// iteration.
    pub const fn preliminary_stable(&self) -> Result<&StableSeasonalityTest, &Unavailable> {
        self.preliminary_stable.as_ref()
    }
    /// Stable-seasonality test on the final SI ratios (table D8.A).
    pub const fn final_stable(&self) -> Result<&StableSeasonalityTest, &Unavailable> {
        self.final_stable.as_ref()
    }
    /// Moving-seasonality test (table D8.A).
    pub const fn moving(&self) -> Result<&MovingSeasonalityTest, &Unavailable> {
        self.moving.as_ref()
    }
    /// Kruskal-Wallis test (table D8.A).
    pub const fn kruskal_wallis(&self) -> Result<&KruskalWallisTest, &Unavailable> {
        self.kruskal_wallis.as_ref()
    }
    /// Combined seasonality test.
    pub const fn combined(&self) -> Result<&CombinedSeasonality, &Unavailable> {
        self.combined.as_ref()
    }
    /// Residual seasonality on the seasonally adjusted series over the full
    /// span.
    pub const fn residual_full(&self) -> Result<&ResidualSeasonalityTest, &Unavailable> {
        self.residual_full.as_ref()
    }
    /// Residual seasonality on the seasonally adjusted series over the last
    /// three years.
    pub const fn residual_recent_three_years(
        &self,
    ) -> Result<&ResidualSeasonalityTest, &Unavailable> {
        self.residual_recent_three_years.as_ref()
    }
    /// Residual seasonality on the extreme-modified adjusted series (E2).
    pub const fn modified_residual_full(&self) -> Result<&ResidualSeasonalityTest, &Unavailable> {
        self.modified_residual_full.as_ref()
    }
    /// The same test over the last three years, run only when the full-span
    /// test succeeds. X-13ARIMA-SEATS computes it only when it prints table E2;
    /// this crate always does, and warns when it cannot
    /// ([`Warning::SeasonalityTestUnavailable`](crate::Warning::SeasonalityTestUnavailable)).
    pub const fn modified_residual_recent_three_years(
        &self,
    ) -> Result<&ResidualSeasonalityTest, &Unavailable> {
        self.modified_residual_recent_three_years.as_ref()
    }
}
