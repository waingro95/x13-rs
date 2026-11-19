//! X-11 quality diagnostics: Build 62's Part F (`x11pt4`), which compares the
//! components' average changes and variances and derives the M1–M11 measures
//! and Q (`f3cal`).
mod measures;
use crate::diagnostics::{
    Sample, Unavailable, UnavailableDetail, X11QualitySeries as Series, X11QualityStage as Stage,
    X11QualityUnavailableReason as Reason,
};
use crate::numerics::build_62_equal;
use crate::time::AxisSpan;
use crate::x11::spectral::LevelShiftRoundTrip;
use crate::{diagnostics, x11};

/// What the run's seasonal filters mean for the quality measures.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QualityFilterHistory {
    /// A stable seasonal filter was requested, so M8–M11 are not computed.
    pub(crate) stable: bool,
    /// M6 has a weight in Q, which it has only for a 3x5 seasonal filter.
    pub(crate) include_m6: bool,
}

impl QualityFilterHistory {
    /// From the requested `filters`, the `selected` filters the run used and
    /// the `outcome` of the moving seasonality ratio's choice, if it made one.
    pub(super) fn resolve(
        filters: &[crate::x11::SeasonalFilter],
        selected: &[crate::x11::SeasonalFilter],
        outcome: Option<&crate::x11::MsrSelection>,
    ) -> Self {
        use crate::{x11, x11::SeasonalFilter as Filter};
        let allowed = |filter: &Filter| matches!(filter, Filter::S3x5 | Filter::X11Default);
        Self {
            stable: filters.contains(&Filter::Stable),
            include_m6: (filters.iter().all(allowed)
                || matches!(
                    outcome,
                    Some(x11::MsrSelection::Selected {
                        filter: x11::MsrFilter::S3x5,
                        ..
                    })
                ))
                && selected.iter().all(allowed),
        }
    }
}

/// What Part F reads from the X-11 run. It is kept apart from the final
/// components because restoring regARIMA effects changes those afterwards.
#[derive(Debug)]
pub(crate) struct QualityWorkspace {
    /// The dates of every index of the extended series, of which `observed`
    /// is the observed range.
    pub(crate) dates: AxisSpan,
    pub(crate) observed: std::ops::Range<usize>,
    /// Seasonal factors from the start of the extended series through the
    /// last observation.
    pub(crate) seasonal: Vec<f64>,
    /// Trend-cycle over the observed range.
    pub(crate) base_trend: Vec<f64>,
    /// Irregular over the observed range.
    pub(crate) base_irregular: Vec<f64>,
    /// [`intermediate_eligibility`] of each observation.
    pub(crate) intermediate_eligible: Vec<bool>,
    /// The final I/C ratio that M3 reads (Build 62's `ratic`).
    pub(crate) irregular_trend_ratio: f64,
    /// The final moving seasonality ratio that M6 reads (Build 62's `ratis`).
    pub(crate) moving_seasonality_ratio: Result<f64, Reason>,
    pub(crate) history: QualityFilterHistory,
}

#[cfg(test)]
impl QualityWorkspace {
    /// A workspace with no retained operands, for tests that build components
    /// by hand.
    pub(crate) fn empty() -> Self {
        Self {
            dates: crate::time::TimeAxis::monthly(2000, 1)
                .unwrap()
                .span(0)
                .unwrap(),
            observed: 0..0,
            seasonal: Vec::new(),
            base_trend: Vec::new(),
            base_irregular: Vec::new(),
            intermediate_eligible: Vec::new(),
            irregular_trend_ratio: 0.0,
            moving_seasonality_ratio: Ok(0.0),
            history: QualityFilterHistory {
                stable: false,
                include_m6: false,
            },
        }
    }
}

/// Build 62's `Gudval` as `x11pt3` leaves it before the final trend-cycle and
/// seasonal adjustment: in pseudo-additive mode, an observation whose modified
/// seasonally adjusted value is not positive is left out of Part F. Other
/// modes leave none out.
pub(super) fn intermediate_eligibility(
    mode: x11::Mode,
    modified_adjusted: &[f64],
    observed: std::ops::Range<usize>,
) -> Vec<bool> {
    modified_adjusted[observed]
        .iter()
        .map(|&value| mode != x11::Mode::PseudoAdditive || value > 0.0)
        .collect()
}

/// `Gudval` as Part F reads it: `x11pt3` and `chkzro` also leave out an
/// observation whose original or seasonally adjusted value is not positive,
/// and never bring one back.
fn final_eligibility(
    mode: x11::Mode,
    intermediate: &[bool],
    original: &[f64],
    adjusted: &[f64],
) -> Vec<bool> {
    original
        .iter()
        .zip(adjusted)
        .zip(intermediate)
        .map(|((&o, &sa), &eligible)| {
            mode != x11::Mode::PseudoAdditive || (eligible && o > 0.0 && sa > 0.0)
        })
        .collect()
}

/// The adjustment factors Part F reads, over the observed dates and on the
/// scale of the original series. `Some` says the adjustment was made, even
/// when every factor is neutral.
#[derive(Default)]
pub(crate) struct QualityAdjustments {
    /// Trading-day and holiday factors combined.
    pub(crate) calendar: Option<Vec<f64>>,
    /// The outlier and user-defined regression effects combined.
    pub(crate) prior: Option<Vec<f64>>,
    pub(crate) level_shift: Option<Vec<f64>>,
    pub(crate) additive_outlier: Option<Vec<f64>>,
    pub(crate) temporary_change: Option<Vec<f64>>,
}

/// The first `count` periods of `dates`.
fn span(dates: AxisSpan, count: usize) -> Sample {
    Sample {
        first: dates.start(),
        last: dates.period(count.saturating_sub(1)),
        count,
    }
}

/// The quality diagnostics of a finished run. It also leaves in the spectral
/// workspace how far Part F got and the trend-cycle it ended with.
pub(super) fn complete(
    components: &mut super::Components,
    original: &[f64],
    observed_dates: AxisSpan,
    adjustments: QualityAdjustments,
) -> Result<diagnostics::QualityMeasures, Unavailable> {
    let workspace = &components.quality_workspace;
    let observed_span = span(observed_dates, original.len());
    let seasonal_history_span = span(workspace.dates, workspace.observed.end);
    let (progress, summary) = calculate(components, original, observed_dates, &adjustments);
    let work = &mut components.spectral_workspace;
    work.level_shift_round_trip = progress.level_shift_round_trip;
    work.part_f_trend = progress.trend;
    let trend_only = components.decomposition == crate::x11::Decomposition::TrendOnly;
    match summary {
        Ok(summary) => {
            let m = summary.measures;
            Ok(diagnostics::QualityMeasures {
                m: [m.m1, m.m2, m.m3, m.m4, m.m5, m.m7],
                m6: (!trend_only).then_some(m.m6),
                seasonal_evolution: summary.evolution.filter(|_| !trend_only).map(|evolution| {
                    diagnostics::SeasonalEvolution {
                        m: evolution.values(),
                    }
                }),
                q: summary.q,
                q_without_m2: summary.q_without_m2,
                failed_measure_count: summary.failed,
                m6_included_in_q: !trend_only && workspace.history.include_m6,
                observed_span,
                seasonal_history_span,
            })
        }
        Err(reason) => Err(Unavailable(UnavailableDetail::Quality {
            reason,
            observed_span,
            seasonal_history_span,
        })),
    }
}

/// Why an average change or a variance has no value. [`at`] adds the stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArithmeticFailure {
    NoEligibleValues,
    ZeroDenominator,
    NonFinite,
}

fn finite(value: f64) -> Result<f64, ArithmeticFailure> {
    value
        .is_finite()
        .then_some(value)
        .ok_or(ArithmeticFailure::NonFinite)
}

/// Names the Part F stage an arithmetic failure happened in.
fn at<T>(value: Result<T, ArithmeticFailure>, stage: Stage) -> Result<T, Reason> {
    value.map_err(|error| match error {
        ArithmeticFailure::NoEligibleValues => Reason::NoEligibleValues { stage },
        ArithmeticFailure::ZeroDenominator => Reason::ZeroDenominator { stage },
        ArithmeticFailure::NonFinite => Reason::NonFiniteArithmetic { stage },
    })
}

/// `value`, or the non-finite arithmetic failure of `stage`.
fn checked(value: f64, stage: Stage) -> Result<f64, Reason> {
    at(finite(value), stage)
}

/// Takes `effect` out of `values`: by division when multiplicative, else by
/// subtraction.
fn remove(
    values: &mut [f64],
    effect: Option<&[f64]>,
    multiplicative: bool,
    stage: Stage,
) -> Result<(), Reason> {
    if let Some(effect) = effect {
        for (value, &factor) in values.iter_mut().zip(effect) {
            if multiplicative && factor == 0.0 {
                return Err(Reason::ZeroDenominator { stage });
            }
            *value = checked(
                if multiplicative {
                    *value / factor
                } else {
                    *value - factor
                },
                stage,
            )?;
        }
    }
    Ok(())
}

/// Puts `effect` back into `values`.
fn restore(
    values: &mut [f64],
    effect: Option<&[f64]>,
    multiplicative: bool,
    stage: Stage,
) -> Result<(), Reason> {
    if let Some(effect) = effect {
        for (value, &factor) in values.iter_mut().zip(effect) {
            *value = checked(
                if multiplicative {
                    *value * factor
                } else {
                    *value + factor
                },
                stage,
            )?;
        }
    }
    Ok(())
}

/// A series' mean absolute change (percent when multiplicative) at each lag
/// `1..=period`, and its square.
#[derive(Clone, Copy)]
struct LagChanges {
    mean: [f64; 12],
    square: [f64; 12],
}

impl LagChanges {
    /// A component Part F does not have contributes no change.
    const NONE: Self = Self {
        mean: [0.0; 12],
        square: [0.0; 12],
    };

    fn mean(&self, lag: usize) -> f64 {
        self.mean[lag - 1]
    }

    fn square(&self, lag: usize) -> f64 {
        self.square[lag - 1]
    }
}

/// `sumry`'s changes at every lag up to one year.
fn changes(
    values: &[f64],
    good: &[bool],
    period: usize,
    multiplicative: bool,
    stage: Stage,
) -> Result<LagChanges, Reason> {
    let mut changes = LagChanges::NONE;
    let slots = changes.mean.iter_mut().zip(&mut changes.square);
    for (lag, (mean, square)) in (1..=period).zip(slots) {
        (*mean, *square) = at(lag_changes(values, good, lag, multiplicative), stage)?;
    }
    Ok(changes)
}

/// Runs Part F: the measures or why there are none, and how far it got
/// either way.
fn calculate(
    components: &super::Components,
    original: &[f64],
    dates: AxisSpan,
    adjustments: &QualityAdjustments,
) -> (Progress, Result<measures::Summary, Reason>) {
    let mut progress = Progress {
        level_shift_round_trip: LevelShiftRoundTrip::Unknown,
        trend: None,
    };
    let summary = progress.run(components, original, dates, adjustments);
    (progress, summary)
}

/// What the spectral diagnostics read from Part F, set as it goes so that a
/// failure of the measures does not lose it.
struct Progress {
    level_shift_round_trip: LevelShiftRoundTrip,
    /// The antilogged Part F trend-cycle (`Stc`), once it is computed.
    trend: Option<Vec<f64>>,
}

/// What every Part F stage reads.
struct PartFInput<'a> {
    components: &'a super::Components,
    work: &'a QualityWorkspace,
    original: &'a [f64],
    /// The observed dates, one per element of `original`.
    dates: AxisSpan,
    adjustments: &'a QualityAdjustments,
    period: usize,
    multiplicative: bool,
    trend_only: bool,
    /// `x11pt4`'s `Gudval`: the observations the changes and variances read.
    good: Vec<bool>,
}

/// `sumry`'s changes of each component Part F compares, and the variances it
/// later normalizes.
struct ComponentChanges {
    prior: LagChanges,
    calendar: LagChanges,
    irregular: LagChanges,
    modified_irregular: LagChanges,
    seasonal: LagChanges,
    trend: LagChanges,
    prior_variance: f64,
    /// The modified irregular's variance.
    irregular_variance: f64,
    /// The irregular's average duration of run.
    duration: f64,
}

/// The M1–M7 operands that variance normalization produces.
struct Normalized {
    prior_variance: f64,
    irregular_variance: f64,
    /// The irregular and prior shares at the one-quarter span.
    irregular_share: f64,
    prior_share: f64,
    /// Each span's I/C ratio; the first `period` entries are read.
    span_ratios: [f64; 12],
}

impl Progress {
    /// Part F by stage: the components' changes, the detrended original, the
    /// constant-series return, variance normalization, M1–M7, the
    /// seasonal-evolution measures and the summary.
    fn run(
        &mut self,
        components: &super::Components,
        original: &[f64],
        dates: AxisSpan,
        adjustments: &QualityAdjustments,
    ) -> Result<measures::Summary, Reason> {
        let work = &components.quality_workspace;
        let period = usize::from(dates.start().frequency().periods_per_year());
        let n = original.len();
        if n <= period {
            return Err(Reason::InsufficientHistory {
                stage: Stage::IrregularChanges,
                actual: n,
                minimum: period + 1,
            });
        }
        let multiplicative = components.mode != x11::Mode::Additive;
        let trend_only = components.decomposition == crate::x11::Decomposition::TrendOnly;
        let pseudo_trend_only = trend_only && components.mode == x11::Mode::PseudoAdditive;
        let good = if pseudo_trend_only {
            vec![false; n]
        } else {
            final_eligibility(
                components.mode,
                &work.intermediate_eligible,
                original,
                &components.seasonally_adjusted,
            )
        };
        let input = PartFInput {
            components,
            work,
            original,
            dates,
            adjustments,
            period,
            multiplicative,
            trend_only,
            good,
        };
        let (mut modified, modified_irregular) = modify_original_and_irregular(
            original,
            &work.base_irregular,
            &components.intermediate_extreme_weights,
            multiplicative,
        )?;
        if pseudo_trend_only {
            return self.pseudo_trend_only_return(&input, &modified, &modified_irregular);
        }
        let mut reasons = Vec::new();
        let changes = input.component_changes(&mut modified, &modified_irregular, &mut reasons)?;
        let detrended_variance = self.detrended_variance(&input, &mut modified, &mut reasons)?;
        if !reasons.is_empty() {
            self.level_shift_round_trip = LevelShiftRoundTrip::NotReached;
            return Err(Reason::ConstantSeries { series: reasons });
        }
        let detrended_variance = detrended_variance.ok_or(Reason::NoEligibleValues {
            stage: Stage::VarianceNormalization,
        })?;
        let normalized = changes.normalize(detrended_variance, period)?;
        // `x11pt4` takes the level shift out of the modified seasonally
        // adjusted series and puts it back right after this normalization,
        // whatever becomes of the measures below.
        self.level_shift_round_trip = LevelShiftRoundTrip::Reached;
        let initial = input.initial_measures(&changes, &normalized)?;
        let evolution = if !trend_only && !work.history.stable && work.observed.end / period >= 6 {
            Some(measures::evolution(&work.seasonal, period, multiplicative)?)
        } else {
            None
        };
        Ok(measures::summarize(
            initial,
            evolution,
            !trend_only && work.history.include_m6,
        ))
    }

    /// The detrended original's variance (`None` for `varlog`'s DNOTST), after
    /// the original's and the detrended original's constant-series checks.
    /// `modified` leaves less its linear trend.
    fn detrended_variance(
        &mut self,
        input: &PartFInput<'_>,
        modified: &mut [f64],
        reasons: &mut Vec<Series>,
    ) -> Result<Option<f64>, Reason> {
        let &PartFInput {
            components,
            work,
            original,
            dates,
            adjustments,
            multiplicative,
            ref good,
            ..
        } = input;
        let linear_trend = self.detrend(components.mode, work, original.len(), dates)?;
        let mut original_without_outliers = original.to_vec();
        for effect in [
            &adjustments.level_shift,
            &adjustments.additive_outlier,
            &adjustments.temporary_change,
        ] {
            remove(
                &mut original_without_outliers,
                effect.as_deref(),
                multiplicative,
                Stage::ModifiedOriginalChanges,
            )?;
        }
        if same(&original_without_outliers, good) {
            reasons.push(Series::Original {
                outliers_removed: adjustments.level_shift.is_some()
                    || adjustments.additive_outlier.is_some()
                    || adjustments.temporary_change.is_some(),
            });
        }
        remove(
            modified,
            Some(&linear_trend),
            multiplicative,
            Stage::Detrending,
        )?;
        let detrended_variance = at(
            variance(modified, good, multiplicative),
            Stage::VarianceNormalization,
        )?
        .map(|value| value / 100.0);
        // `x11pt4` divides `varlog`'s DNOTST (-999) by 100 before it compares
        // the variance with DNOTST, so a log sample with no eligible value
        // does not report the detrended original as constant.
        if detrended_variance.is_some_and(|value| build_62_equal(value, 0.0)) {
            reasons.push(Series::DetrendedOriginal);
        }
        Ok(detrended_variance)
    }

    /// Part F of a trend-only pseudo-additive run, which Build 62 always ends
    /// with its constant-series return: `chkzro` clears every `Gudval` when
    /// `Kfulsm` is 2, so `sumry` has no changes to average, `issame` compares
    /// nothing and `varlog` returns zero. The mode takes no regARIMA
    /// adjustments, so none are removed.
    fn pseudo_trend_only_return(
        &mut self,
        input: &PartFInput<'_>,
        modified: &[f64],
        modified_irregular: &[f64],
    ) -> Result<measures::Summary, Reason> {
        let &PartFInput {
            work,
            original,
            dates,
            ref good,
            ..
        } = input;
        let mut series = Vec::new();
        if same(modified, good) {
            series.push(Series::ModifiedOriginal);
        }
        let irregular_variance = at(
            zero_mean_variance(modified_irregular, good, true),
            Stage::IrregularChanges,
        )?;
        if build_62_equal(irregular_variance, 0.0) || same(&work.base_irregular, good) {
            series.push(Series::Irregular);
        }
        if same(&work.base_trend, good) {
            series.push(Series::Trend);
        }
        // `x11pt4` takes the trend-cycle's logarithm and antilogarithm before
        // it returns, which leaves `Stc` changed.
        if let Some(index) = crate::numerics::first_nonpositive(&work.base_trend) {
            return Err(Reason::NonPositiveLogInput {
                series: Series::Trend,
                period: dates.period(index),
            });
        }
        let trend: Vec<_> = work
            .base_trend
            .iter()
            .map(|value| value.ln().exp())
            .collect();
        if trend.iter().all(|value| value.is_finite()) {
            self.trend = Some(trend);
        }
        if same(original, good) {
            series.push(Series::Original {
                outliers_removed: false,
            });
        }
        // `varlog` has no eligible value of the detrended original. Its
        // DNOTST over 100 is neither DNOTST nor zero, so that series is not
        // reported as constant.
        self.level_shift_round_trip = LevelShiftRoundTrip::NotReached;
        Err(Reason::ConstantSeries { series })
    }

    /// Part F's detrending line, fitted to the (log) trend-cycle by least
    /// squares and returned in the original domain. For pseudo-additive it
    /// also keeps the antilogged trend (`Stc`).
    fn detrend(
        &mut self,
        mode: x11::Mode,
        work: &QualityWorkspace,
        n: usize,
        dates: AxisSpan,
    ) -> Result<Vec<f64>, Reason> {
        let multiplicative = mode != x11::Mode::Additive;
        let mut trend = work.base_trend.clone();
        if multiplicative {
            for (i, value) in trend.iter_mut().enumerate() {
                if *value <= 0.0 {
                    return Err(Reason::NonPositiveLogInput {
                        series: Series::Trend,
                        period: dates.period(i),
                    });
                }
                *value = checked(value.ln(), Stage::Detrending)?;
            }
        }
        let mut total = 0.0;
        // `x11pt4`'s `2*i-Pos1ob-Posfob`, with `i` counted from the first
        // observation.
        for (i, value) in trend.iter().enumerate() {
            total = checked(
                total + value * (2.0 * i as f64 - (n - 1) as f64),
                Stage::Detrending,
            )?;
        }
        let nf = n as f64;
        let slope = checked(6.0 * total / (nf * (nf * nf - 1.0)), Stage::Detrending)?;
        let mut linear_trend = (0..n)
            .map(|i| {
                checked(
                    slope * (work.observed.start + i + 1) as f64,
                    Stage::Detrending,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if multiplicative {
            for value in &mut trend {
                *value = checked(value.exp(), Stage::Detrending)?;
            }
            if mode == x11::Mode::PseudoAdditive {
                self.trend = Some(trend); // `x11pt4`'s `antilg(Stc)`
            }
            for value in &mut linear_trend {
                *value = checked(value.exp(), Stage::Detrending)?;
            }
        }
        Ok(linear_trend)
    }
}

impl PartFInput<'_> {
    /// `sumry`'s changes of each component, with the constant-series checks
    /// that interleave them. `modified` leaves without its AO and TC effects.
    fn component_changes(
        &self,
        modified: &mut [f64],
        modified_irregular: &[f64],
        reasons: &mut Vec<Series>,
    ) -> Result<ComponentChanges, Reason> {
        let &Self {
            work,
            adjustments,
            period,
            multiplicative,
            trend_only,
            ref good,
            ..
        } = self;
        remove(
            modified,
            adjustments.additive_outlier.as_deref(),
            multiplicative,
            Stage::ModifiedOriginalChanges,
        )?;
        remove(
            modified,
            adjustments.temporary_change.as_deref(),
            multiplicative,
            Stage::ModifiedOriginalChanges,
        )?;
        let (prior_variance, prior) = if let Some(prior) = &adjustments.prior {
            let prior_changes = changes(prior, good, period, multiplicative, Stage::PriorChanges)?;
            let variance = at(variance(prior, good, multiplicative), Stage::PriorChanges)?.ok_or(
                Reason::NoEligibleValues {
                    stage: Stage::PriorChanges,
                },
            )?;
            (variance, prior_changes)
        } else {
            (0.0, LagChanges::NONE)
        };
        let calendar = if let Some(calendar) = &adjustments.calendar {
            changes(
                calendar,
                good,
                period,
                multiplicative,
                Stage::CalendarChanges,
            )?
        } else {
            LagChanges::NONE
        };
        let irregular = changes(
            &work.base_irregular,
            good,
            period,
            multiplicative,
            Stage::IrregularChanges,
        )?;
        let duration = average_duration(&work.base_irregular);
        // `x11pt4` takes the level shift out here and puts it back below; the
        // round trip can change the last bit of `modified`.
        remove(
            modified,
            adjustments.level_shift.as_deref(),
            multiplicative,
            Stage::ModifiedOriginalChanges,
        )?;
        if same(modified, good) {
            reasons.push(Series::ModifiedOriginal);
        }
        let modified_irregular_changes = changes(
            modified_irregular,
            good,
            period,
            multiplicative,
            Stage::IrregularChanges,
        )?;
        let irregular_variance = at(
            zero_mean_variance(modified_irregular, good, multiplicative),
            Stage::IrregularChanges,
        )?;
        if build_62_equal(irregular_variance, 0.0) || same(&work.base_irregular, good) {
            reasons.push(Series::Irregular);
        }
        restore(
            modified,
            adjustments.level_shift.as_deref(),
            multiplicative,
            Stage::ModifiedOriginalChanges,
        )?;
        let seasonal = if trend_only {
            LagChanges::NONE
        } else {
            changes(
                &work.seasonal[work.observed.clone()],
                good,
                period,
                multiplicative,
                Stage::SeasonalEvolution,
            )?
        };
        let trend = changes(
            &work.base_trend,
            good,
            period,
            multiplicative,
            Stage::TrendChanges,
        )?;
        if same(&work.base_trend, good) {
            reasons.push(Series::Trend);
        }
        Ok(ComponentChanges {
            prior,
            calendar,
            irregular,
            modified_irregular: modified_irregular_changes,
            seasonal,
            trend,
            prior_variance,
            irregular_variance,
            duration,
        })
    }

    /// M1–M7 from the normalized operands and the final stable and moving
    /// seasonality F statistics.
    fn initial_measures(
        &self,
        changes: &ComponentChanges,
        normalized: &Normalized,
    ) -> Result<measures::Measures, Reason> {
        let stable_f = match &self.components.seasonality.final_stable {
            Ok(value) => value.test().statistic(),
            Err(_) => {
                return Err(Reason::PrerequisiteUnavailable {
                    stage: crate::diagnostics::SeasonalityTest::FinalStable,
                });
            }
        };
        let moving_f = match &self.components.seasonality.moving {
            Ok(value) => value.test().statistic(),
            Err(_) => {
                return Err(Reason::PrerequisiteUnavailable {
                    stage: crate::diagnostics::SeasonalityTest::Moving,
                });
            }
        };
        measures::initial(&measures::Inputs {
            trend_only: self.trend_only,
            period: self.period,
            history_count: self.work.observed.end,
            irregular_share: normalized.irregular_share,
            prior_share: normalized.prior_share,
            irregular_variance: normalized.irregular_variance,
            prior_variance: normalized.prior_variance,
            irregular_trend_ratio: self.work.irregular_trend_ratio,
            average_duration: changes.duration,
            span_ratios: &normalized.span_ratios[..self.period],
            moving_seasonality_ratio: self.work.moving_seasonality_ratio.clone()?,
            stable_f,
            moving_f,
        })
    }
}

impl ComponentChanges {
    /// The variances over the detrended original's, the irregular and prior
    /// shares at the one-quarter span, and each span's I/C ratio.
    fn normalize(&self, detrended_variance: f64, period: usize) -> Result<Normalized, Reason> {
        let prior_variance = checked(
            self.prior_variance / detrended_variance,
            Stage::VarianceNormalization,
        )?;
        let irregular_variance = checked(
            self.irregular_variance / detrended_variance,
            Stage::VarianceNormalization,
        )?;
        // M1 reads the shares at the one-quarter span: lag 3 monthly, lag 1 quarterly.
        let quarter_lag = period / 4;
        let mut span_ratios = [0.0; 12];
        let mut irregular_share = 0.0;
        let mut prior_share = 0.0;
        for (lag, ratio) in (1..=period).zip(&mut span_ratios) {
            let total = checked(
                self.modified_irregular.square(lag)
                    + self.trend.square(lag)
                    + self.seasonal.square(lag)
                    + self.prior.square(lag)
                    + self.calendar.square(lag),
                Stage::VarianceNormalization,
            )?;
            if total == 0.0 {
                return Err(Reason::ZeroDenominator {
                    stage: Stage::VarianceNormalization,
                });
            }
            if lag == quarter_lag {
                irregular_share = self.modified_irregular.square(lag) / total;
                prior_share = self.prior.square(lag) / total;
            }
            let trend_mean = self.trend.mean(lag);
            if trend_mean == 0.0 {
                return Err(Reason::ZeroDenominator {
                    stage: Stage::MonthOrQuarterDominance,
                });
            }
            *ratio = checked(
                self.irregular.mean(lag) / trend_mean,
                Stage::MonthOrQuarterDominance,
            )?;
        }
        Ok(Normalized {
            prior_variance,
            irregular_variance,
            irregular_share,
            prior_share,
            span_ratios,
        })
    }
}

/// The original and the irregular modified for extremes: where the extreme
/// weight is zero, the original gives up its irregular and the irregular
/// becomes neutral.
fn modify_original_and_irregular(
    original: &[f64],
    irregular: &[f64],
    weights: &[f64],
    multiplicative: bool,
) -> Result<(Vec<f64>, Vec<f64>), Reason> {
    let mut modified = original.to_vec();
    let mut modified_irregular = irregular.to_vec();
    let outputs = modified.iter_mut().zip(&mut modified_irregular);
    let inputs = original.iter().zip(irregular).zip(weights);
    for ((modified, modified_irregular), ((&original, &irregular), &weight)) in outputs.zip(inputs)
    {
        if weight <= 0.0 {
            if multiplicative && irregular == 0.0 {
                return Err(Reason::ZeroDenominator {
                    stage: Stage::ModifiedOriginalChanges,
                });
            }
            *modified = checked(
                if multiplicative {
                    original / irregular
                } else {
                    original - irregular
                },
                Stage::ModifiedOriginalChanges,
            )?;
            *modified_irregular = if multiplicative { 1.0 } else { 0.0 };
        }
    }
    Ok((modified, modified_irregular))
}

/// `sumry`: the mean absolute change over `lag` periods (in percent when
/// multiplicative) and its square. A change counts when the observation it
/// starts from is eligible.
fn lag_changes(
    values: &[f64],
    good: &[bool],
    lag: usize,
    multiplicative: bool,
) -> Result<(f64, f64), ArithmeticFailure> {
    let mut total = 0.0;
    let mut count = 0.0;
    let pairs = values.iter().zip(values.iter().skip(lag)).zip(good);
    for ((&start, &end), _) in pairs.filter(|(_, eligible)| **eligible) {
        let mut change = finite(end - start)?;
        if multiplicative {
            if start == 0.0 {
                return Err(ArithmeticFailure::ZeroDenominator);
            }
            change = finite(finite(change * 100.0)? / start)?;
        }
        total = finite(total + change.abs())?;
        count += 1.0;
    }
    if count == 0.0 {
        return Err(ArithmeticFailure::NoEligibleValues);
    }
    let mean = total / count;
    Ok((mean, finite(mean * mean)?))
}

/// Whether `varlog` reads a value. Additive variance is unmasked, so `good` is
/// read only when multiplicative.
fn variance_eligible(good: &[bool], multiplicative: bool, index: usize, value: f64) -> bool {
    !multiplicative || (good[index] && value > 0.0)
}

/// Sum of squares of the eligible values about `mean`.
fn square_sum(
    values: &[f64],
    good: &[bool],
    multiplicative: bool,
    mean: f64,
) -> Result<f64, ArithmeticFailure> {
    let mut sum = 0.0;
    for (i, &value) in values.iter().enumerate() {
        if variance_eligible(good, multiplicative, i, value) {
            let difference = if multiplicative { value.ln() } else { value } - mean;
            sum = finite(sum + difference * difference)?;
        }
    }
    Ok(sum)
}

/// `varlog` about the sample mean. `None` stands for its DNOTST return when no
/// value is eligible.
fn variance(
    values: &[f64],
    good: &[bool],
    multiplicative: bool,
) -> Result<Option<f64>, ArithmeticFailure> {
    let mut mean = 0.0;
    let mut count = 0.0;
    for (i, &value) in values.iter().enumerate() {
        if variance_eligible(good, multiplicative, i, value) {
            mean = finite(mean + if multiplicative { value.ln() } else { value })?;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return Ok(None);
    }
    mean /= count;
    square_sum(values, good, multiplicative, mean).map(Some)
}

/// `varlog` about zero, which never takes the DNOTST return.
fn zero_mean_variance(
    values: &[f64],
    good: &[bool],
    multiplicative: bool,
) -> Result<f64, ArithmeticFailure> {
    square_sum(values, good, multiplicative, 0.0)
}

/// `issame`: whether every eligible value after the first equals the first.
fn same(values: &[f64], good: &[bool]) -> bool {
    values
        .iter()
        .zip(good)
        .skip(1)
        .all(|(&value, &good)| !good || build_62_equal(value, values[0]))
}

/// Average duration of run: the number of changes per run of strict rises or
/// falls. A leading plateau takes the direction of its first strict move.
fn average_duration(values: &[f64]) -> f64 {
    let last = values.len() - 1;
    let (mut rising, start) = if values[0] < values[1] {
        (true, 0)
    } else if build_62_equal(values[0], values[1]) {
        let first_move = (1..last).find_map(|i| {
            if values[i] < values[i + 1] {
                Some((true, i))
            } else if values[i] > values[i + 1] {
                Some((false, i))
            } else {
                None
            }
        });
        let Some(first_move) = first_move else {
            return last as f64;
        };
        first_move
    } else {
        (false, 0)
    };
    let mut runs = 1.0;
    for pair in values[start + 1..].windows(2) {
        if (rising && pair[0] > pair[1]) || (!rising && pair[0] < pair[1]) {
            rising = !rising;
            runs += 1.0;
        }
    }
    last as f64 / runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::UnavailableKind;
    use crate::time::TimeAxis;

    fn calculate(
        components: &super::super::Components,
        original: &[f64],
        axis: TimeAxis,
        adjustments: &QualityAdjustments,
    ) -> Result<measures::Summary, Reason> {
        let dates = axis.span(original.len()).unwrap();
        super::calculate(components, original, dates, adjustments).1
    }

    #[test]
    fn eligibility_stays_cleared_after_a_positive_final_adjustment() {
        let axis = TimeAxis::quarterly(2000, 1).unwrap();
        let original = [2.0; 3];
        let seasonal = [3.0; 3];
        let intermediate = crate::x11::pseudo_additive::adjusted_series(
            &original,
            &[2.0, 1.0, 0.5],
            &seasonal,
            axis.span(3).unwrap(),
            crate::error::X11PseudoAdditiveStage::FinalAdjustment,
        )
        .unwrap();
        let adjusted = crate::x11::pseudo_additive::adjusted_series(
            &original,
            &[0.5; 3],
            &seasonal,
            axis.span(3).unwrap(),
            crate::error::X11PseudoAdditiveStage::FinalAdjustment,
        )
        .unwrap();
        assert_eq!(intermediate, [-2.0, 0.0, 1.0]);
        assert_eq!(adjusted, [1.0; 3]);
        let eligible = intermediate_eligibility(x11::Mode::PseudoAdditive, &intermediate, 0..3);
        assert_eq!(eligible, [false, false, true]);
        assert_eq!(
            final_eligibility(x11::Mode::PseudoAdditive, &eligible, &original, &adjusted),
            [false, false, true],
        );
    }

    #[test]
    fn ineligible_start_is_left_out_of_percent_changes() {
        let original = [2.0, 8.0, 4.0, 6.0];
        let good = final_eligibility(
            x11::Mode::PseudoAdditive,
            &[false, true, true, true],
            &original,
            &[1.0; 4],
        );
        // `sumry` works in percent: the eligible starts give 50 and 50, and
        // the first start would add 300.
        assert_eq!(
            lag_changes(&original, &good, 1, true).unwrap(),
            (50.0, 2500.0)
        );
        assert_eq!(
            lag_changes(&original, &[true; 4], 1, true).unwrap().0,
            400.0 / 3.0
        );
    }

    #[test]
    fn only_pseudo_additive_clears_eligibility() {
        let intermediate = [-1.0, 7.0, -3.0, 0.0, 5.0, -6.0];
        assert_eq!(
            intermediate_eligibility(x11::Mode::PseudoAdditive, &intermediate, 1..5),
            [true, false, false, true],
        );
        assert_eq!(
            final_eligibility(
                x11::Mode::PseudoAdditive,
                &[false, true, true, true],
                &[1.0, 0.0, 2.0, 3.0],
                &[1.0, 1.0, 0.0, 4.0],
            ),
            [false, false, false, true],
        );
        for mode in [
            x11::Mode::Additive,
            x11::Mode::Multiplicative,
            x11::Mode::LogAdditive,
        ] {
            assert_eq!(
                intermediate_eligibility(mode, &intermediate, 1..5),
                [true; 4]
            );
            assert_eq!(
                final_eligibility(mode, &[false; 4], &[0.0; 4], &[0.0; 4]),
                [true; 4]
            );
        }
    }

    #[test]
    fn zero_weight_replaces_original_and_irregular() {
        assert_eq!(
            modify_original_and_irregular(&[10.0; 3], &[2.0; 3], &[0.0, 0.3, 1.0], false).unwrap(),
            (vec![8.0, 10.0, 10.0], vec![0.0, 2.0, 2.0])
        );
        assert_eq!(
            modify_original_and_irregular(&[10.0; 3], &[2.0; 3], &[0.0, 0.3, 1.0], true).unwrap(),
            (vec![5.0, 10.0, 10.0], vec![1.0, 2.0, 2.0])
        );
    }

    #[test]
    fn level_shift_round_trip_rounds() {
        let mut additive = [1.0];
        remove(
            &mut additive,
            Some(&[1e16]),
            false,
            Stage::ModifiedOriginalChanges,
        )
        .unwrap();
        restore(
            &mut additive,
            Some(&[1e16]),
            false,
            Stage::ModifiedOriginalChanges,
        )
        .unwrap();
        assert_eq!(additive, [0.0]);
        let mut multiplicative = [0.1];
        remove(
            &mut multiplicative,
            Some(&[1.3498588075760032]),
            true,
            Stage::ModifiedOriginalChanges,
        )
        .unwrap();
        restore(
            &mut multiplicative,
            Some(&[1.3498588075760032]),
            true,
            Stage::ModifiedOriginalChanges,
        )
        .unwrap();
        assert_eq!(multiplicative, [0.10000000000000002]);
    }

    #[test]
    fn s3x15_replaced_by_stable_counts_as_a_stable_filter() {
        let axis = TimeAxis::monthly(2000, 3).unwrap();
        let prepared = crate::x11::PreparedOptions::new(
            crate::x11::Options::default().with_seasonal_filter(crate::x11::SeasonalFilter::S3x15),
            axis,
            120,
        )
        .unwrap();
        let history = QualityFilterHistory::resolve(
            prepared.seasonal_filters.as_slice(),
            prepared.seasonal_filters.as_slice(),
            None,
        );
        assert!(history.stable);
        assert!(!history.include_m6);
    }

    #[test]
    fn m6_enters_q_only_for_a_requested_or_selected_3x5() {
        use crate::{x11, x11::SeasonalFilter as F};
        let requested = [F::Msr; 4];
        let resolved = [F::S3x5; 4];
        assert!(
            !QualityFilterHistory::resolve(
                &requested,
                &resolved,
                Some(&x11::MsrSelection::ShortHistoryFallback { passes: 0 })
            )
            .include_m6
        );
        assert!(
            QualityFilterHistory::resolve(
                &requested,
                &resolved,
                Some(&x11::MsrSelection::Selected {
                    filter: x11::MsrFilter::S3x5,
                    passes: 1
                })
            )
            .include_m6
        );
        assert!(
            QualityFilterHistory::resolve(
                &[F::X11Default, F::S3x5, F::X11Default, F::S3x5],
                &resolved,
                None
            )
            .include_m6
        );
        let mixed = QualityFilterHistory::resolve(
            &[F::Msr, F::Stable, F::S3x5, F::S3x5],
            &[F::S3x5, F::Stable, F::S3x5, F::S3x5],
            Some(&x11::MsrSelection::Selected {
                filter: x11::MsrFilter::S3x5,
                passes: 1,
            }),
        );
        assert!(mixed.stable);
        assert!(!mixed.include_m6);
        assert!(
            !QualityFilterHistory::resolve(
                &[F::S3x1, F::Msr, F::S3x5, F::S3x5],
                &[F::S3x1, F::S3x5, F::S3x5, F::S3x5],
                Some(&x11::MsrSelection::SuppressedByFirstPeriodS3x1)
            )
            .include_m6
        );
    }

    fn components(count: usize) -> (Vec<f64>, TimeAxis, crate::x11::Components) {
        let values = (0..count)
            .map(|i| {
                100.0
                    + i as f64 * 0.3
                    + [
                        3.0, -2.0, 1.0, -4.0, 2.0, 7.0, -1.0, 4.0, 2.0, -3.0, 1.0, 6.0,
                    ][i % 12]
                    + ((i * 17) % 11) as f64 * 0.17
            })
            .collect::<Vec<_>>();
        let axis = TimeAxis::monthly(2000, 3).unwrap();
        let c = crate::x11::adjust(&values, axis, x11::Mode::Additive, Default::default()).unwrap();
        (values, axis, c)
    }

    #[test]
    fn part_f_records_how_far_it_got() {
        use crate::x11::spectral::LevelShiftRoundTrip as RoundTrip;
        let (values, axis, mut c) = components(180);
        c.quality_workspace.moving_seasonality_ratio = Err(Reason::NonFiniteArithmetic {
            stage: Stage::FinalMovingSeasonalityRatio,
        });
        let quality = complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        assert!(quality.is_err());
        assert_eq!(
            c.spectral_workspace.level_shift_round_trip,
            RoundTrip::Reached
        );

        let values = vec![100.0; 120];
        let mut c =
            crate::x11::adjust(&values, axis, x11::Mode::PseudoAdditive, Default::default())
                .unwrap();
        let _ = complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        assert_eq!(
            c.spectral_workspace.level_shift_round_trip,
            RoundTrip::NotReached
        );

        let (values, axis, mut c) = components(180);
        c.quality_workspace.base_irregular[7] = f64::INFINITY;
        let _ = complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        assert_eq!(
            c.spectral_workspace.level_shift_round_trip,
            RoundTrip::Unknown
        );
    }

    #[test]
    fn constant_series_returns_before_the_moving_seasonality_ratio_is_read() {
        let axis = TimeAxis::monthly(1980, 7).unwrap();
        let values = vec![100.0; 120];
        let mut c =
            crate::x11::adjust(&values, axis, x11::Mode::PseudoAdditive, Default::default())
                .unwrap();
        c.quality_workspace.moving_seasonality_ratio = Err(Reason::NonFiniteArithmetic {
            stage: Stage::FinalMovingSeasonalityRatio,
        });
        assert!(matches!(
            calculate(&c, &values, axis, &Default::default()),
            Err(Reason::ConstantSeries { .. })
        ));
    }

    #[test]
    fn empty_log_sample_does_not_report_the_detrended_original() {
        let (values, axis, mut c) = components(180);
        c.mode = x11::Mode::Multiplicative;
        c.intermediate_extreme_weights.fill(0.0);
        c.quality_workspace.base_irregular = (0..180).map(|i| -1.0 - i as f64 * 0.01).collect();
        // Zero weights make the modified irregular all ones, which is
        // constant. They also make the modified original negative, so its
        // detrended logarithms have no sample, but `varlog`'s -999 over 100
        // is not DNOTST and the detrended original is not reported.
        assert_eq!(
            calculate(&c, &values, axis, &Default::default()).unwrap_err(),
            Reason::ConstantSeries {
                series: vec![Series::Irregular]
            }
        );
    }

    #[test]
    fn nonfinite_irregular_makes_quality_unavailable() {
        let (values, axis, mut c) = components(180);
        c.quality_workspace.base_irregular[7] = f64::INFINITY;
        let quality = complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        let unavailable = quality.unwrap_err();
        assert_eq!(unavailable.kind(), UnavailableKind::Numerical);
        assert_eq!(
            unavailable.to_string(),
            "the M and Q measures are unavailable: changes in the irregular: a value is not finite"
        );
        assert!(matches!(
            unavailable,
            Unavailable(UnavailableDetail::Quality {
                reason: Reason::NonFiniteArithmetic {
                    stage: Stage::IrregularChanges
                },
                ..
            })
        ));
    }

    #[test]
    fn nonpositive_trend_is_reported_at_its_date() {
        let (values, axis, mut c) = components(180);
        c.mode = x11::Mode::Multiplicative;
        // The last value starts no change, so the failure is the logarithm
        // of the trend-cycle and not an earlier percent change.
        c.quality_workspace.base_trend[179] = 0.0;
        assert_eq!(
            calculate(&c, &values, axis, &Default::default()).unwrap_err(),
            Reason::NonPositiveLogInput {
                series: Series::Trend,
                period: axis.period_at(179).unwrap()
            }
        );
    }

    #[test]
    fn lag_changes_reports_no_start_a_zero_start_and_overflow() {
        assert_eq!(
            lag_changes(&[1.0, 2.0], &[false; 2], 1, false),
            Err(ArithmeticFailure::NoEligibleValues)
        );
        assert_eq!(
            lag_changes(&[0.0, 2.0], &[true; 2], 1, true),
            Err(ArithmeticFailure::ZeroDenominator)
        );
        assert_eq!(
            lag_changes(&[f64::MAX, -f64::MAX], &[true; 2], 1, false),
            Err(ArithmeticFailure::NonFinite)
        );
    }

    #[test]
    fn seasonal_evolution_needs_six_complete_years() {
        for (count, available) in [(71, false), (72, true)] {
            let (values, axis, mut c) = components(count);
            let value = complete(
                &mut c,
                &values,
                axis.span(values.len()).unwrap(),
                Default::default(),
            )
            .expect("the measures are available");
            assert_eq!(value.seasonal_evolution().is_some(), available);
        }
    }

    #[test]
    fn sumry_masks_only_the_start_and_squares_the_mean_absolute_change() {
        assert_eq!(
            lag_changes(&[1.0, 3.0, 9.0], &[true, false, true], 1, false).unwrap(),
            (2.0, 4.0)
        );
        assert_eq!(
            lag_changes(&[1.0, 3.0, 9.0], &[true; 3], 1, false).unwrap(),
            (4.0, 16.0)
        );
    }

    #[test]
    fn sumry_multiplicative_changes_use_starting_denominator() {
        assert_eq!(
            lag_changes(&[2.0, 4.0, 2.0], &[true; 3], 1, true).unwrap(),
            (75.0, 5625.0)
        );
    }

    #[test]
    fn additive_variance_is_unmasked_sum_of_squares() {
        assert_eq!(
            variance(&[1.0, 3.0, 8.0], &[true, false, true], false).unwrap(),
            Some(26.0)
        );
    }

    #[test]
    fn logarithmic_variance_skips_nonpositive_and_ineligible_values() {
        assert_eq!(
            zero_mean_variance(&[1.0, 0.0, -2.0], &[true; 3], true).unwrap(),
            0.0
        );
        assert_eq!(
            zero_mean_variance(&[2.0, 8.0], &[false; 2], true).unwrap(),
            0.0
        );
    }

    #[test]
    fn same_compares_with_the_first_value_even_when_it_is_ineligible() {
        assert!(!same(&[1.0, 2.0, 2.0], &[false, true, true]));
        assert!(same(&[1.0, 9.0, 1.0], &[false, false, true]));
    }

    #[test]
    fn duration_counts_strict_turns_and_keeps_plateaus_inside_runs() {
        assert_eq!(average_duration(&[1.0, 1.0, 2.0, 2.0, 1.0]), 2.0);
        assert_eq!(average_duration(&[1.0, 2.0, 3.0, 4.0]), 3.0);
    }

    #[test]
    fn duration_gives_a_leading_plateau_the_direction_of_its_first_move() {
        assert_eq!(average_duration(&[2.0; 5]), 4.0);
        assert_eq!(average_duration(&[3.0, 3.0]), 1.0);
        assert_eq!(average_duration(&[1.0, 1.0, 1.0, 0.5, 2.0, 3.0]), 2.5);
        // Signed zeros are equal.
        assert_eq!(average_duration(&[-0.0, 0.0, 1.0, 0.0]), 1.5);
        assert_eq!(average_duration(&[0.0, -0.0, -0.0]), 2.0);
    }

    fn quality(
        mode: x11::Mode,
        options: crate::x11::Options,
    ) -> Result<diagnostics::QualityMeasures, Unavailable> {
        let (values, axis, _) = components(180);
        let mut c = crate::x11::adjust(&values, axis, mode, options).unwrap();
        complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        )
    }

    #[test]
    fn trend_only_has_no_m6_and_no_seasonal_evolution() {
        let options =
            crate::x11::Options::default().with_decomposition(crate::x11::Decomposition::TrendOnly);
        let value = quality(x11::Mode::Additive, options).expect("the measures are available");
        assert_eq!(value.m6(), None);
        assert!(!value.m6_included_in_q());
        assert_eq!(value.seasonal_evolution(), None);
    }

    #[test]
    fn stable_filter_has_no_seasonal_evolution() {
        let options =
            crate::x11::Options::default().with_seasonal_filter(crate::x11::SeasonalFilter::Stable);
        let value = quality(x11::Mode::Additive, options).expect("the measures are available");
        assert!(value.m6().is_some());
        assert!(!value.m6_included_in_q());
        assert_eq!(value.seasonal_evolution(), None);
    }

    #[test]
    fn trend_only_pseudo_additive_reports_every_series_but_the_detrended_original() {
        let options =
            crate::x11::Options::default().with_decomposition(crate::x11::Decomposition::TrendOnly);
        let unavailable = quality(x11::Mode::PseudoAdditive, options)
            .expect_err("Part F returns on a constant series");
        assert_eq!(unavailable.kind(), UnavailableKind::NotApplicable);
        let Unavailable(UnavailableDetail::Quality { reason, .. }) = unavailable else {
            panic!("the quality measures are what is unavailable")
        };
        assert_eq!(
            reason,
            Reason::ConstantSeries {
                series: vec![
                    Series::ModifiedOriginal,
                    Series::Irregular,
                    Series::Trend,
                    Series::Original {
                        outliers_removed: false
                    },
                ]
            }
        );
    }

    #[test]
    fn part_f_keeps_its_trend_only_for_pseudo_additive() {
        let (values, axis, mut c) = components(180);
        let _ = complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        assert_eq!(c.spectral_workspace.part_f_trend, None);

        let mut c =
            crate::x11::adjust(&values, axis, x11::Mode::PseudoAdditive, Default::default())
                .unwrap();
        let _ = complete(
            &mut c,
            &values,
            axis.span(values.len()).unwrap(),
            Default::default(),
        );
        // The logarithm and antilogarithm do not round back to the trend.
        let expected: Vec<_> = c
            .quality_workspace
            .base_trend
            .iter()
            .map(|value| value.ln().exp())
            .collect();
        assert_ne!(expected, c.quality_workspace.base_trend);
        assert_eq!(c.spectral_workspace.part_f_trend, Some(expected));
    }
}
