//! Build 62 `f3cal`: the M1–M11 quality measures and their Q summaries.
use super::checked;
use crate::diagnostics::{X11QualityStage as Stage, X11QualityUnavailableReason as Reason};

/// Part F's operands for M1–M7.
pub(super) struct Inputs<'a> {
    pub(super) trend_only: bool,
    pub(super) period: usize,
    pub(super) history_count: usize,
    /// The irregular's share of the change variance at the one-quarter span (M1).
    pub(super) irregular_share: f64,
    /// The prior adjustment's share of that variance (M1).
    pub(super) prior_share: f64,
    /// Irregular variance as a percent of the detrended original's (M2).
    pub(super) irregular_variance: f64,
    /// Prior-adjustment variance on the same scale (M2).
    pub(super) prior_variance: f64,
    /// The final I/C ratio (M3).
    pub(super) irregular_trend_ratio: f64,
    /// Average duration of run of the irregular (M4).
    pub(super) average_duration: f64,
    /// I/C ratio of the mean changes at each span `1..=period` (M5).
    pub(super) span_ratios: &'a [f64],
    /// The moving seasonality ratio (M6).
    pub(super) moving_seasonality_ratio: f64,
    pub(super) stable_f: f64,
    pub(super) moving_f: f64,
}

/// M1–M7, which every quality summary computes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Measures {
    pub(super) m1: f64,
    pub(super) m2: f64,
    pub(super) m3: f64,
    pub(super) m4: f64,
    pub(super) m5: f64,
    /// Zero for a trend-only decomposition, which has no M6.
    pub(super) m6: f64,
    pub(super) m7: f64,
}

impl Measures {
    fn values(self) -> [f64; 7] {
        [
            self.m1, self.m2, self.m3, self.m4, self.m5, self.m6, self.m7,
        ]
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            m1: f(self.m1),
            m2: f(self.m2),
            m3: f(self.m3),
            m4: f(self.m4),
            m5: f(self.m5),
            m6: f(self.m6),
            m7: f(self.m7),
        }
    }
}

/// M8–M11, the seasonal-evolution measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Evolution {
    /// Year-to-year change over the whole history.
    pub(super) m8: f64,
    /// Linear movement over the whole history.
    pub(super) m9: f64,
    /// Year-to-year change in the recent years.
    pub(super) m10: f64,
    /// Linear movement in the recent years.
    pub(super) m11: f64,
}

impl Evolution {
    pub(super) fn values(self) -> [f64; 4] {
        [self.m8, self.m9, self.m10, self.m11]
    }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            m8: f(self.m8),
            m9: f(self.m9),
            m10: f(self.m10),
            m11: f(self.m11),
        }
    }
}

/// M1–M7, before they are clamped.
pub(super) fn initial(input: &Inputs<'_>) -> Result<Measures, Reason> {
    let months_per_period = (12 / input.period).max(1) as f64;
    if 1.0 - input.prior_share == 0.0 {
        return Err(Reason::ZeroDenominator { stage: Stage::M1 });
    }
    let m1 = checked(
        (input.irregular_share / (1.0 - input.prior_share)) / 0.10,
        Stage::M1,
    )?;
    if (100.0 - input.prior_variance).abs() == 0.0 {
        return Err(Reason::ZeroDenominator { stage: Stage::M2 });
    }
    let m2 = checked(
        (input.irregular_variance / (100.0 - input.prior_variance).abs()) / 0.10,
        Stage::M2,
    )?;
    let m3 = checked(
        (input.irregular_trend_ratio * months_per_period - 1.0) / 2.0,
        Stage::M3,
    )?;
    let history = input.history_count as f64;
    if input.average_duration == 0.0 {
        return Err(Reason::ZeroDenominator { stage: Stage::M4 });
    }
    let m4 = checked(
        (3.0 * (history - 1.0) / input.average_duration - 2.0 * history + 1.0).abs()
            / ((1.6 * history - 2.9).sqrt() * 2.577),
        Stage::M4,
    )?;
    let m5 = m5(input.span_ratios, input.period)?;
    let m6 = if input.trend_only {
        0.0
    } else {
        checked(
            (input.moving_seasonality_ratio - 4.0).abs() / 2.5,
            Stage::M6,
        )?
    };
    let stable = input.stable_f;
    let mut stable_term = 9.0;
    if 9.0 * stable >= 7.0 {
        stable_term = 7.0 / stable;
    }
    let moving_term = if stable > 0.0 {
        // An infinite ratio is a failure, not a value for the cap of nine.
        checked((3.0 * input.moving_f) / stable, Stage::M7)?.min(9.0)
    } else {
        9.0
    };
    let m7 = checked(((stable_term + moving_term) / 2.0).sqrt(), Stage::M7)?;
    Ok(Measures {
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
    })
}

/// M5 from the months (or quarters) for cyclical dominance, MCD: the first
/// span whose I/C ratio falls below one, interpolated between spans.
fn m5(span_ratios: &[f64], period: usize) -> Result<f64, Reason> {
    let mut span = period;
    while span > 0 && span_ratios[span - 1] < 1.0 {
        span -= 1;
    }
    span = (span + 1).min(period);
    let months_per_period = (12 / period).max(1) as f64;
    let dominance = if span == 1 {
        let denominator = span_ratios[0] - span_ratios[1];
        if denominator == 0.0 {
            return Err(Reason::ZeroDenominator { stage: Stage::M5 });
        }
        checked(1.0 + (span_ratios[0] - 1.0) / denominator, Stage::M5)?.clamp(0.5, 1.0)
    } else {
        let difference = span_ratios[span - 2] - span_ratios[span - 1];
        if (difference < 0.0 || crate::numerics::build_62_equal(difference, 0.0)) && span == period
        {
            months_per_period * 15.5
        } else {
            if difference == 0.0 {
                return Err(Reason::ZeroDenominator { stage: Stage::M5 });
            }
            checked(
                span as f64 + (span_ratios[span - 1] - 1.0) / difference,
                Stage::M5,
            )?
        }
    };
    checked((dominance * months_per_period - 0.5) / 5.0, Stage::M5)
}

/// M8–M11, from the seasonal factors scaled by their deviation about the
/// neutral value.
pub(super) fn evolution(
    seasonal: &[f64],
    period: usize,
    multiplicative: bool,
) -> Result<Evolution, Reason> {
    let center = if multiplicative { 1.0 } else { 0.0 };
    let scale = seasonal_scale(seasonal, center)?;
    if scale == 0.0 {
        return Err(Reason::ZeroDenominator {
            stage: Stage::SeasonalEvolution,
        });
    }
    let normalized = seasonal
        .iter()
        .map(|value| checked((value - center) / scale, Stage::SeasonalEvolution))
        .collect::<Result<Vec<_>, _>>()?;
    let mut year_to_year = 0.0;
    let mut year_to_year_count = 0.0;
    let mut linear = 0.0;
    let mut recent_year_to_year = 0.0;
    let mut recent_year_to_year_count = 0.0;
    let mut recent_linear = 0.0;
    let mut recent_slots = 0.0;
    for slot in 0..period {
        let mut slot_changes = 0.0;
        let mut last = slot;
        for year in (slot + period..seasonal.len()).step_by(period) {
            slot_changes += 1.0;
            year_to_year_count += 1.0;
            year_to_year += (normalized[year] - normalized[year - period]).abs();
            last = year;
        }
        linear += (normalized[last] - normalized[slot]).abs() / slot_changes;
        // The recent three years end two years before the slot's last value.
        let recent_end = last - 2 * period;
        if let Some(recent_start) = recent_end.checked_sub(3 * period) {
            recent_linear += (normalized[recent_end] - normalized[recent_start]).abs() / 3.0;
            recent_slots += 1.0;
            for year in (recent_start + period..=recent_end).step_by(period) {
                recent_year_to_year_count += 1.0;
                recent_year_to_year += (normalized[year] - normalized[year - period]).abs();
            }
        }
    }
    year_to_year /= year_to_year_count;
    linear /= period as f64;
    if !crate::numerics::build_62_equal(recent_slots, 0.0) {
        recent_linear /= recent_slots;
        recent_year_to_year /= recent_year_to_year_count;
    }
    let evolution = Evolution {
        m8: 10.0 * year_to_year,
        m9: 10.0 * linear,
        m10: 10.0 * recent_year_to_year,
        m11: 10.0 * recent_linear,
    };
    for value in evolution.values() {
        checked(value, Stage::SeasonalEvolution)?;
    }
    Ok(evolution)
}

/// `sdev`: the root mean square of the seasonal factors about `center`.
fn seasonal_scale(seasonal: &[f64], center: f64) -> Result<f64, Reason> {
    let mut sum = 0.0;
    let mut count = 0.0;
    for &value in seasonal {
        // `sdev` leaves out a value equal to Build 62's DNOTST (-999) and
        // does not consult `Gudval`.
        if crate::numerics::build_62_equal(value, -999.0) {
            continue;
        }
        let difference = value - center;
        sum = checked(sum + difference * difference, Stage::SeasonalEvolution)?;
        count += 1.0;
    }
    if count == 0.0 {
        return Err(Reason::NoEligibleValues {
            stage: Stage::SeasonalEvolution,
        });
    }
    Ok((sum / count).sqrt())
}

/// The measures clamped to `[0, 3]`, with Q and Q without M2.
#[derive(Debug, PartialEq)]
pub(super) struct Summary {
    pub(super) measures: Measures,
    /// Present when the history allows the seasonal-evolution measures.
    pub(super) evolution: Option<Evolution>,
    pub(super) q: f64,
    pub(super) q_without_m2: f64,
    pub(super) failed: usize,
}

/// Clamps the measures, counts the failures and averages Q, in which M6 has
/// a weight only when `include_m6`.
pub(super) fn summarize(
    measures: Measures,
    evolution: Option<Evolution>,
    include_m6: bool,
) -> Summary {
    let clamp = |value: f64| value.clamp(0.0, 3.0);
    let measures = measures.map(clamp);
    let evolution = evolution.map(|evolution| evolution.map(clamp));
    // An excluded M6 still counts as a failure.
    let failed = measures
        .values()
        .into_iter()
        .chain(evolution.into_iter().flat_map(Evolution::values))
        .filter(|&value| value >= 1.0)
        .count();
    let Measures {
        m1,
        m2,
        m3,
        m4,
        m5,
        m6,
        m7,
    } = measures;
    // Without the evolution measures, f3cal weights M7 in place of M8 and M9,
    // M1 in place of M10 and M2 in place of M11.
    let [m8, m9, m10, m11] = evolution.map_or([m7, m7, m1, m2], Evolution::values);
    let terms = [
        Some((m1, 10.0)),
        Some((m2, 11.0)),
        Some((m3, 10.0)),
        Some((m4, 8.0)),
        Some((m5, 11.0)),
        include_m6.then_some((m6, 10.0)),
        Some((m7, 18.0)),
        Some((m8, 7.0)),
        Some((m9, 7.0)),
        Some((m10, 4.0)),
        Some((m11, 4.0)),
    ];
    let mut q = 0.0;
    let mut total_weight = 0.0;
    for (value, weight) in terms.into_iter().flatten() {
        q += value * weight;
        total_weight += weight;
    }
    q /= total_weight;
    // M2's weight, including the M11 slot it fills without the evolution measures.
    let m2_weight = if evolution.is_some() { 11.0 } else { 15.0 };
    let q_without_m2 = ((q * total_weight) - (m2 * m2_weight)) / (total_weight - m2_weight);
    Summary {
        measures,
        evolution,
        q,
        q_without_m2,
        failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measures(m: [f64; 11]) -> (Measures, Evolution) {
        (
            Measures {
                m1: m[0],
                m2: m[1],
                m3: m[2],
                m4: m[3],
                m5: m[4],
                m6: m[5],
                m7: m[6],
            },
            Evolution {
                m8: m[7],
                m9: m[8],
                m10: m[9],
                m11: m[10],
            },
        )
    }

    /// `summarize` over M1–M11 in order, with or without M8–M11.
    fn summarize_all(m: [f64; 11], with_evolution: bool, include_m6: bool) -> Summary {
        let (initial, evolution) = measures(m);
        summarize(initial, with_evolution.then_some(evolution), include_m6)
    }

    #[test]
    fn seasonal_scale_skips_only_dnotst_values() {
        assert_eq!(
            seasonal_scale(&[-999.0, 3.0, 4.0], 0.0).unwrap(),
            (25.0_f64 / 2.0).sqrt()
        );
        assert_eq!(
            seasonal_scale(&[0.0, 3.0, 4.0], 0.0).unwrap(),
            (25.0_f64 / 3.0).sqrt()
        );
        assert_eq!(
            seasonal_scale(&[-999.0], 0.0),
            Err(Reason::NoEligibleValues {
                stage: Stage::SeasonalEvolution
            })
        );
    }

    #[test]
    fn seasonal_evolution_handles_a_partial_last_year() {
        // The scale is exactly 2, so every normalized value is exactly ±1.
        let values = (0..26)
            .map(|i| if (i / 4) % 2 == 0 { 2.0 } else { -2.0 })
            .collect::<Vec<_>>();
        assert_eq!(
            evolution(&values, 4, false).unwrap().values(),
            [20.0, 2.0, 20.0, 6.666666666666666]
        );
        assert_eq!(
            evolution(&values[..24], 4, false).unwrap().values(),
            [20.0, 4.0, 20.0, 6.666666666666666]
        );
    }

    #[test]
    fn m7_averages_the_stable_and_moving_terms() {
        let input = Inputs {
            trend_only: false,
            period: 4,
            history_count: 80,
            irregular_share: 0.01,
            prior_share: 0.0,
            irregular_variance: 1.0,
            prior_variance: 0.0,
            irregular_trend_ratio: 1.0,
            average_duration: 1.5,
            span_ratios: &[1.0; 4],
            moving_seasonality_ratio: 4.0,
            stable_f: 10.0,
            moving_f: 2.0,
        };
        // In f64, (0.7 + 0.6) / 2 is 0.6499999999999999, not 0.65.
        assert_eq!(initial(&input).unwrap().m7, 0.6499999999999999_f64.sqrt());
    }

    #[test]
    fn quarterly_m5_with_and_without_a_dominant_span() {
        // No span's I/C ratio falls below one, and the three months per
        // quarter then apply twice: (3 * 15.5 * 3 - 0.5) / 5.
        assert_eq!(m5(&[1.0; 4], 4).unwrap(), 27.8);
        assert_eq!(m5(&[0.9, 0.7, 0.5, 0.4], 4).unwrap(), 0.20000000000000012);
        assert_eq!(m5(&[0.75, 0.25, 0.125, 0.0625], 4).unwrap(), 0.2);
    }

    #[test]
    fn q_without_m2_can_round_below_zero() {
        let mut m = [0.0; 11];
        m[1] = 0.039;
        let result = summarize_all(m, true, true);
        assert_eq!(result.q, 0.0042899999999999995);
        assert_eq!(result.q_without_m2, -6.237208003512115e-19);
    }

    #[test]
    fn all_computed_half_measures_have_half_quality() {
        let (measures, evolution) = measures([0.5; 11]);
        assert_eq!(
            summarize_all([0.5; 11], true, true),
            Summary {
                measures,
                evolution: Some(evolution),
                q: 0.5,
                q_without_m2: 0.5,
                failed: 0,
            }
        );
    }

    #[test]
    fn excluded_m6_still_counts_as_failed() {
        let mut m = [0.0; 11];
        m[5] = 3.0;
        let (measures, evolution) = measures(m);
        assert_eq!(
            summarize_all(m, true, false),
            Summary {
                measures,
                evolution: Some(evolution),
                q: 0.0,
                q_without_m2: 0.0,
                failed: 1
            }
        );
    }

    #[test]
    fn failure_count_uses_unrounded_values_at_one() {
        for (value, failed) in [(1.0_f64.next_down(), 0), (1.0, 1), (1.0_f64.next_up(), 1)] {
            let mut m = [0.0; 11];
            m[0] = value;
            assert_eq!(summarize_all(m, true, true).failed, failed);
        }
    }

    #[test]
    fn clamps_before_quality_and_failure_count() {
        let result = summarize_all(
            [-1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            true,
            true,
        );
        assert_eq!([result.measures.m1, result.measures.m2], [0.0, 3.0]);
        assert_eq!(result.failed, 1);
        assert_eq!(result.q, 0.33);
        assert_eq!(result.q_without_m2, 0.0);
    }

    #[test]
    fn q_weights_depend_on_the_evolution_measures_and_m6() {
        let m = [1.2, 0.3, 2.0, 0.7, 4.0, 1.5, 0.2, 0.9, 1.1, 0.4, 2.2];
        let summary = |with_evolution, include_m6| {
            let result = summarize_all(m, with_evolution, include_m6);
            (result.q, result.q_without_m2, result.failed)
        };
        assert_eq!(
            summary(false, false),
            (0.958888888888889, 1.090666666666667, 4)
        );
        assert_eq!(
            summary(false, true),
            (1.0130000000000001, 1.1388235294117648, 4)
        );
        assert_eq!(
            summary(true, false),
            (1.132222222222222, 1.2481012658227846, 6)
        );
    }

    #[test]
    fn without_evolution_m7_m1_and_m2_fill_the_missing_weights() {
        let result = summarize_all(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 99.0, 99.0, 99.0, 99.0],
            false,
            true,
        );
        assert_eq!(result.failed, 1);
        assert_eq!(result.q, 0.32);
        assert_eq!(result.q_without_m2, 32.0 / 85.0);
    }
}
