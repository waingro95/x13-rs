//! The X-11 Easter adjustment, estimated from the irregular by Easter date
//! group: Build 62 `holday.f`, `holidy.f` and `easter.f`.

use crate::easter_dates::{EASTER_DATE_OFFSETS, FIRST_EASTER_YEAR};
use crate::error::NumericalDetail;
use crate::time::TimeAxis;
use crate::x11::{X11EasterStage as Stage, X11EasterUnavailableReason as Reason};
use crate::{Error, x11, x11::EasterDateGroup as Group};

/// `easter.f`'s long-run frequency of each Easter date, March 22 first.
const FREQUENCIES: [f64; 35] = [
    0.0100, 0.0150, 0.0050, 0.0175, 0.0300, 0.0325, 0.0250, 0.0300, 0.0300, 0.0400, 0.0375, 0.0350,
    0.0250, 0.0275, 0.0425, 0.0425, 0.0275, 0.0300, 0.0225, 0.0400, 0.0425, 0.0325, 0.0300, 0.0350,
    0.0300, 0.0425, 0.0375, 0.0350, 0.0300, 0.0250, 0.0350, 0.0300, 0.0100, 0.0100, 0.0100,
];

/// Easter factors for the requested span, and whether they were estimated.
pub(super) struct EstimatedEasterPrior {
    /// One factor for each backcast, observation and forecast.
    pub(super) factors: Vec<f64>,
    pub(super) status: x11::EasterStatus,
}

fn failure(stage: Stage, period: Option<crate::Period>, reason: Reason) -> Error {
    Error::from(NumericalDetail::X11Easter {
        stage,
        period,
        reason,
    })
}

fn finite(value: f64, stage: Stage) -> Result<f64, Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(failure(stage, None, Reason::NonFiniteArithmetic))
    }
}

/// The date groups in `means` order: early, late, low (middle) and high (middle).
const GROUPS: [Group; 4] = [
    Group::OnOrBeforeApril1,
    Group::OnOrAfterApril16,
    Group::April2Through8,
    Group::April9Through15,
];

/// The group of an Easter date given as its offset from March 22.
fn group(date: f64) -> Group {
    if date <= 10.0 {
        Group::OnOrBeforeApril1
    } else if date > 24.0 {
        Group::OnOrAfterApril16
    } else if date <= 17.0 {
        Group::April2Through8
    } else {
        Group::April9Through15
    }
}

/// The group's position in [`GROUPS`] and in each per-group array.
fn slot(group: Group) -> usize {
    match group {
        Group::OnOrBeforeApril1 => 0,
        Group::OnOrAfterApril16 => 1,
        Group::April2Through8 => 2,
        Group::April9Through15 => 3,
    }
}

/// `holday.f`: estimates the Easter factors of the observed `irregular`, which
/// starts at `axis`, and extends them over the requested backcasts and
/// forecasts. The factors are all one when a date group has no observed
/// Easter.
pub(super) fn estimate_requested(
    irregular: &[f64],
    axis: TimeAxis,
    forecasts: usize,
    backcasts: usize,
) -> Result<EstimatedEasterPrior, Error> {
    let calendar = EasterCalendar::new(axis, irregular.len(), forecasts, backcasts)?;
    let indices = calendar.observed_marches()?;
    let missing: Vec<_> = GROUPS
        .into_iter()
        .zip(&indices)
        .filter_map(|(group, marches)| marches.is_empty().then_some(group))
        .collect();
    if !missing.is_empty() {
        return Ok(EstimatedEasterPrior {
            factors: vec![1.0; backcasts + irregular.len() + forecasts],
            status: x11::EasterStatus::MissingDateGroups(missing),
        });
    }
    Ok(EstimatedEasterPrior {
        factors: calendar.estimate(irregular, &indices, axis.start().subperiod() == 4)?,
        status: x11::EasterStatus::Estimated,
    })
}

/// holday's Easter calendar for one request: each year's Easter date in the
/// months holidy writes it to, and where the observed irregular and the
/// requested factors sit among those month indices. Indices count months from
/// January of the requested (backcast) start year.
struct EasterCalendar {
    /// The Easter date (offset from March 22) of each written month.
    indicators: Vec<Option<f64>>,
    /// Names indicator months in errors.
    padded_axis: TimeAxis,
    /// The requested start's index (Pos1bk - 1).
    requested_offset: usize,
    /// The observed irregular's first index and one past its last.
    offset: usize,
    end: usize,
    /// One past the last requested factor's index.
    extent: usize,
    /// The first observed March.
    march: usize,
    /// The first March whose factors are adjusted (holidy's marbk).
    factor_march: usize,
}

impl EasterCalendar {
    fn new(
        axis: TimeAxis,
        observed: usize,
        forecasts: usize,
        backcasts: usize,
    ) -> Result<Self, Error> {
        let requested_axis = axis.shift(
            -i64::try_from(backcasts)
                .map_err(|_| failure(Stage::CalendarLookup, None, Reason::OutsideDateTable))?,
        )?;
        let requested_offset = usize::from(requested_axis.start().subperiod()) - 1;
        let offset = requested_offset + backcasts;
        let end = offset + observed;
        let forward = if forecasts > 0 { forecasts } else { 12 };
        // Without forecasts holday.f still generates one more year of
        // factors. That year reads no observed data and feeds neither the
        // trimming nor the normalization, so it is left out here: reading its
        // unwritten dates would reject an otherwise valid adjustment.
        let extent = end + forecasts;
        let padded_axis = TimeAxis::monthly(requested_axis.start().year(), 1)?;
        // holday counts from Pos1bk, excluding leading calendar padding.
        let years = (backcasts + observed + forward).div_ceil(12);
        let mut indicators = vec![None; (years * 12 + 1).max(extent + 1)];
        for y in 0..years {
            let year = i64::from(requested_axis.start().year()) + y as i64;
            let index = usize::try_from(year - i64::from(FIRST_EASTER_YEAR))
                .ok()
                .filter(|&i| i < EASTER_DATE_OFFSETS.len());
            let date = index
                .map(|i| f64::from(EASTER_DATE_OFFSETS[i]))
                .ok_or_else(|| {
                    failure(
                        Stage::CalendarLookup,
                        padded_axis.period_at(y * 12).ok(),
                        Reason::OutsideDateTable,
                    )
                })?;
            // holidy.f writes the year's date into four months, starting one
            // after January. Only the Easter date is read here.
            for indicator in &mut indicators[y * 12 + 1..y * 12 + 5] {
                *indicator = Some(date);
            }
        }
        let march = (offset / 12) * 12 + if offset % 12 >= 3 { 14 } else { 2 };
        // holidy.f: with backcasts, marbk is March of the requested start
        // year, whatever month the request starts in.
        let factor_march = if backcasts == 0 { march } else { 2 };
        Ok(Self {
            indicators,
            padded_axis,
            requested_offset,
            offset,
            end,
            extent,
            march,
            factor_march,
        })
    }

    /// The Easter date written at index `i`; reading an unwritten month is
    /// undefined in Build 62.
    fn read(&self, i: usize) -> Result<f64, Error> {
        self.indicators.get(i).copied().flatten().ok_or_else(|| {
            failure(
                Stage::IndicatorRead,
                self.padded_axis.period_at(i).ok(),
                Reason::UninitializedOperand,
            )
        })
    }

    /// Each date group's observed March indices, in [`GROUPS`] order.
    fn observed_marches(&self) -> Result<[Vec<usize>; 4], Error> {
        let mut indices: [Vec<usize>; 4] = Default::default();
        for i in (self.march..self.end).step_by(12) {
            indices[slot(group(self.read(i)?))].push(i);
        }
        Ok(indices)
    }

    /// easter's per-year estimate: the group means (trimmed), the interpolated
    /// and normalized March/April factors, as ratios over the requested span.
    /// Every date group has at least one observed March in `indices`.
    fn estimate(
        &self,
        irregular: &[f64],
        indices: &[Vec<usize>; 4],
        starts_in_april: bool,
    ) -> Result<Vec<f64>, Error> {
        let Self {
            requested_offset,
            offset,
            end,
            extent,
            march,
            factor_march,
            ..
        } = *self;
        // The irregular in percent, with April mirrored about 200 so that each
        // March index i and i + 1 read the same Easter effect.
        let mut percent = vec![0.0; extent + 1];
        // x11int initializes backing to one; holday fills only the requested span.
        let mut factors = vec![1.0; extent + 1];
        factors[requested_offset..extent].fill(100.0);
        for (scratch, &value) in percent[offset..end].iter_mut().zip(irregular) {
            *scratch = finite(value * 100.0, Stage::Trimming)?;
        }
        for i in (march..end.saturating_sub(1)).step_by(12) {
            percent[i + 1] = 200.0 - percent[i + 1];
        }
        let mut means = [0.0; 4];
        let mut counts = [0.0; 4];
        for slot in 0..4 {
            for &i in &indices[slot] {
                means[slot] += percent[i];
                counts[slot] += 1.0;
                if i + 1 < end {
                    means[slot] += percent[i + 1];
                    counts[slot] += 1.0;
                }
            }
            means[slot] = finite(means[slot] / counts[slot], Stage::Trimming)?;
            // The early and late groups trim about their means now; the middle
            // groups trim about the interpolated factors below.
            if slot < 2 {
                let mut variance = 0.0;
                for &i in &indices[slot] {
                    variance += (percent[i] - means[slot]).powi(2);
                    if i + 1 < end {
                        variance += (percent[i + 1] - means[slot]).powi(2);
                    }
                }
                let sd = finite(variance.sqrt() / counts[slot].sqrt() * 2.0, Stage::Trimming)?;
                let (sum, count) = trim(&percent, &indices[slot], end, |_| means[slot], sd);
                means[slot] = retained_mean(sum, count, means[slot], false)?;
            }
        }
        // Only the observed years feed the middle groups' trimming, but the
        // requested extensions get factors too, and an unwritten date among
        // them is an error.
        for i in (march..extent).step_by(12) {
            let date = self.read(i)?;
            factors[i] = finite(interpolate(date, means), Stage::Trimming)?;
            if date < 11.0 || date > 24.0 {
                factors[i + 1] = 200.0 - factors[i];
            }
        }
        let mut retained_low = 0;
        for slot in 2..4 {
            let mut variance = 0.0;
            for &i in &indices[slot] {
                variance += (percent[i] - factors[i]).powi(2);
                if i + 1 < end {
                    variance += (percent[i + 1] - factors[i]).powi(2);
                }
            }
            let sd = finite(variance.sqrt() / counts[slot].sqrt() * 2.0, Stage::Trimming)?;
            let (sum, count) = trim(&percent, &indices[slot], end, |i| factors[i], sd);
            means[slot] = retained_mean(sum, count, means[slot], true)?;
            if slot == 2 {
                retained_low = count;
            }
        }
        for i in (factor_march..extent).step_by(12) {
            let date = self.read(i)?;
            if date > 10.0 && date < 25.0 {
                factors[i] = finite(interpolate(date, means), Stage::Trimming)?;
                factors[i + 1] = 200.0 - factors[i];
            }
        }
        let mut seasonal = 0.0;
        for (i, &frequency) in FREQUENCIES.iter().enumerate() {
            // easter.f's long-run mean changes branch at date 21, one date
            // before the yearly fit does, so date 21 takes the later
            // expression here.
            let value = if i == 21 {
                means[1] + (25.0 - i as f64) * (means[3] - means[1]) / 4.0
            } else {
                interpolate(i as f64, means)
            };
            seasonal += frequency * value;
        }
        finite(seasonal, Stage::Normalization)?;
        if seasonal == 0.0 || 200.0 - seasonal == 0.0 {
            return Err(failure(Stage::Normalization, None, Reason::ZeroDivisor));
        }
        for i in (factor_march..extent).step_by(12) {
            factors[i] = normalize(factors[i], seasonal)?;
            factors[i + 1] = normalize(factors[i + 1], 200.0 - seasonal)?;
        }
        if starts_in_april {
            let date = self.read(2)?;
            factors[3] = normalize(april_initial(date, means, retained_low), 200.0 - seasonal)?;
        }
        for factor in &mut factors[..extent] {
            *factor = finite(*factor / 100.0, Stage::Normalization)?;
        }
        Ok(factors[requested_offset..extent].to_vec())
    }
}

/// `value` as a percentage of the nonzero `divisor`.
fn normalize(value: f64, divisor: f64) -> Result<f64, Error> {
    finite((value * 100.0) / divisor, Stage::Normalization)
}

/// Divides `values`, which start at `axis`, by their Easter `factors`.
pub(super) fn apply_prior(
    values: &[f64],
    factors: &[f64],
    axis: TimeAxis,
) -> Result<Vec<f64>, Error> {
    values
        .iter()
        .zip(factors)
        .enumerate()
        .map(|(i, (&value, &factor))| {
            let period = Some(axis.period_at(i)?);
            if factor == 0.0 {
                return Err(failure(Stage::Normalization, period, Reason::ZeroDivisor));
            }
            let value = value / factor;
            if !value.is_finite() {
                return Err(failure(
                    Stage::Normalization,
                    period,
                    Reason::NonFiniteArithmetic,
                ));
            }
            Ok(value)
        })
        .collect()
}

/// `easter.f`'s March effect for an Easter `date`: the early or late group
/// mean, or a linear interpolation between the group means in between.
fn interpolate(date: f64, means: [f64; 4]) -> f64 {
    let [early, late, low, high] = means;
    if date < 11.0 {
        early
    } else if date > 24.0 {
        late
    } else if date <= 14.0 {
        low + (14.0 - date) * (early - low) / 4.0
    } else if date <= 21.0 {
        high + (21.0 - date) * (low - high) / 7.0
    } else {
        late + (25.0 - date) * (high - late) / 4.0
    }
}

/// The April effect of a series that starts in April, before normalization.
fn april_initial(date: f64, means: [f64; 4], retained_low: usize) -> f64 {
    // For a late Easter easter.f subtracts `nsuma`, the count of values the
    // low middle group retained, where a mean would be expected.
    if date > 24.0 {
        200.0 - retained_low as f64
    } else {
        200.0 - interpolate(date, means)
    }
}

/// The sum and count of a group's percent values within `sd` of their mean.
fn trim(
    percent: &[f64],
    indices: &[usize],
    end: usize,
    mean: impl Fn(usize) -> f64,
    sd: f64,
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for &i in indices {
        let keep_march = (percent[i] - mean(i)).abs() < sd;
        let keep_april = i + 1 < end && (percent[i + 1] - mean(i)).abs() < sd;
        let march = if keep_march { percent[i] } else { 0.0 };
        let april = if keep_april { percent[i + 1] } else { 0.0 };
        // easter.f adds the pair before the running sum.
        sum += march + april;
        count += usize::from(keep_march) + usize::from(keep_april);
    }
    (sum, count)
}

/// A group's mean after trimming. A middle group that retained nothing keeps
/// its `original` mean; an early or late group is an error.
fn retained_mean(sum: f64, count: usize, original: f64, middle: bool) -> Result<f64, Error> {
    if count == 0 {
        if middle {
            Ok(original)
        } else {
            Err(failure(Stage::Trimming, None, Reason::EmptyRetainedGroup))
        }
    } else {
        finite(sum / count as f64, Stage::Trimming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NumericalFailure;

    fn estimate(irregular: &[f64], axis: TimeAxis) -> Result<EstimatedEasterPrior, Error> {
        estimate_requested(irregular, axis, 0, 0)
    }

    /// Twenty-one years in which every date group is observed and retained.
    fn irregular() -> Vec<f64> {
        (0..252)
            .map(|i| 1.0 + ((i * 7) % 13) as f64 * 0.01 - ((i * 5) % 11) as f64 * 0.004)
            .collect()
    }

    #[test]
    fn missing_groups_are_ordered_and_neutral() {
        let result = estimate(&vec![1.0; 36], TimeAxis::monthly(1983, 1).unwrap()).unwrap();
        assert_eq!(
            result.status,
            x11::EasterStatus::MissingDateGroups(vec![
                Group::OnOrBeforeApril1,
                Group::April9Through15
            ])
        );
        assert!(result.factors.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn date_table_lookup_precedes_the_missing_group_check() {
        assert!(matches!(
            estimate(&vec![1.0; 36], TimeAxis::monthly(2099, 1).unwrap()),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    stage: Stage::CalendarLookup,
                    reason: Reason::OutsideDateTable,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn constant_early_group_retains_nothing() {
        assert!(matches!(
            estimate(&vec![1.0; 252], TimeAxis::monthly(1980, 1).unwrap()),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    stage: Stage::Trimming,
                    reason: Reason::EmptyRetainedGroup,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn only_an_empty_middle_group_keeps_its_mean() {
        assert_eq!(retained_mean(0.0, 0, 103.0, true).unwrap(), 103.0);
        assert!(matches!(
            retained_mean(0.0, 0, 103.0, false),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    reason: Reason::EmptyRetainedGroup,
                    ..
                }
            )))
        ));
        assert!(matches!(
            retained_mean(f64::INFINITY, 1, 103.0, false),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    reason: Reason::NonFiniteArithmetic,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn overflow_and_zero_factors_are_errors() {
        assert!(matches!(
            normalize(f64::MAX, 100.0),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    stage: Stage::Normalization,
                    reason: Reason::NonFiniteArithmetic,
                    ..
                }
            )))
        ));
        let axis = TimeAxis::monthly(1980, 3).unwrap();
        assert!(matches!(
            apply_prior(&[1.0], &[0.0], axis),
            Err(Error::Numerical(NumericalFailure(NumericalDetail::X11Easter {
                period: Some(period),
                reason: Reason::ZeroDivisor,
                ..
            }))) if period == axis.start()
        ));
        assert!(matches!(
            apply_prior(&[f64::MAX], &[0.1], axis),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    reason: Reason::NonFiniteArithmetic,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn late_april_start_uses_the_retained_low_count() {
        assert_eq!(april_initial(28.0, [101.0, 98.0, 102.0, 99.0], 11), 189.0);
        assert_eq!(april_initial(9.0, [101.0, 98.0, 102.0, 99.0], 11), 99.0);
        assert_eq!(april_initial(12.0, [101.0, 98.0, 102.0, 99.0], 11), 98.5);
        assert_eq!(april_initial(21.0, [101.0, 98.0, 102.0, 99.0], 11), 101.0);
    }

    #[test]
    fn trim_threshold_is_strict_and_pairs_are_summed_first() {
        let (sum, n) = trim(&[102.0, 98.0, 101.0, 99.0], &[0, 2], 4, |_| 100.0, 2.0);
        assert_eq!((sum, n), (200.0, 2));
        // Summing each March and April pair first is not a flat sum.
        let values = [1e16, 0.0, -1e16, 1.0];
        assert_eq!(trim(&values, &[0, 2], 4, |_| 0.0, 2e16), (0.0, 4));
        assert_eq!(values.iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn interpolation_joins_the_group_means() {
        assert_eq!(interpolate(11.0, [100.0, 90.0, 104.0, 96.0]), 101.0);
        assert_eq!(interpolate(24.0, [100.0, 90.0, 104.0, 96.0]), 91.5);
    }

    #[test]
    fn only_march_and_april_are_adjusted() {
        let prior = estimate(&irregular(), TimeAxis::monthly(1980, 1).unwrap()).unwrap();
        assert_eq!(prior.status, x11::EasterStatus::Estimated);
        assert_eq!(prior.factors.len(), 252);
        for (i, &factor) in prior.factors.iter().enumerate() {
            if !matches!(i % 12, 2 | 3) {
                assert_eq!(factor, 1.0, "month {i}");
            }
        }
        // Easter 1980 fell on April 6.
        assert_eq!(
            prior.factors[2..4],
            [1.0117550908758763, 0.9883441936245618]
        );
    }

    #[test]
    fn extensions_get_factors_and_leave_the_observed_ones() {
        let axis = TimeAxis::monthly(1980, 1).unwrap();
        let observed = estimate(&irregular(), axis).unwrap().factors;
        let forecast = estimate_requested(&irregular(), axis, 12, 0).unwrap();
        assert_eq!(forecast.factors[..252], observed);
        let backcast = estimate_requested(&irregular(), axis, 0, 12).unwrap();
        assert_eq!(backcast.factors[12..], observed);
        // Easter fell on April 15 in both 1979 and 2001.
        let mut extension = [1.0; 12];
        extension[2] = 0.9994260181512641;
        extension[3] = 1.0005691339490717;
        assert_eq!(forecast.factors[252..], extension);
        assert_eq!(backcast.factors[..12], extension);
    }

    #[test]
    fn april_start_adjusts_its_first_month() {
        let prior = estimate(&irregular(), TimeAxis::monthly(1980, 4).unwrap()).unwrap();
        assert_eq!(prior.status, x11::EasterStatus::Estimated);
        assert_eq!(prior.factors[0], 0.9932385311625364);
        assert_eq!(prior.factors[1..11], [1.0; 10]);
    }

    #[test]
    fn mid_year_start_ignores_the_unwritten_tail() {
        // The year holday.f adds without forecasts has no date written for
        // its March.
        let axis = TimeAxis::monthly(1980, 7).unwrap();
        let prior = estimate(&irregular(), axis).unwrap();
        assert_eq!(prior.status, x11::EasterStatus::Estimated);
        assert_eq!(prior.factors.len(), 252);
        assert!(prior.factors.iter().all(|value| value.is_finite()));
        // An empty early group is still reported, and so is a missing one.
        assert!(matches!(
            estimate(&[1.0; 252], axis),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    stage: Stage::Trimming,
                    reason: Reason::EmptyRetainedGroup,
                    ..
                }
            )))
        ));
        let short = estimate(&[1.0; 36], TimeAxis::monthly(1983, 7).unwrap()).unwrap();
        assert!(matches!(
            short.status,
            x11::EasterStatus::MissingDateGroups(_)
        ));
    }

    #[test]
    fn neutral_factors_cover_backcasts_and_forecasts() {
        let axis = TimeAxis::monthly(1983, 1).unwrap();
        for (forecasts, backcasts, expected) in [(0, 0, 36), (1, 0, 37), (5, 0, 41), (5, 12, 53)] {
            let prior = estimate_requested(&[1.0; 36], axis, forecasts, backcasts).unwrap();
            assert_eq!(prior.factors, vec![1.0; expected]);
            assert!(matches!(
                prior.status,
                x11::EasterStatus::MissingDateGroups(_)
            ));
        }
    }

    #[test]
    fn forecast_march_without_a_written_date_is_undefined() {
        // From a July start, a year of backcasts and one of forecasts reach a
        // March that holday.f never writes a date for.
        let values: Vec<_> = (0..144)
            .map(|i| 1.0 + ((i * 7) % 13) as f64 * 0.01)
            .collect();
        let axis = TimeAxis::monthly(1980, 7).unwrap();
        assert!(matches!(
            estimate_requested(&values, axis, 12, 12),
            Err(Error::Numerical(NumericalFailure(NumericalDetail::X11Easter {
                stage: Stage::IndicatorRead,
                period: Some(period),
                reason: Reason::UninitializedOperand,
            }))) if period.year() == 1993 && period.subperiod() == 3
        ));
        // A missing date group returns before that March is read.
        let neutral =
            estimate_requested(&[1.0; 36], TimeAxis::monthly(1983, 7).unwrap(), 12, 12).unwrap();
        assert!(matches!(
            neutral.status,
            x11::EasterStatus::MissingDateGroups(_)
        ));
        assert_eq!(neutral.factors, vec![1.0; 60]);
        // So does an empty early group.
        assert!(matches!(
            estimate_requested(&[1.0; 144], axis, 12, 12),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Easter {
                    stage: Stage::Trimming,
                    reason: Reason::EmptyRetainedGroup,
                    ..
                }
            )))
        ));
    }
}
