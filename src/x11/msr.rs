//! The moving seasonality ratio and the seasonal filter it selects: Build 62
//! `sfmsr.f`, `vsfa.f` and `fis.f`.

use super::seasonal::SeasonalFilterSet;
use crate::time::AxisSpan;
use crate::{numerics, x11};

/// The seasonal filters after MSR selection, and how the selection ended.
pub(super) struct Selection {
    /// The declared filters with only the MSR entries replaced.
    pub(super) filters: SeasonalFilterSet<x11::SeasonalFilter>,
    /// `None` when no entry asked for MSR selection.
    pub(super) outcome: Option<x11::MsrSelection>,
}

/// Replaces the MSR entries of `filters` with `replacement` and keeps the
/// others.
pub(super) fn resolve(
    filters: &SeasonalFilterSet<x11::SeasonalFilter>,
    replacement: x11::SeasonalFilter,
) -> SeasonalFilterSet<x11::SeasonalFilter> {
    filters.map(|filter| {
        if filter == x11::SeasonalFilter::Msr {
            replacement
        } else {
            filter
        }
    })
}

/// `sfmsr.f`: chooses the filter for the MSR entries of `filters` from the
/// moving seasonality ratio of `si[..observed_end]`, dropping one year from
/// the end while the ratio falls between the selection bands.
///
/// `si` and `dates` start at the first backcast, or at the first observation
/// when there is none. Forecasts, which follow `observed_end`, take no part.
pub(super) fn select_checked(
    si: &[f64],
    dates: AxisSpan,
    observed_end: usize,
    mode: x11::Mode,
    filters: &SeasonalFilterSet<x11::SeasonalFilter>,
) -> Result<Selection, crate::Error> {
    let frequency = dates.start().frequency();
    let period = usize::from(frequency.periods_per_year());
    debug_assert_eq!(filters.frequency(), frequency);
    let mut selection = Selection {
        filters: *filters,
        outcome: None,
    };
    if !filters.as_slice().contains(&x11::SeasonalFilter::Msr) {
        return Ok(selection);
    }
    // editor.f scans for code 6 only while the first entry's code is < 6.
    // S3x1 is code 7, so it suppresses selection for later MSR entries.
    if filters[0] == x11::SeasonalFilter::S3x1 {
        selection.filters = resolve(filters, x11::SeasonalFilter::S3x5);
        selection.outcome = Some(x11::MsrSelection::SuppressedByFirstPeriodS3x1);
        return Ok(selection);
    }
    let offset = usize::from(dates.start().subperiod()) - 1;
    let mut end = observed_end - (offset + observed_end) % period;
    let mut passes = 0;
    loop {
        if end < 5 * period {
            selection.filters = resolve(filters, x11::SeasonalFilter::S3x5);
            selection.outcome = Some(x11::MsrSelection::ShortHistoryFallback { passes });
            return Ok(selection);
        }
        let (ratio, _) = global_ratio(&si[..end], period, mode, dates)?;
        passes += 1;
        if let Some(filter) = filter_for_ratio(ratio) {
            let replacement = match filter {
                x11::MsrFilter::S3x3 => x11::SeasonalFilter::S3x3,
                x11::MsrFilter::S3x5 => x11::SeasonalFilter::S3x5,
                x11::MsrFilter::S3x9 => x11::SeasonalFilter::S3x9,
            };
            selection.filters = resolve(filters, replacement);
            selection.outcome = Some(x11::MsrSelection::Selected { filter, passes });
            return Ok(selection);
        }
        end -= period;
    }
}

/// `sfmsr.f`'s selection bands. A ratio between them selects nothing.
fn filter_for_ratio(ratio: f64) -> Option<x11::MsrFilter> {
    if ratio <= 2.5 {
        Some(x11::MsrFilter::S3x3)
    } else if ratio >= 6.5 {
        Some(x11::MsrFilter::S3x9)
    } else if (3.5..=5.5).contains(&ratio) {
        Some(x11::MsrFilter::S3x5)
    } else {
        None
    }
}

/// The moving seasonality ratio of the whole of `si`, for the quality
/// statistics.
pub(super) fn quality_ratio(
    si: &[f64],
    period: usize,
    mode: x11::Mode,
    dates: AxisSpan,
) -> Result<f64, crate::diagnostics::X11QualityUnavailableReason> {
    use crate::diagnostics::{X11QualityStage as Stage, X11QualityUnavailableReason as Reason};
    let stage = Stage::FinalMovingSeasonalityRatio;
    if si.len() < 3 * period {
        return Err(Reason::InsufficientHistory {
            stage,
            actual: si.len(),
            minimum: 3 * period,
        });
    }
    let (ratio, finite) =
        global_ratio(si, period, mode, dates).map_err(|_| Reason::NonFiniteArithmetic { stage })?;
    if !finite || !ratio.is_finite() {
        return Err(Reason::NonFiniteArithmetic { stage });
    }
    Ok(ratio)
}

/// `vsfa.f`: the global moving seasonality ratio of `si`, and whether every
/// irregular and running sum stayed finite. Each calendar period's changes
/// are corrected and then summed, starting with the period of the first
/// value.
fn global_ratio(
    si: &[f64],
    period: usize,
    mode: x11::Mode,
    dates: AxisSpan,
) -> Result<(f64, bool), crate::Error> {
    let mut irregular_sum = 0.0;
    let mut seasonal_sum = 0.0;
    let mut finite = true;
    for first in 0..period {
        // The calendar subseries, padded at each end with three copies of
        // its end means for the 7-term average.
        let mut padded = vec![0.0; 3];
        padded.extend(si.iter().skip(first).step_by(period).copied());
        let end = padded.len();
        let first_mean = (padded[3] + padded[4] + padded[5]) / 3.0;
        // vsfa.f sums the last three values from the end inward.
        let last_mean = (padded[end - 1] + padded[end - 2] + padded[end - 3]) / 3.0;
        padded[..3].fill(first_mean);
        padded.extend([last_mean; 3]);
        let mut smoothed = vec![0.0; padded.len()];
        numerics::moving_average(&padded, &mut smoothed, 0..padded.len(), 1, 7);
        let (subseries, smooth) = (&padded[3..end], &smoothed[3..end]);
        let irregular: Vec<_> = subseries
            .iter()
            .zip(smooth)
            .map(|(&value, &average)| match mode {
                x11::Mode::Additive | x11::Mode::LogAdditive => value - average,
                x11::Mode::Multiplicative => value / average,
                x11::Mode::PseudoAdditive => value - average + 1.0,
            })
            .collect();
        finite &= irregular.iter().all(|value| value.is_finite());
        let (irregular_correction, seasonal_correction) = corrections(subseries.len() - 1);
        let steps = irregular.windows(2).zip(smooth.windows(2));
        let mut irregular_change = 0.0;
        let mut seasonal_change = 0.0;
        match mode {
            x11::Mode::Multiplicative | x11::Mode::PseudoAdditive => {
                for (year, (irregular, smooth)) in steps.enumerate() {
                    let irregular_step = (irregular[1] - irregular[0]).abs();
                    let seasonal_step = (smooth[1] - smooth[0]).abs();
                    if mode == x11::Mode::PseudoAdditive {
                        let index = first + year * period;
                        let stage = crate::error::X11PseudoAdditiveStage::SeasonalSelection;
                        super::pseudo_additive::divide(
                            irregular_step,
                            irregular[0],
                            dates,
                            index,
                            stage,
                        )?;
                        super::pseudo_additive::divide(
                            seasonal_step,
                            smooth[0],
                            dates,
                            index,
                            stage,
                        )?;
                    }
                    irregular_change += irregular_step / irregular[0];
                    seasonal_change += seasonal_step / smooth[0];
                }
                irregular_change = irregular_change * 100.0 * irregular_correction;
                seasonal_change = seasonal_change * 100.0 * seasonal_correction;
            }
            x11::Mode::Additive | x11::Mode::LogAdditive => {
                for (irregular, smooth) in steps {
                    // vsfa.f corrects each additive change before adding it.
                    irregular_change += (irregular[1] - irregular[0]).abs() * irregular_correction;
                    seasonal_change += (smooth[1] - smooth[0]).abs() * seasonal_correction;
                }
            }
        }
        // The ratio is of these sums, not of the per-period averages vsfa.f
        // goes on to report.
        irregular_sum += irregular_change;
        seasonal_sum += seasonal_change;
        finite &= irregular_sum.is_finite() && seasonal_sum.is_finite();
        if mode == x11::Mode::PseudoAdditive {
            let stage = crate::error::X11PseudoAdditiveStage::SeasonalSelection;
            super::pseudo_additive::checked(irregular_sum, dates, first, stage)?;
            super::pseudo_additive::checked(seasonal_sum, dates, first, stage)?;
        }
    }
    let ratio =
        if irregular_sum <= 999.0 * seasonal_sum && !numerics::build_62_equal(seasonal_sum, 0.0) {
            irregular_sum / seasonal_sum
        } else {
            999.99
        };
    Ok((ratio, finite))
}

/// `fis.f`: the irregular and seasonal finite-history corrections for a
/// calendar subseries with `changes` year-to-year changes.
fn corrections(changes: usize) -> (f64, f64) {
    const IRREGULAR: [f64; 4] = [1.00000, 1.02584, 1.01779, 1.01383];
    const SEASONAL: [f64; 4] = [1.00000, 3.00000, 1.55291, 1.30095];
    if changes < 6 {
        (IRREGULAR[changes - 2], SEASONAL[changes - 2])
    } else {
        let n = changes as f64;
        let after_six = (changes - 6) as f64;
        let seasonal = n * 1.732051 / (8.485281 + after_six * 1.732051);
        let irregular = n * 12.247449 / (73.239334 + after_six * 12.247449);
        (irregular, seasonal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeAxis;

    /// Selection with every calendar period requesting MSR.
    fn select(si: &[f64], axis: TimeAxis, observed_end: usize, mode: x11::Mode) -> Selection {
        let filters = SeasonalFilterSet::uniform(axis.frequency(), x11::SeasonalFilter::Msr);
        select_checked(
            si,
            axis.span(si.len()).unwrap(),
            observed_end,
            mode,
            &filters,
        )
        .unwrap()
    }

    /// The ratio of a series that starts in the first month or quarter.
    fn ratio(si: &[f64], period: usize, mode: x11::Mode) -> f64 {
        let axis = if period == 4 {
            TimeAxis::quarterly(2000, 1)
        } else {
            TimeAxis::monthly(2000, 1)
        }
        .unwrap();
        let (ratio, _) = global_ratio(si, period, mode, axis.span(si.len()).unwrap()).unwrap();
        ratio
    }

    /// Eighteen years of quarterly SI with a moving seasonal pattern.
    fn quarterly_si() -> Vec<f64> {
        (0..72)
            .map(|i| 1.0 + (i % 4) as f64 * 0.05 + ((i * 7) % 11) as f64 * 0.003)
            .collect()
    }

    #[test]
    fn pseudo_additive_rejects_zero_divisors_in_the_observed_span() {
        let axis = TimeAxis::quarterly(1999, 3).unwrap();
        let dates = axis.span(48).unwrap();
        let (ratio, _) = global_ratio(&[-2.0; 40], 4, x11::Mode::PseudoAdditive, dates).unwrap();
        assert_eq!(ratio, 999.99);
        assert_eq!(
            global_ratio(&[0.0; 40], 4, x11::Mode::PseudoAdditive, dates).unwrap_err(),
            crate::Error::from(crate::error::NumericalDetail::X11PseudoAdditive {
                stage: crate::error::X11PseudoAdditiveStage::SeasonalSelection,
                index: 0,
                period: axis.start(),
                reason: crate::error::X11PseudoAdditiveUnavailableReason::ZeroDivisor
            })
        );
        // Zero forecasts are not divided by.
        let mut si = vec![-2.0; 40];
        si.extend([0.0; 8]);
        let filters = SeasonalFilterSet::uniform(axis.frequency(), x11::SeasonalFilter::Msr);
        assert!(select_checked(&si, dates, 40, x11::Mode::PseudoAdditive, &filters).is_ok());
    }

    #[test]
    fn ratio_by_mode() {
        let si = quarterly_si();
        assert_eq!(ratio(&si, 4, x11::Mode::Additive), 12.057581331154363);
        assert_eq!(ratio(&si, 4, x11::Mode::Multiplicative), 12.082208278003248);
        assert_eq!(ratio(&si, 4, x11::Mode::PseudoAdditive), 13.123707465701665);
        // vsfa.f treats log-additive SI as additive.
        assert_eq!(ratio(&si, 4, x11::Mode::LogAdditive), 12.057581331154363);
    }

    #[test]
    fn nonfinite_si_is_flagged_and_gives_the_sentinel_ratio() {
        let mut si = quarterly_si();
        si[9] = f64::NAN;
        let axis = TimeAxis::quarterly(2000, 2).unwrap();
        let dates = axis.span(si.len()).unwrap();
        for mode in [
            x11::Mode::Additive,
            x11::Mode::Multiplicative,
            x11::Mode::LogAdditive,
        ] {
            assert_eq!(global_ratio(&si, 4, mode, dates), Ok((999.99, false)));
        }
        assert_eq!(
            quality_ratio(&si, 4, x11::Mode::Additive, dates),
            Err(
                crate::diagnostics::X11QualityUnavailableReason::NonFiniteArithmetic {
                    stage: crate::diagnostics::X11QualityStage::FinalMovingSeasonalityRatio
                }
            )
        );
        // A pseudo-additive run stops at the first year of that quarter.
        assert_eq!(
            global_ratio(&si, 4, x11::Mode::PseudoAdditive, dates),
            Err(crate::Error::from(
                crate::error::NumericalDetail::X11PseudoAdditive {
                    stage: crate::error::X11PseudoAdditiveStage::SeasonalSelection,
                    index: 1,
                    period: axis.period_at(1).unwrap(),
                    reason: crate::error::X11PseudoAdditiveUnavailableReason::NonFiniteValue,
                }
            ))
        );
    }

    #[test]
    fn selection_bands_include_their_endpoints() {
        use x11::MsrFilter::{S3x3, S3x5, S3x9};
        // sfmsr.f's bands, with the adjacent representable values.
        for (ratio, expected) in [
            (0.0, Some(S3x3)),
            (2.5, Some(S3x3)),
            (2.5_f64.next_up(), None),
            (3.5_f64.next_down(), None),
            (3.5, Some(S3x5)),
            (5.5, Some(S3x5)),
            (5.5_f64.next_up(), None),
            (6.5_f64.next_down(), None),
            (6.5, Some(S3x9)),
            (999.99, Some(S3x9)),
        ] {
            assert_eq!(filter_for_ratio(ratio), expected, "ratio {ratio:?}");
        }
    }

    #[test]
    fn corrections_follow_the_tables_then_the_formula() {
        assert_eq!(corrections(2), (1.0, 1.0));
        assert_eq!(corrections(3), (1.02584, 3.0));
        assert_eq!(corrections(5), (1.01383, 1.30095));
        assert_eq!(corrections(6), (1.0033501123863304, 1.2247450614776338));
        assert_eq!(corrections(9), (1.0022309169833474, 1.1393878010155953));
    }

    #[test]
    fn zero_seasonal_change_gives_the_sentinel_ratio() {
        for mode in [x11::Mode::Additive, x11::Mode::Multiplicative] {
            assert_eq!(ratio(&[1.0; 24], 4, mode), 999.99);
        }
    }

    #[test]
    fn selection_ends_at_a_calendar_year_end_and_ignores_forecasts() {
        // Constant SI gives the sentinel ratio, which selects S3x9, once
        // five calendar years end within the observations. The 24 values
        // after them are forecasts.
        for (axis, last_partial_end, december_end) in [
            (TimeAxis::monthly(1980, 7).unwrap(), 65, 66),
            (TimeAxis::quarterly(1980, 3).unwrap(), 21, 22),
        ] {
            let si = vec![1.0; december_end + 24];
            assert_eq!(
                select(&si, axis, last_partial_end, x11::Mode::Additive).outcome,
                Some(x11::MsrSelection::ShortHistoryFallback { passes: 0 }),
            );
            assert_eq!(
                select(&si, axis, december_end, x11::Mode::Additive).outcome,
                Some(x11::MsrSelection::Selected {
                    filter: x11::MsrFilter::S3x9,
                    passes: 1
                }),
            );
        }
    }

    #[test]
    fn quality_ratio_uses_the_whole_span() {
        let axis = TimeAxis::monthly(2000, 7).unwrap();
        let si = (0..180)
            .map(|i| (i % 12) as f64 + (i / 12) as f64 * 0.13 + ((i * 17) % 31) as f64 * 0.07)
            .collect::<Vec<_>>();
        let dates = axis.span(si.len()).unwrap();
        let full = quality_ratio(&si, 12, x11::Mode::Additive, dates).unwrap();
        // Selection would stop at the last complete calendar year.
        let (truncated, _) = global_ratio(&si[..174], 12, x11::Mode::Additive, dates).unwrap();
        assert!(full.is_finite() && full != 999.99);
        assert_ne!(full, truncated);
        assert_eq!(
            quality_ratio(&[0.0; 24], 12, x11::Mode::Additive, dates),
            Err(
                crate::diagnostics::X11QualityUnavailableReason::InsufficientHistory {
                    stage: crate::diagnostics::X11QualityStage::FinalMovingSeasonalityRatio,
                    actual: 24,
                    minimum: 36
                }
            )
        );
    }
}
