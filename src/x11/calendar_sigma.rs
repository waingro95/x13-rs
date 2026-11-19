//! The automatic choice of extreme-value scale (Build 62 `vtest.f`): Cochran's
//! test of whether the irregular's variance differs between calendar months
//! or quarters.

use crate::{numerics::build_62_equal, x11, x11::AutomaticExtremeScale as Choice};
use std::ops::Range;

/// `vtest.f`'s critical values for a monthly series: entry `r` is for `r + 1`
/// degrees of freedom.
const MONTHLY: [f64; 40] = [
    0.5410, 0.3924, 0.3264, 0.2880, 0.2624, 0.2440, 0.2299, 0.2187, 0.2096, 0.2020, 0.1955, 0.1900,
    0.1851, 0.1808, 0.1770, 0.1735, 0.1704, 0.1676, 0.1650, 0.1627, 0.1605, 0.1585, 0.1566, 0.1548,
    0.1532, 0.1517, 0.1502, 0.1488, 0.1476, 0.1463, 0.1452, 0.1441, 0.1431, 0.1421, 0.1411, 0.1402,
    0.1393, 0.1385, 0.1377, 0.1370,
];
/// The same for a quarterly series.
const QUARTERLY: [f64; 40] = [
    0.9065, 0.7679, 0.6839, 0.6287, 0.5894, 0.5598, 0.5365, 0.5175, 0.5018, 0.4884, 0.4769, 0.4668,
    0.4579, 0.4500, 0.4430, 0.4365, 0.4307, 0.4254, 0.4205, 0.4159, 0.4118, 0.4079, 0.4042, 0.4008,
    0.3976, 0.3946, 0.3918, 0.3891, 0.3866, 0.3842, 0.3819, 0.3797, 0.3776, 0.3756, 0.3737, 0.3719,
    0.3702, 0.3685, 0.3669, 0.3654,
];

/// Tests the irregular over `span`: a scale for each calendar period when the
/// largest period variance's share of their total reaches the critical value,
/// otherwise the rolling five-year scale.
pub(super) fn select(
    irregular: &[f64],
    span: Range<usize>,
    period: usize,
    mode: x11::Mode,
) -> Choice {
    let test = cochran(irregular, span, period, mode);
    decision(test.statistic, test.critical)
}

/// [`select`], except that a pseudo-additive run's nonfinite total or
/// statistic is an error at the span's first observation.
pub(super) fn select_checked(
    irregular: &[f64],
    span: Range<usize>,
    period: usize,
    mode: x11::Mode,
    dates: crate::time::AxisSpan,
) -> Result<Choice, crate::Error> {
    let first = span.start;
    let test = cochran(irregular, span, period, mode);
    if mode == x11::Mode::PseudoAdditive {
        let stage = crate::error::X11PseudoAdditiveStage::SeasonalSelection;
        super::pseudo_additive::checked(test.total, dates, first, stage)?;
        super::pseudo_additive::checked(test.statistic, dates, first, stage)?;
    }
    Ok(decision(test.statistic, test.critical))
}

/// Cochran's statistic, its critical value and the total it divides by.
struct Cochran {
    statistic: f64,
    critical: f64,
    total: f64,
}

/// One calendar slot's mean squared deviation and the number of values it
/// averages.
struct Slot {
    mean: f64,
    count: usize,
}

/// Each calendar slot's mean squared deviation from the mode's center. Like
/// `vtest.f`, the slots start at the span's first observation, not at January
/// or the first quarter.
fn slots(
    irregular: &[f64],
    span: Range<usize>,
    period: usize,
    mode: x11::Mode,
) -> impl Iterator<Item = Slot> {
    // The caller checks this geometry, which leaves every slot at least two
    // values.
    debug_assert!(matches!(period, 4 | 12) && span.end <= irregular.len());
    debug_assert!(span.len() >= 2 * period);
    // vtest.f centers log-additive irregulars at 1, unlike the working
    // extremes (`x11::Mode::center`), which center them at 0.
    let center = match mode {
        x11::Mode::Additive => 0.0,
        x11::Mode::Multiplicative | x11::Mode::LogAdditive | x11::Mode::PseudoAdditive => 1.0,
    };
    (span.start..span.start + period).map(move |first| {
        // gfortran fuses each square into the running sum in vtest.f; the
        // division is separate.
        let (sum, count) = irregular[first..span.end].iter().step_by(period).fold(
            (0.0, 0),
            |(sum, count), &value| {
                let deviation = value - center;
                (deviation.mul_add(deviation, sum), count + 1)
            },
        );
        Slot {
            mean: sum / count as f64,
            count,
        }
    })
}

fn cochran(irregular: &[f64], span: Range<usize>, period: usize, mode: x11::Mode) -> Cochran {
    // `f64::max` skips NaN means; a NaN mean also makes the total, and so
    // the statistic, NaN.
    let (total, maximum, fewest) = slots(irregular, span, period, mode).fold(
        (0.0, f64::NEG_INFINITY, usize::MAX),
        |(total, maximum, fewest), slot| {
            (
                total + slot.mean,
                maximum.max(slot.mean),
                fewest.min(slot.count),
            )
        },
    );
    // Row r holds the critical value for r + 1 degrees of freedom (the fewest
    // values less one); vtest.f's tables stop at 40.
    let row = (fewest - 2).min(39);
    let critical = if period == 4 {
        QUARTERLY[row]
    } else {
        MONTHLY[row]
    };
    Cochran {
        statistic: statistic(maximum, total),
        critical,
        total,
    }
}

/// The largest slot mean's share of the total, or the total itself when it is
/// zero within Build 62's tolerance.
fn statistic(maximum: f64, total: f64) -> f64 {
    if build_62_equal(total, 0.0) {
        total
    } else {
        maximum / total
    }
}

fn decision(statistic: f64, critical: f64) -> Choice {
    if statistic >= critical {
        Choice::ByCalendarPeriod
    } else {
        Choice::RollingFiveYear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot_means(irregular: &[f64], span: Range<usize>, period: usize) -> Vec<f64> {
        slots(irregular, span, period, x11::Mode::Additive)
            .map(|slot| slot.mean)
            .collect()
    }

    #[test]
    fn pseudo_additive_rejects_a_nonfinite_total() {
        let axis = crate::time::TimeAxis::quarterly(1999, 3).unwrap();
        let dates = axis.span(24).unwrap();
        let error =
            select_checked(&[1e200; 24], 0..24, 4, x11::Mode::PseudoAdditive, dates).unwrap_err();
        assert_eq!(
            error,
            crate::Error::from(crate::error::NumericalDetail::X11PseudoAdditive {
                stage: crate::error::X11PseudoAdditiveStage::SeasonalSelection,
                index: 0,
                period: axis.start(),
                reason: crate::error::X11PseudoAdditiveUnavailableReason::NonFiniteValue,
            })
        );
    }

    #[test]
    fn log_additive_centers_at_one() {
        let values = [1.0; 48];
        assert_eq!(
            cochran(&values, 0..48, 4, x11::Mode::LogAdditive).total,
            0.0
        );
        assert_eq!(cochran(&values, 0..48, 4, x11::Mode::Additive).total, 4.0);
    }

    #[test]
    fn deviations_are_measured_from_the_mode_center() {
        for (period, mode, center) in [
            (12, x11::Mode::Additive, 0.0),
            (4, x11::Mode::Multiplicative, 1.0),
        ] {
            let mut input = vec![center; period * 5];
            for index in (0..input.len()).step_by(period) {
                input[index] += 2.0;
            }
            assert_eq!(
                select(&input, 0..input.len(), period, mode),
                Choice::ByCalendarPeriod
            );
            input.fill(center + 2.0);
            assert_eq!(
                select(&input, 0..input.len(), period, mode),
                Choice::RollingFiveYear
            );
        }
    }

    #[test]
    fn total_within_zero_tolerance_selects_rolling() {
        for (power, expected) in [
            (-33, Choice::RollingFiveYear),
            (-32, Choice::ByCalendarPeriod),
        ] {
            let mut input = vec![0.0; 48];
            let value = f64::from_bits(((1023 + power) as u64) << 52);
            for index in (0..48).step_by(12) {
                input[index] = value;
            }
            assert_eq!(select(&input, 0..48, 12, x11::Mode::Additive), expected);
        }
    }

    #[test]
    fn partial_years_divide_by_each_slot_count() {
        let input = [2.0, 4.0, 8.0, 16.0, 2.0, 4.0, 8.0, 16.0, 2.0, 4.0, 8.0];
        assert_eq!(slot_means(&input, 0..11, 4), [4.0, 16.0, 64.0, 256.0]);
        // The last slot has two values, so one degree of freedom.
        let test = cochran(&input, 0..11, 4, x11::Mode::Additive);
        assert_eq!(test.critical, 0.9065);
        assert_eq!(test.statistic, 256.0 / 340.0);
    }

    #[test]
    fn total_accumulates_from_the_span_start() {
        let mut input = vec![1.0; 48];
        for index in (0..48).step_by(12) {
            input[index] = 100_000_000.0;
        }
        let first = cochran(&input, 0..48, 12, x11::Mode::Additive);
        let shifted = cochran(&input, 1..37, 12, x11::Mode::Additive);
        assert_eq!(first.total, 10_000_000_000_000_000.0);
        assert_eq!(shifted.total, 10_000_000_000_000_012.0);
    }

    #[test]
    fn critical_values_stop_at_forty_degrees_of_freedom() {
        for (period, first, before_last, last) in
            [(12, 0.5410, 0.1377, 0.1370), (4, 0.9065, 0.3669, 0.3654)]
        {
            for (count, critical) in [(2, first), (40, before_last), (41, last), (45, last)] {
                let input = vec![2.0; period * count];
                let test = cochran(&input, 0..input.len(), period, x11::Mode::Additive);
                assert_eq!(test.critical, critical);
            }
        }
    }

    #[test]
    fn zero_guard_is_strict_and_decision_includes_equality() {
        let guard: f64 = 3.834e-20;
        assert_eq!(
            statistic(guard, f64::from_bits(guard.to_bits() - 1)),
            f64::from_bits(guard.to_bits() - 1)
        );
        assert_eq!(statistic(guard, guard), 1.0);
        assert!(statistic(guard, f64::from_bits(guard.to_bits() + 1)) > 0.99);
        assert_eq!(statistic(0.0, 0.0), 0.0);
        for critical in [0.5410_f64, 0.3654] {
            assert_eq!(
                decision(f64::from_bits(critical.to_bits() - 1), critical),
                Choice::RollingFiveYear
            );
            assert_eq!(decision(critical, critical), Choice::ByCalendarPeriod);
            assert_eq!(
                decision(f64::from_bits(critical.to_bits() + 1), critical),
                Choice::ByCalendarPeriod
            );
        }
    }

    #[test]
    fn nan_or_infinite_slot_selects_rolling() {
        // Signed-zero deviations square to +0.0 means; the zero total guard
        // then returns the total itself.
        let zero = cochran(&[-0.0; 12], 0..12, 4, x11::Mode::Additive);
        assert_eq!([zero.total.to_bits(), zero.statistic.to_bits()], [0, 0]);
        // A NaN or infinite slot leaves a NaN statistic whatever the maximum,
        // which rejects the calendar choice.
        for (index, value, total) in [(1, f64::NAN, f64::NAN), (0, 1e200, f64::INFINITY)] {
            let mut input = vec![1.5; 12];
            input[index] = value;
            let test = cochran(&input, 0..12, 4, x11::Mode::Multiplicative);
            assert_eq!(test.total.to_bits(), total.to_bits());
            assert!(test.statistic.is_nan());
            assert_eq!(
                decision(test.statistic, test.critical),
                Choice::RollingFiveYear
            );
        }
        // The same sample with a finite outlier.
        let mut input = vec![1.5; 12];
        input[5] = -0.0;
        let test = cochran(&input, 0..12, 4, x11::Mode::Multiplicative);
        assert_eq!([test.total, test.statistic], [1.25, 0.4]);
    }

    #[test]
    fn squares_accumulate_fused_then_divide() {
        let mut input = vec![0.0; 12];
        input[0] = 0.05711422845691383;
        input[4] = 0.07722772277227723;
        input[8] = 0.4064885496183206;
        let means = slot_means(&input, 0..12, 4);
        assert_eq!(means[0].to_bits(), 0x3fadc63bf8b8b818);
    }
}
