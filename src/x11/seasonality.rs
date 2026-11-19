//! X-11 seasonality tests: the stable (`ftest`), moving (`mstest`) and
//! Kruskal-Wallis (`kwtest`) tests, the combined test for identifiable
//! seasonality (`combft`) and the residual seasonality tests.

use crate::diagnostics::{
    Sample, Unavailable, UnavailableDetail, X11DiagnosticUnavailableReason as Reason,
};
use crate::time::TimeAxis;
use crate::{Error, Frequency, Warning, diagnostics, diagnostics::SeasonalityTest as Stage, x11};
use std::{num::NonZeroUsize, ops::Range};

/// The dates and count of `span`, or `None` when it is empty.
fn sample(axis: TimeAxis, span: Range<usize>) -> Option<Sample> {
    (span.start < span.end)
        .then(|| {
            Some(Sample {
                first: axis.period_at(span.start).ok()?,
                last: axis.period_at(span.end - 1).ok()?,
                count: span.end - span.start,
            })
        })
        .flatten()
}

/// Why `test` has no value. `lag` is the differencing lag of a residual
/// seasonality test, kept only with a sample.
fn missing(test: Stage, reason: Reason, sample: Option<Sample>, lag: Option<usize>) -> Unavailable {
    Unavailable(UnavailableDetail::SeasonalityTest {
        test,
        reason,
        differencing_lag: sample.as_ref().and(lag),
        sample,
    })
}

fn unavailable<T>(test: Stage, reason: Reason) -> Result<T, Unavailable> {
    Err(missing(test, reason, None, None))
}

/// `sum + value²` as one fused multiply-add, which is how gfortran compiled
/// the sums of squares in `ftest` and `mstest`. It fused none of their other
/// sums.
fn square_add(sum: f64, value: f64) -> f64 {
    value.mul_add(value, sum)
}

/// gfortran's real-to-integer power, `z**k`: binary exponentiation by
/// repeated squaring, whose rounding differs from `powi` and `powf`.
fn gfortran_powi(mut base: f64, mut exponent: usize) -> f64 {
    let mut result = if exponent & 1 == 1 { base } else { 1.0 };
    exponent >>= 1;
    while exponent != 0 {
        base *= base;
        if exponent & 1 == 1 {
            result *= base;
        }
        exponent >>= 1;
    }
    result
}

/// `fvalue`: the F statistic `x` and its upper-tail probability with `m` and
/// `n` degrees of freedom. As in Build 62, a statistic that is not positive,
/// or whose cumulative probability is not, comes back as `(0, 1)`.
fn fvalue(x: f64, m: usize, n: usize) -> Result<(f64, f64), Reason> {
    if !x.is_finite() {
        return Err(Reason::NonFiniteArithmetic);
    }
    if m == 0 || n == 0 {
        return Err(Reason::InsufficientSample);
    }
    if x <= 0.0 {
        return Ok((0.0, 1.0));
    }
    if x > 90.0 || (x > 40.0 && n > 150) {
        return Ok((x, 0.0));
    }
    let l = if m.is_multiple_of(2) { 2 } else { 1 };
    let k = if n.is_multiple_of(2) { 2 } else { 1 };
    let w = x * m as f64 / n as f64;
    let mut z = 1.0 / (1.0 + w);
    let (mut p, mut d) = match (l, k) {
        (2, 2) => (w * z, z * z),
        (2, 1) => {
            let root = z.sqrt();
            (1.0 - root, 0.5 * z * root)
        }
        (1, 2) => {
            let root = (w * z).sqrt();
            (root, 0.5 * root * z / w)
        }
        _ => {
            let root = w.sqrt();
            #[expect(
                clippy::approx_constant,
                reason = "the literal in fvalue.f, not the nearest 1/pi"
            )]
            let y = 0.31830988618379;
            (2.0 * y * root.atan(), y * z / root)
        }
    };
    let mut y = 2.0 * w / z;
    if n >= k + 2 {
        if l != 1 {
            let zk = gfortran_powi(z, (n - 1) / 2);
            d = d * zk * n as f64 / k as f64;
            let term = w * z * (zk - 1.0) / (z - 1.0);
            p = zk.mul_add(p, term);
        } else {
            for j in ((k + 2)..=n).step_by(2) {
                d = (1.0 + l as f64 / (j - 2) as f64) * d * z;
                p += d * y / (j - 1) as f64;
            }
        }
    }
    y = w * z;
    if m >= l + 2 {
        z = 2.0 / z;
        for i in ((l + 2)..=m).step_by(2) {
            let zk = (i + n - 2) as f64;
            d = y * d * zk / (i - 2) as f64;
            p -= z * d / zk;
        }
    }
    if ![w, z, y, p, d].iter().all(|v| v.is_finite()) {
        return Err(Reason::NonFiniteArithmetic);
    }
    if p >= 1.0 {
        Ok((x, 0.0))
    } else if p > 0.0 {
        Ok((x, 1.0 - p))
    } else {
        Ok((0.0, 1.0))
    }
}

/// Whether the tested values are ratios about one, which the tests scale to
/// percent.
fn multiplicative(mode: x11::Mode) -> bool {
    matches!(mode, x11::Mode::Multiplicative | x11::Mode::PseudoAdditive)
}

/// `ftest`: the one-way analysis of variance of `values[span]` between
/// calendar periods. `lag` is the differencing lag of a residual seasonality
/// test.
fn f_test(
    values: &[f64],
    axis: TimeAxis,
    span: Range<usize>,
    mode: x11::Mode,
    lag: Option<usize>,
) -> Result<diagnostics::FTest, Reason> {
    let period = usize::from(axis.frequency().periods_per_year());
    let count = span.end.saturating_sub(span.start);
    if span.end > values.len() || count <= period {
        return Err(Reason::InsufficientSample);
    }
    let sample = sample(axis, span.clone()).ok_or(Reason::InsufficientSample)?;
    let mut total = 0.0;
    let mut total_ss = 0.0;
    let mut between_ss = 0.0;
    // `ftest` takes its columns from the span's first period, not from
    // January.
    for offset in 0..period {
        let mut slot_count = 0;
        let mut slot_sum = 0.0;
        for j in ((span.start + offset)..span.end).step_by(period) {
            let value = values[j];
            if !value.is_finite() {
                return Err(Reason::NonFiniteArithmetic);
            }
            slot_count += 1;
            slot_sum += value;
            total_ss = square_add(total_ss, value);
        }
        total += slot_sum;
        between_ss += slot_sum * slot_sum / slot_count as f64;
    }
    let scale = if multiplicative(mode) { 10000.0 } else { 1.0 };
    total_ss = (total_ss - total * total / count as f64) * scale;
    between_ss = (between_ss - total * total / count as f64) * scale;
    let residual_ss = total_ss - between_ss;
    let numerator = period - 1;
    let denominator = count - period;
    let between_ms = between_ss / numerator as f64;
    let residual_ms = residual_ss / denominator as f64;
    if ![total, total_ss, between_ss, between_ms, residual_ms]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(Reason::NonFiniteArithmetic);
    }
    if crate::numerics::build_62_equal(residual_ms, 0.0) {
        return Err(Reason::ZeroResidualMeanSquare);
    }
    let (statistic, probability) = fvalue(between_ms / residual_ms, numerator, denominator)?;
    Ok(diagnostics::FTest {
        statistic,
        upper_tail_probability_percent: probability * 100.0,
        numerator_degrees_of_freedom: numerator,
        denominator_degrees_of_freedom: denominator,
        sample,
        differencing_lag: lag,
        small_residual_mean_square: lag.is_none() && multiplicative(mode) && residual_ms <= 0.01,
    })
}

fn stable_conclusion(p: f64) -> diagnostics::StableSeasonality {
    if p <= 0.1 {
        diagnostics::StableSeasonality::PresentAtPointOnePercent
    } else {
        diagnostics::StableSeasonality::NotPresentAtPointOnePercent
    }
}

fn moving_conclusion(p: f64) -> diagnostics::MovingSeasonality {
    if p <= 1.0 {
        diagnostics::MovingSeasonality::PresentAtOnePercent
    } else if p >= 5.0 {
        diagnostics::MovingSeasonality::NotPresentAtFivePercent
    } else {
        diagnostics::MovingSeasonality::PresentAtFivePercent
    }
}

fn residual_conclusion(p: f64, recent: bool) -> diagnostics::ResidualSeasonality {
    if p <= 1.0 {
        diagnostics::ResidualSeasonality::PresentAtOnePercent
    } else if !recent {
        diagnostics::ResidualSeasonality::NotPresentAtOnePercent
    } else if p <= 5.0 {
        diagnostics::ResidualSeasonality::PresentAtFivePercent
    } else {
        diagnostics::ResidualSeasonality::NotPresentAtFivePercent
    }
}

fn kw_conclusion(p: f64) -> diagnostics::KruskalWallisSeasonality {
    if p <= 1.0 {
        diagnostics::KruskalWallisSeasonality::PresentAtOnePercent
    } else {
        diagnostics::KruskalWallisSeasonality::NotPresentAtOnePercent
    }
}

/// The stable seasonality F-test of `values[span]`.
pub(super) fn stable(
    values: &[f64],
    axis: TimeAxis,
    span: Range<usize>,
    mode: x11::Mode,
    stage: Stage,
) -> Result<diagnostics::StableSeasonalityTest, Unavailable> {
    match f_test(values, axis, span.clone(), mode, None) {
        Ok(test) => Ok(diagnostics::StableSeasonalityTest {
            conclusion: stable_conclusion(test.upper_tail_probability_percent),
            test,
        }),
        Err(reason) => Err(missing(stage, reason, sample(axis, span), None)),
    }
}

/// `mstest`: the moving seasonality F-test, a two-way analysis of variance
/// over the complete calendar years of `span`.
fn moving(
    values: &[f64],
    axis: TimeAxis,
    span: Range<usize>,
    mode: x11::Mode,
) -> Result<diagnostics::MovingSeasonalityTest, Unavailable> {
    let period = usize::from(axis.frequency().periods_per_year());
    let initial = usize::from(axis.start().subperiod()) - 1;
    let start = span.start + (period - (initial + span.start) % period) % period;
    let end = span.end.saturating_sub((initial + span.end) % period);
    let span = start..end;
    let eligible = sample(axis, span.clone());
    let calculate = || -> Result<diagnostics::FTest, Reason> {
        if end > values.len() || end.saturating_sub(start) < 2 * period {
            return Err(Reason::InsufficientSample);
        }
        let deviations = values[span.clone()]
            .iter()
            .map(|&x| {
                if multiplicative(mode) {
                    (x - 1.0).abs()
                } else {
                    x.abs()
                }
            })
            .collect::<Vec<_>>();
        if !deviations.iter().all(|x| x.is_finite()) {
            return Err(Reason::NonFiniteArithmetic);
        }
        let years = deviations.len() / period;
        let period_len = period as f64;
        let year_count = years as f64;
        let scale = if multiplicative(mode) { 10000.0 } else { 1.0 };
        let mut sum = 0.0;
        for &value in &deviations {
            sum += value;
        }
        let grand_mean = sum / (period_len * year_count);
        let mut slot_ss = 0.0;
        for slot in 0..period {
            let mut slot_mean = 0.0;
            for j in (slot..deviations.len()).step_by(period) {
                slot_mean += deviations[j];
            }
            slot_mean /= year_count;
            slot_ss = square_add(slot_ss, slot_mean - grand_mean);
        }
        slot_ss = slot_ss * year_count * scale;
        let mut year_ss = 0.0;
        for year in deviations.chunks_exact(period) {
            let mut year_mean = 0.0;
            for &value in year {
                year_mean += value;
            }
            year_mean /= period_len;
            year_ss = square_add(year_ss, year_mean - grand_mean);
        }
        year_ss = year_ss * period_len * scale;
        let mut total_ss = 0.0;
        for &value in &deviations {
            total_ss = square_add(total_ss, value - grand_mean);
        }
        let residual_ss = scale.mul_add(total_ss, -slot_ss) - year_ss;
        let numerator = years - 1;
        let denominator = (period - 1) * numerator;
        let year_ms = year_ss / numerator as f64;
        let residual_ms = residual_ss / ((period_len - 1.0) * (year_count - 1.0));
        if ![
            grand_mean,
            slot_ss,
            year_ss,
            total_ss,
            residual_ss,
            year_ms,
            residual_ms,
        ]
        .iter()
        .all(|v| v.is_finite())
        {
            return Err(Reason::NonFiniteArithmetic);
        }
        if crate::numerics::build_62_equal(residual_ms, 0.0) {
            return Err(Reason::ZeroResidualMeanSquare);
        }
        let (statistic, probability) = fvalue(year_ms / residual_ms, numerator, denominator)?;
        Ok(diagnostics::FTest {
            statistic,
            upper_tail_probability_percent: probability * 100.0,
            numerator_degrees_of_freedom: numerator,
            denominator_degrees_of_freedom: denominator,
            sample: eligible.clone().ok_or(Reason::InsufficientSample)?,
            differencing_lag: None,
            small_residual_mean_square: false,
        })
    };
    match calculate() {
        Ok(test) => Ok(diagnostics::MovingSeasonalityTest {
            conclusion: moving_conclusion(test.upper_tail_probability_percent),
            test,
        }),
        Err(reason) => Err(missing(Stage::Moving, reason, eligible, None)),
    }
}

/// Degrees of freedom across one year's calendar periods.
const fn calendar_degrees_of_freedom(frequency: Frequency) -> NonZeroUsize {
    match frequency {
        Frequency::Monthly => const { NonZeroUsize::new(11).unwrap() },
        Frequency::Quarterly => const { NonZeroUsize::new(3).unwrap() },
    }
}

/// `kwtest`: the Kruskal-Wallis test of `values[span]` between calendar
/// periods.
fn kruskal_wallis(
    values: &[f64],
    axis: TimeAxis,
    span: Range<usize>,
) -> Result<diagnostics::KruskalWallisTest, Unavailable> {
    let eligible = sample(axis, span.clone());
    let calculate = || -> Result<diagnostics::KruskalWallisTest, Reason> {
        let period = usize::from(axis.frequency().periods_per_year());
        if span.end > values.len() || span.end.saturating_sub(span.start) < period {
            return Err(Reason::InsufficientSample);
        }
        let mut data = values[span.clone()].to_vec();
        if !data.iter().all(|v| v.is_finite()) {
            return Err(Reason::NonFiniteArithmetic);
        }
        // The slot rank sums total n(n+1)/2 over `period` slots, so by
        // pigeonhole the largest is at least the ceiling of that over
        // `period`. Above isqrt(i32::MAX) = 46,340 its i32 square below (or
        // the sum itself) overflows for every ranking, so the result is
        // already `RankIntegerOverflow`. Return it before the quadratic tie
        // ranking: monthly from n = 1,055, quarterly from n = 609.
        let n = data.len() as u128;
        let largest_slot_sum = (n * (n + 1) / 2).div_ceil(period as u128);
        if largest_slot_sum > u128::from(i32::MAX.unsigned_abs().isqrt()) {
            return Err(Reason::RankIntegerOverflow);
        }
        let mut indices = span.clone().collect::<Vec<_>>();
        indices.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
        // With distinct finite values there is exactly one ranking, so
        // sorting the indices gives `kwtest`'s ranks without its quadratic
        // swaps. Equal values (signed zeros included) need its
        // strict-greater swaps, which leave ties in neither stable nor
        // average ranks.
        if indices
            .windows(2)
            .any(|pair| values[pair[0]] == values[pair[1]])
        {
            for (index, original) in indices.iter_mut().zip(span.clone()) {
                *index = original;
            }
            for i in 0..data.len() {
                for j in i..data.len() {
                    if data[i] > data[j] {
                        data.swap(i, j);
                        indices.swap(i, j);
                    }
                }
            }
        }
        let mut sums = [0_i32; 12];
        let mut counts = [0_i32; 12];
        for (rank, index) in indices.into_iter().enumerate() {
            let slot = (usize::from(axis.start().subperiod()) - 1 + index) % period;
            counts[slot] = counts[slot]
                .checked_add(1)
                .ok_or(Reason::RankIntegerOverflow)?;
            sums[slot] = sums[slot]
                .checked_add(i32::try_from(rank + 1).map_err(|_| Reason::RankIntegerOverflow)?)
                .ok_or(Reason::RankIntegerOverflow)?;
        }
        let mut weighted_rank_squares = 0.0;
        for slot in 0..period {
            debug_assert!(counts[slot] > 0, "a span of one year fills every slot");
            let squared = sums[slot]
                .checked_mul(sums[slot])
                .ok_or(Reason::RankIntegerOverflow)?;
            weighted_rank_squares += f64::from(squared) / f64::from(counts[slot]);
        }
        let n = i32::try_from(data.len()).map_err(|_| Reason::RankIntegerOverflow)?;
        let next = n.checked_add(1).ok_or(Reason::RankIntegerOverflow)?;
        let divisor = n.checked_mul(next).ok_or(Reason::RankIntegerOverflow)?;
        let centered = 3_i32.checked_mul(next).ok_or(Reason::RankIntegerOverflow)?;
        let statistic = 12.0 * weighted_rank_squares / f64::from(divisor) - f64::from(centered);
        let degrees_of_freedom = calendar_degrees_of_freedom(axis.frequency());
        let probability =
            crate::numerics::probability::chi_square_upper_tail(statistic, degrees_of_freedom)
                .map_err(|_| Reason::NonFiniteArithmetic)?
                * 100.0;
        Ok(diagnostics::KruskalWallisTest {
            statistic,
            upper_tail_probability_percent: probability,
            degrees_of_freedom: degrees_of_freedom.get(),
            sample: eligible.clone().ok_or(Reason::InsufficientSample)?,
            conclusion: kw_conclusion(probability),
        })
    };
    calculate().map_err(|reason| missing(Stage::KruskalWallis, reason, eligible, None))
}

/// `combft`: the combined test for identifiable seasonality. A test whose
/// result it does not come to read may be unavailable.
fn combined(
    stable: &Result<diagnostics::StableSeasonalityTest, Unavailable>,
    moving: &Result<diagnostics::MovingSeasonalityTest, Unavailable>,
    kw: &Result<diagnostics::KruskalWallisTest, Unavailable>,
) -> Result<diagnostics::CombinedSeasonality, Unavailable> {
    let missing = |stage| unavailable(Stage::Combined, Reason::PrerequisiteUnavailable { stage });
    let Ok(stable) = stable else {
        return missing(Stage::FinalStable);
    };
    if stable.test.upper_tail_probability_percent >= 0.1 {
        return Ok(diagnostics::CombinedSeasonality::NotPresent);
    }
    let Ok(moving) = moving else {
        return missing(Stage::Moving);
    };
    let fs = stable.test.statistic;
    let fm = moving.test.statistic;
    let scaled_stable = fs * 9.0;
    let scaled_moving = if fs > 0.0 { 3.0 * fm } else { 0.0 };
    if !scaled_stable.is_finite() || !scaled_moving.is_finite() {
        return unavailable(Stage::Combined, Reason::NonFiniteArithmetic);
    }
    let test1 = if scaled_stable >= 7.0 { 7.0 / fs } else { 9.0 };
    let mut test2 = if fs > 0.0 { scaled_moving / fs } else { 9.0 };
    if !test2.is_finite() {
        return unavailable(Stage::Combined, Reason::NonFiniteArithmetic);
    }
    if test2 > 9.0 {
        test2 = 9.0;
    }
    if ![test1, test2].iter().all(|v| v.is_finite()) {
        return unavailable(Stage::Combined, Reason::NonFiniteArithmetic);
    }
    if moving.test.upper_tail_probability_percent <= 5.0 && (test1 + test2) / 2.0 >= 1.0 {
        return Ok(diagnostics::CombinedSeasonality::NotPresent);
    }
    if test1 >= 1.0 || test2 >= 1.0 {
        return Ok(diagnostics::CombinedSeasonality::ProbablyNotPresent);
    }
    let Ok(kw) = kw else {
        return missing(Stage::KruskalWallis);
    };
    Ok(if kw.upper_tail_probability_percent <= 0.1 {
        diagnostics::CombinedSeasonality::Present
    } else {
        diagnostics::CombinedSeasonality::ProbablyNotPresent
    })
}

/// The seasonality tests of the final unmodified SI ratios `d8`: stable over
/// the observed range, Kruskal-Wallis and moving from the start of the
/// extended series, and their combination. The residual tests are
/// placeholders until [`complete_residual`] and [`complete_modified`] run.
pub(super) fn core(
    preliminary_stable: Result<diagnostics::StableSeasonalityTest, Unavailable>,
    d8: &[f64],
    axis: TimeAxis,
    observed: Range<usize>,
    mode: x11::Mode,
) -> diagnostics::SeasonalityTests {
    let final_stable = stable(d8, axis, observed.clone(), mode, Stage::FinalStable);
    let kruskal_wallis = kruskal_wallis(d8, axis, 0..observed.end);
    let moving = moving(d8, axis, 0..observed.end, mode);
    let combined = combined(&final_stable, &moving, &kruskal_wallis);
    diagnostics::SeasonalityTests {
        preliminary_stable,
        final_stable,
        moving,
        kruskal_wallis,
        combined,
        residual_full: unavailable(Stage::ResidualFull, Reason::InsufficientSample),
        modified_residual_full: unavailable(
            Stage::ModifiedResidualFull,
            Reason::InsufficientSample,
        ),
        modified_residual_recent_three_years: unavailable(
            Stage::ModifiedResidualRecentThreeYears,
            Reason::PrerequisiteUnavailable {
                stage: Stage::ModifiedResidualFull,
            },
        ),
        residual_recent_three_years: unavailable(
            Stage::ResidualRecentThreeYears,
            Reason::PrerequisiteUnavailable {
                stage: Stage::ResidualFull,
            },
        ),
    }
}

/// Warns about a test that is unavailable for a reason Build 62 prints a note
/// for: a zero residual mean square or a constant original series.
pub(super) fn append_warning<T>(record: &Result<T, Unavailable>, warnings: &mut Vec<Warning>) {
    if let Err(Unavailable(UnavailableDetail::SeasonalityTest {
        test,
        reason: Reason::ZeroResidualMeanSquare | Reason::ConstantOriginalSeries,
        ..
    })) = record
    {
        warnings.push(Warning::SeasonalityTestUnavailable { test: *test });
    }
}

/// The residual seasonality tests of the final seasonally adjusted series
/// (D11): over the whole series and over its last three years.
pub(super) fn complete_residual(
    components: &mut super::Components,
    original: &[f64],
    axis: TimeAxis,
) {
    let (full, recent) = residual_pair(
        &components.seasonally_adjusted,
        original,
        axis,
        components.mode,
        Stage::ResidualFull,
        Stage::ResidualRecentThreeYears,
    );
    append_warning(&full, &mut components.warnings);
    append_warning(&recent, &mut components.warnings);
    components.seasonality.residual_full = full;
    components.seasonality.residual_recent_three_years = recent;
}

/// The seasonally adjusted series modified for extremes (table E2): where the
/// extreme weight is zero, the trend-cycle takes the value's place.
pub(crate) fn modified_adjusted(
    adjusted: &[f64],
    base_trend: &[f64],
    weights: &[f64],
) -> Result<Vec<f64>, Error> {
    if adjusted.len() != base_trend.len() || adjusted.len() != weights.len() {
        return Err(Error::internal("table E2 inputs differ in length"));
    }
    Ok(adjusted
        .iter()
        .zip(base_trend)
        .zip(weights)
        .map(|((&value, &trend), &weight)| if weight > 0.0 { value } else { trend })
        .collect())
}

/// The residual seasonality tests of the modified seasonally adjusted series,
/// which is also kept for its spectrum.
pub(super) fn complete_modified(
    components: &mut super::Components,
    modified: Vec<f64>,
    original: &[f64],
    axis: TimeAxis,
) {
    let (full, recent) = residual_pair(
        &modified,
        original,
        axis,
        components.mode,
        Stage::ModifiedResidualFull,
        Stage::ModifiedResidualRecentThreeYears,
    );
    super::spectral::retain_modified(components, &modified);
    append_warning(&full, &mut components.warnings);
    append_warning(&recent, &mut components.warnings);
    components.seasonality.modified_residual_full = full;
    components.seasonality.modified_residual_recent_three_years = recent;
}

/// `ftest` on the changes of `values` over one quarter: for the whole series
/// and, when that test is available, for the last three years. A constant
/// original series has neither.
fn residual_pair(
    values: &[f64],
    original: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
    full_stage: Stage,
    recent_stage: Stage,
) -> (
    Result<diagnostics::ResidualSeasonalityTest, Unavailable>,
    Result<diagnostics::ResidualSeasonalityTest, Unavailable>,
) {
    let period = usize::from(axis.frequency().periods_per_year());
    let lag = period / 4;
    let count = values.len();
    let span = lag..count;
    let mode = if mode == x11::Mode::LogAdditive {
        x11::Mode::Multiplicative
    } else {
        mode
    };
    let mut differences = vec![0.0; count];
    let full = if crate::numerics::is_constant_series(original) {
        Err(missing(
            full_stage,
            Reason::ConstantOriginalSeries,
            sample(axis, span.clone()),
            Some(lag),
        ))
    } else {
        for i in span.clone() {
            differences[i] = values[i] - values[i - lag];
        }
        residual(&differences, axis, span, mode, lag, full_stage)
    };
    let recent = if full.is_ok() {
        if let Some(start) = count.checked_sub(3 * period).filter(|start| *start >= lag) {
            residual(&differences, axis, start..count, mode, lag, recent_stage)
        } else {
            unavailable(recent_stage, Reason::InsufficientSample)
        }
    } else {
        unavailable(
            recent_stage,
            Reason::PrerequisiteUnavailable { stage: full_stage },
        )
    };
    (full, recent)
}

/// The residual seasonality warning of the preliminary pass of a legacy
/// Easter adjustment. With `Khol = 1`, `ftest` still notes a constant
/// original series or a zero residual mean square before it returns without
/// printing, and the last-three-years test is not run.
pub(super) fn preliminary_easter_warning(
    values: &[f64],
    original: &[f64],
    axis: TimeAxis,
    mode: x11::Mode,
) -> Vec<Unavailable> {
    let lag = usize::from(axis.frequency().periods_per_year()) / 4;
    let span = lag..values.len();
    let reason = if crate::numerics::is_constant_series(original) {
        Some(Reason::ConstantOriginalSeries)
    } else {
        let mut differences = vec![0.0; values.len()];
        for i in span.clone() {
            differences[i] = values[i] - values[i - lag];
        }
        f_test(&differences, axis, span.clone(), mode, Some(lag)).err()
    };
    reason
        .filter(|r| {
            matches!(
                r,
                Reason::ConstantOriginalSeries | Reason::ZeroResidualMeanSquare
            )
        })
        .map(|reason| missing(Stage::ResidualFull, reason, sample(axis, span), Some(lag)))
        .into_iter()
        .collect()
}

/// One residual seasonality test of the differenced `values[span]`.
fn residual(
    values: &[f64],
    axis: TimeAxis,
    span: Range<usize>,
    mode: x11::Mode,
    lag: usize,
    stage: Stage,
) -> Result<diagnostics::ResidualSeasonalityTest, Unavailable> {
    let recent = matches!(
        stage,
        Stage::ResidualRecentThreeYears | Stage::ModifiedResidualRecentThreeYears
    );
    match f_test(values, axis, span.clone(), mode, Some(lag)) {
        Ok(test) => Ok(diagnostics::ResidualSeasonalityTest {
            conclusion: residual_conclusion(test.upper_tail_probability_percent, recent),
            test,
        }),
        Err(reason) => Err(missing(stage, reason, sample(axis, span), Some(lag))),
    }
}

#[cfg(test)]
mod tests;
