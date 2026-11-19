//! Shrinkage of the final seasonal factors: Build 62 `shrink.f`, `glbshk.f`,
//! `locshk.f` and `lkshnk.f`.

use super::seasonal::ResolvedSeasonalFilter;
use crate::error::{
    NumericalDetail, X11ShrinkageStage as Stage, X11ShrinkageUnavailableReason as Reason,
};
use crate::numerics::BUILD_62_PI;
use crate::time::{AxisSpan, TimeAxis};
use crate::{Error, x11};
use std::borrow::Cow;
use std::ops::Range;

/// Where the seasonal factors to shrink sit in the run.
pub(super) struct ShrinkageContext {
    pub(super) mode: x11::Mode,
    /// The dates of the seasonal factors, backcasts and forecasts included.
    pub(super) dates: AxisSpan,
    /// The start of the requested span, which the run's own span may not
    /// reach back to.
    pub(super) requested_axis: TimeAxis,
    /// The observations among the seasonal factors.
    pub(super) observed: Range<usize>,
    /// The seasonal filter applied last.
    pub(super) last_filter: ResolvedSeasonalFilter,
}

impl ShrinkageContext {
    fn period(&self) -> usize {
        usize::from(self.dates.start().frequency().periods_per_year())
    }

    fn error(&self, policy: x11::SeasonalShrinkage, stage: Stage, reason: Reason) -> Error {
        Error::from(NumericalDetail::X11Shrinkage {
            policy,
            first: self.dates.period(self.observed.start),
            last: self.dates.period(self.observed.end.saturating_sub(1)),
            stage,
            reason,
        })
    }

    /// The number of periods from January of the requested start year to the
    /// first seasonal factor. Build 62's indices (`setxpt.f`, `nofcst.f`)
    /// count from the requested start year even when the run does not reach
    /// back to it.
    fn prefix(&self) -> usize {
        let start = self.dates.start();
        let years = i64::from(start.year()) - i64::from(self.requested_axis.start().year());
        debug_assert!(years >= 0);
        years as usize * self.period() + usize::from(start.subperiod()) - 1
    }
}

/// `forcst.f`: one more year of seasonal factors, each calendar period's last
/// factor plus half its last yearly change.
pub(super) fn prepare_tail(seasonal: &[f64], context: &ShrinkageContext) -> Vec<f64> {
    let period = context.period();
    debug_assert!(seasonal.len() >= 2 * period);
    let end = seasonal.len();
    (0..period)
        .map(|i| {
            let previous = seasonal[end - period + i];
            // gfortran fuses this multiply-add in forcst.f.
            (previous - seasonal[end - 2 * period + i]).mul_add(0.5, previous)
        })
        .collect()
}

/// `shrink.f`: the sum of the seasonal filter's squared weights, accumulated
/// term by term. The stable filter has none.
fn filter_weight(filter: ResolvedSeasonalFilter) -> Option<f64> {
    let length = match filter {
        ResolvedSeasonalFilter::S3x1 => return Some(1.0 / 3.0),
        ResolvedSeasonalFilter::S3x3 => 3,
        ResolvedSeasonalFilter::S3x5 => 5,
        ResolvedSeasonalFilter::S3x9 => 9,
        ResolvedSeasonalFilter::S3x15 => 15,
        ResolvedSeasonalFilter::Stable => return None,
    };
    let width = length as f64 * 3.0;
    let denominator = width * width;
    let mut weight = 10.0 / denominator;
    for _ in 0..length - 2 {
        weight += 9.0 / denominator;
    }
    Some(weight)
}

/// `shrink.f`, global: the mean over calendar slots of each slot's sample
/// variance of SI about the final seasonal factors. Each slot needs two
/// observations.
fn global_variance(si: &[f64], seasonal: &[f64], observed: Range<usize>, period: usize) -> f64 {
    let mut total = 0.0;
    for first in observed.start..observed.start + period {
        let mut sum = 0.0;
        let mut count = 0.0;
        for j in (first..observed.end).step_by(period) {
            count += 1.0;
            let difference = si[j] - seasonal[j];
            // gfortran fuses the square into the sum in shrink.f.
            sum = difference.mul_add(difference, sum);
        }
        debug_assert!(count > 1.0);
        total += sum / (count - 1.0);
    }
    total / period as f64
}

/// `shrink.f`, local: the squared SI deviations about the final seasonal
/// factors, divided by the observed count less one year.
fn local_variance(si: &[f64], seasonal: &[f64], observed: Range<usize>, period: usize) -> f64 {
    debug_assert!(observed.len() > period);
    let denominator = (observed.len() - period) as f64;
    let mut total = 0.0;
    for j in observed {
        let difference = si[j] - seasonal[j];
        // gfortran fuses the square into the sum in shrink.f.
        total = difference.mul_add(difference, total);
    }
    total / denominator
}

/// `lkshnk.f`: the normal density of `first - second` for the given variance,
/// with Build 62's own value of pi.
fn density(first: f64, second: f64, variance: f64) -> f64 {
    let difference = first - second;
    (1.0 / (variance * 2.0 * BUILD_62_PI).sqrt())
        * (-0.5 * ((difference * difference) / variance)).exp()
}

/// `glbshk.f`: `seasonal` moved toward one by `weight`. gfortran rounds
/// `1 - weight`, then fuses the multiply-add.
fn global_blend(seasonal: f64, weight: f64) -> f64 {
    seasonal.mul_add(1.0 - weight, weight)
}

/// `locshk.f`: one more term of a weighted sum of seasonal factors, which
/// gfortran fuses.
fn local_accumulate(seasonal: f64, weight: f64, sum: f64) -> f64 {
    seasonal.mul_add(weight, sum)
}

/// SI or the forecast tail in the original domain: log-additive values are
/// antilogged before shrinkage.
fn original_domain(values: &[f64], mode: x11::Mode) -> Cow<'_, [f64]> {
    if mode == x11::Mode::LogAdditive {
        Cow::Owned(values.iter().map(|v| v.exp()).collect())
    } else {
        Cow::Borrowed(values)
    }
}

/// The final seasonal factors at Build 62's `Sts` indices, which count from
/// January of the requested start year.
struct AbsoluteFactors<'a> {
    /// The number of indices before the first stored factor. They read as
    /// `identity`.
    prefix: usize,
    /// Zero for an additive run, otherwise one.
    identity: f64,
    seasonal: &'a [f64],
    /// The forecast tail beyond the stored factors.
    tail: &'a [f64],
}

impl AbsoluteFactors<'_> {
    fn at(&self, index: usize) -> f64 {
        if index < self.prefix {
            self.identity
        } else {
            let i = index - self.prefix;
            if i < self.seasonal.len() {
                self.seasonal[i]
            } else {
                self.tail[i - self.seasonal.len()]
            }
        }
    }

    /// One past the last stored factor's index.
    fn stored_end(&self) -> usize {
        self.prefix + self.seasonal.len()
    }
}

/// A failed shrinkage step, before the policy and span are attached.
type StepFailure = (Stage, Reason);

/// `shrink.f`: shrinks the observed `seasonal` factors in place by `policy`.
/// `si` is the final SI the factors were smoothed from and `tail` the year
/// [`prepare_tail`] extrapolated beyond them.
pub(super) fn apply(
    policy: x11::SeasonalShrinkage,
    si: &[f64],
    seasonal: &mut [f64],
    tail: &[f64],
    context: ShrinkageContext,
) -> Result<(), Error> {
    if policy == x11::SeasonalShrinkage::None {
        return Ok(());
    }
    let fail = |stage, reason| context.error(policy, stage, reason);
    let period = context.period();
    let observed = context.observed.clone();
    let forecasts = seasonal.len() - observed.end;
    // locshk.f copies Sts only through Posffc, then reads its copy up to
    // Posfob+Ny/2-1. Without these forecasts it reads undefined values, so
    // this is an error.
    if policy == x11::SeasonalShrinkage::Local && forecasts < period / 2 - 1 {
        return Err(fail(
            Stage::LocalLikelihood,
            Reason::InsufficientAcceptedForecasts {
                actual: forecasts,
                minimum: period / 2 - 1,
            },
        ));
    }
    let weight = filter_weight(context.last_filter).ok_or_else(|| {
        fail(
            Stage::FilterWeight,
            Reason::UnsupportedFilter {
                filter: context.last_filter.into(),
            },
        )
    })?;
    let global = policy == x11::SeasonalShrinkage::Global;
    // An X-11 run has three years of observations, more than either variance
    // needs.
    let needed = if global {
        2 * period
    } else {
        period + period / 2
    };
    debug_assert!(observed.len() >= needed);
    let si = original_domain(si, context.mode);
    let tail = original_domain(tail, context.mode);
    if !si
        .iter()
        .chain(tail.iter())
        .chain(seasonal.iter())
        .all(|v| v.is_finite())
    {
        return Err(fail(Stage::SeasonalFactors, Reason::NonFiniteArithmetic));
    }
    let variance = if global {
        global_variance(&si, seasonal, observed.clone(), period)
    } else {
        local_variance(&si, seasonal, observed.clone(), period)
    };
    let variance = weight * variance;
    if !variance.is_finite() {
        return Err(fail(Stage::Variance, Reason::NonFiniteArithmetic));
    }
    let prefix = context.prefix();
    let span = prefix + observed.start..prefix + observed.end;
    let factors = AbsoluteFactors {
        prefix,
        identity: if context.mode == x11::Mode::Additive {
            0.0
        } else {
            1.0
        },
        seasonal: &*seasonal,
        tail: &tail,
    };
    let shrunk = if global {
        global_shrink(&factors, span, period, variance)
    } else {
        local_shrink(&factors, span, period, variance)
    };
    let output = shrunk.map_err(|(stage, reason)| fail(stage, reason))?;
    if !output.iter().all(|v| v.is_finite()) {
        return Err(fail(Stage::SeasonalFactors, Reason::NonFiniteArithmetic));
    }
    seasonal[observed].copy_from_slice(&output);
    Ok(())
}

/// `glbshk.f`: moves each observed factor toward one, by a weight from its
/// calendar year's spread about the identity. `span` is the observed span at
/// `Sts` indices.
fn global_shrink(
    factors: &AbsoluteFactors<'_>,
    span: Range<usize>,
    period: usize,
    variance: f64,
) -> Result<Vec<f64>, StepFailure> {
    let (first, last) = (span.start, span.end);
    let count = span.len();
    let offset = first % period;
    let years = (offset + count).div_ceil(period);
    // The observed factors in whole calendar years, completed from the
    // neighbouring year at each end.
    let mut aligned = vec![0.0; years * period];
    aligned[offset..offset + count]
        .copy_from_slice(&factors.seasonal[first - factors.prefix..last - factors.prefix]);
    for i in 1..=offset {
        let source = first - i;
        aligned[offset - i] = factors.at(if source >= factors.prefix {
            source
        } else {
            source + period
        });
    }
    for (i, value) in aligned.iter_mut().enumerate().skip(offset + count) {
        let source = last + i - offset - count;
        *value = factors.at(if source < factors.stored_end() {
            source
        } else {
            source - period
        });
    }
    let mut weights = Vec::with_capacity(years);
    for year in 0..years {
        let mut sum = 0.0;
        for i in year * period..(year + 1) * period {
            let difference = factors.at(i) - factors.identity;
            // gfortran fuses the square into the sum in glbshk.f.
            sum = difference.mul_add(difference, sum);
        }
        let mut excess = sum / (period - 1) as f64 - variance;
        if excess < 0.0 {
            excess = 0.0;
        }
        let denominator = variance + excess;
        if denominator == 0.0 {
            return Err((Stage::GlobalWeight, Reason::ZeroDenominator));
        }
        let weight = ((period - 3) as f64 / (period - 1) as f64) * variance / denominator;
        if !denominator.is_finite() || !weight.is_finite() {
            return Err((Stage::GlobalWeight, Reason::NonFiniteArithmetic));
        }
        weights.push(weight);
    }
    Ok((offset..offset + count)
        .map(|i| global_blend(aligned[i], weights[i / period]))
        .collect())
}

/// `locshk.f`: replaces each observed factor by a likelihood-weighted mean of
/// the factors within half a year of it, the weights averaged over its
/// calendar period. `span` is the observed span at `Sts` indices and holds at
/// least a year and a half.
fn local_shrink(
    factors: &AbsoluteFactors<'_>,
    span: Range<usize>,
    period: usize,
    variance: f64,
) -> Result<Vec<f64>, StepFailure> {
    if variance <= 0.0 {
        return Err((Stage::Variance, Reason::NonPositiveLocalVariance));
    }
    let (first, last) = (span.start, span.end);
    let half = period / 2;
    let mut likelihoods = vec![[0.0; 12]; span.len() - half];
    for (row, i) in likelihoods.iter_mut().zip(first + half..last) {
        let mut sum = 0.0;
        for (j, value) in row.iter_mut().take(period).enumerate() {
            *value = density(factors.at(i), factors.at(i + j - half), variance);
            sum += *value;
        }
        if !sum.is_finite() {
            return Err((Stage::LocalLikelihood, Reason::NonFiniteArithmetic));
        }
        if sum <= 0.0 {
            return Err((Stage::LocalLikelihood, Reason::NonPositiveLikelihoodSum));
        }
        for value in &mut row[..period] {
            *value /= sum;
        }
    }
    let mut weights = [[0.0; 12]; 12];
    for first_row in 0..period {
        let calendar = (first + half + first_row) % period;
        let mut years = 0.0;
        for row in likelihoods.iter().skip(first_row).step_by(period) {
            years += 1.0;
            for (weight, &likelihood) in weights[calendar].iter_mut().zip(&row[..period]) {
                *weight += likelihood;
            }
        }
        debug_assert!(years > 0.0);
        for value in &mut weights[calendar][..period] {
            *value /= years;
        }
    }
    Ok(span
        .map(|i| {
            let mut sum = 0.0;
            for (j, &weight) in weights[i % period][..period].iter().enumerate() {
                // locshk.f reads a neighbour before the first weighted factor
                // one year later.
                let source = if i + j < first + half {
                    i + j + period - half
                } else {
                    i + j - half
                };
                sum = local_accumulate(factors.at(source), weight, sum);
            }
            sum
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::NumericalFailure;

    /// Twelve additive quarterly observations from 2000Q1, last smoothed by a
    /// 3x1 filter.
    fn context() -> ShrinkageContext {
        ShrinkageContext {
            mode: x11::Mode::Additive,
            dates: TimeAxis::quarterly(2000, 1).unwrap().span(12).unwrap(),
            requested_axis: TimeAxis::quarterly(2000, 1).unwrap(),
            observed: 0..12,
            last_filter: ResolvedSeasonalFilter::S3x1,
        }
    }

    #[test]
    fn variances_fuse_the_square_into_the_sum() {
        let mut si = [0.; 12];
        si[0] = 2_f64.powi(-27);
        si[4] = si[0];
        si[8] = 1.0 + f64::EPSILON;
        let expected = f64::from_bits(1.0f64.to_bits() + 3) / 8.0;
        assert_eq!(
            global_variance(&si, &[0.; 12], 0..12, 4).to_bits(),
            expected.to_bits()
        );
        assert_eq!(
            local_variance(&si, &[0.; 12], 0..12, 4).to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn blend_and_weighted_sum_are_fused() {
        // The exact product plus the addend, rounded once.
        assert_eq!(global_blend(0.1, 1.0 / 3.0).to_bits(), 0x3fd9999999999999);
        assert_eq!(
            local_accumulate(2.3, 0.7, 0.1).to_bits(),
            0x3ffb5c28f5c28f5b
        );
    }

    #[test]
    fn tail_extrapolates_one_year() {
        assert_eq!(
            prepare_tail(
                &[1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8.],
                &context()
            ),
            vec![6., 7., 8., 9.]
        );
    }

    #[test]
    fn tail_rounds_only_at_the_addition() {
        let tiny = f64::from_bits(1);
        let mut values = [2.0 * tiny; 8];
        values[4..].fill(3.0 * tiny);
        let tail = prepare_tail(&values, &context());
        assert!(tail.iter().all(|value| value.to_bits() == 4));
    }

    #[test]
    fn global_variance_is_by_calendar_slot_and_local_is_not() {
        let si = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
        assert_eq!(global_variance(&si, &[0.; 9], 0..9, 4), 57.875);
        assert_eq!(local_variance(&si, &[0.; 9], 0..9, 4), 57.0);
    }

    #[test]
    fn filter_weights_accumulate_term_by_term() {
        assert_eq!(filter_weight(ResolvedSeasonalFilter::S3x1), Some(1.0 / 3.0));
        assert_eq!(
            filter_weight(ResolvedSeasonalFilter::S3x3),
            Some(10.0 / 81.0 + 9.0 / 81.0)
        );
        let mut expected = 10.0 / 2025.0;
        for _ in 0..13 {
            expected += 9.0 / 2025.0;
        }
        assert_eq!(filter_weight(ResolvedSeasonalFilter::S3x15), Some(expected));
        assert_eq!(
            filter_weight(ResolvedSeasonalFilter::S3x9)
                .unwrap()
                .to_bits(),
            0x3fb9a2970059e603
        );
        assert_ne!(
            filter_weight(ResolvedSeasonalFilter::S3x9).unwrap(),
            73.0 / 729.0
        );
        assert_eq!(filter_weight(ResolvedSeasonalFilter::Stable), None);
    }

    #[test]
    fn prefix_counts_from_the_requested_start_year() {
        // The run's factors start at the requested start when it uses the
        // backcasts, and at the first observation when it does not.
        for (observed_axis, backcasts, uses_backcasts, expected) in [
            (TimeAxis::monthly(2000, 7).unwrap(), 0, true, 6),
            (TimeAxis::monthly(2000, 1).unwrap(), 13, true, 11),
            (TimeAxis::monthly(2000, 1).unwrap(), 13, false, 24),
            (TimeAxis::monthly(-2000, 7).unwrap(), 5, true, 1),
            (TimeAxis::monthly(-2000, 7).unwrap(), 5, false, 6),
        ] {
            let requested = observed_axis.shift(-backcasts).unwrap();
            let axis = if uses_backcasts {
                requested
            } else {
                observed_axis
            };
            let context = ShrinkageContext {
                dates: axis.span(120).unwrap(),
                requested_axis: requested,
                observed: 0..120,
                ..context()
            };
            assert_eq!(context.prefix(), expected);
        }
    }

    #[test]
    fn global_year_weights_read_the_identity_before_the_first_factor() {
        // From Q3, the first year's sum of squares holds two identities and
        // the last year's two tail values.
        let axis = TimeAxis::quarterly(2000, 3).unwrap();
        for (mode, expected_first, expected_rest) in [
            (x11::Mode::Additive, 1.9375, 1.96875),
            (x11::Mode::Multiplicative, 1.75, 1.875),
        ] {
            let context = ShrinkageContext {
                mode,
                dates: axis.span(12).unwrap(),
                requested_axis: axis,
                ..context()
            };
            let mut seasonal = [2.; 12];
            apply(
                x11::SeasonalShrinkage::Global,
                &[3.; 12],
                &mut seasonal,
                &[2.; 4],
                context,
            )
            .unwrap();
            assert_eq!(&seasonal[..2], &[expected_first; 2]);
            assert_eq!(&seasonal[2..], &[expected_rest; 10]);
        }
    }

    #[test]
    fn log_additive_shrinks_the_antilogged_si_and_tail() {
        let axis = TimeAxis::quarterly(2000, 3).unwrap();
        let seasonal: Vec<_> = (0..12).map(|i| 0.9 + (i % 4) as f64 * 0.07).collect();
        let log_si: Vec<_> = (0..12).map(|i| ((i * 5) % 7) as f64 * 0.03 - 0.1).collect();
        let log_tail = [-0.1, -0.03, 0.04, 0.11];
        let shrink = |mode, si: &[f64], tail: &[f64]| {
            let context = ShrinkageContext {
                mode,
                dates: axis.span(12).unwrap(),
                requested_axis: axis,
                ..context()
            };
            let mut shrunk = seasonal.clone();
            apply(
                x11::SeasonalShrinkage::Global,
                si,
                &mut shrunk,
                tail,
                context,
            )
            .unwrap();
            shrunk
        };
        let si: Vec<_> = log_si.iter().map(|value| value.exp()).collect();
        let tail = log_tail.map(f64::exp);
        let expected = shrink(x11::Mode::Multiplicative, &si, &tail);
        assert_ne!(expected, seasonal);
        assert_eq!(shrink(x11::Mode::LogAdditive, &log_si, &log_tail), expected);
    }

    #[test]
    fn density_underflows_to_zero_far_from_its_center() {
        assert_eq!(density(0.0, 0.0, 0.5).to_bits(), 0x3fe2_0dd7_5042_9b6f);
        assert_eq!(density(1.0, 100.0, 0.0001), 0.0);
        let center = density(1.0, 1.0, 0.0001);
        assert!(center.is_finite() && center > 0.0);
    }

    #[test]
    fn local_rejects_a_zero_likelihood_sum() {
        // The variance is finite, but its product with 2 pi overflows and
        // every density is zero.
        let mut si = [1.; 7];
        si[0] = 1.33e154;
        let context = ShrinkageContext {
            observed: 0..6,
            ..context()
        };
        assert!(matches!(
            apply(
                x11::SeasonalShrinkage::Local,
                &si,
                &mut [1.; 7],
                &[1.; 4],
                context
            ),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Shrinkage {
                    stage: Stage::LocalLikelihood,
                    reason: Reason::NonPositiveLikelihoodSum,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn local_needs_half_a_year_of_forecasts_less_one() {
        for (axis, period) in [
            (TimeAxis::monthly(2000, 7).unwrap(), 12),
            (TimeAxis::quarterly(2000, 3).unwrap(), 4),
        ] {
            let count = period * 6;
            let minimum = period / 2 - 1;
            for forecasts in [minimum - 1, minimum] {
                let mut seasonal = (0..count + forecasts)
                    .map(|i| 1.0 + (i % period) as f64)
                    .collect::<Vec<_>>();
                let si = seasonal.iter().map(|s| s + 0.2).collect::<Vec<_>>();
                let tail = vec![2.; period];
                let context = ShrinkageContext {
                    dates: axis.span(count + forecasts).unwrap(),
                    requested_axis: axis,
                    observed: 0..count,
                    ..context()
                };
                let result = apply(
                    x11::SeasonalShrinkage::Local,
                    &si,
                    &mut seasonal,
                    &tail,
                    context,
                );
                if forecasts < minimum {
                    assert!(matches!(
                        result,
                        Err(Error::Numerical(NumericalFailure(NumericalDetail::X11Shrinkage {
                            reason: Reason::InsufficientAcceptedForecasts {
                                actual,
                                minimum: expected
                            },
                            ..
                        }))) if actual == forecasts && expected == minimum
                    ));
                } else {
                    result.unwrap();
                }
            }
        }
    }

    #[test]
    fn local_with_a_tiny_variance_returns_the_factors() {
        // Each factor's likelihood is then one at itself and zero elsewhere.
        let mut seasonal = vec![1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 9.];
        let original = seasonal.clone();
        let si = seasonal.iter().map(|s| s + 0.001).collect::<Vec<_>>();
        apply(
            x11::SeasonalShrinkage::Local,
            &si,
            &mut seasonal,
            &[100.; 4],
            context(),
        )
        .unwrap();
        assert_eq!(seasonal, original);
    }

    #[test]
    fn zero_and_nonfinite_variances_are_errors() {
        let mut zero = vec![0.; 12];
        assert!(matches!(
            apply(
                x11::SeasonalShrinkage::Global,
                &[0.; 12],
                &mut zero,
                &[0.; 4],
                context()
            ),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Shrinkage {
                    stage: Stage::GlobalWeight,
                    reason: Reason::ZeroDenominator,
                    ..
                }
            )))
        ));
        let mut local = vec![1.; 13];
        assert!(matches!(
            apply(
                x11::SeasonalShrinkage::Local,
                &[1.; 13],
                &mut local,
                &[1.; 4],
                context()
            ),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Shrinkage {
                    stage: Stage::Variance,
                    reason: Reason::NonPositiveLocalVariance,
                    ..
                }
            )))
        ));
        assert!(matches!(
            apply(
                x11::SeasonalShrinkage::Local,
                &[f64::MAX; 13],
                &mut local,
                &[1.; 4],
                context()
            ),
            Err(Error::Numerical(NumericalFailure(
                NumericalDetail::X11Shrinkage {
                    stage: Stage::Variance,
                    reason: Reason::NonFiniteArithmetic,
                    ..
                }
            )))
        ));
    }

    #[test]
    fn global_moves_additive_factors_toward_one() {
        let mut seasonal = vec![2.; 12];
        apply(
            x11::SeasonalShrinkage::Global,
            &[3.; 12],
            &mut seasonal,
            &[2.; 4],
            context(),
        )
        .unwrap();
        // The slot variances are 3/2 and the filter weight 1/3, so the
        // variance is 1/2. Each year's sum of squares is 16, so its weight is
        // (1/3)(1/2)/(16/3) = 1/32.
        assert_eq!(seasonal, vec![1.96875; 12]);
        // SI equal to the factors gives a zero variance and no shrinkage.
        let mut unchanged = vec![2.; 12];
        apply(
            x11::SeasonalShrinkage::Global,
            &[2.; 12],
            &mut unchanged,
            &[2.; 4],
            context(),
        )
        .unwrap();
        assert_eq!(unchanged, vec![2.; 12]);
    }

    #[test]
    fn stable_filter_is_unsupported() {
        let mut context = context();
        context.last_filter = ResolvedSeasonalFilter::Stable;
        let error = apply(
            x11::SeasonalShrinkage::Global,
            &[3.; 12],
            &mut [2.; 12],
            &[2.; 4],
            context,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Numerical(NumericalFailure(NumericalDetail::X11Shrinkage {
                policy: x11::SeasonalShrinkage::Global,
                stage: Stage::FilterWeight,
                reason: Reason::UnsupportedFilter {
                    filter: crate::x11::SeasonalFilter::Stable
                },
                ..
            }))
        ));
    }
}
