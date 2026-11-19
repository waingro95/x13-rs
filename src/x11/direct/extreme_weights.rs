//! `xtrm.f` extreme weights and the replacement of extreme values.

use std::ops::Range;

use super::extreme_scale::{ExtremeWeightPolicy, OperativeScale, RobustContext, ScaleSample};
use crate::error::X11ExtremeWeightingPass;
use crate::numerics::build_62_equal;
use crate::{Error, x11};

/// The two `xtrm.f` weighting passes, in order.
const WEIGHTING_PASSES: [X11ExtremeWeightingPass; 2] = [
    X11ExtremeWeightingPass::Initial,
    X11ExtremeWeightingPass::Reweighted,
];

/// `sdxtrm.f`: the reweighting pass leaves out values the initial pass
/// weighted to zero.
pub(super) fn in_scale_sample(pass: X11ExtremeWeightingPass, weight: f64) -> bool {
    !(pass == X11ExtremeWeightingPass::Reweighted && build_62_equal(weight, 0.0))
}

/// The spans of one extreme weighting: `application` is weighted, and
/// `estimation`, inside it, supplies the scale samples.
pub(super) struct ExtremeSpans {
    pub(super) application: Range<usize>,
    pub(super) estimation: Range<usize>,
}

impl ExtremeSpans {
    pub(super) fn new(application: Range<usize>, estimation: Range<usize>) -> Self {
        debug_assert!(!application.is_empty() && !estimation.is_empty());
        debug_assert!(application.start <= estimation.start && estimation.end <= application.end);
        Self {
            application,
            estimation,
        }
    }

    /// The spans of a section weighted over `application`. Its scales come
    /// from the same span, or from `observed` less `trim` at each end when
    /// the extensions are left out.
    pub(super) fn for_section(
        application: Range<usize>,
        observed: Range<usize>,
        trim: usize,
        policy: crate::x11::ExtremeEstimationSpan,
    ) -> Self {
        let estimation = match policy {
            crate::x11::ExtremeEstimationSpan::WithExtensions => application.clone(),
            crate::x11::ExtremeEstimationSpan::ObservedOnly => {
                observed.start + trim..observed.end - trim
            }
        };
        Self::new(application, estimation)
    }
}

/// The scale of one sample: the robust estimate when one is requested,
/// otherwise the standard deviation.
fn sample_scale(
    robust: Option<RobustContext>,
    irregular: &[f64],
    weights: &[f64],
    indices: impl IntoIterator<Item = usize>,
    sample: ScaleSample,
) -> Result<f64, Error> {
    match robust {
        Some(context) => context.scale(irregular, weights, indices, sample),
        None => Ok(irregular_standard_deviation(
            irregular,
            weights,
            indices,
            sample.pass,
            sample.center,
        )),
    }
}

/// Clear the reweighting pass's `-1.0` marks, which `xtrm.f` does on the
/// estimation span only. Marks on the extensions stay for `replac.f` and the
/// extreme component.
fn clear_rejection_marks(weights: &mut [f64]) {
    for weight in weights {
        if *weight + 1.0 <= 0.0 {
            *weight = 0.0;
        }
    }
}

/// `xtrm.f`: the extreme weights of `irregular` under `policy`'s scale, after
/// both weighting passes.
pub(super) fn calendar_extreme_weights(
    irregular: &[f64],
    spans: ExtremeSpans,
    period: usize,
    start_subperiod: usize,
    mode: x11::Mode,
    limits: x11::SigmaLimits,
    policy: impl Into<ExtremeWeightPolicy>,
) -> Result<Vec<f64>, Error> {
    use crate::{x11::SigmaGroup as Group, x11::SigmaGroups as Groups};
    use OperativeScale as Scale;
    let ExtremeWeightPolicy { scale, robust } = policy.into();
    if let Scale::RollingFiveYear = scale {
        return rolling_extreme_weights(
            irregular,
            spans,
            period,
            start_subperiod,
            mode,
            limits,
            robust,
        );
    }
    let ExtremeSpans {
        application,
        estimation,
    } = spans;
    let offset = start_subperiod - 1;
    let selected = match scale {
        Scale::Grouped(Groups::Monthly(groups)) => groups.map(|group| group == Group::Selected),
        Scale::Grouped(Groups::Quarterly(groups)) => {
            let mut selected = [false; 12];
            selected[..4].copy_from_slice(&groups.map(|group| group == Group::Selected));
            selected
        }
        _ => [false; 12],
    };
    let group_filter = selected[..period].iter().any(|&selected| selected);
    let mut weights = vec![1.0; irregular.len()];
    for pass in WEIGHTING_PASSES {
        if matches!(scale, Scale::Grouped(_)) {
            // `xtrm.f` computes both groups' scales before it applies either's
            // weights. `sdxtrm.f` takes every value when no period is selected.
            let mut deviations = [0.0; 2];
            for group in [true, false] {
                let indices = estimation
                    .clone()
                    .filter(|&index| !group_filter || selected[(index + offset) % period] == group);
                deviations[usize::from(!group)] = sample_scale(
                    robust,
                    irregular,
                    &weights,
                    indices,
                    ScaleSample {
                        first: estimation.start,
                        last: estimation.end - 1,
                        stride: 1,
                        pass,
                        center: mode.center(),
                        group: Some(if group {
                            Group::Selected
                        } else {
                            Group::Complement
                        }),
                    },
                )?;
            }
            for index in application.clone() {
                let deviation = deviations[usize::from(!selected[(index + offset) % period])];
                if deviation > 0.0 {
                    weights[index] = extreme_weight(
                        irregular[index],
                        deviation,
                        pass,
                        weights[index],
                        mode.center(),
                        limits,
                    );
                }
            }
        } else {
            // `xtrm.f` computes each calendar period's scale and applies its
            // weights before the next period's. A period that starts before
            // the estimation span samples from that span's first value, which
            // is another period's, so on the reweighting pass its sample reads
            // weights this pass already changed.
            for first in application.start..application.start + period {
                let scale_first = if first < estimation.start {
                    estimation.start
                } else {
                    first
                };
                let scale_last = ((estimation.end - 1 - first) / period) * period + first;
                let deviation = sample_scale(
                    robust,
                    irregular,
                    &weights,
                    (scale_first..scale_last + 1).step_by(period),
                    ScaleSample {
                        first: scale_first,
                        last: scale_last,
                        stride: period,
                        pass,
                        center: mode.center(),
                        group: None,
                    },
                )?;
                if !build_62_equal(deviation, 0.0) {
                    for index in (first..application.end).step_by(period) {
                        weights[index] = extreme_weight(
                            irregular[index],
                            deviation,
                            pass,
                            weights[index],
                            mode.center(),
                            limits,
                        );
                    }
                }
            }
        }
    }
    clear_rejection_marks(&mut weights[estimation]);
    Ok(weights)
}

/// `xtrm.f` under the rolling five-year scale: each year is weighted by the
/// scale of the five years around it.
pub(super) fn rolling_extreme_weights(
    irregular: &[f64],
    spans: ExtremeSpans,
    period: usize,
    start_subperiod: usize,
    mode: x11::Mode,
    limits: x11::SigmaLimits,
    robust: Option<RobustContext>,
) -> Result<Vec<f64>, Error> {
    let ExtremeSpans {
        application,
        estimation,
    } = spans;
    debug_assert!(application.end <= irregular.len());
    debug_assert!((1..=period).contains(&start_subperiod));
    // Calendar positions count from period 1 of the first year, so each year
    // starts at a multiple of `period`.
    let calendar_offset = start_subperiod - 1;
    let to_calendar =
        |span: &Range<usize>| span.start + calendar_offset..span.end + calendar_offset;
    let to_axis = |span: Range<usize>| span.start - calendar_offset..span.end - calendar_offset;

    let mut weights = vec![1.0; irregular.len()];
    for pass in WEIGHTING_PASSES {
        for window in rolling_windows(to_calendar(&application), to_calendar(&estimation), period) {
            let sample = to_axis(window.sample);
            let standard_deviation = sample_scale(
                robust,
                irregular,
                &weights,
                sample.clone(),
                ScaleSample {
                    first: sample.start,
                    last: sample.end - 1,
                    stride: 1,
                    pass,
                    center: mode.center(),
                    group: None,
                },
            )?;
            if !build_62_equal(standard_deviation, 0.0) {
                let weighted = to_axis(window.weighted);
                for (weight, &value) in weights[weighted.clone()]
                    .iter_mut()
                    .zip(&irregular[weighted])
                {
                    *weight = extreme_weight(
                        value,
                        standard_deviation,
                        pass,
                        *weight,
                        mode.center(),
                        limits,
                    );
                }
            }
        }
    }
    clear_rejection_marks(&mut weights[estimation]);
    Ok(weights)
}

/// One `xtrm.f` rolling window: the positions it weights, and the positions
/// whose irregular values estimate its scale.
pub(super) struct RollingWindow {
    pub(super) weighted: Range<usize>,
    pub(super) sample: Range<usize>,
}

/// `xtrm.f`'s rolling windows over calendar positions (each year starts at a
/// multiple of `period`), in order.
///
/// Windows start on year boundaries, one year apart, from the first year's
/// start to the last window. A middle window weights its third year and takes
/// its scale over all five. The first window instead weights from the start of
/// `application` through its own third year, sampling from the first
/// estimation value through five years past the first full year. The last
/// weights from its third year through the end of `application`, sampling
/// from the last full five-year window through the last estimation value.
/// `ExtremeSpans::new` keeps `application` around `estimation`, so no weight
/// range inverts. Without five full years past the first full year,
/// one window weights all of `application` from all of `estimation`.
///
/// `xtrm.f` takes the window geometry from the estimation span
/// (`Kfdax`/`Kldax`) and the first and last weight ranges from the application
/// span (`Kfda`/`Klda`).
pub(super) fn rolling_windows(
    application: Range<usize>,
    estimation: Range<usize>,
    period: usize,
) -> impl Iterator<Item = RollingWindow> {
    let five_years = 5 * period;
    let first_year = estimation.start / period * period;
    let first_full_year = estimation.start.div_ceil(period) * period;
    // The last full window ends with the last complete year; the last window
    // ends with the year that holds the last estimation value.
    let last_full_window = (estimation.end / period * period).checked_sub(five_years);
    let (last_window, last_sample) = match last_full_window {
        Some(window) if window >= first_full_year => (
            estimation.end.div_ceil(period) * period - five_years,
            window,
        ),
        _ => (first_year, first_year),
    };
    (first_year..=last_window)
        .step_by(period)
        .map(move |start| {
            if last_window <= first_year {
                RollingWindow {
                    weighted: application.clone(),
                    sample: estimation.clone(),
                }
            } else if start == first_year {
                RollingWindow {
                    weighted: application.start..first_year + 3 * period,
                    sample: estimation.start..first_full_year + five_years,
                }
            } else if start < last_window {
                RollingWindow {
                    weighted: start + 2 * period..start + 3 * period,
                    sample: start..start + five_years,
                }
            } else {
                RollingWindow {
                    weighted: last_window + 2 * period..application.end,
                    sample: last_sample..estimation.end,
                }
            }
        })
}

/// `sdxtrm.f`: the root mean square deviation from `center` of the irregular
/// values at `span` that `pass` keeps.
pub(super) fn irregular_standard_deviation(
    irregular: &[f64],
    weights: &[f64],
    span: impl IntoIterator<Item = usize>,
    pass: X11ExtremeWeightingPass,
    center: f64,
) -> f64 {
    let mut sum_squares = 0.0;
    let mut count = 0.0;
    for index in span {
        if in_scale_sample(pass, weights[index]) {
            count += 1.0;
            let deviation = irregular[index] - center;
            // gfortran fuses each square into the running sum in `sdxtrm.f`;
            // the subtraction is separate.
            sum_squares = deviation.mul_add(deviation, sum_squares);
        }
    }
    (sum_squares / count).sqrt()
}

/// `wtxtrm.f`: the initial pass only zeroes values beyond the upper limit. The
/// reweighting pass grades values between the limits and marks newly rejected
/// values `-1.0`, which `clear_rejection_marks` resets on the estimation span.
pub(super) fn extreme_weight(
    value: f64,
    standard_deviation: f64,
    pass: X11ExtremeWeightingPass,
    previous: f64,
    center: f64,
    limits: x11::SigmaLimits,
) -> f64 {
    let reweighting = pass == X11ExtremeWeightingPass::Reweighted;
    let deviation = (value - center).abs() / standard_deviation;
    if deviation <= limits.upper() {
        if deviation > limits.lower() && reweighting {
            (limits.upper() - deviation) / (limits.upper() - limits.lower())
        } else {
            previous
        }
    } else if !reweighting {
        0.0
    } else if previous > 0.0 {
        -1.0
    } else {
        previous
    }
}

/// `replac.f`: replace each value in `span` whose weight is not 1 with the
/// weighted mean of itself and four full-weight values of its calendar
/// period: the nearest on either side, or the first or last four at the ends
/// of the span. With fewer than four it takes the period's mean.
pub(super) fn replace_extremes(
    values: &mut [f64],
    weights: &[f64],
    span: Range<usize>,
    period: usize,
) {
    for seasonal_offset in 0..period {
        let indices = (span.start + seasonal_offset..span.end)
            .step_by(period)
            .collect::<Vec<_>>();
        let average =
            indices.iter().map(|&index| values[index]).sum::<f64>() / indices.len() as f64;
        for position in 0..indices.len() {
            let index = indices[position];
            if build_62_equal(weights[index], 1.0) {
                continue;
            }
            let mut neighbors = Vec::with_capacity(4);
            if position <= 1 {
                neighbors.extend(
                    indices
                        .iter()
                        .copied()
                        .filter(|&candidate| weights[candidate] >= 1.0)
                        .take(4),
                );
            } else if position + 2 >= indices.len() {
                neighbors.extend(
                    indices
                        .iter()
                        .rev()
                        .copied()
                        .filter(|&candidate| weights[candidate] >= 1.0)
                        .take(4),
                );
            } else {
                let right = indices[position + 1..]
                    .iter()
                    .copied()
                    .filter(|&candidate| build_62_equal(weights[candidate], 1.0))
                    .collect::<Vec<_>>();
                let left = indices[..position]
                    .iter()
                    .rev()
                    .copied()
                    .filter(|&candidate| build_62_equal(weights[candidate], 1.0))
                    .collect::<Vec<_>>();
                neighbors.extend(right.iter().take(2).copied());
                neighbors.extend(left.iter().take(4 - neighbors.len()).copied());
                neighbors.extend(right.iter().skip(2).take(4 - neighbors.len()).copied());
            }
            if neighbors.len() == 4 {
                let total = neighbors
                    .iter()
                    .fold(weights[index] * values[index], |total, &neighbor| {
                        total + values[neighbor]
                    });
                values[index] = total / (4.0 + weights[index]);
            } else {
                values[index] = average;
            }
        }
    }
}

/// `x11pt2.f`'s `Stex`: each irregular's extreme component under its weight.
pub(super) fn extreme_components<'a>(
    mode: x11::Mode,
    irregular: &'a [f64],
    weights: &'a [f64],
) -> impl Iterator<Item = f64> + 'a {
    irregular
        .iter()
        .zip(weights)
        .map(move |(&irregular, &weight)| mode.extreme_component(irregular, weight))
}
