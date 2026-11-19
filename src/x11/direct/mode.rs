//! How each decomposition mode combines and separates components.

use std::ops::Range;

use crate::error::X11PseudoAdditiveStage as PseudoStage;
use crate::time::AxisSpan;
use crate::x11::pseudo_additive as pseudo;
use crate::{Error, x11};

/// `divsub.f`: `left` with the component `right` taken out.
pub(super) fn remove(mode: x11::Mode, left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| mode.remove(*left, *right))
        .collect()
}

/// `remove` over `span`, with the mode's center elsewhere. Pseudo-additive
/// divisions are checked and report `stage`.
pub(super) fn remove_checked(
    mode: x11::Mode,
    left: &[f64],
    right: &[f64],
    span: Range<usize>,
    dates: AxisSpan,
    stage: PseudoStage,
) -> Result<Vec<f64>, Error> {
    if mode != x11::Mode::PseudoAdditive {
        return Ok(remove_on_span(mode, left, right, span));
    }
    let mut result = vec![1.0; left.len()];
    for i in span {
        result[i] = pseudo::divide(left[i], right[i], dates, i, stage)?;
    }
    Ok(result)
}

/// `x11pt2.f`: the modified series of a trend-only pseudo-additive run. The
/// quotient is rounded before the multiplication.
pub(super) fn pseudo_trend_modified(
    trend: &[f64],
    irregular: &[f64],
    extreme: &[f64],
    dates: AxisSpan,
) -> Result<Vec<f64>, Error> {
    (0..trend.len())
        .map(|i| {
            let stage = PseudoStage::ExtremeReconstruction;
            let ratio = pseudo::divide(irregular[i], extreme[i], dates, i, stage)?;
            pseudo::checked(trend[i] * ratio, dates, i, stage)
        })
        .collect()
}

/// `x11pt2.f`: the modified series of a pseudo-additive run.
pub(super) fn pseudo_modified(
    trend: &[f64],
    seasonal: &[f64],
    irregular: &[f64],
    extreme: &[f64],
    dates: AxisSpan,
) -> Result<Vec<f64>, Error> {
    (0..trend.len())
        .map(|i| {
            let stage = PseudoStage::ExtremeReconstruction;
            pseudo::divide(irregular[i], extreme[i], dates, i, stage)?;
            pseudo::checked(
                pseudo::modified(trend[i], seasonal[i], irregular[i], extreme[i]),
                dates,
                i,
                stage,
            )
        })
        .collect()
}

/// `remove` over `span`, with the mode's center elsewhere.
pub(super) fn remove_on_span(
    mode: x11::Mode,
    left: &[f64],
    right: &[f64],
    span: Range<usize>,
) -> Vec<f64> {
    let mut result = vec![mode.center(); left.len()];
    for index in span {
        result[index] = mode.remove(left[index], right[index]);
    }
    result
}

impl x11::Mode {
    /// The value of a component that has no effect: 0 when components add, 1
    /// when they multiply.
    pub(super) const fn center(self) -> f64 {
        match self {
            Self::Additive | Self::LogAdditive => 0.0,
            Self::Multiplicative | Self::PseudoAdditive => 1.0,
        }
    }

    /// `divsub.f`: take a component out.
    pub(super) fn remove(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Additive | Self::LogAdditive => left - right,
            Self::Multiplicative | Self::PseudoAdditive => left / right,
        }
    }

    /// `addmul.f`: put back a component that `remove` took out.
    pub(super) fn restore(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Additive | Self::LogAdditive => left + right,
            Self::Multiplicative | Self::PseudoAdditive => left * right,
        }
    }

    /// The absolute change from `current` to `next`, relative to `current`
    /// when components multiply.
    pub(super) fn change(self, current: f64, next: f64) -> f64 {
        match self {
            Self::Additive | Self::LogAdditive => (next - current).abs(),
            Self::Multiplicative | Self::PseudoAdditive => (next - current).abs() / current,
        }
    }

    /// `x11pt2.f`'s `Stex`: the part of `irregular` its extreme weight
    /// removes.
    pub(super) fn extreme_component(self, irregular: f64, weight: f64) -> f64 {
        match self {
            Self::Additive | Self::LogAdditive => irregular * (1.0 - weight),
            // gfortran fuses this multiply-add in `x11pt2.f`; the subtraction
            // and the division are separate.
            Self::Multiplicative | Self::PseudoAdditive => {
                let deviation = irregular - 1.0;
                irregular / weight.mul_add(deviation, 1.0)
            }
        }
    }
}
