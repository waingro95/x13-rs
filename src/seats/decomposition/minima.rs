//! SPECTRU's minima (`spectrum.f`): the minimum of the trend, cycle and
//! seasonal spectra over their frequency ranges, and the irregular variance
//! that is left once each component gives up its minimum.

use super::harmonic::Harmonics;
use super::rational::{RationalPrefix, rational_prefix};
use super::spectrum::{Spectrum, SpectrumComponent, global_minimum, grid_minimum, local_minimum};
use crate::Frequency;
use crate::numerics::BUILD_62_PI;
use crate::seats::airline::model::ResolvedAirline;
use crate::seats::failure::Error;

/// The cycle's numerator and the minimum of its spectrum, SPECTRU's `enoc`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CycleMinimum {
    pub(super) numerator: Vec<f64>,
    pub(crate) minimum: f64,
}

/// The component numerators with the minimum of each component's spectrum.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SpectralMinima {
    pub(super) harmonics: Harmonics,
    pub(super) trend_numerator: Vec<f64>,
    pub(super) seasonal_numerator: Vec<f64>,
    /// The minimum of the trend spectrum.
    pub(crate) enot: f64,
    /// The minimum of the seasonal spectrum.
    pub(crate) estar: f64,
    pub(crate) cycle: Option<CycleMinimum>,
    /// SPECTRU's `qt1`, the irregular variance before any model change. A
    /// negative value means the model has no admissible decomposition.
    pub(crate) irregular_candidate: f64,
}

/// SPECTRU's `MIN(e1, e2)` of the two half-range minima. gfortran's minimum
/// of a positive and a negative zero is the negative one, in either order.
fn trend_minimum(first: f64, second: f64) -> f64 {
    if first == 0.0 && second == 0.0 {
        f64::from_bits(first.to_bits() | second.to_bits())
    } else if second < first {
        second
    } else {
        first
    }
}

/// The lowest interval minimum, starting from SPECTRU's 10.
fn seasonal_minimum(intervals: &[f64]) -> f64 {
    let mut minimum = 10.0;
    for &interval in intervals {
        if interval < minimum {
            minimum = interval;
        }
    }
    minimum
}

/// SPECTRU's test for the grid search, `numerator(1) - minimum*factor(1) < 0`.
/// gfortran fuses the product into the subtraction; rounding the product
/// first can change the sign.
fn grid_required(first: f64, factor: f64, minimum: f64) -> bool {
    (-factor).mul_add(minimum, first) < 0.0
}

/// The grid search replaces a minimum only with a strictly lower value.
fn adopt_grid(minimum: &mut f64, candidate: f64) {
    if candidate < *minimum {
        *minimum = candidate;
    }
}

/// SPECTRU's `qt(1) + enot + estar + enoc` without a cycle, where enoc is a
/// positive zero.
fn irregular_without_cycle(quotient: f64, enot: f64, estar: f64) -> f64 {
    let first = quotient + enot;
    let second = first + estar;
    second + 0.0
}

/// SPECTRU's minima for the airline model.
pub(crate) fn spectral_minima(model: &ResolvedAirline) -> Result<SpectralMinima, Error> {
    let prefix = rational_prefix(model)?;
    Ok(spectral_minima_prefix(model.frequency(), prefix))
}

/// SPECTRU's minima for a model with differencing.
pub(crate) fn spectral_minima_prefix(
    frequency: Frequency,
    prefix: RationalPrefix,
) -> SpectralMinima {
    regular_ar_spectral_minima(frequency, prefix, true)
}

/// SPECTRU's minima. `has_differencing` says whether D+BD is positive, which
/// keeps the trend search off the zero frequency.
pub(crate) fn regular_ar_spectral_minima(
    frequency: Frequency,
    prefix: RationalPrefix,
    has_differencing: bool,
) -> SpectralMinima {
    let RationalPrefix {
        harmonics,
        quotient,
        trend_numerator,
        seasonal_numerator,
        cycle_numerator,
    } = prefix;
    let enot = if harmonics.trend.len() == 1 {
        // SPECTRU skips the trend search when nchi=1 and keeps enot=0.
        0.0
    } else {
        let trend = Spectrum::new(SpectrumComponent::Trend, &trend_numerator, &harmonics.trend)
            .with_differencing(has_differencing);
        // The upper half of the range comes first.
        let upper = global_minimum(&trend, 0.5 * BUILD_62_PI, BUILD_62_PI);
        let lower = global_minimum(&trend, 0.0, 0.5 * BUILD_62_PI);
        let mut enot = trend_minimum(upper, lower);
        if grid_required(trend_numerator[0], harmonics.trend[0], enot) {
            adopt_grid(&mut enot, grid_minimum(&trend, frequency));
        }
        enot
    };

    // SPECTRU searches the cycle before the seasonal component.
    let cycle = cycle_numerator.map(|numerator| {
        let cycle = Spectrum::new(SpectrumComponent::Cycle, &numerator, &harmonics.cycle);
        let upper = global_minimum(&cycle, 0.5 * BUILD_62_PI, BUILD_62_PI);
        let lower = global_minimum(&cycle, 0.0, 0.5 * BUILD_62_PI);
        let mut minimum = trend_minimum(upper, lower);
        if grid_required(numerator[0], harmonics.cycle[0], minimum) {
            adopt_grid(&mut minimum, grid_minimum(&cycle, frequency));
        }
        CycleMinimum { numerator, minimum }
    });

    let estar = if harmonics.seasonal.len() == 1 {
        // Without a seasonal factor SPECTRU keeps estar=0.
        0.0
    } else {
        let seasonal = Spectrum::new(
            SpectrumComponent::Seasonal,
            &seasonal_numerator,
            &harmonics.seasonal,
        );
        // One search between each pair of neighbouring seasonal frequencies.
        // The first starts at zero, the others at the middle of their range.
        let bins = usize::from(frequency.periods_per_year()) / 2;
        let mut intervals = Vec::with_capacity(bins);
        intervals.push(local_minimum(
            &seasonal,
            0.0,
            0.0,
            BUILD_62_PI / bins as f64,
        ));
        for i in 2..=bins {
            let start = ((i as f64 - 0.5) * BUILD_62_PI) / bins as f64;
            let lower = ((i - 1) as f64 * BUILD_62_PI) / bins as f64;
            let upper = (i as f64 * BUILD_62_PI) / bins as f64;
            intervals.push(local_minimum(&seasonal, start, lower, upper));
        }
        let mut estar = seasonal_minimum(&intervals);
        if grid_required(seasonal_numerator[0], harmonics.seasonal[0], estar) {
            adopt_grid(&mut estar, grid_minimum(&seasonal, frequency));
        }
        estar
    };

    let irregular_candidate = if harmonics.numerator.len() > harmonics.denominator.len() {
        // A quotient of positive degree is already in the transitory
        // spectrum, so SPECTRU leaves qt(1) out.
        (enot + estar) + cycle.as_ref().map_or(0.0, |cycle| cycle.minimum)
    } else if let Some(cycle) = &cycle {
        ((quotient[0] + enot) + estar) + cycle.minimum
    } else {
        irregular_without_cycle(quotient[0], enot, estar)
    };
    SpectralMinima {
        harmonics,
        trend_numerator,
        seasonal_numerator,
        enot,
        estar,
        cycle,
        irregular_candidate,
    }
}

#[cfg(test)]
mod tests;
