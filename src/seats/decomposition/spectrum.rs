//! A component's spectrum and the searches for its minimum (`ansub2.f`):
//! FUNC0 and Fbis evaluate the ratio of two harmonic polynomials, MINIM and
//! MINIMbis search one interval from one start, globalMinim takes the lowest
//! of several starts, and minimGrid scans a grid of angles.

use crate::Frequency;
use crate::numerics::BUILD_62_PI;

/// MINIM's first step `d`, 0.01.
const SEARCH_STEP: f64 = f64::from_bits(0x3f84_7ae1_47ae_147b);
/// MINIM's `dstop`, 5e-6: the search stops once two bracket points are closer.
const STOP_SPACING: f64 = f64::from_bits(0x3ed4_f8b5_88e3_68f1);
/// Fbis's `epsilon`, two degrees in radians: the trend search of a
/// differenced model evaluates no angle below it.
const TREND_ANGLE_FLOOR: f64 = f64::from_bits(0x3fa1_df46_a252_9d34);
/// minimGrid's `paso`, pi/100000.
const GRID_STEP: f64 = f64::from_bits(0x3f00_7893_0430_48ca);
/// FUNC0's floor on the size of the denominator, 1e-13.
const DENOMINATOR_FLOOR: f64 = f64::from_bits(0x3d3c_25c2_6849_7682);
/// MINIM gives up once two bracket points are closer than this, 1e-12.
const SPACING_FLOOR: f64 = f64::from_bits(0x3d71_9799_812d_ea11);
/// MINIM moves a trial closer than this, 1e-4, to the upper bound onto it.
const UPPER_SNAP: f64 = f64::from_bits(0x3f1a_36e2_eb1c_432d);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpectrumComponent {
    Trend,
    Seasonal,
    Cycle,
}

/// The ratio of two harmonic polynomials, as a function of the angle.
pub(super) struct Spectrum<'a> {
    component: SpectrumComponent,
    numerator: &'a [f64],
    denominator: &'a [f64],
    has_differencing: bool,
}

impl<'a> Spectrum<'a> {
    pub(super) fn new(
        component: SpectrumComponent,
        numerator: &'a [f64],
        denominator: &'a [f64],
    ) -> Self {
        // FUNC0 holds 32 cosines.
        debug_assert!((1..=32).contains(&numerator.len()));
        debug_assert!((1..=32).contains(&denominator.len()));
        Self {
            component,
            numerator,
            denominator,
            has_differencing: true,
        }
    }

    /// Says whether the model has any differencing (D+BD > 0), which decides
    /// whether the trend search avoids the zero frequency.
    pub(super) fn with_differencing(mut self, has_differencing: bool) -> Self {
        self.has_differencing = has_differencing;
        self
    }

    /// FUNC0: the spectrum at angle `x`.
    pub(super) fn evaluate(&self, x: f64) -> f64 {
        let mut cosines = [0.0; 32];
        cosines[0] = 1.0;
        let mut angle = x;
        for cosine in &mut cosines[1..self.numerator.len().max(self.denominator.len())] {
            *cosine = angle.cos();
            angle += x;
        }
        let mut numerator = 0.0;
        for (&coefficient, &cosine) in self.numerator.iter().zip(&cosines) {
            numerator = coefficient.mul_add(cosine, numerator);
        }
        let mut denominator = 0.0;
        for (&coefficient, &cosine) in self.denominator.iter().zip(&cosines) {
            denominator = coefficient.mul_add(cosine, denominator);
        }
        numerator / clamp_denominator(denominator)
    }

    /// The angle Fbis evaluates for `x`. It compares the raw angle, not its
    /// size, and floors it only for the trend of a differenced model.
    fn evaluation_angle(&self, x: f64) -> f64 {
        if self.component == SpectrumComponent::Trend
            && self.has_differencing
            && x < TREND_ANGLE_FLOOR
        {
            TREND_ANGLE_FLOOR
        } else {
            x
        }
    }

    /// Fbis: the spectrum at the angle the searches use for `x`.
    fn evaluate_for_search(&self, x: f64) -> f64 {
        self.evaluate(self.evaluation_angle(x))
    }
}

fn clamp_denominator(denominator: f64) -> f64 {
    if denominator.abs() < DENOMINATOR_FLOOR {
        DENOMINATOR_FLOOR.copysign(denominator)
    } else {
        denominator
    }
}

/// globalMinim: the lowest minimum MINIMbis finds from 13 starts across
/// `[lower, upper]`. A start's result counts whether or not it converged.
pub(super) fn global_minimum(spectrum: &Spectrum<'_>, lower: f64, upper: f64) -> f64 {
    debug_assert_ne!(spectrum.component, SpectrumComponent::Seasonal);
    debug_assert!(lower.is_finite() && upper.is_finite() && lower < upper);
    let step = (upper - lower) / 12.0;
    let mut start = lower;
    // globalMinim starts from 10.0d20, above any spectrum value.
    let mut minimum = f64::from_bits(0x444b_1ae4_d6e2_ef50);
    while start <= upper {
        let trial = local_minimum(spectrum, start, lower, upper);
        if trial < minimum {
            minimum = trial;
        }
        start += step;
    }
    minimum
}

fn scan_grid(spectrum: &Spectrum<'_>, minimum: &mut f64, mut x: f64, upper: f64) {
    while x <= upper {
        let value = spectrum.evaluate(x);
        if value < *minimum {
            *minimum = value;
        }
        x += GRID_STEP;
    }
}

/// minimGrid: the lowest value on a grid of step pi/100000. The trend grid
/// starts two degrees above zero, and the seasonal grid leaves out two
/// degrees on each side of every seasonal frequency.
pub(super) fn grid_minimum(spectrum: &Spectrum<'_>, frequency: Frequency) -> f64 {
    // minimGrid calls FUNC0, not Fbis. Each bound and angle is a separate
    // operation, and gfortran does not fuse the two-epsilon sums.
    let two_pi = 2.0 * BUILD_62_PI;
    let epsilon = (two_pi * 2.0) / 360.0;
    let location = match spectrum.component {
        SpectrumComponent::Seasonal => 0.0,
        SpectrumComponent::Trend => 0.0 + epsilon,
        SpectrumComponent::Cycle => 0.0,
    };
    let mut minimum = spectrum.evaluate(location);
    match spectrum.component {
        SpectrumComponent::Seasonal => {
            let period = usize::from(frequency.periods_per_year());
            let interval = two_pi / period as f64;
            let mut upper = interval - epsilon;
            scan_grid(spectrum, &mut minimum, GRID_STEP, upper);
            for _ in 1..period / 2 {
                let lower = upper + (epsilon + epsilon);
                upper = (lower + interval) - (epsilon + epsilon);
                scan_grid(spectrum, &mut minimum, lower, upper);
            }
        }
        SpectrumComponent::Trend => {
            let lower = location + GRID_STEP;
            scan_grid(spectrum, &mut minimum, lower, BUILD_62_PI);
        }
        SpectrumComponent::Cycle => scan_grid(spectrum, &mut minimum, 0.0, BUILD_62_PI),
    }
    minimum
}

/// MINIM (seasonal) and MINIMbis (trend and cycle): the spectrum at the last
/// trial point of a three-point search of `[lower, upper]` from `start`.
///
/// The two routines differ only in their limit on trial evaluations, 50 and
/// 500; both inner halving loops stop after 50. A search that runs out of
/// trials has already halved its trial point without evaluating it, and
/// returns the value of the trial before.
pub(super) fn local_minimum(spectrum: &Spectrum<'_>, start: f64, lower: f64, upper: f64) -> f64 {
    let outer_limit = match spectrum.component {
        SpectrumComponent::Seasonal => 50,
        SpectrumComponent::Trend => 500,
        SpectrumComponent::Cycle => 500,
    };
    let d = SEARCH_STEP;
    let long_step = 10.0 * d;
    let mut x2 = start;
    let mut x1 = start - d;
    let mut x3 = start + d;
    debug_assert!((x1 - x2).abs() >= SPACING_FLOOR && (x3 - x2).abs() >= SPACING_FLOOR);
    let mut f1 = spectrum.evaluate_for_search(x1);
    let mut f2 = spectrum.evaluate_for_search(x2);
    let mut f3 = spectrum.evaluate_for_search(x3);
    // The first pass always evaluates a trial: x1 and x3 start `d` from x2,
    // far above SPACING_FLOOR.
    let mut last_f4 = f64::NAN;
    let mut count = 0;
    'search: loop {
        if count > outer_limit || (x1 - x2).abs() < SPACING_FLOOR || (x3 - x2).abs() < SPACING_FLOOR
        {
            break;
        }
        let h1 = (f2 - f1) / (x2 - x1);
        let h3 = (f3 - f2) / (x3 - x2);
        if h3 <= h1 {
            if x3 <= upper && !(x1 > lower && f1 < f3) {
                let mut x4 = 25.0_f64.mul_add(d, x3);
                if x4 > upper {
                    x4 = 0.5 * (x3 + upper);
                }
                let f4 = loop {
                    let f4 = spectrum.evaluate_for_search(x4);
                    last_f4 = f4;
                    count += 1;
                    if f4 <= f2 {
                        break f4;
                    }
                    x4 = 0.5 * (x4 + x3);
                    if count > 50 {
                        break 'search;
                    }
                };
                x1 = x2;
                f1 = f2;
                x2 = x3;
                f2 = f3;
                x3 = x4;
                f3 = f4;
                continue;
            }
            let mut x4 = (-25.0_f64).mul_add(d, x1);
            if x4 < lower {
                x4 = 0.5 * (x1 + lower);
            }
            let f4 = loop {
                let f4 = spectrum.evaluate_for_search(x4);
                last_f4 = f4;
                count += 1;
                if f4 <= f2 {
                    break f4;
                }
                x4 = 0.5 * (x4 + x1);
                if count > 50 {
                    break 'search;
                }
            };
            x3 = x2;
            f3 = f2;
            x2 = x1;
            f2 = f1;
            x1 = x4;
            f1 = f4;
        } else {
            // The h1 product rounds on its own, the other product is fused
            // into the sum, the division is separate, and the half of the
            // quotient is fused into x2. The 10*d comparisons are not fused.
            let product = h1 * (x3 - x2);
            let quotient = h3.mul_add(x2 - x1, product) / (h3 - h1);
            let mut x4 = (-0.5_f64).mul_add(quotient, x2);
            if x4 < x1 - long_step {
                if x4 < lower {
                    x4 = 0.5 * (x1 + lower);
                }
                let f4 = spectrum.evaluate_for_search(x4);
                last_f4 = f4;
                count += 1;
                x3 = x1;
                f3 = f1;
                if f4 >= f1 {
                    x1 = x4;
                    f1 = f4;
                } else {
                    x2 = x4;
                    f2 = f4;
                    x1 = x2 - long_step;
                    f1 = spectrum.evaluate_for_search(x1);
                    continue;
                }
            } else if x4 > x3 + long_step {
                if x4 > upper {
                    x4 = 0.5 * (x3 + upper);
                }
                let f4 = spectrum.evaluate_for_search(x4);
                last_f4 = f4;
                count += 1;
                x1 = x3;
                f1 = f3;
                if f4 >= f3 {
                    x3 = x4;
                    f3 = f4;
                } else {
                    x2 = x4;
                    f2 = f4;
                    x3 = x2 + long_step;
                    f3 = spectrum.evaluate_for_search(x3);
                    continue;
                }
            } else {
                if (x4 - upper).abs() < UPPER_SNAP {
                    x4 = upper;
                }
                let f4 = spectrum.evaluate_for_search(x4);
                last_f4 = f4;
                count += 1;
                if (x4 - x2).abs() < STOP_SPACING
                    || (x4 - x1).abs() < STOP_SPACING
                    || (x4 - x3).abs() < STOP_SPACING
                {
                    break;
                }
                if x4 >= x2 {
                    x1 = x2;
                    f1 = f2;
                    if x4 > x3 {
                        x2 = x3;
                        f2 = f3;
                        x3 = x4;
                        f3 = f4;
                        continue;
                    }
                } else {
                    x3 = x2;
                    f3 = f2;
                    if x4 < x1 {
                        x2 = x1;
                        f2 = f1;
                        x1 = x4;
                        f1 = f4;
                        continue;
                    }
                }
                x2 = x4;
                f2 = f4;
                continue;
            }
            x2 = 0.5 * (x1 + x3);
            f2 = spectrum.evaluate_for_search(x2);
        }
    }
    last_f4
}

#[cfg(test)]
mod tests;
