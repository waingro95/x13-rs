//! A regular AR(2) or AR(3) in the decomposition: TRANS0's import of its
//! coefficients, RPQ's roots, F1RST's allocation of each root to the trend,
//! seasonal or cycle factor (`sigsub.f`), and SIGEX's polynomials for the
//! allocated factors.

use super::polynomial::{DenominatorFactors, SpectralInputs, convolve};
use crate::Frequency;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::import::{AR_BOUND, trans0_clip};
use crate::seats::roots::polynomial_roots;
use crate::seats::tails::DifferenceOrder;
use crate::seats::trading_day::EffectiveTradingDay;

/// The coefficients of a regular AR(2) or AR(3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum RegularArCoefficients {
    Two([f64; 2]),
    Three([f64; 3]),
}
impl RegularArCoefficients {
    pub(crate) fn as_slice(&self) -> &[f64] {
        match self {
            Self::Two(v) => v,
            Self::Three(v) => v,
        }
    }
    /// The AR degree, as a model order.
    pub(crate) const fn degree(&self) -> u8 {
        match self {
            Self::Two(_) => 2,
            Self::Three(_) => 3,
        }
    }
    pub(crate) fn map(self, f: impl FnMut(f64) -> f64) -> Self {
        match self {
            Self::Two(v) => Self::Two(v.map(f)),
            Self::Three(v) => Self::Three(v.map(f)),
        }
    }
    /// `1 - c1 B - c2 B^2 - ...`.
    pub(crate) fn polynomial(self) -> Vec<f64> {
        std::iter::once(1.0)
            .chain(self.as_slice().iter().map(|v| -v))
            .collect()
    }
}

/// One root of the regular AR polynomial, with RPQ's modulus, angle in
/// degrees and period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ArRoot {
    pub real: f64,
    pub imaginary: f64,
    pub modulus: f64,
    pub angle: f64,
    pub period: f64,
}

/// One of F1RST's six factors of the AR polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GeometryFactor {
    TrendStationary,
    TrendNonstationary,
    SeasonalStationary,
    SeasonalNonstationary,
    CycleStationary,
    CycleNonstationary,
}

const INPUT: &str = "regular AR coefficient";
const IMPORTED: &str = "imported regular AR coefficient";
const SPECTRAL_INPUT: &str = "spectral input coefficient";

/// The six factors of the AR polynomial, differences included.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AllocatedGeometry {
    pub trend_stationary: Vec<f64>,
    pub trend_nonstationary: Vec<f64>,
    pub seasonal_stationary: Vec<f64>,
    pub seasonal_nonstationary: Vec<f64>,
    pub cycle_stationary: Vec<f64>,
    pub cycle_nonstationary: Vec<f64>,
    /// Whether a complex pair in the cycle is near the trading-day frequency.
    pub is_close_to_trading_day: bool,
}

fn finite(value: f64, quantity: &'static str, index: usize) -> Result<f64, Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Failure::Unusable {
            quantity,
            index: Some(index),
            value,
        }
        .at(Stage::CanonicalDecomposition))
    }
}

fn invalid_root_pair(index: usize) -> Error {
    Failure::UnusableRoots {
        step: "root allocation",
        index: Some(index),
    }
    .at(Stage::CanonicalDecomposition)
}

/// The coefficients as SEATS imports them: NMLMDL's signs, then TRANS0 with
/// IPRS=2, UR=1 and XL=.99, and TRANS2 back.
pub(crate) fn import_regular_ar(
    coefficients: RegularArCoefficients,
) -> Result<RegularArCoefficients, Error> {
    // ANALTS negates the transformed coordinates back to this convention.
    // That second sign flip of C is a plain double negation, an exact
    // identity, so C is returned as computed. NMLMDL's `-(0.0 - x)` is not
    // (it maps +0 to -0) and stays.
    let phith = |value: f64, index| finite(-(0.0 - finite(value, INPUT, index)?), IMPORTED, index);
    let snap = |x: f64| trans0_clip(x, AR_BOUND);
    match coefficients {
        RegularArCoefficients::Two(values) => {
            let first = phith(values[0], 0)?;
            let mut second = phith(values[1], 1)?;
            if (1.0 - second).abs() < 1e-9 {
                second = 1.0;
            }
            // TRANS0 quirk: with PHITH(2) reset to UR=1 the quotient divides
            // by exactly zero (and a finite one can overflow). Build 62 keeps
            // the IEEE ±infinity, which the clip maps to ±UR.
            let x1 = snap(first / (1.0 - second));
            let x2 = snap(second);
            // 0/0 leaves a NaN that both clip comparisons pass over. Build 62's
            // SEATS then hangs; this returns an error.
            let c1 = finite(x1 * (1.0 - x2), IMPORTED, 0)?;
            Ok(RegularArCoefficients::Two([c1, x2]))
        }
        RegularArCoefficients::Three(values) => {
            let first = phith(values[0], 0)?;
            let second = phith(values[1], 1)?;
            let mut third = phith(values[2], 2)?;
            if (third - 1.0).abs() < 1e-9 {
                third = 1.0_f64.copysign(third);
            }
            // TRANS0 quirk: each division can produce an IEEE ±infinity, which
            // Build 62 keeps and the clip maps to ±UR: 1 - PHITH(3) after the
            // reset to UR, 1 + PHITH(3) at exactly -1, 1 + X(1) after the reset
            // to -UR, or an overflowing finite quotient.
            let mut x1 = 0.5 * ((first + second) / (1.0 - third) + 1.0);
            let x2 = 1.0 + (first - second) / (1.0 + third);
            if (x1 + 1.0).abs() < 1e-9 {
                x1 = -1.0;
            }
            let x2 = x2 / (1.0 + x1) - 1.0;
            let c = cubic_coefficients([snap(x1), snap(x2), snap(third)]);
            // A 0/0 (or ∞/∞) leaves a NaN that both clip comparisons pass over
            // into C, an error for the same reason.
            for (i, value) in c.into_iter().enumerate() {
                finite(value, IMPORTED, i)?;
            }
            Ok(RegularArCoefficients::Three(c))
        }
    }
}

/// TRANS2 and TRANSC's coefficients of a cubic from its three transformed
/// coordinates. gfortran fuses the products in both sums.
pub(crate) fn cubic_coefficients(x: [f64; 3]) -> [f64; 3] {
    let first = x[0].mul_add(2.0, -1.0);
    let complement = 1.0 - x[2];
    let d = (1.0 + x[0]).mul_add(1.0 + x[1], -1.0) * (1.0 + x[2]);
    [
        0.5 * first.mul_add(complement, d),
        0.5 * first.mul_add(complement, -d),
        x[2],
    ]
}

/// RPQ's roots of the AR polynomial, one per coefficient.
pub(crate) fn regular_ar_roots(coefficients: RegularArCoefficients) -> Result<Vec<ArRoot>, Error> {
    let values = coefficients.as_slice();
    for (index, &value) in values.iter().enumerate() {
        finite(value, INPUT, index)?;
    }
    let output = polynomial_roots(&coefficients.polynomial())?;
    Ok((0..values.len())
        .map(|i| ArRoot {
            real: output.real[i],
            imaginary: output.imaginary[i],
            modulus: output.modulus[i],
            angle: output.argument_degrees[i],
            period: output.period[i],
        })
        .collect())
}

impl GeometryFactor {
    /// The factor's name in a failure.
    fn label(self) -> &'static str {
        match self {
            Self::TrendStationary => "stationary trend",
            Self::TrendNonstationary => "nonstationary trend",
            Self::SeasonalStationary => "stationary seasonal",
            Self::SeasonalNonstationary => "nonstationary seasonal",
            Self::CycleStationary => "stationary cycle",
            Self::CycleNonstationary => "nonstationary cycle",
        }
    }

    /// F1RST's declared size for the factor's coefficient array.
    fn maximum(self) -> usize {
        match self {
            Self::TrendStationary => 5,
            Self::TrendNonstationary => 8,
            Self::SeasonalStationary => 16,
            Self::SeasonalNonstationary => 27,
            Self::CycleStationary => 17,
            Self::CycleNonstationary => 5,
        }
    }
}

impl AllocatedGeometry {
    /// The six factors in SIGEX's storage order.
    fn factors(&self) -> [(GeometryFactor, &[f64]); 6] {
        [
            (GeometryFactor::TrendStationary, &self.trend_stationary),
            (
                GeometryFactor::TrendNonstationary,
                &self.trend_nonstationary,
            ),
            (
                GeometryFactor::SeasonalStationary,
                &self.seasonal_stationary,
            ),
            (
                GeometryFactor::SeasonalNonstationary,
                &self.seasonal_nonstationary,
            ),
            (GeometryFactor::CycleStationary, &self.cycle_stationary),
            (
                GeometryFactor::CycleNonstationary,
                &self.cycle_nonstationary,
            ),
        ]
    }

    fn factor_mut(&mut self, factor: GeometryFactor) -> &mut Vec<f64> {
        match factor {
            GeometryFactor::TrendStationary => &mut self.trend_stationary,
            GeometryFactor::TrendNonstationary => &mut self.trend_nonstationary,
            GeometryFactor::SeasonalStationary => &mut self.seasonal_stationary,
            GeometryFactor::SeasonalNonstationary => &mut self.seasonal_nonstationary,
            GeometryFactor::CycleStationary => &mut self.cycle_stationary,
            GeometryFactor::CycleNonstationary => &mut self.cycle_nonstationary,
        }
    }

    fn multiply(&mut self, factor: GeometryFactor, polynomial: &[f64]) -> Result<(), Error> {
        let values = self.factor_mut(factor);
        let count = values.len() + polynomial.len() - 1;
        check_capacity(factor, count)?;
        *values = convolve(polynomial, values);
        for (index, &value) in values.iter().enumerate() {
            finite(value, "allocated factor coefficient", index)?;
        }
        Ok(())
    }
}

fn check_capacity(factor: GeometryFactor, count: usize) -> Result<(), Error> {
    let maximum = factor.maximum();
    if count > maximum {
        Err(Failure::FactorCapacity {
            factor: factor.label(),
            count,
            maximum,
        }
        .at(Stage::CanonicalDecomposition))
    } else {
        Ok(())
    }
}

fn is_real(root: &ArRoot) -> bool {
    -1e-13 < root.imaginary && root.imaginary < 1e-13
}

fn validate_roots(roots: &[ArRoot]) -> Result<(), Error> {
    if roots.len() > 3 {
        return Err(Failure::FactorCapacity {
            factor: "regular AR",
            count: roots.len(),
            maximum: 3,
        }
        .at(Stage::CanonicalDecomposition));
    }
    for (index, root) in roots.iter().enumerate() {
        for (coordinate, value) in [
            root.real,
            root.imaginary,
            root.modulus,
            root.angle,
            root.period,
        ]
        .into_iter()
        .enumerate()
        {
            finite(value, "regular AR root coordinate", index * 5 + coordinate)?;
        }
        if root.modulus < 0.0 {
            return Err(invalid_root_pair(index));
        }
    }
    let Some(first) = roots.first() else {
        return Ok(());
    };
    if is_real(first) {
        if let Some(index) = roots.iter().position(|root| !is_real(root)) {
            return Err(invalid_root_pair(index));
        }
    } else if roots.len() < 2
        || is_real(&roots[1])
        || first.real != roots[1].real
        || first.imaginary != -roots[1].imaginary
        || first.modulus != roots[1].modulus
        || roots.get(2).is_some_and(|root| !is_real(root))
    {
        return Err(invalid_root_pair(0));
    }
    Ok(())
}

/// SIGEX's factors for a model with no seasonal AR: the differences, then
/// F1RST's allocation of the regular roots.
pub(crate) fn allocate_roots_with_differences(
    frequency: Frequency,
    roots: &[ArRoot],
    differences: DifferenceOrder,
) -> Result<AllocatedGeometry, Error> {
    let period = usize::from(frequency.periods_per_year());
    // Each difference (the seasonal one included) multiplies 1 - B into the
    // trend, in place over the coefficients up to the new degree.
    let d = differences.ordinary() + usize::from(differences.seasonal());
    let mut trend_nonstationary = vec![0.0; d + 1];
    trend_nonstationary[0] = 1.0;
    for degree in 1..=d {
        for k in (1..=degree).rev() {
            trend_nonstationary[k] -= trend_nonstationary[k - 1];
        }
    }
    let allocation = AllocatedGeometry {
        trend_stationary: vec![1.0],
        trend_nonstationary,
        seasonal_stationary: vec![1.0],
        // 1 + B + ... + B^(s-1).
        seasonal_nonstationary: vec![1.0; if differences.seasonal() { period } else { 1 }],
        cycle_stationary: vec![1.0],
        cycle_nonstationary: vec![1.0],
        is_close_to_trading_day: false,
    };
    allocate_regular_roots(frequency, roots, allocation)
}

/// F1RST: multiplies each regular root into the factor it belongs to.
fn allocate_regular_roots(
    frequency: Frequency,
    roots: &[ArRoot],
    mut allocation: AllocatedGeometry,
) -> Result<AllocatedGeometry, Error> {
    validate_roots(roots)?;
    let period = usize::from(frequency.periods_per_year());
    let Some(first) = roots.first() else {
        return Ok(allocation);
    };
    // F1RST fixes RmodS once, before allocating any regular roots.
    let seasonal_cutoff = if allocation.seasonal_nonstationary.len() > 1
        || allocation.seasonal_stationary.len() > 1
    {
        0.5
    } else {
        0.9
    };
    if is_real(first) {
        // Real roots: F1RST tests the unit neighbourhood first.
        for root in roots {
            let real = root.real;
            let factor = if real <= 0.0 {
                if (1.0 + real).abs() < 1e-6 {
                    GeometryFactor::SeasonalNonstationary
                } else if real.abs() >= seasonal_cutoff {
                    GeometryFactor::SeasonalStationary
                } else {
                    GeometryFactor::CycleStationary
                }
            } else if (1.0 - real).abs() < 1e-6 {
                GeometryFactor::TrendNonstationary
            } else if real.abs() >= 0.5 {
                GeometryFactor::TrendStationary
            } else {
                GeometryFactor::CycleStationary
            };
            allocation.multiply(factor, &[1.0, -real])?;
        }
    } else {
        // A complex pair: F1RST reads its first root only. gfortran fuses the
        // squares in the quadratic's constant. The trend and seasonal window
        // tests are strict; the RmodS test is not.
        let polynomial = [
            1.0,
            -2.0 * first.real,
            first
                .real
                .mul_add(first.real, first.imaginary * first.imaginary),
        ];
        let unit = (first.modulus - 1.0).abs() < 1e-6;
        let factor = if first.modulus > 0.5 && first.angle.abs() < 360.0 / (2.0 * period as f64) {
            if unit {
                GeometryFactor::TrendNonstationary
            } else {
                GeometryFactor::TrendStationary
            }
        } else {
            // Within 2 degrees of a seasonal harmonic, at modulus RmodS or more.
            let k = (360 / period) as f64;
            let seasonal = (1..=period / 2).any(|i| {
                let harmonic = k * i as f64;
                first.angle.abs() > harmonic - 2.0
                    && first.angle.abs() < harmonic + 2.0
                    && first.modulus >= seasonal_cutoff
            });
            if seasonal {
                if unit {
                    GeometryFactor::SeasonalNonstationary
                } else {
                    GeometryFactor::SeasonalStationary
                }
            } else {
                allocation.is_close_to_trading_day = is_close_td(first.angle, frequency);
                if unit {
                    GeometryFactor::CycleNonstationary
                } else {
                    GeometryFactor::CycleStationary
                }
            }
        };
        allocation.multiply(factor, &polynomial)?;
        if let Some(root) = roots.get(2) {
            // The real root after a complex pair. Here F1RST compares a
            // positive root with RmodS and a negative one with Rmod, the
            // reverse of the all-real branch.
            let real = root.real;
            let factor = if real > 0.0 {
                if (1.0 - real).abs() < 1e-6 {
                    GeometryFactor::TrendNonstationary
                } else if real.abs() >= seasonal_cutoff {
                    GeometryFactor::TrendStationary
                } else {
                    GeometryFactor::CycleStationary
                }
            } else if (1.0 + real).abs() < 1e-6 {
                GeometryFactor::SeasonalNonstationary
            } else if real.abs() >= 0.5 {
                GeometryFactor::SeasonalStationary
            } else {
                GeometryFactor::CycleStationary
            };
            allocation.multiply(factor, &[1.0, -real])?;
        }
    }
    Ok(allocation)
}

/// isCloseTD (`sigsub.f`): whether an angle in degrees is near the
/// trading-day frequency. Every comparison is strict.
fn is_close_td(angle: f64, frequency: Frequency) -> bool {
    match frequency {
        Frequency::Monthly => angle.abs() < 0.732 * 180.0 && angle.abs() > 0.680556 * 180.0,
        Frequency::Quarterly => (angle.abs() - 0.0892 * 180.0).abs() < 0.02 * 180.0,
    }
}

/// SIGEX's factors for a seasonal AR (BP=1), before any regular roots. A
/// negative seasonal AR goes to the cycle whole; F1RST's rules for negative
/// regular roots do not apply to it.
fn allocate_seasonal_ar(
    frequency: Frequency,
    seasonal_ar: f64,
    differences: DifferenceOrder,
) -> Result<AllocatedGeometry, Error> {
    let seasonal_ar = finite(seasonal_ar, "seasonal AR coefficient", 0)?;
    let s = usize::from(frequency.periods_per_year());
    let mut allocation = allocate_roots_with_differences(frequency, &[], differences)?;
    if seasonal_ar > 0.0 {
        let cmu = finite(seasonal_ar.powf(1.0 / s as f64), "seasonal AR root", 0)?;
        let unit = (1.0 - cmu).abs() < 1e-13;
        allocation.multiply(
            if unit {
                GeometryFactor::TrendNonstationary
            } else {
                GeometryFactor::TrendStationary
            },
            &[1.0, -cmu],
        )?;
        // 1 + cmu B + ... + cmu^(s-1) B^(s-1), each power one product on.
        let polynomial = std::iter::successors(Some(1.0), |&power| Some(cmu * power))
            .take(s)
            .collect::<Vec<_>>();
        allocation.multiply(
            if unit {
                GeometryFactor::SeasonalNonstationary
            } else {
                GeometryFactor::SeasonalStationary
            },
            &polynomial,
        )?;
    } else if seasonal_ar < 0.0 {
        allocation.cycle_stationary = seasonal_lag(s, seasonal_ar);
    }
    Ok(allocation)
}

/// SIGEX's factors for a seasonal AR with regular roots: the seasonal AR's
/// factors first, then F1RST's allocation into the same polynomials.
pub(crate) fn allocate_mixed_ar(
    frequency: Frequency,
    roots: &[ArRoot],
    seasonal_ar: f64,
    differences: DifferenceOrder,
) -> Result<AllocatedGeometry, Error> {
    let allocation = allocate_seasonal_ar(frequency, seasonal_ar, differences)?;
    allocate_regular_roots(frequency, roots, allocation)
}

/// `1 - coefficient B^period`.
fn seasonal_lag(period: usize, coefficient: f64) -> Vec<f64> {
    let mut polynomial = vec![0.0; period + 1];
    polynomial[0] = 1.0;
    polynomial[period] = -coefficient;
    polynomial
}

/// SIGEX's polynomials for the allocated factors. `regular_ma` holds the
/// regular MA coefficients (none for Q=0); `seasonal_ma` is `None` when the
/// model has no seasonal MA (BQ=0). A stochastic trading day keeps the cycle
/// out of the trend-cycle and seasonally adjusted denominators.
pub(crate) fn spectral_inputs(
    frequency: Frequency,
    allocation: &AllocatedGeometry,
    regular_ma: &[f64],
    seasonal_ma: Option<f64>,
    trading_day: EffectiveTradingDay,
) -> Result<SpectralInputs, Error> {
    let mut index = 0;
    for (factor, values) in allocation.factors() {
        check_capacity(factor, values.len())?;
        if values.is_empty() {
            return Err(invalid_root_pair(index));
        }
        for &value in values {
            finite(value, SPECTRAL_INPUT, index)?;
            index += 1;
        }
    }
    // The seasonal MA keeps its coordinate even when absent.
    if let Some(ma) = seasonal_ma {
        finite(ma, SPECTRAL_INPUT, index)?;
    }
    index += 1;
    for (offset, &ma) in regular_ma.iter().enumerate() {
        finite(ma, SPECTRAL_INPUT, index + offset)?;
    }
    let period = usize::from(frequency.periods_per_year());
    let theta = std::iter::once(1.0)
        .chain(regular_ma.iter().map(|ma| -ma))
        .collect::<Vec<_>>();
    let seasonal_theta = seasonal_ma.map_or_else(|| vec![1.0], |ma| seasonal_lag(period, ma));
    let aggregate_ma = convolve(&theta, &seasonal_theta);
    // The total denominator keeps the cycle even with a stochastic trading
    // day.
    let trend_combined = convolve(
        &allocation.trend_stationary,
        &allocation.trend_nonstationary,
    );
    let seasonal_combined = convolve(
        &allocation.seasonal_stationary,
        &allocation.seasonal_nonstationary,
    );
    let cycle_combined = convolve(
        &allocation.cycle_stationary,
        &allocation.cycle_nonstationary,
    );
    let ordinary_trend_cycle = convolve(&trend_combined, &cycle_combined);
    let seasonal_cycle = convolve(&seasonal_combined, &cycle_combined);
    let trend_seasonal = convolve(&trend_combined, &seasonal_combined);
    let aggregate_denominator = convolve(&seasonal_combined, &ordinary_trend_cycle);
    let (trend_cycle, adjustment_stationary, adjustment_nonstationary) = match trading_day {
        EffectiveTradingDay::Stochastic => (
            trend_combined.clone(),
            allocation.trend_stationary.clone(),
            allocation.trend_nonstationary.clone(),
        ),
        EffectiveTradingDay::Ordinary => (
            ordinary_trend_cycle,
            convolve(&allocation.cycle_stationary, &allocation.trend_stationary),
            convolve(
                &allocation.cycle_nonstationary,
                &allocation.trend_nonstationary,
            ),
        ),
    };
    let inputs = SpectralInputs {
        theta,
        seasonal_theta,
        aggregate_ma,
        trend: DenominatorFactors {
            stationary: allocation.trend_stationary.clone(),
            nonstationary: allocation.trend_nonstationary.clone(),
            combined: trend_combined,
        },
        seasonal: DenominatorFactors {
            stationary: allocation.seasonal_stationary.clone(),
            nonstationary: allocation.seasonal_nonstationary.clone(),
            combined: seasonal_combined,
        },
        cycle: DenominatorFactors {
            stationary: allocation.cycle_stationary.clone(),
            nonstationary: allocation.cycle_nonstationary.clone(),
            combined: cycle_combined,
        },
        trend_cycle,
        seasonal_cycle,
        trend_seasonal,
        adjustment_stationary,
        adjustment_nonstationary,
        aggregate_denominator,
    };
    for (index, &value) in inputs
        .aggregate_ma
        .iter()
        .chain(&inputs.trend.combined)
        .chain(&inputs.seasonal.combined)
        .chain(&inputs.cycle.combined)
        .chain(&inputs.trend_cycle)
        .chain(&inputs.seasonal_cycle)
        .chain(&inputs.trend_seasonal)
        .chain(&inputs.adjustment_stationary)
        .chain(&inputs.adjustment_nonstationary)
        .chain(&inputs.aggregate_denominator)
        .enumerate()
    {
        finite(value, SPECTRAL_INPUT, index)?;
    }
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use super::GeometryFactor::{
        CycleNonstationary, CycleStationary, SeasonalNonstationary, SeasonalStationary,
        TrendNonstationary, TrendStationary,
    };
    use super::*;

    fn complex_pair(modulus: f64, degrees: f64) -> [ArRoot; 2] {
        let radians = degrees.to_radians();
        let root = ArRoot {
            real: modulus * radians.cos(),
            imaginary: modulus * radians.sin(),
            modulus,
            angle: degrees,
            period: 360.0 / degrees,
        };
        [
            root,
            ArRoot {
                imaginary: -root.imaginary,
                angle: -degrees,
                period: -root.period,
                ..root
            },
        ]
    }

    fn real_root(real: f64) -> ArRoot {
        ArRoot {
            real,
            imaginary: 0.0,
            modulus: real.abs(),
            angle: if real < 0.0 { 180.0 } else { 0.0 },
            period: 999.0,
        }
    }

    /// The factors a monthly model's roots are multiplied into.
    fn grown(roots: &[ArRoot], differences: DifferenceOrder) -> Vec<GeometryFactor> {
        let monthly = Frequency::Monthly;
        let before = allocate_roots_with_differences(monthly, &[], differences).unwrap();
        let after = allocate_roots_with_differences(monthly, roots, differences).unwrap();
        before
            .factors()
            .into_iter()
            .zip(after.factors())
            .filter(|((_, before), (_, after))| after.len() > before.len())
            .map(|((factor, _), _)| factor)
            .collect()
    }

    #[test]
    fn real_roots_go_to_the_factor_their_sign_and_size_select() {
        let stationary = DifferenceOrder::Stationary;
        for (real, factor) in [
            (0.7, TrendStationary),
            (0.3, CycleStationary),
            (-0.3, CycleStationary),
            (-0.6, CycleStationary),
            (-0.95, SeasonalStationary),
            (1.0, TrendNonstationary),
            (-1.0, SeasonalNonstationary),
        ] {
            assert_eq!(grown(&[real_root(real)], stationary), [factor], "{real}");
        }
        // A seasonal difference lowers the seasonal cutoff from 0.9 to 0.5.
        assert_eq!(
            grown(&[real_root(-0.6)], DifferenceOrder::First),
            [SeasonalStationary]
        );
        let allocation =
            allocate_roots_with_differences(Frequency::Monthly, &[real_root(0.7)], stationary)
                .unwrap();
        assert_eq!(allocation.trend_stationary, [1.0, -0.7]);
    }

    #[test]
    fn complex_pairs_go_to_the_factor_their_angle_and_modulus_select() {
        let stationary = DifferenceOrder::Stationary;
        for (modulus, degrees, factor) in [
            (0.9, 10.0, TrendStationary),
            (1.0, 10.0, TrendNonstationary),
            (0.45, 10.0, CycleStationary),
            (0.95, 60.0, SeasonalStationary),
            (1.0, 60.0, SeasonalNonstationary),
            (0.6, 60.0, CycleStationary),
            (0.8, 126.0, CycleStationary),
            (1.0, 126.0, CycleNonstationary),
        ] {
            assert_eq!(
                grown(&complex_pair(modulus, degrees), stationary),
                [factor],
                "{modulus} {degrees}"
            );
        }
        assert_eq!(
            grown(&complex_pair(0.6, 60.0), DifferenceOrder::First),
            [SeasonalStationary]
        );
    }

    #[test]
    fn a_real_root_after_a_pair_swaps_the_two_cutoffs() {
        let stationary = DifferenceOrder::Stationary;
        let with_real = |real| {
            let mut roots = complex_pair(0.8, 126.0).to_vec();
            roots.push(real_root(real));
            grown(&roots, stationary)
        };
        // Alone, 0.7 is a trend root and -0.7 a cycle root.
        assert_eq!(with_real(0.95), [TrendStationary, CycleStationary]);
        assert_eq!(with_real(0.7), [CycleStationary]);
        assert_eq!(with_real(-0.7), [SeasonalStationary, CycleStationary]);
        assert_eq!(with_real(-0.3), [CycleStationary]);
        assert_eq!(with_real(1.0), [TrendNonstationary, CycleStationary]);
        assert_eq!(with_real(-1.0), [SeasonalNonstationary, CycleStationary]);
    }

    #[test]
    fn a_cycle_pair_near_the_trading_day_frequency_is_flagged() {
        let monthly = allocate_roots_with_differences(
            Frequency::Monthly,
            &complex_pair(0.8, 126.0),
            DifferenceOrder::First,
        )
        .unwrap();
        assert!(monthly.is_close_to_trading_day);
        let quarterly = allocate_roots_with_differences(
            Frequency::Quarterly,
            &complex_pair(0.45, 16.0),
            DifferenceOrder::First,
        )
        .unwrap();
        assert!(quarterly.is_close_to_trading_day);
        let seasonal = allocate_roots_with_differences(
            Frequency::Monthly,
            &complex_pair(0.95, 60.0),
            DifferenceOrder::Stationary,
        )
        .unwrap();
        assert!(!seasonal.is_close_to_trading_day);
        assert_eq!(seasonal.seasonal_stationary.len(), 3);
    }
}
