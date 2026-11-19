//! SPECTRU's rational split (`spectrum.f`): DIVFCN divides the aggregate
//! spectrum's numerator harmonics by its denominator, and PARFRA splits the
//! remainder into the trend, seasonal and cycle numerators by partial
//! fractions (both in `ansub2.f`).

use std::cmp::Ordering;
use std::ops::Range;

use super::harmonic::{Harmonics, build_harmonics, multiply_harmonics};
use super::polynomial::{MaFamily, SpectralInputs, spectral_inputs};
use crate::numerics::RowMajorMatrix;
use crate::seats::airline::model::ResolvedAirline;
use crate::seats::failure::{Error, Stage};
use crate::seats::linear::solve;

/// DIVFCN's quotient and PARFRA's component numerators.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RationalPrefix {
    pub(crate) harmonics: Harmonics,
    pub(crate) quotient: Vec<f64>,
    pub(crate) trend_numerator: Vec<f64>,
    pub(crate) seasonal_numerator: Vec<f64>,
    pub(crate) cycle_numerator: Option<Vec<f64>>,
}

/// DIVFCN for equal orders: one quotient coefficient, and a remainder that
/// keeps its exact-zero slots.
fn divide_equal_order(numerator: &[f64], denominator: &[f64]) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(numerator.len() == denominator.len() && denominator.len() >= 2);
    let mut remainder = numerator.to_vec();
    let factor = remainder[numerator.len() - 1] / denominator[denominator.len() - 1];
    for (value, &coefficient) in remainder.iter_mut().zip(denominator) {
        // The product rounds before the subtraction: gfortran does not fuse
        // them in DIVFCN.
        let product = factor * coefficient;
        *value -= product;
    }
    remainder.truncate(denominator.len() - 1);
    (vec![factor], remainder)
}

/// DIVFCN for a numerator of higher order: the quotient harmonics from the
/// top down. Each step subtracts at both folded lags, which can be the same
/// slot, so the two subtractions run in order.
fn divide_higher_order(numerator: &[f64], denominator: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut work = numerator.to_vec();
    let mut quotient = vec![0.0; numerator.len() - denominator.len() + 1];
    if denominator.len() == 1 {
        return (
            numerator.iter().map(|x| x / denominator[0]).collect(),
            vec![0.0],
        );
    }
    for k in (0..quotient.len()).rev() {
        let factor = work[k + denominator.len() - 1] / denominator[denominator.len() - 1];
        quotient[k] = if k == 0 { factor } else { 2.0 * factor };
        for (i, &coefficient) in denominator.iter().enumerate() {
            let product = factor * coefficient;
            work[i + k] -= product;
            if k != 0 {
                work[i.abs_diff(k)] -= product;
            }
        }
    }
    work.truncate(denominator.len() - 1);
    (quotient, work)
}

/// SPECTRU's branch for a numerator of lower order: a zero quotient, and the
/// numerator padded with positive zeros up to pstar-1 as the remainder.
fn divide_lower_order(harmonics: &Harmonics) -> (Vec<f64>, Vec<f64>) {
    let mut remainder = harmonics.numerator.clone();
    remainder.resize(harmonics.denominator.len() - 1, 0.0);
    (vec![0.0], remainder)
}

/// SPECTRU adds a quotient of positive degree to the transitory numerator
/// rather than to the irregular variance.
fn include_polynomial_cycle(prefix: &mut RationalPrefix) {
    if prefix.harmonics.numerator.len() <= prefix.harmonics.denominator.len() {
        return;
    }
    let polynomial = multiply_harmonics(&prefix.quotient, &prefix.harmonics.cycle);
    let cycle = prefix.cycle_numerator.get_or_insert_with(Vec::new);
    let shared = cycle.len().min(polynomial.len());
    for (value, &coefficient) in cycle.iter_mut().zip(&polynomial).take(shared) {
        *value += coefficient;
    }
    if polynomial.len() > cycle.len() {
        cycle.extend_from_slice(&polynomial[cycle.len()..]);
    }
}

/// SPECTRU with no trend, seasonal or cycle factor (nchi=npsi=ncyc=1): a
/// seasonal AR whose estimate is exactly zero, so SIGEX allocated nothing,
/// with no regular AR and no differencing.
///
/// SIGEX still counts BP in `pstar = p + d + mq*(bd+bp) + 1`, so pstar is
/// mq+1 while the total denominator is the constant 1 (Build 62 prints
/// "WARNING: DIMENSION PROBLEM"). Without a seasonal MA, qstar is at most
/// q+1 <= 4 < pstar, so SPECTRU takes its lower-order branch: Qt=0 and RT is
/// the numerator. None of the component branches that follow matches, so RT
/// goes to no component and no minimum is searched.
///
/// SPECTRU leaves enot, estar and enoc uninitialized in this branch. Build 62
/// prints the irregular variance as 0, which the minima's zero defaults
/// reproduce; no saved component reads them.
fn no_component_prefix(harmonics: Harmonics) -> RationalPrefix {
    RationalPrefix {
        harmonics,
        quotient: vec![0.0],
        trend_numerator: Vec::new(),
        seasonal_numerator: Vec::new(),
        cycle_numerator: None,
    }
}

/// SPECTRU with a trend only: the remainder is the trend numerator, with no
/// PARFRA call.
pub(crate) fn nonseasonal_ma_prefix(inputs: &SpectralInputs) -> RationalPrefix {
    let harmonics = build_harmonics(inputs);
    let (quotient, remainder) = divide_equal_order(&harmonics.numerator, &harmonics.denominator);
    RationalPrefix {
        harmonics,
        quotient,
        trend_numerator: remainder,
        seasonal_numerator: Vec::new(),
        cycle_numerator: None,
    }
}

/// SPECTRU's division and partial fractions for any mix of trend, seasonal
/// and cycle factors.
///
/// A stationary AR whose roots all go to the seasonal component, such as
/// (1,0,0) with phi <= -0.9, leaves no trend or cycle factor. SPECTRU then
/// copies RT into V and never calls PARFRA on the empty trend-cycle
/// denominator. `adjusted::factor_adjusted` has the matching guard for the
/// seasonally adjusted series.
pub(crate) fn lower_degree_rational_prefix(
    inputs: &SpectralInputs,
) -> Result<RationalPrefix, Error> {
    let harmonics = build_harmonics(inputs);
    let has_cycle = inputs.cycle.combined.len() > 1;
    if inputs.trend.combined.len() == 1
        && inputs.seasonal.combined.len() == 1
        && !has_cycle
        && inputs.seasonal_theta.len() == 1
    {
        return Ok(no_component_prefix(harmonics));
    }
    let (quotient, remainder) = match harmonics.numerator.len().cmp(&harmonics.denominator.len()) {
        Ordering::Less => divide_lower_order(&harmonics),
        Ordering::Equal => divide_equal_order(&harmonics.numerator, &harmonics.denominator),
        Ordering::Greater => divide_higher_order(&harmonics.numerator, &harmonics.denominator),
    };
    if inputs.seasonal.combined.len() > 1 && inputs.trend.combined.len() == 1 && !has_cycle {
        // A seasonal factor with no trend or cycle: SPECTRU puts RT straight
        // into V and calls no PARFRA, so UT stays empty.
        let mut prefix = RationalPrefix {
            harmonics,
            quotient,
            trend_numerator: Vec::new(),
            seasonal_numerator: remainder,
            cycle_numerator: None,
        };
        include_polynomial_cycle(&mut prefix);
        return Ok(prefix);
    }
    if inputs.seasonal.combined.len() == 1 {
        // No seasonal factor, so no seasonal PARFRA system. With a cycle
        // SPECTRU splits RT between the trend and the cycle directly.
        let (trend_numerator, cycle_numerator) = if has_cycle {
            let (trend, cycle) = partial_fractions(&remainder, &harmonics.trend, &harmonics.cycle)?;
            (trend, Some(cycle))
        } else {
            (remainder, None)
        };
        let mut prefix = RationalPrefix {
            harmonics,
            quotient,
            trend_numerator,
            seasonal_numerator: Vec::new(),
            cycle_numerator,
        };
        include_polynomial_cycle(&mut prefix);
        return Ok(prefix);
    }
    let (mut trend_numerator, seasonal_numerator) =
        partial_fractions(&remainder, &harmonics.trend_cycle, &harmonics.seasonal)?;
    let cycle_numerator = if has_cycle {
        // SPECTRU pads U to ncyc+nchi-1; PARFRA reads its first m+n slots.
        trend_numerator.resize(
            inputs.cycle.combined.len() + inputs.trend.combined.len() - 1,
            0.0,
        );
        let (mut cycle, trend) =
            partial_fractions(&trend_numerator, &harmonics.cycle, &harmonics.trend)?;
        // SPECTRU floors a constant cycle numerator at 1e-15.
        if cycle.len() == 1 && cycle[0].abs() < 1e-15 {
            cycle[0] = 1e-15;
        }
        trend_numerator = trend;
        Some(cycle)
    } else {
        None
    };
    let mut prefix = RationalPrefix {
        harmonics,
        quotient,
        trend_numerator,
        seasonal_numerator,
        cycle_numerator,
    };
    include_polynomial_cycle(&mut prefix);
    Ok(prefix)
}

/// PARFRA: the numerators of `remainder` over `left` and over `right`.
fn partial_fractions(
    remainder: &[f64],
    left: &[f64],
    right: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), Error> {
    let m = left.len() - 1;
    let n = right.len() - 1;
    let matrix = partial_fractions_matrix(remainder, left, right);
    let solved =
        solve(matrix, m + n, 1).map_err(|failure| failure.at(Stage::CanonicalDecomposition))?;
    Ok(solved_numerators(&solved, m))
}

/// PARFRA's two numerators: the solved right-hand column, split after the
/// first `m` rows.
fn solved_numerators(matrix: &RowMajorMatrix, m: usize) -> (Vec<f64>, Vec<f64>) {
    let p = matrix.row_count();
    let mut column = matrix.rows().map(|row| row[p]);
    let left = column.by_ref().take(m).collect();
    (left, column.collect())
}

/// CONVM then CONJM (`ansub2.f`): `coefficients` times a block of ones, added
/// into `columns` of the row-major `matrix` of rows `width` cells wide.
/// Column `j` of the block gets coefficient `i` in row `i + j` (CONVM), then
/// in row `|i - j|` (CONJM).
fn add_product_columns(
    matrix: &mut [f64],
    width: usize,
    coefficients: &[f64],
    columns: Range<usize>,
) {
    // The coefficients run in the outer loop. gfortran fuses each
    // accumulation, with the unit factor as its multiplicand.
    let count = columns.len();
    for (i, &coefficient) in coefficients.iter().enumerate() {
        for (row, column) in (i..i + count).zip(columns.clone()) {
            let cell = &mut matrix[row * width + column];
            *cell = coefficient.mul_add(1.0, *cell);
        }
    }
    for (i, &coefficient) in coefficients.iter().enumerate() {
        for (j, column) in columns.clone().enumerate() {
            let cell = &mut matrix[i.abs_diff(j) * width + column];
            *cell = coefficient.mul_add(1.0, *cell);
        }
    }
}

/// PARFRA's `p × (p + 1)` system, `p = m + n`: the first `m` columns multiply
/// the numerator over `left`, the next `n` the numerator over `right`, and
/// the last holds the first `p` values of `remainder`.
fn partial_fractions_matrix(remainder: &[f64], left: &[f64], right: &[f64]) -> RowMajorMatrix {
    let m = left.len() - 1;
    let n = right.len() - 1;
    let p = m + n;
    let width = p + 1;
    debug_assert!(remainder.len() >= p);
    let mut matrix = vec![0.0; p * width];
    add_product_columns(&mut matrix, width, right, 0..m);
    add_product_columns(&mut matrix, width, left, m..p);
    for (row, &value) in matrix.chunks_exact_mut(width).zip(remainder) {
        for cell in &mut row[..p] {
            *cell /= 2.0;
        }
        row[p] = value;
    }
    RowMajorMatrix::new(matrix, width)
}

/// SPECTRU's division and partial fractions for the airline model.
pub(super) fn rational_prefix(model: &ResolvedAirline) -> Result<RationalPrefix, Error> {
    rational_prefix_inputs(&spectral_inputs(model), MaFamily::Airline)
}

/// SPECTRU's division and partial fractions for a model whose trend is two
/// differences and whose seasonal factor is the seasonal sum, with no cycle.
pub(crate) fn rational_prefix_inputs(
    inputs: &SpectralInputs,
    family: MaFamily,
) -> Result<RationalPrefix, Error> {
    let harmonics = build_harmonics(inputs);
    debug_assert_eq!(harmonics.trend.len(), 3);
    debug_assert_eq!(harmonics.cycle.len(), 1);
    debug_assert_eq!(
        harmonics.numerator.len(),
        family.ma_coefficients(harmonics.seasonal.len())
    );
    let (quotient, remainder) = match family {
        MaFamily::Airline => divide_equal_order(&harmonics.numerator, &harmonics.denominator),
        MaFamily::SeasonalMa => divide_lower_order(&harmonics),
    };
    let (trend_numerator, seasonal_numerator) =
        partial_fractions(&remainder, &harmonics.trend, &harmonics.seasonal)?;
    Ok(RationalPrefix {
        harmonics,
        quotient,
        trend_numerator,
        seasonal_numerator,
        cycle_numerator: None,
    })
}

#[cfg(test)]
mod tests;
