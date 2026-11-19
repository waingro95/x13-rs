//! Roots of the real polynomials SEATS factors: SYMPOLY's reduction of a
//! symmetric polynomial, the C02AEF root finder, and RPQ's root metadata.
//!
//! C02AEF and its evaluator C02AEZ are Build 62's polynomial root search
//! (`ansub2.f`), after Grant and Hitchins (1971). The arithmetic here
//! (operation order, constants, and the multiply-adds gfortran fused) follows
//! Build 62 exactly: the root allocation's spectral split turns a one-ulp root
//! change into visible drift in the SEATS outputs, so another root finder
//! would not reproduce Build 62's decomposition.

use std::f64::consts::LN_2;

use crate::numerics::BUILD_62_PI;
use crate::seats::failure::{Error, Failure, Stage};

/// C02AEF's convergence tolerance, as Build 62 computes it.
const TOLERANCE: f64 = f64::from_bits(0x3d19_0d56_b873_f4c5);
/// Build 62's machine epsilon: the search gives up once a step falls below it.
const STEP_EPSILON: f64 = f64::from_bits(0x3caf_ffff_f417_8da3);
/// Build 62's largest finite value. Its square root caps backward deflation.
const LARGEST: f64 = f64::from_bits(0x7fef_ffff_d7b9_609a);
/// Build 62's `dpeq` threshold for "equal to zero".
const ZERO_THRESHOLD: f64 = f64::from_bits(0x3be6_a1c6_e8d9_8029);
/// RPQ's period for a real root: the REAL constant 999.99, rounded to
/// binary32 and then widened to binary64.
pub(crate) const REAL_ROOT_PERIOD: f64 = f64::from_bits(0x408f_3feb_8000_0000);

/// MAK1 (ansub2.f) receives RPQ's `nrpoly-1` roots in `rez1(32)`, so the
/// longest numerator it can factor has 33 coefficients; its `poly(34)` and
/// SYMPOLY's `rdpoly(34)` are looser. Seasonal numerators reach 26.
pub(super) const MAK1_MAX_COEFFICIENTS: usize = 33;

/// C02AEF's search point and the values C02AEZ last evaluated there (Build
/// 62's COMMON `/ac02ae/`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RootState {
    /// Real part of the search point (`X`).
    x: f64,
    /// Imaginary part of the search point (`Y0`).
    y: f64,
    /// Real part of p(x + iy) (`R`).
    r: f64,
    /// Real part of p'(x + iy) (`RX`).
    rx: f64,
    /// Imaginary part of p(x + iy) (`J`).
    j: f64,
    /// Imaginary part of p'(x + iy) (`JX`).
    jx: f64,
    /// Whether Adams's rounding-error test accepted the point (`SAT`).
    satisfied: bool,
}

impl RootState {
    /// |p(x + iy)|², fused as Build 62 computes it.
    fn residual(&self) -> f64 {
        self.r.mul_add(self.r, self.j * self.j)
    }
}

/// C02AEF's working arrays.
struct Solver {
    /// The working coefficients, rescaled and deflated in place.
    coefficients: Vec<f64>,
    /// Coefficients still in play; 1 once every root is found.
    remaining_count: usize,
    /// Roots, filled from the back as they are deflated off.
    real: Vec<f64>,
    imaginary: Vec<f64>,
    state: RootState,
}

/// A polynomial's roots with RPQ's metadata, complex roots first.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RootOutput {
    pub(super) real: Vec<f64>,
    pub(super) imaginary: Vec<f64>,
    pub(super) modulus: Vec<f64>,
    pub(super) argument_degrees: Vec<f64>,
    pub(super) period: Vec<f64>,
}

/// The search stalled: the derivative vanished or the step underflowed.
struct Stalled;

/// Accepts the 3 to 33 finite coefficients MAK1's arrays hold.
fn check_input(poly: &[f64]) -> Result<(), Error> {
    if !(3..=MAK1_MAX_COEFFICIENTS).contains(&poly.len()) {
        return Err(Failure::Unusable {
            quantity: "root polynomial coefficient count",
            index: None,
            value: poly.len() as f64,
        }
        .at(Stage::CanonicalDecomposition));
    }
    if let Some(index) = poly.iter().position(|value| !value.is_finite()) {
        return Err(Failure::Unusable {
            quantity: "root polynomial coefficient",
            index: Some(index),
            value: poly[index],
        }
        .at(Stage::CanonicalDecomposition));
    }
    Ok(())
}

/// SYMPOLY: rewrites a polynomial symmetric in z and 1/z as a polynomial in
/// (z + 1/z). Coefficients are in descending powers, as in Build 62.
pub(super) fn reduce_symmetric(poly: &[f64]) -> Vec<f64> {
    debug_assert!(check_input(poly).is_ok());
    let basis = symmetric_basis(poly.len());
    let mut reduced: Vec<f64> = (0..poly.len())
        .map(|row| {
            // Build 62's vectorized loop fuses each row's terms in ascending
            // column order; it does not sum lanes horizontally.
            basis
                .iter()
                .zip(poly.iter().rev())
                .fold(0.0, |sum, (column, &coefficient)| {
                    let entry = column.get(row).copied().unwrap_or(0.0);
                    entry.mul_add(coefficient, sum)
                })
        })
        .collect();
    reduced.reverse();
    reduced
}

/// Column k holds z^k + z^-k in ascending powers of (z + 1/z); column 0 holds
/// the constant 1. The entries are small integers, so building them is exact.
fn symmetric_basis(count: usize) -> Vec<Vec<f64>> {
    let mut columns = vec![vec![1.0], vec![0.0, 1.0]];
    // The recurrence's seed is z^0 + z^-0 = 2, unlike column 0.
    let mut previous = vec![2.0];
    while columns.len() < count {
        let current = &columns[columns.len() - 1];
        let mut next = vec![0.0; current.len() + 1];
        next[1..].copy_from_slice(current);
        for (value, older) in next.iter_mut().zip(&previous) {
            *value -= older;
        }
        previous = current.clone();
        columns.push(next);
    }
    columns.truncate(count);
    columns
}

/// Build 62's `dpeq(value, 0)`.
fn is_negligible(value: f64) -> bool {
    value.abs() < ZERO_THRESHOLD
}

/// 2^exponent by GCC's `__powidf2`: square-and-multiply on the magnitude, then
/// one reciprocal for a negative exponent. `unsigned_abs` keeps `i32::MIN`.
fn two_power(exponent: i32) -> f64 {
    let magnitude = exponent.unsigned_abs();
    let mut base = 2.0;
    let mut value = if magnitude & 1 == 0 { 1.0 } else { base };
    let mut remaining = magnitude >> 1;
    while remaining != 0 {
        base *= base;
        if remaining & 1 != 0 {
            value *= base;
        }
        remaining >>= 1;
    }
    if exponent < 0 { 1.0 / value } else { value }
}

/// C02AEZ: evaluates p and p' at the search point w = x + iy by synthetic
/// division by the real quadratic (z - w)(z - w̄), then applies Adams's test:
/// is |p(w)| within the rounding error of evaluating it?
fn evaluate(coefficients: &[f64], state: &mut RootState) {
    let (x, y) = (state.x, state.y);
    let twice_x = x + x;
    let norm_squared = x.mul_add(x, y * y);
    let norm = norm_squared.sqrt();
    // One synthetic-division step, fused the way gfortran compiled it.
    let divide = |latest: f64, older: f64, addend: f64| {
        (-norm_squared).mul_add(older, twice_x.mul_add(latest, addend))
    };

    let len = coefficients.len();
    let leading = coefficients[0];
    let (mut quotient, mut quotient_older) = (leading, 0.0);
    let (mut slope, mut slope_older) = (leading, 0.0);
    let mut error = leading.abs() * 0.8;
    for &coefficient in &coefficients[1..len - 2] {
        (quotient, quotient_older) = (divide(quotient, quotient_older, coefficient), quotient);
        error = norm.mul_add(error, quotient.abs());
        (slope, slope_older) = (divide(slope, slope_older, quotient), slope);
    }
    (quotient, quotient_older) = (
        divide(quotient, quotient_older, coefficients[len - 2]),
        quotient,
    );

    let x_quotient = x * quotient;
    state.r = (-norm_squared).mul_add(quotient_older, coefficients[len - 1] + x_quotient);
    state.j = quotient * y;
    state.rx = (-y).mul_add(y * (slope_older + slope_older), quotient);
    state.jx = (-x).mul_add(slope_older, slope) * (y + y);
    error = norm.mul_add(norm.mul_add(error, quotient.abs()), state.r.abs());
    let magnitude = state.r.mul_add(state.r, state.j * state.j).sqrt();
    let correction = quotient.abs().mul_add(norm, state.r.abs()) * 8.0;
    let bound = x_quotient
        .abs()
        .mul_add(2.0, error.mul_add(10.0, -correction))
        * TOLERANCE;
    state.satisfied = magnitude < bound;
}

impl Solver {
    fn new(poly: &[f64]) -> Self {
        let root_count = poly.len() - 1;
        Self {
            coefficients: poly.to_vec(),
            remaining_count: poly.len(),
            // NaN until written, so a root the search never reached fails the
            // finite check.
            real: vec![f64::NAN; root_count],
            imaginary: vec![f64::NAN; root_count],
            state: RootState::default(),
        }
    }

    fn is_finite(&self) -> bool {
        let state = self.state;
        self.coefficients
            .iter()
            .chain(&self.real)
            .chain(&self.imaginary)
            .chain(&[state.x, state.y, state.r, state.rx, state.j, state.jx])
            .all(|value| value.is_finite())
    }

    /// C02AEF's main loop: find the root nearest the origin, deflate it off,
    /// and repeat until a closed form finishes the last one or two.
    fn find_roots(&mut self) -> Result<(), Stalled> {
        let Self {
            coefficients,
            remaining_count: len,
            real,
            imaginary,
            state,
        } = self;
        let stall = TOLERANCE.powf(1.5);
        let cap = LARGEST.sqrt();
        let mut root_scale = 1.0;
        let mut forward = vec![0.0; coefficients.len()];
        let mut backward = vec![0.0; coefficients.len()];
        let mut scratch = vec![0.0; coefficients.len()];

        // Coefficients that vanish at the low end are roots at zero.
        while *len >= 2 && is_negligible(coefficients[*len - 1]) {
            real[*len - 2] = 0.0;
            imaginary[*len - 2] = 0.0;
            *len -= 1;
        }

        loop {
            let n = *len;
            let poly = &mut coefficients[..n];
            normalize(poly);
            match n {
                1 => return Ok(()),
                2 => {
                    real[0] = -(root_scale * (poly[1] / poly[0]));
                    imaginary[0] = 0.0;
                    *len = 1;
                    return Ok(());
                }
                3 => {
                    quadratic(poly, root_scale, real, imaginary, state);
                    *len = 1;
                    return Ok(());
                }
                _ => {}
            }

            // While every root lies outside the unit circle, halve them all
            // (z -> 2z) so the search from near zero finds the smallest.
            forward[..n].copy_from_slice(poly);
            while all_roots_outside_unit_circle(&mut forward[..n], &mut scratch) {
                root_scale += root_scale;
                halve_roots(poly);
                forward[..n].copy_from_slice(poly);
            }

            search_root(poly, state, stall)?;
            let scaled_x = state.x * root_scale;
            if is_real_root(poly, state) {
                real[n - 2] = scaled_x;
                imaginary[n - 2] = 0.0;
                *len = n - 1;
                deflate_real(poly, state.x, &mut forward, &mut backward, cap);
            } else {
                let scaled_y = state.y * root_scale;
                (real[n - 3], imaginary[n - 3]) = (scaled_x, -scaled_y);
                (real[n - 2], imaginary[n - 2]) = (scaled_x, scaled_y);
                *len = n - 2;
                deflate_conjugate(poly, state, &mut forward, &mut backward, cap);
            }
            let n = *len;
            recombine(&mut coefficients[..n], &forward[..n], &backward[..n], stall);
        }
    }
}

/// Scales the polynomial by the power of two nearest the reciprocal geometric
/// mean of its non-negligible coefficients.
fn normalize(poly: &mut [f64]) {
    let log_sum = poly
        .iter()
        .map(|value| value.abs())
        .filter(|&magnitude| magnitude >= 1e-5)
        .fold(0.0, |sum, magnitude| sum + magnitude.ln());
    let exponent = (log_sum / (poly.len() as f64 * LN_2) + 0.5) as i32;
    let scale = two_power(exponent.wrapping_neg());
    for value in poly {
        *value *= scale;
    }
}

/// Schur-Cohn test on the reversed polynomial. Running to completion means no
/// root lies on or inside the unit circle. Overwrites `poly` with the reduction.
fn all_roots_outside_unit_circle(poly: &mut [f64], scratch: &mut [f64]) -> bool {
    for len in (2..=poly.len()).rev() {
        let constant = poly[len - 1];
        if is_negligible(constant) {
            return false;
        }
        let ratio = poly[0] / constant;
        if ratio.abs() >= 1.0 {
            return false;
        }
        for (index, value) in scratch[..len - 1].iter_mut().enumerate() {
            *value = (-ratio).mul_add(poly[len - 2 - index], poly[index + 1]);
        }
        poly[..len - 1].copy_from_slice(&scratch[..len - 1]);
    }
    true
}

/// Substitutes z -> 2z, which halves every root: the coefficient of z^k gains
/// a factor 2^k. The constant term is unchanged.
fn halve_roots(poly: &mut [f64]) {
    let mut factor = 1.0;
    let last = poly.len() - 1;
    for value in poly[..last].iter_mut().rev() {
        factor += factor;
        *value *= factor;
    }
}

/// Damped Newton from 0.001 + 0.1i, halving each step until |p| falls far
/// enough, until Adams's test accepts the point.
fn search_root(poly: &[f64], state: &mut RootState, stall: f64) -> Result<(), Stalled> {
    state.x = 1e-3;
    state.y = 0.1;
    evaluate(poly, state);
    let mut residual = state.residual();
    loop {
        let slope = state.rx.mul_add(state.rx, state.jx * state.jx);
        if slope < residual * stall {
            return Err(Stalled);
        }
        // The Newton step -p/p', in real arithmetic.
        let along = state.rx.mul_add(state.r, state.jx * state.j) / slope;
        let mut step_x = -along;
        let mut step_y = state.jx.mul_add(state.r, -(state.rx * state.j)) / slope;
        let mut required_drop = 2e-4;
        let length = along.mul_add(along, step_y * step_y).sqrt();
        if length > 1.0 {
            step_x /= length;
            step_y /= length;
            required_drop /= length;
        }
        state.x += step_x;
        state.y += step_y;
        loop {
            evaluate(poly, state);
            if state.satisfied {
                return Ok(());
            }
            let trial = state.residual();
            if residual - trial >= required_drop * residual {
                residual = trial;
                break;
            }
            step_x *= 0.5;
            step_y *= 0.5;
            if step_x.abs() <= STEP_EPSILON * state.x.abs()
                && step_y.abs() <= STEP_EPSILON * state.y.abs()
            {
                return Err(Stalled);
            }
            required_drop *= 0.5;
            state.x -= step_x;
            state.y -= step_y;
        }
    }
}

/// Accepts x as a real root when |y| <= 0.1 and Adams's test passes at y = 0.
/// That evaluation stays in `state`, as it does in Build 62's COMMON.
fn is_real_root(poly: &[f64], state: &mut RootState) -> bool {
    if state.y.abs() <= 0.1 {
        let y = state.y;
        state.y = 0.0;
        evaluate(poly, state);
        state.y = y;
        state.satisfied
    } else {
        false
    }
}

fn within_cap(value: f64, cap: f64) -> bool {
    value.abs() <= cap
}

/// Divides out (z - x) both ways: `forward` from the leading coefficient,
/// `backward` from the constant, capped at `cap` once it grows past it.
fn deflate_real(poly: &[f64], x: f64, forward: &mut [f64], backward: &mut [f64], cap: f64) {
    let len = poly.len() - 1;
    forward[0] = poly[0];
    for index in 1..len {
        forward[index] = x.mul_add(forward[index - 1], poly[index]);
    }
    backward[len - 1] = -(poly[len] / x);
    let mut capped = false;
    for index in (0..len - 1).rev() {
        if !capped {
            backward[index] = (backward[index + 1] - poly[index + 1]) / x;
            capped = !within_cap(backward[index], cap);
        }
        if capped {
            backward[index] = cap;
        }
    }
}

/// Divides out z² - 2x·z + |w|² both ways, as `deflate_real` does.
fn deflate_conjugate(
    poly: &[f64],
    state: &mut RootState,
    forward: &mut [f64],
    backward: &mut [f64],
    cap: f64,
) {
    let len = poly.len() - 2;
    let x = state.x;
    let twice_x = x + x;
    let norm_squared = x.mul_add(x, state.y * state.y);
    // C02AEF reuses COMMON for the divisor: R holds 2x and J holds -|w|².
    state.r = twice_x;
    state.j = -norm_squared;

    forward[0] = poly[0];
    forward[1] = twice_x.mul_add(forward[0], poly[1]);
    for index in 2..len {
        forward[index] = (-norm_squared).mul_add(
            forward[index - 2],
            twice_x.mul_add(forward[index - 1], poly[index]),
        );
    }
    backward[len - 1] = poly[len + 1] / norm_squared;
    backward[len - 2] = twice_x.mul_add(backward[len - 1], poly[len]) / norm_squared;
    let mut capped = false;
    for index in (0..len.saturating_sub(2)).rev() {
        if !capped {
            let numerator =
                twice_x.mul_add(backward[index + 1], poly[index + 2] - backward[index + 2]);
            // Build 62 divides the negated numerator and negates the quotient,
            // which is exact; a NaN here is capped below.
            backward[index] = numerator / norm_squared;
            capped = !within_cap(backward[index], cap);
        }
        if capped {
            backward[index] = cap;
        }
    }
}

/// Joins the forward quotient (accurate at the leading end) and the backward
/// one (accurate at the trailing end) where they agree best, averaging that
/// coefficient. Takes the backward quotient whole when none qualifies.
fn recombine(poly: &mut [f64], forward: &[f64], backward: &[f64], stall: f64) {
    let mut best = 1.0 / stall;
    let mut join = None;
    for (index, (&ahead, &behind)) in forward.iter().zip(backward).enumerate() {
        let size = ahead.abs() + behind.abs();
        if size > TOLERANCE {
            let disagreement = (ahead - behind).abs() / size;
            if disagreement < best {
                best = disagreement;
                join = Some(index);
            }
        }
    }
    match join {
        Some(index) => {
            poly[..index].copy_from_slice(&forward[..index]);
            poly[index] = 0.5 * (forward[index] + backward[index]);
            poly[index + 1..].copy_from_slice(&backward[index + 1..]);
        }
        None => poly.copy_from_slice(backward),
    }
}

/// Both roots of a quadratic in closed form. The discriminant stays in R.
fn quadratic(
    poly: &[f64],
    root_scale: f64,
    real: &mut [f64],
    imaginary: &mut [f64],
    state: &mut RootState,
) {
    let [a, b, c] = [poly[0], poly[1], poly[2]];
    // 4·a and then ·c round separately before the fused subtraction.
    let discriminant = b.mul_add(b, -((4.0 * a) * c));
    state.r = discriminant;
    if discriminant > 0.0 {
        let larger = if b < 0.0 {
            (0.5 * (discriminant.sqrt() - b)) / a * root_scale
        } else if is_negligible(b) {
            -(root_scale * ((0.5 * discriminant.sqrt()) / a))
        } else {
            (0.5 * (-b - discriminant.sqrt())) / a * root_scale
        };
        real[0] = larger;
        real[1] = c / (larger * a) * root_scale * root_scale;
        imaginary[0] = 0.0;
        imaginary[1] = 0.0;
    } else {
        let real_part = -(root_scale * ((b * 0.5) / a));
        real[0] = real_part;
        real[1] = real_part;
        imaginary[1] = (0.5 * (-discriminant).sqrt()) / a * root_scale;
        imaginary[0] = -imaginary[1];
    }
}

/// C02AEF then RPQ: the roots of `poly` (descending powers), complex roots
/// ahead of real ones, each with its modulus, argument in degrees and period.
pub(super) fn polynomial_roots(poly: &[f64]) -> Result<RootOutput, Error> {
    check_input(poly)?;
    let mut solver = Solver::new(poly);
    if solver.find_roots().is_err() {
        return Err(Failure::RootSolverStalled {
            degree: poly.len() - 1,
        }
        .at(Stage::CanonicalDecomposition));
    }
    debug_assert_eq!(solver.remaining_count, 1);
    if !solver.is_finite() {
        return Err(unusable_roots("solver"));
    }

    let Solver {
        mut real,
        mut imaginary,
        ..
    } = solver;
    while let Some((complex, earlier_real)) = misplaced_root(&imaginary) {
        real.swap(complex, earlier_real);
        imaginary.swap(complex, earlier_real);
    }

    let count = real.len();
    let mut modulus = Vec::with_capacity(count);
    let mut argument_degrees = Vec::with_capacity(count);
    let mut period = Vec::with_capacity(count);
    for (&re, &im) in real.iter().zip(&imaginary) {
        let root_modulus = re.mul_add(re, im * im).sqrt();
        let cosine = if root_modulus < 1e-8 {
            0.0
        } else {
            re / root_modulus
        };
        let radians = if cosine.abs() <= 1.0 {
            let angle = cosine.acos();
            if im < 0.0 { -angle } else { angle }
        } else if re < 0.0 {
            BUILD_62_PI
        } else {
            0.0
        };
        let complex = im > TOLERANCE || im < -TOLERANCE;
        modulus.push(root_modulus);
        period.push(if radians.abs() > 1e-8 && complex {
            (2.0 * BUILD_62_PI) / radians
        } else {
            REAL_ROOT_PERIOD
        });
        argument_degrees.push(180.0 * radians / BUILD_62_PI);
    }

    if !modulus
        .iter()
        .chain(&argument_degrees)
        .chain(&period)
        .all(|value| value.is_finite())
    {
        return Err(unusable_roots("metadata"));
    }
    Ok(RootOutput {
        real,
        imaginary,
        modulus,
        argument_degrees,
        period,
    })
}

fn unusable_roots(step: &'static str) -> Error {
    Failure::UnusableRoots { step, index: None }.at(Stage::CanonicalDecomposition)
}

/// RPQ's reordering scan: the first complex root that follows a real one,
/// paired with the last real root before it. Swapping one pair per scan, as
/// RPQ does, fixes the order the allocation later reads.
fn misplaced_root(imaginary: &[f64]) -> Option<(usize, usize)> {
    let mut last_real = None;
    for (index, &value) in imaginary.iter().enumerate() {
        if value < TOLERANCE && value > -TOLERANCE {
            last_real = Some(index);
        } else if let Some(real_index) = last_real {
            return Some((index, real_index));
        }
    }
    None
}

#[cfg(test)]
mod tests;
