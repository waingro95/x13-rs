//! Roots of the ARIMA factor polynomials (degree 1 to 3).
//!
//! An independent implementation, not a translation of Build 62's RPOLY:
//! closed-form quadratic and cubic formulas with Newton polishing, and
//! Kahan (1986) for the direction in which a cubic is deflated.
//!
//! # Underflow
//!
//! Only a product or a quotient can underflow; a sum or difference that lands
//! below the normal range is exact. An underflowed result is off by at most
//! 2^-1075, half the smallest subnormal. That is harmless when the value is
//! next added into a sum that stays normal, because the error is then at most
//! half an ulp of the sum. Anywhere else it can change the roots, and
//! [`Underflow`] records it. A quadratic or cubic that records one is solved
//! again after an exact power-of-two rescaling that brings its leading
//! coefficient and its largest root near 1 ([`rescaled_roots`]). If the
//! rescaled problem underflows too, which takes roots spread over hundreds of
//! binary orders of magnitude, the solver returns `Err`. A polynomial that
//! records no underflow is never rescaled, so its roots are exactly those of
//! the plain method.

use super::polynomial::{ComplexRoot, RootComputationError};

/// Newton steps allowed when polishing one real root of a cubic.
const MAX_NEWTON_STEPS: usize = 8;

/// 2^-968. An fma recovers the rounding error of a product at least this
/// large exactly; below it, that error can itself underflow.
const EXACT_PRODUCT_ERROR_FLOOR: f64 = power_of_two(-968);

/// 2^-1071, sixteen times the largest error one underflow leaves. Per unit of
/// max(1, |x|)³, it bounds what the underflows in one evaluation of a cubic
/// can add to p(x): at most three per Horner step, each carried through the
/// remaining steps.
const UNDERFLOW_NOISE: f64 = 8.0 * f64::from_bits(1);

/// Roots of c[0]·z^n + c[1]·z^(n-1) + … + c[n], for n in 1..=3.
/// The caller guarantees c[0] != 0, finite coefficients and n <= 3.
pub(super) fn polynomial_roots(
    descending: &[f64],
) -> Result<Vec<ComplexRoot>, RootComputationError> {
    // Input the caller should not pass is refused.
    if descending.first().is_none_or(|&leading| leading == 0.0)
        || descending
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return Err(RootComputationError);
    }
    if let [a, b] = *descending {
        // Degree 1 is a single division, never rescaled.
        return sorted_output(vec![Root::Real(-b / a)]);
    }
    let mut underflow = Underflow::default();
    let roots = solve(descending, &mut underflow);
    // An error that follows a harmful underflow may be its consequence (a
    // division by an underflowed zero), so it gets the rescaled attempt too.
    // An overflow with no underflow before it stays an error.
    let roots = if underflow.harmful {
        rescaled_roots(descending)?
    } else {
        roots?
    };
    sorted_output(roots)
}

/// One real root, or a conjugate pair `real ± i·imaginary` (two roots) with
/// `imaginary > 0`. Keeping a pair as one value until the output is written
/// guarantees its halves stay adjacent and exact conjugates.
#[derive(Debug, Clone, Copy)]
enum Root {
    Real(f64),
    Pair { real: f64, imaginary: f64 },
}

impl Root {
    /// The root itself, or the pair's +imaginary member.
    fn representative(self) -> ComplexRoot {
        match self {
            Root::Real(real) => ComplexRoot {
                real,
                imaginary: 0.0,
            },
            Root::Pair { real, imaginary } => ComplexRoot { real, imaginary },
        }
    }

    fn is_finite(self) -> bool {
        let ComplexRoot { real, imaginary } = self.representative();
        real.is_finite() && imaginary.is_finite()
    }

    /// A nonzero root with no part in the normal range. Its value has lost
    /// precision, however it was computed. One subnormal part beside a normal
    /// one is harmless: it moves the root by under 2^-1074, far below an ulp
    /// of its modulus.
    fn is_subnormal(self) -> bool {
        let ComplexRoot { real, imaginary } = self.representative();
        (real != 0.0 || imaginary != 0.0) && real.abs().max(imaginary.abs()) < f64::MIN_POSITIVE
    }
}

/// Whether an underflow may have changed a solve's roots. See the module
/// documentation for which underflows count.
#[derive(Debug, Default)]
struct Underflow {
    harmful: bool,
}

impl Underflow {
    /// `left · right`; its underflow is harmful.
    fn product(&mut self, left: f64, right: f64) -> f64 {
        let product = left * right;
        self.harmful |= underflows(product, left, right);
        product
    }

    /// `numerator / denominator`; its underflow is harmful.
    fn quotient(&mut self, numerator: f64, denominator: f64) -> f64 {
        let quotient = numerator / denominator;
        self.harmful |= underflows(quotient, numerator, denominator);
        quotient
    }

    /// `addend + left·right`, the product rounded first; the product's
    /// underflow is harmful only if the sum is below the normal range.
    fn sum_with_product(&mut self, addend: f64, left: f64, right: f64) -> f64 {
        let product = left * right;
        let sum = addend + product;
        self.absorb(underflows(product, left, right), sum);
        sum
    }

    /// Records an underflow in a term of `sum`, harmful only if `sum` is
    /// below the normal range.
    fn absorb(&mut self, term_underflowed: bool, sum: f64) {
        self.harmful |= term_underflowed && sum.abs() < f64::MIN_POSITIVE;
    }
}

/// Whether `result`, the rounded product or quotient of nonzero `left` and
/// `right`, fell below the normal range (to a subnormal or to zero).
fn underflows(result: f64, left: f64, right: f64) -> bool {
    left != 0.0 && right != 0.0 && result.abs() < f64::MIN_POSITIVE
}

/// Roots of a quadratic or a cubic, recording harmful underflow.
fn solve(descending: &[f64], underflow: &mut Underflow) -> Result<Vec<Root>, RootComputationError> {
    let roots = match *descending {
        [a, b, c] => quadratic_roots(a, b, c, underflow)?,
        [a, b, c, d] => cubic_roots([a, b, c, d], underflow)?,
        _ => return Err(RootComputationError),
    };
    underflow.harmful |= roots.iter().any(|root| root.is_subnormal());
    Ok(roots)
}

/// Solves the polynomial again after two exact rescalings: substituting
/// z = 2^s·w, and dividing every coefficient by 2^e, where 2^e is the leading
/// coefficient's binary order of magnitude and s is the smallest integer with
/// |c[k]| < 2^(e + s·k) for every k ≥ 1. Coefficient k is multiplied by
/// 2^(−e − s·k), which is exact while it stays in the normal range. The
/// leading coefficient lands in [1, 2), every other below 1 and at least one
/// above 1/16, so the largest root w is roughly between 0.1 and 2.
fn rescaled_roots(descending: &[f64]) -> Result<Vec<Root>, RootComputationError> {
    let leading_exponent = binary_exponent(descending[0]);
    let root_exponent = descending
        .iter()
        .zip(0..)
        .skip(1)
        .filter(|&(&coefficient, _)| coefficient != 0.0)
        .map(|(&coefficient, power)| {
            ceiling_division(binary_exponent(coefficient) + 1 - leading_exponent, power)
        })
        .max()
        .unwrap_or(0);
    let scaled: Vec<f64> = descending
        .iter()
        .zip(0..)
        .map(|(&coefficient, power)| {
            times_power_of_two(coefficient, -leading_exponent - root_exponent * power)
        })
        .collect();
    // A coefficient pushed below the normal range was not scaled exactly.
    if scaled
        .iter()
        .zip(descending)
        .any(|(&scaled, &original)| original != 0.0 && scaled.abs() < f64::MIN_POSITIVE)
    {
        return Err(RootComputationError);
    }

    let mut underflow = Underflow::default();
    let roots = solve(&scaled, &mut underflow)?;
    if underflow.harmful {
        return Err(RootComputationError);
    }
    // Scaling back is exact unless a root leaves the normal range. Overflow
    // is left to the output's finiteness check; underflow is refused here.
    let mut unscaled = Vec::with_capacity(descending.len() - 1);
    for root in roots {
        let root = match root {
            Root::Real(real) => Root::Real(times_power_of_two(real, root_exponent)),
            Root::Pair { real, imaginary } => Root::Pair {
                real: times_power_of_two(real, root_exponent),
                imaginary: times_power_of_two(imaginary, root_exponent),
            },
        };
        if root.is_subnormal() {
            return Err(RootComputationError);
        }
        match root {
            // As in the quadratic, an imaginary part that underflows to zero
            // leaves a double real root.
            Root::Pair {
                real,
                imaginary: 0.0,
            } => {
                unscaled.extend([Root::Real(real), Root::Real(real)]);
            }
            root => unscaled.push(root),
        }
    }
    Ok(unscaled)
}

/// ⌊log2 |x|⌋ for finite nonzero `x`, subnormals included.
fn binary_exponent(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    match (bits >> 52) as i32 {
        // A subnormal is its significand times 2^-1074.
        0 => 63 - bits.leading_zeros() as i32 - 1074,
        biased => biased - 1023,
    }
}

/// ⌈numerator / denominator⌉ for a positive denominator.
fn ceiling_division(numerator: i32, denominator: i32) -> i32 {
    -(-numerator).div_euclid(denominator)
}

/// `x · 2^exponent`, exact unless the result leaves the normal range.
fn times_power_of_two(x: f64, exponent: i32) -> f64 {
    // Factors of at most 2^±1000 are representable. The value moves
    // monotonically toward its final size, so no step leaves the normal range
    // unless the result does.
    const STEP: i32 = 1000;
    let mut value = x;
    let mut remaining = exponent;
    while remaining > STEP {
        value *= power_of_two(STEP);
        remaining -= STEP;
    }
    while remaining < -STEP {
        value *= power_of_two(-STEP);
        remaining += STEP;
    }
    value * power_of_two(remaining)
}

/// 2^exponent, for an exponent in −1022..=1023.
const fn power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Roots of a·z² + b·z + c by the quadratic formulas: two real roots or one
/// pair.
fn quadratic_roots(
    a: f64,
    b: f64,
    c: f64,
    underflow: &mut Underflow,
) -> Result<Vec<Root>, RootComputationError> {
    let four_a = 4.0 * a;
    let (square, product) = (b * b, four_a * c);
    let discriminant = square - product;
    // The usual harmless underflow is b² underflowing while 4ac is normal.
    underflow.absorb(
        underflows(square, b, b) || underflows(product, four_a, c),
        discriminant,
    );
    if !discriminant.is_finite() {
        return Err(RootComputationError);
    }
    if discriminant >= 0.0 {
        // q = −(b + sign(b)·√D)/2 adds two terms of the same sign, so it never
        // cancels. It gives the larger-magnitude root q/a; the product of the
        // roots, c/a, gives the other one as c/q.
        let sign = if b < 0.0 { -1.0 } else { 1.0 };
        let q = underflow.quotient(-(b + sign * discriminant.sqrt()), 2.0);
        if q == 0.0 {
            return Ok(vec![Root::Real(0.0), Root::Real(0.0)]);
        }
        Ok(vec![
            Root::Real(underflow.quotient(q, a)),
            Root::Real(underflow.quotient(c, q)),
        ])
    } else {
        let real = -b / (2.0 * a);
        let imaginary = (-discriminant).sqrt() / (2.0 * a.abs());
        // Checked before a zero imaginary part turns the pair into two reals.
        underflow.harmful |= Root::Pair { real, imaginary }.is_subnormal();
        // An imaginary part that underflows to zero leaves a double real root.
        if imaginary == 0.0 {
            return Ok(vec![Root::Real(real), Root::Real(real)]);
        }
        Ok(vec![Root::Pair { real, imaginary }])
    }
}

/// Roots of a cubic: one real root from the closed form, polished, deflated
/// out; the quadratic's real roots polished again on the original cubic.
fn cubic_roots(
    cubic: [f64; 4],
    underflow: &mut Underflow,
) -> Result<Vec<Root>, RootComputationError> {
    let start = closed_form_real_root(&cubic, underflow)?;
    let first = polish_real_root(&cubic, start, underflow);

    // Synthetic division by (z − first), dropping the remainder. Working down
    // from the leading coefficient multiplies each rounding error by |first|;
    // working up from the constant divides it by |first|. So divide from the
    // constant end when |first| exceeds the geometric mean of the other two
    // roots' moduli, √|d/(a·first)|, that is when |first|³ > |d/a|. This is
    // the deflation-direction choice of the QBC step in W. Kahan, "To Solve a
    // Real Cubic Equation" (1986).
    let [a, b, c, d] = cubic;
    let magnitude = first.abs();
    let square = magnitude * magnitude;
    let cube = square * magnitude;
    let ratio = d / a;
    // Underflow on one side of the comparison cannot flip it, since the other
    // side is normal and so the larger. On both sides it decides nothing.
    underflow.harmful |= (underflows(square, magnitude, magnitude)
        || underflows(cube, square, magnitude))
        && underflows(ratio, d, a);
    let (linear, constant) = if cube > ratio.abs() {
        let constant = underflow.quotient(-d, first);
        (underflow.quotient(constant - c, first), constant)
    } else {
        let linear = underflow.sum_with_product(b, first, a);
        (linear, underflow.sum_with_product(c, first, linear))
    };

    let mut roots = vec![Root::Real(first)];
    for root in quadratic_roots(a, linear, constant, underflow)? {
        roots.push(match root {
            Root::Real(start) => Root::Real(polish_real_root(&cubic, start, underflow)),
            pair => pair,
        });
    }
    Ok(roots)
}

/// One real root of the cubic from the closed-form solution of its monic
/// form z³ + b·z² + c·z + d, depressed by z = t − b/3 to t³ + p·t + q.
fn closed_form_real_root(
    cubic: &[f64; 4],
    underflow: &mut Underflow,
) -> Result<f64, RootComputationError> {
    let [leading, b, c, d] = *cubic;
    let b = underflow.quotient(b, leading);
    let c = underflow.quotient(c, leading);
    let d = underflow.quotient(d, leading);
    let shift = underflow.quotient(b, 3.0);
    // p = c − b·shift and q = (2·shift² − c)·shift + d.
    let p = underflow.sum_with_product(c, -b, shift);
    let curvature = underflow.sum_with_product(-c, 2.0 * shift, shift);
    let q = underflow.sum_with_product(d, curvature, shift);
    let half_q = underflow.quotient(q, 2.0);
    let third_p = underflow.quotient(p, 3.0);
    let q_term = half_q * half_q;
    let p_square = third_p * third_p;
    let p_term = p_square * third_p;
    let discriminant = q_term + p_term;
    underflow.absorb(
        underflows(q_term, half_q, half_q)
            || underflows(p_square, third_p, third_p)
            || underflows(p_term, p_square, third_p),
        discriminant,
    );
    if !discriminant.is_finite() {
        return Err(RootComputationError);
    }

    let t = if discriminant < 0.0 {
        // Three distinct real roots (so p < 0): t = 2m·cos θ with
        // m = √(−p/3) and cos 3θ = −q/(2m³). Take the root of largest |t|,
        // sign(−q)·2m·cos(acos(|q|/(2m³))/3): it is the one farthest from the
        // other two, so it stays simple even when the other two nearly merge.
        // The ratio may underflow harmlessly: it moves acos's argument near 0
        // by under 2^-1074, and t by as little relative to m.
        let m = (-third_p).sqrt();
        let cos_triple_angle = (half_q.abs() / underflow.product(-third_p, m)).min(1.0);
        let magnitude = 2.0 * m * (cos_triple_angle.acos() / 3.0).cos();
        if q > 0.0 { -magnitude } else { magnitude }
    } else {
        // One real root (or a multiple root): Cardano's t = u + v with
        // u³ = −q/2 − sign(q)·√Δ, choosing the sign that adds magnitudes, and
        // v = −p/(3u) from u·v = −p/3. A nonzero |u| is at least the cube
        // root of the smallest subnormal, about 1.6e-108, so p/(3u) can only
        // underflow into a normal sum.
        let u_magnitude = (half_q.abs() + discriminant.sqrt()).cbrt();
        if u_magnitude == 0.0 {
            // p = q = 0: the triple root t = 0.
            0.0
        } else {
            let u = if q > 0.0 { -u_magnitude } else { u_magnitude };
            u - third_p / u
        }
    };
    let root = t - shift;
    if !root.is_finite() {
        return Err(RootComputationError);
    }
    Ok(root)
}

/// Newton's method on the cubic from `start`, keeping each step only while
/// it strictly reduces |p(x)|, for at most `MAX_NEWTON_STEPS` steps. The root
/// stays finite: a nonfinite candidate never reduces |p(x)|.
fn polish_real_root(cubic: &[f64; 4], start: f64, underflow: &mut Underflow) -> f64 {
    let mut root = start;
    let (mut value, mut slope, mut underflowed) = value_and_slope(cubic, root);
    for _ in 0..MAX_NEWTON_STEPS {
        if value == 0.0 || slope == 0.0 {
            break;
        }
        let candidate = root - value / slope;
        let (candidate_value, candidate_slope, candidate_underflowed) =
            value_and_slope(cubic, candidate);
        underflowed |= candidate_underflowed;
        // A NaN or infinite candidate fails this comparison too.
        if candidate_value.abs() < value.abs() {
            root = candidate;
            value = candidate_value;
            slope = candidate_slope;
        } else {
            break;
        }
    }
    // Underflow perturbs each computed p(x) by at most
    // UNDERFLOW_NOISE·max(1, |x|)³, which can stop Newton up to that much
    // over |p'| away from the root. It is harmful when that distance could
    // exceed half an ulp of the root.
    let reach = root.abs().max(1.0);
    underflow.harmful |= underflowed
        && UNDERFLOW_NOISE * reach * reach * reach > 0.5 * f64::EPSILON * (root * slope).abs();
    root
}

/// p(x) and p'(x) by Horner's scheme, and whether an underflow may have
/// perturbed p(x). The value is compensated: each multiply's and add's
/// rounding error is recovered exactly (an fma for the product, a two-sum
/// for the addition) and carried along by a second Horner pass, so p(x) is
/// about as accurate as if evaluated in twice the precision. Near a close
/// pair of roots, plain Horner's rounding noise is larger than |p| itself,
/// and Newton would wander on it; the slope needs no such care.
fn value_and_slope(cubic: &[f64; 4], x: f64) -> (f64, f64, bool) {
    let mut value = cubic[0];
    let mut value_error = 0.0;
    let mut slope = 0.0;
    let mut underflowed = false;
    for &coefficient in &cubic[1..] {
        slope = slope * x + value;
        let product = value * x;
        let product_error = value.mul_add(x, -product);
        let carried_error = value_error * x;
        underflowed |= (value != 0.0 && x != 0.0 && product.abs() < EXACT_PRODUCT_ERROR_FLOOR)
            || underflows(carried_error, value_error, x);
        let (sum, sum_error) = two_sum(product, coefficient);
        value = sum;
        value_error = carried_error + (product_error + sum_error);
    }
    (value + value_error, slope, underflowed)
}

/// `left + right` and the exact rounding error of that addition.
fn two_sum(left: f64, right: f64) -> (f64, f64) {
    let sum = left + right;
    let right_part = sum - left;
    let left_part = sum - right_part;
    (sum, (left - left_part) + (right - right_part))
}

/// The output rules: finite roots, +0.0 imaginary parts on real
/// roots, conjugate pairs adjacent with +imaginary first, and ascending
/// modulus with ties broken by ascending real part, then descending
/// imaginary part.
fn sorted_output(roots: Vec<Root>) -> Result<Vec<ComplexRoot>, RootComputationError> {
    if roots.iter().any(|root| !root.is_finite()) {
        return Err(RootComputationError);
    }
    // A pair sorts by its +imaginary member, which is where its first root
    // belongs.
    let mut keyed: Vec<_> = roots
        .into_iter()
        .map(|root| (sort_modulus(root.representative()), root))
        .collect();
    keyed.sort_by(|(left_modulus, left), (right_modulus, right)| {
        let (left, right) = (left.representative(), right.representative());
        left_modulus
            .total_cmp(right_modulus)
            .then(left.real.total_cmp(&right.real))
            .then(right.imaginary.total_cmp(&left.imaginary))
    });

    let mut output = Vec::with_capacity(3);
    for (_, root) in keyed {
        // Adding +0.0 turns a −0.0 real part into +0.0 and leaves any other
        // value unchanged; `details.rs` would read a −0.0 real root through
        // atan2 as frequency 1/2.
        match root {
            Root::Real(real) => output.push(ComplexRoot {
                real: real + 0.0,
                imaginary: 0.0,
            }),
            Root::Pair { real, imaginary } => {
                let real = real + 0.0;
                output.push(ComplexRoot { real, imaginary });
                output.push(ComplexRoot {
                    real,
                    imaginary: -imaginary,
                });
            }
        }
    }
    Ok(output)
}

/// The modulus a root sorts by. It is `checked_modulus`, the modulus the
/// consumers compare, wherever the squares stay in the normal range. A finite
/// root whose squared modulus overflows has none and sorts after every other
/// root. For a root below 2^-500, whose squares would underflow, the same
/// arithmetic runs on the root scaled by 2^600 and the result is scaled back,
/// so tiny roots keep their true order instead of all tying at 0.
fn sort_modulus(root: ComplexRoot) -> f64 {
    const TINY: f64 = power_of_two(-500);
    const SCALE: i32 = 600;
    if root.real.abs().max(root.imaginary.abs()) < TINY {
        let scaled = ComplexRoot {
            real: times_power_of_two(root.real, SCALE),
            imaginary: times_power_of_two(root.imaginary, SCALE),
        };
        return scaled
            .checked_modulus()
            .map_or(f64::INFINITY, |modulus| times_power_of_two(modulus, -SCALE));
    }
    root.checked_modulus().unwrap_or(f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulp(value: f64) -> f64 {
        value.abs().next_up() - value.abs()
    }

    fn real(value: f64) -> ComplexRoot {
        ComplexRoot {
            real: value,
            imaginary: 0.0,
        }
    }

    fn bits(roots: &[ComplexRoot]) -> Vec<(u64, u64)> {
        roots
            .iter()
            .map(|root| (root.real.to_bits(), root.imaginary.to_bits()))
            .collect()
    }

    /// Product of two real polynomials in descending order.
    fn multiply(left: &[f64], right: &[f64]) -> Vec<f64> {
        let mut product = vec![0.0; left.len() + right.len() - 1];
        for (i, l) in left.iter().enumerate() {
            for (j, r) in right.iter().enumerate() {
                product[i + j] += l * r;
            }
        }
        product
    }

    /// Multiplies out `leading · Π (z − root)` into descending complex
    /// coefficients `(real, imaginary)`.
    fn expand(leading: f64, roots: &[ComplexRoot]) -> Vec<(f64, f64)> {
        let mut product = vec![(leading, 0.0)];
        for root in roots {
            let mut next = product.clone();
            next.push((0.0, 0.0));
            for k in 1..next.len() {
                let (real, imaginary) = product[k - 1];
                next[k].0 -= root.real * real - root.imaginary * imaginary;
                next[k].1 -= root.real * imaginary + root.imaginary * real;
            }
            product = next;
        }
        product
    }

    fn sample_polynomials() -> Vec<Vec<f64>> {
        vec![
            vec![3.0, 1.0],
            vec![1.0, -1.0e8, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![-1.0, 1.0, -1.0],
            vec![1.0, -1.0, 0.25],
            vec![0.125, 0.25, -1.0],
            vec![1.0, -0.4, 0.25, -0.63],
            vec![0.03125, -0.25, 0.75, -1.0],
            vec![1.0, 3.0, 3.0, 1.0 - 1e-12],
            vec![1.0, 0.0, 0.0, -1.0],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![-1.0, 1.0, 1.0, -1.0],
            vec![1.0, -3.0, -1.0, 3.0],
        ]
    }

    #[test]
    fn linear_root_is_one_division() {
        for (leading, constant) in [
            (3.0, 1.0),
            (0.1, 0.7),
            (-0.3, 2.0),
            (1e-30, 1e-10),
            (1e-300, 1e-10),
        ] {
            let roots = polynomial_roots(&[leading, constant]).unwrap();
            assert_eq!(roots.len(), 1);
            assert_eq!(roots[0].real.to_bits(), (-constant / leading).to_bits());
            assert_eq!(roots[0].imaginary.to_bits(), 0);
        }
    }

    #[test]
    fn quadratic_roots_avoid_cancellation() {
        let roots = polynomial_roots(&[1.0, -1.0e8, 1.0]).unwrap();
        assert_eq!(roots.len(), 2);
        let small = roots[0].real;
        assert!(
            (small - 1e-8).abs() <= ulp(1e-8),
            "small root {small:e} is not within 1 ulp of 1e-8"
        );
        assert!((roots[1].real - 1e8).abs() <= ulp(1e8));
    }

    #[test]
    fn complex_pairs_are_exact_adjacent_conjugates() {
        for coefficients in [
            &[1.0, 0.0, 1.0][..],
            &[1.0, 1.0, 1.0],
            &[-1.0, 1.0, -1.0],
            &[1.0, -0.4, 0.25, -0.63],
            &[0.03125, -0.25, 0.75, -1.0],
            &[1.0, 3.0, 3.0, 1.0 - 1e-12],
            &[1.0, 0.0, 0.0, -1.0],
            &[1.0, -3.0, 1.0, -3.0],
        ] {
            let roots = polynomial_roots(coefficients).unwrap();
            assert_eq!(roots.len(), coefficients.len() - 1);
            let mut pairs = 0;
            let mut index = 0;
            while index < roots.len() {
                let root = roots[index];
                if root.imaginary != 0.0 {
                    assert!(
                        root.imaginary > 0.0,
                        "{coefficients:?}: negative imaginary first"
                    );
                    let conjugate = roots[index + 1];
                    assert_eq!(conjugate.real.to_bits(), root.real.to_bits());
                    assert_eq!(conjugate.imaginary.to_bits(), (-root.imaginary).to_bits());
                    pairs += 1;
                    index += 1;
                }
                index += 1;
            }
            assert_eq!(pairs, 1, "{coefficients:?} should have one complex pair");
        }
    }

    #[test]
    fn real_roots_have_positive_zero_imaginary() {
        for coefficients in [
            &[1.0, 2.0][..],
            &[-2.0, -1.0],
            &[1.0, 0.0],
            &[1.0, 3.0, 2.0],
            &[1.0, 1.0, 0.0],
            &[-1.0, 0.0, 1.0],
            &[1.0, -3.0, -1.0, 3.0],
            &[1.0, 6.0, 11.0, 6.0],
            &[1.0, 1.0, -1.0, -1.0],
            &[-1.0, 1.0, 1.0, -1.0],
            &[1.0, -0.4, 0.25, -0.63],
        ] {
            for root in polynomial_roots(coefficients).unwrap() {
                if root.imaginary == 0.0 {
                    assert_eq!(root.imaginary.to_bits(), 0, "{coefficients:?}");
                    assert_ne!(root.real.to_bits(), (-0.0_f64).to_bits());
                }
            }
        }
        // Polynomials whose roots are all real and well separated.
        for coefficients in [&[1.0, 3.0, 2.0][..], &[1.0, 6.0, 11.0, 6.0]] {
            let roots = polynomial_roots(coefficients).unwrap();
            assert!(roots.iter().all(|root| root.imaginary.to_bits() == 0));
        }
    }

    #[test]
    fn double_root_is_stable() {
        // (z − 0.5)² and every 1-ulp neighbour of each coefficient.
        let base: [f64; 3] = [1.0, -1.0, 0.25];
        let mut cases = vec![base];
        for index in 0..base.len() {
            for neighbour in [base[index].next_up(), base[index].next_down()] {
                let mut coefficients = base;
                coefficients[index] = neighbour;
                cases.push(coefficients);
            }
        }
        for coefficients in cases {
            let first = polynomial_roots(&coefficients).unwrap();
            assert_eq!(first.len(), 2);
            let is_pair = first[0].imaginary != 0.0;
            for _ in 0..4 {
                let again = polynomial_roots(&coefficients).unwrap();
                assert_eq!(again[0].imaginary != 0.0, is_pair, "{coefficients:?}");
                assert_eq!(bits(&again), bits(&first), "{coefficients:?}");
            }
            let expanded = expand(coefficients[0], &first);
            for (&(real, imaginary), wanted) in expanded.iter().zip(coefficients) {
                assert!(
                    (real - wanted).abs() <= 4.0 * ulp(wanted),
                    "{coefficients:?}: reconstructed {real:e}, wanted {wanted:e}"
                );
                assert!(imaginary.abs() <= 4.0 * ulp(wanted), "{coefficients:?}");
            }
        }
    }

    #[test]
    fn cubic_roots_reconstruct() {
        let cases = [
            multiply(&[1.0, -0.9], &[1.0, 0.5, 0.7]),
            multiply(&multiply(&[1.0, -0.5], &[1.0, -0.500_000_1]), &[1.0, 2.0]),
            vec![1.0, 3.0, 3.0, 1.0 - 1e-12],
        ];
        for coefficients in cases {
            let roots = polynomial_roots(&coefficients).unwrap();
            assert_eq!(roots.len(), 3);
            let largest = coefficients.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
            let tolerance = 1e-14 * largest;
            let expanded = expand(coefficients[0], &roots);
            for (&(real, imaginary), wanted) in expanded.iter().zip(&coefficients) {
                assert!(
                    (real - wanted).abs() <= tolerance,
                    "{coefficients:?}: reconstructed {real:e}, wanted {wanted:e}, \
                     roots {roots:?}"
                );
                assert!(imaginary.abs() <= tolerance, "{coefficients:?}: {roots:?}");
            }
        }
    }

    #[test]
    fn order_is_ascending_modulus() {
        // Equal moduli break ties by ascending real part.
        let roots = polynomial_roots(&[1.0, 0.0, -1.0]).unwrap();
        assert_eq!(bits(&roots), bits(&[real(-1.0), real(1.0)]));
        let roots = polynomial_roots(&[1.0, -3.0, -1.0, 3.0]).unwrap();
        assert_eq!(bits(&roots), bits(&[real(-1.0), real(1.0), real(3.0)]));

        // A real root inside a complex pair, and a complex pair inside a real root.
        let roots = polynomial_roots(&[1.0, -0.5, 4.0, -2.0]).unwrap();
        assert!((roots[0].real - 0.5).abs() < 1e-15 && roots[0].imaginary == 0.0);
        assert!((roots[1].imaginary - 2.0).abs() < 1e-15);
        assert!((roots[2].imaginary + 2.0).abs() < 1e-15);
        let roots = polynomial_roots(&[1.0, -3.0, 1.0, -3.0]).unwrap();
        assert!((roots[0].imaginary - 1.0).abs() < 1e-15);
        assert!((roots[1].imaginary + 1.0).abs() < 1e-15);
        assert!((roots[2].real - 3.0).abs() < 1e-15 && roots[2].imaginary == 0.0);

        // Every output is sorted by (modulus, real ascending, imaginary descending).
        for coefficients in sample_polynomials() {
            let roots = polynomial_roots(&coefficients).unwrap();
            for window in roots.windows(2) {
                let (left, right) = (window[0], window[1]);
                let order = left
                    .checked_modulus()
                    .unwrap()
                    .total_cmp(&right.checked_modulus().unwrap())
                    .then(left.real.total_cmp(&right.real))
                    .then(right.imaginary.total_cmp(&left.imaginary));
                assert!(order.is_le(), "{coefficients:?}: {roots:?}");
            }
        }
    }

    #[test]
    fn overflow_is_an_error() {
        assert_eq!(
            polynomial_roots(&[-1.0, -1e308, -1.0]).unwrap_err(),
            RootComputationError
        );
        assert_eq!(
            polynomial_roots(&[1e-300, 1e10]).unwrap_err(),
            RootComputationError
        );
    }

    #[test]
    fn overflowing_modulus_sorts_last() {
        // The roots are 1e-150 and -1e160, whose squared modulus overflows.
        assert_eq!(
            super::super::polynomial::has_strictly_inside_root(&[1e150, 1e-10]),
            Ok(true)
        );
        let roots = polynomial_roots(&[1e-10, 1e150, -1.0]).unwrap();
        assert_eq!(bits(&roots), bits(&[real(1e-150), real(-1e160)]));
    }

    #[test]
    fn harmless_underflow_changes_nothing() {
        // b² underflows while 4ac is normal: the plain formulas, unscaled.
        let roots = polynomial_roots(&[1.0, 1e-170, 1.0]).unwrap();
        let pair = [
            ComplexRoot {
                real: -5e-171,
                imaginary: 1.0,
            },
            ComplexRoot {
                real: -5e-171,
                imaginary: -1.0,
            },
        ];
        assert_eq!(bits(&roots), bits(&pair));
    }

    #[test]
    fn underflowing_polynomials_are_rescaled_exactly() {
        // The true roots of each polynomial (its coefficients taken as exact
        // binary values), correctly rounded, in the solver's output order:
        // computed by Durand–Kerner in 90-digit decimal arithmetic and checked
        // against the coefficients. Unscaled, each of these underflows in a
        // way that corrupts its roots.
        type Case = (&'static [f64], &'static [(f64, f64)]);
        const HALF_ROOT_THREE: f64 = 0.866_025_403_784_438_6;
        let cases: [Case; 9] = [
            (
                &[1.0, 0.0, 1e-110, 1e-170],
                &[
                    (-9.999_999_998_999_999e-61, 0.0),
                    (4.999_999_999_499_999_5e-61, 1.000_000_000_037_500_1e-55),
                    (4.999_999_999_499_999_5e-61, -1.000_000_000_037_500_1e-55),
                ],
            ),
            (
                &[1.0, 1e-200, 0.0, 0.0],
                &[(0.0, 0.0), (0.0, 0.0), (-1e-200, 0.0)],
            ),
            (&[1e-200, -3e-200, 2e-200], &[(1.0, 0.0), (2.0, 0.0)]),
            (&[1e-200, 0.0, 1e-200], &[(0.0, 1.0), (0.0, -1.0)]),
            (&[1e-200, 0.0, -1e-200], &[(-1.0, 0.0), (1.0, 0.0)]),
            (
                &[1e-310, 1e-310, 1e-310],
                &[(-0.5, HALF_ROOT_THREE), (-0.5, -HALF_ROOT_THREE)],
            ),
            (
                &[1e-200, -6e-200, 11e-200, -6e-200],
                &[
                    (1.000_000_000_000_000_2, 0.0),
                    (1.999_999_999_999_999, 0.0),
                    (3.000_000_000_000_001, 0.0),
                ],
            ),
            (
                &[1.0, -6e-100, 11e-200, -6e-300],
                &[
                    (1.000_000_000_000_000_7e-100, 0.0),
                    (1.999_999_999_999_997_5e-100, 0.0),
                    (3.000_000_000_000_002e-100, 0.0),
                ],
            ),
            // Reachable from the wrapper; unscaled, a spurious double root
            // replaced the pair ±7.6e-64.
            (
                &[-1.25e211, -2.35e127, 7.3e84, -1.0],
                &[
                    (1.369_863_013_698_630_2e-85, 0.0),
                    (-7.641_989_269_817_121e-64, 0.0),
                    (7.641_989_269_817_121e-64, 0.0),
                ],
            ),
        ];
        for (coefficients, expected) in cases {
            let mut underflow = Underflow::default();
            let _ = solve(coefficients, &mut underflow);
            assert!(underflow.harmful, "{coefficients:?} does not underflow");

            // Err would also be acceptable here; every one of these is solved.
            let roots = polynomial_roots(coefficients).unwrap();
            assert_eq!(roots.len(), expected.len(), "{coefficients:?}");
            for (root, &(real, imaginary)) in roots.iter().zip(expected) {
                let error = (root.real - real).hypot(root.imaginary - imaginary);
                assert!(
                    error <= 1e-14 * real.hypot(imaginary),
                    "{coefficients:?}: {root:?}, wanted ({real:e}, {imaginary:e})"
                );
            }
        }
    }

    #[test]
    fn close_and_spread_roots_reconstruct_each_coefficient() {
        for coefficients in [
            // (z − 0.5)(z − 0.500001)(z − 2)
            [1.0, -3.000_001, 2.250_002_500_000_000_3, -0.500_001],
            // (z − 1e6)(z² + 0.5z + 0.7)
            [1.0, -999_999.5, -499_999.3, -700_000.0],
        ] {
            let roots = polynomial_roots(&coefficients).unwrap();
            let expanded = expand(coefficients[0], &roots);
            for (&(real, imaginary), wanted) in expanded.iter().zip(coefficients) {
                assert!(
                    (real - wanted).abs() <= 1e-14 * wanted.abs()
                        && imaginary.abs() <= 1e-14 * wanted.abs(),
                    "{coefficients:?}: reconstructed ({real:e}, {imaginary:e}), \
                     wanted {wanted:e}, roots {roots:?}"
                );
            }
        }
    }
}
