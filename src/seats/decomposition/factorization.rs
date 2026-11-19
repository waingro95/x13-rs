//! MAK1 (`ansub2.f`): the invertible moving-average factor of a component's
//! autocovariance numerator, with its innovation variance.
//!
//! MAK1 reduces the symmetric numerator with SYMPOLY and takes its roots with
//! RPQ. ROOTC and SQROOTC turn each into a reciprocal pair of roots in B,
//! JoinRoot merges the single roots on the unit circle, HalfRoots keeps the
//! half on or inside it, and MPBC multiplies the kept factors out.

use super::canonical::CanonicalPreparation;
use super::harmonic::conjugate;
use crate::numerics::BUILD_62_PI;
use crate::seats::failure::{Error, Failure, Stage};
use crate::seats::roots::{
    MAK1_MAX_COEFFICIENTS, REAL_ROOT_PERIOD, polynomial_roots, reduce_symmetric,
};

const MATCH_EPSILON: f64 = 1e-30;
const UNIT_EPSILON: f64 = 1e-13;
const ADJACENT_EPSILON: f64 = 1e-10;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaFactor {
    pub(super) coefficients: Vec<f64>,
    pub(crate) variance: f64,
}

impl MaFactor {
    /// DecompSpectrum's initial THETA=[1] with zero variance, kept by a
    /// component whose MAK1 call DecompSpectrum skips.
    pub(super) fn absent() -> Self {
        Self {
            coefficients: vec![1.0],
            variance: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Factorization {
    pub(crate) preparation: CanonicalPreparation,
    pub(super) trend: MaFactor,
    pub(super) seasonal: MaFactor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Complex {
    real: f64,
    imaginary: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Root {
    value: Complex,
    modulus: f64,
    argument_degrees: f64,
    period: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Group {
    root: Root,
    count: usize,
}

/// MAK1 on the lags 0..n of a symmetric numerator: the factor's coefficients
/// in B and its innovation variance.
pub(super) fn factor_numerator(numerator: &[f64]) -> Result<MaFactor, Error> {
    // The upper bound is MAK1's own array bound; a seasonal AR with three
    // seasonal regular roots reaches 26 coefficients.
    if !(2..=MAK1_MAX_COEFFICIENTS).contains(&numerator.len()) {
        return Err(Failure::Unusable {
            quantity: "spectral numerator coefficient count",
            index: None,
            value: numerator.len() as f64,
        }
        .at(Stage::CanonicalDecomposition));
    }
    if let Some(index) = numerator.iter().position(|v| !v.is_finite()) {
        return Err(Failure::Unusable {
            quantity: "spectral numerator coefficient",
            index: Some(index),
            value: numerator[index],
        }
        .at(Stage::CanonicalDecomposition));
    }
    // MAK1 reverses all nonzero lags, then doubles lag zero before SYMPOLY.
    let mut poly: Vec<_> = numerator[1..].iter().rev().copied().collect();
    poly.push(numerator[0] + numerator[0]);
    check_finite(&poly, "input")?;
    let (real, imaginary) = if numerator.len() == 2 {
        // With two coefficients MAK1 bypasses SYMPOLY and RPQ and supplies
        // the single root directly.
        let root = -poly[1] / poly[0];
        if !root.is_finite() {
            return Ok(unselected_direct_root(numerator[0]));
        }
        (vec![root], vec![0.0])
    } else {
        let output = polynomial_roots(&reduce_symmetric(&poly))?;
        (output.real, output.imaginary)
    };
    let mut roots = Vec::with_capacity(2 * real.len());
    for (&real, &imaginary) in real.iter().zip(&imaginary) {
        for value in reciprocal_roots(Complex {
            real: -real,
            imaginary: -imaginary,
        }) {
            // Unlike ROOTC/SQROOTC, MAK1's imaginary square is rounded first,
            // then its real square is fused into the sum.
            let modulus = value
                .real
                .mul_add(value.real, value.imaginary * value.imaginary)
                .sqrt();
            let mut argument = if modulus < 1e-8 {
                0.0
            } else {
                value.real / modulus
            };
            if argument.abs() <= 1.0 {
                argument = argument.acos();
                if value.imaginary < 0.0 {
                    argument = -argument;
                }
            } else {
                argument = if value.real < 0.0 { BUILD_62_PI } else { 0.0 };
            }
            let period = if value.imaginary > 1e-5 || value.imaginary < -1e-5 {
                2.0 * BUILD_62_PI / argument
            } else {
                REAL_ROOT_PERIOD
            };
            let root = Root {
                value,
                modulus,
                argument_degrees: 180.0 * argument / BUILD_62_PI,
                period,
            };
            check_root(root, "reciprocal", roots.len())?;
            roots.push(root);
        }
    }
    let grouped = group_roots(&roots)?;
    let selected = half_roots(&grouped, numerator.len() - 1)?;
    // The product starts as the first factor itself, not as [1] times it:
    // MPBC's +0 accumulators would turn a -0 coefficient into +0.
    let mut product = Vec::with_capacity(selected.len() + 1);
    product.extend(monic_factor(&selected[0]));
    let mut scratch = Vec::with_capacity(selected.len() + 1);
    for root in &selected[1..] {
        multiply_into(&product, &monic_factor(root), &mut scratch);
        std::mem::swap(&mut product, &mut scratch);
    }
    for (index, value) in product.iter().enumerate() {
        if !(value.real.is_finite() && value.imaginary.is_finite()) {
            return Err(unusable_roots("product", Some(index)));
        }
    }
    let coefficients: Vec<_> = product.iter().map(|value| value.real).collect();
    let squared_norm = coefficients
        .iter()
        .fold(0.0, |norm, &value| value.mul_add(value, norm));
    let variance = numerator[0] / squared_norm;
    if !variance.is_finite() {
        return Err(unusable_roots("variance", None));
    }
    // MAK1's TOTERR, the squared error of the factor against the numerator.
    let reconstructed = conjugate(&coefficients, &coefficients);
    let mut reconstruction_error = 0.0;
    for (index, (&value, &target)) in reconstructed.iter().zip(numerator).enumerate() {
        let residual = value.mul_add(variance, -target);
        reconstruction_error = residual.mul_add(residual, reconstruction_error);
        if !reconstruction_error.is_finite() {
            return Err(unusable_roots("reconstruction", Some(index)));
        }
    }
    Ok(MaFactor {
        coefficients,
        variance,
    })
}

/// MAK1 with NUFIN=2 and a zero lag-one ACF (the transitory spectrum is
/// exactly zero). Build 62 divides by zero there; ROOTC's 2*REZ*IMZ is then
/// ∞·0 or NaN, so both reciprocal roots are NaN, every HalfRoots comparison
/// on them is false and IA=0. THETA then keeps the THETA(1)=1 DecompSpectrum
/// gave it and VAR=GAMZER/1. That is the white-noise factor of the ACF, so it
/// is a branch here instead of a nonfinite root carried through.
fn unselected_direct_root(gamzer: f64) -> MaFactor {
    MaFactor {
        coefficients: vec![1.0],
        variance: gamzer / 1.0,
    }
}

fn unusable_roots(step: &'static str, index: Option<usize>) -> Error {
    Failure::UnusableRoots { step, index }.at(Stage::CanonicalDecomposition)
}

fn check_finite(values: &[f64], step: &'static str) -> Result<(), Error> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(unusable_roots(step, Some(index)));
    }
    Ok(())
}

fn check_root(root: Root, step: &'static str, index: usize) -> Result<(), Error> {
    let finite = [
        root.value.real,
        root.value.imaginary,
        root.modulus,
        root.argument_degrees,
        root.period,
    ]
    .iter()
    .all(|value| value.is_finite());
    if finite {
        Ok(())
    } else {
        Err(unusable_roots(step, Some(index)))
    }
}

// SQROOTC: all squares and the norm addition are unfused. Both signed
// imaginary zeros choose the negative candidate in the negative-real branch.
fn square_root(value: Complex) -> Complex {
    let norm = (value.real * value.real + value.imaginary * value.imaginary).sqrt();
    if value.real >= 0.0 {
        let real = ((value.real + norm) * 0.5).sqrt();
        let imaginary = if real.abs() < 1e-8 {
            0.0
        } else {
            value.imaginary / (real + real)
        };
        Complex { real, imaginary }
    } else {
        let magnitude = ((value.real.abs() + norm) * 0.5).sqrt();
        let imaginary = if value.imaginary > 0.0 {
            magnitude
        } else {
            -magnitude
        };
        let real = if imaginary.abs() < 1e-8 {
            0.0
        } else {
            value.imaginary / (imaginary + imaginary)
        };
        Complex { real, imaginary }
    }
}

fn reciprocal_roots(value: Complex) -> [Complex; 2] {
    let delta = value.real * value.real - value.imaginary * value.imaginary - 4.0;
    let delta_imaginary = (value.real + value.real) * value.imaginary;
    let square = square_root(Complex {
        real: delta,
        imaginary: delta_imaginary,
    });
    [
        Complex {
            real: (square.real - value.real) * 0.5,
            imaginary: (square.imaginary - value.imaginary) * 0.5,
        },
        Complex {
            real: (-value.real - square.real) * 0.5,
            imaginary: (-value.imaginary - square.imaginary) * 0.5,
        },
    ]
}

fn matches(value: Complex, target: Complex) -> bool {
    (value.real - target.real).abs() <= MATCH_EPSILON
        && (value.imaginary - target.imaginary).abs() <= MATCH_EPSILON
}

fn get_root(groups: &[Group], target: Complex) -> Option<usize> {
    groups
        .iter()
        .position(|group| matches(group.root.value, target))
}

fn get_conjugate(roots: &[Root], start: usize, target: Complex) -> Option<usize> {
    (start..roots.len())
        .rev()
        .find(|&index| matches(roots[index].value, target))
}

fn closest_root(groups: &[Group], start: usize, target: Complex) -> Option<usize> {
    let mut closest = None;
    let mut minimum = 1e10;
    for (index, group) in groups.iter().enumerate().skip(start) {
        let real = group.root.value.real - target.real;
        let imaginary = group.root.value.imaginary - target.imaginary;
        // The inlined closestRoot rounds imaginary squared, then fuses real.
        let distance = real.mul_add(real, imaginary * imaginary);
        if distance < minimum {
            closest = Some(index);
            minimum = distance;
        }
    }
    closest
}

fn group_roots(roots: &[Root]) -> Result<Vec<Group>, Error> {
    let mut groups = Vec::new();
    for (index, &root) in roots.iter().enumerate() {
        check_root(root, "grouping", index)?;
        if let Some(found) = get_root(&groups, root.value) {
            groups[found].count += 1;
        } else if root.value.imaginary.abs() > MATCH_EPSILON {
            let target = Complex {
                real: root.value.real,
                imaginary: -root.value.imaginary,
            };
            let Some(partner) = get_conjugate(roots, index + 1, target) else {
                return Err(unusable_roots("grouping", None));
            };
            groups.push(Group { root, count: 1 });
            groups.push(Group {
                root: roots[partner],
                count: 0,
            });
        } else {
            let mut root = root;
            root.value.imaginary = 0.0;
            groups.push(Group { root, count: 1 });
        }
    }
    let mut index = 0;
    while index + 1 < groups.len() {
        if groups[index].count == 1 && (groups[index].root.modulus - 1.0).abs() < UNIT_EPSILON {
            let Some(partner) = closest_root(&groups, index + 1, groups[index].root.value) else {
                return Err(unusable_roots("grouping", None));
            };
            join_root(&mut groups, index, partner)?;
        }
        index += 1;
    }
    Ok(groups)
}

fn join_root(groups: &mut Vec<Group>, index: usize, partner: usize) -> Result<(), Error> {
    let left = groups[index];
    let right = groups[partner];
    let count = left.count + right.count;
    let combined_modulus = left.root.modulus * right.root.modulus;
    // JoinRoot rounds the partner product before fusing the current product.
    let mut real = left.root.value.real.mul_add(
        left.count as f64,
        right.root.value.real * right.count as f64,
    ) / count as f64;
    let mut imaginary = left.root.value.imaginary.mul_add(
        left.count as f64,
        right.root.value.imaginary * right.count as f64,
    ) / count as f64;
    let norm = real.mul_add(real, imaginary * imaginary);
    let scale = (combined_modulus / norm).sqrt();
    real *= scale;
    imaginary *= scale;
    let argument_degrees = if real > 0.0 {
        (imaginary / real).atan() * 180.0 / BUILD_62_PI
    } else if real < 0.0 {
        let angle = 180.0 + (imaginary / real).atan() * 180.0 / BUILD_62_PI;
        if angle > 180.0 { 180.0 - angle } else { angle }
    } else if imaginary > 0.0 {
        90.0
    } else {
        -90.0
    };
    let period = if argument_degrees != 0.0 {
        360.0 / argument_degrees
    } else {
        REAL_ROOT_PERIOD
    };
    let root = Root {
        value: Complex { real, imaginary },
        modulus: combined_modulus.sqrt(),
        argument_degrees,
        period,
    };
    check_root(root, "grouping", index)?;
    groups[index] = Group { root, count };
    groups.remove(partner);
    Ok(())
}

fn half_roots(groups: &[Group], expected: usize) -> Result<Vec<Root>, Error> {
    let mut selected = Vec::new();
    let mut index = 0;
    while index < groups.len() {
        let group = groups[index];
        check_root(group.root, "selection", index)?;
        let repeats = if (group.root.modulus - 1.0).abs() < UNIT_EPSILON {
            let repeats = group.count / 2;
            if group.count == 1 {
                let Some(next) = groups.get(index + 1) else {
                    return Err(unusable_roots("selection", None));
                };
                if next.count == 1 && (next.root.modulus - 1.0).abs() < UNIT_EPSILON {
                    index += 1;
                    let mut root = next.root;
                    if root.value.real > 0.0 {
                        root.value = Complex {
                            real: root.modulus,
                            imaginary: 0.0,
                        };
                        root.period = 999.0;
                        root.argument_degrees = 180.0;
                    } else {
                        root.value = Complex {
                            real: -root.modulus,
                            imaginary: 0.0,
                        };
                        root.period = 2.0;
                        root.argument_degrees = 0.0;
                    }
                    selected.push(root);
                }
            }
            repeats
        } else if group.root.modulus < 1.0 {
            group.count
        } else {
            0
        };
        let current = groups[index].root;
        let complex = current.value.imaginary.abs() > ADJACENT_EPSILON;
        let adjacent = groups.get(index + 1);
        if complex && repeats > 0 && adjacent.is_none() {
            return Err(unusable_roots("selection", None));
        }
        let paired = adjacent.filter(|next| {
            complex && (current.value.real - next.root.value.real).abs() < ADJACENT_EPSILON
        });
        for _ in 0..repeats {
            selected.push(current);
            if let Some(next) = paired {
                selected.push(next.root);
            }
        }
        index += if paired.is_some() && repeats > 0 {
            2
        } else {
            1
        };
    }
    if selected.len() != expected {
        return Err(unusable_roots("selection", None));
    }
    Ok(selected)
}

/// `1 - root·B`.
fn monic_factor(root: &Root) -> [Complex; 2] {
    [
        Complex {
            real: 1.0,
            imaginary: 0.0,
        },
        Complex {
            real: -root.value.real,
            imaginary: -root.value.imaginary,
        },
    ]
}

/// MPBC's complex product `left · right`. MPBC copies both inputs before
/// clearing its possibly aliased output; a separate `output` does the same.
fn multiply_into(left: &[Complex], right: &[Complex], output: &mut Vec<Complex>) {
    output.clear();
    output.resize(left.len() + right.len() - 1, Complex::default());
    for (i, a) in left.iter().enumerate() {
        for (j, b) in right.iter().enumerate() {
            let imaginary = a.real.mul_add(b.imaginary, a.imaginary * b.real);
            let real = a.real.mul_add(b.real, -(b.imaginary * a.imaginary));
            // gfortran fuses MPBC's complex product but rounds these
            // accumulator additions separately.
            output[i + j].real += real;
            output[i + j].imaginary += imaginary;
        }
    }
}

#[cfg(test)]
mod tests;
