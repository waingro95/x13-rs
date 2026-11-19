use super::polynomial::{RootComputationError, roots};
use crate::numerics::build_62_equal;

/// One root as `roots.f` reports it. The frequency is in cycles per period.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RootDetails {
    pub(crate) real: f64,
    pub(crate) imaginary: f64,
    pub(crate) modulus: f64,
    pub(crate) frequency: f64,
}

/// The roots of one factor, in order of modulus.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DetailedRoots {
    pub(crate) roots: Vec<RootDetails>,
    pub(crate) all_outside_or_on_unit_circle: bool,
}

/// The roots of `1 - sum c_k B^k` with their moduli and frequencies, as
/// `roots.f` reports them.
pub(crate) fn detailed_roots(coefficients: &[f64]) -> Result<DetailedRoots, RootComputationError> {
    // `roots.f` divides the angle by this value, which is not 2π
    // (6.283185307...).
    const FREQUENCY_DIVISOR: f64 = 6.283_187_307_079_59;
    let raw = roots(coefficients)?;
    let mut details = Vec::with_capacity(raw.len());
    let mut all_outside_or_on_unit_circle = true;
    let mut raw = raw.into_iter();
    while let Some(root) = raw.next() {
        let modulus = root.checked_modulus()?;
        let frequency = root.imaginary.atan2(root.real) / FREQUENCY_DIVISOR;
        if !frequency.is_finite() {
            return Err(RootComputationError);
        }
        if modulus < 1.0 {
            all_outside_or_on_unit_circle = false;
        }
        details.push(RootDetails {
            real: root.real,
            imaginary: root.imaginary,
            modulus,
            frequency,
        });
        // A complex root's conjugate follows it and copies its modulus and
        // negated frequency. `0.0 - frequency` is not `-frequency` at +0.0.
        if !build_62_equal(root.imaginary, 0.0) {
            let conjugate = raw.next().ok_or(RootComputationError)?;
            let copied_frequency = 0.0 - frequency;
            if !copied_frequency.is_finite() {
                return Err(RootComputationError);
            }
            details.push(RootDetails {
                real: conjugate.real,
                imaginary: conjugate.imaginary,
                modulus,
                frequency: copied_frequency,
            });
        }
    }
    Ok(DetailedRoots {
        roots: details,
        all_outside_or_on_unit_circle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_of_a_linear_factor() {
        assert_eq!(
            detailed_roots(&[0.5]).unwrap(),
            DetailedRoots {
                roots: vec![RootDetails {
                    real: 2.0,
                    imaginary: 0.0,
                    modulus: 2.0,
                    frequency: 0.0
                }],
                all_outside_or_on_unit_circle: true,
            }
        );
    }

    #[test]
    fn a_negative_root_has_frequency_one_half_and_real_roots_are_ordered_by_modulus() {
        let negative = detailed_roots(&[-0.5]).unwrap();
        assert_eq!(negative.roots[0].real, -2.0);
        assert_eq!(negative.roots[0].imaginary, 0.0);
        assert_eq!(negative.roots[0].modulus, 2.0);
        assert!((negative.roots[0].frequency - 0.499_999_840_853_065).abs() < 1e-15);
        let quadratic = detailed_roots(&[0.25, 0.125]).unwrap();
        assert_eq!(
            quadratic
                .roots
                .iter()
                .map(|root| root.real)
                .collect::<Vec<_>>(),
            [2.0, -4.0]
        );
        assert!(quadratic.roots.iter().all(|root| root.imaginary == 0.0));
        assert_eq!(
            quadratic
                .roots
                .iter()
                .map(|root| root.modulus)
                .collect::<Vec<_>>(),
            [2.0, 4.0]
        );
    }

    #[test]
    fn zero_coefficients_have_no_roots_and_invalid_ones_are_errors() {
        for coefficients in [&[][..], &[0.0][..], &[0.0, 0.0, 0.0][..]] {
            assert_eq!(
                detailed_roots(coefficients).unwrap(),
                DetailedRoots {
                    roots: vec![],
                    all_outside_or_on_unit_circle: true
                }
            );
        }
        assert_eq!(detailed_roots(&[0.5, -0.25, 0.0]).unwrap().roots.len(), 2);
        let immediately_below = f64::from_bits(3.834e-20f64.to_bits() - 1);
        assert!(
            detailed_roots(&[immediately_below])
                .unwrap()
                .roots
                .is_empty()
        );
        assert_eq!(detailed_roots(&[3.834e-20]).unwrap().roots.len(), 1);
        for coefficients in [
            &[0.0, 0.0, 0.0, 0.0][..],
            &[f64::NAN][..],
            &[f64::INFINITY][..],
            &[f64::NEG_INFINITY][..],
            &[-1e308, -1.0][..],
        ] {
            assert_eq!(detailed_roots(coefficients), Err(RootComputationError));
        }
    }

    #[test]
    fn a_root_on_the_unit_circle_is_not_inside_it() {
        let below = f64::from_bits(1.0f64.to_bits() - 1);
        let above = f64::from_bits(1.0f64.to_bits() + 1);
        assert!(
            !detailed_roots(&[above])
                .unwrap()
                .all_outside_or_on_unit_circle
        );
        assert!(
            detailed_roots(&[1.0])
                .unwrap()
                .all_outside_or_on_unit_circle
        );
        assert!(
            detailed_roots(&[below])
                .unwrap()
                .all_outside_or_on_unit_circle
        );
        assert!(
            detailed_roots(&[0.5])
                .unwrap()
                .all_outside_or_on_unit_circle
        );
    }

    #[test]
    fn a_conjugate_root_copies_the_modulus_and_negates_the_frequency() {
        let roots = detailed_roots(&[0.75, -0.25, 0.03125]).unwrap().roots;
        assert_eq!(roots.len(), 3);
        assert!((roots[0].real - 2.0).abs() < 2e-15);
        assert!((roots[0].imaginary - 2.0).abs() < 2e-15);
        assert!((roots[1].real - 2.0).abs() < 2e-15);
        assert!((roots[1].imaginary + 2.0).abs() < 2e-15);
        assert!((roots[2].real - 4.0).abs() < 2e-15);
        assert_eq!(roots[0].modulus, roots[1].modulus);
        assert_eq!(roots[1].frequency, 0.0 - roots[0].frequency);
    }
}
