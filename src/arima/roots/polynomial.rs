use super::solver;
use crate::numerics::build_62_equal;

const MAX_DEGREE: usize = 3;
const COEFFICIENT_CAPACITY: usize = MAX_DEGREE + 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RootComputationError;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ComplexRoot {
    pub(super) real: f64,
    pub(super) imaginary: f64,
}

impl ComplexRoot {
    pub(super) fn checked_modulus(self) -> Result<f64, RootComputationError> {
        // Build 62 rounds the imaginary square before fusing the real square.
        let imaginary_square = self.imaginary * self.imaginary;
        let modulus = self.real.mul_add(self.real, imaginary_square).sqrt();
        if !modulus.is_finite() {
            return Err(RootComputationError);
        }
        Ok(modulus)
    }
}

pub(crate) fn has_strictly_inside_root(coefficients: &[f64]) -> Result<bool, RootComputationError> {
    for root in roots(coefficients)? {
        if root.checked_modulus()? < 1.0 {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Build 62 `roots.f` for supported ARIMA factors: reverse, append the constant
/// `-1`, strip leading zeros, then solve with [`solver`]. Input
/// coefficients exclude the polynomial's constant `-1`.
pub(super) fn roots(coefficients: &[f64]) -> Result<Vec<ComplexRoot>, RootComputationError> {
    if coefficients.len() > MAX_DEGREE
        || coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return Err(RootComputationError);
    }

    let degree = coefficients.len();

    // `setmdl.f` supplies `[-1, theta_1, ...]`; `roots.f` reverses it into
    // decreasing-power order for the solver.
    let mut polynomial = [0.0; COEFFICIENT_CAPACITY];
    for (destination, coefficient) in polynomial[..degree]
        .iter_mut()
        .zip(coefficients.iter().rev())
    {
        *destination = *coefficient;
    }
    polynomial[degree] = -1.0;

    // Strip leading zero coefficients; with none left, there are no roots.
    let leading_zeros = polynomial[..degree]
        .iter()
        .take_while(|&&coefficient| build_62_equal(coefficient, 0.0))
        .count();
    if leading_zeros == degree {
        return Ok(Vec::new());
    }
    solver::polynomial_roots(&polynomial[leading_zeros..=degree])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_root_strictly_inside_the_unit_circle_is_found_for_each_degree() {
        for (coefficients, expected) in [
            (&[0.2, 1.0][..], true),
            (&[0.0, 1.0], false),
            (&[0.2, -1.0], false),
            (&[1.0, -1.0], true),
            (&[2.0, -1.0], false),
            (&[f64::from_bits(2.0_f64.to_bits() + 1), -1.0], true),
            (&[0.2, 0.2, 1.0], true),
            (&[0.2, -0.2, 1.0], false),
            (&[0.2, 0.2, -1.0], false),
            (&[0.2, -0.2, -1.0], true),
            (&[0.0, 0.0, 1.0], true),
            (&[0.0, 0.0, -1.0], true),
            (&[0.0, 0.0, f64::from_bits(1.0_f64.to_bits() - 1)], true),
            (&[0.0, 0.0, -f64::from_bits(1.0_f64.to_bits() - 1)], true),
            (&[1.0, 1.0, -1.0], false),
            // Roots exactly on the unit circle, which rounding puts inside it
            // in Build 62.
            (&[-1.0, 1.0, 1.0], false),
            (&[0.5, 0.0, 0.5], false),
            (&[0.5, 0.2, 0.3], false),
            (&[1.4, -1.0, 0.0], false),
        ] {
            assert_eq!(
                has_strictly_inside_root(coefficients),
                Ok(expected),
                "coefficients={coefficients:?}"
            );
        }
    }

    #[test]
    fn trailing_zero_coefficients_lower_the_degree() {
        assert_eq!(has_strictly_inside_root(&[0.2, 1.0, 0.0]), Ok(true));
        assert_eq!(has_strictly_inside_root(&[0.0, 1.0, 0.0]), Ok(false));
        assert_eq!(has_strictly_inside_root(&[0.0, 0.0, 0.0]), Ok(false));
        // A negative zero is dropped like a positive one.
        assert_eq!(has_strictly_inside_root(&[0.2, 1.0, -0.0]), Ok(true));
        assert_eq!(has_strictly_inside_root(&[-0.0, -0.0, -0.0]), Ok(false));
    }

    #[test]
    fn a_degree_above_three_or_a_nonfinite_coefficient_is_an_error() {
        assert_eq!(
            has_strictly_inside_root(&[0.0, 0.0, 0.0, 0.0]),
            Err(RootComputationError)
        );
        for coefficients in [
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [f64::NEG_INFINITY, 0.0, 0.0],
        ] {
            assert_eq!(
                has_strictly_inside_root(&coefficients),
                Err(RootComputationError)
            );
        }
    }
}
