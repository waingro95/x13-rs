//! Chi-square and normal probabilities (`chisq.f`, `gauss.f`).

use std::num::NonZeroUsize;

/// `chisq.f` `C=0.797884560802864D0`: sqrt(2/pi) to the digits the routine
/// writes.
const SQRT_TWO_OVER_PI: f64 = 0.797884560802864;

/// Why a probability could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ProbabilityError {
    #[error("chi-square statistic is nonfinite")]
    NonFiniteInput,
    #[error("chi-square probability calculation is nonfinite")]
    NonFiniteResult,
}

/// Build 62 `chisq.f`: the chi-square upper-tail probability.
pub(crate) fn chi_square_upper_tail(
    statistic: f64,
    degrees_of_freedom: NonZeroUsize,
) -> Result<f64, ProbabilityError> {
    if !statistic.is_finite() {
        return Err(ProbabilityError::NonFiniteInput);
    }
    if statistic <= 0.0 {
        return Ok(1.0);
    }
    if statistic >= 90.0 {
        return Ok(0.0);
    }

    let half_df = degrees_of_freedom.get() / 2;
    let mut c1 = 1.0;
    let mut c2 = 1.0;
    let result = if degrees_of_freedom.get().is_multiple_of(2) {
        let y = statistic / 2.0;
        for index in 1..half_df {
            c2 = c2 * y / index as f64;
            c1 += c2;
        }
        c1 * (-y).exp()
    } else {
        let mut c3 = 0.0;
        for _ in 0..half_df {
            c1 = c1 * statistic / c2;
            c3 += c1;
            c2 += 2.0;
        }
        let root = statistic.sqrt();
        1.0 - central_normal(root) + SQRT_TWO_OVER_PI * c3 * (-statistic / 2.0).exp() / root
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ProbabilityError::NonFiniteResult)
    }
}

/// Build 62 `gauss.f`: `P(-x < Z < x)` for a standard normal `Z`.
fn central_normal(x: f64) -> f64 {
    if super::build_62_equal(x, 0.0) {
        return 0.0;
    }
    let mut y = x.abs() / 2.0;
    if y >= 3.0 {
        return 1.0;
    }
    if y >= 1.0 {
        y -= 2.0;
        let coefficients = [
            -0.000045255659_f64,
            0.000152529290,
            -0.000019538132,
            -0.000676904986,
            0.001390604284,
            -0.000794620820,
            -0.002034254874,
            0.006549791214,
            -0.010557625006,
            0.011630447319,
            -0.009279453341,
            0.005353579108,
            -0.002141268741,
            0.000535310849,
            0.999936657524,
        ];
        coefficients[1..]
            .iter()
            .fold(coefficients[0], |sum, &coefficient| {
                sum.mul_add(y, coefficient)
            })
    } else {
        let w = y * y;
        let coefficients = [
            0.000124818987_f64,
            -0.001075204047,
            0.005198775019,
            -0.019198292004,
            0.059054035642,
            -0.151968751364,
            0.319152932694,
            -0.531923007300,
            0.797884560593,
        ];
        coefficients[1..]
            .iter()
            .fold(coefficients[0], |sum, &coefficient| {
                sum.mul_add(w, coefficient)
            })
            * y
            * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn df(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    #[test]
    fn chisq_cutoffs_and_even_odd_recurrences() {
        for statistic in [-4.0, 0.0] {
            assert_eq!(chi_square_upper_tail(statistic, df(1)), Ok(1.0));
        }
        assert_eq!(chi_square_upper_tail(90.0, df(1)), Ok(0.0));
        assert_eq!(chi_square_upper_tail(90.0, df(8)), Ok(0.0));
        assert_eq!(
            chi_square_upper_tail(f64::from_bits(90.0_f64.to_bits() + 1), df(8)),
            Ok(0.0)
        );
        assert!(
            chi_square_upper_tail(f64::from_bits(90.0_f64.to_bits() - 1), df(8)).unwrap() > 0.0
        );
        assert_eq!(chi_square_upper_tail(2.0, df(2)).unwrap(), (-1.0_f64).exp());
        assert_eq!(
            chi_square_upper_tail(2.0, df(4)).unwrap(),
            2.0 * (-1.0_f64).exp()
        );
        assert_eq!(
            chi_square_upper_tail(1.0, df(1)).unwrap(),
            1.0 - central_normal(1.0)
        );
        let odd_three = chi_square_upper_tail(1.0, df(3)).unwrap();
        let build_62_increment = 0.797884560802864 * (-0.5_f64).exp();
        assert!(
            (odd_three - (chi_square_upper_tail(1.0, df(1)).unwrap() + build_62_increment)).abs()
                < 1e-15
        );
    }

    #[test]
    fn normal_equality_polynomials_and_saturation() {
        assert_eq!(central_normal(0.0), 0.0);
        assert_eq!(central_normal(3.833e-20), 0.0);
        assert!(central_normal(3.835e-20) > 0.0);
        assert_eq!(central_normal(0.324).to_bits(), 0x3fd0428d1e99ff05);
        assert_eq!(central_normal(4.0), 0.999936657524);
        assert_eq!(central_normal(6.0), 1.0);
        assert_eq!(central_normal(-6.0), 1.0);
    }

    #[test]
    fn nonfinite_input_is_an_error() {
        for statistic in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                chi_square_upper_tail(statistic, df(1)),
                Err(ProbabilityError::NonFiniteInput)
            );
        }
    }
}
