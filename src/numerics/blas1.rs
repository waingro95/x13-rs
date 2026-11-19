//! Build 62's `ddot.f` and `daxpy.f` on slices.
//!
//! The unit-stride forms take two equal-length slices. The strided forms take
//! each operand as an iterator, so a call site names the elements it means,
//! such as one column of a row-major matrix, `values[column..].iter().step_by(width)`.
//! Strides are positive.
//!
//! `dot` accumulates left to right from +0.0 with fused multiply-adds and
//! skips the terms DDOT's log guard rejects (`dot_term_is_retained`). `axpy`
//! updates in order with fused multiply-adds and does nothing when `alpha` is
//! `dpeq.f`-equal to zero.

use super::{BUILD_62_MIN_POSITIVE, build_62_equal};

/// Build 62 `ddot.f` with unit increments: the sum of `x[i] * y[i]`.
///
/// # Panics
///
/// If `x` and `y` differ in length.
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    // The zip would otherwise silently drop the longer operand's tail.
    assert_eq!(x.len(), y.len(), "dot operands differ in length");
    dot_terms(x.iter().zip(y))
}

/// [`dot`] with positive increments: each operand yields its elements in
/// order, for example through `step_by`. The sum stops at the shorter one.
pub(crate) fn dot_strided<'x, 'y>(
    x: impl IntoIterator<Item = &'x f64>,
    y: impl IntoIterator<Item = &'y f64>,
) -> f64 {
    dot_terms(x.into_iter().zip(y))
}

fn dot_terms<'x, 'y>(terms: impl Iterator<Item = (&'x f64, &'y f64)>) -> f64 {
    let minimum_log = BUILD_62_MIN_POSITIVE.log10();
    terms.fold(0.0, |total, (&x, &y)| {
        if dot_term_is_retained(x, y, minimum_log) {
            x.mul_add(y, total)
        } else {
            total
        }
    })
}

#[inline]
pub(crate) fn dot_term_is_retained(x: f64, y: f64, minimum_log: f64) -> bool {
    // DDOT's logarithmic guard is necessarily true for finite operands
    // with magnitudes >= 1e-150: the log sum is >= -300, safely above
    // log10(2.225074e-308) (~-307.65). Skip only that guard computation;
    // accumulation remains fused and ordered. Zero, NaN, infinity and tiny
    // operands retain the reference path, including in packed covariance.
    (x.is_finite() && y.is_finite() && x.abs() >= 1.0e-150 && y.abs() >= 1.0e-150)
        || (x != 0.0 && y != 0.0 && x.abs().log10() + y.abs().log10() > minimum_log)
}

/// Build 62 `daxpy.f` with unit increments: `y[i] += alpha * x[i]`.
///
/// # Panics
///
/// If `x` and `y` differ in length.
pub(crate) fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    // The zip would otherwise silently drop the longer operand's tail.
    assert_eq!(x.len(), y.len(), "axpy operands differ in length");
    axpy_terms(alpha, x.iter().zip(y));
}

/// [`axpy`] with positive increments, as [`dot_strided`] takes them. The
/// update stops at the shorter operand.
pub(crate) fn axpy_strided<'x, 'y>(
    alpha: f64,
    x: impl IntoIterator<Item = &'x f64>,
    y: impl IntoIterator<Item = &'y mut f64>,
) {
    axpy_terms(alpha, x.into_iter().zip(y));
}

fn axpy_terms<'x, 'y>(alpha: f64, terms: impl Iterator<Item = (&'x f64, &'y mut f64)>) {
    if build_62_equal(alpha, 0.0) {
        return;
    }
    for (&x, y) in terms {
        *y = alpha.mul_add(x, *y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_strided_reads_every_stride_th_element() {
        let x = [1.0, 99.0, 2.0, 99.0, 3.0];
        let y = [4.0, 99.0, 5.0, 99.0, 6.0];

        assert_eq!(dot_strided(x.iter().step_by(2), y.iter().step_by(2)), 32.0);
    }

    #[test]
    fn dot_preserves_left_to_right_accumulation() {
        let x = [1.0e16, 1.0, -1.0e16, 3.0];
        let y = [1.0; 4];

        assert_eq!(dot(&x, &y), 3.0);
    }

    #[test]
    fn dot_preserves_underflow_boundaries_and_special_operands() {
        // Powers of two give independently exact products on either side of
        // the fast-path boundary and DDOT's rounded minimum-normal constant.
        let power = |exponent: i32| f64::from_bits(((1023 + exponent) as u64) << 52);
        for (x, y, expected) in [
            (power(-498), power(-498), power(-996)),
            (power(-499), power(-499), power(-998)),
            (power(-510), power(-510), power(-1020)),
            (power(-511), power(-511), 0.0),
            (power(-600), power(600), 1.0),
            (-power(-600), power(600), -1.0),
            (BUILD_62_MIN_POSITIVE, 1.0, 0.0),
            (0.0, f64::INFINITY, 0.0),
            (-0.0, f64::NEG_INFINITY, 0.0),
            (f64::NAN, 1.0, 0.0),
            (1.0, f64::NAN, 0.0),
            (f64::INFINITY, 1.0, f64::INFINITY),
            (f64::NEG_INFINITY, 1.0, f64::NEG_INFINITY),
        ] {
            assert_eq!(dot(&[x], &[y]).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn dot_handles_signed_zero_empty_nan_and_strides() {
        let positive_zero = 0.0_f64.to_bits();
        assert_eq!(dot(&[], &[]).to_bits(), positive_zero);
        assert_eq!(dot(&[-0.0], &[-0.0]).to_bits(), positive_zero);
        assert_eq!(dot(&[-0.0, -1.0], &[1.0, 0.0]).to_bits(), positive_zero);
        assert_eq!(dot(&[-1.0, 1.0], &[1.0, 1.0]).to_bits(), positive_zero);
        assert_eq!(dot(&[2.0, f64::NAN, 3.0], &[1.0, 5.0, 1.0]), 5.0);
        assert!(dot(&[f64::INFINITY, 1.0], &[1.0, f64::NEG_INFINITY]).is_nan());

        let x = [1.0, 99.0, 2.0, 99.0, 3.0];
        let y = [4.0, 99.0, 99.0, 5.0, 99.0, 99.0, 6.0];
        assert_eq!(dot_strided(x.iter().step_by(2), y.iter().step_by(3)), 32.0);
        assert_eq!(
            dot_strided(x.iter().step_by(2).take(2), y.iter().step_by(3)),
            14.0
        );
    }

    #[test]
    fn axpy_handles_signed_zero_empty_nan_and_strides() {
        let mut output = [-0.0, -0.0];
        axpy(2.0, &[0.0, -0.0], &mut output);
        assert_eq!(
            output.map(f64::to_bits),
            [0.0_f64.to_bits(), (-0.0_f64).to_bits()]
        );

        let mut output = [-0.0];
        axpy(-0.0, &[1.0], &mut output);
        assert_eq!(output[0].to_bits(), (-0.0_f64).to_bits());

        let mut output = [1.0];
        axpy(1.0, &[], &mut output[..0]);
        assert_eq!(output, [1.0]);

        let mut output = [1.0, 1.0];
        axpy(f64::NAN, &[1.0, 1.0], &mut output);
        assert!(output.iter().all(|value| value.is_nan()));

        let mut output = [1.0];
        axpy(1.0, &[f64::NAN], &mut output);
        assert!(output[0].is_nan());

        let mut output = [1.0, 9.0, 2.0, 9.0];
        axpy_strided(3.0, &[1.0, 2.0], output.iter_mut().step_by(2));
        assert_eq!(output, [4.0, 9.0, 8.0, 9.0]);
    }

    #[test]
    fn axpy_matches_build_62_fused_multiply_add() {
        let alpha = 1.0 + 2.0_f64.powi(-27);
        let input = [1.0 - 2.0_f64.powi(-27)];
        let mut output = [-1.0];

        axpy(alpha, &input, &mut output);

        assert_eq!(output[0], alpha.mul_add(input[0], -1.0));
    }
}
