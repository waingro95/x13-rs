//! Column-packed upper triangles and the LINPACK routines that work on them.
//! Columns are 0-based; use `packed_len`, `packed_column` and
//! `packed_diagonal` rather than repeating `n * (n + 1) / 2`.

use std::ops::Range;

use super::blas1::{axpy, axpy_strided, dot, dot_strided, dot_term_is_retained};
use super::{BUILD_62_MACHINE_EPSILON, BUILD_62_MIN_POSITIVE};

/// Entries in a column-packed upper triangle of the given `order`.
pub(crate) const fn packed_len(order: usize) -> usize {
    order * (order + 1) / 2
}

/// [`packed_len`], or `None` where `order * (order + 1)` overflows.
pub(crate) fn checked_packed_len(order: usize) -> Option<usize> {
    order
        .checked_add(1)
        .and_then(|next| order.checked_mul(next))
        .map(|entries| entries / 2)
}

/// The entries of 0-based `column` in a column-packed upper triangle: rows
/// `0..=column`, with the diagonal last.
pub(crate) const fn packed_column(column: usize) -> Range<usize> {
    packed_len(column)..packed_len(column + 1)
}

/// The index of 0-based `column`'s diagonal entry.
pub(crate) const fn packed_diagonal(column: usize) -> usize {
    packed_len(column) + column
}

/// Build 62 `xprmx.f`: the packed upper triangle of `trans(X) X` for a
/// row-major `X`, with `ddot.f`'s guard on every term.
pub(crate) fn packed_rank_product(
    row_major_matrix: &[f64],
    row_count: usize,
    column_count: usize,
    packed: &mut [f64],
) {
    if column_count == 0 {
        return;
    }
    let packed = &mut packed[..packed_len(column_count)];
    packed.fill(0.0);
    let minimum_log = BUILD_62_MIN_POSITIVE.log10();
    // X'X's entries are independent: visit rows in the original DDOT order,
    // but update adjacent packed entries together. Every entry retains its
    // exact operand order, guard and fused accumulation; only inter-entry
    // scheduling changes. No reduction is split or reassociated.
    for row in row_major_matrix[..row_count * column_count].chunks_exact(column_count) {
        // Classify each operand once, not once per pair. For finite nonzero
        // magnitudes >= 1e-150 DDOT's log guard is true (log sum >= -300).
        // Still skip zero terms exactly; tiny/nonfinite rows use the original
        // guard below. This exposes independent entries to vectorization
        // without splitting any entry's accumulator or changing its bits.
        let ordinary = row
            .iter()
            .all(|x| x.is_finite() && (*x == 0.0 || x.abs() >= 1.0e-150));
        let dense = ordinary && row.iter().all(|x| *x != 0.0);
        let mut remaining = &mut packed[..];
        for (column, &x) in row.iter().enumerate() {
            let (entries, tail) = remaining.split_at_mut(column + 1);
            if dense {
                // Every term is retained by DDOT. Vector lanes update distinct
                // entries, never separate partial sums of the same entry.
                for (total, &y) in entries.iter_mut().zip(row) {
                    *total = x.mul_add(y, *total);
                }
            } else if ordinary {
                if x != 0.0 {
                    for (total, &y) in entries.iter_mut().zip(row) {
                        if y != 0.0 {
                            *total = x.mul_add(y, *total);
                        }
                    }
                }
            } else {
                for (total, &y) in entries.iter_mut().zip(row) {
                    if dot_term_is_retained(x, y, minimum_log) {
                        *total = x.mul_add(y, *total);
                    }
                }
            }
            remaining = tail;
        }
    }
}

/// `dppfa.f`'s nonzero `info`, zero-based: the factorization stopped at this
/// column because the column's Schur complement was not positive.
///
/// The column is kept because `olsreg.f` callers accept `Info.eq.Ncxy`: on an
/// augmented `[X:y]` matrix, a stop at the response column is a perfect fit,
/// not a singular `X`. That acceptance lives at the call sites
/// (`regression::gls`, `automdl::hannan_rissanen`), because other callers
/// treat every failure as fatal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NotPositiveDefinite {
    pub(crate) column: usize,
}

/// Build 62 `dppfa.f`: factor a packed symmetric matrix of the given `order`
/// in place as `trans(R) R`.
pub(crate) fn packed_cholesky(packed: &mut [f64], order: usize) -> Result<(), NotPositiveDefinite> {
    for column in 0..order {
        let (factored, rest) = packed.split_at_mut(packed_len(column));
        let current = &mut rest[..=column];
        let mut square_sum = 0.0;
        for row in 0..column {
            let preceding = &factored[packed_column(row)];
            let value = (current[row] - dot(&preceding[..row], &current[..row])) / preceding[row];
            current[row] = value;
            square_sum = value.mul_add(value, square_sum);
        }

        let schur_value = current[column] - square_sum;
        if schur_value > 0.0 {
            current[column] = schur_value.sqrt();
        } else {
            if schur_value >= -BUILD_62_MACHINE_EPSILON {
                current[column] = 0.0;
            }
            // `dppfa.f`'s `info` is this column plus one.
            return Err(NotPositiveDefinite { column });
        }
    }

    Ok(())
}

/// Build 62 `dppsl.f`, forward half: solve `trans(R) y = b` for the packed
/// upper-triangular Cholesky factor `R` of the given `order`.
///
/// `matrix` holds `column_count` right-hand sides in row-major order, so
/// column `c` of row `r` is `matrix[r * column_count + c]`; pass
/// `column_count == 1` for a plain vector. Each column runs the single-vector
/// recursion with the same `ddot` guard, in the same operand order.
pub(crate) fn packed_forward_solve(
    factor: &[f64],
    order: usize,
    column_count: usize,
    matrix: &mut [f64],
) {
    for row in 0..order {
        let factor_column = &factor[packed_column(row)];
        let (above, diagonal) = (&factor_column[..row], factor_column[row]);
        let (solved, current) = matrix.split_at_mut(row * column_count);
        for (column, value) in current[..column_count].iter_mut().enumerate() {
            let solved_column = solved.iter().skip(column).step_by(column_count);
            *value = (*value - dot_strided(above, solved_column)) / diagonal;
        }
    }
}

/// Build 62 `dppsl.f`, backward half: solve `R x = y`. Shares the layout of
/// [`packed_forward_solve`] and runs the `daxpy` update with its near-zero
/// skip, so it must follow the forward half (or start from a `y` produced the
/// same way).
pub(crate) fn packed_backward_solve(
    factor: &[f64],
    order: usize,
    column_count: usize,
    matrix: &mut [f64],
) {
    for row in (0..order).rev() {
        let factor_column = &factor[packed_column(row)];
        let (above, diagonal) = (&factor_column[..row], factor_column[row]);
        let (unsolved, current) = matrix.split_at_mut(row * column_count);
        for (column, value) in current[..column_count].iter_mut().enumerate() {
            *value /= diagonal;
            let unsolved_column = unsolved.iter_mut().skip(column).step_by(column_count);
            axpy_strided(-*value, above, unsolved_column);
        }
    }
}

/// Invert the packed Cholesky factor in place using Build 62 `dppdi.f` order.
pub(crate) fn packed_inverse(factor: &mut [f64], order: usize) {
    // inverse(R), column by column.
    for column in 0..order {
        let entries = packed_column(column);
        let diagonal = packed_diagonal(column);
        factor[diagonal] = 1.0 / factor[diagonal];
        let scale = -factor[diagonal];
        for value in &mut factor[entries.start..diagonal] {
            *value *= scale;
        }

        for target_column in column + 1..order {
            let (source, rest) = factor.split_at_mut(packed_len(target_column));
            // Rows `0..=column` of the target column; the entry in row
            // `column` is zeroed before the update adds into it.
            let target = &mut rest[..=column];
            let alpha = target[column];
            target[column] = 0.0;
            axpy(alpha, &source[entries.clone()], target);
        }
    }

    // inverse(R) trans(inverse(R)), column by column.
    for column in 0..order {
        let (preceding, rest) = factor.split_at_mut(packed_len(column));
        let current = &mut rest[..=column];
        for row in 0..column {
            axpy(
                current[row],
                &current[..=row],
                &mut preceding[packed_column(row)],
            );
        }
        let scale = current[column];
        for value in current {
            *value *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::blas1::dot_strided;

    #[test]
    fn layout_helpers_index_the_column_packed_upper_triangle() {
        assert_eq!([0, 1, 2, 3, 4].map(packed_len), [0, 1, 3, 6, 10]);
        assert_eq!([0, 1, 2, 3].map(packed_column), [0..1, 1..3, 3..6, 6..10]);
        assert_eq!([0, 1, 2, 3].map(packed_diagonal), [0, 2, 5, 9]);
        assert_eq!(checked_packed_len(4), Some(10));
        assert_eq!(checked_packed_len(usize::MAX), None);
        assert_eq!(checked_packed_len(usize::MAX / 2), None);
    }

    #[test]
    fn rank_product_keeps_column_dot_order_and_guard() {
        // Cancellation distinguishes ordered FMA from regrouped reductions;
        // the tiny column and zeros exercise DDOT's fallback predicate.
        let large = f64::from_bits((1023 + 54) << 52);
        let tiny = f64::from_bits((1023 - 530) << 52);
        let matrix = [large, 1.0, tiny, 1.0, 1.0, 0.0, -large, 1.0, tiny];
        let mut packed = [99.0; 6];
        packed_rank_product(&matrix, 3, 3, &mut packed);
        let expected = [
            dot_strided(matrix.iter().step_by(3), matrix.iter().step_by(3)),
            0.0,
            3.0,
            0.0,
            2.0 * tiny,
            0.0,
        ];
        assert_eq!(bits(&packed), bits(&expected));
        packed_rank_product(&[1.0, 0.0, 3.0, 4.0, 5.0, 0.0], 2, 3, &mut packed);
        assert_eq!(packed, [17.0, 20.0, 25.0, 3.0, 0.0, 9.0]);
        packed_rank_product(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &mut packed);
        assert_eq!(packed, [17.0, 22.0, 29.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn packed_inverse_uses_build_62_zero_gate_for_inversion_updates() {
        let mut factor = [1.0, 1e-20, 1.0];

        packed_inverse(&mut factor, 2);

        assert_eq!(
            bits(&factor),
            [0x3ff0000000000000, 0x8000000000000000, 0x3ff0000000000000,]
        );
    }

    #[test]
    fn packed_inverse_uses_build_62_zero_gate_for_product_updates() {
        let mut factor = [1.0, 0.0, 1e20, -1e10, 1.0, 1.0];

        packed_inverse(&mut factor, 3);

        assert_eq!(
            bits(&factor),
            [
                0x4415af1d78b58c40,
                0x8000000000000000,
                0x37a16c262777579c,
                0x4202a05f20000000,
                0xbbc79ca10c924223,
                0x3ff0000000000000,
            ]
        );
    }

    #[test]
    fn packed_inverse_preserves_non_gated_accumulation_order() {
        let mut factor = [3.625, 0.0625, 2.0, 0.4375, 0.75, 5.75];

        packed_inverse(&mut factor, 3);

        assert_eq!(
            bits(&factor),
            [
                0x3fb39a039a726121,
                0xbf68b27cc9a64173,
                0x3fd045afaad3f652,
                0xbf6c4d3a96d1e499,
                0xbf873a8e46a770c1,
                0x3f9ef8bdb389ebac,
            ]
        );
    }

    #[test]
    fn cholesky_factor_and_failure_states() {
        let mut packed = [4.0, 2.0, 5.0, -2.0, 1.0, 6.0];
        assert_eq!(packed_cholesky(&mut packed, 3), Ok(()));
        assert_eq!(
            bits(&packed),
            [
                0x4000000000000000,
                0x3ff0000000000000,
                0x4000000000000000,
                0xbff0000000000000,
                0x3ff0000000000000,
                0x4000000000000000,
            ]
        );

        let mut cancelling = [3.0, 1.0, 7.0, 2.0, -1.0, 9.0];
        assert_eq!(packed_cholesky(&mut cancelling, 3), Ok(()));
        assert_eq!(
            bits(&cancelling),
            [
                0x3ffbb67ae8584caa,
                0x3fe279a74590331d,
                0x4004a7e9cb8a3491,
                0x3ff279a74590331d,
                0xbfe4a7e9cb8a3492,
                0x40058a68a4a8d9f3,
            ]
        );

        let mut empty: [f64; 0] = [];
        assert_eq!(packed_cholesky(&mut empty, 0), Ok(()));

        // A Schur complement in [-eps, 0] is zeroed before the failure.
        let mut singular = [1.0, 1.0, 1.0];
        assert_eq!(
            packed_cholesky(&mut singular, 2),
            Err(NotPositiveDefinite { column: 1 })
        );
        assert_eq!(bits(&singular), bits(&[1.0, 1.0, 0.0]));

        // Below -eps the failed diagonal is left as it was.
        let mut indefinite = [1.0, 2.0, 1.0];
        assert_eq!(
            packed_cholesky(&mut indefinite, 2),
            Err(NotPositiveDefinite { column: 1 })
        );
        assert_eq!(bits(&indefinite), bits(&[1.0, 2.0, 1.0]));

        let mut negative_zero = [-0.0, 5.0, 5.0];
        assert_eq!(
            packed_cholesky(&mut negative_zero, 2),
            Err(NotPositiveDefinite { column: 0 })
        );
        assert_eq!(bits(&negative_zero), bits(&[0.0, 5.0, 5.0]));

        let mut not_a_number = [4.0, f64::NAN, 1.0];
        assert_eq!(
            packed_cholesky(&mut not_a_number, 2),
            Err(NotPositiveDefinite { column: 1 })
        );
        assert_eq!(not_a_number[0], 2.0);
        assert!(not_a_number[1].is_nan() && not_a_number[2] == 1.0);
    }

    #[test]
    fn rank_product_fills_only_the_packed_prefix() {
        let mut packed = [99.0; 4];
        packed_rank_product(&[1.0, 2.0], 1, 2, &mut packed);
        assert_eq!(packed, [1.0, 2.0, 4.0, 99.0]);
        let mut untouched = [99.0];
        packed_rank_product(&[], 0, 0, &mut untouched);
        assert_eq!(untouched, [99.0]);
    }

    fn solve(factor: &[f64], order: usize, column_count: usize, matrix: &mut [f64]) {
        packed_forward_solve(factor, order, column_count, matrix);
        packed_backward_solve(factor, order, column_count, matrix);
    }

    #[test]
    fn reverse_solve_preserves_build_62_near_zero_axpy_branch() {
        const DELTA: f64 = 3.834e-20;
        let factor = [1.0, 2.0, 1.0];

        let mut zero = [0.0, 0.0];
        solve(&factor, 2, 1, &mut zero);
        assert_eq!(zero, [0.0, 0.0]);

        let mut positive_tiny = [0.0, 1.0e-20];
        solve(&factor, 2, 1, &mut positive_tiny);
        assert_eq!(positive_tiny, [0.0, 1.0e-20]);

        let mut negative_tiny = [0.0, -1.0e-20];
        solve(&factor, 2, 1, &mut negative_tiny);
        assert_eq!(negative_tiny, [0.0, -1.0e-20]);

        let mut threshold = [0.0, DELTA];
        solve(&factor, 2, 1, &mut threshold);
        assert_eq!(threshold, [-2.0 * DELTA, DELTA]);

        let mut two_columns = [0.0, 0.0, 1.0e-20, -1.0e-20];
        solve(&factor, 2, 2, &mut two_columns);
        assert_eq!(two_columns, [0.0, 0.0, 1.0e-20, -1.0e-20]);
    }

    #[test]
    fn reverse_solve_uses_factor_and_matrix_column_strides() {
        const DELTA: f64 = 3.834e-20;
        let factor = [1.0, 0.0, 1.0, 3.0, 5.0, 1.0];
        let mut two_columns = [0.0, 0.0, 0.0, 0.0, DELTA, 1.0];

        solve(&factor, 3, 2, &mut two_columns);

        assert_eq!(
            two_columns,
            [-3.0 * DELTA, -3.0, -5.0 * DELTA, -5.0, DELTA, 1.0]
        );
    }

    #[test]
    fn multi_column_solve_equals_independent_single_column_solves() {
        let factor = [2.0, 0.5, 1.5, -0.25, 0.75, 3.0];
        let columns = [[1.0, -2.0, 4.0], [0.5, 3.0, -1.0]];
        let mut together = [0.0; 6];
        for (column, values) in columns.iter().enumerate() {
            for (row, value) in values.iter().enumerate() {
                together[row * 2 + column] = *value;
            }
        }
        solve(&factor, 3, 2, &mut together);
        for (column, values) in columns.iter().enumerate() {
            let mut single = *values;
            solve(&factor, 3, 1, &mut single);
            for (row, value) in single.iter().enumerate() {
                assert_eq!(together[row * 2 + column].to_bits(), value.to_bits());
            }
        }
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }
}
