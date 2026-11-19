//! MLTSOL (`ansub2.f`): the Gauss-Jordan solver behind SEATS's linear
//! systems, PARFRA's partial fractions and ESTBUR's filter equations.
//!
//! The pivot search, the elimination order, the clamps of tiny values and
//! the one multiply-add gfortran fused follow Build 62: the solutions feed
//! every component, so their low-order bits matter.

use super::decomposition::polynomial::MaFamily;
use super::failure::{Failure, usable};
use crate::Frequency;
use crate::numerics::RowMajorMatrix;

/// ESTBUR's systems for the airline and seasonal-MA families, sized by the
/// frequency alone. The other families size their systems from the aggregate
/// polynomials and call [`solve`] directly.
#[derive(Debug, Clone, Copy)]
pub(super) enum EstburSystem {
    /// `s + 2` partition equations with trend, seasonal and cycle right-hand
    /// sides.
    Partition,
    /// `s + q` endpoint equations with six right-hand sides.
    Endpoint(MaFamily),
}

pub(super) fn solve_estbur(
    frequency: Frequency,
    system: EstburSystem,
    matrix: RowMajorMatrix,
) -> Result<RowMajorMatrix, Failure> {
    let s = usize::from(frequency.periods_per_year());
    let (n, rhs) = match system {
        EstburSystem::Partition => (s + 2, 3),
        EstburSystem::Endpoint(family) => (s + family.ma_coefficients(s), 6),
    };
    solve(matrix, n, rhs)
}

/// MLTSOL on an `n × (n + rhs)` system: the first `n` columns are the
/// coefficients, the rest are right-hand sides that come back as solutions.
pub(super) fn solve(
    matrix: RowMajorMatrix,
    n: usize,
    rhs: usize,
) -> Result<RowMajorMatrix, Failure> {
    check_shape(&matrix, n, rhs)?;
    let columns = matrix.column_count();
    let mut values = matrix.into_values();
    mltsol(&mut values, n, rhs)?;
    Ok(RowMajorMatrix::new(values, columns))
}

/// Rejects a system without `n` rows of `n + rhs` cells. Every row has the
/// matrix's one width, so the first row reports a wrong width.
fn check_shape(matrix: &RowMajorMatrix, n: usize, rhs: usize) -> Result<(), Failure> {
    let rows = matrix.row_count();
    if rows != n {
        return Err(Failure::Unusable {
            quantity: "linear system row count",
            index: None,
            value: rows as f64,
        });
    }
    let columns = matrix.column_count();
    if rows > 0 && columns != n + rhs {
        return Err(Failure::Unusable {
            quantity: "linear system column count",
            index: Some(0),
            value: columns as f64,
        });
    }
    Ok(())
}

// MLTSOL's constants as gfortran folded them. Its `10.d-30` is 1e-29, not
// 1e-30.
const PIVOT_FLOOR: f64 = f64::from_bits(0x39e9_5a5e_fea6_b347);
const MIN1: f64 = f64::from_bits(0x3cd2_03af_9ee7_5616);
const MIN2: f64 = f64::from_bits(0xbcd2_03af_9ee7_5616);
const ELIMINATION_CUTOFF: f64 = f64::from_bits(0x3d3c_25c2_6849_7682);

/// The first column without a pivot whose magnitude is strictly the largest
/// above MLTSOL's floor.
fn select_pivot(row: &[f64], pivot_rows: &[Option<usize>]) -> Option<usize> {
    debug_assert_eq!(row.len(), pivot_rows.len());
    let mut magnitude = PIVOT_FLOOR;
    let mut selected = None;
    for (column, (&value, pivot_row)) in row.iter().zip(pivot_rows).enumerate() {
        if value.abs() > magnitude && pivot_row.is_none() {
            magnitude = value.abs();
            selected = Some(column);
        }
    }
    selected
}

fn clamp_tiny(value: f64) -> f64 {
    if (MIN2..=MIN1).contains(&value) {
        0.0
    } else {
        value
    }
}

fn should_eliminate(value: f64) -> bool {
    value.abs() >= ELIMINATION_CUTOFF
}

fn eliminate_cell(pivot_row_cell: f64, factor: f64, old: f64) -> f64 {
    // gfortran's fused multiply-subtract negates the pivot-row multiplicand,
    // not the result.
    (-pivot_row_cell).mul_add(factor, old)
}

/// MLTSOL in place on a row-major `n × (n + rhs)` system.
fn mltsol(values: &mut [f64], n: usize, rhs: usize) -> Result<(), Failure> {
    let columns = n + rhs;
    let at = |row: usize, column: usize| row * columns + column;
    for (index, &cell) in values.iter().enumerate() {
        usable("linear system coefficient", Some(index / columns), cell)?;
    }
    // Each column's pivot row. The right-hand-side columns never get one.
    let mut pivot_rows = vec![None; columns];
    for i in (0..n).rev() {
        let selected = select_pivot(&values[at(i, 0)..at(i, n)], &pivot_rows[..n])
            .ok_or(Failure::SingularSystem { row: i })?;
        pivot_rows[selected] = Some(i);
        let pivot = clamp_tiny(usable(
            "pivot reciprocal",
            Some(i),
            1.0 / values[at(i, selected)],
        )?);
        for j in (0..n).filter(|&j| j != i) {
            if should_eliminate(values[at(j, selected)]) {
                // The factor rounds on its own before the fused elimination.
                let factor = usable(
                    "elimination factor",
                    Some(j),
                    pivot * values[at(j, selected)],
                )?;
                for (k, pivot_row) in pivot_rows.iter().enumerate() {
                    // MLTSOL zeroes a tiny pivot-row cell only here, inside an
                    // eliminated row's loop, and in columns that already have
                    // a pivot too. With no row to eliminate it stays.
                    if (MIN2..=MIN1).contains(&values[at(i, k)]) {
                        values[at(i, k)] = 0.0;
                    } else if pivot_row.is_none() {
                        values[at(j, k)] = usable(
                            "eliminated coefficient",
                            Some(j),
                            eliminate_cell(values[at(i, k)], factor, values[at(j, k)]),
                        )?;
                    }
                }
            }
        }
        for (k, pivot_row) in pivot_rows.iter().enumerate() {
            if pivot_row.is_none() {
                // A plain product: gfortran does not fuse the normalization.
                let normalized = pivot * values[at(i, k)];
                values[at(i, k)] = usable("normalized coefficient", Some(i), normalized)?;
            }
        }
    }
    // MLTSOL permutes one right-hand side at a time through a scratch column.
    let mut solution = vec![0.0; n];
    for column in n..columns {
        for (index, (slot, pivot_row)) in solution.iter_mut().zip(&pivot_rows).enumerate() {
            let row = pivot_row.ok_or(Failure::SingularSystem { row: index })?;
            *slot = usable("linear system solution", Some(row), values[at(row, column)])?;
        }
        for (row, &value) in solution.iter().enumerate() {
            values[at(row, column)] = value;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
