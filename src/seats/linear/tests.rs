use super::*;

fn row_major(rows: Vec<Vec<f64>>) -> RowMajorMatrix {
    let columns = rows[0].len();
    RowMajorMatrix::new(rows.concat(), columns)
}

fn cell(matrix: &RowMajorMatrix, row: usize, column: usize) -> f64 {
    matrix.as_slice()[row * matrix.column_count() + column]
}

#[test]
fn solves_a_tridiagonal_system() {
    for n in [6, 14] {
        let mut matrix = vec![vec![0.0; n + 1]; n];
        for row in 0..n {
            matrix[row][row] = 3.0 + row as f64 / 16.0;
            matrix[row][n] = (row as f64 + 1.0) / 7.0;
            if row > 0 {
                matrix[row][row - 1] = -0.375;
            }
            if row + 1 < n {
                matrix[row][row + 1] = 0.625;
            }
        }
        let solved = solve(row_major(matrix.clone()), n, 1).unwrap();
        let solution: Vec<f64> = solved.rows().map(|row| row[n]).collect();
        for cells in &matrix {
            let product: f64 = cells[..n].iter().zip(&solution).map(|(a, x)| a * x).sum();
            assert!(
                (product - cells[n]).abs() <= 1e-14,
                "{product} != {}",
                cells[n]
            );
        }
    }
}

#[test]
fn right_hand_sides_keep_their_column_order() {
    for frequency in [Frequency::Quarterly, Frequency::Monthly] {
        let s = usize::from(frequency.periods_per_year());
        for (n, rhs) in [(s + 2, 3), (2 * s + 2, 6)] {
            let mut matrix = vec![vec![0.0; n + rhs]; n];
            for (row, cells) in matrix.iter_mut().enumerate() {
                let pivot_column = n - row - 1;
                cells[pivot_column] = (row + 2) as f64;
                for column in 0..rhs {
                    cells[n + column] = ((row + 1) * (column + 3)) as f64;
                }
            }
            let system = if rhs == 3 {
                EstburSystem::Partition
            } else {
                EstburSystem::Endpoint(MaFamily::Airline)
            };
            let solved = solve_estbur(frequency, system, row_major(matrix)).unwrap();
            for (row, cells) in solved.rows().enumerate() {
                let mirrored_row = n - row - 1;
                for column in 0..rhs {
                    assert_eq!(
                        cells[n + column].to_bits(),
                        ((1.0 / (mirrored_row + 2) as f64)
                            * ((mirrored_row + 1) * (column + 3)) as f64)
                            .to_bits(),
                        "{frequency:?} row={row} rhs={column}"
                    );
                }
            }
        }
    }
}

#[test]
fn diagonal_systems_solve_every_right_hand_side() {
    for (n, rhs) in [(14, 1), (6, 1), (3, 1), (15, 3), (7, 3), (26, 6), (10, 6)] {
        let mut matrix = vec![vec![0.0; n + rhs]; n];
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = 2.0;
            for j in 0..rhs {
                row[n + j] = 2.0 * (i + j + 1) as f64;
            }
        }
        let solved = solve(row_major(matrix), n, rhs).unwrap();
        for (i, row) in solved.rows().enumerate() {
            for j in 0..rhs {
                assert_eq!(row[n + j], (i + j + 1) as f64);
            }
        }
    }
}

#[test]
fn rejects_wrong_shape_missing_pivot_and_nonfinite_input() {
    let short = row_major(vec![vec![0.0; 9]; 5]);
    assert_eq!(
        solve_estbur(Frequency::Quarterly, EstburSystem::Partition, short),
        Err(Failure::Unusable {
            quantity: "linear system row count",
            index: None,
            value: 5.0,
        })
    );
    // A row-major matrix has one width, so the first row reports it.
    let narrow = row_major(vec![vec![0.0; 15]; 10]);
    let airline_endpoint = EstburSystem::Endpoint(MaFamily::Airline);
    assert_eq!(
        solve_estbur(Frequency::Quarterly, airline_endpoint, narrow),
        Err(Failure::Unusable {
            quantity: "linear system column count",
            index: Some(0),
            value: 15.0,
        })
    );
    let singular = row_major(vec![vec![0.0; 9]; 6]);
    assert_eq!(
        solve_estbur(Frequency::Quarterly, EstburSystem::Partition, singular),
        Err(Failure::SingularSystem { row: 5 })
    );
    let mut nonfinite = vec![vec![0.0; 16]; 10];
    for (index, row) in nonfinite.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    nonfinite[3][14] = f64::INFINITY;
    assert_eq!(
        solve_estbur(Frequency::Quarterly, airline_endpoint, row_major(nonfinite)),
        Err(Failure::Unusable {
            quantity: "linear system coefficient",
            index: Some(3),
            value: f64::INFINITY,
        })
    );
}

#[test]
fn clamped_reciprocal_zeroes_each_right_hand_side_with_its_sign() {
    let mut matrix = vec![vec![0.0; 9]; 6];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = if i == 5 { 1e16 } else { 1.0 };
        row[6] = (i + 1) as f64;
        row[7] = -((i + 2) as f64);
        row[8] = (i + 3) as f64;
    }
    let solved = solve_estbur(
        Frequency::Quarterly,
        EstburSystem::Partition,
        row_major(matrix),
    )
    .unwrap();
    assert_eq!(cell(&solved, 5, 6).to_bits(), 0.0_f64.to_bits());
    assert_eq!(cell(&solved, 5, 7).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(cell(&solved, 5, 8).to_bits(), 0.0_f64.to_bits());
    assert_eq!(cell(&solved, 0, 6).to_bits(), 1.0_f64.to_bits());
    assert_eq!(cell(&solved, 0, 7).to_bits(), (-2.0_f64).to_bits());
    assert_eq!(cell(&solved, 0, 8).to_bits(), 3.0_f64.to_bits());
}

#[test]
fn factor_overflow_is_a_nonfinite_failure() {
    let mut overflowing = vec![vec![0.0; 7]; 6];
    for (index, row) in overflowing.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    overflowing[5][5] = PIVOT_FLOOR.next_up();
    overflowing[0][5] = f64::MAX;
    assert_eq!(
        solve(row_major(overflowing), 6, 1),
        Err(Failure::Unusable {
            quantity: "elimination factor",
            index: Some(0),
            value: f64::INFINITY,
        })
    );
}

#[test]
fn pivot_clamp_and_elimination_thresholds() {
    assert_eq!(select_pivot(&[PIVOT_FLOOR], &[None]), None);
    assert_eq!(select_pivot(&[PIVOT_FLOOR.next_up()], &[None]), Some(0));
    assert_eq!(
        select_pivot(&[3.0, -3.0, 2.0], &[None, None, None]),
        Some(0)
    );
    assert_eq!(
        select_pivot(&[3.0, -3.0, 2.0], &[Some(0), None, None]),
        Some(1)
    );
    assert_eq!(
        select_pivot(&[3.0, -3.0, 2.0], &[Some(0), Some(1), None]),
        Some(2)
    );
    for value in [MIN1, MIN2, 0.0, -0.0] {
        assert_eq!(clamp_tiny(value).to_bits(), 0.0_f64.to_bits());
    }
    for value in [MIN1.next_up(), MIN2.next_down()] {
        assert_eq!(clamp_tiny(value).to_bits(), value.to_bits());
    }
    assert!(should_eliminate(ELIMINATION_CUTOFF));
    assert!(should_eliminate(-ELIMINATION_CUTOFF));
    assert!(!should_eliminate(ELIMINATION_CUTOFF.next_down()));
    assert!(!should_eliminate(-ELIMINATION_CUTOFF.next_down()));
}

#[test]
fn elimination_negates_the_pivot_row_multiplicand() {
    let e = f64::EPSILON;
    let cell = 1.0 + e;
    let factor = 1.0 - e;
    assert_eq!(
        eliminate_cell(cell, factor, 1.0).to_bits(),
        (e * e).to_bits()
    );
    let separate = 1.0 - cell * factor;
    assert_eq!(separate.to_bits(), 0.0_f64.to_bits());
    // Negating the fused result instead would turn this +0 into -0.
    assert_eq!(eliminate_cell(0.0, 0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        (-(0.0_f64.mul_add(0.0, -0.0))).to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn pivot_row_clamp_needs_an_eliminated_row() {
    let tiny = f64::from_bits(0x3cd2_03af_9ee7_5616);
    let mut matrix = vec![vec![0.0; 6]; 5];
    for (row, rhs) in [1.0, 2.0, 3.0, 4.0, 6.0].into_iter().enumerate() {
        matrix[row][row] = if row == 4 { 2.0 } else { 1.0 };
        matrix[row][5] = rhs;
    }
    matrix[4][0] = tiny;
    let mut eligible = matrix.clone();
    eligible[3][4] = 1.0;
    let solved = solve(row_major(matrix), 5, 1).unwrap();
    assert_eq!(cell(&solved, 4, 0).to_bits(), (tiny * 0.5).to_bits());
    assert_eq!(
        solved.rows().map(|row| row[5]).collect::<Vec<_>>(),
        [1.0, 2.0, 3.0, 4.0, 3.0]
    );
    let eligible = solve(row_major(eligible), 5, 1).unwrap();
    assert_eq!(cell(&eligible, 4, 0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        eligible.rows().map(|row| row[5]).collect::<Vec<_>>(),
        [1.0, 2.0, 3.0, 1.0, 3.0]
    );
}
