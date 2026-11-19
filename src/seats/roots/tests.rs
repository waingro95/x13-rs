use super::*;

#[test]
fn two_power_squares_then_takes_one_reciprocal() {
    for (exponent, expected) in [
        (0, 0x3ff0000000000000),
        (1, 0x4000000000000000),
        (-1, 0x3fe0000000000000),
        (1023, 0x7fe0000000000000),
        (1024, 0x7ff0000000000000),
        (-1023, 0x0008000000000000),
        (-1024, 0),
        (i32::MAX, 0x7ff0000000000000),
        (i32::MIN, 0),
    ] {
        assert_eq!(
            two_power(exponent).to_bits(),
            expected,
            "exponent {exponent}"
        );
    }
}

#[test]
fn negligible_is_strictly_below_the_threshold() {
    assert!(is_negligible(0.0));
    assert!(is_negligible(-0.0));
    assert!(is_negligible(f64::from_bits(0x3be6a1c6e8d98028)));
    assert!(!is_negligible(ZERO_THRESHOLD));
    assert!(!is_negligible(f64::from_bits(0x3be6a1c6e8d9802a)));
    assert!(!is_negligible(-ZERO_THRESHOLD));
}

#[test]
fn symmetric_reduction_rewrites_in_powers_of_the_sum() {
    // c0 + c1 (z + 1/z) + c2 (z² + 1/z²) = c2 u² + c1 u + (c0 - 2 c2).
    assert_eq!(reduce_symmetric(&[1.0, 3.0, 5.0]), [1.0, 3.0, 3.0]);
    // z³ + 1/z³ = u³ - 3u.
    assert_eq!(
        reduce_symmetric(&[1.0, 0.0, 0.0, 0.0]),
        [1.0, 0.0, -3.0, 0.0]
    );
}

#[test]
fn quadratic_branches_on_sign_discriminant_and_signed_zero() {
    for (input, real, imaginary) in [
        ([1.0, -3.0, 2.0], [2.0, 1.0], [0.0, 0.0]),
        ([1.0, 3.0, 2.0], [-2.0, -1.0], [0.0, 0.0]),
        ([1.0, 0.0, -1.0], [-1.0, 1.0], [0.0, 0.0]),
        ([1.0, -0.0, -1.0], [-1.0, 1.0], [0.0, 0.0]),
        ([1.0, 1e-21, -1.0], [-1.0, 1.0], [0.0, 0.0]),
        ([1.0, 0.0, 1.0], [-0.0, -0.0], [-1.0, 1.0]),
        ([1.0, -0.0, 1.0], [0.0, 0.0], [-1.0, 1.0]),
        // C02AEF: a zero discriminant takes the complex branch, where
        // sqrt(-0) = -0 makes the second imaginary part -0 and the first +0.
        ([1.0, -2.0, 1.0], [1.0, 1.0], [0.0, -0.0]),
        ([1.0, -2.0, 0.0], [2.0, 0.0], [0.0, 0.0]),
    ] {
        let mut solver = Solver::new(&input);
        assert!(solver.find_roots().is_ok());
        assert_eq!(solver.remaining_count, 1);
        assert_eq!(
            solver.real.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            real.map(f64::to_bits)
        );
        assert_eq!(
            solver
                .imaginary
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            imaginary.map(f64::to_bits)
        );
    }
}

#[test]
fn evaluation_gives_value_derivative_and_adams_test() {
    for (a, x, y, r, rx, j, jx, satisfied) in [
        (vec![1.0, 0.0, 1.0], 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, true),
        (vec![1.0, 0.0, 1.0], 1.0, 0.0, 2.0, 2.0, 0.0, 0.0, false),
        (
            vec![1.0, -6.0, 11.0, -6.0],
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            true,
        ),
        (
            vec![1.0, 0.0, 0.0, 0.0, 1.0],
            0.0,
            1.0,
            2.0,
            0.0,
            0.0,
            -4.0,
            false,
        ),
    ] {
        let mut state = RootState {
            x,
            y,
            ..RootState::default()
        };
        evaluate(&a, &mut state);
        assert_eq!([state.r, state.rx, state.j, state.jx], [r, rx, j, jx]);
        assert_eq!(state.satisfied, satisfied);
    }
}

#[test]
fn roots_come_back_complex_first_with_rpq_metadata() {
    // (z² + 1)(z - 1)(z - 2).
    let output = polynomial_roots(&[1.0, -3.0, 3.0, -3.0, 2.0]).unwrap();
    let close = |actual: f64, expected: f64| (actual - expected).abs() <= 1e-10;
    assert!(close(output.real[0], 0.0) && close(output.real[1], 0.0));
    assert!(close(output.imaginary[0].abs(), 1.0));
    assert_eq!(output.imaginary[0], -output.imaginary[1]);
    let mut real_roots = [output.real[2], output.real[3]];
    real_roots.sort_by(f64::total_cmp);
    assert!(close(real_roots[0], 1.0) && close(real_roots[1], 2.0));
    assert_eq!(output.imaginary[2..], [0.0, 0.0]);
    for index in 0..2 {
        assert!(close(output.modulus[index], 1.0));
        assert!(close(output.argument_degrees[index].abs(), 90.0));
        assert!(close(output.period[index].abs(), 4.0));
    }
    assert_eq!(output.argument_degrees[2..], [0.0, 0.0]);
    assert_eq!(output.period[2..], [REAL_ROOT_PERIOD; 2]);
}

#[test]
fn rejects_unsupported_counts_and_nonfinite_coefficients() {
    for count in [0, 1, 2, MAK1_MAX_COEFFICIENTS + 1] {
        assert_eq!(
            polynomial_roots(&vec![1.0; count]),
            Err(Failure::Unusable {
                quantity: "root polynomial coefficient count",
                index: None,
                value: count as f64,
            }
            .at(Stage::CanonicalDecomposition))
        );
    }
    for value in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            polynomial_roots(&[1.0, value, 1.0]),
            Err(Failure::Unusable {
                quantity: "root polynomial coefficient",
                index: Some(1),
                value,
            }
            .at(Stage::CanonicalDecomposition))
        );
    }
    let Err(error) = polynomial_roots(&[1.0, f64::NAN, 1.0]) else {
        panic!("a NaN coefficient must be rejected");
    };
    assert_eq!(error.stage, Stage::CanonicalDecomposition);
    assert!(matches!(
        error.failure,
        Failure::Unusable {
            quantity: "root polynomial coefficient",
            index: Some(1),
            value,
        } if value.is_nan()
    ));
}
