use super::*;

fn root(real: f64, imaginary: f64, modulus: f64) -> Root {
    Root {
        value: Complex { real, imaginary },
        modulus,
        argument_degrees: 0.0,
        period: 999.99,
    }
}

fn assert_complex_bits(actual: Complex, real: f64, imaginary: f64) {
    assert_eq!(
        [actual.real.to_bits(), actual.imaginary.to_bits()],
        [real.to_bits(), imaginary.to_bits()]
    );
}

#[test]
fn square_root_branches_and_signed_zeros() {
    for (real, imaginary, want_real, want_imaginary) in [
        (4.0, 0.0, 2.0, 0.0),
        (-4.0, 0.0, -0.0, -2.0),
        (-4.0, -0.0, 0.0, -2.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, -0.0, 0.0, 0.0),
        (3.0, 4.0, 2.0, 1.0),
        (-3.0, 4.0, 1.0, 2.0),
        (-3.0, -4.0, 1.0, -2.0),
        (0.0, 1e-16, (5e-17_f64).sqrt(), 0.0),
        (-5e-17, 0.0, 0.0, -(5e-17_f64).sqrt()),
        (1e-16, -0.0, 1e-8, -0.0),
        (-1e-16, 0.0, -0.0, -1e-8),
    ] {
        assert_complex_bits(
            square_root(Complex { real, imaginary }),
            want_real,
            want_imaginary,
        );
    }
}

#[test]
fn reciprocal_roots_keep_order_and_signed_zeros() {
    let roots = reciprocal_roots(Complex {
        real: -2.0,
        imaginary: -0.0,
    });
    assert_complex_bits(roots[0], 1.0, 0.0);
    assert_complex_bits(roots[1], 1.0, 0.0);
    let roots = reciprocal_roots(Complex {
        real: 0.0,
        imaginary: 0.0,
    });
    // ROOTC: SQROOTC(-4,+0) returns (-0,-2), then (-0 - +0)/2.
    assert_complex_bits(roots[0], -0.0, -1.0);
    // ROOTC: (-rez - a)/2 is (-0 - -0)/2 = +0.
    assert_complex_bits(roots[1], 0.0, 1.0);
}

#[test]
fn root_searches_are_inclusive_and_keep_first_ties() {
    let groups = [
        Group {
            root: root(1e-30, 0.0, 1.0),
            count: 1,
        },
        Group {
            root: root(0.0, 0.0, 1.0),
            count: 1,
        },
    ];
    assert_eq!(get_root(&groups, Complex::default()), Some(0));
    let roots = [
        root(0.5, 0.5, 1.0),
        root(0.5, -0.5, 1.0),
        root(0.5, -0.5, 1.0),
    ];
    assert_eq!(
        get_conjugate(
            &roots,
            1,
            Complex {
                real: 0.5,
                imaginary: -0.5
            }
        ),
        Some(2)
    );
    let groups = [
        Group {
            root: root(-1.0, 0.0, 1.0),
            count: 1,
        },
        Group {
            root: root(1.0, 0.0, 1.0),
            count: 1,
        },
    ];
    assert_eq!(closest_root(&groups, 0, Complex::default()), Some(0));
}

#[test]
fn closest_root_fuses_the_real_square() {
    // closestRoot rounds the imaginary difference squared, then fuses the
    // real one. For the stored 0.3 and 1.3, candidate 0's distance is
    // 3ffc7ae147ae147c and candidate 1's is 3ffc7ae147ae147b; fusing the
    // other square instead would exchange them and keep candidate 0.
    let small = f64::from_bits(0x3fd3_3333_3333_3333);
    let large = f64::from_bits(0x3ff4_cccc_cccc_cccd);
    let groups = [
        Group {
            root: root(small, large, 1.0),
            count: 1,
        },
        Group {
            root: root(large, small, 1.0),
            count: 1,
        },
    ];
    assert_eq!(closest_root(&groups, 0, Complex::default()), Some(1));
}

#[test]
fn grouping_joins_single_unit_roots() {
    let groups = group_roots(&[root(1.0, 0.0, 1.0), root(1.0 + 4e-16, 0.0, 1.0)]).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].count, 2);
    assert_complex_bits(groups[0].root.value, 1.0, 0.0);
    assert_eq!(groups[0].root.modulus, 1.0);
    assert_eq!(groups[0].root.argument_degrees, 0.0);
    assert_eq!(groups[0].root.period.to_bits(), 0x408f_3feb_8000_0000);
    let groups = group_roots(&[root(-1.0, 0.0, 1.0), root(-1.0 + 4e-16, 0.0, 1.0)]).unwrap();
    assert_complex_bits(groups[0].root.value, -1.0, 0.0);
    assert_eq!(groups[0].root.argument_degrees, 180.0);
    assert_eq!(groups[0].root.period, 2.0);
}

#[test]
fn join_root_fuses_the_current_weighted_products() {
    // JoinRoot rounds the partner's product, then fuses the current one;
    // counts of three expose both. The current coordinates are 1+2^-52 and
    // 1+3*2^-52 and the partner is (-1,-1), so the averages are 2^-53 and
    // 3*2^-53 and the norm is 10*2^-106.
    let mut groups = vec![
        Group {
            root: root(
                f64::from_bits(0x3ff0_0000_0000_0001),
                f64::from_bits(0x3ff0_0000_0000_0003),
                1.0,
            ),
            count: 3,
        },
        Group {
            root: root(-1.0, -1.0, 1.0),
            count: 3,
        },
    ];
    join_root(&mut groups, 0, 1).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].count, 6);
    assert_complex_bits(
        groups[0].root.value,
        f64::from_bits(0x3fd4_3d13_6248_490f),
        f64::from_bits(0x3fee_5b9d_136c_6d96),
    );
}

#[test]
fn half_roots_thresholds_adjacency_and_integer_halving() {
    let below_unit = f64::from_bits((1.0_f64 - 1e-13).to_bits() - 1);
    let groups = [Group {
        root: root(0.5, 0.0, 1.0),
        count: 3,
    }];
    assert_eq!(half_roots(&groups, 1).unwrap().len(), 1);
    let groups = [Group {
        root: root(0.5, 0.0, below_unit),
        count: 3,
    }];
    assert_eq!(half_roots(&groups, 3).unwrap().len(), 3);
    let inside_unit = f64::from_bits((1.0_f64 - 1e-13).to_bits() + 1);
    let groups = [Group {
        root: root(0.5, 0.0, inside_unit),
        count: 3,
    }];
    assert_eq!(half_roots(&groups, 1).unwrap().len(), 1);
    for (imaginary, want_count) in [(1e-10, 1), (f64::from_bits(1e-10_f64.to_bits() + 1), 2)] {
        let groups = [
            Group {
                root: root(0.5, imaginary, 0.5),
                count: 1,
            },
            Group {
                root: root(0.5, -imaginary, 2.0),
                count: 1,
            },
        ];
        assert_eq!(half_roots(&groups, want_count).unwrap().len(), want_count);
    }
    let groups = [
        Group {
            root: root(0.0, 0.5, 0.5),
            count: 1,
        },
        Group {
            root: root(1e-10, -0.5, 2.0),
            count: 1,
        },
    ];
    assert_eq!(half_roots(&groups, 1).unwrap().len(), 1);
}

/// MPBC with its output aliased to the left operand, which it copies first.
fn multiply_in_place(output: &mut Vec<Complex>, right: &[Complex]) {
    let left = std::mem::take(output);
    multiply_into(&left, right, output);
}

#[test]
fn multiply_fuses_products_and_rounds_accumulation_separately() {
    let x = 1.0 + 2.0_f64.powi(-27);
    let y = 1.0 - 2.0_f64.powi(-27);
    let mut output = vec![Complex {
        real: x,
        imaginary: 1.0,
    }];
    multiply_in_place(
        &mut output,
        &[Complex {
            real: y,
            imaginary: 1.0,
        }],
    );
    assert_complex_bits(output[0], -2.0_f64.powi(-54), 2.0);
    let mut output = vec![
        Complex {
            real: -1.0,
            imaginary: 0.0,
        },
        Complex {
            real: x,
            imaginary: 0.0,
        },
    ];
    multiply_in_place(
        &mut output,
        &[
            Complex {
                real: y,
                imaginary: 0.0,
            },
            Complex {
                real: 1.0,
                imaginary: 0.0,
            },
        ],
    );
    assert_complex_bits(output[1], 0.0, 0.0);
    assert_ne!(x.mul_add(y, -1.0).to_bits(), output[1].real.to_bits());
    let mut output = vec![
        Complex {
            real: 1.0,
            imaginary: 0.0,
        },
        Complex {
            real: -2.0,
            imaginary: -0.0,
        },
    ];
    multiply_in_place(
        &mut output,
        &[
            Complex {
                real: 1.0,
                imaginary: 0.0,
            },
            Complex {
                real: -3.0,
                imaginary: -0.0,
            },
        ],
    );
    for (actual, expected) in output.iter().zip([1.0, -5.0, 6.0]) {
        assert_complex_bits(*actual, expected, 0.0);
    }
}

#[test]
fn unpaired_or_unselectable_roots_are_invalid() {
    let unusable =
        |step| Failure::UnusableRoots { step, index: None }.at(Stage::CanonicalDecomposition);
    let missing = [root(0.5, 0.5, 0.5)];
    assert_eq!(group_roots(&missing), Err(unusable("grouping")));
    let groups = [Group {
        root: missing[0],
        count: 2,
    }];
    assert_eq!(half_roots(&groups, 2), Err(unusable("selection")));
    let groups = [Group {
        root: root(0.5, 0.0, 0.5),
        count: 1,
    }];
    assert_eq!(half_roots(&groups, 2), Err(unusable("selection")));
}

#[test]
fn rejects_unsupported_numerators() {
    let mut beyond = vec![0.0; MAK1_MAX_COEFFICIENTS + 1];
    beyond[0] = 1.0;
    for input in [vec![], vec![1.0], beyond] {
        assert_eq!(
            factor_numerator(&input),
            Err(Failure::Unusable {
                quantity: "spectral numerator coefficient count",
                index: None,
                value: input.len() as f64,
            }
            .at(Stage::CanonicalDecomposition))
        );
    }
    assert_eq!(
        factor_numerator(&[1.0, f64::INFINITY, 1.0]),
        Err(Failure::Unusable {
            quantity: "spectral numerator coefficient",
            index: Some(1),
            value: f64::INFINITY,
        }
        .at(Stage::CanonicalDecomposition))
    );
}

#[test]
fn recovers_a_known_factor() {
    let direct = factor_numerator(&[2.0, -2.0]).unwrap();
    assert_eq!(direct.coefficients, [1.0, -1.0]);
    assert_eq!(direct.variance, 1.0);
    for n in [5, 13] {
        let mut ma = vec![0.0; n];
        ma[0] = 1.0;
        ma[n - 1] = -0.4;
        let output = factor_numerator(&conjugate(&ma, &ma)).unwrap();
        assert_eq!(output.coefficients.len(), n);
        assert!((output.variance - 1.0).abs() < 1e-10);
        for (&a, &b) in output.coefficients.iter().zip(&ma) {
            assert!((a - b).abs() < 1e-10);
        }
    }
    // SYMPOLY loses accuracy at these degrees, in Build 62 too, so only the
    // factor's length is checked.
    for n in [26, MAK1_MAX_COEFFICIENTS] {
        let mut ma = vec![0.0; n];
        ma[0] = 1.0;
        ma[n - 1] = -0.4;
        let output = factor_numerator(&conjugate(&ma, &ma)).unwrap();
        assert_eq!(output.coefficients.len(), n);
    }
}

#[test]
fn zero_lag_one_gives_the_white_noise_factor() {
    // MAK1's zero divisor leaves IA=0: THETA stays [1] and VAR=GAMZER, the
    // signed zero included.
    let white = factor_numerator(&[1.5, 0.0]).unwrap();
    assert_eq!(white.coefficients, [1.0]);
    assert_eq!(white.variance, 1.5);
    let zero = factor_numerator(&[-0.0, 0.0]).unwrap();
    assert_eq!(zero.coefficients, [1.0]);
    assert_eq!(zero.variance.to_bits(), (-0.0f64).to_bits());
}
