use super::*;

#[test]
fn lm_parameter_respects_a_rectangular_jacobians_leading_dimension() {
    let order = 2;
    let mut compact_r = vec![2.0, 0.0, 0.5, 1.5];
    let mut rectangular_r = vec![0.0; 5 * order];
    let mut rectangular = ColMajorMut::new(&mut rectangular_r, 5);
    rectangular[(0, 0)] = 2.0;
    rectangular[(0, 1)] = 0.5;
    rectangular[(1, 1)] = 1.5;
    let region = TrustRegion {
        permutation: &[0, 1],
        diagonal: &[1.0, 1.0],
        q_transpose_b: &[1.0, -0.5],
        delta: 0.1,
    };

    let compact = solve_lm_parameter(ColMajorMut::new(&mut compact_r, order), &region);
    let rectangular = solve_lm_parameter(rectangular, &region);

    assert_eq!(rectangular, compact);
}

#[test]
fn qr_solve_uses_build_62_subtractive_givens_rounding() {
    // Expected bits were calculated independently from `qrsolv.f`'s rotation
    // order: round sine * old first, then fuse cosine * old - it.
    let mut r = [
        0.9079292834884649,
        0.0,
        0.0,
        1.596980825451498,
        -1.4733658655180597,
        0.0,
        -0.859017959827407,
        1.732584331438071,
        -0.19152613041941668,
    ];
    let permutation = [0, 1, 2];
    let diagonal = [-0.18701844952440094, 1.155596137037977, 1.9423182707984923];
    let q_transpose_b = [
        -0.039610386787262364,
        1.8081538697707384,
        0.08863631344025524,
    ];
    let mut solution = [0.0; 3];
    let mut s_diagonal = [0.0; 3];
    let mut work = [0.0; 3];
    qr_solve(
        &mut ColMajorMut::new(&mut r, 3),
        &permutation,
        &diagonal,
        &q_transpose_b,
        &mut solution,
        &mut s_diagonal,
        &mut work,
    );
    assert_eq!(
        solution.map(f64::to_bits),
        [
            0x3ff2_1347_a6d9_fbf8,
            0xbfe2_15a6_0d3d_fc06,
            0x3fce_bd91_10ea_d3d3,
        ]
    );
}

#[test]
fn qr_solve_rounds_the_transformed_rhs_like_qrsolv() {
    // This 3x3 case is sensitive to the other subtractive rotation: the
    // update of `qtbpj` before a later row reads it.
    let mut r = [
        0.7065579158663979,
        0.0,
        0.0,
        0.294842342412712,
        -1.6018359964156224,
        0.0,
        -0.6696782535293173,
        -0.06987415261265939,
        -0.16293884745475085,
    ];
    let permutation = [0, 1, 2];
    let diagonal = [0.13984863743376286, 0.17400725193325206, 1.5370405408225434];
    let q_transpose_b = [
        0.8228164196338179,
        -0.31352049881582955,
        0.03760784465995348,
    ];
    let mut solution = [0.0; 3];
    let mut s_diagonal = [0.0; 3];
    let mut work = [0.0; 3];
    qr_solve(
        &mut ColMajorMut::new(&mut r, 3),
        &permutation,
        &diagonal,
        &q_transpose_b,
        &mut solution,
        &mut s_diagonal,
        &mut work,
    );
    assert_eq!(
        solution.map(f64::to_bits),
        [
            0x3ff0_8238_5b76_5b22,
            0x3fc9_3c05_780c_07e2,
            0xbf85_bd92_3905_5f36,
        ]
    );
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn options(
    function_tolerance: f64,
    parameter_tolerance: f64,
    gradient_tolerance: f64,
    maximum_iterations: usize,
) -> LmOptions {
    LmOptions {
        function_tolerance,
        parameter_tolerance,
        gradient_tolerance,
        maximum_iterations,
        finite_difference_epsilon: 0.0,
        factor: 100.0,
    }
}

/// One `lmdif` run, bit for bit: exit, counters, final parameters,
/// residuals and the returned work diagonal.
fn lmdif_summary(
    parameters: &mut [f64],
    residual_count: usize,
    options: LmOptions,
    mut residual: impl FnMut(&[f64], &mut [f64], LmEvaluation),
) -> String {
    let mut residuals = vec![0.0; residual_count];
    let mut counters = LmCounters::default();
    let outcome = lmdif(
        parameters,
        &mut residuals,
        options,
        &mut counters,
        |candidate, output, evaluation| {
            residual(candidate, output, evaluation);
            Ok(())
        },
    );
    let tail = format!(
        "{}/{} p={:x?} r={:x?}",
        counters.iterations,
        counters.function_evaluations,
        bits(parameters),
        bits(&residuals)
    );
    match outcome {
        Ok(outcome) => format!(
            "{:?} {tail} d={:x?}",
            outcome.exit,
            outcome.work_diagonal.as_deref().map(bits)
        ),
        Err(error) => format!("{error:?} {tail}"),
    }
}

const DECAY_TIMES: [f64; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
const DECAY_VALUES: [f64; 6] = [2.0, 1.3, 0.9, 0.55, 0.4, 0.25];

fn decay(parameters: &[f64], output: &mut [f64], _: LmEvaluation) {
    for ((output, time), value) in output.iter_mut().zip(DECAY_TIMES).zip(DECAY_VALUES) {
        *output = parameters[0] * (-parameters[1] * time).exp() - value;
    }
}

fn rosenbrock(parameters: &[f64], output: &mut [f64], _: LmEvaluation) {
    output[0] = 10.0 * (parameters[1] - parameters[0] * parameters[0]);
    output[1] = 1.0 - parameters[0];
}

#[test]
fn lmdif_fits_an_exponential_decay() {
    assert_eq!(
        lmdif_summary(&mut [1.0, 0.1], 6, options(1e-10, 0.0, 0.0, 100), decay),
        "RelativeReduction 5/16 p=[3fffe9aaa9be1141, 3fda7430a06e889a] r=[bf76555641eebf00, \
        3f93ba79ab994280, bf9c0c065ec8d240, 3f9bd40faafc6880, bf92ac2365550a60, \
        3f6496e35eee5680] d=Some([c00b605ccf0e8919, bff179672c61b0c4])",
    );
}

#[test]
fn lmdif_solves_the_rosenbrock_valley() {
    assert_eq!(
        lmdif_summary(
            &mut [-1.2, 1.0],
            2,
            options(1e-12, 0.0, 0.0, 200),
            rosenbrock,
        ),
        "Orthogonal 16/56 p=[3ff0000000000000, 3ff0000000000000] r=[0, 0] \
        d=Some([4034066561ccb931, 3fdff5c77101c9c5])",
    );
}

#[test]
fn lmdif_stops_at_the_iteration_limit() {
    assert_eq!(
        lmdif_summary(&mut [-1.2, 1.0], 2, options(1e-12, 0.0, 0.0, 2), rosenbrock,),
        "EvaluationOrIterationLimit 2/9 p=[bfe8dd05c0186c62, 3fe2ee5b9a6fc707] \
        r=[bfbefb211b00b260, 3ffc6e82e00c3631] d=Some([c031fa0301e75e80, \
        bfe1cd087d1e879b])",
    );
}

#[test]
fn lmdif_parameter_tolerance_exits() {
    for (function_tolerance, parameter_tolerance, expected) in [
        (
            1e-10,
            0.5,
            "RelativeStep 2/7 p=[3fffe93b501d5ead, 3fda5affdf9d78d2] r=[bf76c4afe2a15300, \
            3f95bc2b634d2940, bf9957c0a6004160, 3f9e8777c5f56a60, bf9048201e12da90, \
            3f723893dca3b800] d=Some([c0086daf7227509e, bff163308f75dc4a])",
        ),
        (
            1.0,
            0.5,
            "RelativeReduction 1/4 p=[3ffddc6c12cb6d92, 3fdb8190447ec852] \
            r=[bfc11c9f69a49370, bfb5ef44db026420, bfbc22a6d8b34858, bfa26494b8e0d2d0, \
            bfb0c5a8cb7aff94, bfa09286c0a230cc] d=Some([c013ee1b491e94a8, \
            bff4639026bab171])",
        ),
        (
            1e-10,
            1e-3,
            "RelativeStep 4/13 p=[3fffe9aa9f6c1b3a, 3fda74308440dc96] r=[bf76556093e4c600, \
            3f93ba7a49860380, bf9c0c046ce82d20, 3f9bd411f885ce60, bf92ac2133454a20, \
            3f6496f28bd92a00] d=Some([c00b608137caf698, bff1796a2f658fb4])",
        ),
        (
            1.0,
            1e6,
            "RelativeReductionAndStep 1/4 p=[3ffddc6c12cb6d92, 3fdb8190447ec852] \
            r=[bfc11c9f69a49370, bfb5ef44db026420, bfbc22a6d8b34858, bfa26494b8e0d2d0, \
            bfb0c5a8cb7aff94, bfa09286c0a230cc] d=Some([c013ee1b491e94a8, \
            bff4639026bab171])",
        ),
    ] {
        assert_eq!(
            lmdif_summary(
                &mut [1.0, 0.1],
                6,
                options(function_tolerance, parameter_tolerance, 0.0, 100),
                decay,
            ),
            expected,
        );
    }
}

#[test]
fn lmdif_recovers_from_rejected_constrained_trials() {
    assert_eq!(
        lmdif_summary(
            &mut [0.0, 0.0],
            3,
            options(1e-10, 0.0, 0.0, 50),
            |parameters, output, evaluation| {
                if evaluation == LmEvaluation::ConstrainedTrial && parameters[0].abs() >= 1.5 {
                    output.fill(1e3);
                } else {
                    output[0] = parameters[0] - 2.0;
                    output[1] = 0.5f64.mul_add(parameters[1], -0.3);
                    output[2] = parameters[0] * parameters[1] - 0.1;
                }
            },
        ),
        "RelativeReduction 28/104 p=[3ff7fffffffcd66d, 3fc7241360e453d3] \
        r=[bfe0000000065326, bfcad45cb5f43c7c, 3fc5e95044851e0b] d=Some([bff94c5838749822, \
        3ff006b035dace7a])",
    );
}

#[test]
fn lmdif_handles_a_rank_deficient_jacobian() {
    assert_eq!(
        lmdif_summary(
            &mut [0.0, 0.0],
            3,
            options(1e-10, 0.0, 0.0, 50),
            |parameters, output, _| {
                let sum = parameters[0] + parameters[1];
                output.copy_from_slice(&[sum - 1.0, 2.0 * sum - 1.0, sum - 3.0]);
            },
        ),
        "RelativeReduction 1/7 p=[403d5e155b8addc8, c03c5e155b852a3b] r=[3e16ce3400000000, \
        3ff0000000b671a0, bfffffffffa4c730] d=Some([c003988e155166d6, bc6f024b58a1e3d4])",
    );
}

#[test]
fn lmdif_handles_a_zero_jacobian_column() {
    assert_eq!(
        lmdif_summary(
            &mut [0.5, 0.5],
            2,
            options(1e-10, 0.0, 0.0, 50),
            |parameters, output, _| {
                output.copy_from_slice(&[parameters[0] - 1.0, 2.0 * parameters[0] - 3.0]);
            },
        ),
        "RelativeReduction 2/7 p=[3ff6666666666667, 3fe0000000000000] r=[3fd999999999999c, \
        bfc9999999999990] d=Some([c001e3779c4522fd, 8000000000000000])",
    );
}

#[test]
fn lmdif_solves_an_exact_linear_system() {
    assert_eq!(
        lmdif_summary(
            &mut [0.0, 0.0],
            3,
            options(0.0, 0.0, 0.0, 50),
            |parameters, output, _| {
                output.copy_from_slice(&[
                    parameters[0] - 1.0,
                    parameters[1] - 2.0,
                    parameters[0] + parameters[1] - 3.0,
                ]);
            },
        ),
        "ParameterToleranceTooSmall 2/11 p=[3ff0000000000001, 4000000000000000] \
        r=[3cb0000000000000, 0, 0] d=Some([bff6a09e67e0f1e8, 3ff3988e13996806])",
    );
}

#[test]
fn lmdif_reports_a_nonfinite_trial_residual() {
    assert_eq!(
        lmdif_summary(
            &mut [0.0, 0.0],
            2,
            options(1e-10, 0.0, 0.0, 50),
            |parameters, output, evaluation| {
                output[0] = parameters[0] - 1.0;
                output[1] = if evaluation == LmEvaluation::ConstrainedTrial {
                    f64::NAN
                } else {
                    parameters[1]
                };
            },
        ),
        "NonFiniteResidual { index: 1 } 0/4 p=[0, 0] r=[bff0000000000000, 0]",
    );
}

fn qr_summary(row_count: usize, column_count: usize, mut matrix: Vec<f64>) -> String {
    let mut factors = QrFactors::new(column_count);
    qr_factor(&mut ColMajorMut::new(&mut matrix, row_count), &mut factors);
    format!(
        "{:?} r={:x?} n={:x?} m={:x?}",
        factors.permutation,
        bits(&factors.r_diagonal),
        bits(&factors.column_norms),
        bits(&matrix)
    )
}

#[test]
fn qr_factor_pivots_and_renormalizes_columns() {
    // The second column is nearly collinear with the first, so its
    // downdated norm falls under the qrfac renormalization threshold.
    assert_eq!(
        qr_summary(
            4,
            3,
            vec![
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0 + 1e-9,
                3.0,
                -1.0,
                2.0,
                0.5,
            ],
        ),
        "[2, 1, 0] r=[c00e3307cc56cf5c, bff9b1ced4162b08, be0ce6924cb25d55] \
        n=[4000000000000000, 4000000000112e0c, 400e3307cc56cf5c] m=[3ffcb72bb45a7242, \
        bfd0f43a45cdedad, 3fe0f43a45cdedad, 3fc0f43a45cdedad, bff312c18e90c589, \
        3ffd2fd6bf5a5366, 3fcc2076d78f9e40, 3fe0b3e59aa1dc7c, bff312c18e87ab64, \
        bff9b1ced3f22210, 3ff1fd7b2d16c7af, bfefc06191c5bd31]",
    );
}

#[test]
fn qr_factor_handles_tied_and_zero_columns() {
    assert_eq!(
        qr_summary(
            4,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,],
        ),
        "[1, 2, 0] r=[c000000000000000, c000000000000000, 8000000000000000] n=[0, \
        4000000000000000, 4000000000000000] m=[3ff8000000000000, bfe0000000000000, \
        3fe0000000000000, 3fe0000000000000, 3c90000000000000, 3ffaaaaaaaaaaaaa, \
        bfe5555555555555, 3fd5555555555556, 0, 0, 0, 0]",
    );
}

#[test]
fn lm_parameter_with_singular_r_and_an_inactive_trust_region() {
    // R's second diagonal is zero, so lmpar takes the singular branch;
    // the large delta then accepts the Gauss-Newton step outright.
    let r = [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.3, 1.0];
    for (delta, expected) in [
        (
            0.1,
            "40345ea80434b803 x=[3fb1ff517946aa03, 3f76b86f893cc9c3, 3fc1ff517946aa04] \
            s=[4013bf0cf223da52, 40037a7daec78617, 4022317843192f42]",
        ),
        (100.0, "0 x=[3fe0000000000000, 0, 0] s=[0, 0, 0]"),
    ] {
        let (parameter, solution, s_diagonal) = solve_lm_parameter(
            ColMajorMut::new(&mut r.clone(), 3),
            &TrustRegion {
                permutation: &[0, 2, 1],
                diagonal: &[1.0, 2.0, 0.5],
                q_transpose_b: &[1.0, -0.5, 0.25],
                delta,
            },
        );
        assert_eq!(
            format!(
                "{:x} x={:x?} s={:x?}",
                parameter.to_bits(),
                bits(&solution),
                bits(&s_diagonal)
            ),
            expected,
        );
    }
}

fn solve_lm_parameter(
    mut r: ColMajorMut<'_>,
    region: &TrustRegion<'_>,
) -> (f64, Vec<f64>, Vec<f64>) {
    let order = region.permutation.len();
    let mut solution = vec![0.0; order];
    let mut scratch = LmParameterScratch::new(order);
    let parameter = lm_parameter(&mut r, region, 0.0, &mut solution, &mut scratch);
    (parameter, solution, scratch.s_diagonal)
}

#[test]
fn orthogonal_exit_keeps_pivot_order_work_diagonal() {
    let mut parameters = [0.0, 0.0];
    let mut residuals = [0.0; 3];
    let mut counters = LmCounters::default();
    let outcome = lmdif(
        &mut parameters,
        &mut residuals,
        options(0.0, 0.0, 1.0, 10),
        &mut counters,
        |candidate, output, _| {
            output.copy_from_slice(&[candidate[0], 2.0 * candidate[1], 0.0]);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(outcome.exit, LmExit::Orthogonal);
    assert_eq!(
        counters.iterations, 0,
        "orthogonal exit must not take a step"
    );
    assert_eq!(outcome.work_diagonal, Some(vec![-2.0, 1.0]));
    assert_eq!(parameters, [0.0, 0.0]);
}

#[test]
fn converged_exit_retains_qr_diagonal() {
    let mut parameters = [0.0];
    let mut residuals = [0.0];
    let mut counters = LmCounters::default();
    let outcome = lmdif(
        &mut parameters,
        &mut residuals,
        options(2.0, 0.0, 0.0, 10),
        &mut counters,
        |candidate, output, _| {
            output[0] = 1.0 + candidate[0];
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(outcome.exit, LmExit::RelativeReduction);
    assert_eq!(counters.iterations, 1);
    // First QR uses `((1 + h) - 1) / h` with the existing fdjac2 epsilon.
    assert_eq!(
        outcome.work_diagonal.unwrap()[0].to_bits(),
        0xbfeffffffdf4392f
    );
}

#[test]
fn improper_input_returns_before_evaluating() {
    let mut parameters = [0.0, 0.0];
    let mut residuals = [0.0];
    let mut counters = LmCounters::default();
    let outcome = lmdif(
        &mut parameters,
        &mut residuals,
        options(0.0, 0.0, 0.0, 10),
        &mut counters,
        |_, _, _| panic!("improper input must not evaluate residuals"),
    )
    .unwrap();
    assert_eq!(outcome.exit, LmExit::ImproperInput);
    assert_eq!(outcome.work_diagonal, None);
    assert_eq!(counters, LmCounters::default());
}
