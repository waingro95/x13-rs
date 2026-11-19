use super::*;

#[test]
fn a_linear_model_has_its_least_squares_covariance() {
    for (variance, expected_matrix) in [(0.1, vec![0.1, 0.0, 0.0, 0.025]), (0.0, vec![0.0; 4])] {
        let mut parameters = [0.0, 0.0];
        let covariance = parameter_covariance(
            &mut parameters,
            &[0.0, 0.0, 0.0],
            0.0,
            variance,
            |candidate, output| {
                output.copy_from_slice(&[candidate[0], 2.0 * candidate[1], 0.0]);
            },
        );
        assert_eq!(
            covariance,
            ParameterCovariance::Calculated {
                matrix: expected_matrix,
                unscaled_diagonal: vec![1.0, 0.25],
            }
        );
    }
}

/// The matrix, if there is one, and the unscaled diagonal, as bits.
fn bits(covariance: &ParameterCovariance) -> (Option<Vec<u64>>, Vec<u64>) {
    let bits = |values: &[f64]| values.iter().map(|value| value.to_bits()).collect();
    match covariance {
        ParameterCovariance::Calculated {
            matrix,
            unscaled_diagonal,
        } => (Some(bits(matrix)), bits(unscaled_diagonal)),
        ParameterCovariance::Singular { unscaled_diagonal } => (None, bits(unscaled_diagonal)),
    }
}

/// A three-parameter nonlinear covariance at several rank tolerances: zero
/// (machine epsilon), a small positive one, and two large enough to cut the
/// pivoted triangle after its second and first columns.
#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    ignore = "pins exact results of the Apple silicon math library"
)]
fn a_nonlinear_covariance_at_each_rank_tolerance() {
    const TIMES: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
    const VALUES: [f64; 5] = [1.1, 0.9, 0.85, 0.7, 0.72];
    let full_rank = (
        Some(vec![
            0x3fa3_2414_449e_d2ca,
            0xbfa5_3a1c_493c_8751,
            0x3f85_6d0d_06f5_764a,
            0xbfa5_3a1c_493c_8751,
            0x3fc4_84a8_f8c3_2963,
            0xbfac_35e2_43da_2675,
            0x3f85_6d0d_06f5_764a,
            0xbfac_35e2_43da_2675,
            0x3f95_4f55_d7f4_329b,
        ]),
        vec![
            0x3fed_e85f_ab38_295c,
            0x4010_07a4_0258_7855,
            0x3fe0_a5fb_10b6_c789,
        ],
    );
    for (tolerance, expected) in [
        (0.0, full_rank.clone()),
        (1e-4, full_rank),
        (
            0.1,
            (None, vec![0x3fe5_5458_c37e_63f0, 0, 0x3fa7_f520_ab4d_c944]),
        ),
        (0.9, (None, vec![0, 0, 0x3fa1_1111_19a6_296d])),
    ] {
        let mut parameters = [1.0, -0.3, 0.2];
        let residuals: Vec<f64> = TIMES
            .iter()
            .zip(VALUES)
            .map(|(time, value)| (-0.3 * time).exp() + 0.2 * time - value)
            .collect();
        let covariance = parameter_covariance(
            &mut parameters,
            &residuals,
            tolerance,
            0.04,
            |candidate, output| {
                for ((output, time), value) in output.iter_mut().zip(TIMES).zip(VALUES) {
                    *output =
                        candidate[0] * (candidate[1] * time).exp() + candidate[2] * time - value;
                }
            },
        );
        assert_eq!(bits(&covariance), expected, "tolerance={tolerance}");
    }
}

#[test]
fn a_rank_deficient_jacobian_has_no_matrix_but_keeps_the_diagonal() {
    let mut parameters = [0.0, 0.0];
    let covariance = parameter_covariance(
        &mut parameters,
        &[0.0, 0.0, 0.0],
        0.0,
        0.1,
        |candidate, output| {
            output.copy_from_slice(&[candidate[0] + 2.0 * candidate[1], 0.0, 0.0]);
        },
    );
    assert_eq!(
        covariance,
        ParameterCovariance::Singular {
            unscaled_diagonal: vec![0.0, 0.25]
        }
    );
}
