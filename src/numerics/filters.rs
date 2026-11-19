//! The X-11 moving averages: M-of-N simple averages, the 3x5 seasonal
//! filter, and Henderson trend filters with their endpoint weights
//! (`averag.f`, `vsfb.f`, `hender.f`, `hndtrn.f`, `hndend.f`).

use std::ops::Range;

use super::BUILD_62_PI;
use crate::x11;

/// Build 62 `averag.f`: apply an M-of-N moving average within `span`.
pub(crate) fn moving_average(
    input: &[f64],
    output: &mut [f64],
    span: Range<usize>,
    m: usize,
    n: usize,
) {
    debug_assert_eq!(input.len(), output.len());
    // Bad orders or an inverted span would otherwise return wrong or unwritten
    // output without a panic.
    assert!(m > 0 && n > 0 && (m + n).is_multiple_of(2));
    assert!(span.start <= span.end && span.end <= input.len());

    let offset = (m + n) / 2 - 1;
    if span.len() < 2 * offset + 1 {
        return;
    }

    for (center, output_value) in output[..span.end - offset]
        .iter_mut()
        .enumerate()
        .skip(span.start + offset)
    {
        let first_window = center - offset;
        let mut total = 0.0;
        for window_start in first_window..first_window + m {
            for value in &input[window_start..window_start + n] {
                total += value;
            }
        }
        *output_value = total / (m * n) as f64;
    }
}

/// Build 62 `hender.f`: generate the full symmetric N-term filter.
pub(crate) fn henderson_weights(terms: usize) -> Vec<f64> {
    debug_assert!(terms > 1 && terms % 2 == 1);

    let y = ((terms + 3) / 2) as f64;
    let y1 = (y - 1.0) * (y - 1.0);
    let y2 = y * y;
    let y3 = (y + 1.0) * (y + 1.0);
    let y4 = 3.0 * y2 - 16.0;
    let y5 = 4.0 * y2;
    let denominator = 8.0 * y * (y2 - 1.0) * (y5 - 1.0) * (y5 - 9.0) * (y5 - 25.0) / 315.0;
    let half_length = terms.div_ceil(2);
    let mut center_out = Vec::with_capacity(half_length);
    for index in 0..half_length {
        let x = (index * index) as f64;
        center_out.push((y1 - x) * (y2 - x) * (y3 - x) * (y4 - 11.0 * x) / denominator);
    }

    let mut weights = Vec::with_capacity(terms);
    weights.extend(center_out[1..].iter().rev().copied());
    weights.extend(center_out);
    weights
}

/// Build 62 `vsfb.f`: S3x5 smoothing for one calendar subseries.
pub(crate) fn s3x5_subseries(input: &[f64]) -> Vec<f64> {
    debug_assert!(input.len() >= 4);

    let last = input.len() - 1;
    let mut output = vec![f64::NAN; input.len()];
    moving_average(input, &mut output, 0..input.len(), 3, 5);

    // In `vsfb.f`'s endpoint formulas gfortran computes one product on its
    // own and fuses every other multiply into its addition; the division
    // comes last. The right endpoints mirror the left ones.
    output[0] = 17.0f64.mul_add(input[0] + input[1] + input[2], 9.0 * input[3]) / 60.0;
    output[last] = 17.0f64.mul_add(
        input[last] + input[last - 1] + input[last - 2],
        9.0 * input[last - 3],
    ) / 60.0;

    if input.len() == 4 {
        let average = input.iter().sum::<f64>() / 4.0;
        output[1] = average;
        output[2] = average;
        return output;
    }

    output[1] = 4.0f64.mul_add(
        input[4],
        15.0f64.mul_add(input[0] + input[1] + input[2], 11.0 * input[3]),
    ) / 60.0;
    output[last - 1] = 4.0f64.mul_add(
        input[last - 4],
        15.0f64.mul_add(
            input[last] + input[last - 1] + input[last - 2],
            11.0 * input[last - 3],
        ),
    ) / 60.0;

    if input.len() == 5 {
        output[2] = input.iter().sum::<f64>() / 5.0;
    } else {
        output[2] = 4.0f64.mul_add(
            input[5],
            8.0f64.mul_add(
                input[4],
                9.0f64.mul_add(input[0], 13.0 * (input[1] + input[2] + input[3])),
            ),
        ) / 60.0;
        output[last - 2] = 4.0f64.mul_add(
            input[last - 5],
            8.0f64.mul_add(
                input[last - 4],
                9.0f64.mul_add(
                    input[last],
                    13.0 * (input[last - 1] + input[last - 2] + input[last - 3]),
                ),
            ),
        ) / 60.0;
    }

    output
}

/// Build 62 `hndtrn.f`: central Henderson weights plus Doherty endpoints.
///
/// Returns the filtered series and the endpoint I/C ratio `hndtrn.f` leaves
/// behind: the legacy 7-term endpoints reset it to 0.001, which later trend
/// passes then inherit.
pub(crate) fn henderson_filter(
    input: &[f64],
    terms: usize,
    mut trend_ic: f64,
    policy: x11::SevenTermEndpoints,
) -> (Vec<f64>, f64) {
    debug_assert!(terms > 1 && terms % 2 == 1);
    // A series one term short would otherwise skip the central filter silently.
    assert!(input.len() >= terms);

    let mut endpoint_terms = terms;
    let mut output = vec![0.0; input.len()];
    apply_symmetric_henderson(
        input,
        &mut output,
        terms,
        terms / 2..input.len() - terms / 2,
    );

    // Build 62 defaults to the legacy X-11 behavior for a 7-term filter:
    // use the 5-term central filter at the next pair of endpoint positions.
    if terms == 7 && policy == x11::SevenTermEndpoints::LegacyFiveTerm {
        endpoint_terms = 5;
        apply_symmetric_henderson(input, &mut output, 5, 2..3);
        apply_symmetric_henderson(input, &mut output, 5, input.len() - 3..input.len() - 2);
        trend_ic = 0.001;
    }

    let central = henderson_weights(endpoint_terms);
    let half = endpoint_terms / 2;
    // `hndtrn.f` writes the shortened pi.
    let r_beta = 4.0 / (trend_ic * trend_ic * BUILD_62_PI);
    for endpoint_index in 0..half {
        let available = half + 1 + endpoint_index;
        let weights = henderson_end_weights(available, &central, r_beta);
        let mut beginning = 0.0;
        let mut end = 0.0;
        for input_index in 0..available {
            let weight = weights[available - input_index - 1];
            beginning = weight.mul_add(input[input_index], beginning);
            end = weight.mul_add(input[input.len() - input_index - 1], end);
        }
        output[endpoint_index] = beginning;
        output[input.len() - endpoint_index - 1] = end;
    }

    (output, trend_ic)
}

#[cfg(feature = "__conditioning-scales")]
pub(crate) fn henderson_filter_operation_scales(
    input: &[f64],
    terms: usize,
    trend_ic: f64,
    policy: x11::SevenTermEndpoints,
) -> Vec<f64> {
    debug_assert!(terms > 1 && terms % 2 == 1);
    // A series one term short would otherwise skip the central filter silently.
    assert!(input.len() >= terms);

    let mut endpoint_terms = terms;
    let mut scales = vec![0.0; input.len()];
    apply_symmetric_henderson_operation_scales(
        input,
        &mut scales,
        terms,
        terms / 2..input.len() - terms / 2,
    );

    if terms == 7 && policy == x11::SevenTermEndpoints::LegacyFiveTerm {
        endpoint_terms = 5;
        apply_symmetric_henderson_operation_scales(input, &mut scales, 5, 2..3);
        apply_symmetric_henderson_operation_scales(
            input,
            &mut scales,
            5,
            input.len() - 3..input.len() - 2,
        );
    }

    let central = henderson_weights(endpoint_terms);
    let half = endpoint_terms / 2;
    let r_beta = 4.0 / (trend_ic * trend_ic * BUILD_62_PI);
    for endpoint_index in 0..half {
        let available = half + 1 + endpoint_index;
        let weights = henderson_end_weights(available, &central, r_beta);
        let mut beginning = 0.0;
        let mut end = 0.0;
        for input_index in 0..available {
            let weight = weights[available - input_index - 1];
            beginning += (weight * input[input_index]).abs();
            end += (weight * input[input.len() - input_index - 1]).abs();
        }
        scales[endpoint_index] = beginning.max(1.0);
        scales[input.len() - endpoint_index - 1] = end.max(1.0);
    }

    scales
}

fn apply_symmetric_henderson(
    input: &[f64],
    output: &mut [f64],
    terms: usize,
    centers: Range<usize>,
) {
    let weights = henderson_weights(terms);
    let half = terms / 2;
    for center in centers {
        let mut value = weights[half] * input[center];
        for distance in 1..=half {
            value = weights[half + distance]
                .mul_add(input[center - distance] + input[center + distance], value);
        }
        output[center] = value;
    }
}

#[cfg(feature = "__conditioning-scales")]
fn apply_symmetric_henderson_operation_scales(
    input: &[f64],
    scales: &mut [f64],
    terms: usize,
    centers: Range<usize>,
) {
    let weights = henderson_weights(terms);
    let half = terms / 2;
    for center in centers {
        let mut scale = (weights[half] * input[center]).abs();
        for distance in 1..=half {
            scale += (weights[half + distance]
                * (input[center - distance] + input[center + distance]))
                .abs();
        }
        scales[center] = scale.max(1.0);
    }
}

/// Build 62 `hndend.f`: Doherty endpoint weights for `available` input terms.
fn henderson_end_weights(available: usize, central: &[f64], r_beta: f64) -> Vec<f64> {
    let mut omitted_sum = 0.0;
    let mut omitted_moment = 0.0;
    for (index, &weight) in central.iter().enumerate().skip(available) {
        omitted_sum += weight;
        let delta = (index + 1) as f64 - (available + 1) as f64 / 2.0;
        // gfortran fuses this multiply-add in `hndend.f`.
        omitted_moment = weight.mul_add(delta, omitted_moment);
    }

    let mut end_weights = Vec::with_capacity(available);
    for (index, &weight) in central.iter().take(available).enumerate() {
        let c1 = ((index + 1) as f64 - (available + 1) as f64 / 2.0) * r_beta;
        let denominator_factor = (available * (available - 1) * (available + 1)) as f64 / 12.0;
        // gfortran fuses this multiply-add in `hndend.f` too, and nothing in
        // the weight below.
        let c2 = r_beta.mul_add(denominator_factor, 1.0);
        end_weights.push(weight + omitted_sum / available as f64 + omitted_moment * (c1 / c2));
    }
    end_weights
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "__conditioning-scales")]
    use super::{henderson_end_weights, henderson_filter_operation_scales};
    use super::{henderson_filter, henderson_weights, moving_average, s3x5_subseries};
    use crate::x11::SevenTermEndpoints::{HendersonSevenTerm, LegacyFiveTerm};

    #[test]
    fn nine_term_right_endpoint_matches_build_62() {
        let input = [
            4.46837133920651,
            4.022805424498783,
            3.5715149682325955,
            3.118643018591178,
            2.6662801561500302,
            2.217785883481174,
            1.769802271387526,
            1.3274341837831947,
            0.9105597497947402,
        ];
        let (filtered, _) = henderson_filter(&input, 9, 1.0, LegacyFiveTerm);
        assert_eq!(filtered[6].to_bits(), 0x3ffc_56c7_0f54_7446);
    }

    #[test]
    fn henderson_endpoints_mirror_their_weights() {
        // One endpoint weight times a single nonzero input, at each end.
        let input = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0];
        let (output, _) = henderson_filter(&input, 9, 1.0, LegacyFiveTerm);
        assert_eq!(output[2].to_bits(), 0xc01b_2368_8305_3f52);
        let reversed: Vec<_> = input.into_iter().rev().collect();
        let (output, _) = henderson_filter(&reversed, 9, 1.0, LegacyFiveTerm);
        assert_eq!(output[6].to_bits(), 0xc01b_2368_8305_3f52);
    }

    #[cfg(feature = "__conditioning-scales")]
    #[test]
    fn henderson_operation_scale_uses_matching_coefficients() {
        let input = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0];
        let scales = henderson_filter_operation_scales(&input, 9, 1.0, LegacyFiveTerm);
        assert_eq!(scales[2].to_bits(), 0x401b_2368_8305_3f52);
        let reversed: Vec<_> = input.into_iter().rev().collect();
        let scales = henderson_filter_operation_scales(&reversed, 9, 1.0, LegacyFiveTerm);
        assert_eq!(scales[6].to_bits(), 0x401b_2368_8305_3f52);
    }

    #[test]
    fn henderson_omitted_moment_is_fused() {
        // `hndend.f` fuses each omitted weight times its centered index.
        // Without that fusion the first weight differs by one ULP
        // (0xbf97_2874_7d75_1e25).
        let central = henderson_weights(9);
        let r_beta = 4.0 / crate::numerics::BUILD_62_PI;
        let weights = super::henderson_end_weights(7, &central, r_beta);
        assert_eq!(weights[0].to_bits(), 0xbf97_2874_7d75_1e24);
    }

    #[test]
    fn henderson_denominator_is_fused() {
        // `hndend.f` fuses `R*(M*(M-1)*(M+1)/12)+1`. With one omitted weight
        // the moment fusion is exact, so only the denominator's fusion
        // matters here; without it the last weight is one ULP higher
        // (0x3faa_7c42_e4c9_0f6c).
        let central = [0.5, 0.25, 0.2, 0.07, -0.02];
        let r_beta = 1.5348175182481751;
        let weights = super::henderson_end_weights(4, &central, r_beta);
        assert_eq!(weights[3].to_bits(), 0x3faa_7c42_e4c9_0f6b);
    }

    #[test]
    fn two_by_four_moving_average_fills_only_complete_windows() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0];
        let mut output = [f64::NAN; 8];

        moving_average(&input, &mut output, 0..input.len(), 2, 4);

        assert!(output[..2].iter().all(|value| value.is_nan()));
        assert_eq!(&output[2..6], &[4.5, 5.125, 5.75, 5.75]);
        assert!(output[6..].iter().all(|value| value.is_nan()));
    }

    #[test]
    fn five_term_henderson_weights_match_build_62() {
        assert_eq!(
            henderson_weights(5),
            [
                -0.073_426_573_426_573_42,
                0.293_706_293_706_293_7,
                0.559_440_559_440_559_5,
                0.293_706_293_706_293_7,
                -0.073_426_573_426_573_42,
            ]
        );
    }

    #[cfg(feature = "__conditioning-scales")]
    #[test]
    fn henderson_operation_scale_sums_absolute_reference_terms() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0];

        let scales = henderson_filter_operation_scales(&input, 5, 0.001, LegacyFiveTerm);

        assert_eq!(scales[2], 1_462.0 / 286.0);
    }

    #[cfg(feature = "__conditioning-scales")]
    #[test]
    fn henderson_operation_scale_preserves_endpoint_weight_reversal() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0];
        let central = henderson_weights(5);
        let r_beta = 4.0 / (0.001_f64.powi(2) * crate::numerics::BUILD_62_PI);
        let weights = henderson_end_weights(3, &central, r_beta);

        let scales = henderson_filter_operation_scales(&input, 5, 0.001, LegacyFiveTerm);

        let beginning = (weights[2] * input[0]).abs()
            + (weights[1] * input[1]).abs()
            + (weights[0] * input[2]).abs();
        let end = (weights[2] * input[7]).abs()
            + (weights[1] * input[6]).abs()
            + (weights[0] * input[5]).abs();
        assert_eq!(scales[0], beginning.max(1.0));
        assert_eq!(scales[7], end.max(1.0));
    }

    #[test]
    fn s3x5_subseries_uses_build_62_endpoint_filters() {
        let filtered = s3x5_subseries(&[2.0, 5.0, 1.0, 9.0, 4.0, 8.0]);

        assert_eq!(
            filtered,
            [
                217.0 / 60.0,
                235.0 / 60.0,
                277.0 / 60.0,
                302.0 / 60.0,
                346.0 / 60.0,
                366.0 / 60.0,
            ]
        );
    }

    #[test]
    fn s3x5_endpoints_fuse_like_build_62() {
        // Each left-endpoint formula gives different bits without the fused
        // multiply-adds; reversing the input runs the same operations at the
        // right end.
        let input = [
            2.2803468208092488,
            5.916184971098266,
            3.7890173410404624,
            1.6618497109826589,
            5.297687861271676,
            3.1705202312138727,
            1.0433526011560694,
            4.679190751445087,
        ];
        let first_bits = [
            0x400d_2955_ba5a_2ad1,
            0x400d_3be1_9d8b_c9f0,
            0x400d_c7f7_5195_a76a,
        ];
        let last_bits = [
            0x400a_83cc_4e86_c1f2,
            0x400a_7140_6b55_22d2,
            0x4009_e52a_b74b_455a,
        ];
        for (values, first, last) in [
            (input.to_vec(), first_bits, last_bits),
            (input.into_iter().rev().collect(), last_bits, first_bits),
        ] {
            let actual = s3x5_subseries(&values);
            for index in 0..3 {
                assert_eq!(
                    actual[index].to_bits(),
                    first[index],
                    "left endpoint {index}"
                );
                assert_eq!(
                    actual[actual.len() - 1 - index].to_bits(),
                    last[index],
                    "right endpoint {index}"
                );
            }
        }
    }

    #[test]
    fn s3x5_short_subseries_use_plain_means() {
        assert_eq!(
            s3x5_subseries(&[2.0, 5.0, 1.0, 9.0]),
            [217.0 / 60.0, 17.0 / 4.0, 17.0 / 4.0, 273.0 / 60.0]
        );
        assert_eq!(
            s3x5_subseries(&[2.0, 5.0, 1.0, 9.0, 4.0]),
            [
                217.0 / 60.0,
                235.0 / 60.0,
                21.0 / 5.0,
                273.0 / 60.0,
                283.0 / 60.0
            ]
        );
    }

    #[test]
    fn five_term_henderson_filter_matches_build_62_endpoints() {
        let (filtered, _) = henderson_filter(
            &[2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0],
            5,
            0.001,
            LegacyFiveTerm,
        );

        assert_eq!(
            filtered,
            [
                f64::from_bits(0x400A_479B_B5E0_8F0A),
                f64::from_bits(0x4008_23CD_D932_715A),
                f64::from_bits(0x4010_EC4E_C4EC_4EC5),
                f64::from_bits(0x4016_3220_3949_660B),
                f64::from_bits(0x401B_BF8D_6D33_EA85),
                f64::from_bits(0x4015_6D33_EA84_79BC),
                f64::from_bits(0x4015_4072_9392_3BC0),
                f64::from_bits(0x4015_6440_7769_3B2C),
            ]
        );
    }

    #[test]
    fn henderson_central_values_match_build_62_bits() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0];
        let (five_term, _) = henderson_filter(&input, 5, 0.001, LegacyFiveTerm);
        let (seven_term, _) = henderson_filter(&input, 7, 0.001, LegacyFiveTerm);

        // Bits from Build 62's `hndtrn.f` central loop.
        assert_eq!(
            five_term[2..6],
            [
                f64::from_bits(0x4010_EC4E_C4EC_4EC5),
                f64::from_bits(0x4016_3220_3949_660B),
                f64::from_bits(0x401B_BF8D_6D33_EA85),
                f64::from_bits(0x4015_6D33_EA84_79BC),
            ]
        );
        assert_eq!(
            seven_term[3..5],
            [
                f64::from_bits(0x4016_9B63_E4C9_EF88),
                f64::from_bits(0x4018_B196_BC54_7F77),
            ]
        );
    }

    #[test]
    fn henderson_endpoint_values_match_build_62_bits() {
        let input = [2.1, 5.3, 1.7, 9.2, 4.4, 8.6, 3.8, 7.9];
        let (filtered, _) = henderson_filter(&input, 5, 0.001, LegacyFiveTerm);

        // Bits from Build 62's `hndtrn.f` endpoint loop.
        assert_eq!(
            filtered[0..2],
            [
                f64::from_bits(0x400A_C938_328D_61AD),
                f64::from_bits(0x400B_2819_C094_4BCA),
            ]
        );
        assert_eq!(
            filtered[6..8],
            [
                f64::from_bits(0x4018_5D17_46EB_D14B),
                f64::from_bits(0x4019_10A6_250A_D34F),
            ]
        );
    }

    #[test]
    fn legacy_seven_term_filter_resets_the_endpoint_parameter() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0];
        let (_, trend_ic) = henderson_filter(&input, 7, 4.5, LegacyFiveTerm);

        assert_eq!(trend_ic, 0.001);

        #[cfg(feature = "__conditioning-scales")]
        {
            let scales = henderson_filter_operation_scales(&input, 7, trend_ic, LegacyFiveTerm);
            let five_term = henderson_weights(5);
            let seven_term = henderson_weights(7);
            let overridden = (five_term[2] * input[2]).abs()
                + (five_term[3] * (input[1] + input[3])).abs()
                + (five_term[4] * (input[0] + input[4])).abs();
            let central = (seven_term[3] * input[3]).abs()
                + (seven_term[4] * (input[2] + input[4])).abs()
                + (seven_term[5] * (input[1] + input[5])).abs()
                + (seven_term[6] * (input[0] + input[6])).abs();
            assert_eq!(scales[2], overridden.max(1.0));
            assert_eq!(scales[3], central.max(1.0));
        }
    }

    #[test]
    fn seven_term_endpoints_keep_the_ratio_and_change_values() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0];
        let (legacy, legacy_ic) = henderson_filter(&input, 7, 0.75, LegacyFiveTerm);
        let (seven_term, seven_term_ic) = henderson_filter(&input, 7, 0.75, HendersonSevenTerm);
        assert_eq!(seven_term_ic, 0.75);
        assert_eq!(legacy_ic, 0.001);
        assert_ne!(seven_term[2], legacy[2]);
        assert_eq!(&seven_term[3..5], &legacy[3..5]);
        let (alternate, _) = henderson_filter(&input, 7, 4.5, HendersonSevenTerm);
        assert_ne!(seven_term[0], alternate[0]);
    }

    #[test]
    fn other_lengths_ignore_the_seven_term_policy() {
        let input = [2.0, 5.0, 1.0, 9.0, 4.0, 8.0, 3.0, 7.0, 12.0];
        for terms in [3, 5, 9] {
            let (left, left_ic) = henderson_filter(&input, terms, 0.75, LegacyFiveTerm);
            let (right, right_ic) = henderson_filter(&input, terms, 0.75, HendersonSevenTerm);
            assert_eq!(left, right);
            assert_eq!((left_ic, right_ic), (0.75, 0.75));
            #[cfg(feature = "__conditioning-scales")]
            assert_eq!(
                henderson_filter_operation_scales(&input, terms, left_ic, LegacyFiveTerm),
                henderson_filter_operation_scales(&input, terms, right_ic, HendersonSevenTerm)
            );
        }
    }

    #[cfg(feature = "__conditioning-scales")]
    #[test]
    fn genuine_seven_term_operation_scales_use_seven_term_endpoint_coefficients() {
        let input = [2.0, -5.0, 1.0, -9.0, 4.0, 8.0, -3.0, 7.0];
        let central = henderson_weights(7);
        let r_beta = 4.0 / (0.75 * 0.75 * crate::numerics::BUILD_62_PI);
        let weights = henderson_end_weights(6, &central, r_beta);
        let expected = weights
            .iter()
            .rev()
            .zip(input)
            .map(|(weight, value)| (weight * value).abs())
            .sum::<f64>()
            .max(1.0);
        let scales = henderson_filter_operation_scales(&input, 7, 0.75, HendersonSevenTerm);
        assert_eq!(scales[2], expected);
        assert_ne!(
            scales[0],
            henderson_filter_operation_scales(&input, 7, 4.5, HendersonSevenTerm)[0]
        );
    }
}
