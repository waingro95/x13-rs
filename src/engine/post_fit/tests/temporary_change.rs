//! Allocation of the temporary-change effect to the trend or the irregular.

use super::{components, final_options, outlier_effects, restore};
use crate::regarima::OutlierTypes;
use crate::{Transform, x11};

#[test]
fn additive_allocation_moves_the_temporary_change_between_trend_and_irregular() {
    let effects = outlier_effects(Transform::None, [2.0, 3.0, 4.0]);
    let cases = [
        (x11::FinalOutliers::Retain, 90.0, 77.0, 73.0),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE),
            88.0,
            77.0,
            73.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT),
            87.0,
            74.0,
            70.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::TEMPORARY_CHANGE),
            86.0,
            73.0,
            73.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT),
            85.0,
            74.0,
            70.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_TEMPORARY_CHANGE),
            84.0,
            73.0,
            73.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT_AND_TEMPORARY_CHANGE),
            83.0,
            70.0,
            70.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ALL),
            81.0,
            70.0,
            70.0,
        ),
    ];
    for (outliers, expected_d11, trend_d12, irregular_d12) in cases {
        for (allocation, expected_d12, expected_d13) in [
            (
                x11::TemporaryChangeAllocation::Irregular,
                irregular_d12,
                17.0,
            ),
            (x11::TemporaryChangeAllocation::Trend, trend_d12, 13.0),
        ] {
            let mut result = components(x11::Mode::Additive, 10.0, 70.0);
            restore(
                &mut result,
                100.0,
                &effects,
                Transform::None,
                final_options(outliers, allocation),
            );
            assert_eq!(
                result.seasonally_adjusted,
                [expected_d11],
                "{outliers:?}/{allocation:?} D11"
            );
            assert_eq!(
                result.trend_cycle,
                [expected_d12],
                "{outliers:?}/{allocation:?} D12"
            );
            assert_eq!(
                result.irregular,
                [expected_d13],
                "{outliers:?}/{allocation:?} D13"
            );
        }
    }
}

/// `(1e16 - 1e16) + 1` keeps the temporary change; adding it to the trend
/// before the level shift would round it away.
#[test]
fn additive_trend_adds_ls_before_tc() {
    let effects = outlier_effects(Transform::None, [0.0, -1.0e16, 1.0]);
    let mut result = components(x11::Mode::Additive, 0.0, 1.0e16);
    restore(
        &mut result,
        1.0e16,
        &effects,
        Transform::None,
        final_options(
            x11::FinalOutliers::Retain,
            x11::TemporaryChangeAllocation::Trend,
        ),
    );
    assert_eq!(result.trend_cycle, [1.0]);
}

/// As for the trend: the irregular gets the additive outlier back before the
/// temporary change.
#[test]
fn additive_irregular_adds_ao_before_tc() {
    let effects = outlier_effects(Transform::None, [-1.0e16, 0.0, 1.0]);
    let mut result = components(x11::Mode::Additive, 0.0, 0.0);
    restore(
        &mut result,
        0.0,
        &effects,
        Transform::None,
        final_options(
            x11::FinalOutliers::Retain,
            x11::TemporaryChangeAllocation::Irregular,
        ),
    );
    assert_eq!(result.irregular, [1.0]);
}

#[test]
fn log_allocation_multiplies_the_temporary_change_into_trend_or_irregular() {
    let effects = outlier_effects(Transform::Log, [0.1, 0.2, -0.05]);
    let pre_final = 200.0 / 2.0;
    let ao = 0.1_f64.exp();
    let ls = 0.2_f64.exp();
    let tc = (-0.05_f64).exp();
    let raw_irregular = (((pre_final / 50.0) / ls) / ao) / tc;
    for (outliers, expected_d11, trend_d12, irregular_d12) in [
        (
            x11::FinalOutliers::Retain,
            pre_final,
            tc * (ls * 50.0),
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE),
            pre_final / ao,
            tc * (ls * 50.0),
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT),
            pre_final / ls,
            tc * 50.0,
            50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::TEMPORARY_CHANGE),
            pre_final / tc,
            ls * 50.0,
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT),
            (pre_final / ao) / ls,
            tc * 50.0,
            50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_TEMPORARY_CHANGE),
            (pre_final / ao) / tc,
            ls * 50.0,
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT_AND_TEMPORARY_CHANGE),
            (pre_final / ls) / tc,
            50.0,
            50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ALL),
            ((pre_final / ao) / ls) / tc,
            50.0,
            50.0,
        ),
    ] {
        for (allocation, expected_d12, expected_d13) in [
            (
                x11::TemporaryChangeAllocation::Irregular,
                irregular_d12,
                tc * (ao * raw_irregular),
            ),
            (
                x11::TemporaryChangeAllocation::Trend,
                trend_d12,
                ao * raw_irregular,
            ),
        ] {
            let mut result = components(x11::Mode::Multiplicative, 2.0, 50.0);
            restore(
                &mut result,
                200.0,
                &effects,
                Transform::Log,
                final_options(outliers, allocation),
            );
            assert_eq!(
                result.seasonally_adjusted[0].to_bits(),
                expected_d11.to_bits(),
                "{outliers:?}/{allocation:?} D11"
            );
            assert_eq!(
                result.trend_cycle[0].to_bits(),
                expected_d12.to_bits(),
                "{outliers:?}/{allocation:?} D12"
            );
            assert_eq!(
                result.irregular[0].to_bits(),
                expected_d13.to_bits(),
                "{outliers:?}/{allocation:?} D13"
            );
        }
    }
}
