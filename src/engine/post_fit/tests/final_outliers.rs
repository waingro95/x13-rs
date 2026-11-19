//! Removal of outlier effects from the final seasonally adjusted series.

use super::{components, final_options, outlier_effects, restore};
use crate::regarima::OutlierTypes;
use crate::{Transform, x11};

/// The quality statistics read the irregular before any outlier is restored
/// to it.
#[test]
fn quality_statistics_use_the_irregular_without_outliers() {
    let effects = outlier_effects(Transform::None, [2.0, 3.0, 4.0]);
    for outliers in [
        x11::FinalOutliers::Retain,
        x11::FinalOutliers::Remove(OutlierTypes::ALL),
    ] {
        for allocation in [
            x11::TemporaryChangeAllocation::Irregular,
            x11::TemporaryChangeAllocation::Trend,
        ] {
            let mut result = components(x11::Mode::Additive, 10.0, 70.0);
            restore(
                &mut result,
                100.0,
                &effects,
                Transform::None,
                final_options(outliers, allocation),
            );
            let without_outliers = &result.quality_workspace.base_irregular;
            assert_eq!(without_outliers, &[11.0], "{outliers:?}");
            assert_ne!(&result.irregular, without_outliers, "{outliers:?}");
        }
    }
}

#[test]
fn additive_removal_takes_each_selected_type_out_of_the_adjusted_series() {
    let effects = outlier_effects(Transform::None, [3.0, 2.0, 4.0]);
    for (outliers, adjusted, trend) in [
        (x11::FinalOutliers::Retain, 90.0, 72.0),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE),
            87.0,
            72.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT),
            88.0,
            70.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::TEMPORARY_CHANGE),
            86.0,
            72.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT),
            85.0,
            70.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_TEMPORARY_CHANGE),
            83.0,
            72.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT_AND_TEMPORARY_CHANGE),
            84.0,
            70.0,
        ),
        (x11::FinalOutliers::Remove(OutlierTypes::ALL), 81.0, 70.0),
    ] {
        let mut result = components(x11::Mode::Additive, 10.0, 70.0);
        restore(
            &mut result,
            100.0,
            &effects,
            Transform::None,
            final_options(outliers, x11::TemporaryChangeAllocation::Irregular),
        );
        assert_eq!(result.seasonally_adjusted, [adjusted], "{outliers:?}");
        assert_eq!(result.trend_cycle, [trend], "{outliers:?}");
        assert_eq!(result.irregular, [18.0], "{outliers:?}");
    }
}

/// These effects cancel differently in every other order of subtraction.
#[test]
fn additive_removal_subtracts_ao_then_ls_then_tc() {
    let (original, ao, ls, tc) = (1.0e16_f64, 1.0e16 + 2.0, -1.0, -1.0e-5);
    let effects = outlier_effects(Transform::None, [ao, ls, tc]);
    let mut result = components(x11::Mode::Additive, 0.0, 70.0);
    restore(
        &mut result,
        original,
        &effects,
        Transform::None,
        final_options(
            x11::FinalOutliers::Remove(OutlierTypes::ALL),
            x11::TemporaryChangeAllocation::Irregular,
        ),
    );
    let adjusted = result.seasonally_adjusted[0];
    assert_eq!(adjusted.to_bits(), (((original - ao) - ls) - tc).to_bits());
    for other in [
        ((original - ls) - ao) - tc,
        ((original - ao) - tc) - ls,
        ((original - tc) - ao) - ls,
        original - (ao + ls + tc),
    ] {
        assert_ne!(adjusted.to_bits(), other.to_bits());
    }
}

#[test]
fn log_removal_divides_by_ao_then_ls_then_tc() {
    let effects = outlier_effects(Transform::Log, [0.1, 0.2, -0.05]);
    let ao = 0.1_f64.exp();
    let ls = 0.2_f64.exp();
    let tc = (-0.05_f64).exp();
    let irregular = (((((100.0 / 50.0) / ls) / ao) / tc) * ao) * tc;
    for (outliers, adjusted, trend) in [
        (x11::FinalOutliers::Retain, 100.0, ls * 50.0),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE),
            100.0 / ao,
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT),
            100.0 / ls,
            50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::TEMPORARY_CHANGE),
            100.0 / tc,
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT),
            (100.0 / ao) / ls,
            50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ADDITIVE_AND_TEMPORARY_CHANGE),
            (100.0 / ao) / tc,
            ls * 50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::LEVEL_SHIFT_AND_TEMPORARY_CHANGE),
            (100.0 / ls) / tc,
            50.0,
        ),
        (
            x11::FinalOutliers::Remove(OutlierTypes::ALL),
            ((100.0 / ao) / ls) / tc,
            50.0,
        ),
    ] {
        let mut result = components(x11::Mode::Multiplicative, 2.0, 50.0);
        restore(
            &mut result,
            200.0,
            &effects,
            Transform::Log,
            final_options(outliers, x11::TemporaryChangeAllocation::Irregular),
        );
        assert_eq!(
            result.seasonally_adjusted[0].to_bits(),
            adjusted.to_bits(),
            "{outliers:?}"
        );
        assert_eq!(
            result.trend_cycle[0].to_bits(),
            trend.to_bits(),
            "{outliers:?}"
        );
        assert_eq!(
            result.irregular[0].to_bits(),
            irregular.to_bits(),
            "{outliers:?}"
        );
    }
}
