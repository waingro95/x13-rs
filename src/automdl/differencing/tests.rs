use super::*;
use crate::arima::exact::{DifferenceError, difference_series};
use crate::arima::model::ExpandedModel;
use crate::error::NumericalOperation;
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, Frequency};

fn model(d: u8, seasonal_d: u8) -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(0, d, 0).unwrap(),
        ArimaOrder::new(0, seasonal_d, 0).unwrap(),
    )
    .unwrap()
}

#[test]
fn the_second_difference_is_one_fused_operator_applied_before_the_seasonal_one() {
    let expanded = ExpandedModel::from_spec(model(2, 1), Frequency::Quarterly);
    let original = [-1e12, 1e8, -100.0, -1e16, 100.0, 1.0, -1.0, 1e16];
    // The fused second-difference operator, then the lag-4 seasonal one.
    let expected = [1_000_200_000_197.0, 19_999_999_899_999_804.0];
    assert_eq!(difference_series(&original, &expanded).unwrap(), expected);
    let seasonal_first = (0..4)
        .map(|i| original[i + 4] - original[i])
        .collect::<Vec<_>>();
    let reversed = (0..2)
        .map(|i| {
            let first = (-2.0_f64).mul_add(seasonal_first[i + 1], seasonal_first[i + 2]);
            first + seasonal_first[i]
        })
        .collect::<Vec<_>>();
    let once = original
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let twice = once
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let separated = (0..2).map(|i| twice[i + 4] - twice[i]).collect::<Vec<_>>();
    assert_ne!(
        reversed, expected,
        "seasonal-first order must be detectable"
    );
    assert_ne!(
        separated, expected,
        "two rounded first passes must be detectable"
    );
}

#[test]
fn differencing_rejects_short_and_nonfinite_input() {
    let expanded = ExpandedModel::from_spec(model(2, 1), Frequency::Quarterly);
    assert_eq!(
        difference_series(&[], &expanded),
        Err(DifferenceError::InsufficientObservations)
    );
    assert_eq!(
        difference_series(&[1.0; 6], &expanded),
        Err(DifferenceError::InsufficientObservations)
    );
    assert_eq!(
        difference_series(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0], &expanded),
        Err(DifferenceError::NonFiniteInput)
    );
}

/// A later round's result in `iddiff.f`'s names.
struct LaterFlags {
    regular: u8,
    seasonal: u8,
    icon: u8,
    irm1: bool,
    first_maxima: [f64; 2],
    cancellation_limit: f64,
}

/// Runs [`later_round`] on coefficients in the order regular AR, seasonal
/// AR, regular MA, seasonal MA, and on `rmaxr`/`rmaxs`.
fn later_flags(
    round: u8,
    regular: u8,
    seasonal: u8,
    coefficients: [f64; 4],
    first_maxima: [f64; 2],
    cancellation_limit: f64,
    post_covariance_error: bool,
) -> LaterFlags {
    let [regular_ar, seasonal_ar, regular_ma, seasonal_ma] = coefficients;
    let outcome = later_round(
        round,
        ByFactor::new(regular, seasonal),
        LeadingCoefficients {
            regular_ar,
            seasonal_ar,
            regular_ma,
            seasonal_ma,
        },
        ByFactor::new(first_maxima[0], first_maxima[1]),
        cancellation_limit,
        post_covariance_error,
    );
    let (icon, irm1) = match outcome.signal {
        RoundSignal::Settled => (0, false),
        RoundSignal::Differenced => (1, false),
        RoundSignal::Explosive => (1, true),
        RoundSignal::BothDifferenced => (2, false),
    };
    LaterFlags {
        regular: outcome.orders.regular,
        seasonal: outcome.orders.seasonal,
        icon,
        irm1,
        first_maxima: [outcome.first_maxima.regular, outcome.first_maxima.seasonal],
        cancellation_limit: outcome.cancellation_limit,
    }
}

#[test]
fn a_later_round_differences_a_unit_coefficient_unless_the_ma_cancels_it() {
    let regular = later_flags(3, 0, 0, [0.95, 0.0, 0.0, 0.0], [-999.0; 2], 0.1, false);
    assert_eq!((regular.regular, regular.seasonal, regular.icon), (1, 0, 1));
    let seasonal = later_flags(3, 0, 0, [0.0, 0.95, 0.0, 0.0], [-999.0; 2], 0.1, false);
    assert_eq!(
        (seasonal.regular, seasonal.seasonal, seasonal.icon),
        (0, 1, 1)
    );
    let canceled = later_flags(3, 1, 0, [0.95, 0.95, 0.95, 0.95], [-999.0; 2], 0.1, false);
    assert_eq!(
        (canceled.regular, canceled.seasonal, canceled.icon),
        (1, 0, 0)
    );
    assert_eq!(canceled.cancellation_limit, 0.098);
}

#[test]
fn the_second_round_uses_the_first_round_maxima() {
    let guarded = later_flags(2, 0, 0, [0.7, 0.7, 0.0, 0.0], [0.91, 0.89], 0.1, false);
    assert_eq!((guarded.regular, guarded.seasonal), (1, 0));
    let unset = later_flags(2, 1, 0, [0.95, 0.95, 0.0, 0.0], [-999.0; 2], 0.1, false);
    assert_eq!((unset.regular, unset.seasonal, unset.icon), (1, 1, 2));
    let infinite = later_flags(
        2,
        1,
        0,
        [0.95, 0.95, 0.0, 0.0],
        [f64::NEG_INFINITY; 2],
        0.1,
        false,
    );
    assert_eq!((infinite.regular, infinite.seasonal), (1, 0));
    let diagnostic = later_flags(2, 1, 0, [0.96, 0.94, 0.0, 0.0], [-999.0; 2], 0.1, true);
    assert_eq!((diagnostic.regular, diagnostic.seasonal), (2, 0));
}

#[test]
fn mean_thresholds_depend_on_the_series_length_and_include_equality() {
    for threshold in [1.96_f64, 2.0, 2.55] {
        assert!(!mean_is_significant(threshold.next_down(), threshold));
        assert!(mean_is_significant(threshold, threshold));
        assert!(mean_is_significant(threshold.next_up(), threshold));
    }
    for (count, threshold) in [
        (80, 1.96),
        (81, 1.98),
        (155, 1.98),
        (156, 2.1),
        (230, 2.1),
        (231, 2.3),
        (350, 2.3),
        (351, 2.5),
    ] {
        assert_eq!(fitted_mean_threshold(count), threshold);
        assert!(mean_is_significant(threshold, threshold));
        assert!(!mean_is_significant(threshold.next_down(), threshold));
        assert!(mean_is_significant(threshold.next_up(), threshold));
    }
    // The no-difference statistic is sum/sqrt(sum of squares): k ones give
    // sqrt(k), however many zeros follow.
    for (count, below, at_or_above) in [
        (79, 3, 4),
        (80, 3, 4),
        (81, 3, 4),
        (199, 3, 4),
        (200, 3, 4),
        (201, 6, 7),
        (202, 6, 7),
    ] {
        for (ones, expected) in [(below, false), (at_or_above, true)] {
            let mut values = vec![0.0; count];
            values[..ones].fill(1.0);
            assert_eq!(
                no_difference_mean(&values),
                expected,
                "count {count}, ones {ones}"
            );
        }
    }
}

#[test]
fn the_no_difference_mean_skips_values_of_minus_999() {
    let mut values = vec![0.0; 80];
    values[..3].fill(1.0);
    assert!(!no_difference_mean(&values));
    // The skipped values still count toward the series length.
    values[3..23].fill(-999.0);
    assert!(no_difference_mean(&values));
}

#[test]
fn a_tiny_variance_or_zero_standard_error_gives_a_zero_mean_statistic() {
    let limit = 2.0 * BUILD_62_MACHINE_EPSILON;
    assert_eq!(fitted_mean_statistic(3.0, &[1.0], limit.next_down()), 0.0);
    assert_eq!(fitted_mean_statistic(3.0, &[1.0], limit), 0.0);
    assert!(fitted_mean_statistic(3.0, &[1.0], limit.next_up()) > 0.0);
    assert_eq!(fitted_mean_statistic(3.0, &[f64::INFINITY], 1.0), 0.0);
}

#[test]
fn a_singular_hannan_rissanen_regression_starts_an_exact_fit() {
    let initial = round_start(
        Err(HrFailure::Initial(HrFailureReason::SingularRegression {
            pivot: 1,
        })),
        4,
    )
    .unwrap();
    assert_eq!(initial.parameters, [0.1; 4]);
    assert!(initial.needs_fit);
    let first_stage = HrEstimate {
        parameters: vec![0.2, 0.3, 0.4, 0.5],
        effective_observations: 20,
    };
    let ma_refit = round_start(
        Err(HrFailure::MaRefit {
            reason: HrFailureReason::SingularRegression { pivot: 2 },
            first_stage,
        }),
        4,
    )
    .unwrap();
    assert_eq!(ma_refit.parameters, [0.2, 0.3, 0.4, 0.5]);
    assert!(ma_refit.needs_fit);
}

#[test]
fn empty_short_and_nonfinite_series_are_estimation_failures() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let estimation = Error::numerical(NumericalOperation::Estimation);
    for values in [&[][..], &[1.0][..], &[1.0, 2.0, 3.0][..]] {
        assert_eq!(
            DifferencingState::default().identify_prepared(values, axis, &mut Vec::new()),
            Err(estimation.clone())
        );
    }
    let nonfinite = [1.0, 2.0, f64::NAN, 4.0];
    let working = ExpandedModel::from_spec(
        working_model(1, ByFactor::new(0, 0)).unwrap(),
        axis.frequency(),
    );
    assert!(matches!(
        centered_differences(&nonfinite, &working),
        Err(DifferenceError::NonFiniteInput)
    ));
    assert_eq!(
        DifferencingState::default().identify_prepared(&nonfinite, axis, &mut Vec::new()),
        Err(estimation)
    );
}

#[test]
fn an_all_zero_series_is_an_estimation_failure() {
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    // The autocorrelations of the working model's Hannan-Rissanen estimate
    // have a zero variance to divide by.
    let orders = HrOrders::new(1, 1, 1, 1);
    assert!(matches!(
        estimate_prepared(&[0.0; 144], axis.frequency(), orders),
        Err(HrFailure::Initial(HrFailureReason::Autocorrelation(
            crate::numerics::autocorrelation::AutocorrelationError::ZeroVariance
        )))
    ));
    assert_eq!(
        DifferencingState::default().identify_prepared(&[0.0; 144], axis, &mut Vec::new()),
        Err(Error::numerical(NumericalOperation::Estimation))
    );
}

#[test]
fn a_second_round_always_runs_and_the_seventh_is_the_last() {
    assert!(continue_after_round(2, RoundSignal::Settled));
    assert!(continue_after_round(7, RoundSignal::Differenced));
    assert!(!continue_after_round(8, RoundSignal::Differenced));
    assert!(!continue_after_round(3, RoundSignal::Settled));
    for signal in [
        RoundSignal::Differenced,
        RoundSignal::Explosive,
        RoundSignal::BothDifferenced,
    ] {
        assert!(continue_after_round(7, signal));
        assert!(!continue_after_round(8, signal));
    }
}

/// A later round's result as `(regular, seasonal, icon, irm1)` in
/// `iddiff.f`'s names, from a 0.1 cancellation limit.
fn later(
    round: u8,
    orders: (u8, u8),
    coefficients: [f64; 4],
    first_maxima: [f64; 2],
    post_covariance_error: bool,
) -> (u8, u8, u8, bool) {
    let decision = later_flags(
        round,
        orders.0,
        orders.1,
        coefficients,
        first_maxima,
        0.1,
        post_covariance_error,
    );
    (
        decision.regular,
        decision.seasonal,
        decision.icon,
        decision.irm1,
    )
}

/// The `rmaxr`/`rmaxs` pair a later round leaves, as bits.
fn later_maxima(round: u8, coefficients: [f64; 4], first_maxima: [f64; 2]) -> [u64; 2] {
    later_flags(round, 0, 0, coefficients, first_maxima, 0.1, false)
        .first_maxima
        .map(f64::to_bits)
}

const NO_MAXIMA: [f64; 2] = [-999.0; 2];

#[test]
fn regular_unit_band_and_explosive_thresholds() {
    let regular = |ar: f64, ma: f64, post_covariance_error| {
        later(
            3,
            (0, 1),
            [ar, 0.0, ma, 0.0],
            NO_MAXIMA,
            post_covariance_error,
        )
    };
    assert_eq!(regular(1.02, 0.0, false), (1, 1, 1, false));
    assert_eq!(regular(1.02_f64.next_up(), 0.0, false), (0, 1, 1, true));
    assert_eq!(regular(-1.12, 0.0, false), (0, 1, 0, false));
    assert_eq!(regular(-(1.12_f64.next_up()), 0.0, false), (0, 1, 1, true));
    // `din` is `1.005 - 0.88`, widened by `0.88 - 0.869` without a covariance
    // error: these are the last coefficients inside each band.
    let wide_inside = f64::from_bits(0x3feb_a5e3_53f7_ceda);
    let wide_outside = f64::from_bits(0x3feb_a5e3_53f7_ced9);
    let narrow_inside = f64::from_bits(0x3fec_0000_0000_0001);
    assert_eq!(regular(wide_inside, 0.0, false), (1, 1, 1, false));
    assert_eq!(regular(wide_outside, 0.0, false), (0, 1, 0, false));
    assert_eq!(regular(wide_inside, 0.0, true), (0, 1, 0, false));
    assert_eq!(regular(narrow_inside, 0.0, true), (1, 1, 1, false));
    assert_eq!(regular(0.875, 0.0, true), (0, 1, 0, false));
    assert_eq!(regular(0.95, 0.9, false), (0, 1, 0, false));
    assert_eq!(
        later_flags(3, 0, 1, [0.0; 4], NO_MAXIMA, 0.1, false)
            .cancellation_limit
            .to_bits(),
        0x3fb9_1687_2b02_0c4a
    );
}

#[test]
fn seasonal_unit_band_and_explosive_thresholds() {
    let seasonal = |ar: f64, orders| later(3, orders, [0.0, ar, 0.0, 0.0], NO_MAXIMA, false);
    assert_eq!(seasonal(1.02, (1, 0)), (1, 1, 1, false));
    assert_eq!(seasonal(1.02_f64.next_up(), (1, 0)), (1, 0, 1, true));
    assert_eq!(seasonal(-1.12, (1, 0)), (1, 0, 0, false));
    assert_eq!(seasonal(-(1.12_f64.next_up()), (1, 0)), (1, 0, 1, true));
    assert_eq!(seasonal(0.81, (1, 0)), (1, 1, 1, false));
    assert_eq!(
        seasonal(f64::from_bits(0x3fe9_eb85_1eb8_51eb), (1, 0)),
        (1, 0, 0, false)
    );
    assert_eq!(seasonal(0.95, (1, 1)), (1, 1, 0, false));
}

#[test]
fn flag_interplay_between_regular_and_seasonal_branches() {
    let round = |coefficients| later(3, (0, 0), coefficients, NO_MAXIMA, false);
    // A seasonal unit root clears the regular explosive or far flag...
    assert_eq!(round([1.03, 0.95, 0.0, 0.0]), (0, 1, 1, false));
    assert_eq!(round([1.2, 0.95, 0.0, 0.0]), (0, 1, 1, false));
    // ...but a seasonal explosive or far flag keeps the regular difference.
    assert_eq!(round([0.95, 1.05, 0.0, 0.0]), (1, 0, 1, true));
    assert_eq!(round([0.95, 1.5, 0.0, 0.0]), (1, 0, 1, true));
    // Two explosive flags leave both orders zero for the late check.
    assert_eq!(round([1.03, 1.05, 0.0, 0.0]), (0, 1, 1, true));
    assert_eq!(
        later_maxima(3, [1.03, 1.05, 0.0, 0.0], NO_MAXIMA),
        [1.03_f64.to_bits(), 1.05_f64.to_bits()]
    );
}

#[test]
fn both_unit_roots_keep_one_difference() {
    let both = |orders, coefficients, first_maxima, post_covariance_error| {
        later(3, orders, coefficients, first_maxima, post_covariance_error)
    };
    let units = [0.95, 0.95, 0.0, 0.0];
    assert_eq!(both((0, 0), units, [0.5, 0.7], false), (0, 1, 2, false));
    assert_eq!(both((0, 0), units, [0.7, 0.5], false), (1, 0, 2, false));
    assert_eq!(both((0, 0), units, [0.7, 0.7], false), (0, 1, 2, false));
    assert_eq!(
        both((0, 0), units, [-10_000.0, -20_000.0], false),
        (0, 0, 2, false)
    );
    assert_eq!(
        both((0, 0), units, [-20_000.0, -10_000.0], false),
        (0, 0, 2, false)
    );
    assert_eq!(both((1, 0), units, [0.5, 0.7], false), (1, 1, 2, false));
    // A covariance error compares the coefficients instead of the maxima.
    assert_eq!(
        both((0, 0), [0.95, 0.9, 0.0, 0.0], [-20_000.0; 2], true),
        (1, 0, 2, false)
    );
    assert_eq!(
        both((0, 0), [0.9, 0.95, 0.0, 0.0], NO_MAXIMA, true),
        (0, 1, 2, false)
    );
    assert_eq!(both((0, 0), units, NO_MAXIMA, true), (0, 1, 2, false));
}

#[test]
fn second_round_cancellation_block() {
    let second = |coefficients, first_maxima| later(2, (0, 0), coefficients, first_maxima, false);
    let regular_cancels = [0.5, 0.5, 0.45, 0.0];
    assert_eq!(second(regular_cancels, [0.5, 0.95]), (0, 1, 1, false));
    assert_eq!(second(regular_cancels, [0.95, 0.5]), (1, 0, 1, false));
    assert_eq!(second(regular_cancels, [0.95, 0.95]), (0, 1, 1, false));
    assert_eq!(second(regular_cancels, [0.9, 0.5]), (1, 0, 1, false));
    assert_eq!(
        second(regular_cancels, [0.9_f64.next_down(), 0.5]),
        (0, 0, 0, false)
    );
    assert_eq!(second(regular_cancels, [0.85, 0.85]), (0, 0, 0, false));
    assert_eq!(second([0.5, 0.5, 0.0, 0.45], [0.5, 0.95]), (0, 1, 1, false));
    // A far regular coefficient's flag is replaced by the difference.
    assert_eq!(second([1.5, 0.5, 1.45, 0.0], [0.95, 0.5]), (1, 0, 1, false));
    assert_eq!(
        later(3, (0, 0), regular_cancels, [0.5, 0.95], false),
        (0, 0, 0, false)
    );
}

#[test]
fn second_round_near_unit_block_and_the_lchks2_quirk() {
    let second = |coefficients, first_maxima| later(2, (0, 0), coefficients, first_maxima, false);
    // `lchks2` stays true whenever `lchks` is false, so the second disjunct
    // needs only the maxima.
    let distant = [0.5, 0.5, 0.0, 0.0];
    assert_eq!(second(distant, [0.5, 0.89]), (0, 1, 1, false));
    assert_eq!(
        second(distant, [0.5, 0.89_f64.next_down()]),
        (0, 0, 0, false)
    );
    assert_eq!(second(distant, [0.91, 0.5]), (1, 0, 1, false));
    assert_eq!(
        second(distant, [0.91_f64.next_down(), 0.5]),
        (0, 0, 0, false)
    );
    // `lchks` lets the first disjunct use the lower 0.88 seasonal maximum.
    let lchks_inside = f64::from_bits(0x3fea_e147_ae14_7ae2);
    let lchks_outside = f64::from_bits(0x3fea_e147_ae14_7ae1);
    assert_eq!(
        second([0.5, lchks_inside, 0.0, 0.8], [0.5, 0.88]),
        (0, 1, 1, false)
    );
    assert_eq!(
        second([0.5, lchks_outside, 0.0, 0.8], [0.5, 0.88]),
        (0, 1, 0, false)
    );
    assert_eq!(
        later_maxima(2, [0.5, lchks_outside, 0.0, 0.8], [0.5, 0.88]),
        [0.5_f64.to_bits(), lchks_outside.to_bits()]
    );
    // The regular 0.15 band with the 0.9 regular maximum.
    let regular_edge = f64::from_bits(0x3feb_3333_3333_3334);
    assert_eq!(
        second([regular_edge, 0.5, 0.0, 0.0], [0.9, 0.5]),
        (1, 0, 1, false)
    );
    assert_eq!(
        second([regular_edge, 0.5, 0.0, 0.0], [0.9_f64.next_down(), 0.5]),
        (1, 0, 0, false)
    );
    // A far seasonal coefficient's flag is replaced by the difference.
    assert_eq!(second([0.5, 1.5, 0.0, 0.0], [0.5, 0.89]), (0, 1, 1, false));
}

#[test]
fn late_near_unit_block_replaces_the_maxima() {
    let regular_edge = f64::from_bits(0x3feb_3333_3333_3334);
    assert_eq!(
        later(3, (0, 0), [regular_edge, 0.3, 0.0, 0.0], NO_MAXIMA, false),
        (1, 0, 0, false)
    );
    assert_eq!(
        later_maxima(3, [regular_edge, 0.3, 0.0, 0.0], NO_MAXIMA),
        [regular_edge.to_bits(), 0.3_f64.to_bits()]
    );
    assert_eq!(
        later(3, (0, 0), [0.85, 0.3, 0.0, 0.0], NO_MAXIMA, false),
        (0, 0, 0, false)
    );
    assert_eq!(
        later_maxima(3, [0.85, 0.3, 0.0, 0.0], NO_MAXIMA),
        NO_MAXIMA.map(f64::to_bits)
    );
    assert_eq!(
        later(3, (0, 0), [0.3, 0.75, 0.0, 0.0], NO_MAXIMA, false),
        (0, 1, 0, false)
    );
    assert_eq!(
        later(
            3,
            (0, 0),
            [0.3, 0.75_f64.next_down(), 0.0, 0.0],
            NO_MAXIMA,
            false
        ),
        (0, 0, 0, false)
    );
    assert_eq!(
        later(3, (0, 0), [0.86, 0.86, 0.0, 0.86], NO_MAXIMA, false),
        (0, 1, 0, false)
    );
    assert_eq!(
        later(2, (0, 0), [regular_edge, 0.3, 0.0, 0.0], [0.5, 0.5], false),
        (1, 0, 0, false)
    );
}

#[test]
fn adding_a_difference_counts_up_and_clears_an_explosive_signal() {
    assert_eq!(
        RoundSignal::Settled.add_difference(),
        RoundSignal::Differenced
    );
    assert_eq!(
        RoundSignal::Explosive.add_difference(),
        RoundSignal::Differenced
    );
    assert_eq!(
        RoundSignal::Differenced.add_difference(),
        RoundSignal::BothDifferenced
    );
}

#[test]
fn parameter_layout_reads_each_factor_from_its_slot() {
    let first_round = ParameterLayout::new(working_model(1, ByFactor::new(0, 0)).unwrap());
    assert_eq!(first_round.ar_count(), 3);
    let later = ParameterLayout::new(working_model(2, ByFactor::new(1, 0)).unwrap());
    assert_eq!(later.ar_count(), 2);
    assert_eq!(
        later.leading(&[0.1, 0.2, 0.3, 0.4]),
        LeadingCoefficients {
            regular_ar: 0.1,
            seasonal_ar: 0.2,
            regular_ma: 0.3,
            seasonal_ma: 0.4,
        }
    );
}
