//! Robust scale estimates of the irregular from Build 62 `sdxtrm.f`: the
//! median absolute deviation and its tau-adjusted form, on the values or on
//! their logs.

/// Which robust scale [`scale`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RobustEstimator {
    /// The median absolute deviation from the center, scaled to a standard
    /// deviation.
    Mad,
    /// The median absolute log, converted to a log-normal standard
    /// deviation.
    LogMad,
    /// [`Self::Mad`] with the tau adjustment.
    Tau,
    /// [`Self::LogMad`] with the tau adjustment.
    LogTau,
}

/// Why a sample has no robust scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RobustScaleFailure {
    /// The sample is empty.
    EmptySample,
    /// The tau adjustment would divide a zero deviation by a zero scale.
    ZeroTauBase,
    /// A value, or its deviation from the center, is not finite.
    NonFiniteDeviation,
    /// A log estimator met a value that is not positive.
    NonPositiveLogInput,
    /// The scale is not finite.
    NonFiniteScale,
}

/// The robust scale of `values`. The plain estimators measure deviations
/// from `center`; the log estimators ignore it and use `|ln(value)|`.
///
/// The tau estimators divide their sum by `post_loop_index`, not by the
/// number of values: `sdxtrm.f` divides by its loop variable after the loop
/// has ended (see [`post_loop_index`]).
pub(super) fn scale(
    values: impl IntoIterator<Item = f64>,
    center: f64,
    estimator: RobustEstimator,
    post_loop_index: usize,
) -> Result<f64, RobustScaleFailure> {
    let mut deviations = Vec::new();
    let is_log = matches!(estimator, RobustEstimator::LogMad | RobustEstimator::LogTau);
    for value in values {
        let deviation = if is_log {
            if !value.is_finite() {
                return Err(RobustScaleFailure::NonFiniteDeviation);
            }
            if value <= 0.0 {
                return Err(RobustScaleFailure::NonPositiveLogInput);
            }
            value.ln().abs()
        } else {
            (value - center).abs()
        };
        if !deviation.is_finite() {
            return Err(RobustScaleFailure::NonFiniteDeviation);
        }
        deviations.push(deviation);
    }
    if deviations.is_empty() {
        return Err(RobustScaleFailure::EmptySample);
    }
    crate::numerics::shell_sort(&mut deviations);

    let middle = deviations.len() / 2;
    let median = if deviations.len().is_multiple_of(2) {
        (deviations[middle - 1] + deviations[middle]) / 2.0
    } else {
        deviations[middle]
    };
    let initial = median / crate::numerics::BUILD_62_MAD_SCALE;
    if !initial.is_finite() {
        return Err(RobustScaleFailure::NonFiniteScale);
    }
    let base = if is_log {
        let exponential = (initial * initial).exp();
        (exponential * (exponential - 1.0)).sqrt()
    } else {
        initial
    };
    if !base.is_finite() {
        return Err(RobustScaleFailure::NonFiniteScale);
    }
    if matches!(estimator, RobustEstimator::Mad | RobustEstimator::LogMad) {
        return Ok(base);
    }
    // sdxtrm.f's log conversion can round a positive median's scale to zero.
    // Positive deviations then divide to infinity, which `rho` caps at 6.502,
    // leaving a zero tau scale. Only a zero deviation, 0/0, has no value.
    if base == 0.0 && deviations.contains(&0.0) {
        return Err(RobustScaleFailure::ZeroTauBase);
    }

    let mut sum = 0.0;
    for deviation in deviations {
        sum += rho(deviation / base);
    }
    let adjusted = (base * base * sum / post_loop_index as f64).sqrt();
    if !adjusted.is_finite() {
        return Err(RobustScaleFailure::NonFiniteScale);
    }
    Ok(adjusted)
}

/// The value a Fortran `DO i = first, last, stride` leaves in `i`: one
/// stride past the last index it visits. `None` for a zero stride, a loop
/// that would not run, or an index that overflows.
pub(super) fn post_loop_index(first: usize, last: usize, stride: usize) -> Option<usize> {
    if stride == 0 || first > last {
        return None;
    }
    let steps = (last - first) / stride;
    first.checked_add(steps.checked_add(1)?.checked_mul(stride)?)
}

/// `rho2.f`: the bounded loss the tau adjustment sums, constant beyond
/// 2.798.
fn rho(u: f64) -> f64 {
    if u.abs() > 2.798 {
        6.502
    } else {
        let u2 = u * u;
        (0.9249 * u2) + (0.0812 * u2 * u2) - (0.0119 * u2 * u2 * u2)
    }
}

#[cfg(test)]
mod tests {
    use super::{RobustEstimator, RobustScaleFailure, post_loop_index, rho, scale};

    fn assert_close(actual: f64, bits: u64) {
        let expected = f64::from_bits(bits);
        assert!(actual.is_finite());
        assert!(expected != 0.0);
        assert!(
            ((actual - expected) / expected).abs() <= 6e-15,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn log_estimates_for_odd_and_even_counts() {
        // Expected values worked out from sdxtrm.f in binary64, for tau
        // divisors of 4 and 31. The center makes no difference.
        for (values, mad, tau4, tau31) in [
            (
                vec![0.5, 1.0, 2.0],
                0x4002_9302_8c3e_db44,
                0x3fde_489d_b446_b17c,
                0x3fc5_c1a4_ce7c_8605,
            ),
            (
                vec![0.5, 2.0, 4.0, 8.0],
                0x4024_807b_91bb_7173,
                0x3ff4_ae70_22a2_6a3a,
                0x3fdd_b745_ebcd_e154,
            ),
            (
                vec![1.0 + 2e-8; 3],
                0x3e60_0000_0000_0002,
                0x3e52_3689_a7f1_0d38,
                0x3e3a_2b54_fb6f_d40d,
            ),
            (
                vec![1.0 + 2e-8, 1.0 + 2e-8, 1e200],
                0x3e60_0000_0000_0002,
                0x3e65_b649_ee21_861e,
                0x3e4f_3262_9592_ceae,
            ),
        ] {
            for center in [0.0, 17.0] {
                assert_close(
                    scale(values.iter().copied(), center, RobustEstimator::LogMad, 4).unwrap(),
                    mad,
                );
                assert_close(
                    scale(values.iter().copied(), center, RobustEstimator::LogTau, 4).unwrap(),
                    tau4,
                );
                assert_close(
                    scale(values.iter().copied(), center, RobustEstimator::LogTau, 31).unwrap(),
                    tau31,
                );
            }
        }
    }

    #[test]
    fn log_estimates_reject_nonpositive_and_nonfinite_values() {
        for estimator in [RobustEstimator::LogMad, RobustEstimator::LogTau] {
            assert_eq!(
                scale([], 0.0, estimator, 4),
                Err(RobustScaleFailure::EmptySample)
            );
            for value in [-1.0, 0.0, -0.0] {
                assert_eq!(
                    scale([value], 0.0, estimator, 4),
                    Err(RobustScaleFailure::NonPositiveLogInput)
                );
            }
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    scale([value], 0.0, estimator, 4),
                    Err(RobustScaleFailure::NonFiniteDeviation)
                );
            }
            assert_eq!(
                scale([f64::MAX; 3], 0.0, estimator, 4),
                Err(RobustScaleFailure::NonFiniteScale)
            );
        }
    }

    #[test]
    fn log_tau_needs_a_nonzero_scale_or_deviation() {
        let adjacent = f64::from_bits(1.0_f64.to_bits() + 1);
        for values in [[1.0; 3], [adjacent; 3]] {
            assert_eq!(scale(values, 0.0, RobustEstimator::LogMad, 4), Ok(0.0));
        }
        assert_eq!(
            scale([1.0; 3], 0.0, RobustEstimator::LogTau, 4),
            Err(RobustScaleFailure::ZeroTauBase)
        );
        assert_eq!(
            scale([adjacent; 3], 0.0, RobustEstimator::LogTau, 4),
            Ok(0.0)
        );
    }

    #[test]
    fn mad_measures_from_the_given_center() {
        // An odd and an even count whose median deviation is 2.
        assert_eq!(
            scale([-3.0, 1.0, -2.0], 0.0, RobustEstimator::Mad, 7)
                .unwrap()
                .to_bits(),
            0x4007_b8a5_7ac8_048e
        );
        assert_eq!(
            scale([4.0, 0.0, 3.0, -1.0], 1.0, RobustEstimator::Mad, 8)
                .unwrap()
                .to_bits(),
            0x4007_b8a5_7ac8_048e
        );
    }

    #[test]
    fn mad_of_zeros_is_zero() {
        assert_eq!(scale([0.0, -0.0], 0.0, RobustEstimator::Mad, 3), Ok(0.0));
    }

    #[test]
    fn tau_may_underflow_to_zero() {
        assert_eq!(scale([1.0e-200; 3], 0.0, RobustEstimator::Tau, 4), Ok(0.0));
    }

    #[test]
    fn tau_rejects_a_zero_scale_with_a_zero_deviation() {
        assert_eq!(
            scale([0.0, -0.0], 0.0, RobustEstimator::Tau, 3),
            Err(RobustScaleFailure::ZeroTauBase)
        );
        assert_eq!(
            scale([1.0e-20, 1.0e-20, 2.0e-20], 0.0, RobustEstimator::Tau, 7)
                .unwrap()
                .to_bits(),
            0x3bc5_f457_5b3f_f30c
        );
    }

    #[test]
    fn rho_is_constant_strictly_beyond_its_cutoff() {
        // The polynomial is evaluated term by term, without fusion.
        assert_eq!(rho(0.5).to_bits(), 0x3fce_38fc_5048_16f0);
        assert_eq!(rho(2.798).to_bits(), 0x401a_07d8_3738_6cc8);
        assert_eq!(rho(-2.798).to_bits(), 0x401a_07d8_3738_6cc8);
        assert_eq!(
            rho(f64::from_bits(2.798_f64.to_bits() + 1)).to_bits(),
            0x401a_020c_49ba_5e35
        );
        assert_eq!(rho(3.0).to_bits(), 0x401a_020c_49ba_5e35);
    }

    #[test]
    fn tau_divides_by_the_post_loop_index() {
        assert_eq!(
            scale([3.0, 1.0, 2.0], 0.0, RobustEstimator::Tau, 7)
                .unwrap()
                .to_bits(),
            0x3ff6_68ca_c3fd_68a8
        );
    }

    #[test]
    fn tau_caps_an_overflowed_ratio() {
        assert_eq!(
            scale([1.0e-150, 1.0e-150, 1.0e200], 0.0, RobustEstimator::Tau, 4)
                .unwrap()
                .to_bits(),
            0x20da_5bd8_3072_61af
        );
    }

    #[test]
    fn unusable_samples_have_distinct_reasons() {
        assert_eq!(
            scale([], 0.0, RobustEstimator::Mad, 31),
            Err(RobustScaleFailure::EmptySample)
        );
        assert_eq!(
            scale([f64::NAN], 0.0, RobustEstimator::Mad, 2),
            Err(RobustScaleFailure::NonFiniteDeviation)
        );
        assert_eq!(
            scale([f64::MAX], -f64::MAX, RobustEstimator::Mad, 2),
            Err(RobustScaleFailure::NonFiniteDeviation)
        );
        assert_eq!(
            scale([f64::MAX; 2], 0.0, RobustEstimator::Mad, 3),
            Err(RobustScaleFailure::NonFiniteScale)
        );
        assert_eq!(
            scale([1.0e200; 3], 0.0, RobustEstimator::Tau, 4),
            Err(RobustScaleFailure::NonFiniteScale)
        );
    }

    #[test]
    fn post_loop_index_steps_once_past_the_last_visit() {
        assert_eq!(post_loop_index(7, 19, 12), Some(31));
        assert_eq!(post_loop_index(10, 25, 12), Some(34));
        assert_eq!(post_loop_index(7, 7, 12), Some(19));
        assert_eq!(post_loop_index(7, 19, 0), None);
        assert_eq!(post_loop_index(20, 19, 12), None);
        assert_eq!(post_loop_index(usize::MAX - 1, usize::MAX, 2), None);
    }
}
