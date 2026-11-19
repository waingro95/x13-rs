use super::*;
use crate::diagnostics::{UnavailableDetail::SeasonalityTest, UnavailableKind};

fn monthly() -> TimeAxis {
    TimeAxis::monthly(2000, 4).unwrap()
}

#[test]
fn zero_weight_takes_the_trend_in_the_modified_series() {
    let values =
        modified_adjusted(&[12.0, 13.0, 14.0], &[7.0, 8.0, 9.0], &[0.0, 0.25, 1.0]).unwrap();
    assert_eq!(values, [7.0, 13.0, 14.0]);
}

#[test]
fn constant_original_skips_both_residual_tests() {
    let values = (0..72).map(|i| (i * i % 17) as f64).collect::<Vec<_>>();
    let (full, recent) = residual_pair(
        &values,
        &[1.0; 72],
        monthly(),
        x11::Mode::Additive,
        Stage::ModifiedResidualFull,
        Stage::ModifiedResidualRecentThreeYears,
    );
    let full = full.expect_err("a constant original has no full test");
    assert_eq!(full.kind(), UnavailableKind::NotApplicable);
    assert_eq!(
        full.to_string(),
        "the modified residual-seasonality test over the full span is unavailable: the original \
         series is constant"
    );
    let differences = sample(monthly(), 3..72);
    assert_eq!(differences.as_ref().unwrap().count(), 69);
    assert_eq!(
        full,
        missing(
            Stage::ModifiedResidualFull,
            Reason::ConstantOriginalSeries,
            differences,
            Some(3)
        )
    );
    let recent = recent.expect_err("the recent test needs the full test");
    assert_eq!(recent.kind(), UnavailableKind::Numerical);
    assert_eq!(
        recent,
        missing(
            Stage::ModifiedResidualRecentThreeYears,
            Reason::PrerequisiteUnavailable {
                stage: Stage::ModifiedResidualFull
            },
            None,
            None,
        )
    );
}

#[test]
fn nonfinite_residual_test_adds_no_warning() {
    let original = (0..72).map(|i| (i * i % 17) as f64).collect::<Vec<_>>();
    let (full, recent) = residual_pair(
        &[f64::INFINITY; 72],
        &original,
        monthly(),
        x11::Mode::Additive,
        Stage::ModifiedResidualFull,
        Stage::ModifiedResidualRecentThreeYears,
    );
    let Err(Unavailable(SeasonalityTest { reason, .. })) = &full else {
        panic!("the differences are not finite")
    };
    assert_eq!(*reason, Reason::NonFiniteArithmetic);
    let mut warnings = Vec::new();
    append_warning(&full, &mut warnings);
    append_warning(&recent, &mut warnings);
    assert!(warnings.is_empty());
}

#[test]
fn residual_tests_difference_by_a_quarter_and_recent_spans_three_years() {
    for (axis, period, lag) in [
        (monthly(), 12, 3),
        (TimeAxis::quarterly(2000, 3).unwrap(), 4, 1),
    ] {
        for count in [3 * period + lag - 1, 5 * period] {
            let values = (0..count)
                .map(|i| (i * i % 17) as f64 + i as f64 / 7.0)
                .collect::<Vec<_>>();
            let (full, recent) = residual_pair(
                &values,
                &values,
                axis,
                x11::Mode::Additive,
                Stage::ModifiedResidualFull,
                Stage::ModifiedResidualRecentThreeYears,
            );
            let full = full.expect("the full test is available");
            assert_eq!(full.test().sample().count(), count - lag);
            assert_eq!(full.test().sample().first(), axis.period_at(lag).unwrap());
            assert_eq!(full.test().differencing_lag(), Some(lag));
            if count == 5 * period {
                let recent = recent.expect("the recent test is available");
                assert_eq!(recent.test().differencing_lag(), Some(lag));
                assert_eq!(recent.test().sample().count(), 3 * period);
                assert_eq!(
                    recent.test().sample().first(),
                    axis.period_at(2 * period).unwrap()
                );
                assert_eq!(
                    recent.test().sample().last(),
                    axis.period_at(count - 1).unwrap()
                );
            } else {
                let recent = recent.expect_err("three years of differences are not available");
                assert_eq!(recent.kind(), UnavailableKind::InsufficientData);
                assert_eq!(
                    recent,
                    missing(
                        Stage::ModifiedResidualRecentThreeYears,
                        Reason::InsufficientSample,
                        None,
                        None
                    )
                );
            }
        }
    }
}

fn available_f(statistic: f64, p: f64) -> diagnostics::FTest {
    diagnostics::FTest {
        statistic,
        upper_tail_probability_percent: p,
        numerator_degrees_of_freedom: 11,
        denominator_degrees_of_freedom: 120,
        sample: sample(monthly(), 0..132).unwrap(),
        differencing_lag: None,
        small_residual_mean_square: false,
    }
}

fn stable_record(f: f64, p: f64) -> Result<diagnostics::StableSeasonalityTest, Unavailable> {
    Ok(diagnostics::StableSeasonalityTest {
        test: available_f(f, p),
        conclusion: stable_conclusion(p),
    })
}

fn moving_record(f: f64, p: f64) -> Result<diagnostics::MovingSeasonalityTest, Unavailable> {
    Ok(diagnostics::MovingSeasonalityTest {
        test: available_f(f, p),
        conclusion: moving_conclusion(p),
    })
}

#[test]
#[expect(
    clippy::approx_constant,
    reason = "The expectation uses `fvalue`'s own literal."
)]
fn fvalue_parities_cutoffs_and_nonpositive_fallback() {
    for (m, n, expected) in [
        (1, 1, 1.0 - 0.5 * 0.31830988618379 * std::f64::consts::PI),
        (1, 2, 1.0 - (1.0_f64 / 3.0).sqrt()),
        (2, 1, (1.0_f64 / 3.0).sqrt()),
        (2, 2, 0.5),
    ] {
        let (f, p) = fvalue(1.0, m, n).unwrap();
        assert_eq!(f, 1.0);
        assert!((p - expected).abs() < 4.0 * f64::EPSILON, "{m}/{n}: {p}");
    }
    for f in [-10.0, -0.0, 0.0] {
        assert_eq!(fvalue(f, 11, 120), Ok((0.0, 1.0)));
    }
    assert_eq!(
        fvalue(90.0_f64.next_up(), 1, 1),
        Ok((90.0_f64.next_up(), 0.0))
    );
    assert!(fvalue(90.0, 1, 1).unwrap().1 > 0.0);
    assert_eq!(fvalue(40.0_f64.next_up(), 1, 151).unwrap().1, 0.0);
    assert!(fvalue(f64::INFINITY, 1, 1).is_err());
    assert!(fvalue(1.0, 0, 1).is_err());
    assert_eq!(fvalue(1e-100, 3, 1), Ok((0.0, 1.0)));
    for (f, n) in [(40.0, 150), (40.0, 151), (40.0_f64.next_up(), 150)] {
        let (post_f, p) = fvalue(f, 1, n).unwrap();
        assert_eq!(post_f, f);
        assert!(p.is_finite());
    }
}

#[test]
fn squares_accumulate_fused_and_integer_powers_square_repeatedly() {
    let values = [2.0_f64.powi(-27), 2.0_f64.powi(-27), 1.0 + f64::EPSILON];
    let sum = values
        .iter()
        .fold(0.0, |sum, &value| square_add(sum, value));
    assert_eq!(sum.to_bits(), 0x3ff0000000000003);
    let separate = values.iter().fold(0.0, |sum, &value| sum + value * value);
    assert_eq!(separate.to_bits(), 0x3ff0000000000002);
    assert_eq!(gfortran_powi(1.25, 0), 1.0);
    assert_eq!(gfortran_powi(1.25, 7), 4.76837158203125);
    // Repeated squaring is one ULP from seven multiplications in a row.
    assert_eq!(gfortran_powi(1.1, 7), 1.9487171000000012);
    assert_ne!(gfortran_powi(1.1, 7), (0..7).fold(1.0, |v, _| v * 1.1));
}

#[test]
fn moving_test_uses_complete_calendar_years_and_ties_rank_in_order() {
    let axis = TimeAxis::quarterly(2000, 2).unwrap();
    let values = (0..15)
        .map(|i| 10.0 + ((i * 7) % 11) as f64)
        .collect::<Vec<_>>();
    let m = moving(&values, axis, 0..15, x11::Mode::Additive).expect("three complete years");
    assert_eq!(
        m.test.sample.first,
        TimeAxis::quarterly(2001, 1).unwrap().start()
    );
    assert_eq!(
        m.test.sample.last,
        TimeAxis::quarterly(2003, 4).unwrap().start()
    );
    assert_eq!(m.test.sample.count, 12);
    assert_eq!(m.test.numerator_degrees_of_freedom, 2);
    assert_eq!(m.test.denominator_degrees_of_freedom, 6);
    let kw = kruskal_wallis(&[10.0; 8], axis, 0..8)
        .expect("ties take consecutive ranks, not their average");
    // Ranks 1..8 in order give a total of 172 for the squared rank sums over
    // their counts.
    assert_eq!(kw.statistic, 1.6666666666666679);
    assert_eq!(kw.sample.first, axis.start());
    assert_eq!(kw.sample.last, axis.period_at(7).unwrap());
    assert_eq!(kw.sample.count, 8);
}

#[test]
fn kruskal_wallis_ranks_ties_as_kwtest_swaps_them() {
    let axis = TimeAxis::quarterly(2000, 2).unwrap();
    for values in [
        [3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [0.0, -0.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ] {
        let mut with_prefix = vec![100.0];
        with_prefix.extend(values);
        let kw =
            kruskal_wallis(&with_prefix, axis, 1..9).expect("eight finite values can be ranked");
        // `kwtest`'s strict swaps rank the values in the order
        // [2,1,0,3,4,5,6,7], even when the minimum displaces an equal pair.
        // The quarters' rank sums are then {8,8,8,12}, so
        // 12*168/(8*9) - 27 = 1.
        assert_eq!(kw.statistic.to_bits(), 1.0_f64.to_bits());
        assert_eq!(kw.sample.first, axis.period_at(1).unwrap());
    }
}

#[test]
fn combined_test_decides_capped_cases_without_kruskal_wallis() {
    let missing_kw = unavailable(Stage::KruskalWallis, Reason::RankIntegerOverflow);
    for (fs, fm) in [(0.0, 0.0), (0.5, 0.0), (7.0, 21.0), (7.0, 210.0)] {
        assert_eq!(
            combined(
                &stable_record(fs, 0.0),
                &moving_record(fm, 5.0),
                &missing_kw
            ),
            Ok(diagnostics::CombinedSeasonality::NotPresent)
        );
        assert_eq!(
            combined(
                &stable_record(fs, 0.0),
                &moving_record(fm, 5.0_f64.next_up()),
                &missing_kw
            ),
            Ok(diagnostics::CombinedSeasonality::ProbablyNotPresent)
        );
    }
}

#[test]
fn combined_test_reports_overflow_before_the_cap() {
    let missing_kw = unavailable(Stage::KruskalWallis, Reason::RankIntegerOverflow);
    let result = combined(
        &stable_record(7.0, 0.0),
        &moving_record(f64::MAX, 6.0),
        &missing_kw,
    );
    assert_eq!(
        result,
        unavailable(Stage::Combined, Reason::NonFiniteArithmetic),
        "3 * Fm overflows"
    );
}

#[test]
fn unavailable_tests_keep_their_sample_dates() {
    let axis = TimeAxis::quarterly(2000, 2).unwrap();
    let Err(Unavailable(SeasonalityTest {
        test,
        reason,
        sample,
        differencing_lag,
    })) = residual(
        &[0.0; 20],
        axis,
        1..20,
        x11::Mode::Additive,
        1,
        Stage::ResidualFull,
    )
    else {
        panic!("the residual mean square is zero")
    };
    assert_eq!(test, Stage::ResidualFull);
    assert_eq!(reason, Reason::ZeroResidualMeanSquare);
    assert_eq!(sample.as_ref().unwrap().first, axis.period_at(1).unwrap());
    assert_eq!(sample.as_ref().unwrap().count, 19);
    assert_eq!(differencing_lag, Some(1));
    let Err(Unavailable(SeasonalityTest {
        test,
        reason,
        sample,
        ..
    })) = residual(
        &[f64::INFINITY; 20],
        axis,
        8..20,
        x11::Mode::Additive,
        1,
        Stage::ResidualRecentThreeYears,
    )
    else {
        panic!("the values are not finite")
    };
    assert_eq!(test, Stage::ResidualRecentThreeYears);
    assert_eq!(reason, Reason::NonFiniteArithmetic);
    assert_eq!(sample.unwrap().first, axis.period_at(8).unwrap());
    assert_eq!(
        stable(&[], axis, 0..0, x11::Mode::Additive, Stage::FinalStable),
        unavailable(Stage::FinalStable, Reason::InsufficientSample),
        "there are no values"
    );
}

#[test]
fn conclusions_at_their_thresholds() {
    assert_eq!(
        kw_conclusion(1.0),
        diagnostics::KruskalWallisSeasonality::PresentAtOnePercent
    );
    assert_eq!(
        kw_conclusion(1.0_f64.next_up()),
        diagnostics::KruskalWallisSeasonality::NotPresentAtOnePercent
    );
    assert_eq!(
        stable_conclusion(0.1),
        diagnostics::StableSeasonality::PresentAtPointOnePercent
    );
    assert_eq!(
        stable_conclusion(0.1_f64.next_up()),
        diagnostics::StableSeasonality::NotPresentAtPointOnePercent
    );
    assert_eq!(
        moving_conclusion(1.0),
        diagnostics::MovingSeasonality::PresentAtOnePercent
    );
    assert_eq!(
        moving_conclusion(5.0),
        diagnostics::MovingSeasonality::NotPresentAtFivePercent
    );
    assert_eq!(
        moving_conclusion(5.0_f64.next_down()),
        diagnostics::MovingSeasonality::PresentAtFivePercent
    );
    assert_eq!(
        residual_conclusion(5.0, true),
        diagnostics::ResidualSeasonality::PresentAtFivePercent
    );
    assert_eq!(
        residual_conclusion(1.0, false),
        diagnostics::ResidualSeasonality::PresentAtOnePercent
    );
    assert_eq!(
        residual_conclusion(1.0_f64.next_up(), false),
        diagnostics::ResidualSeasonality::NotPresentAtOnePercent
    );
}

#[test]
fn combined_test_needs_only_the_tests_it_reads() {
    let missing_moving = unavailable(Stage::Moving, Reason::ZeroResidualMeanSquare);
    let missing_kw = unavailable(Stage::KruskalWallis, Reason::RankIntegerOverflow);
    assert_eq!(
        combined(&stable_record(100.0, 0.1), &missing_moving, &missing_kw),
        Ok(diagnostics::CombinedSeasonality::NotPresent)
    );
    assert_eq!(
        combined(
            &stable_record(1.0, 0.01),
            &moving_record(1.0, 5.0),
            &missing_kw
        ),
        Ok(diagnostics::CombinedSeasonality::NotPresent)
    );
    assert_eq!(
        combined(
            &stable_record(7.0, 0.01),
            &moving_record(0.0, 6.0),
            &missing_kw
        ),
        Ok(diagnostics::CombinedSeasonality::ProbablyNotPresent)
    );
    assert_eq!(
        combined(
            &stable_record(70.0, 0.01),
            &moving_record(70.0 / 3.0, 6.0),
            &missing_kw
        ),
        Ok(diagnostics::CombinedSeasonality::ProbablyNotPresent)
    );
    let result = combined(
        &stable_record(70.0, 0.01),
        &moving_record(0.0, 6.0),
        &missing_kw,
    );
    assert_eq!(
        result,
        unavailable(
            Stage::Combined,
            Reason::PrerequisiteUnavailable {
                stage: Stage::KruskalWallis
            }
        ),
        "the Kruskal-Wallis test is read"
    );
    let kw = Ok(diagnostics::KruskalWallisTest {
        statistic: 30.0,
        upper_tail_probability_percent: 0.1,
        degrees_of_freedom: 11,
        sample: sample(monthly(), 0..132).unwrap(),
        conclusion: diagnostics::KruskalWallisSeasonality::PresentAtOnePercent,
    });
    assert_eq!(
        combined(&stable_record(70.0, 0.01), &moving_record(0.0, 6.0), &kw),
        Ok(diagnostics::CombinedSeasonality::Present)
    );
}

#[test]
fn short_overflowing_and_unrankable_samples_are_unavailable() {
    let axis = TimeAxis::quarterly(2000, 2).unwrap();
    assert_eq!(
        reason(stable(
            &[1.0; 4],
            axis,
            0..4,
            x11::Mode::Additive,
            Stage::FinalStable
        )),
        Reason::InsufficientSample,
        "one year leaves no residual degrees of freedom"
    );
    assert_eq!(
        reason(stable(
            &[f64::MAX; 40],
            axis,
            0..40,
            x11::Mode::Additive,
            Stage::FinalStable,
        )),
        Reason::NonFiniteArithmetic,
        "the sums overflow"
    );
    let values = (0..780).map(|i| i as f64).collect::<Vec<_>>();
    assert_eq!(
        reason(kruskal_wallis(&values, axis, 0..780)),
        Reason::RankIntegerOverflow,
        "the squared i32 rank sums overflow"
    );
    assert_eq!(
        reason(moving(&[1.0; 6], axis, 0..6, x11::Mode::Multiplicative)),
        Reason::InsufficientSample,
        "there is only one complete year"
    );
}

/// Why a test has no value.
fn reason<T: std::fmt::Debug>(test: Result<T, Unavailable>) -> Reason {
    match test {
        Err(Unavailable(SeasonalityTest { reason, .. })) => reason,
        other => panic!("not an unavailable seasonality test: {other:?}"),
    }
}

#[test]
fn unavailable_kind_follows_the_reason() {
    for (reason, kind) in [
        (
            Reason::ConstantOriginalSeries,
            UnavailableKind::NotApplicable,
        ),
        (
            Reason::InsufficientSample,
            UnavailableKind::InsufficientData,
        ),
        (Reason::ZeroResidualMeanSquare, UnavailableKind::Numerical),
        (Reason::NonFiniteArithmetic, UnavailableKind::Numerical),
        (Reason::RankIntegerOverflow, UnavailableKind::Numerical),
        (
            Reason::PrerequisiteUnavailable {
                stage: Stage::Moving,
            },
            UnavailableKind::Numerical,
        ),
    ] {
        assert_eq!(
            missing(Stage::Combined, reason, None, None).kind(),
            kind,
            "{reason:?}"
        );
    }
}

#[test]
fn kruskal_wallis_overflow_returns_before_the_quadratic_tie_ranking() {
    // Ties force `kwtest`'s O(n^2) swap ranking; 200,000 tied values would
    // take about 2e10 comparisons. The pigeonhole bound applies from
    // n = 1,055 monthly and n = 609 quarterly, where every ranking overflows.
    for (axis, threshold) in [
        (TimeAxis::monthly(2000, 4).unwrap(), 1055),
        (TimeAxis::quarterly(2000, 2).unwrap(), 609),
    ] {
        for count in [threshold, 200_000] {
            let values = vec![1.0; count];
            let Err(Unavailable(SeasonalityTest { reason, sample, .. })) =
                kruskal_wallis(&values, axis, 0..count)
            else {
                panic!("{count} tied values overflow the i32 rank sums")
            };
            assert_eq!(reason, Reason::RankIntegerOverflow);
            assert_eq!(sample.map(|sample| sample.count), Some(count));
        }
        // One below the bound a ranking still fits: dealing ranks n..1 to the
        // open slot with the smallest sum keeps the largest at 46,334 monthly
        // and 46,284 quarterly, so the early return must not apply there.
        let count = threshold - 1;
        let period = usize::from(axis.frequency().periods_per_year());
        let first = usize::from(axis.start().subperiod()) - 1;
        let mut open = vec![Vec::new(); period];
        for index in 0..count {
            open[(first + index) % period].push(index);
        }
        let mut sums = vec![0; period];
        let mut values = vec![0.0; count];
        for rank in (1..=count).rev() {
            let slot = (0..period)
                .filter(|&slot| !open[slot].is_empty())
                .min_by_key(|&slot| sums[slot])
                .unwrap();
            sums[slot] += rank;
            values[open[slot].pop().unwrap()] = rank as f64;
        }
        assert!(sums.iter().all(|&sum| sum <= 46_340), "{sums:?}");
        let Ok(test) = kruskal_wallis(&values, axis, 0..count) else {
            panic!("{count} balanced ranks fit in i32")
        };
        assert_eq!(test.sample.count, count);
    }
}

#[test]
fn constant_values_have_no_stable_test_and_keep_their_span() {
    let axis = monthly();
    let constant = stable(
        &[100.0; 120],
        axis,
        6..114,
        x11::Mode::Multiplicative,
        Stage::PreliminaryStable,
    );
    let Err(Unavailable(SeasonalityTest { reason, sample, .. })) = constant else {
        panic!("the residual mean square is zero")
    };
    assert_eq!(reason, Reason::ZeroResidualMeanSquare);
    assert_eq!(sample.as_ref().unwrap().first(), axis.period_at(6).unwrap());
    assert_eq!(sample.unwrap().last(), axis.period_at(113).unwrap());
}
