use crate::diagnostics::{
    Unavailable, UnavailableDetail, X11DiagnosticUnavailableReason as Reason,
};
use crate::time::TimeAxis;
use crate::x11::PreparedOptions;
use crate::x11::direct::extreme_scale::{ExtremeScaleState, SigmaState};
use crate::x11::direct::{RunInput, RunPurpose, adjust_easter, run_components, run_preliminary};
use crate::{Warning, x11};

#[test]
fn legacy_easter_reports_the_editor_warning_once() {
    // 3x15 on ten years: `editor.f` substitutes the stable filter and warns.
    let axis = TimeAxis::monthly(1990, 1).unwrap();
    let values: Vec<_> = (0..120)
        .map(|i| 100.0 + i as f64 * 0.5 + [3.0, -2.0, 1.0, -4.0][i % 4])
        .collect();
    let options = x11::Options::new().with_seasonal_filter(x11::SeasonalFilter::S3x15);
    let prepared = PreparedOptions::new(options, axis, values.len()).unwrap();
    let (components, easter) =
        adjust_easter(&values, axis, x11::Mode::Multiplicative, prepared).unwrap();
    assert_eq!(
        components.warnings,
        [Warning::SeasonalFilterReplaced {
            requested: x11::SeasonalFilter::S3x15,
            replacement: x11::SeasonalFilter::Stable,
        }]
    );
    assert!(easter.preliminary_trend_warnings.is_empty());
}

#[test]
fn sigma_entry_states_decide_both_tests_and_the_selection() {
    let axis = TimeAxis::monthly(1980, 1).unwrap();
    let options = x11::Options::new().with_extreme_scale(crate::x11::ExtremeScale::Automatic);
    let flat: Vec<_> = (0..120)
        .map(|i| 1.0 + f64::from((i * 37 % 17) as u8) / 400.0)
        .collect();
    let mut spiky = flat.clone();
    for index in (2..120).step_by(12) {
        spiky[index] *= 1.5 + f64::from((index % 5) as u8) / 4.0;
    }
    let mut outcomes = Vec::new();
    for entry in [
        None,
        Some(SigmaState::Undecided),
        Some(SigmaState::Rolling),
        Some(SigmaState::ByCalendarPeriod),
        Some(SigmaState::Latched),
    ] {
        for irregular in [&flat, &spiky] {
            let mut sigma = ExtremeScaleState::new(options);
            if let Some(entry) = entry {
                sigma = sigma.with_sigma_entry(entry);
            }
            for initial in [true, false] {
                sigma
                    .resolve(
                        irregular,
                        0..120,
                        12,
                        x11::Mode::Multiplicative,
                        initial,
                        axis.span(120).unwrap(),
                    )
                    .unwrap();
            }
            let selection = sigma.selection().unwrap();
            outcomes.push((
                entry,
                sigma.sigma,
                selection.initial_si(),
                selection.preliminary_irregular(),
                sigma.sigma.is_some_and(SigmaState::retains_c_extremes),
            ));
        }
    }
    use crate::x11::AutomaticExtremeScale::{ByCalendarPeriod as C, RollingFiveYear as R};
    use SigmaState::{ByCalendarPeriod, Latched, Rolling, Undecided};
    assert_eq!(
        outcomes,
        [
            (None, None, R, R, false),
            (None, None, C, C, false),
            (Some(Undecided), Some(Rolling), R, R, false),
            (Some(Undecided), Some(ByCalendarPeriod), C, C, false),
            (Some(Rolling), Some(Undecided), C, C, true),
            (Some(Rolling), Some(Latched), C, C, true),
            (Some(ByCalendarPeriod), Some(Rolling), R, R, false),
            (Some(ByCalendarPeriod), Some(ByCalendarPeriod), C, C, false),
            (Some(Latched), Some(Latched), C, C, true),
            (Some(Latched), Some(Latched), C, C, true),
        ]
    );
}

#[test]
fn entsch_dispatches_on_the_entry_state() {
    use SigmaState::{ByCalendarPeriod, Latched, Rolling, Undecided};
    // `entsch.f`'s `(Kersa, Ksdev)`: (0,0) -> (Iv,2); (0,2) and (1,1) ->
    // (Iv,Iv); (1,2) -> (1,2).
    for (entry, rolling, calendar) in [
        (Rolling, Undecided, Latched),
        (Undecided, Rolling, ByCalendarPeriod),
        (ByCalendarPeriod, Rolling, ByCalendarPeriod),
        (Latched, Latched, Latched),
    ] {
        assert_eq!(entry.after_test(false), rolling);
        assert_eq!(entry.after_test(true), calendar);
    }
    let options = x11::Options::new().with_extreme_scale(crate::x11::ExtremeScale::Automatic);
    let mut sigma = ExtremeScaleState::new(options).with_sigma_entry(Rolling);
    let axis = TimeAxis::monthly(1980, 1).unwrap();
    sigma
        .resolve(
            &vec![1.0; 120],
            0..120,
            12,
            x11::Mode::Multiplicative,
            true,
            axis.span(120).unwrap(),
        )
        .unwrap();
    sigma
        .resolve(
            &vec![1.0; 120],
            0..120,
            12,
            x11::Mode::Multiplicative,
            false,
            axis.span(120).unwrap(),
        )
        .unwrap();
    assert_eq!(
        sigma.selection().unwrap().initial_si(),
        crate::x11::AutomaticExtremeScale::ByCalendarPeriod
    );
    assert_eq!(
        sigma.selection().unwrap().preliminary_irregular(),
        crate::x11::AutomaticExtremeScale::ByCalendarPeriod
    );
}

#[test]
fn preliminary_easter_warning_reports_constant_and_exactly_linear_series() {
    let axis = TimeAxis::monthly(1980, 1).unwrap();
    let original = vec![100.0; 120];
    let warnings = crate::x11::seasonality::preliminary_easter_warning(
        &original,
        &original,
        axis,
        x11::Mode::Multiplicative,
    );
    let [
        Unavailable(UnavailableDetail::SeasonalityTest {
            test: crate::diagnostics::SeasonalityTest::ResidualFull,
            reason: Reason::ConstantOriginalSeries,
            sample: Some(sample),
            differencing_lag: Some(3),
        }),
    ] = warnings.as_slice()
    else {
        panic!("one warning for the constant series: {warnings:?}")
    };
    assert_eq!(
        (sample.first().year(), sample.first().subperiod()),
        (1980, 4)
    );
    assert_eq!(
        (sample.last().year(), sample.last().subperiod()),
        (1989, 12)
    );
    assert_eq!(sample.count(), 117);
    assert_eq!(
        warnings[0].to_string(),
        "the residual-seasonality test over the full span is unavailable: the original series is \
         constant"
    );
    let original: Vec<_> = (0..120).map(|i| 100.0 + i as f64).collect();
    let warnings = crate::x11::seasonality::preliminary_easter_warning(
        &original,
        &original,
        axis,
        x11::Mode::Multiplicative,
    );
    assert!(
        matches!(
            warnings.as_slice(),
            [Unavailable(UnavailableDetail::SeasonalityTest {
                reason: Reason::ZeroResidualMeanSquare,
                ..
            })]
        ),
        "{warnings:?}"
    );
}

#[test]
fn final_run_resumes_the_prepass_msr_choice_history_and_eligibility() {
    let axis = TimeAxis::monthly(1983, 1).unwrap();
    let values: Vec<_> = (0..36).map(|i| 100.0 + (i % 13) as f64).collect();
    let options = x11::Options::new().with_seasonal_filter(x11::SeasonalFilter::Msr);
    let prepared = PreparedOptions::new(options, axis, values.len()).unwrap();
    let mut preliminary = run_preliminary(RunInput {
        values: &values,
        raw: &values,
        axis,
        observed: 0..values.len(),
        mode: x11::Mode::PseudoAdditive,
        prepared: prepared.clone(),
    })
    .unwrap();
    assert_eq!(
        preliminary.msr,
        Some(crate::x11::MsrSelection::ShortHistoryFallback { passes: 0 })
    );
    preliminary.state.eligible.as_mut().unwrap()[0] = false;
    assert!(!preliminary.state.history.unwrap().include_m6);
    let prior = crate::x11::easter::estimate_requested(&preliminary.irregular, axis, 0, 0).unwrap();
    assert!(matches!(
        prior.status,
        crate::x11::EasterStatus::MissingDateGroups(_)
    ));
    let final_pass = run_components(
        RunInput {
            values: &values,
            raw: &values,
            axis,
            observed: 0..values.len(),
            mode: x11::Mode::PseudoAdditive,
            prepared,
        },
        RunPurpose::LegacyEasterFinal,
        preliminary.state,
    )
    .unwrap();
    assert_eq!(
        final_pass.msr_selection,
        Some(crate::x11::MsrSelection::SuppressedByEasterEstimation)
    );
    let workspace = final_pass.quality_workspace;
    assert!(final_pass.seasonally_adjusted[0] > 0.0);
    assert!(!workspace.intermediate_eligible[0]);
    assert!(!workspace.history.include_m6);
}
