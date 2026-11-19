use super::ranking::{BestFive, RankedCandidate};
use super::{SearchState, regular_orders_for_revisit};
use crate::arima::PreparedSearchInput;
use crate::error::{NumericalDetail, NumericalFailure};
use crate::regarima::ArimaFactor;
use crate::time::TimeAxis;
use crate::{ArimaModel, ArimaOrder, Error, Transform};

fn model(p: u8, d: u8, q: u8, sp: u8, sd: u8, sq: u8) -> ArimaModel {
    ArimaModel::new(
        ArimaOrder::new(p, d, q).unwrap(),
        ArimaOrder::new(sp, sd, sq).unwrap(),
    )
    .unwrap()
}

fn ranked(entries: &[(ArimaModel, f64)]) -> BestFive {
    let mut ranking = BestFive::default();
    for &(model, bic2) in entries {
        ranking.insert(RankedCandidate { model, bic2 });
    }
    ranking
}

fn ranked_models(ranking: &BestFive) -> Vec<ArimaModel> {
    ranking
        .slots()
        .iter()
        .flatten()
        .map(|entry| entry.model)
        .collect()
}

/// The parsimony pass over `entries` with no differencing.
fn select(entries: &[(ArimaModel, f64)]) -> ArimaModel {
    ranked(entries).select(model(0, 0, 0, 0, 0, 0)).unwrap()
}

fn new_state() -> SearchState {
    SearchState {
        ranking: BestFive::default(),
        converged: true,
    }
}

#[test]
fn a_tie_ranks_behind_the_earlier_candidate() {
    let first = model(0, 0, 0, 0, 0, 0);
    let equal = model(1, 0, 0, 0, 0, 0);
    let ranking = ranked(&[(first, 1.0), (equal, 1.0)]);
    assert_eq!(ranked_models(&ranking), [first, equal]);
}

#[test]
fn the_sixth_candidate_is_dropped_and_can_be_ranked_again() {
    let sixth = model(3, 0, 0, 0, 0, 0);
    let mut ranking = ranked(&[
        (model(0, 0, 0, 0, 0, 0), 1.0),
        (model(1, 0, 0, 0, 0, 0), 1.0),
        (model(0, 0, 1, 0, 0, 0), 2.0),
        (model(2, 0, 0, 0, 0, 0), 3.0),
        (model(0, 0, 2, 0, 0, 0), 4.0),
        (sixth, 5.0),
    ]);
    assert!(!ranking.contains(sixth));
    ranking.insert(RankedCandidate {
        model: sixth,
        bic2: 0.5,
    });
    assert_eq!(ranking.best().unwrap().model, sixth);
    ranking.clear();
    assert!(ranked_models(&ranking).is_empty());
}

#[test]
fn contains_compares_the_differencing_orders_too() {
    let ranking = ranked(&[(model(1, 1, 0, 0, 1, 0), 1.0)]);
    assert!(ranking.contains(model(1, 1, 0, 0, 1, 0)));
    assert!(!ranking.contains(model(1, 0, 0, 0, 1, 0)));
    assert!(!ranking.contains(model(1, 1, 0, 0, 0, 0)));
}

#[test]
fn a_nan_bic2_is_ranked_and_stays_ahead_of_a_finite_one() {
    let nan = model(1, 0, 0, 0, 0, 0);
    let finite = model(0, 0, 1, 0, 0, 0);
    // A comparison with NaN is false, so the finite candidate does not
    // displace it.
    let ranking = ranked(&[(nan, f64::NAN), (finite, 1.0)]);
    assert_eq!(ranked_models(&ranking), [nan, finite]);
    assert!(ranking.best().unwrap().bic2.is_nan());
}

#[test]
fn an_empty_table_has_no_choice() {
    let ranking = BestFive::default();
    assert_eq!(
        regular_orders_for_revisit(&ranking, 0),
        Err(Error::Numerical(NumericalFailure(
            NumericalDetail::NoEstimableArmaCandidate {
                factor: ArimaFactor::Regular
            }
        )))
    );
    assert_eq!(ranking.select(model(0, 0, 0, 0, 0, 0)), None);
}

#[test]
fn the_revisit_takes_the_second_best_regular_orders_when_the_best_has_no_terms() {
    let mut ranking = ranked(&[(model(0, 1, 0, 0, 0, 0), 1.0)]);
    assert_eq!(
        regular_orders_for_revisit(&ranking, 1).unwrap(),
        ArimaOrder::new(0, 1, 0).unwrap()
    );
    ranking.insert(RankedCandidate {
        model: model(1, 1, 0, 0, 0, 0),
        bic2: 2.0,
    });
    assert_eq!(
        regular_orders_for_revisit(&ranking, 1).unwrap(),
        ArimaOrder::new(1, 1, 0).unwrap()
    );
}

#[test]
fn a_fitted_candidate_is_ranked() {
    let values: Vec<f64> = (0..40)
        .map(|index| if index % 2 == 0 { -1.0 } else { 1.0 })
        .collect();
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let input =
        PreparedSearchInput::new(&values, (&values).into(), axis, Transform::None, &[]).unwrap();
    let candidate = model(0, 0, 0, 0, 0, 0);
    let mut state = new_state();
    state
        .try_candidate(&input, candidate, &mut Vec::new())
        .unwrap();
    assert_eq!(ranked_models(&state.ranking), [candidate]);
}

#[test]
fn a_candidate_whose_bic2_cannot_be_computed_fails_the_search() {
    let values: Vec<f64> = (0..40)
        .map(|index| if index % 2 == 0 { 1e-9 } else { -1e-9 })
        .collect();
    let axis = TimeAxis::monthly(2000, 1).unwrap();
    let input =
        PreparedSearchInput::new(&values, (&values).into(), axis, Transform::None, &[]).unwrap();
    let candidate = model(0, 0, 0, 0, 0, 0);
    let mut state = new_state();
    assert!(matches!(
        state.try_candidate(&input, candidate, &mut Vec::new()),
        Err(Error::Numerical(NumericalFailure(NumericalDetail::ModelSelectionUnavailable { candidate: failed }))) if failed == candidate.into()
    ));
    assert!(ranked_models(&state.ranking).is_empty());
}

#[test]
fn each_parsimony_rule_prefers_the_simpler_model() {
    let cases = [
        // A first regular term is added.
        (model(0, 0, 0, 0, 0, 1), model(1, 0, 0, 0, 0, 1), 0.0001),
        // Regular terms are dropped.
        (model(2, 0, 0, 0, 0, 0), model(1, 0, 0, 0, 0, 0), 0.0001),
        // Seasonal terms are dropped.
        (model(0, 0, 1, 2, 0, 0), model(0, 0, 1, 1, 0, 0), 0.0001),
        // A regular AR term is dropped, within the wider threshold.
        (model(2, 0, 1, 0, 0, 0), model(1, 0, 1, 0, 0, 0), 0.004),
    ];
    for (best, simpler, score) in cases {
        assert_eq!(select(&[(best, 0.0), (simpler, score)]), simpler);
    }
}

#[test]
fn the_thresholds_shrink_with_the_spread_of_the_table() {
    let best = model(0, 0, 0, 0, 0, 1);
    let alternative = model(1, 0, 0, 0, 0, 1);
    let entries = |alternative_score, fifth_score| {
        [
            (best, 0.0),
            (alternative, alternative_score),
            (model(0, 0, 1, 0, 0, 1), 0.001),
            (model(0, 0, 2, 0, 0, 1), 0.002),
            (model(2, 0, 2, 0, 0, 1), fifth_score),
        ]
    };
    // A spread below 0.003 scales the thresholds by 1/16, one below 0.03
    // by 1/4.
    assert_eq!(select(&entries(0.0005, 0.003)), alternative);
    assert_eq!(select(&entries(0.0005, 0.003_f64.next_down())), best);
    assert_eq!(select(&entries(0.001, 0.03)), alternative);
    assert_eq!(select(&entries(0.001, 0.03_f64.next_down())), best);
}

#[test]
fn the_thresholds_are_strict() {
    let best = model(0, 0, 0, 0, 0, 1);
    let alternative = model(1, 0, 0, 0, 0, 1);
    let entries = |alternative_score| {
        [
            (best, 0.0),
            (alternative, alternative_score),
            (model(0, 0, 1, 0, 0, 1), 0.01),
            (model(0, 0, 2, 0, 0, 1), 0.02),
            (model(2, 0, 2, 0, 0, 1), 0.1),
        ]
    };
    assert_eq!(select(&entries(0.0025)), best);
    assert_eq!(select(&entries(0.0025_f64.next_down())), alternative);

    let ar2 = model(2, 0, 1, 0, 0, 0);
    let ar1 = model(1, 0, 1, 0, 0, 0);
    assert_eq!(select(&[(ar2, 0.0), (ar1, 0.0075)]), ar2);
    assert_eq!(select(&[(ar2, 0.0), (ar1, 0.0075_f64.next_down())]), ar1);
}

#[test]
fn a_model_without_arma_terms_gives_way_to_the_next_rank() {
    let next = model(0, 0, 1, 0, 0, 0);
    assert_eq!(select(&[(model(0, 0, 0, 0, 0, 0), 0.0), (next, 0.1)]), next);
    // An empty next slot reads as the model with no ARMA terms.
    let only = model(0, 1, 0, 0, 1, 0);
    assert_eq!(ranked(&[(only, 1.0)]).select(only), Some(only));
}

#[test]
fn each_switch_tightens_the_thresholds_by_its_distance_from_the_best() {
    let entries = [
        (model(3, 0, 1, 0, 0, 0), 0.0),
        (model(2, 0, 1, 0, 0, 0), 0.0005),
        (model(1, 0, 1, 0, 0, 0), 0.0028),
        (model(0, 0, 1, 0, 0, 0), 0.0072),
    ];
    assert_eq!(select(&entries), model(1, 0, 1, 0, 0, 0));
}
