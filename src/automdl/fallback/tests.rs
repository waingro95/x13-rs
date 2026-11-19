use super::*;

type Orders = (u8, u8, u8);

const AIRLINE: (Orders, Orders) = ((0, 1, 1), (0, 1, 1));

fn compared(
    (regular, seasonal): (Orders, Orders),
    procedural_mean: bool,
    automatic_outlier_count: usize,
    confidence: f64,
    innovation_scale: f64,
) -> ComparedModel {
    ComparedModel {
        model: ResolvedArimaModel::new(
            [regular.0, regular.1, regular.2],
            [seasonal.0, seasonal.1, seasonal.2],
        )
        .unwrap(),
        procedural_mean,
        automatic_outlier_count,
        metrics: ComparisonMetrics {
            confidence,
            innovation_scale,
        },
    }
}

/// Whether a (2,1,1)(0,1,1) model with the given confidence and scale goes
/// back to an airline reference with `reference_confidence` and
/// `reference_scale`.
fn prefers(
    confidence: f64,
    scale: f64,
    reference_confidence: f64,
    reference_scale: f64,
    stage: LoopStage,
) -> bool {
    prefers_reference(
        &compared(((2, 1, 1), (0, 1, 1)), false, 0, confidence, scale),
        &compared(AIRLINE, false, 0, reference_confidence, reference_scale),
        None,
        stage,
    )
}

#[test]
fn the_models_must_differ_and_the_reference_have_no_more_outliers() {
    let current = compared(AIRLINE, false, 0, 0.9, 2.0);
    let mut reference = compared(AIRLINE, false, 0, 0.5, 1.0);
    assert!(!prefers_reference(
        &current,
        &reference,
        None,
        LoopStage::First
    ));
    // The same orders with a different mean are a different model.
    reference.procedural_mean = true;
    assert!(prefers_reference(
        &current,
        &reference,
        None,
        LoopStage::First
    ));
    reference.automatic_outlier_count = 1;
    assert!(!prefers_reference(
        &current,
        &reference,
        None,
        LoopStage::First
    ));
}

#[test]
fn a_passing_reference_with_a_smaller_scale_is_preferred() {
    let stage = LoopStage::Second;
    assert!(prefers(
        0.5,
        1.0,
        0.75_f64.next_down(),
        1.0_f64.next_down(),
        stage
    ));
    assert!(!prefers(0.5, 1.0, 0.75, 1.0_f64.next_down(), stage));
    assert!(!prefers(0.5, 1.0, 0.75_f64.next_down(), 1.0, stage));
}

#[test]
fn a_passing_reference_replaces_a_failing_model_in_the_first_pass_only() {
    assert!(prefers(0.98, 1.0, 0.8, 10.0, LoopStage::First));
    assert!(prefers(
        0.95,
        1.0,
        0.95_f64.next_down(),
        10.0,
        LoopStage::First
    ));
    assert!(!prefers(
        0.95_f64.next_down(),
        1.0,
        0.8,
        10.0,
        LoopStage::First
    ));
    assert!(!prefers(0.95, 1.0, 0.95, 10.0, LoopStage::First));
    for stage in [LoopStage::Second, LoopStage::Final] {
        assert!(!prefers(0.98, 1.0, 0.8, 10.0, stage));
    }
}

#[test]
fn a_reference_with_better_confidence_may_have_a_slightly_larger_scale() {
    let stage = LoopStage::Second;
    assert!(prefers(0.9, 1.0, 0.5, 1.005, stage));
    assert!(!prefers(0.9, 1.0, 0.75, 1.005, stage));
    assert!(!prefers(0.9, 1.0, 0.5, 1.013, stage));
    assert!(prefers(0.7, 1.0, 0.7_f64.next_down(), 1.005, stage));
    assert!(!prefers(0.7, 1.0, 0.7, 1.005, stage));
    assert!(!prefers(0.7, 1.0, 0.7_f64.next_up(), 1.005, stage));
}

#[test]
fn a_passing_reference_replaces_a_failing_model_of_similar_scale() {
    let stage = LoopStage::Second;
    let limit: f64 = 1.0 / (1.0 - 0.0125);
    assert!(prefers(0.98, 1.0, 0.8, 1.005, stage));
    assert!(prefers(0.98, 1.0, 0.8, limit.next_down(), stage));
    assert!(!prefers(0.98, 1.0, 0.8, limit, stage));
    assert!(!prefers(0.98, 1.0, 0.8, limit.next_up(), stage));
}

#[test]
fn an_ar_coefficient_near_a_difference_prefers_the_reference() {
    let reference = compared(AIRLINE, false, 0, 0.9, 1.0);
    let prefers = |regular, seasonal, first_ar: f64| {
        prefers_reference(
            &compared((regular, seasonal), false, 0, 0.5, 1.0),
            &reference,
            Some(first_ar),
            LoopStage::Second,
        )
    };

    // (1,0,q)(0,1,1) with q <= 1 and a regular AR coefficient of 0.82 or more.
    assert!(prefers((1, 0, 1), (0, 1, 1), 0.82));
    assert!(!prefers((1, 0, 1), (0, 1, 1), 0.82_f64.next_down()));
    for (regular, seasonal) in [
        ((0, 0, 1), (0, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((1, 0, 2), (0, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 0, 1), (0, 0, 1)),
        ((1, 0, 1), (0, 1, 2)),
    ] {
        assert!(!prefers(regular, seasonal, 0.82), "{regular:?}{seasonal:?}");
    }

    // (0,1,1)(1,0,Q) with Q <= 1 and a seasonal AR coefficient of 0.65 or
    // more.
    assert!(prefers((0, 1, 1), (1, 0, 1), 0.65));
    assert!(!prefers((0, 1, 1), (1, 0, 1), 0.65_f64.next_down()));
    for (regular, seasonal) in [
        ((1, 1, 1), (1, 0, 1)),
        ((0, 0, 1), (1, 0, 1)),
        ((0, 1, 0), (1, 0, 1)),
        ((0, 1, 1), (0, 0, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((0, 1, 1), (1, 0, 2)),
    ] {
        assert!(!prefers(regular, seasonal, 0.65), "{regular:?}{seasonal:?}");
    }
}

#[test]
fn acceptance_limits_are_cumulative_sums() {
    let first = acceptance_limit(LoopStage::First);
    assert_eq!(first.to_bits(), (0.95_f64 + 0.025).to_bits());
    for stage in [LoopStage::Second, LoopStage::Final] {
        assert_eq!(acceptance_limit(stage).to_bits(), (first + 0.015).to_bits());
    }
    assert_eq!(acceptance_limit(LoopStage::Second), 0.99);
}

#[test]
fn the_loop_has_three_passes() {
    assert_eq!(LoopStage::First.next(), Some(LoopStage::Second));
    assert_eq!(LoopStage::Second.next(), Some(LoopStage::Final));
    assert_eq!(LoopStage::Final.next(), None);
}
