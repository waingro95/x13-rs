//! The AO, LS and TC regressor columns the outlier search tests (`makotl.f`)
//! and the dates each type may take.

use crate::regarima::{OutlierType, OutlierTypes};
use crate::time::TimeAxis;
use crate::{Error, Period};

/// One AO, LS or TC candidate column over the model span.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) kind: OutlierType,
    pub(crate) index: usize,
    pub(crate) period: Period,
    pub(crate) values: Vec<f64>,
}

/// Generates candidates in Build 62 date-major, AO/LS/TC order.
pub(crate) fn candidates(
    types: OutlierTypes,
    axis: TimeAxis,
    count: usize,
    tc_rate: f64,
) -> Result<Vec<Candidate>, Error> {
    // The resolved rate is valid by construction: a `TemporaryChangeRate` or
    // the gtinpt.f default.
    debug_assert!(
        !types.contains(OutlierType::TemporaryChange)
            || (tc_rate.is_finite() && tc_rate > 0.0 && tc_rate < 1.0)
    );

    let type_count = OutlierType::BUILD_62_ORDER
        .into_iter()
        .filter(|kind| types.contains(*kind))
        .count();
    let mut result = Vec::with_capacity(count.saturating_mul(type_count));
    for index in 0..count {
        let period = axis.period_at(index)?;
        for kind in OutlierType::BUILD_62_ORDER {
            if !types.contains(kind) {
                continue;
            }
            let mut values = vec![0.0; count];
            match kind {
                OutlierType::Additive => values[index] = 1.0,
                OutlierType::LevelShift => values[..index].fill(-1.0),
                OutlierType::TemporaryChange => {
                    let mut value = 1.0;
                    for current in &mut values[index..] {
                        *current = value;
                        value *= tc_rate;
                    }
                }
            }
            result.push(Candidate {
                kind,
                index,
                period,
                values,
            });
        }
    }
    Ok(result)
}

/// Whether Build 62's endpoint rules allow a candidate. The rules count from
/// the ends of the model span, wherever the search span lies inside it.
pub(crate) const fn candidate_is_eligible(
    types: OutlierTypes,
    kind: OutlierType,
    model_index: usize,
    model_count: usize,
) -> bool {
    if model_index >= model_count || !types.contains(kind) {
        return false;
    }
    match kind {
        OutlierType::Additive => true,
        OutlierType::LevelShift => {
            model_index != 0
                && (!types.contains(OutlierType::Additive)
                    || (model_index != 1 && model_index + 1 != model_count))
        }
        OutlierType::TemporaryChange => {
            !types.contains(OutlierType::Additive) || model_index + 1 != model_count
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{candidate_is_eligible, candidates};
    use crate::regarima::{OutlierType, OutlierTypes, TemporaryChangeRate};
    use crate::time::TimeAxis;
    use crate::{ArimaModel, ArimaOrder, Frequency, RegArimaSpec};

    #[test]
    fn candidates_are_ordered_by_date_then_type_with_build_62_columns() {
        let axis = TimeAxis::monthly(2000, 1).unwrap();
        let generated = candidates(OutlierTypes::ALL, axis, 3, 0.5).unwrap();

        assert_eq!(generated.len(), 9);
        assert_eq!(
            generated
                .iter()
                .map(|candidate| (candidate.index, candidate.kind))
                .collect::<Vec<_>>(),
            [
                (0, OutlierType::Additive),
                (0, OutlierType::LevelShift),
                (0, OutlierType::TemporaryChange),
                (1, OutlierType::Additive),
                (1, OutlierType::LevelShift),
                (1, OutlierType::TemporaryChange),
                (2, OutlierType::Additive),
                (2, OutlierType::LevelShift),
                (2, OutlierType::TemporaryChange),
            ]
        );

        assert_candidate(&generated, 0, OutlierType::Additive, &[1.0, 0.0, 0.0]);
        assert_candidate(&generated, 1, OutlierType::Additive, &[0.0, 1.0, 0.0]);
        assert_candidate(&generated, 2, OutlierType::Additive, &[0.0, 0.0, 1.0]);
        assert_candidate(&generated, 0, OutlierType::LevelShift, &[0.0, 0.0, 0.0]);
        assert_candidate(&generated, 1, OutlierType::LevelShift, &[-1.0, 0.0, 0.0]);
        assert_candidate(&generated, 2, OutlierType::LevelShift, &[-1.0, -1.0, 0.0]);
        assert_candidate(
            &generated,
            0,
            OutlierType::TemporaryChange,
            &[1.0, 0.5, 0.25],
        );
        assert_candidate(
            &generated,
            1,
            OutlierType::TemporaryChange,
            &[0.0, 1.0, 0.5],
        );
        assert_candidate(
            &generated,
            2,
            OutlierType::TemporaryChange,
            &[0.0, 0.0, 1.0],
        );

        for candidate in &generated {
            assert_eq!(candidate.period, axis.period_at(candidate.index).unwrap());
        }
    }

    #[test]
    fn temporary_change_columns_use_frequency_defaults_and_explicit_rate() {
        let airline = ArimaOrder::new(0, 1, 1).unwrap();
        let defaults = RegArimaSpec::new(ArimaModel::new(airline, airline).unwrap())
            .resolve_explicit()
            .unwrap();
        let monthly_rate = defaults.resolved_temporary_change_rate(Frequency::Monthly);
        let quarterly_rate = defaults.resolved_temporary_change_rate(Frequency::Quarterly);
        let explicit_rate = TemporaryChangeRate::new(0.625).unwrap().value();

        let monthly = candidates(
            OutlierTypes::TEMPORARY_CHANGE,
            TimeAxis::monthly(2000, 1).unwrap(),
            3,
            monthly_rate,
        )
        .unwrap();
        let quarterly = candidates(
            OutlierTypes::TEMPORARY_CHANGE,
            TimeAxis::quarterly(2000, 1).unwrap(),
            3,
            quarterly_rate,
        )
        .unwrap();
        let explicit = candidates(
            OutlierTypes::TEMPORARY_CHANGE,
            TimeAxis::monthly(2000, 1).unwrap(),
            3,
            explicit_rate,
        )
        .unwrap();

        assert_eq!(monthly[0].values, [1.0, 0.7, 0.489_999_999_999_999_94]);
        assert_eq!(
            quarterly[0].values,
            [1.0, 0.342_999_999_999_999_9, 0.117_648_999_999_999_95]
        );
        assert_eq!(explicit[0].values, [1.0, 0.625, 0.390625]);
    }

    #[test]
    fn endpoint_eligibility_preserves_each_build_62_type_interaction() {
        assert!(!candidate_is_eligible(
            OutlierTypes::LEVEL_SHIFT,
            OutlierType::LevelShift,
            0,
            4,
        ));
        assert!(candidate_is_eligible(
            OutlierTypes::LEVEL_SHIFT,
            OutlierType::LevelShift,
            1,
            4,
        ));
        assert!(candidate_is_eligible(
            OutlierTypes::LEVEL_SHIFT,
            OutlierType::LevelShift,
            3,
            4,
        ));

        assert!(!candidate_is_eligible(
            OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT,
            OutlierType::LevelShift,
            1,
            4,
        ));
        assert!(!candidate_is_eligible(
            OutlierTypes::ADDITIVE_AND_LEVEL_SHIFT,
            OutlierType::LevelShift,
            3,
            4,
        ));
        assert!(!candidate_is_eligible(
            OutlierTypes::ADDITIVE_AND_TEMPORARY_CHANGE,
            OutlierType::TemporaryChange,
            3,
            4,
        ));
        assert!(candidate_is_eligible(
            OutlierTypes::TEMPORARY_CHANGE,
            OutlierType::TemporaryChange,
            3,
            4,
        ));
        assert!(candidate_is_eligible(
            OutlierTypes::ALL,
            OutlierType::Additive,
            3,
            4,
        ));
    }

    fn assert_candidate(
        candidates: &[super::Candidate],
        index: usize,
        kind: OutlierType,
        expected: &[f64],
    ) {
        let candidate = candidates
            .iter()
            .find(|candidate| candidate.index == index && candidate.kind == kind)
            .unwrap();
        assert_eq!(candidate.values, expected);
    }
}
