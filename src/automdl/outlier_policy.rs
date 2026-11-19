//! The outlier critical values `automd.f` searches with: the spec's, lowered
//! each time a model is rejected.

use crate::Error;
use crate::outliers::critical::OutlierCutoffs;
use crate::regarima::{
    OutlierCriticalValue, OutlierCriticalValues, OutlierSpec, OutlierType, OutlierTypes,
};

/// `automd.f` `TWOPT8`: a cutoff must exceed it for any reduction to run, and
/// no reduced cutoff falls below it.
const REDUCED_CUTOFF_FLOOR: f64 = 2.8;

/// `gtinpt.f` `Predcv`, the automdl `reducecv` default: the fraction of each
/// cutoff that one reduction removes.
const CUTOFF_REDUCTION: f64 = 0.14286;

/// The policy for `automd.f`'s `Lotmod` branch, which tests AO at `BIGCV`
/// when the spec requests no outlier search.
pub(super) fn no_request_outlier_policy() -> OutlierSpec {
    OutlierSpec::default()
        .with_types(OutlierTypes::ADDITIVE)
        .with_critical_values(OutlierCriticalValues::shared(
            OutlierCriticalValue::UNREACHABLE,
        ))
}

/// `policy` with `cutoffs` as its critical values. `cutoffs` holds a value
/// for exactly the outlier types `policy` tests.
pub(super) fn active_outlier_policy(
    policy: OutlierSpec,
    cutoffs: OutlierCutoffs,
) -> Result<OutlierSpec, Error> {
    let types = policy.types();
    let value = |kind| -> Result<Option<OutlierCriticalValue>, Error> {
        debug_assert_eq!(types.contains(kind), cutoffs[kind].is_some());
        cutoffs[kind].map(OutlierCriticalValue::new).transpose()
    };
    Ok(policy.with_critical_values(OutlierCriticalValues::per_type(
        value(OutlierType::Additive)?,
        value(OutlierType::LevelShift)?,
        value(OutlierType::TemporaryChange)?,
    )))
}

/// `automd.f`/`pass2.f`: each tested cutoff becomes
/// `DMAX1(TWOPT8, Critvl - Critvl*Predcv)`, but only when one exceeds `TWOPT8`.
pub(super) fn reduced_outlier_cutoffs(active: OutlierCutoffs) -> Option<OutlierCutoffs> {
    if !active
        .tested()
        .any(|(_, value)| value > REDUCED_CUTOFF_FLOOR)
    {
        return None;
    }
    Some(active.map(|value| REDUCED_CUTOFF_FLOOR.max(value - value * CUTOFF_REDUCTION)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cutoffs_are_reduced_only_while_one_exceeds_the_floor() {
        let cutoffs = OutlierCutoffs::in_build_62_order;
        assert_eq!(
            reduced_outlier_cutoffs(cutoffs([Some(2.8), None, None])),
            None
        );
        assert_eq!(
            reduced_outlier_cutoffs(cutoffs([Some(2.8_f64.next_up()), None, None])),
            Some(cutoffs([Some(2.8), None, None]))
        );
        assert_eq!(
            reduced_outlier_cutoffs(cutoffs([Some(2.0), Some(4.0), None])),
            Some(cutoffs([Some(2.8), Some(4.0 - 4.0 * 0.14286), None]))
        );
    }
}
